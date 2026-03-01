"""
src/utils/coco_loader.py
Download and organise a subset of COCO images for the pipeline.
"""

import os
import sys
import json
import shutil
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.helpers import get_logger, ensure_dirs, load_config, save_json

log = get_logger("COCO-Loader")

COCO_ANN_URL  = "http://images.cocodataset.org/annotations/annotations_trainval{year}.zip"
COCO_IMG_BASE = "http://images.cocodataset.org/{split}/{filename}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unzip(zip_path: str, dest_dir: str):
    import zipfile
    log.info(f"  Unzipping {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    log.info("  Unzip complete.")


def _download_file(url: str, dest: str, retries: int = 3):
    import time
    ensure_dirs(os.path.dirname(dest))
    for attempt in range(1, retries + 1):
        log.info(f"  Downloading (attempt {attempt}/{retries}): {url}")
        def _progress(block_num, block_size, total_size):
            done = block_num * block_size
            if total_size > 0:
                pct = min(100, done * 100 // total_size)
                print(f"\r    {pct:3d}%  {done // 1_048_576:.1f} MB", end="", flush=True)
        try:
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp, _progress)
            print()
            if os.path.getsize(tmp) < 1024:
                raise ValueError("Downloaded file too small")
            os.replace(tmp, dest)
            return
        except KeyboardInterrupt:
            for f in [dest, dest + ".part"]:
                if os.path.exists(f): os.remove(f)
            raise
        except Exception as e:
            print()
            log.warning(f"  Attempt {attempt} failed: {e}")
            for f in [dest + ".part"]:
                if os.path.exists(f): os.remove(f)
            if attempt < retries:
                time.sleep(3)
    raise RuntimeError(
        f"Failed to download after {retries} attempts.\n"
        f"Please download manually: {url}\n"
        f"and place at: {dest}"
    )


# ── Annotations ───────────────────────────────────────────────────────────────

def _ensure_annotations(cfg: dict) -> str:
    """
    Find or download the COCO annotations JSON.
    Returns path to instances_<split>.json.

    Search order:
      1. instances_val2017.json already extracted → done
      2. annotations_trainval2017.zip in annotations/ → unzip it
      3. annotations_trainval2017.zip anywhere in dataset/ → copy + unzip
      4. Download from COCO CDN
    """
    import zipfile

    year    = cfg["dataset"]["coco_year"]
    split   = cfg["dataset"]["coco_split"]
    ann_dir = cfg["dataset"]["annotations_dir"]
    ensure_dirs(ann_dir)

    instances_json = os.path.join(ann_dir, f"instances_{split}.json")

    # ── Already extracted ──────────────────────────────────────────────────────
    if os.path.exists(instances_json):
        log.info(f"  Annotations already present: {instances_json}")
        return instances_json

    zip_name = f"annotations_trainval{year}.zip"
    zip_dest = os.path.join(ann_dir, zip_name)

    # ── Search entire project tree for the zip ─────────────────────────────────
    if not os.path.exists(zip_dest):
        project_root = os.path.abspath(os.path.join(ann_dir, "../.."))
        log.info(f"  Searching for {zip_name} under {project_root} ...")
        for root, dirs, files in os.walk(project_root):
            # skip hidden and venv folders
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv','env','.venv','__pycache__','node_modules')]
            for fname in files:
                if fname == zip_name:
                    found = os.path.join(root, fname)
                    size_mb = os.path.getsize(found) / 1_048_576
                    log.info(f"  Found zip at: {found}  ({size_mb:.1f} MB)")
                    if size_mb < 50:
                        log.warning(f"  Zip seems too small ({size_mb:.1f} MB) — may be incomplete, skipping.")
                        continue
                    if found != zip_dest:
                        log.info(f"  Copying to: {zip_dest}")
                        shutil.copy2(found, zip_dest)
                    break
            if os.path.exists(zip_dest):
                break

    # ── Still not found → download ─────────────────────────────────────────────
    if not os.path.exists(zip_dest):
        log.info(f"  Zip not found locally. Downloading (~241 MB) ...")
        _download_file(COCO_ANN_URL.format(year=year), zip_dest)

    # ── Unzip ──────────────────────────────────────────────────────────────────
    size_mb = os.path.getsize(zip_dest) / 1_048_576
    log.info(f"  Using zip: {zip_dest}  ({size_mb:.1f} MB)")
    try:
        _unzip(zip_dest, ann_dir)
    except Exception as e:
        log.error(f"  Unzip failed: {e}")
        os.remove(zip_dest)
        raise RuntimeError(
            f"Zip file appears corrupt: {zip_dest}\n"
            f"Delete it and re-download from:\n  {COCO_ANN_URL.format(year=year)}"
        )

    # The zip extracts to an 'annotations/' subfolder — move files up
    sub = os.path.join(ann_dir, "annotations")
    if os.path.isdir(sub):
        for fname in os.listdir(sub):
            shutil.move(os.path.join(sub, fname), os.path.join(ann_dir, fname))
        shutil.rmtree(sub, ignore_errors=True)

    if not os.path.exists(instances_json):
        raise RuntimeError(
            f"Unzip succeeded but {instances_json} not found.\n"
            f"Contents of {ann_dir}: {os.listdir(ann_dir)}"
        )

    log.info(f"  Annotations ready: {instances_json}")
    return instances_json


# ── Category helpers ──────────────────────────────────────────────────────────

def _build_cat_map(coco_ann: dict) -> dict:
    return {c["id"]: c["supercategory"] for c in coco_ann["categories"]}


def _image_supercategories(coco_ann: dict, cat_map: dict) -> dict:
    img_cats: dict = {}
    for ann in coco_ann["annotations"]:
        iid  = ann["image_id"]
        scat = cat_map.get(ann["category_id"], "unknown")
        img_cats.setdefault(iid, set()).add(scat)
    return img_cats


# ── Main loader ───────────────────────────────────────────────────────────────

def load_coco_dataset(cfg: dict) -> int:
    year        = cfg["dataset"]["coco_year"]
    split       = cfg["dataset"]["coco_split"]
    processed   = cfg["dataset"]["processed_dir"]
    ann_dir     = cfg["dataset"]["annotations_dir"]
    max_images  = cfg["dataset"].get("max_images") or 99999
    wanted_cats = set(cfg["dataset"].get("categories", []))

    log.info(f"Loading COCO {split} (max_images={max_images}) ...")

    ann_json = _ensure_annotations(cfg)
    with open(ann_json) as f:
        coco_ann = json.load(f)

    cat_map  = _build_cat_map(coco_ann)
    img_cats = _image_supercategories(coco_ann, cat_map)

    images_meta = {img["id"]: img for img in coco_ann["images"]}
    selected = []
    for img_id, supercats in img_cats.items():
        matched = supercats & wanted_cats
        if matched and img_id in images_meta:
            primary_cat = next(iter(sorted(matched)))
            selected.append((img_id, primary_cat, images_meta[img_id]))

    selected.sort(key=lambda x: x[0])
    selected = selected[:max_images]
    log.info(f"  {len(selected)} images selected")

    # ── Resolve local image source directories ────────────────────────────────
    # Priority:
    #   1. cfg["dataset"]["local_images_dir"]  — explicitly configured path
    #   2. dataset/<split>/                    — relative to project root
    #   3. COCO CDN download                  — fallback if nothing found locally

    project_root = os.path.abspath(os.path.join(ann_dir, "../.."))

    local_src_dirs = []

    # Check the explicitly configured path first
    configured_local = cfg["dataset"].get("local_images_dir", "")
    if configured_local:
        # Support both absolute and relative paths
        abs_local = (configured_local if os.path.isabs(configured_local)
                     else os.path.join(project_root, configured_local))
        abs_local = os.path.normpath(abs_local)
        if os.path.isdir(abs_local):
            local_src_dirs.append(abs_local)
            log.info(f"  Local image source: {abs_local}")
        else:
            log.warning(f"  Configured local_images_dir not found: {abs_local}")

    # Also check the conventional dataset/<split>/ folder
    conventional = os.path.join(project_root, "dataset", split)
    if os.path.isdir(conventional) and conventional not in local_src_dirs:
        local_src_dirs.append(conventional)
        log.info(f"  Also checking: {conventional}")

    if local_src_dirs:
        log.info(f"  Will copy from local folders (no download needed)")
    else:
        log.info(f"  No local image folders found — will download from COCO CDN")

    from tqdm import tqdm
    meta_records = []
    total  = 0
    failed = 0
    pbar   = tqdm(selected, desc="Setting up images", unit="img")

    for img_id, cat, img_info in pbar:
        filename  = img_info["file_name"]
        dest_dir  = os.path.join(processed, cat)
        ensure_dirs(dest_dir)
        dest_path = os.path.join(dest_dir, filename)

        if not os.path.exists(dest_path):
            # 1. Try each local source directory
            found_locally = False
            for src_dir in local_src_dirs:
                src = os.path.join(src_dir, filename)
                if os.path.exists(src):
                    shutil.copy2(src, dest_path)
                    found_locally = True
                    break

            if not found_locally:
                # 2. Fall back to downloading from COCO CDN
                url        = COCO_IMG_BASE.format(split=split, filename=filename)
                downloaded = False
                for attempt in range(1, 4):
                    try:
                        tmp = dest_path + ".part"
                        urllib.request.urlretrieve(url, tmp)
                        if os.path.getsize(tmp) < 1024:
                            raise ValueError("File too small")
                        os.replace(tmp, dest_path)
                        downloaded = True
                        break
                    except KeyboardInterrupt:
                        for f in [dest_path, dest_path + ".part"]:
                            if os.path.exists(f): os.remove(f)
                        log.info(f"\n  Interrupted. {total} images ready so far.")
                        log.info("  Re-run the same command to resume.")
                        raise
                    except Exception as e:
                        if os.path.exists(dest_path + ".part"):
                            os.remove(dest_path + ".part")
                        if attempt == 3:
                            pbar.write(f"  SKIP {filename}: {e}")
                            failed += 1
                        else:
                            import time; time.sleep(2)
                if not downloaded:
                    continue

        total += 1
        pbar.set_postfix(ok=total, skip=failed)
        meta_records.append({
            "image_id":      img_id,
            "file_name":     filename,
            "path":          dest_path,
            "supercategory": cat,
            "all_supercats": sorted(img_cats.get(img_id, [])),
            "width":         img_info.get("width"),
            "height":        img_info.get("height"),
            "coco_url":      img_info.get("coco_url", ""),
        })

    meta_path = os.path.join(ann_dir, "coco_meta.json")
    save_json(meta_records, meta_path)
    log.info(f"  Done: {total} images ready, {failed} failed → {meta_path}")
    return total


# ── Meta helper ───────────────────────────────────────────────────────────────

def load_coco_meta(cfg: dict) -> dict:
    meta_path = os.path.join(cfg["dataset"]["annotations_dir"], "coco_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        records = json.load(f)
    return {r["path"]: r["supercategory"] for r in records}


if __name__ == "__main__":
    cfg = load_config()
    n   = load_coco_dataset(cfg)
    log.info(f"Done. {n} images ready in '{cfg['dataset']['processed_dir']}'")