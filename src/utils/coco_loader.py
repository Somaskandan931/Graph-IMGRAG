"""
src/utils/coco_loader.py
Download and organise a subset of COCO images for the pipeline.

Usage (standalone):
    python src/utils/coco_loader.py

What it does:
    1. Downloads COCO annotations JSON (instances_<split><year>.json) if missing.
    2. Downloads up to max_images images from the official COCO CDN.
    3. Organises images into dataset/processed/images/<supercategory>/ folders
       so the rest of the pipeline can find them with collect_images().
    4. Writes dataset/annotations/coco_meta.json with per-image metadata.

COCO image URL pattern:
    http://images.cocodataset.org/<split>/<filename>
    e.g. http://images.cocodataset.org/val2017/000000397133.jpg
"""

import os
import sys
import json
import shutil
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))
from src.utils.helpers import get_logger, ensure_dirs, load_config, save_json

log = get_logger("COCO-Loader")

# Official COCO URLs
COCO_ANN_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval{year}.zip"
)
COCO_IMG_BASE = "http://images.cocodataset.org/{split}/{filename}"


# ── Annotation helpers ────────────────────────────────────────────────────────

def _download_file(url: str, dest: str, retries: int = 3):
    """Download url → dest with progress, retries, and size validation."""
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
            # Validate: must be at least 1 KB
            if os.path.getsize(tmp) < 1024:
                raise ValueError(f"Downloaded file too small ({os.path.getsize(tmp)} bytes) — likely a redirect/error page")
            os.replace(tmp, dest)
            log.info(f"  Saved → {dest}  ({os.path.getsize(dest) // 1_048_576} MB)")
            return
        except Exception as e:
            print()
            log.warning(f"  Download attempt {attempt} failed: {e}")
            # Remove any partial file
            for f in [dest, dest + ".part"]:
                if os.path.exists(f):
                    os.remove(f)
            if attempt < retries:
                log.info(f"  Retrying in 3 s ...")
                time.sleep(3)

    raise RuntimeError(
        f"Failed to download {url} after {retries} attempts.\n"
        f"  Please download manually and place at: {dest}\n"
        f"  Direct link: {url}"
    )


def _is_valid_zip(path: str) -> bool:
    """Return True if path exists and is a valid zip file."""
    import zipfile
    if not os.path.exists(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.testzip()
        return True
    except Exception:
        return False


def _ensure_annotations(cfg: dict) -> str:
    """
    Download and unzip COCO annotation file if not present.
    Returns path to the instances JSON file.
    """
    import zipfile

    year    = cfg["dataset"]["coco_year"]
    split   = cfg["dataset"]["coco_split"]
    ann_dir = cfg["dataset"]["annotations_dir"]
    ensure_dirs(ann_dir)

    instances_json = os.path.join(ann_dir, f"instances_{split}.json")
    if os.path.exists(instances_json):
        log.info(f"  Annotations already present: {instances_json}")
        return instances_json

    zip_url  = COCO_ANN_URL.format(year=year)
    zip_dest = os.path.join(ann_dir, f"annotations_trainval{year}.zip")

    # Remove corrupt zip if it exists
    if os.path.exists(zip_dest) and not _is_valid_zip(zip_dest):
        log.warning(f"  Corrupt/incomplete zip found — deleting and re-downloading: {zip_dest}")
        os.remove(zip_dest)

    if not os.path.exists(zip_dest):
        log.info("Downloading COCO annotations (~241 MB) ...")
        _download_file(zip_url, zip_dest)

    # Final validation before unzipping
    if not _is_valid_zip(zip_dest):
        os.remove(zip_dest)
        raise RuntimeError(
            f"Annotation zip is still corrupt after download.\n"
            f"Please download it manually from:\n  {zip_url}\n"
            f"and save it to:\n  {zip_dest}"
        )

    log.info("Unzipping annotations ...")
    with zipfile.ZipFile(zip_dest, "r") as zf:
        zf.extractall(ann_dir)

    # Move extracted JSON to ann_dir root
    extracted = os.path.join(ann_dir, "annotations", f"instances_{split}.json")
    if os.path.exists(extracted):
        shutil.move(extracted, instances_json)
        shutil.rmtree(os.path.join(ann_dir, "annotations"), ignore_errors=True)

    if not os.path.exists(instances_json):
        raise RuntimeError(
            f"Unzip succeeded but instances JSON not found at: {instances_json}\n"
            f"Check the zip contents in: {ann_dir}"
        )

    log.info(f"  Annotations ready: {instances_json}")
    return instances_json


# ── Category mapping ──────────────────────────────────────────────────────────

def _build_cat_map(coco_ann: dict) -> dict:
    """
    Returns {category_id: supercategory_name} from COCO annotations.
    """
    return {c["id"]: c["supercategory"] for c in coco_ann["categories"]}


def _image_supercategories(coco_ann: dict, cat_map: dict) -> dict:
    """
    Returns {image_id: set_of_supercategories} by scanning all annotations.
    """
    img_cats: dict[int, set] = {}
    for ann in coco_ann["annotations"]:
        iid  = ann["image_id"]
        scat = cat_map.get(ann["category_id"], "unknown")
        img_cats.setdefault(iid, set()).add(scat)
    return img_cats


# ── Main loader ───────────────────────────────────────────────────────────────

def load_coco_dataset(cfg: dict) -> int:
    """
    Download and organise COCO images.

    Returns:
        int — total images placed in processed_dir
    """
    year       = cfg["dataset"]["coco_year"]
    split      = cfg["dataset"]["coco_split"]
    processed  = cfg["dataset"]["processed_dir"]
    ann_dir    = cfg["dataset"]["annotations_dir"]
    max_images = cfg["dataset"].get("max_images") or 99999
    wanted_cats = set(cfg["dataset"].get("categories", []))

    log.info(f"Loading COCO {split} (max_images={max_images}) ...")

    # 1. Annotations
    ann_json = _ensure_annotations(cfg)
    with open(ann_json) as f:
        coco_ann = json.load(f)

    cat_map   = _build_cat_map(coco_ann)
    img_cats  = _image_supercategories(coco_ann, cat_map)

    # 2. Build image list — filter to wanted supercategories
    images_meta = {img["id"]: img for img in coco_ann["images"]}
    selected = []
    for img_id, supercats in img_cats.items():
        # Only keep images whose primary supercategory is in wanted list
        matched = supercats & wanted_cats
        if matched and img_id in images_meta:
            # Pick first matched supercategory as folder name
            primary_cat = next(iter(sorted(matched)))
            selected.append((img_id, primary_cat, images_meta[img_id]))

    # Sort for reproducibility, cap at max_images
    selected.sort(key=lambda x: x[0])
    selected = selected[:max_images]

    log.info(f"  {len(selected)} images selected (from {len(img_cats)} annotated)")

    # 3. Download images and place them in category subfolders
    meta_records = []
    total = 0
    failed = 0

    for img_id, cat, img_info in selected:
        filename = img_info["file_name"]
        dest_dir = os.path.join(processed, cat)
        ensure_dirs(dest_dir)
        dest_path = os.path.join(dest_dir, filename)

        if not os.path.exists(dest_path):
            url = COCO_IMG_BASE.format(split=split, filename=filename)
            try:
                urllib.request.urlretrieve(url, dest_path)
                total += 1
            except Exception as e:
                log.warning(f"  Failed to download {filename}: {e}")
                failed += 1
                continue
        else:
            total += 1

        meta_records.append({
            "image_id":     img_id,
            "file_name":    filename,
            "path":         dest_path,
            "supercategory": cat,
            "all_supercats": sorted(img_cats.get(img_id, [])),
            "width":        img_info.get("width"),
            "height":       img_info.get("height"),
            "coco_url":     img_info.get("coco_url", ""),
        })

    # 4. Save metadata
    meta_path = os.path.join(ann_dir, "coco_meta.json")
    save_json(meta_records, meta_path)
    log.info(f"  Placed {total} images, {failed} failed → meta: {meta_path}")
    return total


# ── Category helper (used by helpers.get_category) ───────────────────────────

def load_coco_meta(cfg: dict) -> dict:
    """Return {image_path: supercategory} from saved coco_meta.json."""
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