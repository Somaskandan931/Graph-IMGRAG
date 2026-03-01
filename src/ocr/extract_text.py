"""
src/ocr/extract_text.py
OCR text extraction using EasyOCR.

Pipeline (report §5):
    1. Load image
    2. Detect text regions
    3. Recognise characters
    4. Combine extracted words into one string

Run standalone:
    python src/ocr/extract_text.py
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

from pathlib import Path
from tqdm import tqdm
from src.utils.helpers import (
    load_config, get_logger, ensure_dirs,
    collect_images, save_json, load_json,
)

log = get_logger("OCR")

_reader = None

def _get_reader(cfg):
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(cfg["ocr"]["languages"],
                                  gpu=cfg["ocr"].get("gpu", False))
    return _reader


def extract_one(image_path: str, cfg: dict) -> str:
    """
    Extract text from a single image and enrich it with category metadata.

    Strategy:
      1. Raw OCR text from the image (if any)
      2. COCO supercategory label (always appended, repeated for embedding weight)
      3. Cleaned filename as last resort

    This ensures every image has a semantically meaningful embedding even
    when OCR finds no readable text in the photo — which is common in
    natural photography like COCO val2017.
    """
    from src.utils.helpers import get_category

    reader = _get_reader(cfg)
    try:
        results = reader.readtext(image_path, detail=0,
                                   paragraph=cfg["ocr"].get("paragraph", True))
        ocr_text = " ".join(results).strip()
    except Exception as e:
        log.warning(f"OCR error on {os.path.basename(image_path)}: {e}")
        ocr_text = ""

    # Always enrich with the COCO supercategory — repeated twice so the
    # category pulls the embedding into the right semantic neighbourhood.
    category = get_category(image_path)          # e.g. "animal", "vehicle"
    folder   = Path(image_path).parent.name      # fallback: parent folder name

    parts = []
    if ocr_text:
        parts.append(ocr_text)

    cat_label = category if category not in ("unknown", "", ".") else folder
    if cat_label and cat_label not in ("unknown", "", ".", "images"):
        parts.append(cat_label)
        parts.append(cat_label)      # repeat to increase semantic weight

    enriched = " ".join(parts).strip()
    if not enriched:
        enriched = Path(image_path).stem.replace("_", " ")

    return enriched


def run_ocr(cfg: dict) -> dict:
    """
    Run OCR on every image in dataset/processed/images/.
    
    Saves:
        dataset/processed/ocr_text/<stem>.txt   per-image text files
        outputs/embeddings/ocr_results.json     combined dict

    Returns: {image_path: extracted_text}
    """
    img_dir  = cfg["dataset"]["processed_dir"]
    ocr_dir  = cfg["dataset"]["ocr_text_dir"]
    out_json = "outputs/embeddings/ocr_results.json"

    ensure_dirs(ocr_dir, "outputs/embeddings")

    # Resume: skip if already done
    if os.path.exists(out_json):
        log.info(f"Resuming — loading cached OCR from '{out_json}'")
        return load_json(out_json)

    images = collect_images(img_dir)
    if not images:
        log.error(f"No images found in '{img_dir}'. Run main.py --demo first.")
        sys.exit(1)

    log.info(f"Running OCR on {len(images)} images ...")
    results = {}

    for path in tqdm(images, desc="OCR"):
        text = extract_one(path, cfg)
        results[path] = text
        # Save individual .txt
        txt_out = os.path.join(ocr_dir, Path(path).stem + ".txt")
        with open(txt_out, "w") as f:
            f.write(text)

    save_json(results, out_json)
    log.info(f"OCR complete. {len(results)} results saved → '{out_json}'")
    return results


if __name__ == "__main__":
    cfg = load_config()
    run_ocr(cfg)