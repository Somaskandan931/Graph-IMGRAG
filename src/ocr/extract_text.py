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
    """Extract all text from a single image. Returns a plain string."""
    reader = _get_reader(cfg)
    try:
        results = reader.readtext(image_path, detail=0,
                                   paragraph=cfg["ocr"].get("paragraph", True))
        text = " ".join(results).strip()
    except Exception as e:
        log.warning(f"OCR error on {os.path.basename(image_path)}: {e}")
        text = ""
    return text if text else Path(image_path).stem.replace("_", " ")


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