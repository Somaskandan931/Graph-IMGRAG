"""
src/embeddings/generate_embeddings.py
Convert OCR text into 384-dim sentence embeddings.

Model: all-MiniLM-L6-v2  (report §6)
  - semantic similarity
  - fast inference
  - compact embeddings

Run standalone:
    python src/embeddings/generate_embeddings.py
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np
from src.utils.helpers import (
    load_config, get_logger, ensure_dirs,
    load_json, save_pickle, load_pickle,
)

log = get_logger("Embeddings")

_model = None

def _get_model(cfg):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(cfg["embeddings"]["model"])
    return _model


def generate(ocr_results: dict, cfg: dict):
    """
    Encode OCR text into L2-normalised embeddings.

    Args:
        ocr_results: {image_path: text}
        cfg:         config dict

    Returns:
        image_paths : list[str]
        embeddings  : np.ndarray  shape (N, 384)

    Saves:
        outputs/embeddings/image_embeddings.pkl
        outputs/embeddings/embeddings.npy
        outputs/embeddings/image_paths.json
    """
    out_pkl  = cfg["embeddings"]["output_file"]
    out_npy  = "outputs/embeddings/embeddings.npy"
    out_path = "outputs/embeddings/image_paths.json"

    ensure_dirs("outputs/embeddings")

    # Resume
    if os.path.exists(out_pkl):
        log.info(f"Resuming — loading cached embeddings from '{out_pkl}'")
        data = load_pickle(out_pkl)
        return data["image_paths"], data["embeddings"]

    model       = _get_model(cfg)
    image_paths = list(ocr_results.keys())
    texts       = list(ocr_results.values())

    log.info(f"Encoding {len(texts)} texts with '{cfg['embeddings']['model']}' ...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=cfg["embeddings"].get("normalize", True),
        batch_size=cfg["embeddings"].get("batch_size", 32),
    )

    log.info(f"Embeddings shape: {embeddings.shape}")

    save_pickle({"image_paths": image_paths, "embeddings": embeddings}, out_pkl)
    np.save(out_npy, embeddings)
    from src.utils.helpers import save_json
    save_json(image_paths, out_path)

    log.info(f"Embeddings saved → '{out_pkl}'")
    return image_paths, embeddings


if __name__ == "__main__":
    cfg      = load_config()
    ocr_json = "outputs/embeddings/ocr_results.json"
    if not os.path.exists(ocr_json):
        log.error("Run src/ocr/extract_text.py first.")
        sys.exit(1)
    ocr_results = load_json(ocr_json)
    generate(ocr_results, cfg)
