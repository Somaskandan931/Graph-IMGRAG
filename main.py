"""
main.py — Full pipeline entry point (COCO Dataset Edition).

Usage:
    # Download COCO val2017 subset (max 500 images) and run all steps:
    python main.py --coco

    # Skip download (images already present) and run pipeline:
    python main.py

    # Run with a text query after building:
    python main.py --query "dog park outdoor"

    # Also run evaluation:
    python main.py --coco --eval

    # Re-run everything from scratch:
    python main.py --coco --force

COCO notes:
    - Default split : val2017  (5 000 images, ~1 GB download)
    - Default limit : 500 images  (configurable in configs/config.yaml)
    - Images are placed in dataset/processed/images/<supercategory>/
    - Annotations zip is ~241 MB and cached in dataset/annotations/

    To switch to train2017 (18 GB), change coco_split in config.yaml.
"""

import os
import sys
import argparse

from src.utils.helpers import load_config, get_logger, ensure_dirs, load_pickle

log = get_logger("Main")


# ─────────────────────────────────────────────────────────────────────────────
#  COCO dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_coco_dataset(cfg):
    """Download and organise COCO images into category sub-folders."""
    from src.utils.coco_loader import load_coco_dataset, load_coco_meta
    from src.utils.helpers import set_coco_meta

    n = load_coco_dataset(cfg)
    # Populate the category cache so get_category() is COCO-aware
    meta = load_coco_meta(cfg)
    set_coco_meta(meta)
    log.info(f"COCO dataset ready — {n} images in '{cfg['dataset']['processed_dir']}'")
    return n


def _init_coco_meta(cfg):
    """Load COCO metadata into the category cache without re-downloading."""
    try:
        from src.utils.coco_loader import load_coco_meta
        from src.utils.helpers import set_coco_meta
        meta = load_coco_meta(cfg)
        if meta:
            set_coco_meta(meta)
            log.info(f"Loaded COCO metadata ({len(meta)} images)")
    except Exception:
        pass  # fall back to folder-name categories


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="graph-imgrag — Graph-Based Image Retrieval (COCO Dataset)"
    )
    parser.add_argument(
        "--coco", action="store_true",
        help="Download and organise COCO images before running the pipeline"
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Override COCO split (e.g. val2017, train2017). Overrides config."
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Override max images to download. Overrides config."
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Text or image-path query (used after the pipeline builds)"
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--eval",  action="store_true",
                        help="Run evaluation metrics")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if cached outputs exist")
    args = parser.parse_args()

    cfg = load_config()

    # Apply CLI overrides
    if args.split:
        cfg["dataset"]["coco_split"] = args.split
    if args.max_images:
        cfg["dataset"]["max_images"] = args.max_images

    if args.force:
        import shutil
        for d in ["outputs/embeddings", "outputs/graphs"]:
            if os.path.exists(d):
                shutil.rmtree(d)
        log.info("Cache cleared — re-running all steps")

    print("\n" + "="*65)
    print("  graph-imgrag  |  Graph-Based Image Retrieval (COCO Edition)")
    print("="*65)

    # ── Step 1 — COCO dataset ─────────────────────────────────────────────────
    if args.coco:
        log.info("\n[STEP 1] Downloading & Organising COCO Dataset ...")
        prepare_coco_dataset(cfg)
    else:
        log.info("\n[STEP 1] Skipping COCO download (use --coco to enable)")
        _init_coco_meta(cfg)

    # ── Step 2 — OCR ─────────────────────────────────────────────────────────
    log.info("\n[STEP 2] OCR Text Extraction")
    from src.ocr.extract_text import run_ocr
    ocr_results = run_ocr(cfg)
    log.info(f"  {len(ocr_results)} images processed")

    # ── Step 3 — Embeddings ───────────────────────────────────────────────────
    log.info("\n[STEP 3] Generating Embeddings")
    from src.embeddings.generate_embeddings import generate
    image_paths, embeddings = generate(ocr_results, cfg)
    labels = [os.path.basename(p) for p in image_paths]
    log.info(f"  Embedding matrix: {embeddings.shape}")

    # ── Step 4 — Graph ────────────────────────────────────────────────────────
    log.info("\n[STEP 4] Building KNN Graph")
    from src.graph.build_graph import build, visualize
    G, sim_matrix, labels = build(image_paths, embeddings, cfg)

    # ── Step 5 — Visualise ────────────────────────────────────────────────────
    log.info("\n[STEP 5] Visualising Graph")
    visualize(G, labels, cfg)

    # ── Step 6 — Search ───────────────────────────────────────────────────────
    log.info("\n[STEP 6] Running Search Queries")
    from src.retrieval.search import search, evaluate

    # Default COCO-relevant queries if no --query provided
    default_queries = [
        "dog park outdoor",
        "car road traffic",
        "person crowd street",
        "food pizza plate",
        "sports ball player",
    ]
    queries = ([args.query] if args.query else default_queries)
    for q in queries:
        search(q, image_paths, embeddings, G, labels, cfg, top_k=args.top_k)

    # ── Step 7 — Evaluate ─────────────────────────────────────────────────────
    if args.eval or not args.query:
        log.info("\n[STEP 7] Evaluation")
        evaluate(image_paths, embeddings, G, labels, cfg)

    print("\n" + "="*65)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs → outputs/")
    print(f"  Web UI  → streamlit run app/streamlit_app.py")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
