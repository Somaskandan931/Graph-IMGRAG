"""
main.py — Full pipeline entry point.

─────────────────────────────────────────────────────────────────
 FIRST TIME SETUP (images already on disk):
   python main.py --build

 After that, search instantly without rebuilding:
   python main.py --query "dog park"
   python main.py --query "car traffic" --top_k 10

 Other options:
   python main.py --build --eval         # build + evaluate
   python main.py --build --force        # wipe cache and rebuild
   python main.py --build --max_images 200  # limit to 200 images
   python main.py --stats                # show database stats
─────────────────────────────────────────────────────────────────

How it works:
  Your COCO images in dataset/val2017/ (and train2017/, test2017/)
  are already on disk.  The pipeline:
    1. Reads the COCO annotation file to get category labels
    2. Copies images into dataset/processed/images/<category>/
    3. Runs OCR on each image  →  extracts embedded text
    4. Encodes text as 384-dim sentence embeddings
    5. Builds a KNN similarity graph (k=5)
    6. Saves everything to a SQLite database (outputs/graph_imgrag.db)

  After --build, all searches load from the database — instant,
  no re-processing.
"""

import os
import sys
import argparse

from src.utils.helpers import load_config, get_logger, ensure_dirs
from src.utils.database import (
    init_db, db_exists, upsert_images, upsert_graph_edges,
    load_all_images, load_graph_edges, set_meta, get_stats,
    DEFAULT_DB,
)

log = get_logger("Main")


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_dataset(cfg):
    """
    Organise local COCO images into category subfolders and populate
    the COCO metadata cache.  No downloading takes place if images
    are already on disk.
    """
    from src.utils.coco_loader import load_coco_dataset, load_coco_meta
    from src.utils.helpers import set_coco_meta

    # Confirm that the local image dir is reachable
    local_dir = cfg["dataset"].get("local_images_dir", "")
    if local_dir:
        abs_dir = (local_dir if os.path.isabs(local_dir)
                   else os.path.abspath(local_dir))
        if os.path.isdir(abs_dir):
            log.info(f"Local images found at: {abs_dir}")
        else:
            log.warning(
                f"local_images_dir not found: {abs_dir}\n"
                f"  Check configs/config.yaml → dataset.local_images_dir"
            )

    n    = load_coco_dataset(cfg)
    meta = load_coco_meta(cfg)
    set_coco_meta(meta)
    log.info(f"Dataset ready — {n} images organised in '{cfg['dataset']['processed_dir']}'")
    return meta


def _load_coco_meta(cfg):
    """Load saved COCO metadata into the category cache (no rebuild)."""
    try:
        from src.utils.coco_loader import load_coco_meta
        from src.utils.helpers import set_coco_meta
        meta = load_coco_meta(cfg)
        if meta:
            set_coco_meta(meta)
            log.info(f"Loaded COCO metadata ({len(meta)} images)")
            return meta
    except Exception:
        pass
    return {}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="graph-imgrag — Graph-Based Image Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --build                     # index your local images (run once)
  python main.py --query "dog park"          # search instantly
  python main.py --build --max_images 200    # index only 200 images
  python main.py --stats                     # show database info
  python main.py --build --force             # wipe and rebuild from scratch
        """,
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build (or rebuild) the search index from local COCO images",
    )
    # Keep --coco as a hidden alias for --build so old commands still work
    parser.add_argument("--coco", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--split",      type=str, default=None,
                        help="Override coco_split in config (e.g. val2017)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Override max_images in config")
    parser.add_argument("--query",      type=str, default=None,
                        help="Text query to run after the index is ready")
    parser.add_argument("--top_k",      type=int, default=5,
                        help="Number of results to return (default 5)")
    parser.add_argument("--eval",       action="store_true",
                        help="Run evaluation metrics after building")
    parser.add_argument("--force",      action="store_true",
                        help="Delete cached outputs and rebuild from scratch")
    parser.add_argument("--stats",      action="store_true",
                        help="Print database statistics and exit")
    parser.add_argument("--db",         type=str, default=DEFAULT_DB,
                        help=f"SQLite database path (default: {DEFAULT_DB})")
    args = parser.parse_args()

    # --coco is an alias for --build
    if args.coco:
        args.build = True

    cfg = load_config()
    if args.split:      cfg["dataset"]["coco_split"]    = args.split
    if args.max_images: cfg["dataset"]["max_images"]    = args.max_images
    db_path = args.db

    # ── Stats only ────────────────────────────────────────────────────────────
    if args.stats:
        if not db_exists(db_path):
            print(f"\n  No database found at '{db_path}'.")
            print(f"  Run:  python main.py --build\n")
            return
        stats = get_stats(db_path)
        print("\n" + "=" * 52)
        print(f"  Database : {db_path}")
        print(f"  Images   : {stats['total_images']:,}")
        print(f"  Edges    : {stats['total_edges']:,}")
        print("\n  Images per category:")
        for cat, n in stats["categories"].items():
            print(f"    {cat:20s}  {n:>5} images")
        print("=" * 52 + "\n")
        return

    # ── Force reset ───────────────────────────────────────────────────────────
    if args.force:
        import shutil
        for d in ["outputs/embeddings", "outputs/graphs"]:
            if os.path.exists(d):
                shutil.rmtree(d)
                log.info(f"Deleted: {d}/")
        if os.path.exists(db_path):
            os.remove(db_path)
            log.info(f"Deleted database: {db_path}")
        # Also clear the processed images so category enrichment re-runs
        processed = cfg["dataset"]["processed_dir"]
        if os.path.exists(processed):
            shutil.rmtree(processed)
            log.info(f"Deleted: {processed}/")

    print("\n" + "=" * 65)
    print("  graph-imgrag  |  Graph-Based Image Retrieval")
    print("=" * 65)

    # ── Search only — database already built ──────────────────────────────────
    if not args.build and args.query:
        if not db_exists(db_path):
            log.error(
                f"Database not found at '{db_path}'.\n"
                f"  Build it first with:  python main.py --build"
            )
            sys.exit(1)
        log.info(f"\nLoading from database '{db_path}' ...")
        _load_coco_meta(cfg)
        image_paths, embeddings, ocr_results, categories = load_all_images(db_path)
        labels = [os.path.basename(p) for p in image_paths]
        G = load_graph_edges(image_paths, db_path)
        from src.retrieval.search import search
        search(args.query, image_paths, embeddings, G, labels, cfg, top_k=args.top_k)
        print(f"\n  Results → outputs/retrieval_results/\n")
        return

    # ── Build pipeline ────────────────────────────────────────────────────────
    if not args.build and not db_exists(db_path):
        log.info("No database found — running full build automatically.")
        args.build = True

    init_db(db_path)

    # Step 1 — organise local images by category
    log.info("\n[STEP 1] Organising Local COCO Images ...")
    categories_map = prepare_dataset(cfg)

    # Step 2 — OCR
    log.info("\n[STEP 2] OCR Text Extraction")
    from src.ocr.extract_text import run_ocr
    ocr_results = run_ocr(cfg)
    log.info(f"  {len(ocr_results)} images processed")

    # Step 3 — Embeddings
    log.info("\n[STEP 3] Generating Embeddings")
    from src.embeddings.generate_embeddings import generate
    image_paths, embeddings = generate(ocr_results, cfg)
    labels = [os.path.basename(p) for p in image_paths]
    log.info(f"  Embedding matrix: {embeddings.shape}")

    # Step 4 — KNN graph
    log.info("\n[STEP 4] Building KNN Graph")
    from src.graph.build_graph import build, visualize
    G, sim_matrix, labels = build(image_paths, embeddings, cfg)

    # Step 5 — Persist to SQLite
    log.info("\n[STEP 5] Saving to Database ...")
    upsert_images(image_paths, ocr_results, embeddings, categories_map, db_path)
    upsert_graph_edges(G, image_paths, db_path)
    set_meta("coco_split",   cfg["dataset"]["coco_split"],  db_path)
    set_meta("max_images",   cfg["dataset"]["max_images"],  db_path)
    set_meta("model",        cfg["embeddings"]["model"],    db_path)
    set_meta("total_images", len(image_paths),              db_path)
    stats = get_stats(db_path)
    log.info(
        f"  Saved → '{db_path}'  "
        f"({stats['total_images']} images, {stats['total_edges']} edges)"
    )

    # Step 6 — Visualise graph
    log.info("\n[STEP 6] Visualising Graph")
    visualize(G, labels, cfg)

    # Step 7 — Demo searches
    log.info("\n[STEP 7] Running Sample Searches")
    from src.retrieval.search import search, evaluate
    demo_queries = [args.query] if args.query else [
        "dog park outdoor",
        "car road traffic",
        "person crowd street",
        "food pizza plate",
        "sports ball player",
    ]
    for q in demo_queries:
        search(q, image_paths, embeddings, G, labels, cfg, top_k=args.top_k)

    # Step 8 — Evaluation (optional)
    if args.eval or not args.query:
        log.info("\n[STEP 8] Evaluation")
        evaluate(image_paths, embeddings, G, labels, cfg)

    print("\n" + "=" * 65)
    print("  BUILD COMPLETE")
    print(f"  Database  →  {db_path}")
    print(f"  Outputs   →  outputs/")
    print(f"  Web UI    →  streamlit run app/streamlit_app.py")
    print()
    print("  Searches are now instant — no rebuild needed:")
    print('    python main.py --query "your search here"')
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()