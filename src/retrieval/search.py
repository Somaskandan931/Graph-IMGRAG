"""
src/retrieval/search.py
Query-based image retrieval with graph expansion + evaluation.

Query types (report §8):
    - Text query  : "passport"
    - Image query : path/to/image.jpg  (OCR extracted first)

Run standalone:
    python src/retrieval/search.py --query "passport"
    python src/retrieval/search.py --query "STOP" --top_k 5 --eval
"""

import os, sys, pickle, argparse
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.helpers import (
    load_config, get_logger, ensure_dirs,
    load_pickle, save_json, get_category,
)

log = get_logger("Retrieval")

_model  = None
_reader = None

def _get_model(cfg):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(cfg["embeddings"]["model"])
    return _model

def _get_reader(cfg):
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(cfg["ocr"]["languages"],
                                  gpu=cfg["ocr"].get("gpu", False))
    return _reader


def search(query, image_paths, embeddings, G, labels, cfg, top_k=None):
    """
    Retrieve the top-K most relevant images for a query.

    Steps:
        1. Convert query to embedding
        2. Cosine similarity against all embeddings
        3. Graph neighbour expansion
        4. Re-rank and return top-K

    Returns: list of dicts [{rank, file, path, similarity, category}]
    """
    top_k  = top_k or cfg["retrieval"]["top_k"]
    expand = cfg["retrieval"].get("expand_graph", True)
    depth  = cfg["retrieval"].get("expansion_depth", 2)

    is_img = os.path.isfile(str(query))
    log.info(f"[{'IMAGE' if is_img else 'TEXT'}] Query: '{query}'")

    model = _get_model(cfg)

    # Step 1 — embed query
    if is_img:
        reader = _get_reader(cfg)
        raw    = " ".join(reader.readtext(query, detail=0)).strip()
        raw    = raw if raw else Path(query).stem.replace("_", " ")
        log.info(f"  OCR on query image: '{raw}'")
        qvec   = model.encode([raw], normalize_embeddings=True)
    else:
        qvec = model.encode([str(query)], normalize_embeddings=True)

    # Step 2 — cosine similarity
    sims    = cosine_similarity(qvec, embeddings)[0]
    top_idx = set(np.argsort(sims)[::-1][:top_k].tolist())

    # Step 3 — graph expansion
    if expand and G is not None:
        extra = set()
        for node in list(top_idx):
            if node in G:
                nbrs = sorted(G[node].items(),
                              key=lambda x: x[1]["weight"], reverse=True)
                for nb, _ in nbrs[:depth]:
                    if nb not in top_idx:
                        extra.add(nb)
        candidates = list(top_idx | extra)
    else:
        candidates = list(top_idx)

    # Step 4 — re-rank
    ranked = sorted(candidates, key=lambda i: sims[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked, 1):
        cat = (G.nodes[idx].get("category", get_category(image_paths[idx]))
               if G and idx in G else get_category(image_paths[idx]))
        log.info(f"  #{rank}  {labels[idx]:38s}  sim={sims[idx]:.4f}  [{cat}]")
        results.append({
            "rank": rank, "file": labels[idx],
            "path": image_paths[idx],
            "similarity": float(sims[idx]), "category": cat,
        })

    # Save
    res_dir = cfg["retrieval"]["results_dir"]
    ensure_dirs(res_dir)
    safe    = str(query).replace("/","_").replace("\\","_").replace(" ","_")[:40]
    save_json(results, os.path.join(res_dir, f"query_{safe}.json"))
    _save_grid(results, str(query), res_dir, safe)
    return results


def _save_grid(results, label, res_dir, safe):
    valid = [r for r in results if os.path.exists(r["path"])]
    if not valid:
        return
    fig, axes = plt.subplots(1, len(valid), figsize=(4 * len(valid), 4))
    if len(valid) == 1:
        axes = [axes]
    for ax, r in zip(axes, valid):
        ax.imshow(Image.open(r["path"]))
        ax.set_title(f"#{r['rank']} {r['file'][:16]}\n"
                     f"sim={r['similarity']:.3f} [{r['category']}]", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"Query: '{label}'", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, f"retrieval_{safe}.png"),
                dpi=120, bbox_inches="tight")
    plt.close()


def evaluate(image_paths, embeddings, G, labels, cfg):
    """
    Compute Precision@5, Recall, Average Precision per category.
    Target avg precision ~ 0.82  (report §11).
    """
    model   = _get_model(cfg)
    k       = cfg["evaluation"]["k"]
    queries = cfg["evaluation"]["queries"]
    ensure_dirs(cfg["retrieval"]["results_dir"])

    all_p, all_r, all_ap = [], [], []
    log_rows = []

    print(f"\n  {'Category':22s}  {'P@5':>5}  {'Recall':>7}  {'AP':>5}  "
          f"{'Ret':>5}  {'Total':>6}")
    print("  " + "─"*60)

    for cat, qtext in queries.items():
        qvec         = model.encode([qtext], normalize_embeddings=True)
        sims         = cosine_similarity(qvec, embeddings)[0]
        top_idx      = np.argsort(sims)[::-1][:k]
        relevant     = {i for i, p in enumerate(image_paths)
                        if get_category(p) == cat}
        total        = len(relevant)
        ret_rel      = [i for i in top_idx if i in relevant]
        p  = len(ret_rel) / k
        r  = len(ret_rel) / total if total else 0.0
        hits, ap = 0, 0.0
        for rank, idx in enumerate(top_idx, 1):
            if idx in relevant:
                hits += 1
                ap   += hits / rank
        ap = ap / min(k, total) if total else 0.0
        all_p.append(p); all_r.append(r); all_ap.append(ap)
        print(f"  {cat:22s}  {p:5.2f}  {r:7.2f}  {ap:5.2f}"
              f"  {len(ret_rel):>5}  {total:>6}")
        log_rows.append({"category": cat, "query": qtext,
                         "precision_at_5": p, "recall": r,
                         "average_precision": ap,
                         "retrieved_relevant": len(ret_rel),
                         "total_relevant": total})

    mp, mr, ma = np.mean(all_p), np.mean(all_r), np.mean(all_ap)
    print("  " + "─"*60)
    print(f"\n  Mean Precision@5   : {mp:.4f}")
    print(f"  Mean Recall        : {mr:.4f}")
    print(f"  Mean Avg Precision : {ma:.4f}  (target ~ 0.82)\n")

    summary = {
        "mean_precision_at_5": float(mp),
        "mean_recall":         float(mr),
        "mean_avg_precision":  float(ma),
        "graph_nodes":         G.number_of_nodes() if G else 0,
        "graph_edges":         G.number_of_edges() if G else 0,
        "graph_density":       float(nx_density(G)) if G else 0,
        "per_category":        log_rows,
    }
    save_json(summary,
              os.path.join(cfg["retrieval"]["results_dir"], "evaluation_metrics.json"))
    return summary


def nx_density(G):
    import networkx as nx
    return nx.density(G)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",  required=True)
    parser.add_argument("--top_k",  type=int, default=5)
    parser.add_argument("--eval",   action="store_true")
    args = parser.parse_args()

    cfg  = load_config()
    pkl  = cfg["embeddings"]["output_file"]
    gpkl = cfg["graph"]["output_file"]

    for path, label in [(pkl, "embeddings"), (gpkl, "graph")]:
        if not os.path.exists(path):
            log.error(f"{label} not found at '{path}'. Run main.py --demo first.")
            sys.exit(1)

    data        = load_pickle(pkl)
    image_paths = data["image_paths"]
    embeddings  = data["embeddings"]
    labels      = [os.path.basename(p) for p in image_paths]

    with open(gpkl, "rb") as f:
        G = pickle.load(f)

    search(args.query, image_paths, embeddings, G, labels, cfg, top_k=args.top_k)

    if args.eval:
        evaluate(image_paths, embeddings, G, labels, cfg)
