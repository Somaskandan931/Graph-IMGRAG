"""
tests/test_pipeline.py — Unit tests for the full pipeline.

Run:
    python -m pytest tests/test_pipeline.py -v
    python tests/test_pipeline.py
"""

import os, sys, json, tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

from src.utils.helpers import (
    load_config, ensure_dirs, collect_images,
    save_json, load_json, save_pickle, load_pickle,
    get_category, stem,
)


# ── Config ─────────────────────────────────────────────────────────────────
def test_config_loads():
    cfg = load_config()
    assert cfg["dataset"]["num_images"] == 170
    assert cfg["graph"]["knn_k"] == 5
    assert cfg["embeddings"]["model"] == "all-MiniLM-L6-v2"
    assert len(cfg["dataset"]["categories"]) == 5
    print("PASS: config loads correctly")


# ── Helpers ────────────────────────────────────────────────────────────────
def test_ensure_dirs(tmp_path):
    d = str(tmp_path / "a" / "b" / "c")
    ensure_dirs(d)
    assert os.path.isdir(d)
    print("PASS: ensure_dirs")


def test_json_io(tmp_path):
    data = {"key": "value", "n": 42, "lst": [1, 2, 3]}
    path = str(tmp_path / "test.json")
    save_json(data, path)
    assert load_json(path) == data
    print("PASS: JSON save/load")


def test_pickle_io(tmp_path):
    obj  = {"arr": np.array([1.0, 2.0, 3.0])}
    path = str(tmp_path / "test.pkl")
    save_pickle(obj, path)
    loaded = load_pickle(path)
    assert np.allclose(loaded["arr"], obj["arr"])
    print("PASS: pickle save/load")


def test_get_category():
    path = "dataset/processed/images/receipts/img_01.jpg"
    assert get_category(path) == "receipts"
    print("PASS: get_category")


def test_stem():
    assert stem("dataset/raw/img_01.jpg") == "img_01"
    print("PASS: stem")


def test_collect_images(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.jpg").write_text("x")
    (sub / "b.png").write_text("x")
    (sub / "c.txt").write_text("x")   # should be ignored
    imgs = collect_images(str(tmp_path))
    assert len(imgs) == 2
    print("PASS: collect_images")


# ── Embeddings ─────────────────────────────────────────────────────────────
def test_embedding_shape():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs  = model.encode(["STOP", "passport", "receipt total"],
                          normalize_embeddings=True)
    assert embs.shape == (3, 384)
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)
    print(f"PASS: embeddings shape={embs.shape}, L2-normalised")


# ── Graph ──────────────────────────────────────────────────────────────────
def test_knn_graph():
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity

    N, D, K = 15, 384, 5
    np.random.seed(0)
    embs        = np.random.rand(N, D).astype(np.float32)
    embs       /= np.linalg.norm(embs, axis=1, keepdims=True)
    sim         = cosine_similarity(embs)

    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    for i in range(N):
        for j in np.argsort(sim[i])[::-1][1:K+1]:
            if not G.has_edge(i, int(j)):
                G.add_edge(i, int(j), weight=float(sim[i, j]))

    assert G.number_of_nodes() == N
    assert G.number_of_edges() >= N  # at least N edges for K>=1
    for n in G.nodes():
        assert G.degree(n) >= 1
    print(f"PASS: KNN graph — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ── Retrieval ──────────────────────────────────────────────────────────────
def test_retrieval_ranking():
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs  = ["passport travel document visa", "STOP road sign traffic", "invoice receipt payment"]
    embs  = model.encode(docs, normalize_embeddings=True)
    qvec  = model.encode(["passport document"], normalize_embeddings=True)
    sims  = cosine_similarity(qvec, embs)[0]
    top   = int(np.argmax(sims))
    assert top == 0, f"Expected index 0, got {top}"
    print(f"PASS: retrieval ranking correct — top result sim={sims[top]:.4f}")


# ── Demo image creation ────────────────────────────────────────────────────
def test_demo_images(tmp_path):
    from PIL import Image, ImageDraw
    bg, fg = "#27ae60", "#1a252f"
    img    = Image.new("RGB", (420, 210), color=bg)
    draw   = ImageDraw.Draw(img)
    draw.text((30, 90), "STOP", fill=fg)
    out = str(tmp_path / "test.jpg")
    img.save(out)
    assert os.path.exists(out)
    loaded = Image.open(out)
    assert loaded.size == (420, 210)
    print("PASS: demo image creation")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    print("\n" + "="*55)
    print("  Running graph-imgrag unit tests")
    print("="*55 + "\n")

    test_config_loads()

    with tempfile.TemporaryDirectory() as td:
        test_ensure_dirs(Path(td))
        test_json_io(Path(td))
        test_pickle_io(Path(td))
        test_collect_images(Path(td))
        test_demo_images(Path(td))

    test_get_category()
    test_stem()
    test_embedding_shape()
    test_knn_graph()
    test_retrieval_ranking()

    print("\n" + "="*55)
    print("  ALL TESTS PASSED ✓")
    print("="*55 + "\n")
