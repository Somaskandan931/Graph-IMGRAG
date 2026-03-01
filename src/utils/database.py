"""
src/utils/database.py
SQLite-backed storage for the graph-imgrag pipeline.

Tables:
    images      — image metadata + OCR text + embedding (binary blob)
    graph_edges — KNN similarity edges
    ocr_cache   — per-image OCR text (fast lookup)
    run_meta    — pipeline run info and timestamps

All embeddings are stored as raw numpy float32 bytes (binary blob).
The graph edges are stored as (src_id, dst_id, weight) rows.
"""

import os
import io
import sqlite3
import pickle
import json
import time
import numpy as np
from contextlib import contextmanager
from src.utils.helpers import get_logger, ensure_dirs

log = get_logger("Database")

DEFAULT_DB = "outputs/graph_imgrag.db"


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def get_conn(db_path: str = DEFAULT_DB):
    """Context manager — yields an open SQLite connection with WAL mode."""
    ensure_dirs(os.path.dirname(db_path))
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    filename    TEXT    NOT NULL,
    category    TEXT    NOT NULL DEFAULT 'unknown',
    ocr_text    TEXT    NOT NULL DEFAULT '',
    embedding   BLOB,                          -- float32 numpy array as bytes
    width       INTEGER,
    height      INTEGER,
    created_at  REAL    NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS graph_edges (
    src_id      INTEGER NOT NULL REFERENCES images(id),
    dst_id      INTEGER NOT NULL REFERENCES images(id),
    weight      REAL    NOT NULL,
    PRIMARY KEY (src_id, dst_id)
);

CREATE TABLE IF NOT EXISTS run_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_images_category ON images(category);
CREATE INDEX IF NOT EXISTS idx_images_path     ON images(path);
CREATE INDEX IF NOT EXISTS idx_edges_src       ON graph_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst       ON graph_edges(dst_id);
"""

def init_db(db_path: str = DEFAULT_DB):
    """Create all tables if they don't already exist."""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
    log.info(f"Database initialised → '{db_path}'")


def db_exists(db_path: str = DEFAULT_DB) -> bool:
    """Return True if the database exists and has image rows."""
    if not os.path.exists(db_path):
        return False
    try:
        with get_conn(db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        return n > 0
    except Exception:
        return False


# ── Numpy ↔ BLOB helpers ──────────────────────────────────────────────────────

def _arr_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()

def _blob_to_arr(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


# ── Write operations ──────────────────────────────────────────────────────────

def upsert_images(image_paths: list, ocr_results: dict,
                  embeddings: np.ndarray, categories: dict,
                  db_path: str = DEFAULT_DB):
    """
    Insert or update all images, their OCR text, and embeddings.

    Args:
        image_paths : list of absolute image paths
        ocr_results : {path: text}
        embeddings  : np.ndarray shape (N, D)
        categories  : {path: category_name}
    """
    log.info(f"Upserting {len(image_paths)} images into database ...")
    rows = []
    for i, path in enumerate(image_paths):
        emb_blob = _arr_to_blob(embeddings[i])
        rows.append((
            path,
            os.path.basename(path),
            categories.get(path, "unknown"),
            ocr_results.get(path, ""),
            emb_blob,
        ))

    with get_conn(db_path) as conn:
        conn.executemany("""
            INSERT INTO images (path, filename, category, ocr_text, embedding)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                category  = excluded.category,
                ocr_text  = excluded.ocr_text,
                embedding = excluded.embedding
        """, rows)

    log.info(f"  {len(rows)} image rows saved")


def upsert_graph_edges(G, image_paths: list, db_path: str = DEFAULT_DB):
    """
    Store all KNN graph edges.
    Node indices in G correspond to positions in image_paths.
    """
    log.info("Storing graph edges ...")

    # Build path → db_id mapping
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT id, path FROM images").fetchall()
        path_to_id = {r["path"]: r["id"] for r in rows}

        # Clear old edges
        conn.execute("DELETE FROM graph_edges")

        edge_rows = []
        for u, v, data in G.edges(data=True):
            src_path = image_paths[u]
            dst_path = image_paths[v]
            src_id = path_to_id.get(src_path)
            dst_id = path_to_id.get(dst_path)
            if src_id and dst_id:
                w = data.get("weight", 0.0)
                edge_rows.append((src_id, dst_id, w))
                edge_rows.append((dst_id, src_id, w))  # undirected

        conn.executemany("""
            INSERT OR REPLACE INTO graph_edges (src_id, dst_id, weight)
            VALUES (?, ?, ?)
        """, edge_rows)

    log.info(f"  {len(edge_rows)//2} edges stored (bidirectional)")


def set_meta(key: str, value, db_path: str = DEFAULT_DB):
    """Store a pipeline metadata key-value pair."""
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO run_meta (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value      = excluded.value,
                updated_at = excluded.updated_at
        """, (key, json.dumps(value), time.time()))


# ── Read operations ───────────────────────────────────────────────────────────

def load_all_images(db_path: str = DEFAULT_DB):
    """
    Load all images from the database.

    Returns:
        image_paths : list[str]
        embeddings  : np.ndarray  (N, D)
        ocr_results : dict {path: text}
        categories  : dict {path: category}
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT path, filename, category, ocr_text, embedding FROM images ORDER BY id"
        ).fetchall()

    if not rows:
        return [], np.array([]), {}, {}

    image_paths = [r["path"]    for r in rows]
    ocr_results = {r["path"]: r["ocr_text"]  for r in rows}
    categories  = {r["path"]: r["category"]  for r in rows}
    embeddings  = np.array([_blob_to_arr(r["embedding"]) for r in rows])

    log.info(f"Loaded {len(image_paths)} images from database  "
             f"(embeddings: {embeddings.shape})")
    return image_paths, embeddings, ocr_results, categories


def load_embeddings(db_path: str = DEFAULT_DB):
    """Convenience: return only (image_paths, embeddings)."""
    paths, embs, _, _ = load_all_images(db_path)
    return paths, embs


def load_graph_edges(image_paths: list, db_path: str = DEFAULT_DB):
    """
    Reconstruct a NetworkX graph from stored edges.

    Returns: nx.Graph
    """
    import networkx as nx
    from src.utils.helpers import get_category

    with get_conn(db_path) as conn:
        img_rows = conn.execute(
            "SELECT id, path, category FROM images ORDER BY id"
        ).fetchall()
        edge_rows = conn.execute(
            "SELECT src_id, dst_id, weight FROM graph_edges"
        ).fetchall()

    id_to_idx  = {}
    G = nx.Graph()
    for img in img_rows:
        idx = image_paths.index(img["path"]) if img["path"] in image_paths else None
        if idx is not None:
            id_to_idx[img["id"]] = idx
            G.add_node(idx,
                       label=os.path.basename(img["path"]),
                       path=img["path"],
                       category=img["category"])

    for edge in edge_rows:
        u = id_to_idx.get(edge["src_id"])
        v = id_to_idx.get(edge["dst_id"])
        if u is not None and v is not None and u < v:
            G.add_edge(u, v, weight=edge["weight"])

    log.info(f"Reconstructed graph: {G.number_of_nodes()} nodes, "
             f"{G.number_of_edges()} edges")
    return G


def get_meta(key: str, db_path: str = DEFAULT_DB):
    """Retrieve a pipeline metadata value."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM run_meta WHERE key = ?", (key,)
        ).fetchone()
    return json.loads(row["value"]) if row else None


def get_stats(db_path: str = DEFAULT_DB) -> dict:
    """Return basic database stats."""
    with get_conn(db_path) as conn:
        n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        n_edges  = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        cats     = conn.execute(
            "SELECT category, COUNT(*) as n FROM images GROUP BY category ORDER BY n DESC"
        ).fetchall()
    return {
        "total_images": n_images,
        "total_edges":  n_edges // 2,
        "categories":   {r["category"]: r["n"] for r in cats},
    }


def search_by_text(query_text: str, limit: int = 20, db_path: str = DEFAULT_DB) -> list:
    """
    Simple full-text search on OCR text (fast, no embedding needed).
    Returns list of matching image rows.
    """
    with get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT path, filename, category, ocr_text
            FROM images
            WHERE ocr_text LIKE ?
            LIMIT ?
        """, (f"%{query_text}%", limit)).fetchall()
    return [dict(r) for r in rows]


def filter_by_category(category: str, db_path: str = DEFAULT_DB):
    """
    Return (image_paths, embeddings) filtered to a specific category.
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT path, embedding FROM images WHERE category = ? ORDER BY id",
            (category,)
        ).fetchall()

    if not rows:
        return [], np.array([])

    paths = [r["path"] for r in rows]
    embs  = np.array([_blob_to_arr(r["embedding"]) for r in rows])
    return paths, embs


def list_categories(db_path: str = DEFAULT_DB) -> list:
    """Return all unique categories in the database."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM images ORDER BY category"
        ).fetchall()
    return [r["category"] for r in rows]
