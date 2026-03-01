"""
src/utils/helpers.py
Shared utilities for the graph-imgrag pipeline.
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path

import yaml
import numpy as np

# ── COCO metadata cache (populated by coco_loader) ────────────────────────────
_COCO_META: dict = {}   # {image_path: supercategory}

def set_coco_meta(meta: dict):
    global _COCO_META
    _COCO_META = meta

# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = None) -> dict:
    """Load YAML config. Searches standard locations if path is not given."""
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, "../../.."))
    candidates = ([path] if path else []) + [
        "configs/config.yaml",
        "config.yaml",
        os.path.join(_root, "configs", "config.yaml"),
        os.path.join(_root, "config.yaml"),
        os.path.join("..", "configs", "config.yaml"),
        os.path.join("..", "config.yaml"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        "config.yaml not found. Searched:\n" +
        "\n".join(f"  {os.path.abspath(p)}" for p in candidates if p)
    )


# ── Directories ───────────────────────────────────────────────────────────────

def ensure_dirs(*dirs):
    """Create directories (and parents) if they do not already exist."""
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


# ── Image collection ──────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def collect_images(root: str) -> list:
    """
    Recursively collect all image files under *root*.
    Returns a sorted list of absolute paths.
    """
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def save_json(data, path: str, indent: int = 2):
    ensure_dirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=_json_default)

def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

def load_json(path: str):
    with open(path) as f:
        return json.load(f)


# ── Pickle helpers ────────────────────────────────────────────────────────────

def save_pickle(obj, path: str):
    ensure_dirs(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Path helpers ──────────────────────────────────────────────────────────────

def stem(path: str) -> str:
    """Return the filename stem (no extension) for *path*."""
    return Path(path).stem


def get_category(image_path: str) -> str:
    """
    Return the supercategory for an image.

    Priority:
      1. COCO metadata cache (set by coco_loader)
      2. Parent directory name (folder-based categories)
      3. 'unknown'
    """
    # Normalise path for lookup
    norm = os.path.normpath(image_path)
    if _COCO_META:
        # Try exact match first
        if norm in _COCO_META:
            return _COCO_META[norm]
        # Try with forward slashes
        fwd = image_path.replace("\\", "/")
        if fwd in _COCO_META:
            return _COCO_META[fwd]
        # Try matching by filename
        fname = os.path.basename(image_path)
        for k, v in _COCO_META.items():
            if os.path.basename(k) == fname:
                return v

    # Fall back: use parent folder name
    parent = Path(image_path).parent.name
    return parent if parent not in ("", ".", "images") else "unknown"