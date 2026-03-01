"""
src/utils/helpers.py
Shared utility functions used across the entire project.
"""

import os
import json
import pickle
import logging
import yaml
from pathlib import Path


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config. Looks in project root automatically."""
    # Handle running from any subdirectory
    candidates = [
        path,
        os.path.join(os.path.dirname(__file__), "../../", path),
        os.path.join(os.path.dirname(__file__), "../../../", path),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Config not found: {path}")


# ── Directory helpers ─────────────────────────────────────────────────────────

def ensure_dirs(*paths):
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


# ── Image collection ──────────────────────────────────────────────────────────

def collect_images(directory: str) -> list:
    """Recursively find all jpg/png images under directory."""
    exts = {".jpg", ".jpeg", ".png"}
    images = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if Path(f).suffix.lower() in exts:
                images.append(os.path.join(root, f))
    return sorted(images)


# ── File I/O ──────────────────────────────────────────────────────────────────

def save_json(data, path: str):
    ensure_dirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_pickle(obj, path: str):
    ensure_dirs(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Path helpers ──────────────────────────────────────────────────────────────

# Optional COCO metadata cache: {image_path: supercategory}
_coco_meta_cache: dict = {}

def set_coco_meta(meta: dict):
    """Populate the COCO metadata cache so get_category() can use it."""
    global _coco_meta_cache
    _coco_meta_cache = meta


def get_category(image_path: str) -> str:
    """
    Return the supercategory for an image.
    Prefers COCO metadata (if loaded via set_coco_meta) and falls back
    to the parent folder name, which is the COCO supercategory when images
    are organised by load_coco_dataset().
    """
    if _coco_meta_cache and image_path in _coco_meta_cache:
        return _coco_meta_cache[image_path]
    return Path(image_path).parent.name


def stem(path: str) -> str:
    return Path(path).stem