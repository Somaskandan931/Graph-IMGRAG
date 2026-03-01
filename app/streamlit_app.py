"""
app/streamlit_app.py — Graph-IMGRAG (Updated for Free Image Generation)
Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import pickle
import json
import re
from pathlib import Path

# Support running from project root OR from app/ subdirectory
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, ".."))
for _p in [_root, _here]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Change working directory to project root so relative paths work
os.chdir(_root)

# Load .env file from project root
try:
    from dotenv import load_dotenv
    env_path = os.path.join(_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded .env from {env_path}")
except ImportError:
    pass

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.helpers import load_config, load_pickle, ensure_dirs

# Find config file
_cfg_path = next(
    (p for p in [
        os.path.join(_root, "configs", "config.yaml"),
        os.path.join(_root, "config.yaml"),
        os.path.join("configs", "config.yaml"),
        "config.yaml",
    ] if os.path.exists(p)), None
)

if not _cfg_path:
    st.error("❌ config.yaml not found! Please ensure config file exists.")
    st.stop()

cfg = load_config(_cfg_path)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graph-IMGRAG · Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Professional Dark Theme ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Design tokens — dark palette ── */
:root {
    --bg:        #0F1117;
    --surface:   #1A1D27;
    --surface2:  #222636;
    --surface3:  #2A2F42;
    --border:    #2E3348;
    --border-lt: #363B52;

    --ink:       #E8ECF4;
    --ink2:      #A8B3CC;
    --ink3:      #6B7899;

    /* Brand — electric blue */
    --brand:     #1A3A6E;
    --brand-mid: #1E4FA8;
    --brand-lt:  #172A4A;

    /* Accent — bright blue */
    --accent:    #4A8FFF;
    --accent-hv: #3B7FFF;
    --accent-lt: #0D1F3C;

    /* Generation — violet */
    --gen:       #7C5CFC;
    --gen-hv:    #6B4BEB;
    --gen-lt:    #1A1535;
    --gen-bd:    #3D2D7A;

    /* Free tier badges */
    --free-bg:   #1A472A;
    --free-text: #4ADE80;
    --free-border: #2A5A3A;

    /* Status */
    --green:     #34D399;
    --green-lt:  #0D2318;
    --amber:     #FBBF24;
    --amber-lt:  #231A04;

    --sans: 'Inter', system-ui, sans-serif;
    --mono: 'JetBrains Mono', 'Courier New', monospace;
    --r:     8px;
    --r-lg:  12px;
    --r-xl:  16px;
    --sh-sm: 0 1px 4px rgba(0,0,0,0.4);
    --sh:    0 4px 20px rgba(0,0,0,0.5);
}

/* ── Base ── */
html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .main > div,
[data-testid="block-container"],
.stApp {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--ink) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 5rem !important; max-width: 1280px !important; }
h1,h2,h3,h4 { color: var(--ink) !important; font-family: var(--sans) !important; font-weight: 600 !important; }
p { color: var(--ink2) !important; line-height: 1.65; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.75rem 0 !important; }
code, pre {
    font-family: var(--mono) !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
    color: #7DD3FC !important;
}

/* ── Free tier badge ── */
.free-badge {
    background: var(--free-bg);
    color: var(--free-text);
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 99px;
    border: 1px solid var(--free-border);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}

.free-badge::before {
    content: "✨";
    font-size: 0.7rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #111420 !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.75rem 1.25rem !important; }
section[data-testid="stSidebar"] *                { color: #8AA3C8 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3               { color: #D0DDEF !important; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCheckbox label { color: #6B84AA !important; }
section[data-testid="stSidebar"] hr               { border-top-color: #1E2438 !important; }
section[data-testid="stSidebar"] code,
section[data-testid="stSidebar"] pre              { background: rgba(0,0,0,0.35) !important; border-color: #1E2A45 !important; color: #60A5FA !important; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(130deg, #0D1F40 0%, #162D6B 50%, #1A3580 100%);
    border: 1px solid #1E3460;
    border-radius: var(--r-xl);
    padding: 1.75rem 2rem;
    margin-bottom: 1.75rem;
    position: relative; overflow: hidden;
}
.hero-banner .t { font-size:1.75rem; font-weight:700; color:#E8F2FF; letter-spacing:-0.02em; margin-bottom:0.35rem; }
.hero-banner .s { font-size:0.88rem; color:rgba(200,220,255,0.65); line-height:1.55; }
.hero-banner .tags { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.9rem; }
.hero-tag {
    background: rgba(74,143,255,0.12); color: rgba(180,210,255,0.85);
    font-family: var(--mono); font-size: 0.68rem;
    padding: 0.16rem 0.55rem; border-radius: 99px;
    border: 1px solid rgba(74,143,255,0.25);
}

/* ── Status pills ── */
.pill { display:inline-flex; align-items:center; gap:0.28rem; font-family:var(--mono); font-size:0.63rem; font-weight:500; padding:0.17rem 0.55rem; border-radius:99px; margin-right:0.3rem; }
.pill-ok  { background:rgba(52,211,153,0.12); color:#34D399; border:1px solid rgba(52,211,153,0.25); }
.pill-off { background:rgba(251,191,36,0.1);  color:#FBBF24; border:1px solid rgba(251,191,36,0.2); }
.pill-dot { width:5px; height:5px; border-radius:50%; background:currentColor; display:inline-block; }

/* ── Generation section ── */
.gen-wrap          { background:var(--surface); border:1.5px solid var(--gen-bd); border-radius:var(--r-xl); padding:1.75rem 2rem; margin-top:1.75rem; box-shadow:0 2px 16px rgba(124,92,252,0.12); }
.gen-header-row    { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.35rem; flex-wrap:wrap; }
.gen-title         { font-size:1.05rem; font-weight:700; color:var(--ink); }
.gen-desc          { font-size:0.84rem; color:var(--ink3); line-height:1.55; margin-bottom:1.1rem; }
.prompt-preview    { background:var(--gen-lt); border:1px solid var(--gen-bd); border-radius:var(--r); padding:0.75rem 1rem; margin:0.5rem 0 1rem; }
.prompt-preview-label { font-family:var(--mono); font-size:0.57rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#9D7FFF; margin-bottom:0.3rem; }
.prompt-preview-text  { font-family:var(--mono); font-size:0.78rem; color:var(--ink2); line-height:1.5; word-break:break-word; }
.gen-result-wrap   { background:var(--surface2); border:1px solid var(--gen-bd); border-radius:var(--r-lg); overflow:hidden; box-shadow:var(--sh); margin-top:1rem; }
.gen-result-header { display:flex; justify-content:space-between; align-items:center; padding:0.65rem 1rem; background:var(--gen-lt); border-bottom:1px solid var(--gen-bd); flex-wrap:wrap; gap:0.5rem; }
.gen-result-title  { font-size:0.8rem; font-weight:600; color:#9D7FFF; display:flex; align-items:center; gap:0.4rem; }
.gen-result-meta   { font-family:var(--mono); font-size:0.63rem; color:var(--ink3); }

/* ── Provider status ── */
.provider-status {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0 1rem;
    padding: 0.75rem;
    background: var(--surface2);
    border-radius: var(--r);
    border: 1px solid var(--border);
}
.provider-item {
    font-family: var(--mono);
    font-size: 0.65rem;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    background: var(--surface3);
    color: var(--ink2);
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}
.provider-item.active {
    background: var(--free-bg);
    color: var(--free-text);
    border: 1px solid var(--free-border);
}
.provider-item.inactive {
    opacity: 0.6;
}

/* ── Generate button — violet ── */
.gen-btn-wrap div.stButton > button {
    background: var(--gen) !important; border-color: var(--gen) !important;
    color: #fff !important; font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(124,92,252,0.35) !important;
}
.gen-btn-wrap div.stButton > button:hover {
    background: var(--gen-hv) !important; border-color: var(--gen-hv) !important;
    box-shadow: 0 4px 14px rgba(124,92,252,0.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
from src.utils.database import (
    db_exists, load_all_images, load_graph_edges,
    get_stats, DEFAULT_DB
)

DB_PATH = DEFAULT_DB

def remove_emojis(text):
    """Remove emojis from text for API calls"""
    if text:
        # Remove emojis and other non-ASCII characters
        return re.sub(r'[^\x00-\x7F]+', '', text)
    return text

@st.cache_resource(show_spinner="Loading search index…")
def _load_artifacts():
    """Load all pipeline artifacts exactly once per app session."""
    # Try database first (preferred)
    if db_exists(DB_PATH):
        try:
            image_paths, embeddings, ocr_results, _ = load_all_images(DB_PATH)
            if image_paths and len(image_paths) > 0:
                G = load_graph_edges(image_paths, DB_PATH)
                labels = [os.path.basename(p) for p in image_paths]
                return image_paths, embeddings, G, labels, ocr_results
        except Exception as e:
            st.warning(f"Database load failed: {e}, trying legacy files...")

    # Fall back to legacy pickle files
    pkl = cfg["embeddings"]["output_file"]
    gpkl = cfg["graph"]["output_file"]

    if not os.path.exists(pkl) or not os.path.exists(gpkl):
        return None, None, None, None, {}

    try:
        data = load_pickle(pkl)
        with open(gpkl, "rb") as f:
            G = pickle.load(f)
        labels = [os.path.basename(p) for p in data["image_paths"]]

        ocr_path = "outputs/embeddings/ocr_results.json"
        ocr = json.load(open(ocr_path)) if os.path.exists(ocr_path) else {}

        return data["image_paths"], data["embeddings"], G, labels, ocr
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return None, None, None, None, {}

def _sys_status():
    if db_exists(DB_PATH):
        try:
            stats = get_stats(DB_PATH)
            has_images = stats["total_images"] > 0
            has_edges = stats["total_edges"] > 0
            return {
                "ocr": has_images,
                "embeddings": has_images,
                "graph": has_edges,
                "db": True,
                "db_images": stats["total_images"],
                "db_edges": stats["total_edges"],
            }
        except:
            pass

    return {
        "ocr": os.path.exists("outputs/embeddings/ocr_results.json"),
        "embeddings": os.path.exists(cfg["embeddings"]["output_file"]),
        "graph": os.path.exists(cfg["graph"]["output_file"]),
        "db": False,
        "db_images": 0,
        "db_edges": 0,
    }

def _all_ready():
    s = _sys_status()
    return s["ocr"] and s["embeddings"] and s["graph"]

def _pill(label, ok):
    cls = "pill-ok" if ok else "pill-off"
    return f'<span class="pill {cls}"><span class="pill-dot"></span>{label}</span>'

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="margin-bottom:1.5rem">'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;'
        'text-transform:uppercase;letter-spacing:0.13em;color:rgba(255,255,255,0.38);'
        'margin-bottom:0.3rem">Graph-IMGRAG</div>'
        '<div style="font-size:1.25rem;font-weight:700;color:#F0F6FF;'
        'line-height:1.25;letter-spacing:-0.015em">'
        'Image Search<br><span style="color:#93C5FD;font-weight:400;font-size:1.05rem">+ FREE AI Generation</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

    s = _sys_status()
    st.markdown('<span class="sec-label-sb">System Status</span>', unsafe_allow_html=True)
    db_label = f"DB ({s['db_images']:,} imgs)" if s["db"] else "No DB"
    st.markdown(
        _pill("OCR", s["ocr"]) +
        _pill("Embed", s["embeddings"]) +
        _pill("Graph", s["graph"]) +
        _pill(db_label, s["db"]),
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<span class="sec-label-sb">Search Settings</span>', unsafe_allow_html=True)
    top_k = st.slider("Results to show", min_value=1, max_value=10,
                      value=cfg["retrieval"]["top_k"],
                      help="How many images to retrieve per query")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="sec-label-sb">Display Options</span>', unsafe_allow_html=True)
    show_graph = st.checkbox("Show Similarity Graph", value=False,
                             help="View the KNN graph built from image embeddings")
    show_eval = st.checkbox("Show Evaluation Metrics", value=False,
                            help="Display Precision@K, Recall, MAP per category")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="sec-label-sb">Filter by Category</span>', unsafe_allow_html=True)
    categories = ["All"] + list(cfg["dataset"].get("categories", []))
    selected_cat = st.selectbox("Category", categories, label_visibility="collapsed",
                                help="Narrow results to a specific COCO supercategory")

    st.markdown("---")

    st.markdown("""
    <div class="info-block">
        <div class="info-title">Pipeline</div>
        <div class="info-row"><span class="info-key">OCR Engine</span><span class="info-val">EasyOCR</span></div>
        <div class="info-row"><span class="info-key">Embedding</span><span class="info-val">MiniLM-L6</span></div>
        <div class="info-row"><span class="info-key">Dim</span><span class="info-val">384</span></div>
        <div class="info-row"><span class="info-key">Graph k</span><span class="info-val">5 (KNN)</span></div>
        <div class="info-row"><span class="info-key">Similarity</span><span class="info-val">Cosine</span></div>
    </div>
    <div class="info-block">
        <div class="info-title">Dataset</div>
        <div class="info-row"><span class="info-key">Source</span><span class="info-val">MS-COCO 2017</span></div>
        <div class="info-row"><span class="info-key">Split</span><span class="info-val">val2017</span></div>
        <div class="info-row"><span class="info-key">Max images</span><span class="info-val">500</span></div>
        <div class="info-row"><span class="info-key">Categories</span><span class="info-val">12</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="sec-label-sb">Quick Start</span>', unsafe_allow_html=True)
    st.code("# Build index once\npython main.py --build\n\n# Search instantly\npython main.py --query \"dog park\"", language="bash")

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
  <div class='t'>🔍 Graph-Based Image Retrieval + FREE AI Generation</div>
  <div class='s'>Finds images by reading text embedded inside them — signs, labels, receipts, documents.<br>
  Then generates a brand-new AI image using completely free providers (no API keys needed!)</div>
  <div class='tags'>
    <span class='hero-tag'>🐕 dog park</span>
    <span class='hero-tag'>🚗 car road</span>
    <span class='hero-tag'>🍕 food pizza</span>
    <span class='hero-tag'>👤 person crowd</span>
    <span class='hero-tag'>⚽ sports ball</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Not-ready gate ─────────────────────────────────────────────────────────────
if not _all_ready():
    s = _sys_status()
    st.markdown("""
    <div class="not-ready">
        <div class="not-ready-title">Search index not built yet</div>
        <div class="not-ready-sub">
            Your COCO images are already on disk — no downloading needed.
            Run the pipeline once to extract text, build embeddings, and
            save the search index.
        </div>
        <ul class="step-list">
            <li><span class="step-num">1</span>python main.py --build</li>
            <li><span class="step-num">2</span>Wait for OCR + embedding + graph (10–30 min first run)</li>
            <li><span class="step-num">3</span>Refresh this page — searches are instant</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("OCR Results", "✓ Ready" if s["ocr"] else "✗ Missing")
    c2.metric("Embeddings", "✓ Ready" if s["embeddings"] else "✗ Missing")
    c3.metric("Search Graph", "✓ Ready" if s["graph"] else "✗ Missing")
    st.stop()

# ── Load artifacts ──────────────────────────────────────────────────────────────
image_paths, embeddings, G, labels, ocr_results = _load_artifacts()

if image_paths is None or len(image_paths) == 0:
    st.error("No images found in database. Please run: python main.py --build")
    st.stop()

# ── Category filter ─────────────────────────────────────────────────────────────
if selected_cat != "All":
    filtered = [
        (p, e, l)
        for p, e, l in zip(image_paths, embeddings, labels)
        if os.path.basename(os.path.dirname(p)) == selected_cat
    ]
    if filtered:
        f_paths, f_embs, f_labels = zip(*filtered)
        image_paths_f = list(f_paths)
        embeddings_f = np.array(f_embs)
        labels_f = list(f_labels)
    else:
        st.info(f"No images found in category **{selected_cat}**. Showing all.")
        image_paths_f, embeddings_f, labels_f = image_paths, embeddings, labels
else:
    image_paths_f, embeddings_f, labels_f = image_paths, embeddings, labels

# ── Search panel ────────────────────────────────────────────────────────────────
st.markdown('<div class="search-card">', unsafe_allow_html=True)

with st.expander("ℹ️ How to use"):
    st.markdown("""
**Two ways to search:**

**1. Text query** — Type any word or phrase. The system finds images whose
embedded text is semantically similar to your query.

**2. Upload an image** — Upload any image with visible text (sign, label,
document, screenshot). OCR extracts the text automatically.
    """)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<span class="sec-label">Search by Text</span>', unsafe_allow_html=True)

st.markdown(
    '<div class="chip-row">'
    '<span class="chip">🐕 dog park</span>'
    '<span class="chip">🚗 car road traffic</span>'
    '<span class="chip">🍕 food pizza</span>'
    '<span class="chip">👤 person crowd</span>'
    '<span class="chip">⚽ sports ball</span>'
    '<span class="chip">🪑 furniture chair</span>'
    '<span class="chip">📱 laptop screen</span>'
    '<span class="chip">🐦 bird animal</span>'
    '</div>',
    unsafe_allow_html=True
)

query_text = st.text_input(
    "Text query",
    placeholder="e.g. dog park outdoor / car road traffic / pizza on plate",
    label_visibility="collapsed",
)

st.markdown('<div class="or-divider"><span>or search by image</span></div>', unsafe_allow_html=True)

st.markdown('<span class="sec-label">Upload a Query Image</span>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

extracted_text = ""
tmp_path = "outputs/retrieval_results/query_upload.jpg"

if uploaded:
    col_prev, col_gap, col_ocr = st.columns([1, 0.1, 2])
    with col_prev:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
    with col_ocr:
        ensure_dirs("outputs/retrieval_results")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getvalue())

        with st.spinner("Extracting text from your image…"):
            try:
                import easyocr
                reader = easyocr.Reader(cfg["ocr"]["languages"], gpu=cfg["ocr"].get("gpu", False))
                extracted_text = " ".join(reader.readtext(tmp_path, detail=0)).strip()
            except Exception as e:
                st.warning(f"OCR failed: {e}")
                extracted_text = ""

        if extracted_text:
            st.success("Text detected successfully!")
            st.markdown(
                f'<div class="ocr-box">'
                f'  <div class="ocr-label">Detected Text</div>'
                f'  <div class="ocr-text">{extracted_text}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("No text detected. The filename will be used as a fallback query.")

st.markdown("<br>", unsafe_allow_html=True)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    run_search = st.button("🔍 Search", type="primary", use_container_width=True)
with col_info:
    n_imgs = len(image_paths_f)
    cat_str = f" in **{selected_cat}**" if selected_cat != "All" else ""
    st.markdown(
        f'<p style="font-size:0.78rem;color:#8c8880;margin-top:0.6rem">'
        f'Searching <strong>{n_imgs:,}</strong> images{cat_str} · '
        f'returning top <strong>{top_k}</strong></p>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Execute search & display results ────────────────────────────────────────────
if run_search:
    from src.retrieval.search import search as _search

    if not uploaded and not query_text.strip():
        st.error("⚠️ Please enter a search query or upload an image with text.")
        st.stop()

    if uploaded:
        query = tmp_path
        display_query = extracted_text or "uploaded_image"
        q_label = f'image — "{extracted_text[:60]}{"…" if len(extracted_text) > 60 else ""}"'
    else:
        query = query_text.strip()
        display_query = query
        q_label = f'"{query}"'

    with st.spinner("Searching the similarity graph…"):
        try:
            results = _search(
                query, image_paths_f, embeddings_f,
                G, labels_f, cfg, top_k=top_k
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            results = []

    # Store results in session state
    st.session_state["_imgrag_results"] = results
    st.session_state["_imgrag_query"] = display_query if uploaded else query_text.strip()

    if not results:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔎</div>
            <div class="empty-title">No results found</div>
            <div class="empty-sub">Try a different keyword or adjust the category filter.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown(
            f'<div class="results-header">'
            f'  <span class="results-count">Top {len(results)} results</span>'
            f'  <span class="results-query">for {q_label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        def _render_row(row_results):
            cols = st.columns(len(row_results))
            for col, r in zip(cols, row_results):
                ocr_raw = ocr_results.get(r["path"], "").strip()
                ocr_disp = (ocr_raw[:100] + "…") if len(ocr_raw) > 100 else ocr_raw
                with col:
                    if os.path.exists(r["path"]):
                        st.image(r["path"], use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="height:130px;background:#f4f3f0;border:1px solid #e8e6e1;border-radius:8px;display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:0.65rem;color:#b8b5af">no image</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown(
                        f'<div class="result-meta">'
                        f'  <span class="result-rank">#{r["rank"]}</span>'
                        f'  <div class="result-fname">{r["file"]}</div>'
                        f'  <span class="result-cat">{r["category"]}</span>'
                        f'  <div class="result-sim-label">Similarity: '
                        f'    <span class="result-sim-val">{r["similarity"]:.3f}</span>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.progress(min(float(r["similarity"]), 1.0))
                    ocr_html = ocr_disp if ocr_disp else '<em style="color:#b8b5af">no text detected</em>'
                    st.markdown(
                        f'<div class="result-ocr-wrap">'
                        f'  <div class="result-ocr-label">Detected Text</div>'
                        f'  <div class="result-ocr-text">{ocr_html}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        _render_row(results[:5])
        if len(results) > 5:
            st.markdown("<br>", unsafe_allow_html=True)
            _render_row(results[5:])

# ══════════════════════════════════════════════════════════════════════════════
# ✨ FREE AI IMAGE GENERATION — appears after every successful search
# ══════════════════════════════════════════════════════════════════════════════
_results = st.session_state.get("_imgrag_results", [])
_display_query = st.session_state.get("_imgrag_query", "")

if _results:
    try:
        from src.generative.image_gen import (
            build_prompt, generate_image_bytes, save_generated_image,
            get_style_names, get_aspect_ratios, api_status,
        )
        import random as _random

        st.markdown("---")
        st.markdown('<div class="gen-wrap">', unsafe_allow_html=True)
        st.markdown(
            '<div class="gen-header-row">'
            '  <span class="gen-title">✨ Generate an AI Image</span>'
            '  <span class="free-badge">100% FREE · Multiple providers</span>'
            '</div>'
            '<p class="gen-desc">Generate a brand-new image inspired by your search query. '
            'The system automatically tries multiple free providers and falls back to pattern generation if needed.</p>',
            unsafe_allow_html=True
        )

        # Show provider status
        status = api_status()
        st.markdown(
            f'<div class="provider-status">'
            f'  <span class="provider-item {"active" if status["huggingface"] else "inactive"}">🤗 HuggingFace {"✓" if status["huggingface"] else "(optional)"}</span>'
            f'  <span class="provider-item {"active" if status["prodia"] else "inactive"}">⚡ Prodia {"✓" if status["prodia"] else "(optional)"}</span>'
            f'  <span class="provider-item active">🌸 Pollinations (free)</span>'
            f'  <span class="provider-item active">🧠 felo.ai (free)</span>'
            f'  <span class="provider-item active">🖼️ Fallback (always works)</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Controls
        _c1, _c2, _c3 = st.columns([2, 1.5, 1.5])
        with _c1:
            st.markdown('<span style="font-family:monospace;font-size:0.6rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#6b7280;display:block;margin-bottom:0.45rem;padding-bottom:0.35rem;border-bottom:1px solid #e5e7eb">Generation Prompt</span>', unsafe_allow_html=True)
            _custom = st.text_area(
                "custom", placeholder="Optional extra detail e.g. 'golden hour, wide angle'",
                height=72, label_visibility="collapsed",
            )
        with _c2:
            st.markdown('<span style="font-family:monospace;font-size:0.6rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#6b7280;display:block;margin-bottom:0.45rem;padding-bottom:0.35rem;border-bottom:1px solid #e5e7eb">Style</span>', unsafe_allow_html=True)
            _style = st.selectbox("style", get_style_names(), label_visibility="collapsed")
        with _c3:
            st.markdown('<span style="font-family:monospace;font-size:0.6rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#6b7280;display:block;margin-bottom:0.45rem;padding-bottom:0.35rem;border-bottom:1px solid #e5e7eb">Aspect Ratio</span>', unsafe_allow_html=True)
            _ratio_map = get_aspect_ratios()
            _ratio_name = st.selectbox("ratio", list(_ratio_map.keys()), label_visibility="collapsed")
        _gen_w, _gen_h = _ratio_map[_ratio_name]

        # Prompt preview
        _clean_query = remove_emojis(_display_query)
        _auto_prompt = build_prompt(
            user_query=_clean_query,
            results=_results,
            ocr_results=ocr_results,
            style=_style,
            extra_instruction=_custom,
        )
        st.markdown(
            f'<div class="prompt-preview">'
            f'<div class="prompt-preview-label">📝 Generation Prompt</div>'
            f'<div class="prompt-preview-text">{_auto_prompt}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Generate row
        _gb, _sb, _saveb = st.columns([2, 1.5, 1.5])
        with _gb:
            st.markdown('<div class="gen-btn-wrap">', unsafe_allow_html=True)
            _do_gen = st.button(
                "✨ Generate FREE Image", use_container_width=True,
                help="Generates a new image using multiple free providers",
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with _sb:
            _fixed = st.checkbox("Fixed seed", value=False)
            _seed_val = st.number_input(
                "Seed", min_value=0, max_value=999999,
                value=int(st.session_state.get("_gen_seed", 42)),
                disabled=not _fixed, label_visibility="collapsed",
            )
        with _saveb:
            _save_disk = st.checkbox("Save to disk", value=False, help="Saves to outputs/generated/")

        # Execute generation
        if _do_gen:
            _seed = int(_seed_val) if _fixed else _random.randint(1, 999999)
            st.session_state["_gen_seed"] = _seed

            progress_placeholder = st.empty()
            progress_placeholder.info("🔄 Trying multiple free providers...")

            with st.spinner(f"Generating with free providers... (seed {_seed})"):
                try:
                    _img_bytes = generate_image_bytes(
                        _auto_prompt, width=_gen_w, height=_gen_h,
                        seed=_seed, timeout=90,
                    )

                    progress_placeholder.empty()

                    _saved = ""
                    if _save_disk:
                        _saved = save_generated_image(_img_bytes, _display_query, _style)

                    st.session_state["_gen_bytes"] = _img_bytes
                    st.session_state["_gen_meta"] = {
                        "query": _display_query, "style": _style,
                        "ratio": _ratio_name, "seed": _seed,
                        "saved": _saved, "prompt": _auto_prompt,
                    }
                except Exception as _e:
                    progress_placeholder.empty()
                    st.error(f"⚠️ Generation error: {str(_e)[:100]}")

        # Show result
        if st.session_state.get("_gen_bytes"):
            _m = st.session_state.get("_gen_meta", {})
            _img_data = st.session_state["_gen_bytes"]

            # Determine provider
            if len(_img_data) < 5000:
                provider_used = "Fallback Generator"
                provider_icon = "🖼️"
            else:
                provider_used = "Free API"
                provider_icon = "✨"

            st.markdown(
                f'<div class="gen-result-wrap">'
                f'<div class="gen-result-header">'
                f'<span class="gen-result-title">{provider_icon} Generated Image <span class="free-badge">{provider_used}</span></span>'
                f'<span class="gen-result-meta">{_m.get("style","")} · {_m.get("ratio","").split("(")[0].strip()} · seed {_m.get("seed","")}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.image(_img_data, caption=f'"{_m.get("query","")}" — {_m.get("style","")} style', use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            _dl, _msg = st.columns([1, 3])
            with _dl:
                st.download_button(
                    "⬇ Download Image",
                    data=_img_data,
                    file_name=f'imgrag_{_m.get("query","gen")[:20].replace(" ","_")}_{_m.get("seed","")}.png',
                    mime="image/png",
                    use_container_width=True,
                )
            with _msg:
                if _m.get("saved"):
                    st.success(f'Saved → `{_m["saved"]}`')

                if provider_used == "Fallback Generator":
                    st.info(
                        "💡 **Tip:** Add a HuggingFace token to `.env` for better images!\n\n"
                        "```bash\n"
                        "echo \"HF_TOKEN=your_token_here\" >> .env\n"
                        "```\n"
                        "Get free token: https://huggingface.co/settings/tokens"
                    )

        st.markdown('</div>', unsafe_allow_html=True)

    except ImportError as e:
        st.warning(f"Image generation module not available: {e}")
    except Exception as e:
        st.warning(f"Image generation temporarily unavailable: {e}")

# ── Evaluation ──────────────────────────────────────────────────────────────────
if show_eval:
    st.markdown("---")
    st.markdown("### 📊 Evaluation Metrics")

    eval_json = "outputs/retrieval_results/evaluation_metrics.json"
    if os.path.exists(eval_json):
        with open(eval_json) as f:
            ev = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean P@5", f"{ev.get('mean_precision_at_5', 0):.3f}")
        c2.metric("Mean Recall", f"{ev.get('mean_recall', 0):.3f}")
        c3.metric("Mean AP", f"{ev.get('mean_avg_precision', 0):.3f}")
        c4.metric("Graph Edges", ev.get("graph_edges", "—"))

        if "per_category" in ev:
            import pandas as pd
            rows = [{
                "Category": row["category"],
                "Query": row["query"],
                "Precision@5": f"{row['precision_at_5']:.2f}",
                "Recall": f"{row['recall']:.2f}",
                "Avg Precision": f"{row['average_precision']:.2f}",
            } for row in ev["per_category"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation results yet. Run `python main.py --build --eval`")

# ── Graph ────────────────────────────────────────────────────────────────────────
if show_graph:
    st.markdown("---")
    st.markdown("### 🕸️ Similarity Graph")

    png = cfg["graph"]["output_png"]
    gpkl = cfg["graph"]["output_file"]

    if os.path.exists(png):
        if os.path.exists(gpkl):
            try:
                import networkx as nx
                with open(gpkl, "rb") as f:
                    Gv = pickle.load(f)
                n = Gv.number_of_nodes()
                e = Gv.number_of_edges()
                deg = sum(d for _, d in Gv.degree()) / n if n else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Nodes (Images)", n)
                c2.metric("Edges", e)
                c3.metric("Avg Degree", f"{deg:.1f}")
                st.markdown("<br>", unsafe_allow_html=True)
            except:
                pass

        st.image(png, use_container_width=True, caption="KNN Similarity Graph")
    else:
        st.info("Graph image not found. Build the pipeline first.")