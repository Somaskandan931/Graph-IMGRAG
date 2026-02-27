"""
app/streamlit_app.py — Streamlit Web UI

Run:
    streamlit run app/streamlit_app.py
"""

import os, sys, pickle
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.helpers import load_config, load_pickle, ensure_dirs

cfg = load_config()

st.set_page_config(
    page_title="graph-imgrag | Image Retrieval",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 graph-imgrag")
st.sidebar.caption("OCR → Embeddings → KNN Graph → Retrieval")
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "🏠 Home",
    "🚀 Build Pipeline",
    "🔍 Search",
    "📊 Evaluate",
    "🗺️ Graph",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Quick start:**
```bash
python main.py --coco
streamlit run app/streamlit_app.py
```
""")



# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_artifacts():
    pkl  = cfg["embeddings"]["output_file"]
    gpkl = cfg["graph"]["output_file"]
    if not os.path.exists(pkl) or not os.path.exists(gpkl):
        return None, None, None, None
    data        = load_pickle(pkl)
    image_paths = data["image_paths"]
    embeddings  = data["embeddings"]
    labels      = [os.path.basename(p) for p in image_paths]
    with open(gpkl, "rb") as f:
        G = pickle.load(f)
    return image_paths, embeddings, G, labels


def _pipeline_warning():
    st.warning("⚠️ Pipeline not built yet. Go to **🚀 Build Pipeline** first.")
    st.code("python main.py --demo", language="bash")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("Graph-Based Image Retrieval System")
    st.markdown("**Text-in-image search using OCR + Semantic Embeddings + KNN Graph**")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Pipeline")
        st.markdown("""
        ```
        Raw Images
             ↓
        OCR (EasyOCR)
             ↓
        Text Embeddings (all-MiniLM-L6-v2)
             ↓
        KNN Similarity Graph (k=5)
             ↓
        Query Retrieval
             ↓
        Retrieved Images
        ```
        """)

    with col2:
        st.subheader("📁 Dataset (MS-COCO)")
        st.table({
            "Category":  ["person", "vehicle", "outdoor", "animal",
                          "sports", "food", "furniture", "electronic",
                          "kitchen", "indoor", "**Total (default)**"],
            "Images":    ["~100", "~60", "~40", "~60",
                          "~40", "~40", "~30", "~30",
                          "~20", "~30", "**≤ 500**"],
        })

    st.subheader("🛠️ Tech Stack")
    st.table({
        "Tool":    ["EasyOCR", "all-MiniLM-L6-v2", "NetworkX",
                   "Scikit-learn", "Streamlit"],
        "Purpose": ["Text extraction", "Semantic embeddings", "KNN graph",
                   "Cosine similarity", "Web UI"],
    })


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀 Build Pipeline":
    st.title("🚀 Build Pipeline")

    st.info("This runs: COCO Download → OCR → Embeddings → Graph → Visualisation")

    st.markdown("""
    **COCO Dataset settings** (edit `configs/config.yaml` to change):
    """)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        split_choice = st.selectbox(
            "COCO Split",
            ["val2017", "train2017"],
            index=0,
            help="val2017 ~1 GB · train2017 ~18 GB"
        )
    with col_b:
        max_imgs = st.number_input(
            "Max Images", min_value=10, max_value=5000, value=500, step=50,
            help="Number of COCO images to download and process"
        )
    with col_c:
        force = st.checkbox("Force re-run (clear cache)", value=False)

    download_coco = st.checkbox("Download / refresh COCO images", value=True)

    if st.button("▶ Start Pipeline", type="primary"):
        import shutil

        # Update config in memory
        cfg["dataset"]["coco_split"] = split_choice
        cfg["dataset"]["max_images"] = int(max_imgs)

        if force:
            for d in ["outputs/embeddings", "outputs/graphs"]:
                if os.path.exists(d):
                    shutil.rmtree(d)

        # COCO Download
        if download_coco:
            with st.spinner(f"Downloading COCO {split_choice} (up to {max_imgs} images) …"):
                from src.utils.coco_loader import load_coco_dataset, load_coco_meta
                from src.utils.helpers import set_coco_meta
                n = load_coco_dataset(cfg)
                set_coco_meta(load_coco_meta(cfg))
            st.success(f"✅ COCO dataset ready — {n} images")
        else:
            from src.utils.coco_loader import load_coco_meta
            from src.utils.helpers import set_coco_meta
            set_coco_meta(load_coco_meta(cfg))

        # OCR
        with st.spinner("Running OCR on all images …"):
            from src.ocr.extract_text import run_ocr
            ocr_results = run_ocr(cfg)
        st.success(f"✅ OCR complete — {len(ocr_results)} images")

        # Embeddings
        with st.spinner("Generating embeddings …"):
            from src.embeddings.generate_embeddings import generate
            image_paths, embeddings = generate(ocr_results, cfg)
        st.success(f"✅ Embeddings: {embeddings.shape}")

        # Graph
        with st.spinner("Building KNN graph …"):
            from src.graph.build_graph import build, visualize
            G, _, labels = build(image_paths, embeddings, cfg)
        st.success(f"✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Visualise
        with st.spinner("Rendering graph image …"):
            visualize(G, labels, cfg)
        st.success("✅ Graph visualisation saved")

        st.balloons()
        st.success("🎉 Pipeline complete! Use the sidebar to Search or Evaluate.")




# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Search":
    st.title("🔍 Search Images")

    image_paths, embeddings, G, labels = _load_artifacts()
    if image_paths is None:
        _pipeline_warning()

    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_input("Enter a text query",
                                    placeholder="e.g. dog park  /  car road  /  person crowd  /  pizza plate")
    with col2:
        top_k = st.slider("Top-K", 1, 10, cfg["retrieval"]["top_k"])

    uploaded = st.file_uploader("Or upload a query image",
                                  type=["jpg", "jpeg", "png"])

    if st.button("🔎 Search", type="primary"):
        from src.retrieval.search import search

        if uploaded:
            tmp = "outputs/retrieval_results/query_upload.jpg"
            ensure_dirs("outputs/retrieval_results")
            with open(tmp, "wb") as f:
                f.write(uploaded.read())
            query = tmp
        elif query_text.strip():
            query = query_text.strip()
        else:
            st.error("Please enter a query or upload an image.")
            st.stop()

        with st.spinner("Searching ..."):
            results = search(query, image_paths, embeddings,
                             G, labels, cfg, top_k=top_k)

        st.markdown(f"### Results for: `{query}`")
        if not results:
            st.warning("No results found.")
        else:
            cols = st.columns(len(results))
            for col, r in zip(cols, results):
                with col:
                    if os.path.exists(r["path"]):
                        col.image(r["path"], use_column_width=True)
                    col.markdown(
                        f"**#{r['rank']}** {r['file'][:18]}  \n"
                        f"sim=`{r['similarity']:.3f}` `[{r['category']}]`"
                    )


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluate":
    st.title("📊 Retrieval Evaluation")

    image_paths, embeddings, G, labels = _load_artifacts()
    if image_paths is None:
        _pipeline_warning()

    if st.button("▶ Run Evaluation", type="primary"):
        from src.retrieval.search import evaluate as run_eval

        with st.spinner("Evaluating ..."):
            metrics = run_eval(image_paths, embeddings, G, labels, cfg)

        st.markdown("### Summary Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Precision@5",   f"{metrics['mean_precision_at_5']:.3f}")
        c2.metric("Mean Recall",         f"{metrics['mean_recall']:.3f}")
        c3.metric("Mean Avg Precision",  f"{metrics['mean_avg_precision']:.3f}")

        st.markdown("### Per-Category Results")
        rows = [
            {
                "Category":      r["category"],
                "Query":         r["query"][:45],
                "Precision@5":   round(r["precision_at_5"], 2),
                "Recall":        round(r["recall"], 2),
                "Avg Precision": round(r["average_precision"], 2),
                "Retrieved":     r["retrieved_relevant"],
                "Total":         r["total_relevant"],
            }
            for r in metrics["per_category"]
        ]
        st.dataframe(rows, use_container_width=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        cats    = [r["category"] for r in metrics["per_category"]]
        p5      = [r["precision_at_5"]    for r in metrics["per_category"]]
        rec     = [r["recall"]             for r in metrics["per_category"]]
        ap      = [r["average_precision"]  for r in metrics["per_category"]]
        x       = range(len(cats))
        ax.bar([i - 0.25 for i in x], p5,  width=0.25, label="P@5",    color="#3498db")
        ax.bar([i        for i in x], rec, width=0.25, label="Recall",  color="#2ecc71")
        ax.bar([i + 0.25 for i in x], ap,  width=0.25, label="Avg P",   color="#e74c3c")
        ax.set_xticks(list(x)); ax.set_xticklabels(cats, rotation=20)
        ax.set_ylim(0, 1.15); ax.legend(fontsize=9)
        ax.set_title("Retrieval Metrics by Category")
        ax.axhline(0.82, color="gray", linestyle="--", linewidth=1, label="target 0.82")
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Graph":
    st.title("🗺️ Similarity Graph")

    png  = cfg["graph"]["output_png"]
    html = "outputs/graphs/interactive_graph.html"

    if os.path.exists(png):
        st.image(png, use_column_width=True,
                 caption="KNN Similarity Graph — nodes coloured by detected community")
    else:
        st.warning("Graph image not found. Build the pipeline first.")

    if os.path.exists(html):
        st.markdown("---")
        st.subheader("Interactive Graph")
        with open(html) as f:
            content = f.read()
        st.components.v1.html(content, height=800, scrolling=True)
    elif os.path.exists(png):
        st.info("Install **pyvis** and rebuild to get an interactive HTML graph:\n"
                "```\npip install pyvis\npython main.py --demo --force\n```")
