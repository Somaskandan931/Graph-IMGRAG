# graph-imgrag — Graph-Based Image Retrieval (COCO Dataset Edition)

OCR → Semantic Embeddings → KNN Graph → Query Retrieval, powered by the real **MS-COCO** dataset.

---

## What changed from the demo version?

| Feature | Demo (old) | COCO (new) |
|---|---|---|
| Dataset | 170 synthetic text images | Real MS-COCO photos (val2017 / train2017) |
| Categories | street_signs, products, receipts… | COCO supercategories (person, animal, food…) |
| Download | No download needed | Auto-downloads annotations + images |
| Scale | 170 images | Up to 5 000 (val2017) or 118 000 (train2017) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download COCO val2017 subset (500 images by default) + run pipeline
python main.py --coco

# 3. Launch web UI
streamlit run app/streamlit_app.py
```

---

## CLI Options

```
python main.py --coco               # Download COCO + run pipeline
python main.py --coco --eval        # Also compute evaluation metrics
python main.py --coco --max_images 200   # Use only 200 images
python main.py --coco --split train2017  # Use training split (~18 GB)
python main.py --query "dog park"   # Query without re-downloading
python main.py --force              # Clear cache and re-run everything
```

---

## Configuration (`configs/config.yaml`)

Key settings:

```yaml
dataset:
  coco_split: "val2017"     # val2017 (~1 GB) | train2017 (~18 GB)
  max_images: 500           # cap on number of images to process
  categories:               # COCO supercategories to include
    - person
    - vehicle
    - animal
    - food
    ...
```

---

## Pipeline Steps

```
COCO Download (annotations + images)
         ↓
  OCR (EasyOCR) — extracts text visible in photos
         ↓
  Sentence Embeddings (all-MiniLM-L6-v2)
         ↓
  KNN Similarity Graph (k=5, cosine similarity)
         ↓
  Query Retrieval + Graph Expansion
         ↓
  Evaluation (Precision@5, Recall, MAP)
```

---

## COCO Dataset Notes

- **Annotations** download URL: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` (~241 MB, cached in `dataset/annotations/`)
- **Images** are fetched individually from `http://images.cocodataset.org/val2017/<id>.jpg`
- Images are organised into `dataset/processed/images/<supercategory>/` folders
- A metadata file `dataset/annotations/coco_meta.json` records per-image COCO info

---

## Outputs

| Path | Contents |
|---|---|
| `outputs/embeddings/image_embeddings.pkl` | Embeddings + image paths |
| `outputs/embeddings/ocr_results.json` | Per-image OCR text |
| `outputs/graphs/similarity_graph.png` | Static graph visualisation |
| `outputs/graphs/interactive_graph.html` | Interactive pyvis graph |
| `outputs/retrieval_results/` | Query result JSONs + grid images |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pycocotools` / built-in JSON | COCO annotation parsing |
| `EasyOCR` | Text extraction from images |
| `all-MiniLM-L6-v2` | Semantic text embeddings |
| `NetworkX` | KNN graph construction |
| `scikit-learn` | Cosine similarity |
| `Streamlit` | Web UI |
