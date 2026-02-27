# Graph-Based Image Retrieval System

A text-aware image retrieval pipeline that extracts embedded text from images using OCR, encodes it as semantic embeddings, and organises images into a similarity graph for efficient query-based retrieval.

---

## Overview

Traditional image retrieval systems depend on manual annotations or metadata. Many real-world images contain embedded text such as street signs, product labels, receipts, posters, and identity documents. This system exploits that textual content to build a fully automated retrieval pipeline without requiring any manual labelling.

The core idea is to treat each image as a text document, embed that text into a vector space, and connect images by semantic similarity through a k-nearest-neighbour graph. At query time, the graph structure enables both direct similarity matching and neighbourhood expansion to surface contextually related results.

---

## Pipeline

```
Image Dataset
      |
      v
OCR Text Extraction  (EasyOCR)
      |
      v
Sentence Embeddings  (all-MiniLM-L6-v2)
      |
      v
KNN Similarity Graph  (NetworkX, cosine similarity, k=5)
      |
      v
Query Engine  (text or image query)
      |
      v
Retrieved Images
```

---

## Dataset

The system uses the [MS-COCO 2017](https://cocodataset.org/#download) dataset. Images are downloaded automatically at runtime and organised by COCO supercategory. The default configuration uses the `val2017` split, capped at 500 images for quick experimentation.

**Supported COCO supercategories:**

| Supercategory | Examples |
|---|---|
| person | people, crowds, portraits |
| vehicle | car, bus, truck, bicycle |
| outdoor | street signs, traffic signals |
| animal | dog, cat, bird, horse |
| sports | ball, racket, skateboard |
| food | pizza, sandwich, fruit |
| furniture | chair, table, sofa |
| electronic | laptop, phone, television |
| kitchen | bowl, cup, fork, knife |
| indoor | book, clock, vase |

The original prototype was validated on a manually curated set of 170 text-rich images across five categories (street signs, product labels, advertisements, receipts, documents), achieving a mean average precision of 0.82 at k=5.

---

## Requirements

- Python 3.9 or higher
- Internet connection for initial COCO download (~241 MB annotations + images)

Install all dependencies:

```bash
pip install -r requirements.txt
```

Core libraries:

| Library | Purpose |
|---|---|
| easyocr | OCR text extraction from images |
| sentence-transformers | Semantic text embeddings |
| networkx | Graph construction and traversal |
| scikit-learn | Cosine similarity computation |
| numpy | Numerical operations |
| Pillow | Image loading and processing |
| matplotlib | Graph and result visualisation |
| streamlit | Web interface |
| pyvis | Interactive graph (optional) |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/graph-imgrag.git
cd graph-imgrag

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
# Download COCO val2017 (500 images) and run all steps
python main.py --coco

# Include evaluation metrics at the end
python main.py --coco --eval

# Limit to fewer images for faster testing
python main.py --coco --max_images 100

# Use the full training split (approximately 18 GB)
python main.py --coco --split train2017 --max_images 5000
```

### Query the built index

```bash
# Text query
python main.py --query "dog park outdoor"
python main.py --query "car road traffic"
python main.py --query "food pizza plate"

# Image query (OCR is applied to the image first)
python main.py --query path/to/image.jpg

# Control number of returned results
python main.py --query "sports ball" --top_k 10
```

### Run individual pipeline steps

```bash
python src/utils/coco_loader.py          # Download and organise COCO images
python src/ocr/extract_text.py           # Run OCR on all images
python src/embeddings/generate_embeddings.py   # Generate embeddings
python src/graph/build_graph.py          # Build the KNN graph
python src/retrieval/search.py --query "dog" --eval  # Search and evaluate
```

### Launch the web interface

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in a browser. The interface provides pipeline controls, a search page, evaluation metrics, and a graph visualisation.

### Force a full re-run

```bash
python main.py --coco --force
```

---

## Configuration

All settings are in `configs/config.yaml`.

```yaml
dataset:
  coco_split: "val2017"      # val2017 (~1 GB) | train2017 (~18 GB)
  max_images: 500            # maximum images to download and process
  categories:                # COCO supercategories to include
    - person
    - vehicle
    - animal
    - food
    - sports
    # add or remove as needed

ocr:
  languages: ["en"]
  gpu: false                 # set true if a CUDA GPU is available

embeddings:
  model: "all-MiniLM-L6-v2"
  batch_size: 32
  normalize: true

graph:
  knn_k: 5                   # number of nearest neighbours per node

retrieval:
  top_k: 5
  expand_graph: true         # traverse neighbours during search
  expansion_depth: 2
```

---

## Repository Structure

```
graph-imgrag/
|
|-- main.py                          # Pipeline entry point
|-- requirements.txt
|-- configs/
|   `-- config.yaml                  # All configuration settings
|
|-- src/
|   |-- ocr/
|   |   `-- extract_text.py          # EasyOCR wrapper
|   |-- embeddings/
|   |   `-- generate_embeddings.py   # Sentence embedding generation
|   |-- graph/
|   |   `-- build_graph.py           # KNN graph construction and visualisation
|   |-- retrieval/
|   |   `-- search.py                # Query engine and evaluation
|   `-- utils/
|       |-- helpers.py               # Shared utilities
|       `-- coco_loader.py           # COCO download and organisation
|
|-- app/
|   `-- streamlit_app.py             # Web interface
|
|-- dataset/
|   |-- raw/                         # Raw downloaded files
|   |-- processed/
|   |   |-- images/                  # Images organised by supercategory
|   |   `-- ocr_text/                # Per-image OCR text files
|   `-- annotations/                 # COCO annotation JSON files
|
|-- outputs/
|   |-- embeddings/                  # Cached embeddings and OCR results
|   |-- graphs/                      # Graph files (pickle, GEXF, PNG, HTML)
|   `-- retrieval_results/           # Query result JSONs and grid images
|
|-- tests/
|   `-- test_pipeline.py
|
`-- notebooks/
    `-- experiment.ipynb
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/embeddings/image_embeddings.pkl` | Serialised embeddings and image paths |
| `outputs/embeddings/ocr_results.json` | Extracted text per image |
| `outputs/graphs/similarity_graph.png` | Static graph visualisation |
| `outputs/graphs/similarity_graph.gexf` | Graph file for Gephi or similar tools |
| `outputs/graphs/interactive_graph.html` | Interactive graph (requires pyvis) |
| `outputs/retrieval_results/` | Per-query result JSONs and image grids |
| `dataset/annotations/coco_meta.json` | Per-image COCO metadata |

---

## Evaluation

Retrieval performance is measured using Precision@5, Recall, and Mean Average Precision (MAP) across COCO supercategories.

Run evaluation after building the pipeline:

```bash
python main.py --eval
```

Results are printed to the terminal and saved to `outputs/retrieval_results/evaluation_metrics.json`.

Baseline results on the original 170-image prototype dataset:

| Query Category | Precision@5 | Recall | Avg Precision |
|---|---|---|---|
| Documents | 0.80 | 0.40 | 0.83 |
| Receipts | 0.80 | 0.33 | 0.81 |
| Street Signs | 1.00 | 0.25 | 1.00 |
| Advertisements | 0.80 | 0.27 | 0.83 |
| Products | 0.80 | 0.27 | 0.65 |
| **Mean** | **0.84** | **0.30** | **0.82** |

---

## Known Limitations

- OCR accuracy degrades on blurry images, stylised fonts, and dense overlapping text. Images with no readable text fall back to the filename stem as the text representation.
- Graph connectivity can be sparse when OCR text is very short or noisy, reducing the effectiveness of neighbourhood expansion.
- The pipeline is CPU-only by default. Setting `gpu: true` in the config and having a CUDA-enabled environment will significantly reduce OCR and embedding time on large datasets.

---

## Possible Extensions

- Hybrid visual-textual embeddings using CLIP to handle images with no readable text.
- Graph neural networks for embedding refinement across neighbourhoods.
- Approximate nearest neighbour indexing (e.g. FAISS) for scaling to millions of images.
- Support for additional OCR languages by extending the `languages` list in the config.

---

## License

This project is released for academic and research use. The MS-COCO dataset is subject to its own [terms of use](https://cocodataset.org/#termsofuse).