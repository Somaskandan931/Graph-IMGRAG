# Graph-Based Image Retrieval System

A text-aware image retrieval pipeline that extracts embedded text from images using OCR, encodes it as semantic embeddings, organises images into a similarity graph for efficient query-based retrieval, and generates new AI images inspired by search results.

---

## Introduction

With the rapid growth of digital media, searching large image collections has become increasingly challenging. Traditional retrieval approaches rely on metadata or manual annotations — a process that is slow, expensive, and difficult to scale.

Many real-world images contain **embedded text** that is rich with semantic meaning: passports, receipts, street signs, product labels, and posters all carry readable information that a retrieval system can exploit. By extracting and understanding this text automatically, we can build a retrieval system that requires no manual labelling at all.

This project introduces a **Graph-Based Image Retrieval Architecture**. Each image is treated as a text document — its embedded text is extracted via OCR, converted into a semantic vector embedding, and connected to visually and semantically similar images through a k-nearest-neighbour similarity graph. At query time, both direct similarity matching and graph neighbourhood traversal are used to surface the most relevant results.

On top of retrieval, the system includes two generative AI layers: a **Claude-powered** module for captions, Q&A, and dataset insights, and a **Pollinations AI** module (free, no key required) for generating brand-new images inspired by the search results.

---

## Objectives

1. Extract text from images automatically using OCR.
2. Represent extracted text as dense semantic vector embeddings.
3. Construct a k-nearest-neighbour similarity graph across all images.
4. Implement a query engine that accepts text or image queries.
5. Persist all pipeline state in a SQLite database for instant re-querying.
6. Evaluate retrieval performance using standard information retrieval metrics.
7. Generate AI captions, summaries, and new images from retrieval results.

---

## How It Works

The system runs as a sequential pipeline across six stages:

```
Image Dataset
      │
      ▼
OCR Text Extraction   →  EasyOCR reads embedded text from each image
      │
      ▼
Text Embeddings       →  all-MiniLM-L6-v2 encodes text as 384-dim vectors
      │
      ▼
Similarity Graph      →  Images are connected by cosine similarity (k=5 neighbours)
      │
      ▼
SQLite Database       →  All state persisted for instant re-querying
      │
      ▼
Query Engine          →  Text or image query is embedded and matched against the graph
      │
      ▼
Retrieved Images  →  (optional) AI captions / summaries via Claude
                  →  (optional) New AI image generation via Pollinations AI
```

### Stage 1 — OCR Text Extraction

Text is extracted from each image using **EasyOCR**, a deep learning based optical character recognition engine. The process loads each image, detects text regions, recognises the characters, and concatenates the words into a single text string per image. Each result is then enriched with the COCO supercategory label (repeated for embedding weight), so images with no readable text still receive a meaningful semantic representation rather than falling back to only the filename stem.

```
Input:  stop_sign.jpg      →  Output: "STOP outdoor outdoor"
Input:  receipt_42.jpg     →  Output: "Total $14.99 Thank you indoor indoor"
Input:  dog_park.jpg       →  Output: "animal animal"   (no OCR text found)
```

### Stage 2 — Text Embeddings

The extracted text string for each image is passed through a pre-trained transformer model — **`all-MiniLM-L6-v2`** from the Sentence Transformers library — to produce a compact 384-dimensional embedding vector. Embeddings are L2-normalised so that cosine similarity reduces to a dot product. Semantically related texts (e.g. "passport", "visa document", "travel id") cluster together in this vector space, enabling meaningful similarity comparisons even when the exact wording differs.

### Stage 3 — Similarity Graph Construction

A graph is constructed where every **node** is an image and every **edge** connects two images whose embeddings are among each other's k nearest neighbours (default k = 5), weighted by their cosine similarity score. The graph is built with **NetworkX** and exported in multiple formats for further analysis or visualisation in tools like Gephi.

```
receipt_1 ── receipt_2
    │
invoice_1
```

This graph structure enables both direct nearest-neighbour retrieval and multi-hop neighbourhood expansion to surface contextually related images that may not be the closest match by embedding distance alone.

### Stage 4 — SQLite Persistence

All pipeline state — image paths, OCR text, embeddings (stored as binary blobs), category labels, graph edges, and run metadata — is persisted in a **SQLite database** (`outputs/graph_imgrag.db`). After the index is built once, every subsequent search loads from the database instantly, with no re-processing. The database supports fast category filtering, full-text OCR search, and graph reconstruction.

### Stage 5 — Query Processing

A user submits either a free-text query or an image file. The query is converted to an embedding using the same model, ranked against all graph nodes by cosine similarity, optionally expanded by traversing the graph up to a configurable depth, and the top-k results are returned with similarity scores and their extracted OCR text.

```
Query:   "passport"
Results: passport_1.jpg     (0.94)
         passport_2.jpg     (0.91)
         travel_doc_3.jpg   (0.87)
```

### Stage 6 — Generative AI Layer

Two optional generative modules extend the retrieval results:

**Claude (Anthropic)** — `src/generative/generate_ai.py` — requires `ANTHROPIC_API_KEY`:
- Generate rich natural-language captions for individual images (descriptive, short, poetic, or technical style)
- Answer free-form questions about a specific image
- Summarise a set of retrieved results in natural language
- Suggest 5 related search queries based on the current results
- Generate high-level insights about the indexed dataset

**Pollinations AI** — `src/generative/image_gen.py` — free, no API key required:
- Generate a brand-new image inspired by the search query and retrieved results
- The prompt is automatically enriched with category context and OCR text from the top results
- Supports multiple style presets (Photorealistic, Cinematic, Digital Art, Oil Painting, Watercolor, Minimalist) and aspect ratios
- Available directly in the Streamlit web interface after every successful search

---

## Dataset

### Option A — Automatic COCO Download (default)

The system uses the [MS-COCO 2017](https://cocodataset.org/#download) dataset. Images are downloaded automatically at runtime and organised by COCO supercategory. The default configuration uses the `val2017` split, capped at 500 images for quick experimentation.

**Supported COCO supercategories:**

| Supercategory | Examples |
|---|---|
| person | people, crowds, portraits |
| vehicle | car, bus, truck, bicycle |
| outdoor | street signs, traffic signals |
| animal | dog, cat, bird, horse |
| accessory | bag, umbrella, hat |
| sports | ball, racket, skateboard |
| food | pizza, sandwich, fruit |
| furniture | chair, table, sofa |
| electronic | laptop, phone, television |
| kitchen | bowl, cup, fork, knife |
| appliance | oven, microwave, sink |
| indoor | book, clock, vase |

### Option B — Kaggle Mirror (recommended for offline or faster setup)

If you prefer not to download directly from the COCO servers, the full COCO 2017 dataset is also available as a Kaggle dataset:

**[COCO Dataset 2017 — sabahesaraki/2017-2017](https://www.kaggle.com/datasets/sabahesaraki/2017-2017)**

This mirror packages the standard COCO 2017 train and validation splits along with the official annotation files in a single download.

| Split | Images | Size (approx.) |
|---|---|---|
| `val2017` | ~5,000 images | ~1 GB |
| `train2017` | ~118,000 images | ~18 GB |
| Annotations | Captions, instances, keypoints | ~241 MB |

```bash
# Install the Kaggle CLI
pip install kaggle

# Download the dataset (requires ~/.kaggle/kaggle.json API token)
kaggle datasets download -d sabahesaraki/2017-2017

# Unzip into the expected directory layout
unzip 2017-2017.zip -d dataset/
```

After unzipping, update `local_images_dir` in `configs/config.yaml` to point to your `val2017` folder, then run `python main.py --build` to index without downloading again.

### Option C — Local Images Already on Disk

If you already have COCO images downloaded (e.g. from a previous project), set the path in `configs/config.yaml`:

```yaml
dataset:
  local_images_dir: "D:/datasets/coco/val2017"   # absolute or relative path
```

The pipeline will copy images into the processed folder structure and skip any downloading.

### Original Prototype Dataset

The original prototype was validated on a manually curated set of **170 text-rich images** across five categories:

| Category | Images |
|---|---|
| Street signs | 40 |
| Product labels | 30 |
| Advertisements | 30 |
| Receipts | 35 |
| Documents | 35 |
| **Total** | **170** |

---

## Implementation Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| EasyOCR | OCR text extraction from images |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Semantic text embeddings |
| scikit-learn | Cosine similarity computation |
| NetworkX | Graph construction and traversal |
| NumPy | Numerical operations |
| Pillow | Image loading and processing |
| Matplotlib | Graph and result visualisation |
| SQLite (stdlib) | Persistent storage for all pipeline state |
| Streamlit | Web interface |
| Anthropic Claude API | AI captions, Q&A, summaries, query suggestions |
| Pollinations AI | Free AI image generation (no key required) |
| pyvis | Interactive graph visualisation (optional) |

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

Or use the provided setup script which does all of the above and runs a quick sanity check:

```bash
bash setup.sh
```

### Anthropic API Key (optional — for AI captions and Q&A)

The Claude-powered generative features require an Anthropic API key. The image generation feature (Pollinations AI) works without any key.

```bash
# Linux / macOS
export ANTHROPIC_API_KEY=sk-ant-...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Or add to your shell profile / .env file
```

---

## Usage

### Build the search index (run once)

```bash
# Index your local COCO images — reads from local_images_dir in config
python main.py --build

# Include evaluation metrics after building
python main.py --build --eval

# Limit to fewer images for faster testing
python main.py --build --max_images 100

# Wipe all cached outputs and rebuild from scratch
python main.py --build --force

# Use the training split instead of val
python main.py --build --split train2017 --max_images 5000
```

> **Note:** `--coco` is a legacy alias for `--build` and works identically.

### Query the built index (instant — no rebuild needed)

```bash
# Text queries
python main.py --query "dog park outdoor"
python main.py --query "car road traffic"
python main.py --query "food pizza plate"

# Image query (OCR is applied to the uploaded image first)
python main.py --query path/to/image.jpg

# Control number of returned results
python main.py --query "sports ball" --top_k 10
```

### Inspect the database

```bash
# Print image counts, category breakdown, and edge count
python main.py --stats
```

### Run individual pipeline steps

```bash
python src/utils/coco_loader.py                      # Download and organise COCO images
python src/ocr/extract_text.py                       # Run OCR on all images
python src/embeddings/generate_embeddings.py         # Generate embeddings
python src/graph/build_graph.py                      # Build the KNN graph
python src/retrieval/search.py --query "dog" --eval  # Search and evaluate
```

### Test the generative AI modules

```bash
# Test Pollinations AI connectivity and generate a sample image
python src/generative/image_gen.py

# Test Claude API key validity (requires ANTHROPIC_API_KEY)
python -c "from src.generative.generate_ai import check_api_key; ok, msg = check_api_key(); print(msg)"
```

### Launch the web interface

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in a browser. The interface includes:

- **Search panel** — text query input with example chips, or upload a query image for OCR-based search
- **Category filter** — narrow results to a specific COCO supercategory via the sidebar
- **Result cards** — retrieved images with similarity scores, rank badges, and detected OCR text
- **AI Image Generation** — appears after every successful search; generates a new image inspired by the query and results using Pollinations AI (free, no key needed), with style and aspect ratio controls
- **Similarity Graph viewer** — toggle in the sidebar to view the KNN graph PNG and an interactive HTML version (requires pyvis)
- **Evaluation Metrics panel** — toggle in the sidebar to view Precision@K, Recall, and MAP per category

### Run the test suite

```bash
python -m pytest tests/test_pipeline.py -v
```

### Force a full re-run

```bash
python main.py --build --force
```

---

## Configuration

All settings are in `configs/config.yaml`.

```yaml
dataset:
  local_images_dir: "dataset/val2017"  # path to your downloaded COCO images
  coco_split: "val2017"                # val2017 (~1 GB) | train2017 (~18 GB)
  max_images: 500
  categories:
    - person
    - vehicle
    - outdoor
    - animal
    - accessory
    - sports
    - kitchen
    - food
    - furniture
    - electronic
    - appliance
    - indoor

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
  expand_graph: true         # traverse graph neighbours during search
  expansion_depth: 2
```

---

## Evaluation

Retrieval performance is measured using **Precision@5**, **Recall**, and **Mean Average Precision (MAP)** across image categories.

```bash
python main.py --build --eval
# or, after building:
python src/retrieval/search.py --query "dog" --eval
```

Results are printed to the terminal and saved to `outputs/retrieval_results/evaluation_metrics.json`.

### Baseline results on the 170-image prototype dataset

| Query Category | Precision@5 | Recall | Avg Precision |
|---|---|---|---|
| Documents | 0.80 | 0.40 | 0.83 |
| Receipts | 0.80 | 0.33 | 0.81 |
| Street Signs | 1.00 | 0.25 | 1.00 |
| Advertisements | 0.80 | 0.27 | 0.83 |
| Products | 0.80 | 0.27 | 0.65 |
| **Mean** | **0.84** | **0.30** | **0.82** |

Sample query-level results:

| Query | Relevant Images | Retrieved |
|---|---|---|
| passport | 10 | 9 |
| receipt | 12 | 10 |
| sale | 8 | 7 |

> **Note on COCO val2017 scores:** COCO natural photos contain very little embedded text, so evaluation queries (which match on category labels rather than literal text) will show lower MAP than the prototype numbers above. This reflects the dataset characteristics — the retrieval, graph, and embedding components all function correctly as demonstrated by similarity search results.

---

## Repository Structure

```
graph-imgrag/
│
├── main.py                              # Pipeline entry point (--build / --query / --stats)
├── requirements.txt
├── setup.sh                             # One-command environment setup
├── configs/
│   └── config.yaml                      # All configuration settings
│
├── src/
│   ├── ocr/
│   │   └── extract_text.py              # EasyOCR wrapper + category enrichment
│   ├── embeddings/
│   │   └── generate_embeddings.py       # Sentence embedding generation
│   ├── graph/
│   │   └── build_graph.py               # KNN graph construction and visualisation
│   ├── retrieval/
│   │   └── search.py                    # Query engine and evaluation metrics
│   ├── generative/
│   │   ├── generate_ai.py               # Claude: captions, Q&A, summaries, insights
│   │   └── image_gen.py                 # Pollinations AI: free image generation
│   └── utils/
│       ├── helpers.py                   # Shared utilities (config, logging, I/O)
│       ├── database.py                  # SQLite persistence layer
│       └── coco_loader.py               # COCO annotation parsing and image organisation
│
├── app/
│   └── streamlit_app.py                 # Web interface
│
├── dataset/
│   ├── raw/                             # Raw downloaded files
│   ├── processed/
│   │   ├── images/                      # Images organised by supercategory
│   │   └── ocr_text/                    # Per-image OCR text files
│   └── annotations/                     # COCO annotation JSON files
│
├── outputs/
│   ├── graph_imgrag.db                  # SQLite database (all pipeline state)
│   ├── embeddings/                      # Cached embeddings and OCR results
│   ├── graphs/                          # Graph files (pickle, GEXF, PNG, HTML)
│   ├── generated/                       # AI-generated images (Pollinations AI)
│   └── retrieval_results/               # Per-query result JSONs and image grids
│
└── tests/
    └── test_pipeline.py                 # Unit tests for the full pipeline
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/graph_imgrag.db` | SQLite database — images, embeddings, graph edges, run metadata |
| `outputs/embeddings/image_embeddings.pkl` | Legacy pickle cache of embeddings and image paths |
| `outputs/embeddings/ocr_results.json` | Extracted OCR text per image |
| `outputs/graphs/similarity_graph.png` | Static KNN graph visualisation |
| `outputs/graphs/similarity_graph.gexf` | Graph file for Gephi or similar tools |
| `outputs/graphs/interactive_graph.html` | Interactive graph (requires pyvis) |
| `outputs/generated/` | AI-generated images from Pollinations AI |
| `outputs/retrieval_results/` | Per-query result JSONs and image grid PNGs |
| `outputs/retrieval_results/evaluation_metrics.json` | Precision@K, Recall, MAP per category |
| `dataset/annotations/coco_meta.json` | Per-image COCO metadata (category, dimensions, URL) |

---

## Known Limitations

**OCR accuracy** degrades on blurry images, stylised fonts, and dense overlapping text. The category enrichment step mitigates this for COCO images by appending the supercategory label to each embedding, but retrieval quality is still best on text-rich datasets like receipts, signs, and documents.

**Graph connectivity** can be sparse when OCR text is very short or noisy, which reduces the effectiveness of neighbourhood expansion during retrieval.

**Performance** — the pipeline is CPU-only by default. Setting `gpu: true` in the config with a CUDA-enabled environment will significantly reduce OCR and embedding time on large datasets.

**Pollinations AI** is a free, rate-limited external service. HTTP 530 errors indicate Cloudflare-level overload on Pollinations' servers and are transient — waiting 1–2 minutes and retrying usually resolves them. The generation client automatically retries across multiple models (`flux`, `turbo`, `flux-realism`) before giving up.

---

## Possible Extensions

- **Hybrid multimodal embeddings** using CLIP to handle images with no readable text, combining visual and textual signals.
- **Graph neural networks** for embedding refinement across neighbourhoods.
- **Approximate nearest neighbour indexing** (e.g. FAISS) for scaling to millions of images.
- **Multilingual OCR** by extending the `languages` list in the config.
- **Claude vision integration** — pass retrieved images directly to Claude for richer, vision-based captions rather than relying solely on OCR text.

---

## Applications

The system architecture is well suited to a range of real-world use cases, including document management systems, digital libraries, government record retrieval, invoice and receipt search, e-commerce catalogue search, and surveillance log analysis.

---

## License

This project is released for academic and research use. The MS-COCO dataset is subject to its own [terms of use](https://cocodataset.org/#termsofuse). The Kaggle mirror is additionally subject to [Kaggle's Terms of Service](https://www.kaggle.com/terms).