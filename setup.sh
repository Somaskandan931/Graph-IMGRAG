#!/usr/bin/env bash
# setup.sh — Quick-start script for graph-imgrag
# Usage: bash setup.sh

set -e

echo ""
echo "============================================================"
echo "  graph-imgrag — Setup"
echo "============================================================"
echo ""

# 1. Create virtual environment
echo "[1/4] Creating virtual environment ..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "[2/4] Installing dependencies ..."
pip install --upgrade pip -q
pip install -r requirements.txt

# 3. Quick sanity check (no GPU needed)
echo "[3/4] Verifying core imports ..."
python3 -c "
import networkx, sklearn, numpy, PIL, yaml, tqdm
from sentence_transformers import SentenceTransformer
print('  Core imports OK')
"

echo "[4/4] Setup complete!"
echo ""
echo "  Next steps:"
echo "    source venv/bin/activate"
echo "    python main.py --coco --max_images 100   # small test run (~50 MB)"
echo "    python main.py --coco                    # full 500-image run (~1 GB)"
echo "    streamlit run app/streamlit_app.py       # launch web UI"
echo ""
