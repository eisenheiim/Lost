#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Put the venv outside ~/Desktop so iCloud Drive does not stall torch/chromadb imports.
VENV_DIR="${HEALTHRAG_VENV:-$HOME/.career_tree_rag/venv}"
mkdir -p "$(dirname "$VENV_DIR")"

echo "Creating virtual environment at: $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "Installing packages..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo ""
echo "Done. Next steps:"
echo "  source $VENV_DIR/bin/activate"
echo "  ./index.sh"
echo "  ./ask.sh --interactive"
echo "  ./ui.sh"
echo ""
echo "Note: venv lives at $VENV_DIR (not on Desktop) to avoid iCloud freezes."
