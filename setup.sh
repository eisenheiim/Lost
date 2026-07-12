#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Installing packages..."
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo ""
echo "Done. Next steps:"
echo "  source .venv/bin/activate"
echo "  ./index.sh"
echo "  ./ask.sh --interactive"
