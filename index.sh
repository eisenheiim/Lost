#!/usr/bin/env bash
set -e
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

if [[ -n "${1:-}" ]]; then
  if [[ ! -d "$1" ]]; then
    echo "Not a directory: $1" >&2
    exit 1
  fi
  export HEALTHRAG_EXTRA_DIR="$(cd "$1" && pwd)"
  echo "Indexing extra markdown from: $HEALTHRAG_EXTRA_DIR"
fi

echo "Running python -m rag.ingest …"
echo "Vector store: ${HEALTHRAG_CHROMA_DIR:-$HOME/.career_tree_rag/chroma_db}"
echo ""

python -m rag.ingest
