from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONTENT_DIR = ROOT / "content"
_extra_override = os.environ.get("HEALTHRAG_EXTRA_DIR", "").strip()
EXTRA_CONTENT_DIR = (
    Path(_extra_override).expanduser().resolve()
    if _extra_override
    else (CONTENT_DIR / "extra")
)

# Keep the vector store off iCloud Desktop (~/Desktop) — file hydration there
# often hangs chromadb / torch imports for minutes with 0% CPU.
_chroma_override = os.environ.get("HEALTHRAG_CHROMA_DIR", "").strip()
if _chroma_override:
    CHROMA_DIR = Path(_chroma_override).expanduser().resolve()
else:
    CHROMA_DIR = Path.home() / ".career_tree_rag" / "chroma_db"

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CV_CACHE_DIR = DATA_DIR / "cv_cache"
CAREER_TREE_FILE = DATA_DIR / "career-tree.json"
CAREER_TREE_MD = CONTENT_DIR / "career-tree.md"

SOURCE_URL = "https://www.ibz04.pro/blog/career-tree"
BOARD_JS_URL = "https://www.ibz04.pro/_astro/CareerTreeBoard.D_9RdM3E.js"
COLLECTION_NAME = "career_tree"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
