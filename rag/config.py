from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONTENT_DIR = ROOT / "content"
EXTRA_CONTENT_DIR = CONTENT_DIR / "extra"
CHROMA_DIR = ROOT / "chroma_db"

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CV_CACHE_DIR = DATA_DIR / "cv_cache"
CAREER_TREE_FILE = DATA_DIR / "career-tree.json"
CAREER_TREE_MD = CONTENT_DIR / "career-tree.md"

SOURCE_URL = "https://www.ibz04.pro/blog/career-tree"
BOARD_JS_URL = "https://www.ibz04.pro/_astro/CareerTreeBoard.D_9RdM3E.js"
COLLECTION_NAME = "career_tree"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
