"""Build the Chroma vector index from career tree chunks."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import (
    CAREER_TREE_MD,
    CHROMA_DIR,
    CHUNKS_FILE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EXTRA_CONTENT_DIR,
    SOURCE_URL,
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_extra_markdown() -> list[dict]:
    """Load user-added .md files from content/extra/."""
    EXTRA_CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []

    for path in sorted(EXTRA_CONTENT_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.append(
            {
                "id": f"extra-{path.stem}",
                "text": text,
                "metadata": {
                    "doc_type": "extra",
                    "layer": "",
                    "role": "",
                    "category": "",
                    "full_path": path.name,
                    "source_url": str(path),
                },
            }
        )

    return chunks


def build_index(reset: bool = True) -> chromadb.Collection:
    chunks = load_jsonl(CHUNKS_FILE)

    if CAREER_TREE_MD.exists():
        chunks.append(
            {
                "id": "career-tree-md",
                "text": CAREER_TREE_MD.read_text(encoding="utf-8"),
                "metadata": {
                    "doc_type": "overview",
                    "layer": "overview",
                    "role": "",
                    "category": "",
                    "full_path": "KNOWLEDGE",
                    "source_url": SOURCE_URL,
                },
            }
        )

    extra_chunks = load_extra_markdown()
    chunks.extend(extra_chunks)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            pass

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    path_count = sum(1 for c in chunks if c["metadata"].get("doc_type") == "career_path")
    layer_count = sum(1 for c in chunks if c["metadata"].get("doc_type") == "layer_overview")
    extra_count = len(extra_chunks)
    print(f"Indexed {len(chunks)} chunks into '{COLLECTION_NAME}'")
    print(f"  - {path_count} career paths")
    print(f"  - {layer_count} layer overviews")
    print(f"  - {extra_count} extra files")
    print(f"  - store: {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    build_index()
