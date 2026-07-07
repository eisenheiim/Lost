"""Assemble all chunks into one file and build the Chroma vector index."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import (
    CAREER_TREE_FILE,
    CHROMA_DIR,
    CHUNKS_FILE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EXTRA_CONTENT_DIR,
)
from rag.regenerate import build_chunks, get_paths

MIN_SECTION_CHARS = 60


def load_tree_chunks() -> list[dict]:
    """Build career-tree chunks (overview + layers + paths) from career-tree.json."""
    if not CAREER_TREE_FILE.exists():
        raise SystemExit(
            f"{CAREER_TREE_FILE} not found. Run: uv run career-rag-regenerate"
        )
    tree = json.loads(CAREER_TREE_FILE.read_text(encoding="utf-8"))
    return build_chunks(get_paths(tree))


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split simple `key: value` YAML-style frontmatter from the body."""
    meta: dict = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            block = text[3:end].strip()
            body = text[end + 4 :].lstrip("\n")
            for line in block.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip()
    return meta, body


def _split_sections(body: str) -> list[tuple[str | None, str]]:
    """Split a markdown body into (heading, content) pairs by level-2 (##) headings."""
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if heading is not None or content:
            sections.append((heading, content))

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    flush()
    return sections


def chunk_markdown(path: Path) -> list[dict]:
    """Turn one .md file into chunks, splitting by ## headings.

    Files with no ## headings stay a single chunk. Very short sections are
    merged into the previous chunk so we don't create tiny, low-signal vectors.
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)

    title = meta.get("title")
    if not title:
        for line in body.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    title = title or path.stem
    source = meta.get("source_url", str(path))

    sections = _split_sections(body)

    texts: list[tuple[str | None, str]] = []
    for heading, content in sections:
        if not content and heading is None:
            continue
        if texts and len(content) < MIN_SECTION_CHARS:
            prev_heading, prev_text = texts[-1]
            merged = prev_text + ("\n\n" if prev_text else "")
            merged += (f"## {heading}\n" if heading else "") + content
            texts[-1] = (prev_heading, merged)
        else:
            texts.append((heading, content))

    if not texts:
        texts = [(None, body.strip())]

    chunks = []
    for i, (heading, content) in enumerate(texts):
        header = f"{title}"
        if heading:
            header += f" — {heading}"
        chunk_text = header + "\n\n" + content
        chunks.append(
            {
                "id": f"extra-{path.stem}-{i}",
                "text": chunk_text,
                "metadata": {
                    "doc_type": "extra",
                    "layer": "",
                    "role": "",
                    "category": "",
                    "full_path": f"{path.name}" + (f" > {heading}" if heading else ""),
                    "source_url": source,
                },
            }
        )
    return chunks


def load_extra_markdown() -> list[dict]:
    """Load and chunk user-added .md files from content/extra/."""
    EXTRA_CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []
    for path in sorted(EXTRA_CONTENT_DIR.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(chunk_markdown(path))
    return chunks


def build_index(reset: bool = True) -> chromadb.Collection:
    tree_chunks = load_tree_chunks()
    extra_chunks = load_extra_markdown()
    chunks = tree_chunks + extra_chunks

    # Write every indexed chunk to a single file so it mirrors the vector store.
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

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
    print(f"  - {extra_count} extra chunks")
    print(f"  - store: {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    build_index()
