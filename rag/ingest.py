"""Assemble all chunks into one file and build the Chroma vector index."""

from __future__ import annotations

import json
from pathlib import Path

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
LOW_SIGNAL_HEADINGS = {
    "source",
    "sources",
    "references",
    "reference",
    "links",
    "link",
    "kaynak",
    "kaynaklar",
}
MERGE_PREFERRED_HEADINGS = {
    "note",
    "notes",
    "conclusion",
    "summary",
    "key takeaway",
    "key takeaways",
    "practical guidance",
}
MERGE_HEADING_PREFIXES = (
    "notes for",
    "note for",
    "about this source",
    "about source",
)
CONTEXT_DEPENDENT_HEADING_PARTS = (
    "takeaway",
    "takeaways",
    "summary",
    "conclusion",
    "final thoughts",
    "what this means",
    "why this matters",
    "practical guidance",
    "next steps",
    "notes for",
)


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


def _is_low_signal_section(heading: str | None, content: str) -> bool:
    """Return True for sections that should not become standalone chunks."""
    heading_norm = (heading or "").strip().lower().rstrip(":")
    content_norm = content.strip().lower()

    if heading_norm in LOW_SIGNAL_HEADINGS:
        return True

    # Avoid creating chunks that are only attribution links.
    if content_norm.startswith("source:") and len(content_norm) < 280:
        return True
    if content_norm.startswith("kaynak:") and len(content_norm) < 280:
        return True

    return False


def _should_merge_into_previous(heading: str | None, content: str) -> bool:
    heading_norm = (heading or "").strip().lower().rstrip(":")
    if any(heading_norm.startswith(prefix) for prefix in MERGE_HEADING_PREFIXES):
        return True
    if any(part in heading_norm for part in CONTEXT_DEPENDENT_HEADING_PARTS):
        return True
    if heading_norm in LOW_SIGNAL_HEADINGS or heading_norm in MERGE_PREFERRED_HEADINGS:
        return True
    if len(content.strip()) < MIN_SECTION_CHARS:
        return True
    return False


def _first_paragraph(text: str, max_chars: int = 280) -> str:
    """Return the first non-empty prose paragraph, clipped to max_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        # Skip markdown headings / bare titles; keep the first prose block.
        lines = [
            line.strip()
            for line in paragraph.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            continue
        summary = " ".join(" ".join(lines).split())
        if len(summary) <= max_chars:
            return summary
        clipped = summary[: max_chars - 1].rsplit(" ", 1)[0]
        return clipped.rstrip(".,;:") + "…"
    return ""


def _build_document_context(
    title: str,
    topic: str | None,
    sections: list[tuple[str | None, str]],
) -> str:
    """Build a short document-level summary for every chunk in this file."""
    intro = ""
    for heading, content in sections:
        if heading is None and content.strip():
            intro = _first_paragraph(content)
            break
    if not intro:
        for heading, content in sections:
            if content.strip() and not _is_low_signal_section(heading, content):
                intro = _first_paragraph(content)
                break

    parts = [f'This document is about "{title}".']
    if topic:
        parts.append(f"Topic: {topic}.")
    if intro:
        parts.append(intro)
    return " ".join(parts)


def chunk_markdown(path: Path) -> list[dict]:
    """Turn one .md file into chunks, splitting by ## headings.

    Files with no ## headings stay a single chunk. Context-dependent and short
    sections are merged into the previous chunk. Every chunk also gets a short
    document-level summary so section-only matches still carry enough context.
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
    topic = meta.get("topic")

    sections = _split_sections(body)
    document_context = _build_document_context(title, topic, sections)

    texts: list[tuple[str | None, str]] = []
    for heading, content in sections:
        if not content and heading is None:
            continue
        if texts and _should_merge_into_previous(heading, content):
            prev_heading, prev_text = texts[-1]
            merged = prev_text + ("\n\n" if prev_text else "")
            if not _is_low_signal_section(heading, content):
                merged += (f"## {heading}\n" if heading else "") + content
            elif content.strip():
                # Keep attribution text with the parent chunk instead of standalone chunks.
                merged += "\n" + content.strip()
            texts[-1] = (prev_heading, merged)
        else:
            if _is_low_signal_section(heading, content):
                # If this low-signal block is the first section, skip it.
                continue
            texts.append((heading, content))

    if not texts:
        texts = [(None, body.strip())]

    chunks = []
    for i, (heading, content) in enumerate(texts):
        header = f"{title}"
        if heading:
            header += f" — {heading}"
        # Skip redundant context when the chunk is already the intro (heading is None).
        if heading is None:
            chunk_text = f"{header}\n\n{content}"
        else:
            chunk_text = f"{header}\n\nDocument context: {document_context}\n\n{content}"
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
                    "document_title": title,
                    "document_context": document_context,
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


def build_index(reset: bool = True):
    print("Building chunk list…", flush=True)
    tree_chunks = load_tree_chunks()
    extra_chunks = load_extra_markdown()
    chunks = tree_chunks + extra_chunks
    print(f"  prepared {len(chunks)} chunks", flush=True)

    # Write every indexed chunk to a single file so it mirrors the vector store.
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Loading chromadb (can take 1–2 min on slow/synced disks)…", flush=True)
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    print(f"Opening vector store at {CHROMA_DIR}", flush=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            pass

    print(f"Loading embedding model '{EMBEDDING_MODEL}'…", flush=True)
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
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i, start in enumerate(range(0, len(chunks), batch_size), 1):
        end = start + batch_size
        print(f"  embedding batch {i}/{total_batches}…", flush=True)
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
    print("Starting career-tree index…", flush=True)
    build_index()
