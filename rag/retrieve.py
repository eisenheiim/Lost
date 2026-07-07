"""Retrieve relevant career tree context for a user query."""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def retrieve(
    query: str,
    top_k: int = TOP_K,
    layer: str | None = None,
    doc_type: str | None = None,
    keep_ratio: float = 0.85,
    min_score: float = 0.30,
) -> list[dict]:
    """Retrieve context, dropping weak matches relative to the best hit.

    We over-fetch, then keep only hits whose score is close to the top score
    (>= top_score * keep_ratio) and above an absolute floor (min_score). This
    removes low-relevance career-tree noise for questions that only need one
    or two strong matches, while keeping all of them when several are relevant.
    The single best hit is always kept.
    """
    collection = get_collection()
    where: dict | None = None
    if layer and doc_type:
        where = {"$and": [{"layer": layer}, {"doc_type": doc_type}]}
    elif layer:
        where = {"layer": layer}
    elif doc_type:
        where = {"doc_type": doc_type}

    fetch_k = max(top_k * 3, 12)
    results = collection.query(
        query_texts=[query],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - distance,
            }
        )

    if not hits:
        return hits

    top_score = hits[0]["score"]
    threshold = max(min_score, top_score * keep_ratio)
    filtered = [h for h in hits if h["score"] >= threshold]
    filtered = filtered[:top_k]
    return filtered or hits[:1]


def format_context(hits: list[dict]) -> str:
    sections = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        header = meta.get("full_path") or meta.get("layer") or meta.get("doc_type", "context")
        source = meta.get("source_url", "")
        sections.append(f"[{i}] {header}\n{hit['text']}\nSource: {source}")
    return "\n\n---\n\n".join(sections)


def main() -> None:
    import sys

    question = " ".join(sys.argv[1:]) or "I studied physics, what AI careers exist?"
    hits = retrieve(question, top_k=5)
    print(format_context(hits))


if __name__ == "__main__":
    main()
