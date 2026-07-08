## HealthRAG (Free Local Setup)

This project uses:
- ChromaDB for local vector index + retrieval
- Sentence Transformers for local embeddings
- Foundry Local for local LLM inference

### 1) Install dependencies

```bash
uv sync
```

### 2) Build / rebuild local index

```bash
uv run career-rag-index
```

### 3) Ask questions

```bash
uv run career-rag-ask "I studied physics, what AI careers exist?"
```

Interactive mode:

```bash
uv run career-rag-ask --interactive
```
