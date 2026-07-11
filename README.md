## HealthRAG (Free Local Setup)

This project uses:
- ChromaDB for local vector index + retrieval
- Sentence Transformers for local embeddings
- Foundry Local for local LLM inference

### 1) Install dependencies

```bash
uv sync
# ensure the local package is installed so console scripts work
uv pip install -e .
```

### 2) Build / rebuild local index

Either using console scripts (if they work in your env):

```bash
uv run career-rag-regenerate
uv run career-rag-index
```

Or, reliably via module execution (bypasses console-script import issues):

```bash
uv run python -m rag.regenerate
uv run python -m rag.ingest
```

### 3) Ask questions

Console script (if available):

```bash
uv run career-rag-ask "I studied physics, what AI careers exist?"
```

Module form (recommended if you see ModuleNotFoundError):

```bash
uv run python -m rag.ask "I studied physics, what AI careers exist?"
```

Interactive mode:

```bash
uv run career-rag-ask --interactive
# or
uv run python -m rag.ask --interactive
```

Include a CV to tailor retrieval and answers (local extraction):

```bash
uv run python -m rag.ask --cv /path/to/your_cv.pdf "What roles fit me?"
uv run python -m rag.ask --cv /path/to/your_cv.pdf --interactive
```

Force fresh CV extraction (ignore cache):

```bash
uv run python -m rag.ask --cv /path/to/your_cv.pdf --no-cache "What roles fit me?"
```

Use a stronger model for CV extraction only:

```bash
uv run python -m rag.ask --cv /path/to/your_cv.pdf --extract-model qwen2.5-1.5b "What roles fit me?"
```

#### CV flow (what happens when `--cv` is used)

1. **Read CV** — `.pdf`, `.docx`, `.txt`, or `.md` is converted to text.
2. **Extract profile** — Foundry Local turns the CV into structured JSON (`skills`, `roles`, `education`, `projects`, etc.).
   - Results are cached under `data/cv_cache/` (skipped on the next run unless `--no-cache`).
3. **Build two profile views:**
   - `cv_query` — retrieval-focused search string (skills, roles, education, highlights, certifications). Used only for vector search.
   - `cv_summary` — short human-readable profile. Sent to the LLM so answers are personalized.
4. **Retrieve context** — ChromaDB query becomes:
   - `your question ; user_profile: <cv_query>`
5. **Generate answer** — LLM receives:
   - retrieved chunks (without source URLs)
   - your original question + `cv_summary`
6. **Show sources separately** — used sources are printed in the terminal, not sent to the LLM.

Retrieval and generation intentionally use different inputs: `cv_query` helps find relevant chunks; `cv_summary` helps the model write a clear, tailored answer.

### 4) Recommend roles from a CV

Extract CV info and get tailored role recommendations (local, no external services):

```bash
uv run career-rag-recommend --cv /path/to/your_cv.pdf
uv run career-rag-recommend --cv /path/to/your_cv.pdf --no-cache
```

Only extract JSON (no recommendations):

```bash
uv run career-rag-recommend --cv /path/to/your_cv.pdf --extract-only
```

Cached extraction files live in `data/cv_cache/`.
### Troubleshooting

- ModuleNotFoundError: No module named rag
	- Run: `uv pip install -e .` in the project root, then retry your command.
	- If you previously created a virtualenv, ensure you run commands from the project root so uv picks up .venv.
    
- PDF/DOCX reading issues
	- Ensure dependencies are installed (done by `uv sync`): pypdf, python-docx.
	- For scanned PDFs (images), you’ll need OCR (e.g., Tesseract). This project focuses on text-based PDFs.

