# HealthRAG — Career Tree (local, free)

Local RAG app: ChromaDB + Sentence Transformers + Foundry Local.

## Quick start

```bash
cd /path/to/healthrag
chmod +x setup.sh index.sh ask.sh retrieve.sh regenerate.sh recommend.sh
./setup.sh
source .venv/bin/activate
./index.sh
./ask.sh --interactive
```

## What you need

- Python 3.11+
- Foundry Local — required for `./ask.sh` and `./recommend.sh` (not needed for `./retrieve.sh` or `./index.sh`)

## CLI reference

All `.sh` scripts activate `.venv` automatically. You can also run the same commands with `python -m rag.<module>` after `source .venv/bin/activate`.

### `setup.sh` — one-time install

Creates the virtual environment and installs dependencies.

```bash
./setup.sh
```

Does **not** build the search index. Run `./index.sh` next.

---

### `regenerate.sh` — refresh career tree data (optional)

Downloads the latest Career Tree from [ibz04.pro](https://www.ibz04.pro/blog/career-tree) and writes:

- `data/career-tree.json`
- `content/career-tree.md`

```bash
./regenerate.sh
```

Run this when the online tree has changed. You still need `./index.sh` afterward to update Chroma.

Equivalent: `python -m rag.regenerate`

---

### `index.sh` — build the vector index

Embeds all chunks (career tree + `content/extra/*.md`) into Chroma at `chroma_db/`. Also writes `data/chunks.jsonl`.

```bash
./index.sh
```

- First run can take **3–5 minutes** (downloads/loads the embedding model, embeds ~200+ chunks).
- Required after setup, after `./regenerate.sh`, after adding articles, or if you delete `chroma_db/`.
- Does **not** need Foundry Local.

Equivalent: `python -m rag.ingest`

---

### `retrieve.sh` — search only (no LLM)

Runs vector search and prints matching chunks. Useful for debugging retrieval without loading an LLM.

```bash
./retrieve.sh "path to LLM engineer"
./retrieve.sh "How do I pivot from physics to AI?"
```

Equivalent: `python -m rag.retrieve "your question"`

---

### `ask.sh` — full RAG (retrieve + answer)

Retrieves relevant context from Chroma, then asks Foundry Local to answer using that context.

```bash
# Single question
./ask.sh "How do I plan my career for next year?"

# Multiple questions in one session
./ask.sh --interactive

# CV-aware: retrieval and answer are personalized to your background
./ask.sh --cv /path/to/cv.pdf "What roles fit me?"
./ask.sh --cv /path/to/cv.pdf --interactive --show-cv
```

| Flag | Description |
|------|-------------|
| `-i`, `--interactive` | Ask many questions in one session (type `quit` to exit) |
| `--cv <path>` | PDF, DOCX, TXT, or MD — augments retrieval and personalization |
| `--show-cv` | With `--interactive` + `--cv`: show CV profile and retrieval query per question |
| `--retrieve-only` | Skip the LLM; print retrieved context only |
| `--top-k <n>` | Max chunks to retrieve (default: 5) |
| `--layer <name>` | Filter retrieval to a career-tree layer (e.g. `KNOWLEDGE`) |
| `--model <alias>` | Foundry Local model for answers (default: `qwen2.5-0.5b`) |
| `--extract-model <alias>` | Model for CV parsing (defaults to `--model`) |
| `--no-cache` | Re-extract CV instead of using `data/cv_cache/` |
| `--no-stream` | Disable token streaming |

Equivalent: `python -m rag.ask [flags] [question]`

---

### `recommend.sh` — CV → role recommendations

Extracts structured data from your CV, retrieves matching career paths, then generates a recommendation plan.

```bash
# Full flow: extract → retrieve → recommend
./recommend.sh --cv /path/to/cv.pdf

# Only parse CV to JSON (no recommendations)
./recommend.sh --cv /path/to/cv.pdf --extract-only
```

| Flag | Description |
|------|-------------|
| `--cv <path>` | **Required.** PDF, DOCX, TXT, or MD |
| `--extract-only` | Print extracted CV JSON and stop |
| `--top-k <n>` | How many roles to retrieve (default: 5) |
| `--layer <name>` | Restrict retrieval to one layer |
| `--model <alias>` | Foundry Local model for recommendations |
| `--extract-model <alias>` | Model for CV parsing |
| `--no-cache` | Ignore cached CV extraction |

Equivalent: `python -m rag.cv [flags]`

---

## Typical workflows

**First time**

```bash
./setup.sh && source .venv/bin/activate && ./index.sh
```

**After adding an article to `content/extra/`**

```bash
./index.sh
```

**Refresh tree from the web**

```bash
./regenerate.sh && ./index.sh
```

**Chroma looks empty**

`chroma_db/` is not in git. If the folder exists but has no files, run `./index.sh`.

## Add your own articles

Put `.md` files in `content/extra/`, then run:

```bash
./index.sh
```

Files are split into chunks by `##` headings automatically. `README.md` in that folder is ignored.

## Generated / ignored paths

These are local runtime artifacts (not committed):

- `chroma_db/` — vector index
- `data/chunks.jsonl` — all indexed chunks (regenerated on each `./index.sh`)
- `data/cv_cache/` — cached CV JSON
- `.venv/`
