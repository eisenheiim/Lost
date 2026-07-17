# Career Tree RAG (local)

Local career advisor: **ChromaDB** + **Sentence Transformers** + **Azure Foundry Local**, with an optional Streamlit UI.

Content comes from [Ibrahim’s Career Tree](https://www.ibz04.pro/blog/career-tree) plus markdown articles in `content/extra/`.

## What you need

- Python 3.11+
- [Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/) — for `./ask.sh`, `./recommend.sh`, and the UI chat/CV tabs (not needed for `./index.sh` or `./retrieve.sh`)
- Network once for the embedding model (`all-MiniLM-L6-v2` from Hugging Face) and Foundry models

## Quick start (teammates)

```bash
git clone <repo-url>
cd careerag   # or healthrag — whatever the folder is named

chmod +x setup.sh index.sh ask.sh retrieve.sh regenerate.sh recommend.sh ui.sh _env.sh
./setup.sh
./index.sh          # required before ask / UI search — builds Chroma (~3–5 min first time)
./ask.sh --interactive
# or open the UI:
./ui.sh             # http://127.0.0.1:8501
```

You do **not** need to manually `source` a venv for the `.sh` scripts — they call `_env.sh` and activate the right environment themselves.

### Models

| Model | Role |
|--------|------|
| `all-MiniLM-L6-v2` | Embeddings for indexing & retrieval (Hugging Face → `~/.cache/huggingface/`) |
| `qwen2.5-0.5b` (Foundry Local) | Answers (`ask`) and CV JSON extraction (`recommend` / `--cv`) |

Override LLM with `--model <alias>` (e.g. a stronger Foundry model you already cached).

### Important paths (not in git)

| Path | Purpose |
|------|---------|
| `~/.career_tree_rag/venv` | Python virtualenv (created by `./setup.sh`) |
| `~/.career_tree_rag/chroma_db` | Vector index (created by `./index.sh`) |
| `~/.career_tree_rag/cache/models/` | Foundry Local model weights |
| `data/cv_cache/` | Cached CV extractions |
| `data/chunks.jsonl` | All indexed chunks (rewritten on each `./index.sh`) |

Why off Desktop? If the repo lives under iCloud Desktop (`~/Desktop/...`), installing/running torch & chromadb from a project-local `.venv` often **hangs with 0% CPU**. Setup therefore puts the venv and Chroma under `~/.career_tree_rag/` by default.

Optional overrides:

```bash
export HEALTHRAG_VENV=/path/to/venv
export HEALTHRAG_CHROMA_DIR=/path/to/chroma_db
export HEALTHRAG_EXTRA_DIR=/path/to/extra/markdown   # used by ./index.sh <folder>
```

## Streamlit UI (`./ui.sh`)

Three tabs in `app.py`:

1. **RAG Sohbet** — runs `./ask.sh "<question>" --no-stream`
2. **CV Analiz & Öneri** — upload PDF/TXT → `./recommend.sh --cv … --extract-only` → shows JSON
3. **Doküman İndeksleme** — optional folder path → `./index.sh [folder]`

Equivalent:

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

First LLM call can take **1–3 minutes** while Foundry loads the model. Indexing must already be done (`./index.sh`).

## CLI reference

All `.sh` scripts activate the venv via `_env.sh`. After `source ~/.career_tree_rag/venv/bin/activate` you can also run `python -m rag.<module>`.

### `setup.sh` — one-time install

Creates `~/.career_tree_rag/venv` and installs `requirements.txt` + the package.

```bash
./setup.sh
```

Does **not** build the search index — run `./index.sh` next.

---

### `index.sh` — build the vector index

Embeds career tree + `content/extra/*.md` into Chroma. Also writes `data/chunks.jsonl`.

```bash
./index.sh
./index.sh /path/to/extra/markdown   # optional: set HEALTHRAG_EXTRA_DIR to that folder
```

- First run can take **3–5 minutes**
- Required after setup, after `./regenerate.sh`, after adding articles, or if search returns nothing
- Does **not** need Foundry Local

Equivalent: `python -m rag.ingest`

---

### `regenerate.sh` — refresh career tree from the web (optional)

Writes `data/career-tree.json` and `content/career-tree.md` from [ibz04.pro](https://www.ibz04.pro/blog/career-tree).

```bash
./regenerate.sh && ./index.sh
```

Equivalent: `python -m rag.regenerate`

---

### `retrieve.sh` — search only (no LLM)

```bash
./retrieve.sh "path to LLM engineer"
```

Equivalent: `python -m rag.retrieve "your question"`

---

### `ask.sh` — retrieve + answer

```bash
./ask.sh "How do I plan my career for next year?"
./ask.sh --interactive
./ask.sh --cv /path/to/cv.pdf "What roles fit me?"
./ask.sh --cv /path/to/cv.pdf --interactive --show-cv
```

| Flag | Description |
|------|-------------|
| `-i`, `--interactive` | Many questions in one session (`quit` to exit) |
| `--cv <path>` | PDF, DOCX, TXT, or MD — personalizes retrieval & answer |
| `--show-cv` | Show CV profile / retrieval query (useful with `--interactive`) |
| `--retrieve-only` | Skip LLM; print context only |
| `--top-k <n>` | Max chunks (default: 5) |
| `--layer <name>` | Filter by career-tree layer |
| `--model <alias>` | Foundry model (default: `qwen2.5-0.5b`) |
| `--extract-model <alias>` | Model for CV parsing |
| `--no-cache` | Ignore `data/cv_cache/` |
| `--no-stream` | Disable token streaming |

Equivalent: `python -m rag.ask [flags] [question]`

---

### `recommend.sh` — CV → structured extract / recommendations

```bash
./recommend.sh --cv /path/to/cv.pdf
./recommend.sh --cv /path/to/cv.pdf --extract-only
```

| Flag | Description |
|------|-------------|
| `--cv <path>` | **Required** |
| `--extract-only` | Print CV JSON only |
| `--top-k <n>` | Roles to retrieve (default: 5) |
| `--layer <name>` | Restrict retrieval |
| `--model` / `--extract-model` | Foundry aliases |
| `--no-cache` | Force fresh CV extraction |

Equivalent: `python -m rag.cv [flags]`

## Typical workflows

**First time**

```bash
./setup.sh && ./index.sh
./ui.sh
# or: ./ask.sh --interactive
```

**After adding an article to `content/extra/`**

```bash
./index.sh
```

**Refresh tree from the web**

```bash
./regenerate.sh && ./index.sh
```

**Search returns nothing / Chroma missing**

```bash
./index.sh
```

## Add your own articles

Put `.md` files in `content/extra/`, then run `./index.sh`.

Files are split by `##` headings. `content/extra/README.md` is ignored.

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| Terminal “freezes” on `./index.sh` / imports with 0% CPU | Repo on iCloud Desktop — use `./setup.sh` (venv under `~/.career_tree_rag/`) or move the clone off Desktop |
| `No virtualenv found` | Run `./setup.sh` |
| UI / ask hangs on first question | Foundry Local loading model (1–3 min); watch the `./ui.sh` terminal for progress |
| Empty retrieval / Chroma errors | Run `./index.sh` again |
| `Port 8501 is not available` | Stop the other Streamlit (`Ctrl+C`) or use another port |

## Generated / ignored (not committed)

- `~/.career_tree_rag/venv`
- `~/.career_tree_rag/chroma_db`
- `data/cv_cache/`
- Project-local `.venv/` / `chroma_db/` if present
- `__pycache__/`, `.DS_Store`
