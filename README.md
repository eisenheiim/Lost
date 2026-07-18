# Career Tree RAG (local)

Local career advisor: **ChromaDB** + **Sentence Transformers** + **Azure Foundry Local**, with an optional Streamlit UI.

Career path data is derived from [Ibrahim’s Career Tree](https://www.ibz04.pro/blog/career-tree) (ibz04.pro), with local extensions (e.g. social careers) and optional markdown articles in `content/extra/`. Please credit that source if you republish the tree content.

**License:** MIT — see [`LICENSE`](LICENSE).  
**Version:** 1.0.0

This is a local career-exploration tool, not professional counseling or hiring advice. Answers depend on your indexed content and the local model you choose.

---

## Before you start (common teammate pitfalls)

1. **Install [Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/)** before chat or CV analysis. Indexing (`./index.sh`) does **not** need it.
2. **First LLM call takes 1–3 minutes** while Foundry downloads/loads the model. Watch the terminal — it is not frozen.
3. **Avoid putting the venv / Chroma on iCloud Desktop.** If this repo lives under `~/Desktop` (iCloud), a project-local `.venv` often hangs with 0% CPU. `./setup.sh` installs the venv and Chroma under `~/.career_tree_rag/` on purpose.
4. **Run `./index.sh` after setup** (and after changing the career tree or articles). Without it, search is empty.
5. **Do not run `./regenerate.sh` if you customized `data/career-tree.json`.** Regenerate re-fetches the website tree and **overwrites** local edits (including added social careers). After manual JSON edits, only run `./index.sh`.

---

## What you need

- Python 3.11+
- [Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/) — for `./ask.sh`, `./recommend.sh`, and the UI chat/CV tabs
- Network once for the embedding model (`all-MiniLM-L6-v2` from Hugging Face) and Foundry models

## Quick start (teammates)

```bash
git clone <repo-url>
cd healthrag   # or whatever the folder is named

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
| `qwen2.5-0.5b` (Foundry Local) | Default answers (`ask`) and CV JSON extraction (`recommend` / `--cv`) |

The default LLM is **small and fast**, good for demos, but weak for nuanced career advice. Prefer a larger Foundry alias you already have cached, e.g.:

```bash
./ask.sh --model Phi-4-mini-instruct "What social careers fit someone who likes teaching?"
```

### Privacy (CVs)

- CV upload and extraction run **locally** via Foundry Local; files are not sent to a cloud API by this project.
- Parsed results may be cached under `data/cv_cache/` (gitignored). Do not commit that folder.
- Prefer not to use real sensitive CVs on shared machines; delete `data/cv_cache/` when done.

### Important paths (not in git)

| Path | Purpose |
|------|---------|
| `~/.career_tree_rag/venv` | Python virtualenv (created by `./setup.sh`) |
| `~/.career_tree_rag/chroma_db` | Vector index (created by `./index.sh`) |
| `~/.career_tree_rag/cache/models/` | Foundry Local model weights |
| `data/cv_cache/` | Cached CV extractions (local only) |
| `data/chunks.jsonl` | All indexed chunks (rewritten on each `./index.sh`) |

Optional overrides:

```bash
export HEALTHRAG_VENV=/path/to/venv
export HEALTHRAG_CHROMA_DIR=/path/to/chroma_db
export HEALTHRAG_EXTRA_DIR=/path/to/extra/markdown   # used by ./index.sh <folder>
```

## Streamlit UI (`./ui.sh`)

Three tabs in `app.py`:

1. **Career Chat** — ask career questions (runs retrieval + local LLM)
2. **CV Analysis** — upload PDF/TXT/DOCX → structured profile JSON
3. **Indexing** — optional folder path → rebuild the search index

```bash
./ui.sh
# http://127.0.0.1:8501
```

Indexing must already be done once (`./index.sh`). First chat answer can take **1–3 minutes** while the model loads.

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

Turns the career tree + extra markdown into searchable embeddings in Chroma, and rewrites `data/chunks.jsonl`.

```bash
./index.sh
./index.sh /path/to/extra/markdown   # optional: set HEALTHRAG_EXTRA_DIR to that folder
```

- First run can take **3–5 minutes**
- Required after setup, after editing `data/career-tree.json`, after adding articles, or if search returns nothing
- Does **not** need Foundry Local

Equivalent: `python -m rag.ingest`

---

### `regenerate.sh` — refresh career tree from the web (optional, destructive)

Re-downloads the tree from [ibz04.pro](https://www.ibz04.pro/blog/career-tree) and **overwrites**:

- `data/career-tree.json`
- `content/career-tree.md`

Any local roles you added (e.g. Social Impact & People Layer) will be **lost**. Only use this when you intentionally want the upstream website tree again, then run `./index.sh`.

```bash
./regenerate.sh && ./index.sh
```

If you edited the JSON by hand, **skip regenerate** and only run `./index.sh`.

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

**After editing `data/career-tree.json` or adding an article**

```bash
./index.sh
# do NOT run ./regenerate.sh — that would wipe local tree edits
```

**Reset tree from the website (destructive)**

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
| Local career roles disappeared | You ran `./regenerate.sh` — it overwrites `career-tree.json`; restore from git and `./index.sh` |
| `Port 8501 is not available` | Stop the other Streamlit (`Ctrl+C`) or use another port |

## Generated / ignored (not committed)

- `~/.career_tree_rag/venv`
- `~/.career_tree_rag/chroma_db`
- `data/cv_cache/`
- Project-local `.venv/` / `chroma_db/` if present
- `__pycache__/`, `.DS_Store`
