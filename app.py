"""Career Tree RAG — local Streamlit UI.

Runs project shell scripts via subprocess so Foundry Local stays outside
Streamlit's event loop.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ASK_SH = ROOT / "ask.sh"
RECOMMEND_SH = ROOT / "recommend.sh"
INDEX_SH = ROOT / "index.sh"

# Foundry / indexing can take several minutes on first run
DEFAULT_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_script(
    script: Path,
    args: list[str] | None = None,
    *,
    timeout: int = DEFAULT_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run a project .sh script and return the completed process."""
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = ["bash", str(script), *(args or [])]
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


_MODEL_PROGRESS_RE = re.compile(
    r"^(?:Preparing execution providers\.*|Downloading model\b.*|"
    r"Loading model\b.*|\s*\d{1,3}(?:\.\d)?%\s*)$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_answer(stdout: str) -> str:
    """Pull the LLM answer block from the ask output when present."""
    marker = "=== Answer ==="
    if marker in stdout:
        text = stdout.split(marker, 1)[1].strip()
    else:
        text = _MODEL_PROGRESS_RE.sub("", stdout).strip()
    return text or "(empty answer)"


def ask_career_question(question: str, *, first_question: bool) -> None:
    """Run ask.sh, append the exchange to session state, and render the reply."""
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    spinner = (
        "Thinking… (first question can take a few minutes while the model loads)"
        if first_question
        else "Thinking…"
    )
    with st.chat_message("assistant"):
        with st.spinner(spinner):
            try:
                proc = run_script(ASK_SH, [question, "--no-stream"])
                raw = (proc.stdout or "") + (
                    f"\n\n[stderr]\n{proc.stderr}" if proc.stderr else ""
                )
                if proc.returncode != 0:
                    answer = (
                        "Something went wrong while answering.\n\n"
                        f"```\n{(proc.stderr or proc.stdout or '').strip()[-2000:]}\n```"
                    )
                    st.error(answer)
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "raw": raw,
                            "is_error": True,
                        }
                    )
                else:
                    answer = extract_answer(proc.stdout or "")
                    render_model_answer(answer)
                    with st.expander("Details"):
                        st.code(raw.strip() or "(empty)", language="text")
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": answer, "raw": raw}
                    )
            except subprocess.TimeoutExpired:
                err = "Timed out: the answer took longer than 10 minutes."
                st.error(err)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": err, "is_error": True}
                )
            except Exception as exc:  # noqa: BLE001
                err = f"Unexpected error: {exc}"
                st.error(err)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": err, "is_error": True}
                )

    st.session_state.chat_model_ready = True
    st.rerun()


def extract_json_object(text: str) -> dict | list | None:
    """Find and parse the first top-level JSON object/array in mixed stdout."""
    start_obj = text.find("{")
    start_arr = text.find("[")
    if start_obj == -1 and start_arr == -1:
        return None

    if start_obj == -1:
        start = start_arr
        end_char = "]"
    elif start_arr == -1:
        start = start_obj
        end_char = "}"
    else:
        start = min(start_obj, start_arr)
        end_char = "}" if start == start_obj else "]"

    end = text.rfind(end_char)
    if end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        for match in re.finditer(r"\{[\s\S]*\}", text):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Career Tree RAG",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap');

      html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
      }
      .stApp {
        background:
          radial-gradient(1200px 600px at 10% -10%, #d8efe6 0%, transparent 55%),
          radial-gradient(900px 500px at 100% 0%, #e7eef8 0%, transparent 50%),
          linear-gradient(180deg, #f7faf8 0%, #eef3f0 100%);
      }

      /* Force dark, readable text everywhere */
      .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
      .stMarkdown, div[data-testid="stCaptionContainer"] {
        color: #1c2b24 !important;
      }

      h1, h2, h3 {
        font-family: "Instrument Serif", Georgia, serif !important;
        letter-spacing: -0.02em;
        color: #14231c !important;
      }
      h1 { font-size: 4rem !important; line-height: 1.15 !important; }
      h2, h3 { font-size: 2.4rem !important; line-height: 1.2 !important; }

      .block-container {
        padding-top: 2.8rem;
        padding-bottom: 3rem;
        max-width: 1180px;
        font-size: 1.35rem;
      }

      /* Captions / helper text */
      div[data-testid="stCaptionContainer"],
      div[data-testid="stCaptionContainer"] p {
        font-size: 1.2rem !important;
        line-height: 1.55 !important;
      }

      /* Tabs: Career Chat / CV Analysis / Indexing — large & always readable */
      div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.6rem;
        margin-bottom: 0.5rem;
      }
      div[data-testid="stTabs"] button[role="tab"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #24523c !important;
        opacity: 1 !important;
        padding: 1.4rem 2.2rem !important;
        min-height: 4.2rem !important;
        background: rgba(255, 255, 255, 0.75);
        border-radius: 18px 18px 0 0;
        margin-right: 0.35rem;
      }
      div[data-testid="stTabs"] button[role="tab"] p {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #24523c !important;
        line-height: 1.2 !important;
      }
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #ffffff;
        color: #0d3b26 !important;
        box-shadow: 0 -4px 16px rgba(36, 82, 60, 0.14);
      }
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
        color: #0d3b26 !important;
      }
      div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
        background-color: #2f8f5b !important;
        height: 6px !important;
      }

      .hero-sub {
        color: #3d5248 !important;
        font-size: 1.65rem;
        margin-top: -0.4rem;
        margin-bottom: 2.2rem;
        line-height: 1.45;
      }

      /* Chat bubbles / cards */
      .model-answer {
        background: linear-gradient(135deg, #eef7ff 0%, #f3efff 100%);
        border: 1px solid #bfd7ff;
        border-left: 7px solid #4f8cff;
        border-radius: 20px;
        color: #17324d !important;
        padding: 1.5rem 1.7rem;
        box-shadow: 0 10px 24px rgba(48, 87, 135, 0.10);
        line-height: 1.75;
        font-size: 1.35rem;
        white-space: pre-wrap;
      }
      .model-answer * { color: #17324d !important; }

      div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.72);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        font-size: 1.3rem !important;
      }
      div[data-testid="stChatMessage"] p {
        font-size: 1.3rem !important;
        line-height: 1.65 !important;
      }

      /* Inputs: dark text on white — typing must stay readable */
      div[data-testid="stChatInput"],
      div[data-testid="stChatInput"] > div,
      div[data-testid="stChatInput"] textarea,
      div[data-testid="stChatInput"] [contenteditable="true"],
      div[data-testid="stChatInput"] div[role="textbox"],
      .stChatInput textarea,
      .stTextInput input,
      .stTextInput div[data-baseweb="input"],
      .stTextInput div[data-baseweb="input"] input {
        background-color: #ffffff !important;
        color: #14231c !important;
        -webkit-text-fill-color: #14231c !important;
        caret-color: #14231c !important;
        font-size: 1.35rem !important;
        min-height: 3.2rem !important;
        line-height: 1.5 !important;
      }
      div[data-testid="stChatInput"] textarea::placeholder,
      .stTextInput input::placeholder {
        color: #5a6e64 !important;
        -webkit-text-fill-color: #5a6e64 !important;
        opacity: 1 !important;
        font-size: 1.35rem !important;
      }
      div[data-testid="stChatInput"] {
        border-radius: 18px !important;
      }
      .stTextInput label p,
      div[data-testid="stFileUploader"] label p,
      div[data-testid="stWidgetLabel"] p {
        font-size: 1.25rem !important;
      }

      .stButton button {
        font-size: 1.3rem !important;
        padding: 0.9rem 1.9rem !important;
        border-radius: 14px !important;
        min-height: 3.2rem !important;
      }
      .stButton button p {
        font-size: 1.3rem !important;
      }
      .stButton button[kind="primary"] {
        background: #2f8f5b !important;
        border: none !important;
      }
      .stButton button[kind="primary"] p { color: #ffffff !important; }

      div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.72);
        border-radius: 18px;
        padding: 1.3rem;
        font-size: 1.25rem !important;
      }
      div[data-testid="stFileUploader"] p,
      div[data-testid="stFileUploader"] span {
        font-size: 1.2rem !important;
      }

      div[data-testid="stExpander"] summary p {
        font-size: 1.25rem !important;
        color: #24523c !important;
      }

      /* Alerts / success / warnings */
      div[data-testid="stAlert"] p {
        font-size: 1.25rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_model_answer(text: str) -> None:
    """Render model output in a distinct colored card."""
    safe = escape(text or "(empty answer)")
    st.markdown(f'<div class="model-answer">{safe}</div>', unsafe_allow_html=True)


st.title("Career Tree RAG")
st.markdown(
    '<p class="hero-sub">Your local career advisor · private, on-device AI</p>',
    unsafe_allow_html=True,
)

tab_chat, tab_cv, tab_index = st.tabs(
    ["💬 Career Chat", "📄 CV Analysis", "⚙️ Indexing"]
)


# ---------------------------------------------------------------------------
# Tab 1 — RAG chat
# ---------------------------------------------------------------------------

with tab_chat:
    st.subheader("Ask about your career")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_model_ready" not in st.session_state:
        st.session_state.chat_model_ready = False

    if not st.session_state.chat_model_ready:
        st.caption(
            "Answers are generated locally with retrieval over the Career Tree. "
            "The first question can take 1–3 minutes while the model loads."
        )
    else:
        st.caption("Answers are generated locally with retrieval over the Career Tree.")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and not msg.get("is_error"):
                render_model_answer(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("raw"):
                with st.expander("Details"):
                    st.code(msg["raw"], language="text")

    first_question = not st.session_state.chat_model_ready

    # After the first answer, show a follow-up ask box directly under the chat.
    if st.session_state.chat_messages:
        st.markdown("#### Ask another question")
        with st.form("followup_question", clear_on_submit=True):
            followup = st.text_input(
                "Your question",
                placeholder="Type your next career question…",
                label_visibility="collapsed",
            )
            asked = st.form_submit_button("Ask", type="primary")
        if asked and followup.strip():
            ask_career_question(followup.strip(), first_question=False)
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.session_state.chat_model_ready = False
            st.rerun()
    else:
        question = st.chat_input("Type your career question…")
        if question:
            ask_career_question(question.strip(), first_question=first_question)


# ---------------------------------------------------------------------------
# Tab 2 — CV analysis
# ---------------------------------------------------------------------------

with tab_cv:
    st.subheader("CV analysis & recommendations")
    st.caption(
        "Upload your CV (PDF, TXT, MD, or DOCX). It is parsed locally into a "
        "structured profile — nothing leaves your machine."
    )

    uploaded = st.file_uploader(
        "Upload your CV",
        type=["pdf", "txt", "md", "docx"],
        help="Drag & drop or browse.",
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

    if analyze and uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower() or ".txt"
        try:
            (ROOT / "data" / "cv_cache").mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                dir=str(ROOT / "data" / "cv_cache"),
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            with st.spinner("Analyzing your CV locally… (first run can take a few minutes)"):
                proc = run_script(
                    RECOMMEND_SH,
                    ["--cv", str(tmp_path), "--extract-only"],
                )

            raw = (proc.stdout or "") + (
                f"\n\n[stderr]\n{proc.stderr}" if proc.stderr else ""
            )

            if proc.returncode != 0:
                st.error("CV analysis failed.")
                st.code((proc.stderr or proc.stdout or "")[-3000:], language="text")
            else:
                parsed = extract_json_object(proc.stdout or "")
                if parsed is None:
                    st.warning("Could not parse a structured profile; showing raw output.")
                    st.code(raw.strip() or "(empty)", language="text")
                else:
                    st.success("CV analysis complete")
                    st.json(parsed)
                    with st.expander("Details"):
                        st.code(raw.strip(), language="text")
        except subprocess.TimeoutExpired:
            st.error("Timed out: CV analysis took longer than 10 minutes.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Tab 3 — Document indexing
# ---------------------------------------------------------------------------

with tab_index:
    st.subheader("Build the search index")
    st.caption(
        "Indexes the Career Tree plus your own markdown articles so they become "
        "searchable. Run this once after setup, and again whenever you add articles."
    )

    folder = st.text_input(
        "Folder with extra markdown articles (optional)",
        value=str(ROOT / "content" / "extra"),
        help="All .md files in this folder are indexed as extra content.",
    )

    if st.button("Start indexing", type="primary"):
        folder_path = Path(folder).expanduser() if folder.strip() else None
        args: list[str] = []

        if folder_path is not None:
            if not folder_path.is_dir():
                st.error(f"Folder not found: {folder_path}")
            else:
                args = [str(folder_path.resolve())]

        if folder_path is None or folder_path.is_dir():
            with st.spinner("Indexing… this can take a few minutes."):
                try:
                    proc = run_script(INDEX_SH, args, timeout=DEFAULT_TIMEOUT)
                    raw = (proc.stdout or "") + (
                        f"\n\n[stderr]\n{proc.stderr}" if proc.stderr else ""
                    )
                    if proc.returncode != 0:
                        st.error("Indexing failed.")
                        st.code(raw.strip()[-3000:], language="text")
                    else:
                        st.success("Indexing complete")
                        if proc.stdout.strip():
                            with st.expander("Details"):
                                st.code(proc.stdout.strip(), language="text")
                except subprocess.TimeoutExpired:
                    st.error("Timed out: indexing took longer than 10 minutes.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected error: {exc}")
