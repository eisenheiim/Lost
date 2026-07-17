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


def extract_answer(stdout: str) -> str:
    """Pull the LLM answer block from ask.sh output when present."""
    marker = "=== Answer ==="
    if marker in stdout:
        return stdout.split(marker, 1)[1].strip() or "(empty answer)"
    return stdout.strip() or "(no output)"


def extract_json_object(text: str) -> dict | list | None:
    """Find and parse the first top-level JSON object/array in mixed stdout."""
    # Prefer fenced or indented JSON dumps from recommend --extract-only
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
        # Fallback: try progressively smaller slices ending at each matching brace
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
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Instrument+Serif:ital@0;1&display=swap');

      html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
      }
      .stApp {
        background:
          radial-gradient(1200px 600px at 10% -10%, #d8efe6 0%, transparent 55%),
          radial-gradient(900px 500px at 100% 0%, #e7eef8 0%, transparent 50%),
          linear-gradient(180deg, #f7faf8 0%, #eef3f0 100%);
      }
      h1, h2, h3 {
        font-family: "Instrument Serif", Georgia, serif !important;
        letter-spacing: -0.02em;
      }
      .block-container {
        padding-top: 2rem;
        max-width: 960px;
      }
      div[data-testid="stTabs"] button[role="tab"] {
        font-weight: 500;
      }
      .hero-sub {
        color: #3d5248;
        font-size: 1.05rem;
        margin-top: -0.6rem;
        margin-bottom: 1.5rem;
      }
      .model-answer {
        background: linear-gradient(135deg, #eef7ff 0%, #f3efff 100%);
        border: 1px solid #bfd7ff;
        border-left: 5px solid #4f8cff;
        border-radius: 18px;
        color: #17324d;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(48, 87, 135, 0.10);
        line-height: 1.6;
        white-space: pre-wrap;
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
    '<p class="hero-sub">Yerel kariyer asistanı · Chroma + Foundry Local</p>',
    unsafe_allow_html=True,
)

tab_chat, tab_cv, tab_index = st.tabs(
    ["💬 RAG Sohbet", "📄 CV Analiz & Öneri", "⚙️ Doküman İndeksleme"]
)


# ---------------------------------------------------------------------------
# Tab 1 — RAG chat
# ---------------------------------------------------------------------------

with tab_chat:
    st.subheader("Sohbet")
    st.caption(
        "Sorular `./ask.sh` üzerinden Foundry Local ile yanıtlanır. "
        "İlk çalıştırmada model yüklemesi 1–3 dakika sürebilir."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and not msg.get("is_error"):
                render_model_answer(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("raw"):
                with st.expander("Ham script çıktısı"):
                    st.code(msg["raw"], language="text")

    question = st.chat_input("Kariyer sorunuzu yazın…")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("ask.sh çalışıyor…"):
                try:
                    proc = run_script(ASK_SH, [question, "--no-stream"])
                    raw = (proc.stdout or "") + (
                        f"\n\n[stderr]\n{proc.stderr}" if proc.stderr else ""
                    )
                    if proc.returncode != 0:
                        answer = (
                            f"Script hata ile bitti (exit {proc.returncode}).\n\n"
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
                        with st.expander("Ham script çıktısı"):
                            st.code(raw.strip() or "(boş)", language="text")
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": answer, "raw": raw}
                        )
                except subprocess.TimeoutExpired:
                    err = "Zaman aşımı: ask.sh 10 dakikadan uzun sürdü."
                    st.error(err)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": err, "is_error": True}
                    )
                except Exception as exc:  # noqa: BLE001
                    err = f"Beklenmeyen hata: {exc}"
                    st.error(err)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": err, "is_error": True}
                    )

    if st.session_state.chat_messages and st.button("Sohbeti temizle", key="clear_chat"):
        st.session_state.chat_messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 2 — CV analysis
# ---------------------------------------------------------------------------

with tab_cv:
    st.subheader("CV Analiz & Öneri")
    st.caption(
        "PDF veya TXT yükleyin. Analiz `./recommend.sh --cv … --extract-only` ile "
        "yapılandırılmış JSON üretir."
    )

    uploaded = st.file_uploader(
        "CV dosyası",
        type=["pdf", "txt", "md", "docx"],
        help="Sürükle-bırak veya seç. Maks. boyut Streamlit varsayılanına bağlıdır.",
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        analyze = st.button("Analiz Et", type="primary", disabled=uploaded is None)

    if analyze and uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower() or ".txt"
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                dir=str(ROOT / "data" / "cv_cache"),
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            (ROOT / "data" / "cv_cache").mkdir(parents=True, exist_ok=True)

            with st.spinner("recommend.sh çalışıyor (Foundry Local)…"):
                proc = run_script(
                    RECOMMEND_SH,
                    ["--cv", str(tmp_path), "--extract-only"],
                )

            raw = (proc.stdout or "") + (
                f"\n\n[stderr]\n{proc.stderr}" if proc.stderr else ""
            )

            if proc.returncode != 0:
                st.error(f"Script hata ile bitti (exit {proc.returncode})")
                st.code((proc.stderr or proc.stdout or "")[-3000:], language="text")
            else:
                parsed = extract_json_object(proc.stdout or "")
                if parsed is None:
                    st.warning("JSON bulunamadı; ham çıktı gösteriliyor.")
                    st.code(raw.strip() or "(boş)", language="text")
                else:
                    st.success("CV analizi tamamlandı")
                    st.json(parsed)
                    with st.expander("Ham script çıktısı"):
                        st.code(raw.strip(), language="text")
        except subprocess.TimeoutExpired:
            st.error("Zaman aşımı: recommend.sh 10 dakikadan uzun sürdü.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Beklenmeyen hata: {exc}")


# ---------------------------------------------------------------------------
# Tab 3 — Document indexing
# ---------------------------------------------------------------------------

with tab_index:
    st.subheader("Doküman İndeksleme")
    st.caption(
        "Varsayılan olarak career tree + `content/extra/` indekslenir. "
        "İsterseniz ekstra markdown klasörü verin; `./index.sh <klasör>` çalışır."
    )

    folder = st.text_input(
        "İndekslenecek klasör yolu (opsiyonel)",
        value=str(ROOT / "content" / "extra"),
        help="Bu klasördeki .md dosyaları extra içerik olarak indekslenir.",
    )

    if st.button("İndekslemeyi Başlat", type="primary"):
        folder_path = Path(folder).expanduser() if folder.strip() else None
        args: list[str] = []

        if folder_path is not None:
            if not folder_path.is_dir():
                st.error(f"Klasör bulunamadı: {folder_path}")
            else:
                args = [str(folder_path.resolve())]

        if folder_path is None or folder_path.is_dir():
            with st.spinner("index.sh çalışıyor…"):
                try:
                    proc = run_script(INDEX_SH, args, timeout=DEFAULT_TIMEOUT)
                    raw = (proc.stdout or "") + (
                        f"\n\n[stderr]\n{proc.stderr}" if proc.stderr else ""
                    )
                    if proc.returncode != 0:
                        st.error(f"İndeksleme başarısız (exit {proc.returncode})")
                        st.code(raw.strip()[-3000:], language="text")
                    else:
                        st.success("İndeksleme tamamlandı")
                        if proc.stdout.strip():
                            st.code(proc.stdout.strip(), language="text")
                except subprocess.TimeoutExpired:
                    st.error("Zaman aşımı: index.sh 10 dakikadan uzun sürdü.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Beklenmeyen hata: {exc}")
