"""CLI: retrieve career tree context and answer with a Foundry Local model."""

from __future__ import annotations

import argparse
import re

from rag.llm import DEFAULT_MODEL, load_model, make_chat as _make_chat
from rag.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from rag.cv import (
    DEFAULT_CV_EXTRACTION_MODEL,
    load_or_extract_cv,
    build_query_from_cv,
    compact_cv_summary,
)

NO_RELEVANT_CONTEXT_MESSAGE = (
    "I couldn't find a sufficiently relevant path in the career tree for that question."
)


def answer(
    chat,
    question: str,
    context: str,
    stream: bool = True,
    *,
    echo: bool = True,
) -> str:
    """Run one chat completion against an already-loaded chat client."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(context=context, question=question),
        },
    ]

    if stream:
        parts: list[str] = []
        for chunk in chat.complete_streaming_chat(messages):
            if chunk.choices and chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                parts.append(piece)
                if echo:
                    print(piece, end="", flush=True)
        if echo:
            print()
        return "".join(parts)

    completion = chat.complete_chat(messages)
    text = completion.choices[0].message.content or ""
    if echo:
        print(text)
    return text


def ask(
    question: str,
    *,
    chat,
    top_k: int = 5,
    layer: str | None = None,
    cv_query: str | None = None,
    cv_summary: str | None = None,
) -> dict:
    """Retrieve context and answer once. Returns answer, hits, and source lines.

    Does not print to stdout — intended for UI and programmatic callers.
    """
    from rag.retrieve import retrieve

    retrieval_query = _retrieve_query(question, cv_query)
    hits = retrieve(retrieval_query, top_k=top_k, layer=layer)
    sources = _format_sources(hits)
    if not hits:
        return {
            "answer": NO_RELEVANT_CONTEXT_MESSAGE,
            "hits": [],
            "sources": sources,
            "retrieval_query": retrieval_query,
        }

    llm_question = _question_with_cv_profile(question, cv_summary)
    llm_context = _llm_context_from_hits(hits)
    text = answer(chat, llm_question, llm_context, stream=False, echo=False)
    return {
        "answer": text,
        "hits": hits,
        "sources": sources,
        "retrieval_query": retrieval_query,
    }

def _sanitize_text_for_llm(text: str) -> str:
    """Remove source lines and URLs before sending context to the model."""
    cleaned_lines = []
    for line in text.splitlines():
        lower = line.strip().lower()
        if lower.startswith("source:") or lower.startswith("kaynak:"):
            continue
        # Skip bare attribution lines that are mostly links.
        if "http://" in line or "https://" in line:
            line = re.sub(r"https?://\S+", "", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _llm_context_from_hits(hits: list[dict]) -> str:
    sections = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        header = meta.get("full_path") or meta.get("layer") or meta.get("doc_type", "context")
        text = _sanitize_text_for_llm(hit["text"])
        sections.append(f"[{i}] {header}\n{text}")
    return "\n\n---\n\n".join(sections)


def _format_sources(hits: list[dict]) -> str:
    seen = set()
    rows = []
    for hit in hits:
        meta = hit.get("metadata", {})
        source = (meta.get("source_url") or "").strip()
        if not source or source in seen:
            continue
        seen.add(source)
        label = meta.get("full_path") or meta.get("doc_type") or "context"
        rows.append(f"- {label}: {source}")
    return "\n".join(rows) if rows else "- (no source metadata)"


def _retrieve_query(question: str, cv_query: str | None = None) -> str:
    if cv_query:
        return f"{question} ; user_profile: {cv_query}"
    return question


def _question_with_cv_profile(question: str, cv_summary: str | None = None) -> str:
    if cv_summary:
        return f"{question}\n\nUser profile (from CV):\n{cv_summary}\n"
    return question


def _print_cv_debug(cv_query: str | None, cv_summary: str | None) -> None:
    if not cv_query and not cv_summary:
        return
    print("\n=== CV Profile ===\n")
    if cv_summary:
        print("Summary sent to LLM:")
        print(cv_summary)
        print()
    if cv_query:
        print("Profile used for retrieval search:")
        print(cv_query)
        print()


def _print_retrieval_query(question: str, cv_query: str | None) -> None:
    query = _retrieve_query(question, cv_query)
    if cv_query:
        print("\n=== Retrieval Query ===\n")
        print(query)
        print()


def _answer_question(
    chat,
    question: str,
    top_k: int,
    layer: str | None,
    stream: bool,
    hits: list[dict] | None = None,
) -> None:
    from rag.retrieve import format_context, retrieve

    if hits is None:
        hits = retrieve(question, top_k=top_k, layer=layer)
    display_context = format_context(hits)
    print("\n=== Retrieved Context ===\n")
    print(display_context or "(no sufficiently relevant context)")
    print("\n=== Used Sources ===\n")
    print(_format_sources(hits))
    print("\n=== Answer ===\n")
    if not hits:
        print(NO_RELEVANT_CONTEXT_MESSAGE)
        return
    llm_context = _llm_context_from_hits(hits)
    answer(chat, question, llm_context, stream=stream)


def run_interactive(
    model,
    top_k: int,
    layer: str | None,
    stream: bool,
    cv_query: str | None = None,
    cv_summary: str | None = None,
    show_cv_debug: bool = False,
) -> None:
    from rag.retrieve import retrieve

    chat = _make_chat(model)
    print("\nModel ready. Type your question (or 'exit' / 'quit' to leave).\n")
    try:
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            if show_cv_debug:
                _print_retrieval_query(question, cv_query)
            hits = retrieve(_retrieve_query(question, cv_query), top_k=top_k, layer=layer)
            _answer_question(
                chat,
                _question_with_cv_profile(question, cv_summary),
                top_k,
                layer,
                stream,
                hits=hits,
            )
            print()
    finally:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the career tree RAG")
    parser.add_argument("question", nargs="?", help="User question (omit with --interactive)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--layer", default=None, help="Filter by layer name")
    parser.add_argument("--retrieve-only", action="store_true", help="Skip the LLM, show context only")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Foundry Local model alias")
    parser.add_argument(
        "--extract-model",
        default=None,
        help="Foundry Local model alias for CV extraction (defaults to --model)",
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming")
    parser.add_argument("--cv", default=None, help="Optional path to a CV file to augment retrieval & answer")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached CV extraction results")
    parser.add_argument("-i", "--interactive", action="store_true", help="Ask many questions in one session")
    parser.add_argument(
        "--show-cv",
        action="store_true",
        help="Show CV profile and retrieval query used for each question",
    )
    args = parser.parse_args()

    stream = not args.no_stream

    # Optionally augment with CV
    cv_query = None
    cv_summary = None
    cv_path = None
    if args.cv:
        from pathlib import Path

        cv_path = Path(args.cv)
        if not cv_path.exists():
            raise SystemExit(f"CV not found: {cv_path}")

    use_cache = not args.no_cache
    extract_model = args.extract_model or args.model or DEFAULT_CV_EXTRACTION_MODEL

    if args.retrieve_only:
        from rag.retrieve import format_context, retrieve

        if not args.question:
            parser.error("a question is required with --retrieve-only")
        if cv_path:
            cv_json = load_or_extract_cv(cv_path, model_alias=extract_model, use_cache=use_cache)
            cv_query = build_query_from_cv(cv_json)
            if args.show_cv:
                _print_cv_debug(cv_query, compact_cv_summary(cv_json))
        if cv_query:
            _print_retrieval_query(args.question, cv_query)
        hits = retrieve(_retrieve_query(args.question, cv_query), top_k=args.top_k, layer=args.layer)
        print("=== Retrieved Context ===\n")
        print(format_context(hits) or "(no sufficiently relevant context)")
        return

    needs_model = args.interactive or (not args.retrieve_only and args.question)
    if not needs_model:
        parser.error("provide a question or use --interactive")

    model = None
    if cv_path or args.interactive or args.question:
        model = load_model(args.model)

    try:
        if cv_path:
            print("Reading CV…")
            extract_chat = _make_chat(model, temperature=0.1, max_tokens=2000)
            cv_json = load_or_extract_cv(
                cv_path,
                model_alias=extract_model,
                chat=extract_chat,
                use_cache=use_cache,
            )
            cv_query = build_query_from_cv(cv_json)
            cv_summary = compact_cv_summary(cv_json)
            if args.show_cv or args.interactive:
                _print_cv_debug(cv_query, cv_summary)

        if args.interactive:
            run_interactive(
                model,
                args.top_k,
                args.layer,
                stream,
                cv_query,
                cv_summary,
                show_cv_debug=bool(cv_query),
            )
            return

        if not args.question:
            parser.error("provide a question or use --interactive")

        from rag.retrieve import retrieve

        if cv_query and args.show_cv:
            _print_retrieval_query(args.question, cv_query)
        hits = retrieve(_retrieve_query(args.question, cv_query), top_k=args.top_k, layer=args.layer)
        if not hits:
            print(NO_RELEVANT_CONTEXT_MESSAGE)
            return

        chat = _make_chat(model)
        _answer_question(
            chat,
            _question_with_cv_profile(args.question, cv_summary),
            args.top_k,
            args.layer,
            stream,
            hits=hits,
        )
    finally:
        if model is not None:
            model.unload()


if __name__ == "__main__":
    main()
