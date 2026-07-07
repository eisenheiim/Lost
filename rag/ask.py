"""CLI: retrieve career tree context and answer with a Foundry Local model."""

from __future__ import annotations

import argparse

from rag.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from rag.retrieve import format_context, retrieve

DEFAULT_MODEL = "qwen2.5-0.5b"


def load_model(model_alias: str):
    """Initialize Foundry Local, download + load the model, return an IModel."""
    from foundry_local_sdk import Configuration, FoundryLocalManager

    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name="career_tree_rag"))
    manager = FoundryLocalManager.instance

    print("Preparing execution providers...")
    manager.download_and_register_eps()

    model = manager.catalog.get_model(model_alias)
    if model is None:
        raise SystemExit(f"Model '{model_alias}' not found in the Foundry Local catalog.")

    if not model.is_cached:
        print(f"Downloading model '{model_alias}'...")
        model.download(lambda pct: print(f"\r  {pct:5.1f}%", end="", flush=True))
        print()

    print(f"Loading model '{model_alias}'...")
    model.load()
    return model


def answer(chat, question: str, context: str, stream: bool = True) -> str:
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
                print(piece, end="", flush=True)
        print()
        return "".join(parts)

    completion = chat.complete_chat(messages)
    text = completion.choices[0].message.content or ""
    print(text)
    return text


def _make_chat(model, temperature: float = 0.3, max_tokens: int = 800):
    chat = model.get_chat_client()
    chat.settings.temperature = temperature
    chat.settings.max_tokens = max_tokens
    return chat


def _answer_question(chat, question: str, top_k: int, layer: str | None, stream: bool) -> None:
    hits = retrieve(question, top_k=top_k, layer=layer)
    context = format_context(hits)
    print("\n=== Retrieved Context ===\n")
    print(context)
    print("\n=== Answer ===\n")
    answer(chat, question, context, stream=stream)


def run_interactive(model_alias: str, top_k: int, layer: str | None, stream: bool) -> None:
    model = load_model(model_alias)
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
            _answer_question(chat, question, top_k, layer, stream)
            print()
    finally:
        model.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the career tree RAG")
    parser.add_argument("question", nargs="?", help="User question (omit with --interactive)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--layer", default=None, help="Filter by layer name")
    parser.add_argument("--retrieve-only", action="store_true", help="Skip the LLM, show context only")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Foundry Local model alias")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming")
    parser.add_argument("-i", "--interactive", action="store_true", help="Ask many questions in one session")
    args = parser.parse_args()

    stream = not args.no_stream

    if args.retrieve_only:
        if not args.question:
            parser.error("a question is required with --retrieve-only")
        hits = retrieve(args.question, top_k=args.top_k, layer=args.layer)
        print("=== Retrieved Context ===\n")
        print(format_context(hits))
        return

    if args.interactive:
        run_interactive(args.model, args.top_k, args.layer, stream)
        return

    if not args.question:
        parser.error("provide a question or use --interactive")

    model = load_model(args.model)
    chat = _make_chat(model)
    try:
        _answer_question(chat, args.question, args.top_k, args.layer, stream)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
