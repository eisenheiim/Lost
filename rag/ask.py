"""CLI: retrieve context and optionally answer with Foundry Local."""

from __future__ import annotations

import argparse

from rag.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from rag.retrieve import format_context, retrieve


def answer_with_foundry(question: str, context: str, model_id: str) -> str:
    import openai
    from foundry_local_sdk import Configuration, FoundryLocalManager

    config = Configuration(app_name="career_tree_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    manager.download_and_register_eps()
    model = manager.catalog.get_model(model_id)
    model.download()
    model.load()
    manager.start_web_service()

    client = openai.OpenAI(
        base_url=f"{manager.urls[0]}/v1",
        api_key="none",
    )
    response = client.chat.completions.create(
        model=model.id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    context=context,
                    question=question,
                ),
            },
        ],
        stream=False,
    )

    model.unload()
    manager.stop_web_service()
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the career tree RAG")
    parser.add_argument("question", help="User question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--layer", default=None, help="Filter by layer name")
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--model", default="qwen2.5-0.5b", help="Foundry Local model id")
    args = parser.parse_args()

    hits = retrieve(args.question, top_k=args.top_k, layer=args.layer)
    context = format_context(hits)

    print("=== Retrieved Context ===\n")
    print(context)
    print()

    if args.retrieve_only:
        return

    print("=== Answer ===\n")
    answer = answer_with_foundry(args.question, context, args.model)
    print(answer)


if __name__ == "__main__":
    main()
