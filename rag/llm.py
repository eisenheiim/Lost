"""Foundry Local model loading helpers (kept separate to avoid heavy RAG imports)."""

from __future__ import annotations

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


def make_chat(model, temperature: float = 0.3, max_tokens: int = 450):
    chat = model.get_chat_client()
    chat.settings.temperature = temperature
    chat.settings.max_tokens = max_tokens
    return chat
