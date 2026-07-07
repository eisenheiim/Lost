"""Career Tree RAG package.

On import we enable Hugging Face offline mode *if* the embedding model is
already cached. This skips the per-run network check to the HF Hub (and its
"unauthenticated requests" warning) without breaking the first-ever run, which
still needs network access to download the model. Set HEALTHRAG_ONLINE=1 to
force online mode.
"""

from __future__ import annotations

import os
from pathlib import Path


def _maybe_enable_offline() -> None:
    if os.environ.get("HEALTHRAG_ONLINE") == "1":
        return
    if "HF_HUB_OFFLINE" in os.environ:
        return  # respect an explicit user setting

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = hf_home / "hub"
    cached = hub.exists() and any(
        p.name.endswith("all-MiniLM-L6-v2") for p in hub.glob("models--*")
    )
    if cached:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_maybe_enable_offline()
