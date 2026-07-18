"""Re-fetch the career tree from ibz04.pro and write the JSON + human-readable md.

Chunk building lives here (build_chunks) but is written to disk by ingest.py,
which combines these tree chunks with the extra .md chunks into one file.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict

from rag.config import (
    BOARD_JS_URL,
    CAREER_TREE_FILE,
    CAREER_TREE_MD,
    CONTENT_DIR,
    DATA_DIR,
    SOURCE_URL,
)


def slugify(name: str) -> str:
    s = name.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def fetch_tree() -> dict:
    with urllib.request.urlopen(BOARD_JS_URL) as resp:
        content = resp.read().decode("utf-8")

    start = content.find('id:"root",name:"KNOWLEDGE"')
    const_start = content.rfind("Ft={", 0, start)
    text = content[const_start + 3 :]
    depth = 0
    end = 0
    in_string = False
    escape = False
    quote_char = None

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
        else:
            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

    tree_str = text[:end]
    tree_str = re.sub(r"(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', tree_str)
    return json.loads(tree_str)


def get_paths(node: dict, path: list[str] | None = None, ids: list[str] | None = None) -> list[dict]:
    path = path or []
    ids = ids or []
    current_path = path + [node["name"]]
    current_ids = ids + [node["id"]]
    children = node.get("children", [])
    if not children:
        return [
            {
                "path_id": "-".join(current_ids),
                "layer": current_path[1] if len(current_path) > 1 else None,
                "category": current_path[2] if len(current_path) > 2 else None,
                "role": current_path[-1],
                "full_path_str": " > ".join(current_path),
            }
        ]
    results = []
    for child in children:
        results.extend(get_paths(child, current_path, current_ids))
    return results


def build_chunks(paths: list[dict]) -> list[dict]:
    by_layer: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        by_layer[p["layer"]].append(p)

    chunks: list[dict] = []

    overview = (
        f"Career Tree overview. Source: {SOURCE_URL}\n"
        f"Root: KNOWLEDGE. Total paths: {len(paths)}. Layers: {len(by_layer)}.\n"
        + "\n".join(f"- {layer}: {len(items)} paths" for layer, items in sorted(by_layer.items()))
    )
    chunks.append(
        {
            "id": "overview",
            "text": overview,
            "metadata": {
                "doc_type": "overview",
                "layer": "overview",
                "role": "",
                "category": "",
                "full_path": "KNOWLEDGE",
                "source_url": SOURCE_URL,
            },
        }
    )

    for layer_name, layer_paths in sorted(by_layer.items()):
        slug = slugify(layer_name)
        categories: dict[str, list[str]] = defaultdict(list)
        for p in layer_paths:
            categories[p["category"] or layer_name].append(p["role"])

        lines = [f"Layer: {layer_name}", f"Source: {SOURCE_URL}", ""]
        for cat, roles in sorted(categories.items()):
            lines.append(f"Category: {cat}")
            for role in sorted(roles):
                lines.append(f"- {role}")
            lines.append("")

        chunks.append(
            {
                "id": f"layer-{slug}",
                "text": "\n".join(lines),
                "metadata": {
                    "doc_type": "layer_overview",
                    "layer": layer_name,
                    "role": "",
                    "category": "",
                    "full_path": f"KNOWLEDGE > {layer_name}",
                    "source_url": SOURCE_URL,
                },
            }
        )

    for p in paths:
        layer = p["layer"] or "unknown"
        category = p["category"] or ""
        role = p["role"]
        text = (
            f"Career Path: {role}\n"
            f"Full path: {p['full_path_str']}\n"
            f"Layer: {layer}\n"
            + (f"Category: {category}\n" if category and category != role else "")
            + f"Source: {SOURCE_URL}"
        )
        chunks.append(
            {
                "id": p["path_id"],
                "text": text,
                "metadata": {
                    "doc_type": "career_path",
                    "role": role,
                    "layer": layer,
                    "category": category or layer,
                    "full_path": p["full_path_str"],
                    "source_url": SOURCE_URL,
                },
            }
        )

    return chunks


def write_career_tree_md(paths: list[dict]) -> str:
    by_layer: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        by_layer[p["layer"]].append(p)

    lines = [
        "---",
        f"source_url: {SOURCE_URL}",
        f"total_paths: {len(paths)}",
        "---",
        "",
        "# Career Tree",
        "",
        f"Source: [The Career Tree]({SOURCE_URL})",
        "",
        f"**{len(paths)} career paths** across **{len(by_layer)} layers**, from KNOWLEDGE (root) to specific roles.",
        "",
    ]

    for layer_name, layer_paths in sorted(by_layer.items()):
        lines.append(f"## {layer_name}")
        lines.append("")
        categories: dict[str, list[str]] = defaultdict(list)
        for p in layer_paths:
            categories[p["category"] or layer_name].append(p["role"])

        for cat in sorted(categories.keys()):
            if cat != layer_name:
                lines.append(f"### {cat}")
                lines.append("")
            cat_paths = [p for p in layer_paths if (p["category"] or layer_name) == cat]
            for p in sorted(cat_paths, key=lambda x: x["role"]):
                lines.append(f"- **{p['role']}** — `{p['full_path_str']}`")
            lines.append("")

    return "\n".join(lines)


def regenerate() -> None:
    tree = fetch_tree()
    paths = get_paths(tree)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)

    CAREER_TREE_FILE.write_text(
        json.dumps(tree, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    CAREER_TREE_MD.write_text(write_career_tree_md(paths), encoding="utf-8")

    print("Wrote:")
    print(f"  {CAREER_TREE_MD}")
    print(f"  {CAREER_TREE_FILE}")
    print(f"  ({len(paths)} paths — run ./index.sh to build chunks + index)")


if __name__ == "__main__":
    regenerate()
