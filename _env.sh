#!/usr/bin/env bash
# Shared env for project scripts: prefer a venv outside iCloud Desktop.
# shellcheck disable=SC2034

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PREFERRED_VENV="${HEALTHRAG_VENV:-$HOME/.career_tree_rag/venv}"

if [[ -x "$_PREFERRED_VENV/bin/python" ]]; then
  _VENV="$_PREFERRED_VENV"
elif [[ -x "$_ROOT/.venv/bin/python" ]]; then
  _VENV="$_ROOT/.venv"
else
  echo "No virtualenv found. Run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$_VENV/bin/activate"
export PYTHONPATH="$_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$_ROOT"
