#!/usr/bin/env bash
set -e
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"
python -m rag.ask "$@"
