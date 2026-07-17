#!/usr/bin/env bash
set -e
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="${STREAMLIT_SERVER_FILE_WATCHER_TYPE:-none}"

echo "Starting UI → http://127.0.0.1:8501"
echo "Ctrl+C to stop."
echo ""

exec streamlit run app.py \
  --server.port 8501 \
  --server.address 127.0.0.1 \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.fileWatcherType none \
  "$@"
