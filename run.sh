#!/usr/bin/env bash
# Convenience launcher for the Agentic RAG system.
#
# Usage:
#   ./run.sh api    Start the FastAPI server (http://127.0.0.1:8000)
#                   Includes /health, /query, /docs (Swagger UI).
#   ./run.sh ui     Start the Streamlit UI (http://localhost:8502)
#                   Calls the pipeline in-process; no FastAPI needed.
#   ./run.sh both   Start API on :8000 AND UI on :8502 in the same shell.
#                   Hit Ctrl+C once to stop both.
#
# Defaults: port 8000 for API, 8502 for UI (8501 is left free in case you
# already run another Streamlit project there). Override via the env
# vars API_PORT and UI_PORT.
#
# Prerequisite: index/ must exist. Build it once with `python -m app.ingest`.

set -euo pipefail

cd "$(dirname "$0")"

# Activate the venv if it exists (idempotent — no-op if already activated).
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8502}"
TARGET="${1:-api}"

case "$TARGET" in
  api)
    echo "→ FastAPI on http://127.0.0.1:${API_PORT} (docs: /docs)"
    exec uvicorn app.api:app --reload --host 127.0.0.1 --port "${API_PORT}"
    ;;

  ui)
    echo "→ Streamlit on http://localhost:${UI_PORT}"
    exec streamlit run ui/streamlit_app.py \
      --server.port "${UI_PORT}" \
      --browser.gatherUsageStats false
    ;;

  both)
    echo "→ FastAPI on http://127.0.0.1:${API_PORT}"
    echo "→ Streamlit on http://localhost:${UI_PORT}"
    echo "  (Ctrl+C once to stop both.)"
    # Start API in background; clean up on exit.
    uvicorn app.api:app --reload --host 127.0.0.1 --port "${API_PORT}" &
    API_PID=$!
    trap 'kill ${API_PID} 2>/dev/null || true' EXIT INT TERM
    # Run Streamlit in foreground.
    streamlit run ui/streamlit_app.py \
      --server.port "${UI_PORT}" \
      --browser.gatherUsageStats false
    ;;

  -h|--help|help)
    sed -n '2,17p' "$0"
    exit 0
    ;;

  *)
    echo "Unknown target: '${TARGET}'"
    echo "Usage: $0 {api|ui|both|help}"
    exit 1
    ;;
esac
