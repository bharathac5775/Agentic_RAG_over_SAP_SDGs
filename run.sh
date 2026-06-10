#!/usr/bin/env bash
# One-liner to run the API. Use after `python -m app.ingest` has built ./index/.
set -e
exec uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
