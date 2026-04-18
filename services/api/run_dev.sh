#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source ".venv/bin/activate"
python -m pip install -r requirements.txt

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
exec python -m uvicorn app.main:app --reload --host "$HOST" --port "$PORT"

