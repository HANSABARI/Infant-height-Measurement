#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${JARAM_HEIGHT_WEB_PYTHON:-D:/Anaconda3/envs/jaram-height-web/python.exe}"
PORT="${PORT:-8000}"

if [[ -z "${H_ALIGN_AI_API_KEY:-}" ]]; then
  echo "Set H_ALIGN_AI_API_KEY before starting the AI server." >&2
  echo "Example: export H_ALIGN_AI_API_KEY=\"<shared-ai-api-key>\"" >&2
  exit 1
fi

if [[ ! -f "$PYTHON_EXE" ]]; then
  echo "Python was not found at: $PYTHON_EXE" >&2
  echo "Set JARAM_HEIGHT_WEB_PYTHON if the Conda env lives elsewhere." >&2
  exit 1
fi

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

cd "$PROJECT_ROOT"
exec "$PYTHON_EXE" -m uvicorn app.main:app --reload --host 127.0.0.1 --port "$PORT"
