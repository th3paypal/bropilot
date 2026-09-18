#!/usr/bin/env bash
# macOS / Linux launcher: ./run_app.sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "[bropilot] Первый запуск: создаю окружение и ставлю зависимости..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -e .
else
  source .venv/bin/activate
fi
# `bropilot serve` сам поднимает Streamlit и открывает вкладку; Ctrl+C останавливает.
exec bropilot serve "$@"
