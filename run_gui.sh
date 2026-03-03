#!/usr/bin/env bash

set -e

echo
echo "=== Temporal Market Analysis Bot — Run GUI (macOS / Linux) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR=".venv"

if [ -x "$VENV_DIR/bin/python3" ]; then
  VENV_PYTHON="$VENV_DIR/bin/python3"
elif [ -x "$VENV_DIR/bin/python" ]; then
  VENV_PYTHON="$VENV_DIR/bin/python"
else
  echo
  echo "Virtual environment not found at .venv."
  echo "Run setup.sh first:"
  echo "    ./setup.sh"
  exit 1
fi

echo
echo "Starting GUI using virtual environment ..."

"$VENV_PYTHON" main_gui.py "$@"

