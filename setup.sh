#!/usr/bin/env bash

set -e

echo
echo "=== Temporal Market Analysis Bot — Setup (macOS / Linux) ==="

# Go to project root (folder where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR=".venv"

echo
echo "Project folder: $(pwd)"

# 1) Create virtual environment if missing
if [ ! -d "$VENV_DIR" ]; then
  echo
  echo "Creating virtual environment in .venv ..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$VENV_DIR"
  else
    python -m venv "$VENV_DIR"
  fi
else
  echo
  echo "Virtual environment already exists at .venv"
fi

# Detect python inside venv
if [ -x "$VENV_DIR/bin/python3" ]; then
  VENV_PYTHON="$VENV_DIR/bin/python3"
elif [ -x "$VENV_DIR/bin/python" ]; then
  VENV_PYTHON="$VENV_DIR/bin/python"
else
  echo
  echo "Existing .venv does not look like a Unix virtualenv (probably created on Windows)."
  echo "Recreating .venv for this system ..."
  rm -rf "$VENV_DIR"
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$VENV_DIR"
  else
    python -m venv "$VENV_DIR"
  fi

  if [ -x "$VENV_DIR/bin/python3" ]; then
    VENV_PYTHON="$VENV_DIR/bin/python3"
  elif [ -x "$VENV_DIR/bin/python" ]; then
    VENV_PYTHON="$VENV_DIR/bin/python"
  else
    echo "ERROR: Could not find python inside $VENV_DIR after recreating it." >&2
    exit 1
  fi
fi

# 2) Install / upgrade packages
if [ ! -f "requirements.txt" ]; then
  echo "ERROR: requirements.txt not found in project root." >&2
  exit 1
fi

echo
echo "Upgrading pip inside the virtual environment ..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo
echo "Installing dependencies from requirements.txt ..."
"$VENV_PYTHON" -m pip install -r "requirements.txt"

# Optional: development / testing dependencies
if [ -f "requirements-dev.txt" ]; then
  echo
  echo "Installing development/test dependencies from requirements-dev.txt ..."
  "$VENV_PYTHON" -m pip install -r "requirements-dev.txt"
fi

# 3) Make main_gui.py executable for ./main_gui.py usage
echo
echo "Marking main_gui.py as executable (chmod +x) ..."
chmod +x main_gui.py || true

echo
echo "=== Setup complete ==="
echo
echo "Next steps for NOOBS:"
echo
echo "1) Open a terminal in this folder:"
echo "     $PROJECT_ROOT"
echo
echo "2) To run the desktop GUI, use the helper script:"
echo "     ./run_gui.sh"
echo
echo "3) To run the automated test suite:"
echo "     python run_all_tests.py"
echo
echo "You can still activate the virtualenv manually if you want to tinker:"
echo "     source .venv/bin/activate"

