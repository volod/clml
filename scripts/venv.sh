#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .venv ]; then
  uv venv --python '>=3.10' .venv
else
  echo ".venv already exists; syncing dependencies"
fi
uv sync --all-groups
uv pip install pip
