#!/usr/bin/env bash
set -euo pipefail

if ! command -v uvx >/dev/null 2>&1; then
  apt-get update
  apt-get install -y curl
  curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
  source "$HOME/.local/bin/env"
fi

uvx \
  --with pytest==8.4.1 \
  --with pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

echo 1 > /logs/verifier/reward.txt
