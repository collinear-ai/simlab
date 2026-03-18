#!/usr/bin/env bash
set -euo pipefail

apt-get update -qq > /dev/null
apt-get install -y -qq -o Dpkg::Use-Pty=0 poppler-utils > /dev/null
uv pip install --quiet --system xlsx2csv
