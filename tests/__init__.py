"""Shared import setup for simlab tests."""

from __future__ import annotations

import sys
from pathlib import Path

CLI_SRC = Path(__file__).resolve().parents[1] / "src"
if str(CLI_SRC) not in sys.path:
    sys.path.insert(0, str(CLI_SRC))
