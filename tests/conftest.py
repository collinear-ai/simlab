from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_cli_telemetry_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SIMLAB_TELEMETRY_STATE_PATH", str(tmp_path / "telemetry.json"))
