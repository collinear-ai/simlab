"""Workspace lifecycle helpers for Harbor-generated runs."""

from __future__ import annotations

import shutil
from datetime import datetime
from datetime import timezone
from pathlib import Path


def preserve_harbor_workspace(workspace_dir: Path, *, task_id: str) -> Path:
    """Copy an ephemeral Harbor workspace into ``output/`` for debugging."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = (Path("output") / "harbor_runs" / f"{task_id}_{ts}").resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace_dir, dest)
    return dest


def create_harbor_workspace_root(*, task_label: str) -> Path:
    """Create a persistent Harbor workspace root under ``output/``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = (Path("output") / "harbor_runs" / f"{task_label}_{ts}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    return root
