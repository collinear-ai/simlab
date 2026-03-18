from __future__ import annotations

import importlib
import sys
from types import ModuleType

MODULE_NAMES = [
    "daytona",
    "simlab.cli.main",
    "simlab.cli.tasks",
    "simlab.runtime.daytona_runner",
    "simlab.agents",
]


def pop_simlab_modules() -> dict[str, ModuleType]:
    removed: dict[str, ModuleType] = {}
    for module_name in MODULE_NAMES:
        module = sys.modules.pop(module_name, None)
        if module is not None:
            removed[module_name] = module
    return removed


def restore_simlab_modules(removed: dict[str, ModuleType]) -> None:
    for module_name in MODULE_NAMES:
        sys.modules.pop(module_name, None)
    sys.modules.update(removed)


def test_importing_main_does_not_import_tasks_or_daytona() -> None:
    removed = pop_simlab_modules()
    try:
        importlib.import_module("simlab.cli.main")

        assert "simlab.cli.tasks" not in sys.modules
        assert "simlab.runtime.daytona_runner" not in sys.modules
        assert "daytona" not in sys.modules
    finally:
        restore_simlab_modules(removed)


def test_importing_tasks_does_not_import_daytona_or_agents() -> None:
    removed = pop_simlab_modules()
    try:
        importlib.import_module("simlab.cli.tasks")

        assert "simlab.runtime.daytona_runner" not in sys.modules
        assert "daytona" not in sys.modules
        assert "simlab.agents" not in sys.modules
    finally:
        restore_simlab_modules(removed)
