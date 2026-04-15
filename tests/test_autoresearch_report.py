from __future__ import annotations

from simlab.autoresearch.config import AutoresearchRunConfig
from simlab.autoresearch.report import _render_report_md
from simlab.autoresearch.report import escape_markdown_table_cell


def test_escape_markdown_table_cell_escapes_pipes() -> None:
    assert escape_markdown_table_cell("a|b") == "a\\|b"


def test_render_report_md_includes_stop_reason() -> None:
    cfg = AutoresearchRunConfig.model_validate(
        {
            "run": {
                "env": "env1",
                "tasks_dir": "./tasks",
                "task_ids": ["t1"],
            },
            "agent": {"model": "gpt-4o-mini", "provider": "openai"},
            "proposer": {"model": "gpt-5.4", "provider": "openai"},
            "verifier": {"model": "gpt-5.4", "provider": "openai"},
        }
    )
    report = _render_report_md(
        cfg=cfg,
        history=[],
        best_iteration=0,
        best_result={"objective_value": 0.5},
        baseline_result={"objective_value": 0.4},
        diff_text="No changes.",
        stop_reason="Reached max_iterations=0.",
    )

    assert "Stop reason" in report
    assert "Reached max_iterations=0." in report
