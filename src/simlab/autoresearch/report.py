"""End-of-run markdown and comparison output for autoresearch."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from simlab.autoresearch.config import AutoresearchRunConfig
from simlab.cli.eval import render_markdown_report
from simlab.evaluation import build_report


def escape_markdown_table_cell(value: object) -> str:
    """Escape content for a markdown table cell."""
    text = str(value or "").replace("\n", " ").strip()
    return text.replace("|", "\\|")


def write_end_of_run_reports(
    *,
    run_dir: Path,
    cfg: AutoresearchRunConfig,
    baseline_output: Path,
    best_output: Path,
    history: list[dict[str, object]],
    best_iteration: int,
    best_result: dict[str, object],
    baseline_result: dict[str, object],
    best_prompt: str,
    baseline_prompt: str,
    stop_reason: str = "",
) -> None:
    """Write `report.md` plus an eval-style baseline-vs-best comparison."""
    compare_report = build_report(baseline_output, compare_path=best_output)
    (run_dir / "compare.md").write_text(render_markdown_report(compare_report), encoding="utf-8")
    (run_dir / "compare.json").write_text(json.dumps(compare_report, indent=2), encoding="utf-8")

    diff_text = _prompt_diff(baseline_prompt, best_prompt)
    report_md = _render_report_md(
        cfg=cfg,
        history=history,
        best_iteration=best_iteration,
        best_result=best_result,
        baseline_result=baseline_result,
        diff_text=diff_text,
        stop_reason=stop_reason,
    )
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")


def _render_report_md(
    *,
    cfg: AutoresearchRunConfig,
    history: list[dict[str, object]],
    best_iteration: int,
    best_result: dict[str, object],
    baseline_result: dict[str, object],
    diff_text: str,
    stop_reason: str,
) -> str:
    obj = cfg.objective.type
    lines: list[str] = []
    lines.append("# Autoresearch Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Objective: `{obj}`")
    lines.append(f"- Baseline objective: `{baseline_result.get('objective_value')}`")
    lines.append(f"- Best objective: `{best_result.get('objective_value')}`")
    lines.append(f"- Best iteration: `{best_iteration}`")
    if stop_reason:
        lines.append(f"- Stop reason: `{stop_reason}`")
    lines.append("")
    lines.append("## What SimLab Tried")
    lines.append("")
    lines.append("| iter | accepted | objective | change_type | rationale |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in history:
        it = entry.get("iteration")
        accepted = "yes" if entry.get("accepted") else "no"
        objective = entry.get("objective_value")
        change_type = entry.get("change_type") or ""
        rationale = escape_markdown_table_cell(entry.get("rationale"))
        if len(rationale) > 120:
            rationale = rationale[:117] + "..."
        lines.append(f"| {it} | {accepted} | {objective} | {change_type} | {rationale} |")
    lines.append("")
    lines.append("## Prompt Diff")
    lines.append("")
    lines.append("```diff")
    lines.append(diff_text.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## What To Keep")
    lines.append("")
    lines.append("Keep `best/scenario_prompt.md` if the best result is better than baseline.")
    lines.append("")
    return "\n".join(lines)


def _prompt_diff(baseline_prompt: str, best_prompt: str) -> str:
    baseline_lines = (baseline_prompt or "").splitlines(keepends=True)
    best_lines = (best_prompt or "").splitlines(keepends=True)
    diff = difflib.unified_diff(
        baseline_lines,
        best_lines,
        fromfile="baseline/scenario_prompt.md",
        tofile="best/scenario_prompt.md",
    )
    return "".join(diff) or "No changes."
