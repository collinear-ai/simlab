"""Objective extraction and comparison rules for autoresearch."""

from __future__ import annotations

import math
from collections.abc import Mapping


def extract_metrics(eval_report: dict[str, object]) -> dict[str, float | None]:
    """Extract the numeric metrics used by v1 objectives and tie-breakers."""
    summary_obj = eval_report.get("summary")
    summary = summary_obj if isinstance(summary_obj, dict) else {}
    overview_obj = summary.get("overview")
    overview = overview_obj if isinstance(overview_obj, dict) else {}
    score_summary_obj = summary.get("score_summary")
    score_summary = score_summary_obj if isinstance(score_summary_obj, dict) else {}

    pass_rate = _safe_float(summary.get("pass_rate"))
    tool_error_rate = _safe_float(overview.get("tool_error_rate"))
    reward_model_score_mean = _safe_float(score_summary.get("mean_reward_model_score"))

    rollouts_obj = eval_report.get("rollouts")
    rollouts = rollouts_obj if isinstance(rollouts_obj, list) else []
    rollout_rewards = [rollout.get("reward") for rollout in rollouts if isinstance(rollout, dict)]
    avg_reward = _mean_numbers(rollout_rewards)
    check_pass_rate = _safe_float(score_summary.get("individual_module_check_pass_rate"))
    if check_pass_rate is None:
        check_pass_rate = _safe_float(score_summary.get("module_all_checks_pass_rate"))

    return {
        "pass_rate": pass_rate,
        "avg_reward": avg_reward,
        "check_pass_rate": check_pass_rate,
        "reward_model_score_mean": reward_model_score_mean,
        "tool_error_rate": tool_error_rate,
    }


def extract_objective_value(eval_report: dict[str, object], objective_type: str) -> float:
    """Return the objective value for the chosen objective type."""
    metrics = extract_metrics(eval_report)
    value = metrics.get(objective_type)
    if value is None:
        raise ValueError(f"Objective metric '{objective_type}' was not available in eval report.")
    return float(value)


def objective_direction(objective_type: str) -> str:
    """Return whether an objective is maximized or minimized."""
    if objective_type == "tool_error_rate":
        return "min"
    return "max"


def is_better_result(
    *,
    candidate: Mapping[str, object],
    best: Mapping[str, object],
    objective_type: str,
) -> bool:
    """Compare a candidate result against best-so-far with deterministic tie-breakers."""
    direction = objective_direction(objective_type)
    cand_obj = _safe_float(candidate.get("objective_value"))
    best_obj = _safe_float(best.get("objective_value"))
    if cand_obj is None or best_obj is None:
        raise ValueError("Missing objective_value for comparison.")

    primary_cmp = _compare(cand_obj, best_obj, direction=direction)
    if primary_cmp > 0:
        return True
    if primary_cmp < 0:
        return False

    cand_rm = _safe_float(candidate.get("reward_model_score_mean"))
    best_rm = _safe_float(best.get("reward_model_score_mean"))
    rm_cmp = _compare_optional(cand_rm, best_rm, direction="max")
    if rm_cmp != 0:
        return rm_cmp > 0

    cand_tool = _safe_float(candidate.get("tool_error_rate"))
    best_tool = _safe_float(best.get("tool_error_rate"))
    tool_cmp = _compare_optional(cand_tool, best_tool, direction="min")
    return tool_cmp > 0


def _compare(a: float, b: float, *, direction: str, eps: float = 1e-9) -> int:
    if math.isclose(a, b, rel_tol=0.0, abs_tol=eps):
        return 0
    if direction == "max":
        return 1 if a > b else -1
    if direction == "min":
        return 1 if a < b else -1
    raise ValueError(f"Unknown direction: {direction}")


def _compare_optional(a: float | None, b: float | None, *, direction: str) -> int:
    if a is None and b is None:
        return 0
    if a is None:
        return -1
    if b is None:
        return 1
    return _compare(a, b, direction=direction)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _mean_numbers(values: list[object]) -> float | None:
    nums: list[float] = []
    for v in values:
        fv = _safe_float(v)
        if fv is not None:
            nums.append(fv)
    if not nums:
        return None
    return sum(nums) / len(nums)
