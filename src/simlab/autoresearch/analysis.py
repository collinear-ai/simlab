"""Build the compact analysis artifact passed to the proposer each iteration."""

from __future__ import annotations


def take_first(value: object, limit: int) -> list[object]:
    """Return the first N items when the input is a list."""
    if not isinstance(value, list):
        return []
    return value[:limit]


def compact_tool_error_taxonomy(value: object, limit: int) -> dict[str, object]:
    """Normalize and compact tool error taxonomy for proposer consumption."""
    if not isinstance(value, dict):
        return {
            "total_errors": 0,
            "categories": [],
            "most_error_prone_tools": [],
            "top_error_messages": [],
        }

    total_errors = value.get("total_errors")
    categories = value.get("categories")
    most_error_prone_tools = value.get("most_error_prone_tools")
    top_error_messages = value.get("top_error_messages")

    return {
        "total_errors": (
            total_errors
            if isinstance(total_errors, (int, float)) and not isinstance(total_errors, bool)
            else 0
        ),
        "categories": take_first(categories, limit),
        "most_error_prone_tools": take_first(most_error_prone_tools, limit),
        "top_error_messages": take_first(top_error_messages, limit),
    }


def build_analysis(
    *,
    objective_type: str,
    objective_target: float | None,
    best_iteration: int,
    best_result: dict[str, object],
    best_eval: dict[str, object],
    latest_iteration: int,
    latest_result: dict[str, object],
    latest_eval: dict[str, object],
    history: list[dict[str, object]],
) -> dict[str, object]:
    """Build a proposer-facing snapshot from best and latest evaluation reports."""
    best_summary = best_eval.get("summary") if isinstance(best_eval, dict) else None
    best_summary = best_summary if isinstance(best_summary, dict) else {}
    latest_summary = latest_eval.get("summary") if isinstance(latest_eval, dict) else None
    latest_summary = latest_summary if isinstance(latest_summary, dict) else {}

    top_failed_checks = best_summary.get("lowest_pass_checks")
    never_passed_checks = best_summary.get("never_passed_checks")
    system_failure_rates = best_summary.get("system_failure_rates")
    tool_error_taxonomy = best_summary.get("tool_error_taxonomy")
    reward_model_dimensions = best_summary.get("reward_model_dimensions")
    reward_model_failed_criteria = best_summary.get("reward_model_failed_criteria")

    return {
        "objective": {"type": objective_type, "target": objective_target},
        "best": {
            "iteration": best_iteration,
            "objective_value": best_result.get("objective_value"),
            "pass_rate": best_result.get("pass_rate"),
            "tool_error_rate": best_result.get("tool_error_rate"),
            "reward_model_score_mean": best_result.get("reward_model_score_mean"),
        },
        "latest": {
            "iteration": latest_iteration,
            "objective_value": latest_result.get("objective_value"),
            "pass_rate": latest_result.get("pass_rate"),
            "tool_error_rate": latest_result.get("tool_error_rate"),
            "reward_model_score_mean": latest_result.get("reward_model_score_mean"),
        },
        "history": history,
        "top_failed_checks": take_first(top_failed_checks, 10),
        "never_passed_checks": take_first(never_passed_checks, 10),
        "system_failure_rates": take_first(system_failure_rates, 10),
        "tool_error_taxonomy": compact_tool_error_taxonomy(tool_error_taxonomy, 10),
        "reward_model_dimensions": take_first(reward_model_dimensions, 10),
        "reward_model_failed_criteria": take_first(reward_model_failed_criteria, 10),
        "latest_summary_overview": (latest_summary.get("overview") or {}) if latest_summary else {},
    }
