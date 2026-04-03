# ruff: noqa: ANN401
"""Post-rollout evaluation helpers for ``simlab eval``."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from simlab.runtime.adapters.harbor.verifier import build_harbor_test_verifier_results

RUBRIC_PASS_THRESHOLD = 0.6
CRITERIA_CONTAINER_KEYS = ("results", "criteria_results", "checks")
NATIVE_ROLLOUT_FILENAME = "artifacts.json"
ATIF_ROLLOUT_RELATIVE_PATH = Path("agent") / "trajectory.json"


class EvaluationError(RuntimeError):
    """Raised when an evaluation report cannot be produced."""


def build_report(
    path: Path,
    *,
    compare_path: Path | None = None,
    task_ids: tuple[str, ...] = (),
    model_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a single-rollout, multi-rollout, or comparison evaluation payload."""
    warnings: list[str] = []
    left_dataset = load_rollout_set(path, warnings=warnings)
    left_rollouts = filter_rollouts(
        left_dataset["rollouts"],
        task_ids=task_ids,
        model_names=model_names,
    )
    if not left_rollouts:
        task_display = list(task_ids) or None
        model_display = list(model_names) or None
        raise EvaluationError(
            "No rollouts matched the requested filters under "
            f"{left_dataset['path']} "
            f"(tasks={task_display}, models={model_display})."
        )

    if compare_path is not None:
        right_dataset = load_rollout_set(compare_path, warnings=warnings)
        right_rollouts = filter_rollouts(
            right_dataset["rollouts"],
            task_ids=task_ids,
            model_names=model_names,
        )
        if not right_rollouts:
            task_display = list(task_ids) or None
            model_display = list(model_names) or None
            raise EvaluationError(
                "No rollouts matched the requested filters under "
                f"{right_dataset['path']} "
                f"(tasks={task_display}, models={model_display})."
            )

        left_summary = summarize_rollouts(left_rollouts)
        right_summary = summarize_rollouts(right_rollouts)
        compare_kind = (
            "rollout" if len(left_rollouts) == 1 and len(right_rollouts) == 1 else "dataset"
        )
        report: dict[str, Any] = {
            "mode": "compare",
            "compare_kind": compare_kind,
            "warnings": warnings,
            "left": {
                "path": left_dataset["path"],
                "summary": left_summary,
                "rollouts": [brief_rollout(rollout) for rollout in left_rollouts],
            },
            "right": {
                "path": right_dataset["path"],
                "summary": right_summary,
                "rollouts": [brief_rollout(rollout) for rollout in right_rollouts],
            },
            "comparison": compare_summaries(left_summary, right_summary),
        }
        if compare_kind == "rollout":
            report["left"]["rollout"] = left_rollouts[0]
            report["right"]["rollout"] = right_rollouts[0]
            report["rollout_comparison"] = compare_rollouts(left_rollouts[0], right_rollouts[0])
        return report

    if left_dataset["is_single"] and len(left_rollouts) == 1:
        return {
            "mode": "single_rollout",
            "path": left_dataset["path"],
            "warnings": warnings,
            "rollout": left_rollouts[0],
        }

    return {
        "mode": "multi_rollout",
        "path": left_dataset["path"],
        "warnings": warnings,
        "summary": summarize_rollouts(left_rollouts),
        "rollouts": [brief_rollout(rollout) for rollout in left_rollouts],
    }


def filter_rollouts(
    rollouts: list[dict[str, Any]],
    *,
    task_ids: tuple[str, ...] = (),
    model_names: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Filter rollouts by task and model while keeping identifiers open-ended."""
    requested_tasks = tuple(task_id.strip() for task_id in task_ids if task_id.strip())
    requested_models = tuple(model.strip() for model in model_names if model.strip())

    if not requested_tasks and not requested_models:
        return rollouts

    filtered: list[dict[str, Any]] = []
    for rollout in rollouts:
        if requested_tasks and str(rollout.get("task_id") or "") not in requested_tasks:
            continue
        if requested_models and str(rollout.get("model") or "") not in requested_models:
            continue
        filtered.append(rollout)
    return filtered


def load_rollout_set(path: Path, *, warnings: list[str]) -> dict[str, Any]:
    """Load a single rollout or a rollout collection from disk."""
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise EvaluationError(f"Path does not exist: {resolved_path}")
    if not resolved_path.is_dir():
        raise EvaluationError(f"Expected a directory path: {resolved_path}")

    if is_rollout_dir(resolved_path):
        rollout = load_rollout(resolved_path, warnings=warnings)
        if rollout is None:
            raise EvaluationError(f"No readable rollout artifacts found in {resolved_path}")
        return {
            "path": str(resolved_path),
            "is_single": True,
            "rollouts": [rollout],
        }

    rollout_inputs = collect_rollout_inputs(resolved_path, warnings=warnings)
    if not rollout_inputs:
        raise EvaluationError(
            f"No rollout directories found under {resolved_path}. "
            f"Expected {NATIVE_ROLLOUT_FILENAME} or {ATIF_ROLLOUT_RELATIVE_PATH.as_posix()}."
        )

    rollouts = []
    for rollout_input in rollout_inputs:
        rollout = load_rollout(
            rollout_input["path"],
            warnings=warnings,
            run_name=rollout_input.get("run_name"),
            summary_entry=rollout_input.get("summary_entry"),
        )
        if rollout is not None:
            rollouts.append(rollout)
    if not rollouts:
        raise EvaluationError(f"No valid rollout artifacts could be parsed under {resolved_path}.")

    return {
        "path": str(resolved_path),
        "is_single": False,
        "rollouts": rollouts,
    }


def collect_rollout_inputs(path: Path, *, warnings: list[str]) -> list[dict[str, Any]]:
    """Collect rollout directories under the requested path."""
    if is_run_set_dir(path):
        return collect_run_set_rollouts(path, warnings=warnings)

    rollout_inputs: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        if is_rollout_dir(child):
            rollout_inputs.append({"path": child})
            continue
        if is_run_set_dir(child):
            rollout_inputs.extend(collect_run_set_rollouts(child, warnings=warnings))
    return rollout_inputs


def collect_run_set_rollouts(path: Path, *, warnings: list[str]) -> list[dict[str, Any]]:
    """Collect rollout directories and summary metadata from a run-set directory."""
    summary_by_index: dict[int, dict[str, Any]] = {}
    summary_payload = load_json_file(path / "summary.json", warnings=warnings, required=False)
    if isinstance(summary_payload, dict):
        results = summary_payload.get("results")
        if isinstance(results, list):
            for entry in results:
                if not isinstance(entry, dict):
                    continue
                rollout_idx = safe_int(entry.get("rollout_idx"))
                if rollout_idx is not None:
                    summary_by_index[rollout_idx] = entry

    rollout_inputs: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or not child.name.startswith("rollout_"):
            continue
        if not is_rollout_dir(child):
            warnings.append(
                "Skipping "
                f"{child}: missing {NATIVE_ROLLOUT_FILENAME} or "
                f"{ATIF_ROLLOUT_RELATIVE_PATH.as_posix()}"
            )
            continue

        rollout_input: dict[str, Any] = {
            "path": child,
            "run_name": path.name,
        }
        rollout_idx = rollout_idx_from_dir_name(child.name)
        if rollout_idx is not None and rollout_idx in summary_by_index:
            rollout_input["summary_entry"] = summary_by_index[rollout_idx]
        rollout_inputs.append(rollout_input)
    return rollout_inputs


def load_rollout(
    path: Path,
    *,
    warnings: list[str],
    run_name: str | None = None,
    summary_entry: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Parse one rollout directory into the report shape."""
    artifacts, atif_trajectory = load_rollout_artifacts(path, warnings=warnings)
    if artifacts is None:
        return None

    reward_path = path / "verifier" / "reward.json"
    if not reward_path.is_file():
        alternate_reward_path = path / "reward.json"
        if alternate_reward_path.is_file():
            reward_path = alternate_reward_path

    reward_payload = None
    if reward_path.is_file():
        reward_payload = load_json_file(reward_path, warnings=warnings, required=False)
    if reward_payload is not None and not isinstance(reward_payload, dict):
        reward_payload = None
    reward_payload = reward_payload or extract_reward_payload_from_atif(atif_trajectory)

    raw_verifier_results = reward_payload.get("verifier_results")
    verifier_results = (
        [entry for entry in raw_verifier_results if isinstance(entry, dict)]
        if isinstance(raw_verifier_results, list)
        else build_harbor_test_verifier_results(reward_payload)
    )
    criteria, reward_model = extract_verifier_views(
        reward_payload,
        verifier_results,
    )
    for criterion in criteria:
        criterion["capability"] = bucket_capability(
            criterion["name"],
            detail=criterion.get("detail"),
            source=criterion.get("source"),
        )

    error = coerce_str(artifacts.get("error"))
    rollout_id = path.name
    if run_name and path.parent.name == run_name:
        rollout_id = f"{run_name}/{path.name}"
    metadata: dict[str, Any] = (
        dict(artifacts["metadata"]) if isinstance(artifacts.get("metadata"), dict) else {}
    )

    return {
        "rollout_id": rollout_id,
        "path": str(path),
        "task_id": coerce_str(artifacts.get("task_id")) or path.name,
        "model": coerce_str(artifacts.get("model")),
        "provider": coerce_str(artifacts.get("provider")),
        "created_at": coerce_str(artifacts.get("created_at")),
        "steps_taken": safe_int(artifacts.get("steps_taken")) or 0,
        "max_steps": safe_int(artifacts.get("max_steps")),
        "reward": safe_float(reward_payload.get("reward")),
        "passed": infer_pass_state(
            reward=safe_float(reward_payload.get("reward")),
            verifier_results=verifier_results,
            summary_entry=summary_entry,
            error=error,
        ),
        "error": error,
        "metrics": extract_metrics(metadata, summary_entry, atif_trajectory=atif_trajectory),
        "criteria": criteria,
        "reward_model": reward_model,
        "tool_calls": build_tool_calls(
            artifacts.get("tool_calls"),
            artifacts.get("tool_results"),
            artifacts.get("messages"),
        ),
        "verifier_results": verifier_results,
        "termination_reason": infer_termination_reason(
            error=error,
            steps_taken=safe_int(artifacts.get("steps_taken")) or 0,
            max_steps=safe_int(artifacts.get("max_steps")),
        ),
    }


def extract_verifier_views(
    reward_payload: dict[str, Any],
    verifier_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Split verifier output into programmatic checks and reward-model data."""
    criteria: list[dict[str, Any]] = []
    reward_model: dict[str, Any] | None = None

    for verifier_result in verifier_results:
        source = module_label(coerce_str(verifier_result.get("module")) or "verifier")
        parsed_output = parse_json_blob(coerce_str(verifier_result.get("output")))
        if is_reward_model_result(source, parsed_output):
            reward_model = build_reward_model_result(source=source, parsed_output=parsed_output)
            continue

        checks = extract_checks_from_payload(parsed_output, source=source)
        if checks:
            criteria.extend(checks)
            continue

        criteria.append(
            {
                "name": source,
                "passed": bool(verifier_result.get("success")),
                "detail": coerce_str(verifier_result.get("message"))
                or coerce_str(verifier_result.get("output")),
                "capability": None,
                "source": source,
            }
        )

    if reward_model is None:
        reward_model = build_reward_model_result(
            source="rubric",
            parsed_output=(
                reward_payload.get("rubric_result")
                if isinstance(reward_payload.get("rubric_result"), dict)
                else {}
            ),
        )

    return criteria, reward_model


def is_reward_model_result(source: str, parsed_output: Any) -> bool:
    """Return whether a verifier payload should be treated as reward-model output."""
    lowered_source = source.lower()
    if (
        "universal_verifier" in lowered_source
        or "reward_model" in lowered_source
        or lowered_source == "rubric"
    ):
        return True
    if not isinstance(parsed_output, dict):
        return False
    if any(isinstance(parsed_output.get(key), list) for key in CRITERIA_CONTAINER_KEYS):
        return False
    return any(
        key in parsed_output for key in ("score", "dimension_scores", "verdict", "raw_response")
    )


def build_reward_model_result(
    *,
    source: str,
    parsed_output: Any,
) -> dict[str, Any] | None:
    """Normalize reward-model output into the report shape."""
    if not isinstance(parsed_output, dict):
        return None

    reward_model_payload = dict(parsed_output)
    raw_response_payload = parse_json_blob(coerce_str(reward_model_payload.get("raw_response")))
    if isinstance(raw_response_payload, dict):
        for key, value in raw_response_payload.items():
            reward_model_payload.setdefault(key, value)

    score = safe_float(reward_model_payload.get("score"))
    dimension_scores = []
    raw_dimension_scores = reward_model_payload.get("dimension_scores")
    if isinstance(raw_dimension_scores, list):
        for item in raw_dimension_scores:
            if not isinstance(item, dict):
                continue
            dimension_name = coerce_str(item.get("dimension"))
            dimension_score = safe_float(item.get("score"))
            if not dimension_name or dimension_score is None:
                continue
            dimension_scores.append(
                {
                    "dimension": dimension_name,
                    "score": dimension_score,
                    "reason": coerce_str(item.get("reason")),
                }
            )

    failed_criteria = []
    raw_failed_criteria = reward_model_payload.get("failed_criteria")
    if isinstance(raw_failed_criteria, list):
        for item in raw_failed_criteria:
            criterion = coerce_str(item)
            if criterion:
                failed_criteria.append(criterion)

    notes = []
    raw_notes = reward_model_payload.get("notes")
    if isinstance(raw_notes, list):
        for item in raw_notes:
            note = coerce_str(item)
            if note:
                notes.append(note)

    evidence = []
    raw_evidence = reward_model_payload.get("evidence")
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            detail = coerce_str(item)
            if detail:
                evidence.append(detail)

    if (
        score is None
        and not dimension_scores
        and not failed_criteria
        and not notes
        and not evidence
    ):
        return None

    return {
        "source": source,
        "score": score,
        "confidence": safe_float(reward_model_payload.get("confidence")),
        "verdict": coerce_str(reward_model_payload.get("verdict")),
        "dimension_scores": dimension_scores,
        "failed_criteria": failed_criteria,
        "notes": notes,
        "evidence": evidence,
    }


def analysis_checks_for_rollout(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the checks used in aggregate analysis for one rollout."""
    criteria = rollout.get("criteria")
    return criteria if isinstance(criteria, list) else []


def infer_termination_reason(
    *,
    error: str | None,
    steps_taken: int,
    max_steps: int | None,
) -> str:
    """Infer how a rollout ended from the recorded artifacts."""
    if error:
        return "error"
    if max_steps is not None and steps_taken >= max_steps:
        return "max_steps"
    return "completed"


def brief_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    """Build the abbreviated rollout shape used in list views."""
    return {
        "rollout_id": rollout["rollout_id"],
        "path": rollout["path"],
        "task_id": rollout["task_id"],
        "model": rollout["model"],
        "provider": rollout["provider"],
        "created_at": rollout["created_at"],
        "steps_taken": rollout["steps_taken"],
        "max_steps": rollout["max_steps"],
        "reward": rollout["reward"],
        "passed": rollout["passed"],
        "error": rollout["error"],
        "metrics": rollout["metrics"],
        "criteria_count": len(rollout["criteria"]),
        "tool_call_count": len(rollout["tool_calls"]),
        "reward_model": rollout.get("reward_model"),
        "termination_reason": rollout.get("termination_reason"),
    }


def extract_metrics(
    metadata: dict[str, Any],
    summary_entry: dict[str, Any] | None,
    *,
    atif_trajectory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract cost, token, and timing metrics from rollout metadata."""
    rollout_metrics = metadata.get("rollout_metrics")
    if not isinstance(rollout_metrics, dict):
        rollout_metrics = {}

    token_usage = rollout_metrics.get("token_usage")
    if not isinstance(token_usage, dict):
        token_usage = {}

    timing = rollout_metrics.get("timing")
    if not isinstance(timing, dict):
        timing = {}

    cost = rollout_metrics.get("cost")
    if not isinstance(cost, dict):
        cost = {}

    final_metrics = extract_atif_final_metrics(atif_trajectory)

    prompt_tokens = safe_int(
        token_usage.get("prompt_tokens_total")
        or token_usage.get("prompt_tokens")
        or metadata.get("prompt_tokens_total")
        or final_metrics.get("total_prompt_tokens")
    )
    completion_tokens = safe_int(
        token_usage.get("completion_tokens_total")
        or token_usage.get("completion_tokens")
        or metadata.get("completion_tokens_total")
        or final_metrics.get("total_completion_tokens")
    )
    total_tokens = safe_int(
        token_usage.get("total_tokens_total")
        or metadata.get("total_tokens_total")
        or final_metrics.get("total_tokens")
    )
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    duration_seconds = safe_float(
        timing.get("duration_seconds")
        or timing.get("agent_turn_seconds_total")
        or timing.get("elapsed_seconds_total")
        or (summary_entry.get("duration") if summary_entry else None)
        or extract_atif_duration_seconds(atif_trajectory)
    )
    estimated_cost_usd = safe_float(
        cost.get("estimated_cost_usd")
        or metadata.get("estimated_cost_usd")
        or final_metrics.get("total_cost_usd")
    )

    return {
        "estimated_cost_usd": estimated_cost_usd,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "duration_seconds": duration_seconds,
    }


def extract_checks_from_payload(payload: Any, *, source: str) -> list[dict[str, Any]]:
    """Extract check rows from a verifier payload."""
    if isinstance(payload, list):
        checks: list[dict[str, Any]] = []
        for item in payload:
            checks.extend(extract_checks_from_payload(item, source=source))
        return checks

    if not isinstance(payload, dict):
        return []

    criterion_name = coerce_str(payload.get("criteria"))
    criterion_passed = payload.get("pass") if isinstance(payload.get("pass"), bool) else None
    if criterion_name and criterion_passed is not None:
        return [
            {
                "name": criterion_name,
                "passed": criterion_passed,
                "detail": coerce_str(payload.get("detail"))
                or coerce_str(payload.get("message"))
                or coerce_str(payload.get("reason")),
                "capability": None,
                "source": source,
            }
        ]

    for key in CRITERIA_CONTAINER_KEYS:
        child_payload = payload.get(key)
        if not isinstance(child_payload, list):
            continue
        nested_checks: list[dict[str, Any]] = []
        for item in child_payload:
            nested_checks.extend(extract_checks_from_payload(item, source=source))
        if nested_checks:
            return nested_checks

    dimension_scores = payload.get("dimension_scores")
    if isinstance(dimension_scores, list):
        checks = []
        for item in dimension_scores:
            if not isinstance(item, dict):
                continue
            dimension_name = coerce_str(item.get("dimension"))
            score = safe_float(item.get("score"))
            if not dimension_name or score is None:
                continue
            checks.append(
                {
                    "name": dimension_name,
                    "passed": score >= RUBRIC_PASS_THRESHOLD,
                    "detail": coerce_str(item.get("reason")),
                    "capability": None,
                    "source": source,
                }
            )
        if checks:
            return checks

    failed_criteria = payload.get("failed_criteria")
    if isinstance(failed_criteria, list):
        verdict = coerce_str(payload.get("verdict")) or coerce_str(payload.get("message"))
        return [
            {
                "name": criterion_name,
                "passed": False,
                "detail": verdict,
                "capability": None,
                "source": source,
            }
            for item in failed_criteria
            if (criterion_name := coerce_str(item))
        ]

    return []


def build_tool_calls(
    raw_tool_calls: Any,
    raw_tool_results: Any,
    raw_messages: Any,
) -> list[dict[str, Any]]:
    """Build normalized tool-call rows from rollout artifacts."""
    tool_results = raw_tool_results if isinstance(raw_tool_results, list) else []
    message_summaries = extract_tool_message_summaries(raw_messages)

    tool_calls: list[dict[str, Any]] = []
    tool_call_items = raw_tool_calls if isinstance(raw_tool_calls, list) else []
    for index, call in enumerate(tool_call_items, start=1):
        if not isinstance(call, dict):
            continue

        result = tool_results[index - 1] if index - 1 < len(tool_results) else {}
        summary = None
        is_error = False
        has_recorded_error_flag = False
        if isinstance(result, dict):
            summary = summarize_tool_observation(result.get("observation"))
            if "is_error" in result:
                is_error = bool(result["is_error"])
                has_recorded_error_flag = True
        if summary is None and index - 1 < len(message_summaries):
            summary = message_summaries[index - 1]
        if not has_recorded_error_flag:
            is_error = looks_like_tool_error(summary)

        tool_calls.append(
            {
                "index": index,
                "tool_server": coerce_str(call.get("tool_server")) or "unknown",
                "tool_name": coerce_str(call.get("tool_name")) or "unknown",
                "parameters": (
                    call.get("parameters") if isinstance(call.get("parameters"), dict) else {}
                ),
                "summary": summary,
                "is_error": is_error,
            }
        )
    return tool_calls


def looks_like_tool_error(summary: str | None) -> bool:
    """Return whether a tool summary reads like an error."""
    text = (summary or "").strip().lower()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "error executing tool",
            "timeouterror",
            "timeout",
            "permissionerror",
            "missing payload",
            "failed to create",
            "validationerror",
            "invalid-params",
            "not found",
            "unauthorized",
        )
    )


def extract_tool_message_summaries(raw_messages: Any) -> list[str | None]:
    """Collect summary text from tool messages."""
    summaries: list[str | None] = []
    messages = raw_messages if isinstance(raw_messages, list) else []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, dict):
            summaries.append(
                coerce_str(content.get("summary")) or coerce_str(content.get("tool_name"))
            )
            continue
        summaries.append(coerce_str(content))
    return summaries


def summarize_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the aggregate report summary for a rollout set."""
    reward_values = [rollout["reward"] for rollout in rollouts if rollout["reward"] is not None]
    passed_count = sum(1 for rollout in rollouts if rollout["passed"] is True)
    capability_analysis, cofailure_patterns = summarize_capabilities(rollouts)
    results_by_task = summarize_by_task(rollouts)
    checks_summary = summarize_checks(rollouts)

    return {
        "rollout_count": len(rollouts),
        "task_count": len({rollout["task_id"] for rollout in rollouts}),
        "models": sorted({rollout["model"] for rollout in rollouts if rollout.get("model")}),
        "passed_count": passed_count,
        "pass_rate": passed_count / len(rollouts) if rollouts else 0.0,
        "reward_distribution": {
            "min": min(reward_values) if reward_values else None,
            "p50": statistics.median(reward_values) if reward_values else None,
            "max": max(reward_values) if reward_values else None,
        },
        "aggregate_metrics": {
            "total_cost_usd": sum_numbers(
                rollout["metrics"].get("estimated_cost_usd") for rollout in rollouts
            ),
            "avg_cost_usd": mean_numbers(
                rollout["metrics"].get("estimated_cost_usd") for rollout in rollouts
            ),
            "avg_duration_seconds": mean_numbers(
                rollout["metrics"].get("duration_seconds") for rollout in rollouts
            ),
            "avg_prompt_tokens": mean_numbers(
                rollout["metrics"].get("prompt_tokens") for rollout in rollouts
            ),
            "avg_completion_tokens": mean_numbers(
                rollout["metrics"].get("completion_tokens") for rollout in rollouts
            ),
            "avg_steps": mean_numbers(rollout.get("steps_taken") for rollout in rollouts),
        },
        "overview": summarize_overview(rollouts),
        "score_summary": summarize_score_summary(rollouts),
        "reward_model_gap": summarize_programmatic_reward_model_gap(rollouts),
        "system_failure_rates": summarize_system_failure_rates(rollouts),
        "never_passed_checks": checks_summary["never_passed"],
        "lowest_pass_checks": checks_summary["lowest_pass"],
        "reward_model_dimensions": summarize_reward_model_dimensions(rollouts),
        "reward_model_failed_criteria": summarize_reward_model_failed_criteria(rollouts),
        "tool_error_taxonomy": summarize_tool_errors(rollouts),
        "call_sequence_patterns": summarize_call_sequences(rollouts),
        "results_by_task": results_by_task,
        "task_difficulty": summarize_task_difficulty(results_by_task),
        "model_comparison": summarize_by_model(rollouts),
        "capability_analysis": capability_analysis,
        "cofailure_patterns": cofailure_patterns,
    }


def summarize_overview(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the high-level overview metrics for a rollout set."""
    total_tool_calls = sum(len(rollout["tool_calls"]) for rollout in rollouts)
    total_tool_errors = sum(
        1 for rollout in rollouts for tool_call in rollout["tool_calls"] if tool_call["is_error"]
    )
    rollouts_with_errors = sum(
        1
        for rollout in rollouts
        if any(tool_call["is_error"] for tool_call in rollout["tool_calls"])
    )

    termination_counts = Counter(
        rollout.get("termination_reason") or "completed" for rollout in rollouts
    )
    step_values = [
        steps_taken
        for rollout in rollouts
        if (steps_taken := safe_int(rollout.get("steps_taken"))) is not None
    ]
    termination_reasons = [
        {
            "reason": reason,
            "count": count,
            "rate": count / len(rollouts) if rollouts else 0.0,
        }
        for reason, count in termination_counts.most_common()
    ]

    return {
        "total_rollouts": len(rollouts),
        "unique_tasks": len({rollout["task_id"] for rollout in rollouts}),
        "total_tool_calls": total_tool_calls,
        "total_tool_errors": total_tool_errors,
        "tool_error_rate": total_tool_errors / total_tool_calls if total_tool_calls else 0.0,
        "rollouts_with_errors": rollouts_with_errors,
        "rollouts_with_errors_rate": rollouts_with_errors / len(rollouts) if rollouts else 0.0,
        "total_steps": sum(step_values),
        "avg_steps_per_rollout": mean_numbers(rollout.get("steps_taken") for rollout in rollouts),
        "median_steps_per_rollout": statistics.median(step_values) if step_values else None,
        "rollouts_with_steps": sum(1 for steps_taken in step_values if steps_taken > 0),
        "rollouts_with_steps_rate": (
            sum(1 for steps_taken in step_values if steps_taken > 0) / len(rollouts)
            if rollouts
            else 0.0
        ),
        "avg_duration_seconds": mean_numbers(
            rollout["metrics"].get("duration_seconds") for rollout in rollouts
        ),
        "termination_reasons": termination_reasons,
    }


def summarize_score_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute score-oriented summary metrics for a rollout set."""
    composite_scores = [
        score for rollout in rollouts if (score := rollout_composite_score(rollout)) is not None
    ]
    reward_model_scores = [
        score for rollout in rollouts if (score := rollout_reward_model_score(rollout)) is not None
    ]
    reward_values = [
        reward for rollout in rollouts if (reward := rollout.get("reward")) is not None
    ]

    module_all_checks_pass_count = sum(
        1 for rollout in rollouts if rollout_module_pass_rate(rollout) == 1.0
    )
    total_module_checks = sum(len(analysis_checks_for_rollout(rollout)) for rollout in rollouts)
    passed_module_checks = sum(
        sum(1 for check in analysis_checks_for_rollout(rollout) if check.get("passed"))
        for rollout in rollouts
    )
    reward_model_above_point_eight_count = sum(1 for score in reward_model_scores if score > 0.8)
    reward_earned_count = sum(1 for reward in reward_values if reward == 1.0)

    reward_model_variance = None
    reward_model_stdev = None
    if len(reward_model_scores) >= 2:
        reward_model_variance = statistics.pvariance(reward_model_scores)
        reward_model_stdev = statistics.pstdev(reward_model_scores)

    return {
        "module_all_checks_pass_rate": (
            module_all_checks_pass_count / len(rollouts) if rollouts else 0.0
        ),
        "individual_module_check_pass_rate": (
            passed_module_checks / total_module_checks if total_module_checks else None
        ),
        "mean_composite_score": mean_numbers(composite_scores),
        "mean_reward_model_score": mean_numbers(reward_model_scores),
        "reward_model_score_variance": reward_model_variance,
        "reward_model_score_stdev": reward_model_stdev,
        "reward_model_rollouts_above_point_eight_rate": (
            reward_model_above_point_eight_count / len(reward_model_scores)
            if reward_model_scores
            else None
        ),
        "reward_earned_rate": reward_earned_count / len(reward_values) if reward_values else None,
        "reward_model_rollout_count": len(reward_model_scores),
    }


def summarize_programmatic_reward_model_gap(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure how programmatic and reward-model signals diverge."""
    module_fail_reward_model_high = 0
    module_pass_reward_model_low = 0
    bin_rows: list[dict[str, Any]] = []
    bin_specs = [
        ("100%", 1.0, 1.0000001),
        ("75-99%", 0.75, 1.0),
        ("50-74%", 0.50, 0.75),
        ("25-49%", 0.25, 0.50),
        ("1-24%", 0.01, 0.25),
        ("0%", 0.0, 0.01),
    ]

    for rollout in rollouts:
        module_score = rollout_module_pass_rate(rollout)
        reward_model_score = rollout_reward_model_score(rollout)
        if module_score is None or reward_model_score is None:
            continue
        if module_score < 1.0 and reward_model_score > 0.8:
            module_fail_reward_model_high += 1
        if module_score == 1.0 and reward_model_score < 0.5:
            module_pass_reward_model_low += 1

    for label, lower_bound, upper_bound in bin_specs:
        matching_scores = [
            rollout_reward_model_score(rollout)
            for rollout in rollouts
            if (
                (module_score := rollout_module_pass_rate(rollout)) is not None
                and lower_bound <= module_score < upper_bound
                and rollout_reward_model_score(rollout) is not None
            )
        ]
        matching_scores = [score for score in matching_scores if score is not None]
        bin_rows.append(
            {
                "module_pass_rate": label,
                "rollout_count": len(matching_scores),
                "mean_reward_model_score": mean_numbers(matching_scores),
            }
        )

    return {
        "module_fails_reward_model_above_point_eight_count": module_fail_reward_model_high,
        "module_fails_reward_model_above_point_eight_rate": (
            module_fail_reward_model_high / len(rollouts) if rollouts else 0.0
        ),
        "module_passes_reward_model_below_point_five_count": module_pass_reward_model_low,
        "module_passes_reward_model_below_point_five_rate": (
            module_pass_reward_model_low / len(rollouts) if rollouts else 0.0
        ),
        "module_pass_rate_bins": bin_rows,
    }


def summarize_system_failure_rates(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute pass and fail rates by system bucket."""
    counts_by_system: dict[str, Counter[str]] = defaultdict(Counter)
    for rollout in rollouts:
        for check in analysis_checks_for_rollout(rollout):
            system = bucket_system(
                check.get("name") or "",
                detail=check.get("detail"),
                source=check.get("source"),
            )
            counts_by_system[system]["passed"] += 1 if check.get("passed") else 0
            counts_by_system[system]["failed"] += 0 if check.get("passed") else 1

    rows: list[dict[str, Any]] = []
    for system, counts in counts_by_system.items():
        passed_count = counts["passed"]
        failed_count = counts["failed"]
        total_count = passed_count + failed_count
        rows.append(
            {
                "system": system,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "fail_rate": failed_count / total_count if total_count else 0.0,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["fail_rate"]),
            -int(row["failed_count"]),
            str(row["system"]),
        )
    )
    return rows


def summarize_checks(rollouts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Compute aggregate pass rates for each recorded check."""
    counts_by_check: dict[str, Counter[str]] = defaultdict(Counter)
    for rollout in rollouts:
        for check in analysis_checks_for_rollout(rollout):
            counts_by_check[check["name"]]["attempts"] += 1
            if check.get("passed"):
                counts_by_check[check["name"]]["passed"] += 1

    rows: list[dict[str, Any]] = []
    for name, counts in counts_by_check.items():
        attempts = counts["attempts"]
        passed_count = counts["passed"]
        rows.append(
            {
                "name": name,
                "attempts": attempts,
                "passed_count": passed_count,
                "pass_rate": passed_count / attempts if attempts else 0.0,
            }
        )

    never_passed = [row for row in rows if row["passed_count"] == 0]
    never_passed.sort(key=lambda row: (-int(row["attempts"]), str(row["name"])))

    lowest_pass = [row for row in rows if int(row["attempts"]) > 0]
    lowest_pass.sort(
        key=lambda row: (
            float(row["pass_rate"]),
            -int(row["attempts"]),
            str(row["name"]),
        )
    )

    return {
        "never_passed": never_passed,
        "lowest_pass": lowest_pass,
    }


def summarize_reward_model_dimensions(
    rollouts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate reward-model dimension scores across rollouts."""
    scores_by_dimension: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        reward_model = rollout.get("reward_model")
        if not isinstance(reward_model, dict):
            continue
        dimension_scores = reward_model.get("dimension_scores")
        if not isinstance(dimension_scores, list):
            continue
        for item in dimension_scores:
            if not isinstance(item, dict):
                continue
            dimension_name = coerce_str(item.get("dimension"))
            score = safe_float(item.get("score"))
            if dimension_name and score is not None:
                scores_by_dimension[dimension_name].append(score)

    rows = [
        {
            "dimension": dimension,
            "mean_score": statistics.fmean(scores),
            "sample_count": len(scores),
        }
        for dimension, scores in scores_by_dimension.items()
        if scores
    ]
    rows.sort(key=lambda row: (row["mean_score"], row["dimension"]))
    return {
        "lowest": rows[:10],
        "highest": list(reversed(rows[-10:])),
    }


def summarize_reward_model_failed_criteria(
    rollouts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Count the most common failed reward-model criteria."""
    counter: Counter[str] = Counter()
    for rollout in rollouts:
        reward_model = rollout.get("reward_model")
        if not isinstance(reward_model, dict):
            continue
        failed_criteria = reward_model.get("failed_criteria")
        if not isinstance(failed_criteria, list):
            continue
        for criterion in failed_criteria:
            name = coerce_str(criterion)
            if name:
                counter[name] += 1
    return [{"criterion": name, "count": count} for name, count in counter.most_common()]


def summarize_tool_errors(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate tool-error counts, categories, and messages."""
    error_calls = [
        tool_call
        for rollout in rollouts
        for tool_call in rollout["tool_calls"]
        if tool_call["is_error"]
    ]
    category_counts: Counter[str] = Counter()
    tool_counts: dict[str, Counter[str]] = defaultdict(Counter)
    message_counts: Counter[str] = Counter()
    call_counts: Counter[str] = Counter()

    for rollout in rollouts:
        for tool_call in rollout["tool_calls"]:
            tool_name = tool_label(tool_call)
            call_counts[tool_name] += 1
            if not tool_call["is_error"]:
                continue
            category = tool_error_category(tool_call.get("summary"))
            category_counts[category] += 1
            tool_counts[tool_name]["errors"] += 1
            message = tool_error_message(tool_call.get("summary"))
            if message:
                message_counts[message] += 1

    categories = [
        {
            "category": category,
            "count": count,
            "rate": count / len(error_calls) if error_calls else 0.0,
            "description": tool_error_category_description(category),
        }
        for category, count in category_counts.most_common()
    ]
    most_error_prone_tools = [
        {
            "tool": tool_label,
            "error_rate": counts["errors"] / call_counts[tool_label]
            if call_counts[tool_label]
            else 0.0,
            "error_count": counts["errors"],
            "call_count": call_counts[tool_label],
        }
        for tool_label, counts in tool_counts.items()
        if call_counts[tool_label]
    ]
    most_error_prone_tools.sort(
        key=lambda row: (
            -(safe_float(row["error_rate"]) or 0.0),
            -(safe_int(row["error_count"]) or 0),
            str(row["tool"]),
        )
    )
    zero_error_tools = [
        {"tool": tool_label, "call_count": call_count}
        for tool_label, call_count in call_counts.items()
        if tool_counts[tool_label]["errors"] == 0
    ]
    zero_error_tools.sort(key=lambda row: (-(safe_int(row["call_count"]) or 0), str(row["tool"])))
    top_error_messages = [
        {"message": message, "count": count} for message, count in message_counts.most_common()
    ]

    return {
        "total_errors": len(error_calls),
        "categories": categories,
        "most_error_prone_tools": most_error_prone_tools,
        "zero_error_tools": zero_error_tools,
        "top_error_messages": top_error_messages,
    }


def summarize_call_sequences(rollouts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate common early tool-call sequences."""
    first_calls: Counter[str] = Counter()
    first_five_patterns: Counter[str] = Counter()

    for rollout in rollouts:
        tool_labels = [tool_label(tool_call) for tool_call in rollout["tool_calls"]]
        if not tool_labels:
            continue
        first_calls[tool_labels[0]] += 1
        first_five_patterns[" -> ".join(tool_labels[:5])] += 1

    return {
        "first_calls": [
            {
                "tool": tool,
                "count": count,
                "rate": count / len(rollouts) if rollouts else 0.0,
            }
            for tool, count in first_calls.most_common()
        ],
        "first_five_patterns": [
            {
                "pattern": pattern,
                "count": count,
                "rate": count / len(rollouts) if rollouts else 0.0,
            }
            for pattern, count in first_five_patterns.most_common()
        ],
    }


def summarize_by_task(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate results by task identifier."""
    rollouts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        rollouts_by_task[rollout["task_id"]].append(rollout)

    rows: list[dict[str, Any]] = []
    for task_id, task_rollouts in rollouts_by_task.items():
        passed_count = sum(1 for rollout in task_rollouts if rollout["passed"] is True)
        rollout_count = len(task_rollouts)
        rows.append(
            {
                "task_id": task_id,
                "rollout_count": rollout_count,
                "passed_count": passed_count,
                "pass_rate": passed_count / rollout_count if rollout_count else 0.0,
                "avg_cost_usd": mean_numbers(
                    rollout["metrics"].get("estimated_cost_usd") for rollout in task_rollouts
                ),
                "avg_duration_seconds": mean_numbers(
                    rollout["metrics"].get("duration_seconds") for rollout in task_rollouts
                ),
                "mean_reward": mean_numbers(rollout.get("reward") for rollout in task_rollouts),
                "mean_composite_score": mean_numbers(
                    rollout_composite_score(rollout) for rollout in task_rollouts
                ),
                "mean_reward_model_score": mean_numbers(
                    rollout_reward_model_score(rollout) for rollout in task_rollouts
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["mean_reward_model_score"] if row["mean_reward_model_score"] is not None else 1.0,
            row["pass_rate"],
            row["task_id"],
        )
    )
    return rows


def summarize_task_difficulty(
    results_by_task: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Order tasks by the reported scores."""
    scorable_rows = [
        row
        for row in results_by_task
        if row.get("mean_reward_model_score") is not None
        or row.get("mean_composite_score") is not None
    ]
    ordered = sorted(
        scorable_rows,
        key=lambda row: (
            row["mean_reward_model_score"]
            if row.get("mean_reward_model_score") is not None
            else row.get("mean_composite_score") or 1.0,
            row["task_id"],
        ),
    )
    return {
        "hardest": ordered[:10],
        "easiest": list(reversed(ordered[-10:])),
    }


def summarize_by_model(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate results by model name."""
    rollouts_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        model = rollout.get("model") or "unknown"
        rollouts_by_model[model].append(rollout)

    rows: list[dict[str, Any]] = []
    for model, model_rollouts in rollouts_by_model.items():
        rollout_count = len(model_rollouts)
        passed_count = sum(1 for rollout in model_rollouts if rollout["passed"] is True)
        rows.append(
            {
                "model": model,
                "rollout_count": rollout_count,
                "passed_count": passed_count,
                "pass_rate": passed_count / rollout_count if rollout_count else 0.0,
                "mean_reward": mean_numbers(rollout.get("reward") for rollout in model_rollouts),
                "mean_composite_score": mean_numbers(
                    rollout_composite_score(rollout) for rollout in model_rollouts
                ),
                "mean_reward_model_score": mean_numbers(
                    rollout_reward_model_score(rollout) for rollout in model_rollouts
                ),
                "avg_cost_usd": mean_numbers(
                    rollout["metrics"].get("estimated_cost_usd") for rollout in model_rollouts
                ),
                "avg_duration_seconds": mean_numbers(
                    rollout["metrics"].get("duration_seconds") for rollout in model_rollouts
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["mean_reward_model_score"] if row["mean_reward_model_score"] is not None else 1.0,
            row["model"],
        )
    )
    return rows


def compare_summaries(
    left_summary: dict[str, Any], right_summary: dict[str, Any]
) -> dict[str, Any]:
    """Build the top-level comparison rows for two rollout summaries."""
    return {
        "overview": [
            comparison_row(
                "rollouts", left_summary["rollout_count"], right_summary["rollout_count"]
            ),
            comparison_row("tasks", left_summary["task_count"], right_summary["task_count"]),
            comparison_row(
                "pass_rate",
                left_summary.get("pass_rate"),
                right_summary.get("pass_rate"),
            ),
            comparison_row(
                "mean_composite_score",
                left_summary.get("score_summary", {}).get("mean_composite_score"),
                right_summary.get("score_summary", {}).get("mean_composite_score"),
            ),
            comparison_row(
                "mean_reward_model_score",
                left_summary.get("score_summary", {}).get("mean_reward_model_score"),
                right_summary.get("score_summary", {}).get("mean_reward_model_score"),
            ),
            comparison_row(
                "reward_model_score_variance",
                left_summary.get("score_summary", {}).get("reward_model_score_variance"),
                right_summary.get("score_summary", {}).get("reward_model_score_variance"),
            ),
            comparison_row(
                "reward_rate",
                left_summary.get("score_summary", {}).get("reward_earned_rate"),
                right_summary.get("score_summary", {}).get("reward_earned_rate"),
            ),
            comparison_row(
                "total_tool_errors",
                left_summary.get("overview", {}).get("total_tool_errors"),
                right_summary.get("overview", {}).get("total_tool_errors"),
            ),
            comparison_row(
                "avg_steps_per_rollout",
                left_summary.get("overview", {}).get("avg_steps_per_rollout"),
                right_summary.get("overview", {}).get("avg_steps_per_rollout"),
            ),
            comparison_row(
                "avg_duration_seconds",
                left_summary.get("overview", {}).get("avg_duration_seconds"),
                right_summary.get("overview", {}).get("avg_duration_seconds"),
            ),
            comparison_row(
                "avg_cost_usd",
                left_summary.get("aggregate_metrics", {}).get("avg_cost_usd"),
                right_summary.get("aggregate_metrics", {}).get("avg_cost_usd"),
            ),
        ],
        "results_by_task": compare_named_rows(
            left_summary.get("results_by_task", []),
            right_summary.get("results_by_task", []),
            key="task_id",
            metrics=(
                "pass_rate",
                "mean_reward_model_score",
                "mean_composite_score",
                "rollout_count",
            ),
        ),
        "model_comparison": compare_named_rows(
            left_summary.get("model_comparison", []),
            right_summary.get("model_comparison", []),
            key="model",
            metrics=(
                "pass_rate",
                "mean_reward_model_score",
                "mean_composite_score",
                "rollout_count",
            ),
        ),
        "system_failure_rates": compare_named_rows(
            left_summary.get("system_failure_rates", []),
            right_summary.get("system_failure_rates", []),
            key="system",
            metrics=("fail_rate", "failed_count", "passed_count"),
        ),
        "capability_analysis": compare_named_rows(
            left_summary.get("capability_analysis", []),
            right_summary.get("capability_analysis", []),
            key="capability",
            metrics=("pass_rate", "total_count", "failed_rollout_count"),
        ),
    }


def compare_rollouts(left_rollout: dict[str, Any], right_rollout: dict[str, Any]) -> dict[str, Any]:
    """Build the detailed comparison payload for two single rollouts."""
    return {
        "overview": [
            comparison_row(
                "passed",
                left_rollout.get("passed")
                if isinstance(left_rollout.get("passed"), bool)
                else None,
                right_rollout.get("passed")
                if isinstance(right_rollout.get("passed"), bool)
                else None,
            ),
            comparison_row("reward", left_rollout.get("reward"), right_rollout.get("reward")),
            comparison_row(
                "composite_score",
                rollout_composite_score(left_rollout),
                rollout_composite_score(right_rollout),
            ),
            comparison_row(
                "reward_model_score",
                rollout_reward_model_score(left_rollout),
                rollout_reward_model_score(right_rollout),
            ),
            comparison_row(
                "steps_taken",
                left_rollout.get("steps_taken"),
                right_rollout.get("steps_taken"),
            ),
            comparison_row(
                "tool_calls",
                len(left_rollout.get("tool_calls", [])),
                len(right_rollout.get("tool_calls", [])),
            ),
            comparison_row(
                "tool_errors",
                tool_error_count(left_rollout),
                tool_error_count(right_rollout),
            ),
            comparison_row(
                "duration_seconds",
                left_rollout.get("metrics", {}).get("duration_seconds"),
                right_rollout.get("metrics", {}).get("duration_seconds"),
            ),
            comparison_row(
                "cost_usd",
                left_rollout.get("metrics", {}).get("estimated_cost_usd"),
                right_rollout.get("metrics", {}).get("estimated_cost_usd"),
            ),
        ],
        "criteria": compare_criteria(left_rollout, right_rollout),
        "reward_model_dimensions": compare_reward_model_dimensions(left_rollout, right_rollout),
        "tool_sequence": compare_tool_sequence(left_rollout, right_rollout),
        "left_errors": [
            tool_call_brief(tool_call)
            for tool_call in left_rollout.get("tool_calls", [])
            if tool_call.get("is_error")
        ],
        "right_errors": [
            tool_call_brief(tool_call)
            for tool_call in right_rollout.get("tool_calls", [])
            if tool_call.get("is_error")
        ],
    }


def compare_named_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    key: str,
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Compare row-shaped aggregates keyed by a shared name field."""
    left_by_key = {row[key]: row for row in left_rows if row.get(key) is not None}
    right_by_key = {row[key]: row for row in right_rows if row.get(key) is not None}
    rows = []
    for name in sorted(set(left_by_key) | set(right_by_key)):
        row = {key: name}
        left_row = left_by_key.get(name, {})
        right_row = right_by_key.get(name, {})
        for metric in metrics:
            left_value = left_row.get(metric)
            right_value = right_row.get(metric)
            row[f"left_{metric}"] = left_value
            row[f"right_{metric}"] = right_value
            row[f"delta_{metric}"] = delta_number(left_value, right_value)
        rows.append(row)
    return rows


def comparison_row(metric: str, left_value: Any, right_value: Any) -> dict[str, Any]:
    """Build one comparison row from left and right values."""
    return {
        "metric": metric,
        "left_value": left_value,
        "right_value": right_value,
        "delta": delta_number(left_value, right_value),
    }


def compare_criteria(
    left_rollout: dict[str, Any],
    right_rollout: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare programmatic verifier criteria between two rollouts."""
    left_by_name = {
        criterion["name"]: criterion
        for criterion in left_rollout.get("criteria", [])
        if isinstance(criterion, dict) and criterion.get("name") is not None
    }
    right_by_name = {
        criterion["name"]: criterion
        for criterion in right_rollout.get("criteria", [])
        if isinstance(criterion, dict) and criterion.get("name") is not None
    }
    rows = []
    for name in sorted(set(left_by_name) | set(right_by_name)):
        left_criterion = left_by_name.get(name, {})
        right_criterion = right_by_name.get(name, {})
        rows.append(
            {
                "criterion": name,
                "left_passed": (
                    left_criterion.get("passed")
                    if isinstance(left_criterion.get("passed"), bool)
                    else None
                ),
                "right_passed": (
                    right_criterion.get("passed")
                    if isinstance(right_criterion.get("passed"), bool)
                    else None
                ),
                "left_detail": coerce_str(left_criterion.get("detail")),
                "right_detail": coerce_str(right_criterion.get("detail")),
            }
        )
    return rows


def compare_reward_model_dimensions(
    left_rollout: dict[str, Any],
    right_rollout: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare reward-model dimensions between two rollouts."""
    left_reward_model = left_rollout.get("reward_model")
    right_reward_model = right_rollout.get("reward_model")
    left_dimension_scores: list[Any] = []
    if isinstance(left_reward_model, dict):
        raw_left_dimension_scores = left_reward_model.get("dimension_scores")
        if isinstance(raw_left_dimension_scores, list):
            left_dimension_scores = raw_left_dimension_scores
    right_dimension_scores: list[Any] = []
    if isinstance(right_reward_model, dict):
        raw_right_dimension_scores = right_reward_model.get("dimension_scores")
        if isinstance(raw_right_dimension_scores, list):
            right_dimension_scores = raw_right_dimension_scores
    left_by_name = {
        item["dimension"]: item
        for item in left_dimension_scores
        if isinstance(item, dict) and item.get("dimension") is not None
    }
    right_by_name = {
        item["dimension"]: item
        for item in right_dimension_scores
        if isinstance(item, dict) and item.get("dimension") is not None
    }
    rows = []
    for name in sorted(set(left_by_name) | set(right_by_name)):
        left_item = left_by_name.get(name, {})
        right_item = right_by_name.get(name, {})
        left_score = safe_float(left_item.get("score"))
        right_score = safe_float(right_item.get("score"))
        rows.append(
            {
                "dimension": name,
                "left_score": left_score,
                "right_score": right_score,
                "delta_score": delta_number(left_score, right_score),
                "left_reason": coerce_str(left_item.get("reason")),
                "right_reason": coerce_str(right_item.get("reason")),
            }
        )
    return rows


def compare_tool_sequence(
    left_rollout: dict[str, Any],
    right_rollout: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare the tool-call sequence between two rollouts."""
    left_tool_calls = left_rollout.get("tool_calls", [])
    right_tool_calls = right_rollout.get("tool_calls", [])
    rows = []
    for index in range(max(len(left_tool_calls), len(right_tool_calls))):
        left_call = left_tool_calls[index] if index < len(left_tool_calls) else {}
        right_call = right_tool_calls[index] if index < len(right_tool_calls) else {}
        left_tool = tool_label(left_call) if left_call else None
        right_tool = tool_label(right_call) if right_call else None
        rows.append(
            {
                "step": index + 1,
                "left_tool": left_tool,
                "right_tool": right_tool,
                "same_tool": bool(left_tool and right_tool and left_tool == right_tool),
                "left_error": (
                    left_call.get("is_error")
                    if isinstance(left_call.get("is_error"), bool)
                    else None
                ),
                "right_error": (
                    right_call.get("is_error")
                    if isinstance(right_call.get("is_error"), bool)
                    else None
                ),
                "left_parameters": (
                    left_call.get("parameters")
                    if isinstance(left_call.get("parameters"), dict)
                    else {}
                ),
                "right_parameters": (
                    right_call.get("parameters")
                    if isinstance(right_call.get("parameters"), dict)
                    else {}
                ),
                "left_summary": coerce_str(left_call.get("summary")),
                "right_summary": coerce_str(right_call.get("summary")),
            }
        )
    return rows


def tool_error_count(rollout: dict[str, Any]) -> int:
    """Count tool calls marked as errors in one rollout."""
    return sum(
        1
        for tool_call in rollout.get("tool_calls", [])
        if isinstance(tool_call, dict) and tool_call.get("is_error") is True
    )


def tool_call_brief(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Build the short tool-call shape used in comparisons."""
    return {
        "index": tool_call.get("index"),
        "tool": tool_label(tool_call),
        "summary": coerce_str(tool_call.get("summary")),
        "parameters": (
            tool_call.get("parameters") if isinstance(tool_call.get("parameters"), dict) else {}
        ),
    }


def summarize_capabilities(
    rollouts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate pass rates and cofailures by capability bucket."""
    capability_counts: dict[str, Counter[str]] = defaultdict(Counter)
    capability_criteria_names: dict[str, set[str]] = defaultdict(set)
    capability_failed_rollouts: dict[str, set[str]] = defaultdict(set)
    capability_attempt_rollouts: dict[str, set[str]] = defaultdict(set)
    criterion_failure_sets: dict[str, set[str]] = defaultdict(set)
    criterion_capabilities: dict[str, str] = {}

    for rollout in rollouts:
        for tool_call in rollout["tool_calls"]:
            capability = bucket_capability(
                tool_call["tool_name"],
                detail=tool_call.get("summary"),
                source=tool_call.get("tool_server"),
            )
            capability_attempt_rollouts[capability].add(rollout["rollout_id"])

        for criterion in analysis_checks_for_rollout(rollout):
            capability = criterion.get("capability") or bucket_capability(
                criterion["name"],
                detail=criterion.get("detail"),
                source=criterion.get("source"),
            )
            capability_counts[capability]["total"] += 1
            capability_criteria_names[capability].add(criterion["name"])
            if criterion["passed"]:
                capability_counts[capability]["passed"] += 1
                continue

            capability_failed_rollouts[capability].add(rollout["rollout_id"])
            criterion_failure_sets[criterion["name"]].add(rollout["rollout_id"])
            criterion_capabilities[criterion["name"]] = capability

    patterns = detect_cofailure_patterns(
        criterion_failure_sets,
        criterion_capabilities,
    )
    patterns_by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pattern in patterns:
        patterns_by_capability[pattern["capability"]].append(pattern)

    rows: list[dict[str, Any]] = []
    for capability, counts in capability_counts.items():
        failed_rollouts = capability_failed_rollouts.get(capability, set())
        attempted_failures = sum(
            1
            for rollout_id in failed_rollouts
            if rollout_id in capability_attempt_rollouts[capability]
        )

        attempt_signal = None
        if failed_rollouts:
            attempt_rate = attempted_failures / len(failed_rollouts)
            if attempt_rate <= 0.25:
                attempt_signal = "rarely attempted"
            elif attempt_rate >= 0.75:
                attempt_signal = "attempted but failed"
            else:
                attempt_signal = "mixed"

        total_count = counts["total"]
        rows.append(
            {
                "capability": capability,
                "pass_count": counts["passed"],
                "total_count": total_count,
                "pass_rate": counts["passed"] / total_count if total_count else 0.0,
                "criteria_names": sorted(capability_criteria_names[capability]),
                "failed_rollout_count": len(failed_rollouts),
                "attempt_signal": attempt_signal,
                "cofailure_patterns": patterns_by_capability.get(capability, []),
            }
        )

    rows.sort(key=lambda row: (row["pass_rate"], -row["total_count"], row["capability"]))
    return rows, patterns


def detect_cofailure_patterns(
    criterion_failure_sets: dict[str, set[str]],
    criterion_capabilities: dict[str, str],
) -> list[dict[str, Any]]:
    """Find checks that tend to fail together."""
    grouped_by_fail_set: dict[frozenset[str], list[str]] = defaultdict(list)
    for criterion_name, failure_set in criterion_failure_sets.items():
        if len(failure_set) >= 2:
            grouped_by_fail_set[frozenset(failure_set)].append(criterion_name)

    patterns: list[dict[str, Any]] = []
    for matched_rollouts, criterion_names in grouped_by_fail_set.items():
        if len(criterion_names) < 2:
            continue
        capabilities = {
            criterion_capabilities.get(name, "General Execution") for name in criterion_names
        }
        patterns.append(
            {
                "capability": next(iter(capabilities)) if len(capabilities) == 1 else "Mixed",
                "criteria_names": sorted(criterion_names),
                "fail_count": len(matched_rollouts),
                "rollout_count": len(matched_rollouts),
            }
        )

    if patterns:
        patterns.sort(key=lambda pattern: (-pattern["fail_count"], -len(pattern["criteria_names"])))
        return patterns[:5]

    criterion_names = sorted(criterion_failure_sets)
    pair_patterns: list[dict[str, Any]] = []
    for idx, first_name in enumerate(criterion_names):
        for second_name in criterion_names[idx + 1 :]:
            intersection = criterion_failure_sets[first_name] & criterion_failure_sets[second_name]
            if len(intersection) < 2:
                continue
            capabilities = {
                criterion_capabilities.get(first_name, "General Execution"),
                criterion_capabilities.get(second_name, "General Execution"),
            }
            pair_patterns.append(
                {
                    "capability": next(iter(capabilities)) if len(capabilities) == 1 else "Mixed",
                    "criteria_names": [first_name, second_name],
                    "fail_count": len(intersection),
                    "rollout_count": len(intersection),
                }
            )

    pair_patterns.sort(key=lambda pattern: (-pattern["fail_count"], pattern["criteria_names"]))
    return pair_patterns[:5]


def rollout_composite_score(rollout: dict[str, Any]) -> float | None:
    """Return the average pass rate across one rollout's checks."""
    checks = analysis_checks_for_rollout(rollout)
    if not checks:
        return None
    passed_count = sum(1 for check in checks if check.get("passed"))
    return passed_count / len(checks)


def rollout_module_pass_rate(rollout: dict[str, Any]) -> float | None:
    """Return the programmatic pass rate for one rollout."""
    return rollout_composite_score(rollout)


def rollout_reward_model_score(rollout: dict[str, Any]) -> float | None:
    """Return the reward-model score for one rollout."""
    reward_model = rollout.get("reward_model")
    if not isinstance(reward_model, dict):
        return None
    return safe_float(reward_model.get("score"))


def bucket_system(
    name: str,
    *,
    detail: str | None = None,
    source: str | None = None,
) -> str:
    """Map a check into a system-level reporting bucket."""
    text = " ".join(part for part in (source, name, detail) if part).lower()
    if any(token in text for token in ("sequence", "before", "after", "order", "timing")):
        return "Sequence"
    if any(
        token in text
        for token in ("calendar", "meeting", "event", "attendee", "availability", "schedule")
    ):
        return "Calendar"
    if any(token in text for token in ("email", "mailhog", "recipient", "subject")):
        return "Email"
    if any(token in text for token in ("chat", "rocketchat", "dm", "message", "channel")):
        return "Chat"
    if any(
        token in text
        for token in (
            "hris",
            "frappe",
            "candidate",
            "employee",
            "recruit",
            "benefit",
            "payroll",
            "applicant",
        )
    ):
        return "HRIS"
    if any(token in text for token in ("crm", "deal", "contact", "pipeline")):
        return "CRM"
    if any(token in text for token in ("erp", "invoice", "vendor", "customer", "order")):
        return "ERP"
    return "Other"


def tool_label(tool_call: dict[str, Any]) -> str:
    """Build the canonical tool label used in reports."""
    tool_server = coerce_str(tool_call.get("tool_server")) or "unknown"
    tool_name = coerce_str(tool_call.get("tool_name")) or "unknown"
    return f"{tool_server}__{tool_name}"


def tool_error_category(summary: str | None) -> str:
    """Map a tool error summary into a category."""
    text = (summary or "").lower()
    if any(token in text for token in ("stale", "ref_not_found", "reference", "not found")):
        return "ref_not_found"
    if "timeout" in text:
        return "timeout"
    if any(token in text for token in ("creationerror", "failed to create", "could not create")):
        return "creation_error"
    if any(
        token in text
        for token in (
            "validation",
            "missing payload",
            "required property",
            "invalid-params",
            "field not permitted",
            "schema",
        )
    ):
        return "validation_error"
    if any(token in text for token in ("permission", "forbidden", "unauthorized")):
        return "permission_error"
    if any(token in text for token in ("login", "auth", "not authenticated")):
        return "auth_error"
    return "generic_error"


def tool_error_category_description(category: str) -> str:
    """Describe a tool error category in plain language."""
    descriptions = {
        "ref_not_found": "Missing or stale references in browser and tool state",
        "timeout": "Tool call exceeded its timeout window",
        "creation_error": "Create or update action failed in the backing system",
        "validation_error": "Arguments or fields were rejected before execution",
        "permission_error": "The backing system denied the action",
        "auth_error": "Login or authentication failed before the action could run",
        "generic_error": "Everything else that still surfaced as an execution failure",
    }
    return descriptions.get(category, "Unclassified tool execution failure")


def tool_error_message(summary: str | None) -> str | None:
    """Extract a normalized error message from a tool summary."""
    text = coerce_str(summary)
    if not text:
        return None
    return string_excerpt(text, max_chars=160)


def delta_number(left_value: Any, right_value: Any) -> float | None:
    """Return the left-minus-right numeric delta when both sides are numeric."""
    if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
        return None
    return float(left_value) - float(right_value)


def summarize_tool_observation(observation: Any) -> str | None:
    """Condense a tool observation into report-friendly text."""
    if isinstance(observation, str):
        return string_excerpt(observation)
    if isinstance(observation, dict):
        for key in ("summary", "text", "message", "result"):
            text = coerce_str(observation.get(key))
            if text:
                return string_excerpt(text)
        nested = observation.get("observation")
        if isinstance(nested, dict):
            nested_text = coerce_str(nested.get("text")) or coerce_str(nested.get("summary"))
            if nested_text:
                return string_excerpt(nested_text)
        return string_excerpt(json.dumps(observation))
    if observation is None:
        return None
    return string_excerpt(str(observation))


def bucket_capability(
    name: str,
    *,
    detail: str | None = None,
    source: str | None = None,
) -> str:
    """Map a tool or check into a capability bucket."""
    text = " ".join(part for part in (source, name, detail) if part).lower()
    if any(token in text for token in ("scope", "discipline", "dangerous", "unintended", "safety")):
        return "Safety / Scope Discipline"
    if any(
        token in text
        for token in ("calendar", "meeting", "event", "attendee", "availability", "schedule")
    ):
        return "Calendar Operations"
    if any(token in text for token in ("email", "mailhog", "inbox", "subject", "recipient")):
        return "Email Operations"
    if any(token in text for token in ("chat", "rocketchat", "dm", "message", "channel")):
        return "Chat Communication"
    if any(
        token in text
        for token in (
            "hris",
            "frappe",
            "candidate",
            "employee",
            "benefit",
            "recruit",
            "applicant",
            "payroll",
        )
    ):
        return "HRIS Operations"
    if any(token in text for token in ("crm", "deal", "contact", "account", "pipeline")):
        return "CRM Operations"
    if any(token in text for token in ("erp", "invoice", "order", "vendor", "customer")):
        return "ERP Operations"
    if any(
        token in text
        for token in ("code", "coding", "bash", "file", "repo", "git", "test", "script")
    ):
        return "Coding / Filesystem"
    if any(token in text for token in ("sheet", "spreadsheet", "drive", "docs", "workspace")):
        return "Workspace Documents"
    if any(token in text for token in ("search", "lookup", "find", "list", "read", "get")):
        return "Lookup / Retrieval"
    if any(token in text for token in ("final response", "final_observation", "respond", "answer")):
        return "Final Response"
    return "General Execution"


def infer_pass_state(
    *,
    reward: float | None,
    verifier_results: list[dict[str, Any]],
    summary_entry: dict[str, Any] | None,
    error: str | None,
) -> bool | None:
    """Infer the overall pass state for one rollout."""
    if reward is not None:
        return reward > 0
    if summary_entry is not None:
        summary_passed = summary_entry.get("verification_passed")
        if isinstance(summary_passed, bool):
            return summary_passed
        if "verification_passed" in summary_entry and error is None:
            return True

    verdicts = [bool(result["success"]) for result in verifier_results if "success" in result]
    if verdicts:
        return all(verdicts)
    if error:
        return False
    return None


def load_json_file(
    path: Path,
    *,
    warnings: list[str],
    required: bool,
) -> dict[str, Any] | list[Any] | None:
    """Load a JSON file and record parse warnings when needed."""
    if not path.is_file():
        if required:
            warnings.append(f"Skipping {path.parent}: missing {path.name}")
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Skipping {path}: {exc}")
        return None


def load_rollout_artifacts(
    path: Path,
    *,
    warnings: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load a native rollout or normalize an ATIF rollout into native shape."""
    native_path = path / NATIVE_ROLLOUT_FILENAME
    native_payload = (
        load_json_file(native_path, warnings=warnings, required=False)
        if native_path.is_file()
        else None
    )
    if isinstance(native_payload, dict):
        return native_payload, None

    atif_path = path / ATIF_ROLLOUT_RELATIVE_PATH
    atif_payload = (
        load_json_file(atif_path, warnings=warnings, required=False)
        if atif_path.is_file()
        else None
    )
    if isinstance(atif_payload, dict):
        return atif_to_artifacts(atif_payload), atif_payload

    if not native_path.is_file() and not atif_path.is_file():
        warnings.append(
            "Skipping "
            f"{path}: missing {NATIVE_ROLLOUT_FILENAME} or "
            f"{ATIF_ROLLOUT_RELATIVE_PATH.as_posix()}"
        )
    return None, None


def atif_to_artifacts(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Normalize an ATIF trajectory into the subset of native artifacts eval uses."""
    simlab_extra = extract_atif_simlab_extra(trajectory)
    agent = trajectory.get("agent")
    agent_dict = agent if isinstance(agent, dict) else {}
    agent_extra = agent_dict.get("extra")
    agent_extra_dict = agent_extra if isinstance(agent_extra, dict) else {}
    metadata = simlab_extra.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    tool_calls, tool_results = extract_atif_tool_artifacts(trajectory.get("steps"))

    return {
        "version": coerce_str(agent_dict.get("version")) or "0.1",
        "task_id": coerce_str(simlab_extra.get("task_id")) or "",
        "task": coerce_str(simlab_extra.get("task")) or "",
        "model": coerce_str(agent_dict.get("model_name")),
        "provider": coerce_str(agent_extra_dict.get("provider")),
        "messages": [],
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "metadata": metadata_dict,
        "final_observation": coerce_str(simlab_extra.get("final_observation")),
        "error": coerce_str(simlab_extra.get("run_error")),
        "steps_taken": safe_int(simlab_extra.get("steps_taken")) or len(tool_calls),
        "max_steps": safe_int(simlab_extra.get("max_steps")),
        "created_at": coerce_str(simlab_extra.get("created_at")),
    }


def extract_atif_simlab_extra(trajectory: dict[str, Any] | None) -> dict[str, Any]:
    """Return the embedded SimLab metadata block from an ATIF trajectory."""
    if not isinstance(trajectory, dict):
        return {}
    extra = trajectory.get("extra")
    extra_dict = extra if isinstance(extra, dict) else {}
    simlab = extra_dict.get("simlab")
    return simlab if isinstance(simlab, dict) else {}


def extract_reward_payload_from_atif(trajectory: dict[str, Any] | None) -> dict[str, Any]:
    """Fallback reward payload for ATIF rollouts when reward.json is absent."""
    simlab_extra = extract_atif_simlab_extra(trajectory)
    verifier = simlab_extra.get("verifier")
    verifier_dict = verifier if isinstance(verifier, dict) else {}
    payload = verifier_dict.get("payload")
    if isinstance(payload, dict):
        return payload

    reward = safe_float(verifier_dict.get("reward"))
    if reward is None:
        return {}
    return {"reward": reward}


def extract_atif_tool_artifacts(
    raw_steps: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten ATIF step tool calls into native-style tool call/result arrays."""
    steps = raw_steps if isinstance(raw_steps, list) else []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        call_items = step.get("tool_calls")
        if not isinstance(call_items, list):
            continue

        observation = step.get("observation")
        observation_dict = observation if isinstance(observation, dict) else {}
        raw_results = observation_dict.get("results")
        result_items = raw_results if isinstance(raw_results, list) else []
        results_by_call_id: dict[str, dict[str, Any]] = {}
        for result in result_items:
            if not isinstance(result, dict):
                continue
            call_id = coerce_str(result.get("source_call_id"))
            if call_id:
                results_by_call_id[call_id] = result

        for call in call_items:
            if not isinstance(call, dict):
                continue
            call_id = coerce_str(call.get("tool_call_id")) or f"call_{len(tool_calls) + 1}"
            call_extra = call.get("extra")
            call_extra_dict = call_extra if isinstance(call_extra, dict) else {}
            matched_result = results_by_call_id.get(call_id)
            result_extra = matched_result.get("extra") if isinstance(matched_result, dict) else None
            result_extra_dict = result_extra if isinstance(result_extra, dict) else {}

            tool_server = coerce_str(call_extra_dict.get("tool_server")) or coerce_str(
                result_extra_dict.get("tool_server")
            )
            tool_name = coerce_str(call.get("function_name")) or coerce_str(
                result_extra_dict.get("tool_name")
            )
            arguments = call.get("arguments")
            parameters = arguments if isinstance(arguments, dict) else {}

            tool_calls.append(
                {
                    "tool_server": tool_server or "unknown",
                    "tool_name": tool_name or "unknown",
                    "parameters": parameters,
                }
            )

            tool_result: dict[str, Any] = {}
            raw_observation = result_extra_dict.get("raw_observation")
            if raw_observation is not None:
                tool_result["observation"] = raw_observation
            else:
                content = (
                    coerce_str(matched_result.get("content"))
                    if isinstance(matched_result, dict)
                    else None
                )
                if content is not None:
                    tool_result["observation"] = {"text": content}

            is_error = result_extra_dict.get("is_error")
            if isinstance(is_error, bool):
                tool_result["is_error"] = is_error
            tool_results.append(tool_result)

    return tool_calls, tool_results


def extract_atif_final_metrics(trajectory: dict[str, Any] | None) -> dict[str, Any]:
    """Return ATIF final metrics when present."""
    if not isinstance(trajectory, dict):
        return {}
    final_metrics = trajectory.get("final_metrics")
    return final_metrics if isinstance(final_metrics, dict) else {}


def extract_atif_duration_seconds(trajectory: dict[str, Any] | None) -> float | None:
    """Estimate ATIF duration from the first and last step timestamps."""
    if not isinstance(trajectory, dict):
        return None
    raw_steps = trajectory.get("steps")
    if not isinstance(raw_steps, list):
        return None

    timestamps = [
        parsed
        for step in raw_steps
        if isinstance(step, dict)
        if isinstance(step.get("timestamp"), str)
        if (parsed := parse_iso8601_timestamp(step["timestamp"])) is not None
    ]
    if len(timestamps) < 2:
        return None
    return max((max(timestamps) - min(timestamps)).total_seconds(), 0.0)


def is_rollout_dir(path: Path) -> bool:
    """Return whether a directory looks like a rollout directory."""
    return path.is_dir() and (
        (path / NATIVE_ROLLOUT_FILENAME).is_file() or (path / ATIF_ROLLOUT_RELATIVE_PATH).is_file()
    )


def is_run_set_dir(path: Path) -> bool:
    """Return whether a directory looks like a run-set directory."""
    if not path.is_dir():
        return False
    if (path / "summary.json").is_file():
        return True
    return any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir())


def rollout_idx_from_dir_name(name: str) -> int | None:
    """Extract the rollout index from a rollout directory name."""
    if not name.startswith("rollout_"):
        return None
    return safe_int(name.removeprefix("rollout_"))


def module_label(module_path: str) -> str:
    """Convert a verifier module path into a short display label."""
    parts = [part for part in module_path.split(".") if part]
    return parts[-1] if parts else module_path


def parse_json_blob(raw_text: str | None) -> Any:
    """Parse a JSON blob when text is present."""
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def parse_iso8601_timestamp(raw_text: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into UTC."""
    if not raw_text:
        return None
    try:
        return datetime.fromisoformat(raw_text).astimezone(timezone.utc)
    except ValueError:
        return None


def sum_numbers(values: Any) -> float | None:
    """Return the sum of numeric values when any are present."""
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numbers) if numbers else None


def mean_numbers(values: Any) -> float | None:
    """Return the mean of numeric values when any are present."""
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.fmean(numbers) if numbers else None


def coerce_str(value: Any) -> str | None:
    """Return a stripped string, or ``None`` for non-strings."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def safe_float(value: Any) -> float | None:
    """Parse a float from a numeric value or numeric string."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    """Parse an int from a numeric value or numeric string."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def string_excerpt(value: str, max_chars: int = 120) -> str:
    """Trim long text for report output."""
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
