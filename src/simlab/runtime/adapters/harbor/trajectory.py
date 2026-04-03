"""Render native SimLab artifacts into Harbor ATIF trajectory output."""

from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


def build_atif_trajectory(
    artifacts: Any,  # noqa: ANN401
    *,
    run_id: str | None = None,
    verification_passed: bool | None = None,
    reward: float | None = None,
    reward_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Harbor ATIF-compatible trajectory from SimLab run artifacts."""
    created_at = _normalize_timestamp(getattr(artifacts, "created_at", None))
    steps = _build_steps(
        artifacts,
        created_at=created_at,
        verification_passed=verification_passed,
        reward=reward,
        reward_payload=reward_payload,
    )

    trajectory: dict[str, Any] = {
        "schema_version": "ATIF-v1.4",
        "session_id": run_id or _default_session_id(artifacts, created_at=created_at),
        "agent": _build_agent(artifacts),
        "steps": steps,
    }
    final_metrics = _build_final_metrics(artifacts, total_steps=len(steps))
    if final_metrics:
        trajectory["final_metrics"] = final_metrics
    extra = _build_root_extra(
        artifacts,
        created_at=created_at,
        verification_passed=verification_passed,
        reward=reward,
        reward_payload=reward_payload,
    )
    if extra:
        trajectory["extra"] = extra
    return trajectory


def write_atif_trajectory(
    path: Path,
    *,
    artifacts: Any,  # noqa: ANN401
    run_id: str | None = None,
    verification_passed: bool | None = None,
    reward: float | None = None,
    reward_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write Harbor ATIF output to ``path`` and return the rendered payload."""
    trajectory = build_atif_trajectory(
        artifacts,
        run_id=run_id,
        verification_passed=verification_passed,
        reward=reward,
        reward_payload=reward_payload,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    return trajectory


def _build_steps(
    artifacts: Any,  # noqa: ANN401
    *,
    created_at: str,
    verification_passed: bool | None,
    reward: float | None,
    reward_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    messages = getattr(artifacts, "messages", None)
    tool_calls = list(getattr(artifacts, "tool_calls", None) or [])
    tool_results = list(getattr(artifacts, "tool_results", None) or [])
    message_items = messages if isinstance(messages, list) else []

    steps: list[dict[str, Any]] = []
    consumed_tool_messages: set[int] = set()
    tool_cursor = 0

    for message_index, item in enumerate(message_items):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "system")
        content = item.get("content")
        timestamp = _normalize_timestamp(item.get("timestamp"), fallback=created_at)

        if role == "assistant":
            tool_call_specs = _extract_assistant_tool_calls(content)
            if tool_call_specs:
                following_tool_messages = _collect_following_tool_messages(
                    message_items,
                    start_index=message_index + 1,
                    consumed=consumed_tool_messages,
                )
                rendered_calls: list[dict[str, Any]] = []
                observation_results: list[dict[str, Any]] = []
                for call_spec in tool_call_specs:
                    raw_call = tool_calls[tool_cursor] if tool_cursor < len(tool_calls) else None
                    raw_result = (
                        tool_results[tool_cursor] if tool_cursor < len(tool_results) else None
                    )
                    tool_cursor += 1

                    tool_call_id = _extract_tool_call_id(call_spec, fallback_index=tool_cursor)
                    tool_message_index, tool_message = _match_tool_message(
                        following_tool_messages,
                        tool_call_id=tool_call_id,
                    )
                    if tool_message_index is not None:
                        consumed_tool_messages.add(tool_message_index)

                    function_name, tool_server = _resolve_tool_identity(call_spec, raw_call)
                    arguments = _resolve_tool_arguments(call_spec, raw_call)
                    rendered_call: dict[str, Any] = {
                        "tool_call_id": tool_call_id,
                        "function_name": function_name,
                        "arguments": arguments,
                    }
                    if tool_server:
                        rendered_call["extra"] = {"tool_server": tool_server}
                    rendered_calls.append(rendered_call)

                    observation_result = _build_observation_result(
                        tool_call_id=tool_call_id,
                        raw_result=raw_result,
                        tool_server=tool_server,
                        tool_name=function_name,
                        tool_message=tool_message,
                    )
                    if observation_result is not None:
                        observation_results.append(observation_result)

                step = _make_step(
                    source="agent",
                    timestamp=timestamp,
                    message=_extract_assistant_message(content),
                    artifacts=artifacts,
                )
                step["tool_calls"] = rendered_calls
                if observation_results:
                    step["observation"] = {"results": observation_results}
                extra = _message_extra(item, include_raw_content=True)
                if extra:
                    step["extra"] = extra
                steps.append(step)
                continue

        if role == "tool":
            if message_index in consumed_tool_messages:
                continue
            tool_content = content if isinstance(content, dict) else {}
            observation_result = _build_observation_result(
                tool_call_id=str(tool_content.get("tool_call_id") or f"tool_{len(steps) + 1}"),
                raw_result=None,
                tool_server=_string_or_none(tool_content.get("tool_server")),
                tool_name=_string_or_none(tool_content.get("tool_name")) or "tool",
                tool_message=item,
            )
            if observation_result is None:
                continue
            step = _make_step(
                source="system",
                timestamp=timestamp,
                message="Tool observation",
                artifacts=artifacts,
            )
            step["observation"] = {"results": [observation_result]}
            extra = _message_extra(item, include_raw_content=True)
            if extra:
                step["extra"] = extra
            steps.append(step)
            continue

        step = _make_step(
            source=_normalize_source(role),
            timestamp=timestamp,
            message=_extract_plain_message(content),
            artifacts=artifacts,
        )
        extra = _message_extra(item, include_raw_content=not isinstance(content, str))
        if extra:
            step["extra"] = extra
        steps.append(step)

    final_timestamp = steps[-1]["timestamp"] if steps else created_at
    run_error = _string_or_none(getattr(artifacts, "error", None))
    if run_error:
        status_step = _make_step(
            source="system",
            timestamp=final_timestamp,
            message=run_error,
            artifacts=artifacts,
        )
        status_step["extra"] = {
            "kind": "run_status",
            "status": "timeout" if _looks_like_timeout(run_error) else "error",
        }
        steps.append(status_step)
    if verification_passed is False:
        verifier_step = _make_step(
            source="system",
            timestamp=final_timestamp,
            message="Verifier failed",
            artifacts=artifacts,
        )
        verifier_extra: dict[str, Any] = {"kind": "verifier", "passed": False}
        if reward is not None:
            verifier_extra["reward"] = reward
        if reward_payload:
            verifier_extra["payload"] = reward_payload
        verifier_step["extra"] = verifier_extra
        steps.append(verifier_step)

    for step_id, step in enumerate(steps, start=1):
        step["step_id"] = step_id
    return steps


def _build_agent(artifacts: Any) -> dict[str, Any]:  # noqa: ANN401
    metadata = getattr(artifacts, "metadata", None)
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    cli_runtime = metadata_dict.get("cli_runtime")
    cli_runtime_dict = cli_runtime if isinstance(cli_runtime, dict) else {}
    agent_import_path = _string_or_none(cli_runtime_dict.get("agent_import_path"))

    agent_name = "simlab-reference-agent"
    if agent_import_path:
        agent_name = agent_import_path.split(":")[-1] or agent_name

    agent: dict[str, Any] = {
        "name": agent_name,
        "version": str(getattr(artifacts, "version", "0.1") or "0.1"),
        "model_name": str(getattr(artifacts, "model", "") or "unknown-model"),
    }
    extra: dict[str, Any] = {}
    provider = _string_or_none(getattr(artifacts, "provider", None))
    if provider:
        extra["provider"] = provider
    if agent_import_path:
        extra["agent_import_path"] = agent_import_path
    if extra:
        agent["extra"] = extra
    return agent


def _build_final_metrics(artifacts: object, *, total_steps: int) -> dict[str, Any]:
    metadata = getattr(artifacts, "metadata", None)
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    rollout_metrics = metadata_dict.get("rollout_metrics")
    rollout_metrics_dict = rollout_metrics if isinstance(rollout_metrics, dict) else {}
    token_usage = rollout_metrics_dict.get("token_usage")
    token_usage_dict = token_usage if isinstance(token_usage, dict) else {}
    cost = rollout_metrics_dict.get("cost")
    cost_dict = cost if isinstance(cost, dict) else {}

    final_metrics: dict[str, Any] = {"total_steps": total_steps}
    prompt_tokens = _int_or_none(
        token_usage_dict.get("prompt_tokens_total") or token_usage_dict.get("prompt_tokens")
    )
    completion_tokens = _int_or_none(
        token_usage_dict.get("completion_tokens_total") or token_usage_dict.get("completion_tokens")
    )
    cached_tokens = _int_or_none(
        token_usage_dict.get("cached_tokens_total") or token_usage_dict.get("cached_tokens")
    )
    cost_usd = _float_or_none(
        cost_dict.get("estimated_cost_usd") or metadata_dict.get("estimated_cost_usd")
    )
    if prompt_tokens is not None:
        final_metrics["total_prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        final_metrics["total_completion_tokens"] = completion_tokens
    if cached_tokens is not None:
        final_metrics["total_cached_tokens"] = cached_tokens
    if cost_usd is not None:
        final_metrics["total_cost_usd"] = cost_usd
    return final_metrics


def _build_root_extra(
    artifacts: Any,  # noqa: ANN401
    *,
    created_at: str,
    verification_passed: bool | None,
    reward: float | None,
    reward_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    simlab_extra: dict[str, Any] = {
        "task_id": _string_or_none(getattr(artifacts, "task_id", None)),
        "task": _string_or_none(getattr(artifacts, "task", None)),
        "created_at": created_at,
        "steps_taken": _int_or_none(getattr(artifacts, "steps_taken", None)),
        "max_steps": _int_or_none(getattr(artifacts, "max_steps", None)),
        "final_observation": _string_or_none(getattr(artifacts, "final_observation", None)),
        "run_error": _string_or_none(getattr(artifacts, "error", None)),
        "metadata": getattr(artifacts, "metadata", None) or {},
    }
    if verification_passed is not None or reward is not None or reward_payload:
        verifier_info: dict[str, Any] = {}
        if verification_passed is not None:
            verifier_info["passed"] = verification_passed
        if reward is not None:
            verifier_info["reward"] = reward
        if reward_payload:
            verifier_info["payload"] = reward_payload
        simlab_extra["verifier"] = verifier_info

    log_probs = getattr(artifacts, "log_probs", None)
    if log_probs is not None:
        simlab_extra["log_probs"] = log_probs

    return {"simlab": {key: value for key, value in simlab_extra.items() if value is not None}}


def _make_step(
    *,
    source: str,
    timestamp: str,
    message: str | None,
    artifacts: Any,  # noqa: ANN401
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "timestamp": timestamp,
        "source": source,
    }
    if message is not None:
        step["message"] = message
    if source == "agent":
        model_name = _string_or_none(getattr(artifacts, "model", None))
        if model_name:
            step["model_name"] = model_name
    return step


def _collect_following_tool_messages(
    messages: list[Any],
    *,
    start_index: int,
    consumed: set[int],
) -> list[tuple[int, dict[str, Any]]]:
    results: list[tuple[int, dict[str, Any]]] = []
    for index in range(start_index, len(messages)):
        item = messages[index]
        if not isinstance(item, dict):
            break
        if index in consumed:
            continue
        if item.get("role") != "tool":
            break
        results.append((index, item))
    return results


def _match_tool_message(
    candidates: list[tuple[int, dict[str, Any]]],
    *,
    tool_call_id: str,
) -> tuple[int | None, dict[str, Any] | None]:
    first_available: tuple[int | None, dict[str, Any] | None] = (None, None)
    for index, item in candidates:
        if first_available == (None, None):
            first_available = (index, item)
        content = item.get("content")
        if isinstance(content, dict) and str(content.get("tool_call_id") or "") == tool_call_id:
            return index, item
    return first_available


def _build_observation_result(
    *,
    tool_call_id: str,
    raw_result: Any,  # noqa: ANN401
    tool_server: str | None,
    tool_name: str,
    tool_message: dict[str, Any] | None,
) -> dict[str, Any] | None:
    observation = getattr(raw_result, "observation", None)
    if observation is None and isinstance(raw_result, dict):
        observation = raw_result.get("observation")
    is_error = getattr(raw_result, "is_error", None)
    if is_error is None and isinstance(raw_result, dict):
        is_error = raw_result.get("is_error")

    tool_content = tool_message.get("content") if isinstance(tool_message, dict) else None
    if observation is None and isinstance(tool_content, dict):
        observation = tool_content.get("summary")
        if is_error is None:
            is_error = tool_content.get("is_error")

    content = _normalize_observation_content(observation)
    if content is None:
        return None

    result: dict[str, Any] = {
        "source_call_id": tool_call_id,
        "content": content,
    }
    extra: dict[str, Any] = {}
    if tool_server:
        extra["tool_server"] = tool_server
    if tool_name:
        extra["tool_name"] = tool_name
    if isinstance(is_error, bool):
        extra["is_error"] = is_error
    if observation is not None and not isinstance(observation, str):
        extra["raw_observation"] = observation
    if isinstance(tool_content, dict):
        extra["tool_message"] = tool_content
    if extra:
        result["extra"] = extra
    return result


def _resolve_tool_identity(
    call_spec: dict[str, Any],
    raw_call: Any,  # noqa: ANN401
) -> tuple[str, str | None]:
    function_name = None
    tool_server = None

    function = call_spec.get("function")
    if isinstance(function, dict):
        raw_name = _string_or_none(function.get("name"))
        if raw_name:
            if "__" in raw_name:
                tool_server, function_name = raw_name.split("__", 1)
            else:
                function_name = raw_name

    if raw_call is not None:
        tool_server = _string_or_none(getattr(raw_call, "tool_server", None)) or tool_server
        function_name = _string_or_none(getattr(raw_call, "tool_name", None)) or function_name

    return function_name or "tool", tool_server


def _resolve_tool_arguments(
    call_spec: dict[str, Any],
    raw_call: Any,  # noqa: ANN401
) -> dict[str, Any]:
    function = call_spec.get("function")
    if isinstance(function, dict):
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, dict):
                    return parsed
    parameters = getattr(raw_call, "parameters", None)
    if isinstance(parameters, dict):
        return parameters
    if isinstance(raw_call, dict):
        maybe_parameters = raw_call.get("parameters")
        if isinstance(maybe_parameters, dict):
            return maybe_parameters
    return {}


def _extract_assistant_tool_calls(content: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    if not isinstance(content, dict):
        return []
    tool_calls = content.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [item for item in tool_calls if isinstance(item, dict)]


def _extract_tool_call_id(call_spec: dict[str, Any], *, fallback_index: int) -> str:
    return _string_or_none(call_spec.get("id")) or f"call_{fallback_index}"


def _extract_assistant_message(content: Any) -> str | None:  # noqa: ANN401
    if not isinstance(content, dict):
        return _extract_plain_message(content)
    message = _string_or_none(content.get("content"))
    return message or None


def _extract_plain_message(content: Any) -> str | None:  # noqa: ANN401
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("message", "content", "summary"):
            value = _string_or_none(content.get(key))
            if value is not None:
                return value
        return json.dumps(content, sort_keys=True)
    if content is None:
        return None
    return str(content)


def _message_extra(item: dict[str, Any], *, include_raw_content: bool) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if include_raw_content:
        extra["raw_message"] = item
    elif "timestamp" in item:
        extra["raw_timestamp"] = item["timestamp"]
    return extra


def _normalize_observation_content(observation: Any) -> str | None:  # noqa: ANN401
    if observation is None:
        return None
    if isinstance(observation, str):
        return observation
    if isinstance(observation, dict):
        for key in ("text", "summary", "message"):
            value = observation.get(key)
            if isinstance(value, str):
                return value
        nested = observation.get("observation")
        if isinstance(nested, dict):
            for key in ("text", "summary", "message"):
                value = nested.get(key)
                if isinstance(value, str):
                    return value
        return json.dumps(observation, sort_keys=True)
    return json.dumps(observation, sort_keys=True)


def _default_session_id(artifacts: Any, *, created_at: str) -> str:  # noqa: ANN401
    task_id = _string_or_none(getattr(artifacts, "task_id", None)) or "harbor-run"
    return f"{task_id}:{created_at}"


def _normalize_source(role: str) -> str:
    if role == "assistant":
        return "agent"
    if role in {"user", "system"}:
        return role
    return "system"


def _normalize_timestamp(raw: Any, fallback: str | None = None) -> str:  # noqa: ANN401
    candidate = raw if isinstance(raw, str) and raw else fallback
    if isinstance(candidate, str):
        normalized = candidate.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            pass
        else:
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _looks_like_timeout(text: str) -> bool:
    return text.strip().lower().startswith("rollout timeout")


def _string_or_none(value: Any) -> str | None:  # noqa: ANN401
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_none(value: Any) -> int | None:  # noqa: ANN401
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _float_or_none(value: Any) -> float | None:  # noqa: ANN401
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
