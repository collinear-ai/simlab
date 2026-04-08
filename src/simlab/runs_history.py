"""Helpers for ``simlab runs history``.

This module scans the on-disk ``output/`` directory produced by SimLab CLI runs.
It is intentionally file-based and does not require any database or API access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Literal

RunResult = Literal["pass", "fail", "unknown"]
RunType = Literal["single", "parallel"]


@dataclass(frozen=True, slots=True)
class RunHistoryEntry:
    """A single row in the runs history view."""

    run_type: RunType
    run_id: str
    path: Path
    task_id: str
    model: str | None
    provider: str | None
    created_at: datetime | None
    duration_seconds: float | None
    result: RunResult
    reward: float | None = None
    rollout_count: int | None = None
    passed_count: int | None = None
    failed_count: int | None = None

    def sort_key(self) -> tuple[float, str]:
        """Return the stable sorting key for the history view."""
        timestamp = self.created_at.timestamp() if self.created_at else 0.0
        return (timestamp, self.run_id)

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON shape for ``--json`` output."""
        created_at = None
        if self.created_at is not None:
            created_at = self.created_at.astimezone(timezone.utc).isoformat()
        return {
            "run_type": self.run_type,
            "run_id": self.run_id,
            "path": str(self.path),
            "task_id": self.task_id,
            "model": self.model,
            "provider": self.provider,
            "created_at": created_at,
            "duration_seconds": self.duration_seconds,
            "result": self.result,
            "reward": self.reward,
            "rollout_count": self.rollout_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
        }


def load_runs_history(output_dir: Path) -> tuple[list[RunHistoryEntry], list[str]]:
    """Load run history entries from an output directory."""
    warnings: list[str] = []
    resolved_dir = output_dir.expanduser().resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(str(resolved_dir))
    if not resolved_dir.is_dir():
        raise NotADirectoryError(str(resolved_dir))

    entries: list[RunHistoryEntry] = []
    for child in sorted(resolved_dir.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        if is_parallel_run_dir(child):
            entry = load_parallel_run_entry(child, warnings=warnings)
            if entry is not None:
                entries.append(entry)
            continue
        if is_single_run_dir(child):
            entry = load_single_run_entry(child, warnings=warnings)
            if entry is not None:
                entries.append(entry)
    entries.sort(key=lambda entry: entry.sort_key(), reverse=True)
    return entries, warnings


def is_single_run_dir(path: Path) -> bool:
    """Return whether a directory looks like a single rollout run."""
    if not path.is_dir():
        return False
    return (path / "artifacts.json").is_file() or (path / "agent" / "trajectory.json").is_file()


def is_parallel_run_dir(path: Path) -> bool:
    """Return whether a directory looks like a parallel rollout run-set."""
    if not path.is_dir():
        return False
    if is_single_run_dir(path):
        return False
    if (path / "summary.json").is_file():
        return True
    return any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir())


def load_single_run_entry(path: Path, *, warnings: list[str]) -> RunHistoryEntry | None:
    """Load one single-run directory into the history row shape."""
    artifacts = load_single_run_artifacts(path, warnings=warnings)
    if artifacts is None:
        return None

    metadata = artifacts.get("metadata")
    metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}

    task_id = coerce_str(artifacts.get("task_id")) or path.name
    model = coerce_str(artifacts.get("model"))
    provider = coerce_str(artifacts.get("provider"))
    created_at = parse_timestamp(coerce_str(artifacts.get("created_at")))
    error = coerce_str(artifacts.get("error"))

    reward_payload = load_reward_payload(path, warnings=warnings)
    reward = safe_float(reward_payload.get("reward")) if reward_payload else None
    verifier_results = []
    raw_verifier_results = reward_payload.get("verifier_results") if reward_payload else None
    if isinstance(raw_verifier_results, list):
        verifier_results = [row for row in raw_verifier_results if isinstance(row, dict)]

    passed = infer_pass_state(reward=reward, verifier_results=verifier_results, error=error)
    if error:
        passed = False

    result: RunResult
    if passed is True:
        result = "pass"
    elif passed is False:
        result = "fail"
    else:
        result = "unknown"

    duration_seconds = extract_duration_seconds(metadata_dict)
    if duration_seconds is None:
        duration_seconds = infer_duration_seconds(
            created_at=created_at,
            candidate_paths=[
                path / "artifacts.json",
                path / "agent" / "trajectory.json",
                path / "verifier" / "reward.json",
                path / "verifier" / "reward.txt",
                path / "reward.json",
            ],
        )

    return RunHistoryEntry(
        run_type="single",
        run_id=path.name,
        path=path,
        task_id=task_id,
        model=model,
        provider=provider,
        created_at=created_at,
        duration_seconds=duration_seconds,
        result=result,
        reward=reward,
    )


def load_parallel_run_entry(path: Path, *, warnings: list[str]) -> RunHistoryEntry | None:
    """Load one parallel run directory into the history row shape."""
    summary_payload = load_json_file(path / "summary.json", warnings=warnings)
    if summary_payload is not None and not isinstance(summary_payload, dict):
        summary_payload = None
    summary_payload = summary_payload or {}

    task_id = coerce_str(summary_payload.get("task_id")) or parallel_task_id_from_dir_name(
        path.name
    )
    rollout_count = safe_int(summary_payload.get("rollout_count"))
    passed_count = safe_int(summary_payload.get("passed"))
    failed_count = safe_int(summary_payload.get("failed"))
    duration_seconds = safe_float(summary_payload.get("total_duration_seconds"))

    created_at = parse_parallel_dir_timestamp(path.name)
    if created_at is None:
        created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    model, provider = load_parallel_model_provider(path, warnings=warnings)

    result: RunResult = "unknown"
    if isinstance(failed_count, int) and failed_count > 0:
        result = "fail"
    elif isinstance(failed_count, int) and failed_count == 0 and passed_count is not None:
        result = "pass"

    return RunHistoryEntry(
        run_type="parallel",
        run_id=path.name,
        path=path,
        task_id=task_id,
        model=model,
        provider=provider,
        created_at=created_at,
        duration_seconds=duration_seconds,
        result=result,
        rollout_count=rollout_count,
        passed_count=passed_count,
        failed_count=failed_count,
    )


def load_parallel_model_provider(
    run_set_dir: Path, *, warnings: list[str]
) -> tuple[str | None, str | None]:
    """Load model/provider from the first rollout entry in a parallel run-set."""
    rollout_dirs = [
        child
        for child in sorted(run_set_dir.iterdir(), key=lambda item: item.name)
        if (child.is_dir() and child.name.startswith("rollout_") and rollout_has_artifacts(child))
    ]
    if not rollout_dirs:
        return None, None

    return load_rollout_model_provider(rollout_dirs[0], warnings=warnings)


def rollout_has_artifacts(path: Path) -> bool:
    """Return whether a rollout directory has a native or ATIF artifact payload."""
    if not path.is_dir():
        return False
    return (path / "artifacts.json").is_file() or (path / "agent" / "trajectory.json").is_file()


def load_single_run_artifacts(path: Path, *, warnings: list[str]) -> dict[str, Any] | None:
    """Load a single run as a native-artifacts-shaped payload."""
    native_payload = load_json_file(path / "artifacts.json", warnings=warnings)
    if isinstance(native_payload, dict):
        return native_payload

    atif_payload = load_json_file(path / "agent" / "trajectory.json", warnings=warnings)
    if not isinstance(atif_payload, dict):
        return None
    return atif_to_native_artifacts(atif_payload, fallback_task_id=path.name)


def load_rollout_model_provider(
    rollout_dir: Path, *, warnings: list[str]
) -> tuple[str | None, str | None]:
    """Load model/provider from a rollout directory."""
    artifacts = load_json_file(rollout_dir / "artifacts.json", warnings=warnings)
    if isinstance(artifacts, dict):
        return coerce_str(artifacts.get("model")), coerce_str(artifacts.get("provider"))

    trajectory = load_json_file(rollout_dir / "agent" / "trajectory.json", warnings=warnings)
    if not isinstance(trajectory, dict):
        return None, None

    agent = trajectory.get("agent")
    agent_dict = agent if isinstance(agent, dict) else {}
    agent_extra = agent_dict.get("extra")
    agent_extra_dict = agent_extra if isinstance(agent_extra, dict) else {}

    return coerce_str(agent_dict.get("model_name")), coerce_str(agent_extra_dict.get("provider"))


def atif_to_native_artifacts(
    trajectory: dict[str, Any],
    *,
    fallback_task_id: str,
) -> dict[str, Any]:
    """Normalize an ATIF trajectory into the subset of native artifacts we need."""
    extra = trajectory.get("extra")
    extra_dict = extra if isinstance(extra, dict) else {}
    simlab = extra_dict.get("simlab")
    simlab_dict = simlab if isinstance(simlab, dict) else {}

    agent = trajectory.get("agent")
    agent_dict = agent if isinstance(agent, dict) else {}
    agent_extra = agent_dict.get("extra")
    agent_extra_dict = agent_extra if isinstance(agent_extra, dict) else {}

    metadata = simlab_dict.get("metadata")
    metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}

    return {
        "task_id": coerce_str(simlab_dict.get("task_id")) or fallback_task_id,
        "model": coerce_str(agent_dict.get("model_name")),
        "provider": coerce_str(agent_extra_dict.get("provider")),
        "created_at": coerce_str(simlab_dict.get("created_at")),
        "metadata": metadata_dict,
        "error": coerce_str(simlab_dict.get("run_error")),
    }


def parallel_task_id_from_dir_name(dir_name: str) -> str:
    """Extract task_id from a run-set directory name."""
    if not dir_name.startswith("parallel_run_"):
        return dir_name
    suffix = dir_name.removeprefix("parallel_run_")
    parts = suffix.rsplit("_", 2)
    return parts[0] if parts else dir_name


def parse_parallel_dir_timestamp(dir_name: str) -> datetime | None:
    """Extract UTC timestamp from a run-set directory name."""
    if not dir_name.startswith("parallel_run_"):
        return None
    suffix = dir_name.removeprefix("parallel_run_")
    parts = suffix.rsplit("_", 2)
    if len(parts) != 3:
        return None
    stamp = f"{parts[1]}_{parts[2]}"
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def load_reward_payload(path: Path, *, warnings: list[str]) -> dict[str, Any] | None:
    """Load the reward payload for a rollout, when it exists."""
    reward_path = path / "verifier" / "reward.json"
    if not reward_path.is_file():
        reward_path = path / "reward.json"
    payload = load_json_file(reward_path, warnings=warnings)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        warnings.append(f"Skipping {reward_path}: expected a JSON object")
        return None
    return payload


def extract_duration_seconds(metadata: dict[str, Any]) -> float | None:
    """Extract rollout duration from artifacts metadata."""
    rollout_metrics = metadata.get("rollout_metrics")
    if not isinstance(rollout_metrics, dict):
        return None
    timing = rollout_metrics.get("timing")
    if not isinstance(timing, dict):
        return None
    return safe_float(timing.get("duration_seconds"))


def infer_duration_seconds(
    *,
    created_at: datetime | None,
    candidate_paths: list[Path],
) -> float | None:
    """Infer run duration from file timestamps when explicit metrics are missing."""
    if created_at is None:
        return None
    end_mtime = None
    for candidate in candidate_paths:
        if not candidate.is_file():
            continue
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        end_mtime = mtime if end_mtime is None else max(end_mtime, mtime)
    if end_mtime is None:
        return None
    end_time = datetime.fromtimestamp(end_mtime, tz=timezone.utc)
    duration = (end_time - created_at.astimezone(timezone.utc)).total_seconds()
    if duration < 0:
        return None
    return duration


def infer_pass_state(
    *,
    reward: float | None,
    verifier_results: list[dict[str, Any]],
    error: str | None,
) -> bool | None:
    """Infer whether a run passed, based on reward/verifier results."""
    if reward is not None:
        return reward > 0
    verdicts = [bool(result["success"]) for result in verifier_results if "success" in result]
    if verdicts:
        return all(verdicts)
    if error:
        return False
    return None


def load_json_file(path: Path, *, warnings: list[str]) -> dict[str, Any] | list[Any] | None:
    """Load a JSON file, adding a warning on failure."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Skipping {path}: {exc}")
        return None


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into a timezone-aware datetime."""
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def coerce_str(value: object) -> str | None:
    """Coerce a value into a non-empty string."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def safe_float(value: object) -> float | None:
    """Coerce a value into a float when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def safe_int(value: object) -> int | None:
    """Coerce a value into an int when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
