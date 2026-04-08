"""Rollout metrics helpers for token usage, timing, and cost estimation.

SimLab evaluation surfaces rollout metrics when the run artifacts include:

- metadata["rollout_metrics"]["token_usage"]["prompt_tokens_total"]
- metadata["rollout_metrics"]["token_usage"]["completion_tokens_total"]
- metadata["rollout_metrics"]["timing"]["duration_seconds"]
- metadata["rollout_metrics"]["cost"]["estimated_cost_usd"]
- metadata["rollout_metrics"]["extensions"] (optional)

This module keeps a single canonical on-the-wire shape while tolerating the
various usage payload formats returned by different agent frameworks.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import litellm


def merge_rollout_metrics_payload(
    base: dict[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge a rollout_metrics update into an existing payload."""
    for key, value in update.items():
        if isinstance(value, Mapping):
            existing_value = base.get(key)
            if isinstance(existing_value, dict):
                merge_rollout_metrics_payload(existing_value, value)
                continue
            if isinstance(existing_value, Mapping):
                existing_value_dict: dict[str, Any] = dict(existing_value)
                merge_rollout_metrics_payload(existing_value_dict, value)
                base[key] = existing_value_dict
                continue
            base[key] = dict(value)
            continue
        base[key] = value
    return base


def int_or_none(value: object) -> int | None:
    """Return an int for numeric values, otherwise None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def read_attr(payload: object, key: str) -> object:
    """Read an attribute or mapping key from the given payload."""
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def extract_token_usage(usage_payload: object) -> dict[str, int] | None:
    """Extract prompt/completion/total/cached tokens from a usage payload."""
    if usage_payload is None:
        return None

    prompt_value = read_attr(usage_payload, "prompt_tokens")
    if prompt_value is None:
        prompt_value = read_attr(usage_payload, "input_tokens")
    prompt_tokens = int_or_none(prompt_value)

    completion_value = read_attr(usage_payload, "completion_tokens")
    if completion_value is None:
        completion_value = read_attr(usage_payload, "output_tokens")
    completion_tokens = int_or_none(completion_value)
    total_tokens = int_or_none(read_attr(usage_payload, "total_tokens"))

    cached_tokens = int_or_none(read_attr(usage_payload, "cached_tokens"))
    input_details = read_attr(usage_payload, "input_tokens_details")
    if input_details is None:
        input_details = read_attr(usage_payload, "prompt_tokens_details")
    if cached_tokens is None:
        cached_tokens = int_or_none(read_attr(input_details, "cached_tokens"))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    extracted: dict[str, int] = {}
    if prompt_tokens is not None:
        extracted["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        extracted["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        extracted["total_tokens"] = total_tokens
    if cached_tokens is not None:
        extracted["cached_tokens"] = cached_tokens
    return extracted


def estimate_cost_usd_from_tokens(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Estimate USD cost using LiteLLM's pricing maps."""
    with contextlib.suppress(Exception):
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if isinstance(prompt_cost, (int, float)) and isinstance(completion_cost, (int, float)):
            total = float(prompt_cost) + float(completion_cost)
            if total > 0:
                return total
    return None


@dataclass
class RolloutMetricsTracker:
    """Accumulate rollout metrics across one agent execution."""

    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    total_tokens_total: int = 0
    cached_tokens_total: int = 0
    llm_inference_seconds_total: float = 0.0
    tool_execution_seconds_total: float = 0.0
    duration_seconds: float = 0.0

    def record_token_usage(self, usage_payload: object) -> None:
        """Accumulate token usage from a framework-specific usage payload."""
        extracted = extract_token_usage(usage_payload)
        if not extracted:
            return

        prompt_tokens = extracted.get("prompt_tokens")
        completion_tokens = extracted.get("completion_tokens")
        total_tokens = extracted.get("total_tokens")
        cached_tokens = extracted.get("cached_tokens")

        if prompt_tokens is not None:
            self.prompt_tokens_total += prompt_tokens
        if completion_tokens is not None:
            self.completion_tokens_total += completion_tokens
        if total_tokens is not None:
            self.total_tokens_total += total_tokens
        if cached_tokens is not None:
            self.cached_tokens_total += cached_tokens

    def record_llm_inference_seconds(self, seconds: float) -> None:
        """Accumulate time spent waiting on the model."""
        if seconds > 0:
            self.llm_inference_seconds_total += seconds

    def record_tool_execution_seconds(self, seconds: float) -> None:
        """Accumulate time spent executing tools."""
        if seconds > 0:
            self.tool_execution_seconds_total += seconds

    def record_duration_seconds(self, seconds: float) -> None:
        """Set end-to-end duration seconds for the rollout."""
        if seconds > 0:
            self.duration_seconds = seconds

    def build_metadata(
        self,
        *,
        model: str | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the canonical rollout_metrics dict for RunArtifacts.metadata."""
        payload: dict[str, Any] = {}

        token_usage: dict[str, Any] = {}
        if self.prompt_tokens_total > 0:
            token_usage["prompt_tokens_total"] = self.prompt_tokens_total
        if self.completion_tokens_total > 0:
            token_usage["completion_tokens_total"] = self.completion_tokens_total
        if self.total_tokens_total > 0:
            token_usage["total_tokens_total"] = self.total_tokens_total
        if self.cached_tokens_total > 0:
            token_usage["cached_tokens_total"] = self.cached_tokens_total
        if token_usage:
            payload["token_usage"] = token_usage

        timing: dict[str, Any] = {}
        if self.duration_seconds > 0:
            timing["duration_seconds"] = round(self.duration_seconds, 4)
        if self.llm_inference_seconds_total > 0:
            timing["llm_inference_seconds_total"] = round(self.llm_inference_seconds_total, 4)
        if self.tool_execution_seconds_total > 0:
            timing["tool_execution_seconds_total"] = round(self.tool_execution_seconds_total, 4)
        if timing:
            payload["timing"] = timing

        if model and self.prompt_tokens_total > 0 and self.completion_tokens_total > 0:
            estimated_cost = estimate_cost_usd_from_tokens(
                model=model,
                prompt_tokens=self.prompt_tokens_total,
                completion_tokens=self.completion_tokens_total,
            )
            if estimated_cost is not None:
                payload["cost"] = {
                    "estimated_cost_usd": round(estimated_cost, 8),
                    "pricing_source": "litellm",
                }

        if extensions:
            payload["extensions"] = dict(extensions)

        return payload

    def merge_into(
        self,
        metadata: dict[str, Any],
        *,
        model: str | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> None:
        """Merge the tracked rollout metrics into metadata["rollout_metrics"]."""
        update = self.build_metadata(model=model, extensions=extensions)
        if not update:
            return

        existing = metadata.get("rollout_metrics")
        if isinstance(existing, dict):
            merge_rollout_metrics_payload(existing, update)
            return

        metadata["rollout_metrics"] = update


@dataclass(frozen=True)
class Timer:
    """Simple perf_counter timer."""

    started_at: float

    @staticmethod
    def start() -> Timer:
        """Start a new timer."""
        return Timer(started_at=time.perf_counter())

    def elapsed_seconds(self) -> float:
        """Return elapsed seconds since start."""
        return max(time.perf_counter() - self.started_at, 0.0)
