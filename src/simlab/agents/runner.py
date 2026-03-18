"""External-agent run loop (setup -> run) with timeout/error capture."""

from __future__ import annotations

import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from simlab.agents.base import BaseEnvironment
from simlab.agents.base import RunArtifacts
from simlab.agents.loader import load_agent_class
from simlab.agents.reference_agent import ReferenceAgent


def _run_maybe_async(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
    result = callable_obj(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


def run_with_agent_contract(
    *,
    task_id: str,
    instruction: str,
    model: str,
    provider: str,
    max_steps: int,
    environment: BaseEnvironment,
    agent_import_path: str | None = None,
    timeout_seconds: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    stop_event: threading.Event | None = None,
) -> RunArtifacts:
    """Execute setup() then run() and always return populated artifacts."""
    artifacts = RunArtifacts(
        task_id=task_id,
        task=instruction,
        model=model,
        provider=provider,
        max_steps=max_steps,
    )
    if agent_import_path:
        agent_cls = load_agent_class(agent_import_path)
        agent = agent_cls()
    else:
        agent = ReferenceAgent(provider=provider, api_key=api_key, base_url=base_url)

    def _execute() -> None:
        _run_maybe_async(agent.setup, environment)
        run_kwargs: dict[str, Any] = {}
        if stop_event is not None:
            sig = inspect.signature(agent.run)
            if "stop_event" in sig.parameters:
                run_kwargs["stop_event"] = stop_event
        _run_maybe_async(agent.run, instruction, environment, artifacts, **run_kwargs)

    try:
        if timeout_seconds is None:
            _execute()
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute)
                future.result(timeout=timeout_seconds)
    except FuturesTimeout:
        artifacts.error = "Rollout timeout exceeded"
        artifacts.metadata["timeout"] = True
    except Exception as exc:
        artifacts.error = f"Agent execution failed: {exc}"

    if artifacts.final_observation and not artifacts.messages:
        artifacts.record_message("assistant", artifacts.final_observation)
    return artifacts
