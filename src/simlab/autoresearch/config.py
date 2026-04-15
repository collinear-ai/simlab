"""Autoresearch run configuration.

This module defines the strict on-the-wire TOML contract for `simlab autoresearch`.
Version 1 keeps the evaluation contract fixed and optimizes only the runtime
scenario prompt injected into task instructions.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from textwrap import dedent
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

ObjectiveType = Literal[
    "pass_rate",
    "avg_reward",
    "check_pass_rate",
    "reward_model_score_mean",
    "tool_error_rate",
]
RuntimeType = Literal["daytona", "local"]


class RunSection(BaseModel):
    """Fixed execution contract for an autoresearch run."""

    model_config = {"extra": "forbid"}

    env: str = Field(description="Environment name under the environments root.")
    environments_dir: str | None = Field(
        default=None,
        description="Root directory for environments (overrides SIMLAB_ENVIRONMENTS_DIR).",
    )
    tasks_dir: str = Field(description="Path to a frozen local tasks bundle directory.")
    task_ids: list[str] = Field(min_length=1, description="Task IDs to run.")
    runtime: RuntimeType = Field(
        default="local",
        description="Execution backend.",
    )
    rollout_count: int = Field(default=1, ge=1, description="Rollouts per task per iteration.")
    max_parallel: int = Field(default=1, ge=1, description="Max concurrent Daytona sandboxes.")
    max_steps: int = Field(default=30, ge=1, description="Maximum agent steps per rollout.")
    agent_timeout_seconds: float = Field(
        default=600.0,
        gt=0.0,
        description="Hard timeout for agent setup plus run lifecycle.",
    )
    no_seed: bool = Field(
        default=False,
        description="Skip all seeding and provisioning for the task.",
    )

    @field_validator("env", "tasks_dir")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("environments_dir")
    @classmethod
    def _non_empty_string_or_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("task_ids")
    @classmethod
    def _normalize_task_ids(cls, value: list[str]) -> list[str]:
        normalized = [task_id.strip() for task_id in value if isinstance(task_id, str)]
        normalized = [task_id for task_id in normalized if task_id]
        if not normalized:
            raise ValueError("task_ids must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_runtime_constraints(self) -> RunSection:
        if self.rollout_count != 1:
            raise ValueError("autoresearch requires rollout_count=1")
        if self.max_parallel != 1:
            raise ValueError("autoresearch requires max_parallel=1")
        return self


class ModelSection(BaseModel):
    """Model settings for either the evaluated agent or the proposer."""

    model_config = {"extra": "forbid"}

    model: str = Field(description="LLM model name.")
    provider: str = Field(default="openai", description="LLM provider identifier.")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Env var for API key.")
    base_url_env: str | None = Field(
        default="OPENAI_API_BASE",
        description="Env var for base URL override.",
    )

    @field_validator("model")
    @classmethod
    def _normalize_model(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("model must be a string")
        return value.strip()

    @field_validator("api_key_env")
    @classmethod
    def _non_empty_api_key_env(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip().lower()

    def resolve_api_key(self) -> str | None:
        """Resolve an API key from `api_key_env`."""
        raw = os.environ.get(self.api_key_env)
        return raw.strip() if raw and raw.strip() else None

    def resolve_base_url(self) -> str | None:
        """Resolve an optional base URL override from `base_url_env`."""
        if not self.base_url_env:
            return None
        raw = os.environ.get(self.base_url_env)
        return raw.strip() if raw and raw.strip() else None


class ObjectiveSection(BaseModel):
    """Metric selection and optional early-stop target."""

    model_config = {"extra": "forbid"}

    type: ObjectiveType = Field(default="pass_rate", description="Metric to optimize.")
    target: float | None = Field(
        default=None,
        description="Stop early when objective reaches this threshold.",
    )

    @field_validator("target")
    @classmethod
    def _validate_target(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            raise TypeError("target must be a number")
        return float(value)


class BudgetSection(BaseModel):
    """Stop conditions for the sequential improvement loop."""

    model_config = {"extra": "forbid"}

    max_iterations: int = Field(default=6, ge=0, description="Max candidate iterations.")
    max_minutes: int = Field(
        default=90,
        ge=-1,
        description="Wall clock budget in minutes (-1 disables the time limit).",
    )
    no_improvement_window: int = Field(
        default=2,
        ge=-1,
        description="Stop after this many rejected iterations in a row (-1 disables).",
    )

    @field_validator("max_minutes")
    @classmethod
    def _validate_max_minutes(cls, value: int) -> int:
        if value == -1:
            return value
        if value < 1:
            raise ValueError("max_minutes must be -1 or >= 1")
        return value

    @field_validator("no_improvement_window")
    @classmethod
    def _validate_no_improvement_window(cls, value: int) -> int:
        if value == -1:
            return value
        if value < 0:
            raise ValueError("no_improvement_window must be -1 or >= 0")
        return value


class AutoresearchRunConfig(BaseModel):
    """Full validated run config loaded from TOML."""

    model_config = {"extra": "forbid"}

    run: RunSection
    agent: ModelSection
    proposer: ModelSection
    verifier: ModelSection
    objective: ObjectiveSection = Field(default_factory=ObjectiveSection)
    budget: BudgetSection = Field(default_factory=BudgetSection)


def load_run_config(path: Path) -> AutoresearchRunConfig:
    """Load and validate an autoresearch config TOML file."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("run config must be a TOML table")
    _resolve_run_paths_relative_to_config(raw, config_path=path)
    return AutoresearchRunConfig.model_validate(raw)


def _resolve_run_paths_relative_to_config(raw: dict[str, object], *, config_path: Path) -> None:
    """Resolve relative run paths against the config file directory.

    The on-the-wire contract permits relative paths. For config files, we make
    their meaning stable: relative paths are resolved from the config file's
    directory, not from the caller's working directory.
    """
    run = raw.get("run")
    if not isinstance(run, dict):
        return

    tasks_dir = run.get("tasks_dir")
    if isinstance(tasks_dir, str):
        run["tasks_dir"] = _resolve_path_str(tasks_dir, base_dir=config_path.parent)

    environments_dir = run.get("environments_dir")
    if isinstance(environments_dir, str):
        run["environments_dir"] = _resolve_path_str(environments_dir, base_dir=config_path.parent)


def _resolve_path_str(value: str, *, base_dir: Path) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        return str(candidate.resolve())
    except OSError:
        return str(candidate.absolute())


def render_run_toml_template(
    *,
    env: str,
    tasks_dir: str,
    task_ids: list[str],
    environments_dir: str | None = None,
) -> str:
    """Render a commented starter TOML file for progressive disclosure."""

    def q(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    task_list = ", ".join(f'"{q(task_id)}"' for task_id in task_ids)
    run_lines = [
        "[run]",
        f'env = "{q(env)}"',
    ]
    if environments_dir:
        run_lines.append(f'environments_dir = "{q(environments_dir)}"')
    run_lines.extend(
        [
            f'tasks_dir = "{q(tasks_dir)}"',
            f"task_ids = [{task_list}]",
            'runtime = "local"  # local or daytona',
            "rollout_count = 1",
            "max_parallel = 1",
            "max_steps = 30",
            "agent_timeout_seconds = 600.0",
            "no_seed = false",
        ]
    )
    run_block = "\n".join(run_lines)

    return dedent(
        f"""\
        # SimLab Autoresearch run config
        #
        # This file is the fixed contract for an autoresearch run.
        # The only thing that changes during the loop is the runtime scenario prompt.

        {run_block}

        [agent]
        # Required. Example: "gpt-5.2"
        model = ""
        provider = "openai"
        api_key_env = "OPENAI_API_KEY"
        base_url_env = "OPENAI_API_BASE"

        [proposer]
        # Required. Example: "gpt-5.4"
        model = ""
        provider = "openai"
        api_key_env = "OPENAI_API_KEY"
        base_url_env = "OPENAI_API_BASE"

        [verifier]
        # Controls the rubric judge model used during task runs.
        # Required. Example: "gpt-5.4"
        model = ""
        provider = "openai"
        api_key_env = "OPENAI_API_KEY"
        base_url_env = "OPENAI_API_BASE"

        [objective]
        # pass_rate, avg_reward, check_pass_rate, reward_model_score_mean, tool_error_rate
        type = "pass_rate"
        target = 0.8

        [budget]
        max_iterations = 6
        max_minutes = 90  # set to -1 for no time limit
        no_improvement_window = 2  # set to -1 to disable this stop rule
        """
    )
