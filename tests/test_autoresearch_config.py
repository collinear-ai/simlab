from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from simlab.autoresearch.config import AutoresearchRunConfig
from simlab.autoresearch.config import load_run_config
from simlab.autoresearch.config import render_run_toml_template
from simlab.autoresearch.validate import AutoresearchProposal
from simlab.autoresearch.validate import parse_and_validate_proposal
from simlab.autoresearch.validate import validate_scenario_prompt


def test_load_run_config_parses_valid_toml(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        """
[run]
env = "env1"
tasks_dir = "./tasks"
task_ids = ["t1"]
runtime = "daytona"
rollout_count = 1
max_parallel = 1
max_steps = 30
agent_timeout_seconds = 600.0
no_seed = false

[agent]
model = "gpt-4o-mini"
provider = "openai"
api_key_env = "OPENAI_API_KEY"
base_url_env = "OPENAI_API_BASE"

[proposer]
model = "gpt-5.4"
provider = "openai"
api_key_env = "OPENAI_API_KEY"
base_url_env = "OPENAI_API_BASE"

[verifier]
model = "gpt-5.4"
provider = "openai"
api_key_env = "OPENAI_API_KEY"
base_url_env = "OPENAI_API_BASE"
""",
        encoding="utf-8",
    )
    cfg = load_run_config(path)
    assert isinstance(cfg, AutoresearchRunConfig)
    assert cfg.run.env == "env1"
    assert cfg.agent.model == "gpt-4o-mini"
    assert cfg.proposer.model == "gpt-5.4"


def test_load_run_config_resolves_tasks_dir_relative_to_config_file(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "run.toml"
    path.write_text(
        """
[run]
env = "env1"
tasks_dir = "./tasks"
task_ids = ["t1"]
runtime = "local"
rollout_count = 1
max_parallel = 1
max_steps = 30
agent_timeout_seconds = 600.0
no_seed = false

[agent]
model = "gpt-4o-mini"
provider = "openai"
api_key_env = "OPENAI_API_KEY"
base_url_env = "OPENAI_API_BASE"

[proposer]
model = "gpt-5.4"
provider = "openai"
api_key_env = "OPENAI_API_KEY"
base_url_env = "OPENAI_API_BASE"

[verifier]
model = "gpt-5.4"
provider = "openai"
api_key_env = "OPENAI_API_KEY"
base_url_env = "OPENAI_API_BASE"
""",
        encoding="utf-8",
    )

    cfg = load_run_config(path)
    assert Path(cfg.run.tasks_dir) == (cfg_dir / "tasks").resolve()


def test_render_run_toml_template_escapes_values_for_toml() -> None:
    text = render_run_toml_template(
        env='env"1',
        tasks_dir="C:\\work\\tasks",
        task_ids=['t"1', "t\\2"],
    )
    parsed = tomllib.loads(text)
    assert parsed["run"]["env"] == 'env"1'
    assert parsed["run"]["tasks_dir"] == "C:\\work\\tasks"
    assert parsed["run"]["task_ids"] == ['t"1', "t\\2"]


def test_run_config_rejects_local_multi_rollout() -> None:
    with pytest.raises(ValueError):
        AutoresearchRunConfig.model_validate(
            {
                "run": {
                    "env": "env1",
                    "tasks_dir": "./tasks",
                    "task_ids": ["t1"],
                    "runtime": "local",
                    "rollout_count": 2,
                    "max_parallel": 1,
                    "max_steps": 30,
                    "agent_timeout_seconds": 600.0,
                    "no_seed": False,
                },
                "agent": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_BASE",
                },
                "proposer": {
                    "model": "gpt-5.4",
                    "provider": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_BASE",
                },
                "verifier": {
                    "model": "gpt-5.4",
                    "provider": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_BASE",
                },
            }
        )


def test_run_config_rejects_daytona_multi_rollout() -> None:
    with pytest.raises(ValueError):
        AutoresearchRunConfig.model_validate(
            {
                "run": {
                    "env": "env1",
                    "tasks_dir": "./tasks",
                    "task_ids": ["t1"],
                    "runtime": "daytona",
                    "rollout_count": 2,
                    "max_parallel": 1,
                    "max_steps": 30,
                    "agent_timeout_seconds": 600.0,
                    "no_seed": False,
                },
                "agent": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_BASE",
                },
                "proposer": {
                    "model": "gpt-5.4",
                    "provider": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_BASE",
                },
                "verifier": {
                    "model": "gpt-5.4",
                    "provider": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_BASE",
                },
            }
        )


def test_validate_scenario_prompt_limits_size() -> None:
    validate_scenario_prompt("ok\n")
    with pytest.raises(ValueError):
        validate_scenario_prompt("a" * 20000)


def test_parse_and_validate_proposal_requires_surface() -> None:
    payload = {
        "version": "0.1",
        "iteration": 1,
        "surface": "scenario_prompt",
        "rationale": "x",
        "change_type": "y",
        "changes": {"scenario_prompt": "# H\n\nDo x\n"},
    }
    proposal = parse_and_validate_proposal(payload)
    assert isinstance(proposal, AutoresearchProposal)

    bad = dict(payload)
    bad["surface"] = "tools"
    with pytest.raises(ValueError):
        parse_and_validate_proposal(bad)
