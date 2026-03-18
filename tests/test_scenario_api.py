"""Tests for unified Scenario Manager client and URL resolution."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from simlab.api.client import ScenarioManagerApiError
from simlab.api.client import ScenarioManagerClient
from simlab.api.client import resolve_scenario_manager_api_url
from simlab.api.schemas import ScenarioSummary
from simlab.api.schemas import ScenarioTasksResponse
from simlab.composer.engine import EnvConfig
from simlab.config import DEFAULT_SCENARIO_MANAGER_API_URL


def test_resolve_scenario_manager_api_url_default() -> None:
    """Without env or config, default URL is used."""
    with patch.dict(os.environ, {"SIMLAB_SCENARIO_MANAGER_API_URL": ""}, clear=False):
        url = resolve_scenario_manager_api_url()
    assert url == DEFAULT_SCENARIO_MANAGER_API_URL.rstrip("/")
    assert url == "https://rl-gym-api.collinear.ai"


def test_resolve_scenario_manager_api_url_env_override() -> None:
    """Env var overrides default."""
    with patch.dict(os.environ, {"SIMLAB_SCENARIO_MANAGER_API_URL": "http://localhost:9011"}):
        url = resolve_scenario_manager_api_url()
    assert url == "http://localhost:9011"


def test_resolve_scenario_manager_api_url_from_config_obj() -> None:
    """Config object's scenario_manager_api_url is used when no env."""
    with patch.dict(os.environ, {"SIMLAB_SCENARIO_MANAGER_API_URL": ""}, clear=False):
        config = EnvConfig(scenario_manager_api_url="http://custom:9022")
    url = resolve_scenario_manager_api_url(config=config)
    assert url == "http://custom:9022"


def test_client_uses_simlab_collinear_api_key_env_var() -> None:
    with patch.dict(
        os.environ,
        {
            "SIMLAB_COLLINEAR_API_KEY": "key-from-env",
        },
        clear=False,
    ):
        client = ScenarioManagerClient(base_url="https://api.example.com")
    assert client.api_key == "key-from-env"


def test_list_scenarios_success() -> None:
    """list_scenarios returns list from API."""
    fake_response = [
        {
            "scenario_id": "human_resource",
            "name": "Human Resource",
            "description": "HR scenario",
            "num_tasks": 5,
            "num_npcs": 2,
            "tool_servers": [
                {"name": "frappe-hrms", "server_type": "hrms"},
                {"name": "rocketchat", "server_type": "chat"},
            ],
        },
    ]
    client = ScenarioManagerClient(base_url="https://api.example.com")
    with patch.object(client, "_request", return_value=fake_response):
        result = client.list_scenarios()
    assert isinstance(result[0], ScenarioSummary)
    assert result[0].scenario_id == "human_resource"


def test_list_scenarios_include_hidden_uses_query_flag() -> None:
    client = ScenarioManagerClient(base_url="https://api.example.com")
    with patch.object(client, "_request", return_value=[]) as mocked_request:
        client.list_scenarios(include_hidden=True)
    mocked_request.assert_called_once_with("GET", "/v1/scenarios?include_hidden=true")


def test_list_scenarios_non_list_raises() -> None:
    client = ScenarioManagerClient(base_url="https://api.example.com")
    with (
        patch.object(client, "_request", return_value={"error": "bad"}),
        pytest.raises(ScenarioManagerApiError, match="did not return a list"),
    ):
        client.list_scenarios()


def test_list_scenario_tasks_success() -> None:
    fake_response = {
        "scenario_id": "human_resource",
        "tasks": [
            {"task_id": "t-1", "name": "Task 1", "difficulty": "easy", "tool_servers": []},
        ],
    }
    client = ScenarioManagerClient(base_url="https://api.example.com")
    with patch.object(client, "_request", return_value=fake_response) as mocked_request:
        result = client.list_scenario_tasks("human_resource")
    assert isinstance(result, ScenarioTasksResponse)
    assert result.scenario_id == "human_resource"
    assert len(result.tasks) == 1
    mocked_request.assert_called_once_with(
        "GET", "/v1/scenarios/human_resource/tasks?include_test=true"
    )


def test_list_scenario_tasks_include_hidden_and_test_uses_both_flags() -> None:
    client = ScenarioManagerClient(base_url="https://api.example.com")
    with patch.object(client, "_request", return_value={"scenario_id": "x", "tasks": []}) as mocked:
        client.list_scenario_tasks("customer_support", include_hidden=True, include_test=True)
    mocked.assert_called_once_with(
        "GET",
        "/v1/scenarios/customer_support/tasks?include_hidden=true&include_test=true",
    )


def test_list_scenario_tasks_can_explicitly_exclude_test_tasks() -> None:
    client = ScenarioManagerClient(base_url="https://api.example.com")
    with patch.object(client, "_request", return_value={"scenario_id": "x", "tasks": []}) as mocked:
        client.list_scenario_tasks("customer_support", include_hidden=True, include_test=False)
    mocked.assert_called_once_with(
        "GET",
        "/v1/scenarios/customer_support/tasks?include_hidden=true&include_test=false",
    )


def test_resolve_template_to_backend_id_fetches_with_include_hidden() -> None:
    fake_scenarios = [ScenarioSummary(scenario_id="human_resource", name="Human Resource")]
    client = ScenarioManagerClient(base_url="https://api.example.com")
    with patch.object(client, "list_scenarios", return_value=fake_scenarios) as mocked:
        resolved = client.resolve_template_to_backend_id("human_resource")
    assert resolved == "human_resource"
    mocked.assert_called_once_with(include_hidden=True)
