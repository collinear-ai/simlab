"""HTTP client for Scenario Manager APIs (listing + task generation)."""

from __future__ import annotations

from typing import cast
from urllib.parse import urlparse

import requests

from simlab.api.schemas import ScenarioSummary
from simlab.api.schemas import ScenarioTasksResponse
from simlab.api.schemas import TaskGenJob
from simlab.api.schemas import TaskGenRequest
from simlab.api.schemas import TaskGenResult
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_scenario_manager_api_url
from simlab.telemetry import build_scenario_manager_headers

_DEFAULT_TIMEOUT_SECONDS = 60
_ALLOWED_SCHEMES = ("http", "https")


class ScenarioManagerApiError(Exception):
    """Raised when the Scenario Manager API returns an error."""

    _AUTH_HINT = (
        "Authentication failed.\n"
        "  Get your API key at https://platform.collinear.ai "
        "(Developer Resources → API Keys).\n"
        "  Then run: simlab auth login"
    )

    def __init__(self, status_code: int, detail: str) -> None:  # noqa: D107
        self.status_code = status_code
        self.detail = detail
        if status_code == 401:
            super().__init__(self._AUTH_HINT)
        else:
            super().__init__(f"HTTP {status_code}: {detail}")


class ScenarioManagerClient:
    """Unified client for Scenario Manager APIs."""

    def __init__(  # noqa: D107
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = resolve_scenario_manager_api_url(base_url=base_url)
        self.api_key = resolve_collinear_api_key(api_key)
        self.timeout_seconds = timeout_seconds
        self._validate_base_url(self.base_url)

    def list_scenarios(self, *, include_hidden: bool = False) -> list[ScenarioSummary]:
        """GET /v1/scenarios (authenticated)."""
        path = "/v1/scenarios?include_hidden=true" if include_hidden else "/v1/scenarios"
        data = self._request("GET", path)
        if not isinstance(data, list):
            raise ScenarioManagerApiError(0, "GET /v1/scenarios did not return a list")
        return [ScenarioSummary.model_validate(item) for item in data]

    def list_scenario_tasks(
        self,
        scenario_id: str,
        *,
        include_hidden: bool = False,
        include_test: bool = True,
    ) -> ScenarioTasksResponse:
        """GET /v1/scenarios/{scenario_id}/tasks (authenticated)."""
        query: list[str] = []
        if include_hidden:
            query.append("include_hidden=true")
        query.append(f"include_test={'true' if include_test else 'false'}")
        query_string = f"?{'&'.join(query)}" if query else ""
        path = f"/v1/scenarios/{scenario_id}/tasks{query_string}"
        data = self._request("GET", path)
        if not isinstance(data, dict):
            raise ScenarioManagerApiError(0, "GET /v1/scenarios/.../tasks did not return an object")
        return ScenarioTasksResponse.model_validate(data)

    def resolve_template_to_backend_id(
        self,
        template: str,
        *,
        scenarios: list[ScenarioSummary] | None = None,
    ) -> str:
        """Resolve user-facing template string to backend scenario_id."""
        scenarios_data = (
            scenarios if scenarios is not None else self.list_scenarios(include_hidden=True)
        )
        norm = _normalize_for_match(template)
        norm_singular = norm.rstrip("s") if norm.endswith("s") else norm

        for scenario in scenarios_data:
            sid = scenario.scenario_id.strip()
            name = scenario.name.strip()
            sid_norm = _normalize_for_match(sid)
            family = sid.split(":", 1)[0].strip()
            family_norm = _normalize_for_match(family)
            name_norm = _normalize_for_match(name)
            if sid_norm in (norm, norm_singular):
                return sid
            if family_norm in (norm, norm_singular):
                return sid
            if name_norm in (norm, norm_singular):
                return sid
            if name_norm.startswith((norm_singular, norm)):
                return sid

        options = [
            f"{s.name.strip() or '?'} ({s.scenario_id.strip() or '?'})" for s in scenarios_data
        ]
        options_hint = ", ".join(options[:8]) + ("..." if len(options) > 8 else "")
        raise ScenarioManagerApiError(
            0,
            (
                f"Template '{template}' not found. "
                f"Run templates list. Available: {options_hint or 'none'}"
            ),
        )

    def submit_task_gen(self, request: TaskGenRequest) -> TaskGenJob:
        """POST /v1/task-gen."""
        data = self._request("POST", "/v1/task-gen", body=request.model_dump())
        if not isinstance(data, dict):
            raise ScenarioManagerApiError(0, "POST /v1/task-gen did not return an object")
        return TaskGenJob.model_validate(data)

    def get_task_gen_status(self, job_id: str) -> TaskGenJob:
        """GET /v1/task-gen/{job_id}."""
        data = self._request("GET", f"/v1/task-gen/{job_id}")
        if not isinstance(data, dict):
            raise ScenarioManagerApiError(0, "GET /v1/task-gen/{job_id} did not return an object")
        return TaskGenJob.model_validate(data)

    def get_task_gen_results(self, job_id: str) -> TaskGenResult:
        """GET /v1/task-gen/{job_id}/results."""
        data = self._request("GET", f"/v1/task-gen/{job_id}/results")
        if not isinstance(data, dict):
            raise ScenarioManagerApiError(
                0, "GET /v1/task-gen/{job_id}/results did not return an object"
            )
        return TaskGenResult.model_validate(data)

    def _request(
        self,
        method: str,
        path: str,
        body: object | None = None,
    ) -> object:
        """Low-level HTTP helper with JSON encoding and error handling."""
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["API-Key"] = self.api_key
        headers.update(build_scenario_manager_headers())

        try:
            resp = requests.request(
                method,
                url,
                json=body,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            return cast(object, resp.json())
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else 0
            detail = _parse_error_body(response)
            raise ScenarioManagerApiError(status_code, detail) from exc
        except requests.exceptions.ConnectionError as exc:
            raise ScenarioManagerApiError(
                0, f"Cannot reach server at {self.base_url}: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise ScenarioManagerApiError(0, f"Request to {url} timed out") from exc
        except requests.exceptions.RequestException as exc:
            raise ScenarioManagerApiError(0, str(exc)) from exc
        except ValueError as exc:
            raise ScenarioManagerApiError(
                0, f"Invalid JSON from Scenario Manager {path}: {exc}"
            ) from exc

    @staticmethod
    def _validate_base_url(base_url: str) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise ScenarioManagerApiError(
                0,
                f"Scenario Manager URL must use http or https, got: {parsed.scheme}",
            )


def _parse_error_body(response: requests.Response | None) -> str:
    """Try to extract a human-readable error from the HTTP response."""
    if response is None:
        return "HTTP request failed"
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("detail", body))
        return str(body)
    except Exception:
        return response.reason or f"HTTP {response.status_code}"


def _normalize_for_match(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "-").replace("_", "-")
