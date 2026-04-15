"""Tests for verifier runner: bundle downloads, local bundles, and rubric judge."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Self
from urllib.error import URLError

import pytest
from simlab.verifiers import runner
from simlab.verifiers.runner import RubricJudgeResult


def test_verifier_bundle_download_uses_api_key_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bundle downloads should authenticate with API-Key header."""
    captured: dict[str, object] = {}
    cache_root = tmp_path / "verifier-cache"
    monkeypatch.setattr(
        runner,
        "_get_verifier_cache_root",
        lambda: cache_root,
    )
    shutil.rmtree(cache_root, ignore_errors=True)

    class _FakeRequest:
        def __init__(self, url: str, headers: dict[str, str]) -> None:
            captured["url"] = url
            captured["headers"] = headers

    def _fake_urlopen(_req: object, timeout: int) -> None:
        _ = timeout
        raise URLError("network down")

    monkeypatch.setattr(runner, "Request", _FakeRequest)
    monkeypatch.setattr(runner, "urlopen", _fake_urlopen)

    with pytest.raises(runner.VerifierBundleError, match="Cannot download verifier bundle"):
        runner._ensure_verifier_bundle_cached(
            scenario_id="coding",
            base_url="https://api.example.com",
            api_key="test-api-key",
        )

    assert captured["url"] == "https://api.example.com/v1/scenarios/coding/verifiers/bundle"
    assert captured["headers"] == {"API-Key": "test-api-key"}


def test_run_verifier_supports_local_bundle_file(tmp_path: Path) -> None:
    verifier_file = tmp_path / "generated_task.py"
    verifier_file.write_text(
        "\n".join(
            [
                "def verify(run_artifacts):",
                "    return run_artifacts.task_id == 'generated-task'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.run_verifier(
        "collinear.scenarios.customer_support.verifiers.generated_task",
        run_artifacts_adapter=type("Artifacts", (), {"task_id": "generated-task"})(),
        scenario_id="customer_support",
        local_verifier_path=verifier_file,
    )

    assert result.success is True


def test_run_verifier_supports_local_bundle_helper_imports(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers"
    verifiers_dir.mkdir()
    (verifiers_dir / "common.py").write_text(
        "\n".join(
            [
                "def is_expected(task_id):",
                "    return task_id == 'generated-task'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    verifier_file = verifiers_dir / "generated_task.py"
    verifier_file.write_text(
        "\n".join(
            [
                "from common import is_expected",
                "",
                "def verify(run_artifacts):",
                "    return is_expected(run_artifacts.task_id)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.run_verifier(
        "collinear.scenarios.customer_support.verifiers.generated_task",
        run_artifacts_adapter=type("Artifacts", (), {"task_id": "generated-task"})(),
        scenario_id="customer_support",
        local_verifier_path=verifier_file,
    )

    assert result.success is True


def test_run_verifier_supports_local_bundle_package_imports(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers"
    verifiers_dir.mkdir()
    (verifiers_dir / "common.py").write_text(
        "\n".join(
            [
                "def is_expected(task_id):",
                "    return task_id == 'generated-task'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    verifier_file = verifiers_dir / "generated_task.py"
    verifier_file.write_text(
        "\n".join(
            [
                "from collinear.scenarios.customer_support.verifiers.common import is_expected",
                "",
                "def verify(run_artifacts):",
                "    return is_expected(run_artifacts.task_id)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.run_verifier(
        "collinear.scenarios.customer_support.verifiers.generated_task",
        run_artifacts_adapter=type("Artifacts", (), {"task_id": "generated-task"})(),
        scenario_id="customer_support",
        local_verifier_path=verifier_file,
    )

    assert result.success is True


def test_run_verifier_supports_local_bundle_collinear_core_imports(tmp_path: Path) -> None:
    verifier_file = tmp_path / "generated_task.py"
    verifier_file.write_text(
        "\n".join(
            [
                "from collinear.core.run_artifacts import RunArtifacts",
                "from collinear.core.verifier import VerifierResult",
                "",
                "def verify(run_artifacts: RunArtifacts) -> VerifierResult:",
                "    return VerifierResult(success=run_artifacts.task_id == 'generated-task')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.run_verifier(
        "collinear.scenarios.customer_support.verifiers.generated_task",
        run_artifacts_adapter=type("Artifacts", (), {"task_id": "generated-task"})(),
        scenario_id="customer_support",
        local_verifier_path=verifier_file,
    )

    assert result.success is True


# ---------------------------------------------------------------------------
# collinear.* shim tests
# ---------------------------------------------------------------------------


class _FakeUrlopenResponse:
    """Minimal context-manager response for monkeypatching urlopen."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _cleanup_shim_modules(injected: list[str]) -> None:
    for mod_name in injected:
        sys.modules.pop(mod_name, None)


def test_install_collinear_core_shims_exposes_required_symbols() -> None:
    """The shim must expose Action, Observation, ToolCallingClient, StepResult, VerifierResult."""
    injected: list[str] = []
    try:
        runner._install_collinear_core_shims(injected)

        models_mod = importlib.import_module("collinear.core.models")
        client_mod = importlib.import_module("collinear.core.tool_calling_client")
        verifier_mod = importlib.import_module("collinear.core.verifier")
        ws_task_mod = importlib.import_module("collinear.workspace_controller.task_execution")

        action = models_mod.Action(tool_name="read_file", parameters={"path": "hello.txt"})
        assert action.tool_name == "read_file"
        assert action.parameters == {"path": "hello.txt"}

        # Action with no parameters defaults to {} (mirrors the real contract)
        bare = models_mod.Action(tool_name="list")
        assert bare.parameters == {}

        obs = models_mod.Observation(is_error=False, text="ok")
        assert obs.content == []
        assert obs.metadata == {}

        client = client_mod.ToolCallingClient(base_url="http://server:8020/")
        assert client.base_url == "http://server:8020"
        client.close()  # no-op

        # Both VerifierResult shims point at the same class so verifiers using
        # either import path get the same type back.
        assert ws_task_mod.VerifierResult is verifier_mod.VerifierResult
        assert client_mod.StepResult is not None
    finally:
        _cleanup_shim_modules(injected)


def test_shim_tool_calling_client_step_parses_observation_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shim must read /step responses with the real envelope shape."""
    injected: list[str] = []
    captured: dict[str, object] = {}

    def _fake_urlopen(req: object, timeout: float) -> _FakeUrlopenResponse:
        captured["url"] = req.full_url  # type: ignore[attr-defined]
        captured["data"] = req.data  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        payload = json.dumps(
            {
                "observation": {
                    "is_error": False,
                    "text": "Hello World\n",
                    "content": [{"type": "text", "text": "Hello World"}],
                    "metadata": {"path": "hello.txt"},
                },
                "reward": 1.0,
                "done": False,
            }
        ).encode()
        return _FakeUrlopenResponse(payload)

    try:
        runner._install_collinear_core_shims(injected)
        monkeypatch.setattr(runner, "urlopen", _fake_urlopen)

        models_mod = importlib.import_module("collinear.core.models")
        client_mod = importlib.import_module("collinear.core.tool_calling_client")

        client = client_mod.ToolCallingClient(base_url="http://server:8020/")
        result = client.step(
            models_mod.Action(tool_name="read_file", parameters={"path": "hello.txt"})
        )

        assert captured["url"] == "http://server:8020/step"
        body = json.loads(captured["data"].decode("utf-8"))  # type: ignore[attr-defined]
        assert body["action"]["tool_name"] == "read_file"
        assert body["action"]["parameters"] == {"path": "hello.txt"}

        assert result.observation.is_error is False
        assert result.observation.text == "Hello World\n"
        assert result.observation.content == [{"type": "text", "text": "Hello World"}]
        assert result.observation.metadata == {"path": "hello.txt"}
        assert result.reward == 1.0
        assert result.done is False
    finally:
        _cleanup_shim_modules(injected)


def test_shim_tool_calling_client_step_surfaces_server_error_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When /step returns is_error=true, the shim must surface it (not coerce to false)."""
    injected: list[str] = []

    def _fake_urlopen(_req: object, timeout: float) -> _FakeUrlopenResponse:
        _ = timeout
        payload = json.dumps(
            {
                "observation": {"is_error": True, "text": "ENOENT: no such file"},
                "reward": 0.0,
                "done": False,
            }
        ).encode()
        return _FakeUrlopenResponse(payload)

    try:
        runner._install_collinear_core_shims(injected)
        monkeypatch.setattr(runner, "urlopen", _fake_urlopen)

        models_mod = importlib.import_module("collinear.core.models")
        client_mod = importlib.import_module("collinear.core.tool_calling_client")

        client = client_mod.ToolCallingClient(base_url="http://server:8020")
        result = client.step(
            models_mod.Action(tool_name="read_file", parameters={"path": "missing.txt"})
        )

        assert result.observation.is_error is True
        assert result.observation.text == "ENOENT: no such file"
    finally:
        _cleanup_shim_modules(injected)


def test_run_verifier_supports_local_bundle_collinear_action_and_client_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a verifier importing Action and ToolCallingClient runs via local bundle."""

    def _fake_urlopen(_req: object, timeout: float) -> _FakeUrlopenResponse:
        _ = timeout
        payload = json.dumps({"observation": {"is_error": False, "text": "Hello World"}}).encode()
        return _FakeUrlopenResponse(payload)

    monkeypatch.setattr(runner, "urlopen", _fake_urlopen)

    verifier_file = tmp_path / "hello_world.py"
    verifier_file.write_text(
        "\n".join(
            [
                "from collinear.core.models import Action",
                "from collinear.core.run_artifacts import RunArtifacts",
                "from collinear.core.tool_calling_client import ToolCallingClient",
                "from collinear.core.verifier import VerifierResult",
                "",
                "",
                "def verify(run_artifacts: RunArtifacts) -> VerifierResult:",
                "    client = ToolCallingClient(base_url='http://server:8020')",
                "    try:",
                "        action = Action(tool_name='read_file', parameters={'path': 'hello.txt'})",
                "        result = client.step(action)",
                "        if result.observation.is_error:",
                "            return VerifierResult(success=False, message=result.observation.text)",
                "        if 'Hello World' in result.observation.text:",
                "            return VerifierResult(success=True, message='ok')",
                "        return VerifierResult(success=False, message='missing greeting')",
                "    finally:",
                "        client.close()",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.run_verifier(
        "collinear.scenarios.coding.verifiers.hello_world",
        run_artifacts_adapter=type("Artifacts", (), {"task_id": "hello_world"})(),
        scenario_id="coding",
        local_verifier_path=verifier_file,
    )

    assert result.success is True, result.message


def test_run_verifier_supports_local_bundle_workspace_controller_verifier_result_import(
    tmp_path: Path,
) -> None:
    """A verifier importing VerifierResult from workspace_controller.task_execution must run."""
    verifier_file = tmp_path / "hr_task.py"
    verifier_file.write_text(
        "\n".join(
            [
                "from collinear.core.run_artifacts import RunArtifacts",
                "from collinear.workspace_controller.task_execution import VerifierResult",
                "",
                "",
                "def verify(run_artifacts: RunArtifacts) -> VerifierResult:",
                "    return VerifierResult(success=True, message='ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.run_verifier(
        "collinear.scenarios.hr_people_management.verifiers.hr_task",
        run_artifacts_adapter=type("Artifacts", (), {"task_id": "hr_task"})(),
        scenario_id="hr_people_management",
        local_verifier_path=verifier_file,
    )

    assert result.success is True, result.message


# ---------------------------------------------------------------------------
# Rubric judge tests
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    """Test _parse_judge_response with various LLM outputs."""

    def test_valid_json(self) -> None:
        raw = (
            '{"verdict": "PASS", "score": 0.85, "confidence": 0.9, '
            '"evidence": ["Did X"], "failed_criteria": [], '
            '"dimension_scores": [{"dimension": "Quality", "score": 0.9, "reason": "Good"}]}'
        )
        result = runner._parse_judge_response(raw)
        assert isinstance(result, RubricJudgeResult)
        assert result.score == 0.85
        assert result.verdict == "PASS"
        assert result.confidence == 0.9
        assert result.evidence == ["Did X"]
        assert result.failed_criteria == []
        assert len(result.dimension_scores) == 1
        assert result.dimension_scores[0]["dimension"] == "Quality"
        assert result.error is None

    def test_json_in_markdown_fences(self) -> None:
        raw = (
            "```json\n"
            '{"verdict": "FAIL", "score": 0.3, "confidence": 0.7, '
            '"evidence": [], "failed_criteria": ["A"]}\n'
            "```"
        )
        result = runner._parse_judge_response(raw)
        assert result.score == 0.3
        assert result.verdict == "FAIL"
        assert result.failed_criteria == ["A"]
        assert result.error is None

    def test_json_embedded_in_text(self) -> None:
        raw = (
            "Here is my evaluation:\n"
            '{"verdict": "PASS", "score": 0.7, "confidence": 0.8, '
            '"evidence": ["B"], "failed_criteria": []}\n'
            "That is all."
        )
        result = runner._parse_judge_response(raw)
        assert result.score == 0.7
        assert result.error is None

    def test_unparseable_returns_error(self) -> None:
        raw = "I cannot evaluate this task because reasons."
        result = runner._parse_judge_response(raw)
        assert result.score == 0.0
        assert result.verdict == "ERROR"
        assert result.error is not None
        assert "Could not parse JSON" in result.error

    def test_score_clamped_to_0_1(self) -> None:
        raw = (
            '{"verdict": "PASS", "score": 1.5, "confidence": -0.2, '
            '"evidence": [], "failed_criteria": []}'
        )
        result = runner._parse_judge_response(raw)
        assert result.score == 1.0
        assert result.confidence == 0.0

    def test_missing_fields_use_defaults(self) -> None:
        raw = '{"score": 0.5}'
        result = runner._parse_judge_response(raw)
        assert result.score == 0.5
        assert result.verdict == "FAIL"  # 0.5 < 0.6 threshold
        assert result.evidence == []
        assert result.dimension_scores == []

    def test_score_at_threshold_passes(self) -> None:
        raw = '{"score": 0.6}'
        result = runner._parse_judge_response(raw)
        assert result.verdict == "PASS"


class TestRubricJudgeResultToDict:
    """Test RubricJudgeResult serialization."""

    def test_without_error(self) -> None:
        r = RubricJudgeResult(
            score=0.8,
            verdict="PASS",
            confidence=0.9,
            evidence=["A"],
            failed_criteria=[],
            dimension_scores=[],
        )
        d = r.to_dict()
        assert d["score"] == 0.8
        assert "error" not in d

    def test_with_error(self) -> None:
        r = RubricJudgeResult(
            score=0.0,
            verdict="ERROR",
            confidence=0.0,
            evidence=[],
            failed_criteria=[],
            dimension_scores=[],
            error="LLM call failed",
        )
        d = r.to_dict()
        assert d["error"] == "LLM call failed"


class TestTruncateMessagesForJudge:
    """Test message truncation for judge prompt."""

    def test_short_messages_unchanged(self) -> None:
        msgs: list[dict] = [{"role": "user", "content": "hello"}]
        result = runner._truncate_messages_for_judge(msgs)
        assert "hello" in result

    def test_long_tool_messages_truncated(self) -> None:
        long_content = "x" * 5000
        msgs: list[dict] = [{"role": "tool", "content": long_content}]
        # max_chars=3000 triggers truncation (raw > 3000) but is large enough
        # to include the "[truncated]" marker in the output
        result = runner._truncate_messages_for_judge(msgs, max_chars=3000)
        assert "[truncated]" in result
        assert len(result) < len(long_content)


class TestExtractJudgeJson:
    """Test JSON extraction from LLM output."""

    def test_plain_json(self) -> None:
        assert runner._extract_judge_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self) -> None:
        assert runner._extract_judge_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_embedded(self) -> None:
        assert runner._extract_judge_json('text {"a": 1} more') == {"a": 1}

    def test_not_json(self) -> None:
        assert runner._extract_judge_json("just text") is None
