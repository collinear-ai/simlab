"""Run verifier modules (python_module) against run artifacts.

Verifier modules are imported by path and must define verify(run_artifacts).
We build an adapter from simlab RunArtifacts so verifiers written for
collinear's RunArtifacts interface work without depending on the collinear package.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import re
import shutil
import sys
import tempfile
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from simlab.agents.base import RunArtifacts as CLIRunArtifacts

logger = logging.getLogger(__name__)

_VERIFIER_MODULE_RE = re.compile(r"collinear\.scenarios\.([^.]+)\.verifiers\.")
_BUNDLE_TIMEOUT = 60


class VerifierResult:
    """Result from a verifier execution (success, message, output)."""

    def __init__(self, success: bool, message: str = "", output: str = "") -> None:
        """Initialize with success flag and optional message/output."""
        self.success = success
        self.message = message
        self.output = output

    def __bool__(self) -> bool:
        """Treat success as truthy."""
        return self.success


@dataclass
class _ToolServerInfo:
    name: str
    tool_server_url: str


@dataclass
class _DiffRecord:
    """Minimal diff record for adapter (verifiers may read diffs)."""

    tool_server: str
    scope: str
    summary: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    step_index: int | None = None
    tool_name: str | None = None
    raw_data: dict[str, Any] | None = None


class _VerifierArtifactsAdapter:
    """Adapter that quacks like collinear's RunArtifacts for verifier modules."""

    def __init__(
        self,
        *,
        version: str,
        task_id: str,
        task: str,
        task_file: str,
        tool_server_name: str,
        tool_server_url: str,
        agent_strategy: str,
        model: str,
        provider: str,
        steps_taken: int,
        max_steps: int,
        tool_servers: list[_ToolServerInfo],
        metadata: dict[str, Any],
        messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]] | None = None,
        final_observation: str | None,
        last_tool_message: str | None,
        verifier_input: str | None,
        rollback_input: str | None,
        diffs: list[_DiffRecord],
        created_at: str,
        rollout_trace_path: str | None = None,
        log_probs: list | None = None,
        reference_folder_id: str | None = None,
        seed_data: dict[str, Any] | None = None,
    ) -> None:
        self.version = version
        self.task_id = task_id
        self.task = task
        self.task_file = task_file
        self.tool_server_name = tool_server_name
        self.tool_server_url = tool_server_url
        self.agent_strategy = agent_strategy
        self.model = model
        self.provider = provider
        self.steps_taken = steps_taken
        self.max_steps = max_steps
        self.tool_servers = tool_servers
        self.metadata = metadata
        self.messages = messages
        self.tool_calls = tool_calls or []
        self.final_observation = final_observation
        self.last_tool_message = last_tool_message
        self.verifier_input = verifier_input
        self.rollback_input = rollback_input
        self.diffs = diffs
        self.created_at = created_at
        self.rollout_trace_path = rollout_trace_path
        self.log_probs = log_probs
        self.reference_folder_id = reference_folder_id
        self.seed_data = seed_data or {}

    def server_url(self, name: str | None = None) -> str | None:
        if name:
            for s in self.tool_servers:
                if s.name == name:
                    return s.tool_server_url
        if self.tool_server_url:
            return self.tool_server_url
        if self.tool_servers:
            return self.tool_servers[0].tool_server_url
        return None


def build_verifier_artifacts(
    cli_artifacts: CLIRunArtifacts,
    task_file: Path,
    tool_servers: dict[str, str],
) -> _VerifierArtifactsAdapter:
    """Build an adapter from CLI RunArtifacts + task path + endpoints for verifiers."""
    tool_servers_list = [
        _ToolServerInfo(name=name, tool_server_url=url) for name, url in tool_servers.items()
    ]
    first_name = next(iter(tool_servers), "")
    first_url = tool_servers.get(first_name, "")

    resource_id = (
        cli_artifacts.metadata.get("resource_id")
        or cli_artifacts.metadata.get("resource_key")
        or cli_artifacts.metadata.get("page_id")
        or cli_artifacts.metadata.get("epic_key")
    )
    verifier_input = resource_id or (cli_artifacts.final_observation or "")

    last_tool_message = None
    for msg in reversed(cli_artifacts.messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            last_tool_message = msg.get("content")
            if isinstance(last_tool_message, dict):
                last_tool_message = str(last_tool_message)
            break

    return _VerifierArtifactsAdapter(
        version=getattr(cli_artifacts, "version", "0.1"),
        task_id=cli_artifacts.task_id,
        task=cli_artifacts.task,
        task_file=str(task_file),
        tool_server_name=first_name,
        tool_server_url=first_url,
        agent_strategy="tool-calling",
        model=cli_artifacts.model or "",
        provider=cli_artifacts.provider or "",
        steps_taken=cli_artifacts.steps_taken,
        max_steps=cli_artifacts.max_steps or 0,
        tool_servers=tool_servers_list,
        metadata=dict(cli_artifacts.metadata),
        messages=list(cli_artifacts.messages),
        tool_calls=list(getattr(cli_artifacts, "tool_calls", None) or []),
        final_observation=cli_artifacts.final_observation,
        last_tool_message=last_tool_message,
        verifier_input=verifier_input,
        rollback_input=verifier_input,
        diffs=[],
        created_at=getattr(cli_artifacts, "created_at", datetime.now(timezone.utc).isoformat()),
    )


def infer_scenario_from_evaluator(evaluator: dict[str, Any]) -> str | None:
    """Infer scenario ID from evaluator module path.

    Example: "collinear.scenarios.coding.verifiers.X" -> "coding"
    """
    if evaluator.get("func") != "python_module":
        return None
    module_path = evaluator.get("module", "")
    match = re.match(r"collinear\.scenarios\.([^.]+)\.verifiers\.", module_path)
    return match.group(1) if match else None


def _scenario_id_from_module_path(module_path: str) -> str | None:
    """Infer scenario ID from verifier module path."""
    match = _VERIFIER_MODULE_RE.match(module_path)
    return match.group(1) if match else None


def _get_verifier_cache_root() -> Path:
    """Root directory for cached verifier bundles."""
    return Path.home() / ".cache" / "simlab" / "verifiers"


def _ensure_verifier_bundle_cached(
    scenario_id: str,
    base_url: str,
    api_key: str | None = None,
    verifier_cache_root: Path | None = None,
) -> Path:
    """Download verifier bundle if needed, return cache root."""
    root = verifier_cache_root if verifier_cache_root is not None else _get_verifier_cache_root()
    cache_root = root / scenario_id
    collinear_dir = cache_root / "collinear"
    if (
        collinear_dir.is_dir()
        and (collinear_dir / "scenarios" / scenario_id / "verifiers").is_dir()
    ):
        print(
            f"Using cached verifier bundle for {scenario_id} ({cache_root})",
            file=sys.stderr,
        )
        return cache_root

    print(f"Downloading verifier bundle for {scenario_id} from API...", file=sys.stderr)
    url = f"{base_url.rstrip('/')}/v1/scenarios/{scenario_id}/verifiers/bundle"
    headers: dict[str, str] = {}
    if api_key:
        headers["API-Key"] = api_key
    try:
        req = Request(url, headers=headers)  # noqa: S310
        with urlopen(req, timeout=_BUNDLE_TIMEOUT) as resp:  # noqa: S310
            zip_bytes = resp.read()
    except HTTPError as e:
        raise VerifierBundleError(
            f"Verifier bundle not found for scenario {scenario_id}: {e.code} {e.reason}"
        ) from e
    except (URLError, TimeoutError, OSError) as e:
        raise VerifierBundleError(
            f"Cannot download verifier bundle from {url}. Check API URL and network: {e}"
        ) from e

    cache_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
        zf.extractall(cache_root)

    return cache_root


def _write_package_init(package_dir: Path) -> None:
    """Ensure a regular package has an __init__.py file."""
    package_dir.mkdir(parents=True, exist_ok=True)
    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")


def _stage_local_verifier_bundle(
    verifier_module_path: str,
    local_verifier_path: Path,
) -> tuple[Path | None, str | None]:
    """Build a temporary import root that mirrors the generated verifier package."""
    scenario_id = _scenario_id_from_module_path(verifier_module_path)
    if scenario_id is None:
        return None, None

    import_root = Path(tempfile.mkdtemp(prefix="simlab-local-verifier-"))
    staged_collinear_dir = import_root / "collinear"
    staged_verifiers_dir = staged_collinear_dir / "scenarios" / scenario_id / "verifiers"

    _write_package_init(staged_collinear_dir)
    _write_package_init(staged_collinear_dir / "scenarios")
    _write_package_init(staged_collinear_dir / "scenarios" / scenario_id)
    _write_package_init(staged_verifiers_dir)

    source_dir = local_verifier_path.parent
    if source_dir.name == "verifiers":
        shutil.copytree(
            source_dir,
            staged_verifiers_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    else:
        for source_file in source_dir.iterdir():
            if source_file.is_file() and source_file.suffix == ".py":
                shutil.copy2(source_file, staged_verifiers_dir / source_file.name)

    return import_root, scenario_id


def _collinear_runtime_module_names() -> list[str]:
    """Return runtime module names that should be isolated for verifier imports."""
    exact_names = (
        "collinear.scenarios",
        "collinear.core",
        "collinear.workspace_controller",
    )
    prefixes = (
        "collinear.scenarios.",
        "collinear.core.",
        "collinear.workspace_controller.",
    )
    return [name for name in sys.modules if name in exact_names or name.startswith(prefixes)]


def _install_collinear_core_shims(injected_modules: list[str]) -> None:
    """Inject minimal collinear.core modules for verifier imports."""

    def ensure_shim(name: str) -> ModuleType:
        if name not in sys.modules:
            mod = ModuleType(name)
            mod.__path__ = []
            mod.__package__ = name
            sys.modules[name] = mod
            injected_modules.append(name)
        return sys.modules[name]

    ensure_shim("collinear.core")
    run_artifacts_module = ensure_shim("collinear.core.run_artifacts")
    run_artifacts_module.RunArtifacts = _VerifierArtifactsAdapter  # type: ignore[attr-defined]
    verifier_module = ensure_shim("collinear.core.verifier")
    verifier_module.VerifierResult = VerifierResult  # type: ignore[attr-defined]


class VerifierBundleError(Exception):
    """Failed to fetch or prepare verifier bundle from API."""


def _coerce_verifier_result(result: object) -> VerifierResult:
    """Normalize verifier return values to the CLI result type."""
    if isinstance(result, VerifierResult):
        return result
    if hasattr(result, "success") and hasattr(result, "message"):
        return VerifierResult(
            success=bool(result.success),
            message=getattr(result, "message", "") or "",
            output=getattr(result, "output", "") or "",
        )
    if isinstance(result, tuple) and len(result) == 2:
        success, message = result
        return VerifierResult(success=success, message=str(message))
    return VerifierResult(success=bool(result), message=str(result))


def _run_imported_verifier(
    module: ModuleType,
    verifier_module_path: str,
    run_artifacts_adapter: _VerifierArtifactsAdapter,
) -> VerifierResult:
    """Call a loaded verifier module and normalize the return value."""
    if not hasattr(module, "verify"):
        return VerifierResult(
            success=False,
            message=f"Verifier module {verifier_module_path} has no 'verify' function",
        )
    try:
        return _coerce_verifier_result(module.verify(run_artifacts=run_artifacts_adapter))
    except Exception as e:
        return VerifierResult(success=False, message=f"Verifier error: {e}", output=str(e))


def run_verifier(
    verifier_module_path: str,
    run_artifacts_adapter: _VerifierArtifactsAdapter,
    scenario_id: str,  # noqa: ARG001 -- kept for API compatibility with callers
    scenario_manager_base_url: str | None = None,
    scenario_manager_api_key: str | None = None,
    local_verifier_path: Path | None = None,
    verifier_cache_root: Path | None = None,
) -> VerifierResult:
    """Import the verifier module and call its verify(run_artifacts=...) function.

    When scenario_manager_base_url is set and the module path is under
    collinear.scenarios.<scenario_id>.verifiers, downloads the verifier bundle
    from the API (if not cached) and runs from cache.
    """
    if local_verifier_path is not None:
        import_root, _ = _stage_local_verifier_bundle(verifier_module_path, local_verifier_path)
        import_paths: list[str] = []
        if import_root is not None:
            import_paths.append(str(import_root))
        import_paths.append(str(local_verifier_path.parent))
        if local_verifier_path.parent.name == "verifiers":
            import_paths.append(str(local_verifier_path.parent.parent))

        added_paths: list[str] = []
        injected_modules: list[str] = []
        original_module = sys.modules.get(verifier_module_path)
        original_runtime_modules = {
            name: sys.modules[name] for name in _collinear_runtime_module_names()
        }
        original_collinear = sys.modules.get("collinear")
        original_collinear_path = None
        if original_collinear is not None and hasattr(original_collinear, "__path__"):
            original_collinear_path = list(original_collinear.__path__)

        try:
            for path in reversed(import_paths):
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)

            if import_root is not None and original_collinear is not None:
                staged_collinear_path = str(import_root / "collinear")
                updated_collinear_path = [
                    path for path in original_collinear.__path__ if path != staged_collinear_path
                ]
                original_collinear.__path__ = [
                    staged_collinear_path,
                    *updated_collinear_path,
                ]

            for module_name in _collinear_runtime_module_names():
                sys.modules.pop(module_name, None)

            _install_collinear_core_shims(injected_modules)
            importlib.invalidate_caches()
            if import_root is not None:
                module = importlib.import_module(verifier_module_path)
            else:
                try:
                    spec = importlib.util.spec_from_file_location(
                        verifier_module_path,
                        local_verifier_path,
                    )
                except (ImportError, ValueError) as e:
                    return VerifierResult(
                        success=False,
                        message=f"Failed to load verifier module {verifier_module_path}: {e}",
                        output=str(e),
                    )
                if spec is None or spec.loader is None:
                    return VerifierResult(
                        success=False,
                        message=(
                            f"Failed to load verifier module {verifier_module_path} "
                            f"from {local_verifier_path}"
                        ),
                    )
                module = importlib.util.module_from_spec(spec)
                sys.modules[verifier_module_path] = module
                spec.loader.exec_module(module)

            return _run_imported_verifier(module, verifier_module_path, run_artifacts_adapter)
        except Exception as e:
            return VerifierResult(
                success=False,
                message=f"Verifier error: {e}",
                output=str(e),
            )
        finally:
            importlib.invalidate_caches()
            for path in added_paths:
                if path in sys.path:
                    sys.path.remove(path)
            for module_name in _collinear_runtime_module_names():
                sys.modules.pop(module_name, None)
            for module_name in injected_modules:
                sys.modules.pop(module_name, None)
            sys.modules.update(original_runtime_modules)
            if original_module is None:
                sys.modules.pop(verifier_module_path, None)
            else:
                sys.modules[verifier_module_path] = original_module
            if original_collinear is None:
                sys.modules.pop("collinear", None)
            elif original_collinear_path is not None:
                original_collinear.__path__ = original_collinear_path
            if import_root is not None:
                shutil.rmtree(import_root, ignore_errors=True)

    use_cache = (
        scenario_manager_base_url is not None
        and _scenario_id_from_module_path(verifier_module_path) is not None
    )

    if use_cache:
        sid = _scenario_id_from_module_path(verifier_module_path)
        if sid is None:
            return VerifierResult(
                success=False,
                message=f"Cannot derive scenario from verifier path: {verifier_module_path}",
            )
        base_url = scenario_manager_base_url
        if base_url is None:
            return VerifierResult(
                success=False,
                message="Scenario manager base URL is required for cached verifier bundles",
            )
        try:
            cache_root = _ensure_verifier_bundle_cached(
                sid,
                base_url,
                api_key=scenario_manager_api_key,
                verifier_cache_root=verifier_cache_root,
            )
        except VerifierBundleError as e:
            return VerifierResult(success=False, message=str(e), output=str(e))
        sys.path.insert(0, str(cache_root))
        cached_injected_modules: list[str] = []
        try:
            # Let the bundle provide collinear.scenarios while we supply only the
            # minimal collinear.core modules verifiers expect at runtime.
            _install_collinear_core_shims(cached_injected_modules)

            try:
                module = importlib.import_module(verifier_module_path)
            except ImportError as e:
                return VerifierResult(
                    success=False,
                    message=f"Failed to import verifier module {verifier_module_path}: {e}",
                    output=str(e),
                )
            return _run_imported_verifier(module, verifier_module_path, run_artifacts_adapter)
        except Exception as e:
            return VerifierResult(success=False, message=f"Verifier error: {e}", output=str(e))
        finally:
            sys.path.pop(0)
            for _mod_name in cached_injected_modules:
                sys.modules.pop(_mod_name, None)

    # Direct import (monorepo or local)
    try:
        module = importlib.import_module(verifier_module_path)
    except ImportError as e:
        return VerifierResult(
            success=False,
            message=f"Failed to import verifier module {verifier_module_path}: {e}",
            output=str(e),
        )
    return _run_imported_verifier(module, verifier_module_path, run_artifacts_adapter)


# ---------------------------------------------------------------------------
# Rubric-based LLM judge
# ---------------------------------------------------------------------------

_RUBRIC_JUDGE_SYSTEM_PROMPT = dedent("""\
    You are an impartial evaluator assessing whether an AI agent successfully
    completed a task. Your evaluation will be used for reinforcement learning
    training. Accurate, evidence-based scoring is critical.

    # Instructions

    1. Read the task prompt, rubric, and the agent's rollout messages.
    2. Cross-reference the agent's actions with the rubric requirements.
       Look for concrete proof of completion.
    3. Score each dimension from the rubric independently, then compute an
       overall score.

    # Scoring Calibration

    - **0.8 - 1.0**: All requirements fully met with clear evidence.
    - **0.6 - 0.8**: Core requirements met with minor gaps. (0.6 = PASS threshold)
    - **0.4 - 0.6**: Partial completion, significant gaps remain.
    - **0.2 - 0.4**: Minimal progress, most requirements failed.
    - **0.0 - 0.2**: No meaningful progress or completely incorrect approach.

    # Principles

    - Focus on outcomes, not process.
    - Alternative correct approaches are valid.
    - Be strict on evidence: if you cannot confirm a requirement was met from
      the rollout trace, mark it as unmet.
""")

_RUBRIC_JUDGE_RESPONSE_SCHEMA = dedent("""\
    Respond with valid JSON using this schema (no markdown fences):
    {
      "verdict": "PASS or FAIL with a short headline",
      "score": 0.0-1.0,
      "confidence": 0.0-1.0,
      "evidence": ["Bullet list citing concrete evidence from the rollout"],
      "failed_criteria": ["List any unmet rubric criteria"],
      "dimension_scores": [
        {"dimension": "Name from rubric", "score": 0.0-1.0, "reason": "one-sentence evidence"}
      ]
    }
""")

_RUBRIC_JUDGE_MAX_MESSAGE_CHARS = 80_000


@dataclass
class RubricJudgeResult:
    """Structured result from the rubric-based LLM judge."""

    score: float
    verdict: str
    confidence: float
    evidence: list[str]
    failed_criteria: list[str]
    dimension_scores: list[dict[str, Any]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for reward.json."""
        d: dict[str, Any] = {
            "score": self.score,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "failed_criteria": self.failed_criteria,
            "dimension_scores": self.dimension_scores,
        }
        if self.error:
            d["error"] = self.error
        return d


def _truncate_messages_for_judge(
    messages: list[dict[str, Any]],
    max_chars: int = _RUBRIC_JUDGE_MAX_MESSAGE_CHARS,
) -> str:
    """Serialize messages to JSON, truncating tool results if needed."""
    raw = json.dumps(messages, indent=2, ensure_ascii=False)
    if len(raw) <= max_chars:
        return raw

    # Truncate long tool message content to fit budget
    truncated: list[dict[str, Any]] = []
    for msg in messages:
        entry = msg
        if isinstance(msg, dict) and msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 2000:
                entry = {
                    **msg,
                    "content": content[:1000] + "\n... [truncated] ...\n" + content[-500:],
                }
        truncated.append(entry)
    return json.dumps(truncated, indent=2, ensure_ascii=False)[:max_chars]


def run_rubric_judge(
    *,
    task_description: str,
    rubric_markdown: str,
    messages: list[dict[str, Any]],
    model: str,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    pass_threshold: float = 0.6,
) -> RubricJudgeResult:
    """Run an LLM judge that evaluates rollout messages against a rubric.

    Uses litellm for model-agnostic completion. Returns a structured result
    with score, verdict, evidence, and per-dimension breakdowns.
    """
    import litellm  # noqa: PLC0415

    rollout_text = _truncate_messages_for_judge(messages)
    user_prompt = dedent(f"""\
        # Task Prompt

        {task_description}

        # Rubric

        {rubric_markdown}

        # Agent Rollout Messages

        {rollout_text}

        # Response Format

        {_RUBRIC_JUDGE_RESPONSE_SCHEMA}
    """)

    litellm_model = model
    if provider and not model.startswith(f"{provider}/"):
        litellm_model = f"{provider}/{model}"

    try:
        response = litellm.completion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": _RUBRIC_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Rubric judge LLM call failed: %s", e)
        return RubricJudgeResult(
            score=0.0,
            verdict="ERROR",
            confidence=0.0,
            evidence=[],
            failed_criteria=[],
            dimension_scores=[],
            error=f"LLM call failed: {e}",
        )

    return _parse_judge_response(raw_text, pass_threshold=pass_threshold)


def _parse_judge_response(
    raw_text: str,
    pass_threshold: float = 0.6,
) -> RubricJudgeResult:
    """Parse the LLM judge JSON response into a RubricJudgeResult."""
    payload = _extract_judge_json(raw_text)
    if payload is None:
        return RubricJudgeResult(
            score=0.0,
            verdict="ERROR",
            confidence=0.0,
            evidence=[],
            failed_criteria=[],
            dimension_scores=[],
            error=f"Could not parse JSON from judge response: {raw_text[:500]}",
        )

    score = _safe_float(payload.get("score"), 0.0)
    confidence = _safe_float(payload.get("confidence"), 0.0)
    confidence = max(0.0, min(confidence, 1.0))
    verdict = str(payload.get("verdict") or ("PASS" if score >= pass_threshold else "FAIL"))
    evidence = _coerce_string_list(payload.get("evidence"))
    failed_criteria = _coerce_string_list(payload.get("failed_criteria"))
    dimension_scores = _coerce_dimension_scores_list(payload.get("dimension_scores"))

    return RubricJudgeResult(
        score=max(0.0, min(score, 1.0)),
        verdict=verdict,
        confidence=confidence,
        evidence=evidence,
        failed_criteria=failed_criteria,
        dimension_scores=dimension_scores,
    )


def _extract_judge_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM output (handles markdown fences)."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`\n")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    with suppress(json.JSONDecodeError):
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        with suppress(json.JSONDecodeError):
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
    return None


def _safe_float(value: object, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_string_list(value: object) -> list[str]:
    """Normalize a value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if item is not None]
    return [str(value).strip()]


def _coerce_dimension_scores_list(value: object) -> list[dict[str, Any]]:
    """Normalize dimension score payloads."""
    if not isinstance(value, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or item.get("name") or "").strip()
        if not dimension:
            continue
        score = _safe_float(item.get("score"))
        reason = str(item.get("reason") or item.get("rationale") or "").strip()
        entries.append(
            {"dimension": dimension, "score": max(0.0, min(score, 1.0)), "reason": reason}
        )
    return entries
