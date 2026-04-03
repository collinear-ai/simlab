"""Harbor-specific verifier execution helpers."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComposeExecResult:
    """Result from executing a command inside a compose service."""

    exit_code: int
    output: str


def build_harbor_test_verifier_results(
    reward_payload: dict[str, Any] | None,
    *,
    passed: bool | None = None,
) -> list[dict[str, Any]]:
    """Build synthetic verifier rows for Harbor ``test.sh`` reward payloads."""
    if not isinstance(reward_payload, dict):
        return []

    harbor_test = reward_payload.get("harbor_test_sh")
    harbor_test_dict = harbor_test if isinstance(harbor_test, dict) else {}
    structured_output = _extract_structured_harbor_output(reward_payload)
    if not harbor_test_dict and structured_output is None:
        return []

    if passed is None:
        reward_value = reward_payload.get("reward")
        passed = bool(reward_value >= 1.0) if isinstance(reward_value, int | float) else None
    if passed is None:
        exit_code = harbor_test_dict.get("exit_code")
        passed = bool(exit_code == 0) if isinstance(exit_code, int) else False

    detail = "Harbor test.sh passed" if passed else "Harbor test.sh failed"
    exit_code = harbor_test_dict.get("exit_code")
    if not passed and isinstance(exit_code, int):
        detail = f"Harbor test.sh failed (exit_code={exit_code})"

    output_payload: dict[str, Any]
    if structured_output is not None:
        output_payload = structured_output
    else:
        output_payload = {
            "name": "harbor_test_sh",
            "passed": passed,
            "detail": detail,
        }

    return [
        {
            "module": "harbor_test_sh",
            "success": passed,
            "message": detail,
            "output": json.dumps(output_payload),
        }
    ]


def _extract_structured_harbor_output(reward_payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("checks", "results", "criteria_results"):
        value = reward_payload.get(key)
        if isinstance(value, list):
            payload: dict[str, Any] = {key: [_normalize_harbor_check(item) for item in value]}
            for field in (
                "score",
                "max_score",
                "percentage",
                "earned",
                "possible",
                "detail",
                "message",
            ):
                if field in reward_payload:
                    payload[field] = reward_payload[field]
            return payload
    return None


def _normalize_harbor_check(item: object) -> object:
    if not isinstance(item, dict):
        return item

    normalized = dict(item)
    if "criteria" not in normalized and "check" in normalized:
        normalized["criteria"] = normalized["check"]
    if "pass" not in normalized and isinstance(normalized.get("passed"), bool):
        normalized["pass"] = normalized["passed"]
    if "detail" not in normalized:
        for field in ("description", "note", "message", "reason"):
            value = normalized.get(field)
            if value:
                normalized["detail"] = value
                break
    return normalized


def run_harbor_verifier(
    *,
    env_dir: Path,
    verifier_config: dict[str, Any],
    using_daytona: bool,
    daytona_client_factory: Any,  # noqa: ANN401
    daytona_api_key: str | None = None,
) -> tuple[bool, float, dict[str, Any], str]:
    """Run Harbor's ``tests/test.sh`` verifier and return result, payload, and stdout."""
    service = str(verifier_config.get("service") or "")
    workdir = str(verifier_config.get("workdir") or "/workspace")
    script_path = str(verifier_config.get("script_path") or "/tests/test.sh")
    env_overrides = resolve_runtime_env_values(verifier_config.get("env"))
    timeout_seconds = _timeout_seconds_from_config(
        verifier_config.get("timeout_sec"),
        default=900.0,
    )
    command = (
        "mkdir -p /logs/verifier && "
        f"chmod +x {shlex.quote(script_path)} && "
        f"cd {shlex.quote(workdir)} && "
        f"{shlex.quote(script_path)}"
    )

    exec_result = run_compose_exec(
        env_dir=env_dir,
        service=service,
        command=command,
        env_overrides=env_overrides,
        using_daytona=using_daytona,
        daytona_client_factory=daytona_client_factory,
        daytona_api_key=daytona_api_key,
        timeout=timeout_seconds,
    )
    reward_text = read_compose_file(
        env_dir=env_dir,
        service=service,
        file_path="/logs/verifier/reward.txt",
        using_daytona=using_daytona,
        daytona_client_factory=daytona_client_factory,
        daytona_api_key=daytona_api_key,
    )
    reward_json_raw = read_compose_file(
        env_dir=env_dir,
        service=service,
        file_path="/logs/verifier/reward.json",
        using_daytona=using_daytona,
        daytona_client_factory=daytona_client_factory,
        daytona_api_key=daytona_api_key,
    )

    reward = parse_reward_value(
        reward_text, reward_json_raw, fallback_exit_code=exec_result.exit_code
    )
    passed = reward >= 1.0
    payload: dict[str, Any]
    if reward_json_raw:
        try:
            parsed = json.loads(reward_json_raw)
        except json.JSONDecodeError:
            payload = {"reward": reward}
        else:
            payload = parsed if isinstance(parsed, dict) else {"reward": reward, "raw": parsed}
    else:
        payload = {"reward": reward}

    payload.setdefault("reward", reward)
    payload["harbor_test_sh"] = {
        "service": service,
        "workdir": workdir,
        "script_path": script_path,
        "exit_code": exec_result.exit_code,
        "output": exec_result.output,
    }
    payload.setdefault(
        "verifier_results",
        build_harbor_test_verifier_results(payload, passed=passed),
    )
    return passed, reward, payload, exec_result.output


def _timeout_seconds_from_config(raw: object, *, default: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return default
    return float(raw) if raw > 0 else default


def resolve_runtime_env_values(raw_env: object) -> dict[str, str]:
    """Resolve Harbor verifier environment values, interpolating ``${VAR}`` references."""
    if not isinstance(raw_env, dict):
        return {}
    resolved: dict[str, str] = {}
    for key, value in raw_env.items():
        text = str(value)
        match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", text)
        if match:
            resolved[str(key)] = os.environ.get(match.group(1), "")
        else:
            resolved[str(key)] = text
    return resolved


def run_compose_exec(
    *,
    env_dir: Path,
    service: str,
    command: str,
    env_overrides: dict[str, str],
    using_daytona: bool,
    daytona_client_factory: Any,  # noqa: ANN401
    daytona_api_key: str | None = None,
    timeout: float = 300,
) -> ComposeExecResult:
    """Run ``docker compose exec`` locally or inside Daytona."""
    if using_daytona:
        state = json.loads((env_dir / "daytona-state.json").read_text(encoding="utf-8"))
        sandbox = daytona_client_factory(daytona_api_key=daytona_api_key).get(state["sandbox_id"])
        exec_parts = ["docker", "compose", "exec", "-T"]
        daytona_timeout = _coerce_daytona_timeout_seconds(timeout)
        for key, value in env_overrides.items():
            exec_parts.extend(["-e", f"{key}={value}"])
        exec_parts.extend([service, "bash", "-lc", command])
        try:
            response = sandbox.process.exec(
                shlex.join(exec_parts),
                cwd="/home/daytona",
                timeout=daytona_timeout,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                return ComposeExecResult(
                    exit_code=1,
                    output=_format_timeout_message(daytona_timeout),
                )
            raise
        return ComposeExecResult(
            exit_code=getattr(response, "exit_code", 1),
            output=str(getattr(response, "result", "") or ""),
        )

    args = ["docker", "compose", "exec", "-T"]
    for key, value in env_overrides.items():
        args.extend(["-e", f"{key}={value}"])
    args.extend([service, "bash", "-lc", command])
    try:
        result = subprocess.run(  # noqa: S603
            args,
            cwd=env_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ComposeExecResult(
            exit_code=1,
            output=_format_timeout_message(timeout),
        )
    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    return ComposeExecResult(exit_code=result.returncode, output=combined_output)


def _format_timeout_message(timeout: float) -> str:
    return f"Command timed out after {timeout:.0f}s"


def _coerce_daytona_timeout_seconds(timeout: float) -> int:
    """Daytona's exec API expects an integer timeout in seconds."""
    return max(1, math.ceil(timeout))


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError | subprocess.TimeoutExpired):
        return True
    error_name = exc.__class__.__name__.lower()
    if "timeout" in error_name:
        return True
    return "timed out" in str(exc).lower()


def read_compose_file(
    *,
    env_dir: Path,
    service: str,
    file_path: str,
    using_daytona: bool,
    daytona_client_factory: Any,  # noqa: ANN401
    daytona_api_key: str | None = None,
) -> str | None:
    """Read a file from a compose service if it exists."""
    command = f"if [ -f {shlex.quote(file_path)} ]; then cat {shlex.quote(file_path)}; fi"
    result = run_compose_exec(
        env_dir=env_dir,
        service=service,
        command=command,
        env_overrides={},
        using_daytona=using_daytona,
        daytona_client_factory=daytona_client_factory,
        daytona_api_key=daytona_api_key,
        timeout=120,
    )
    content = result.output.strip()
    return content or None


def parse_reward_value(
    reward_text: str | None,
    reward_json_raw: str | None,
    *,
    fallback_exit_code: int,
) -> float:
    """Parse a Harbor reward value from ``reward.json`` or ``reward.txt``."""
    if reward_json_raw:
        try:
            parsed = json.loads(reward_json_raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            reward_value = parsed.get("reward")
            if isinstance(reward_value, (int, float)):
                return float(reward_value)
    if reward_text:
        stripped = reward_text.strip()
        if stripped in {"1", "1.0", "true", "True"}:
            return 1.0
        if stripped in {"0", "0.0", "false", "False"}:
            return 0.0
        try:
            return float(stripped)
        except ValueError:
            pass
    return 0.0 if fallback_exit_code else 1.0
