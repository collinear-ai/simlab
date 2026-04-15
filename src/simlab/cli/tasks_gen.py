"""CLI commands for task generation.

Provides init, validate, run, and status for server-side task generation
via the Scenario Manager API.
"""

from __future__ import annotations

import json
import shlex
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import click
import tomli_w

from simlab.api.client import ScenarioManagerApiError
from simlab.api.client import ScenarioManagerClient
from simlab.api.schemas import TaskGenJob
from simlab.api.schemas import TaskGenRequest
from simlab.api.schemas import TaskGenResult
from simlab.cli.tasks_gen_presets import PRESETS
from simlab.config import resolve_collinear_api_key
from simlab.config import resolve_scenario_manager_api_url
from simlab.telemetry import TelemetryCaptureConfig
from simlab.telemetry import emit_cli_event
from simlab.telemetry import resolve_scenario_manager_capture_config
from simlab.telemetry import with_command_telemetry

_POLL_INTERVAL_SECONDS = 3

# Step names reported by the pipeline, in expected order.
_STEP_NAMES = [
    "Generating skills document",
    "Step 1: Summarize seed data",
    "Step 2: Draft tasks",
    "Step 3: Refine tasks",
    "Step 3.5: Simplify tasks",
    "Step 4: Format tasks",
    "Step 5: Augment tasks",
    "Step 6: Filter tasks",
    "Step 7: Patch tasks",
    "Step 8: Generate rubrics",
    "Step 9: Generate verifiers",
    "Step 10: Refine rubrics",
]

_AVAILABLE_PRESETS = list(PRESETS.keys())


def _resolve_template(name: str) -> Path | None:
    """Resolve template TOML path from the packaged templates directory."""
    tdir = _templates_dir()
    if tdir is None:
        return None
    p = tdir / f"{name}.toml"
    return p if p.exists() else None


def _templates_dir() -> Path | None:
    """Find the task-gen templates directory (inside the simlab package)."""
    d = Path(__file__).parent.parent / "templates" / "task-gen"
    return d if d.is_dir() else None


@click.group("tasks-gen")
def tasks_gen() -> None:
    """Task generation — init, validate, run, and status."""


def task_gen_capture_config(
    ctx: click.Context | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TelemetryCaptureConfig:
    """Enable telemetry for task generation API flows with a Collinear API key."""
    _ = args
    server_url = kwargs.get("server_url")
    api_key = kwargs.get("api_key")
    return resolve_scenario_manager_capture_config(
        ctx,
        base_url=server_url if isinstance(server_url, str) else None,
        api_key=api_key if isinstance(api_key, str) else None,
    )


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@tasks_gen.command("templates")
def list_templates() -> None:
    """List available task generation templates."""
    tdir = _templates_dir()
    if not tdir:
        click.secho("No templates directory found.", fg="red")
        raise SystemExit(1)

    toml_files = sorted(tdir.glob("*.toml"))
    if not toml_files:
        click.secho("No templates available.", fg="yellow")
        return

    click.secho("Available templates:", fg="cyan", bold=True)
    click.echo()
    for f in toml_files:
        with f.open("rb") as fh:
            data = tomllib.load(fh)
        agent = data.get("agent", {})
        scenario = data.get("scenario", {})
        toolset = data.get("toolset", [])
        tools = [t.get("name", "") for t in toolset] if toolset else []
        desc = agent.get("description", scenario.get("name", ""))
        role = agent.get("role", "")

        click.secho(f"  {f.stem}", fg="green", bold=True)
        if role:
            click.echo(f"    Role:  {role}")
        if desc:
            click.echo(f"    Desc:  {desc[:100]}{'...' if len(desc) > 100 else ''}")
        if tools:
            click.echo(f"    Tools: {', '.join(tools)}")
        click.echo()

    click.echo("Use: simlab tasks-gen init --template <name>")


@tasks_gen.command()
@click.option(
    "--preset",
    type=click.Choice(_AVAILABLE_PRESETS),
    default=None,
    help="Use a preset to skip interactive prompts",
)
@click.option(
    "--template",
    type=str,
    default=None,
    help="Template name (e.g., hr, coding, crm, blank)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=".",
    show_default=True,
    help="Directory to write the generated config.toml",
)
@with_command_telemetry("tasks-gen init", resolver=task_gen_capture_config)
def init(preset: str | None, template: str | None, output_dir: str) -> None:
    r"""Generate an editable config.toml for task generation.

    Use --preset for a quick start with sensible defaults, --template to copy
    a template TOML file, or run without either to create a minimal template.

    \b
    Examples:
      simlab tasks-gen init --preset hr --output-dir ./taskgen
      simlab tasks-gen init --template coding --output-dir ./taskgen
      simlab tasks-gen init --output-dir ./taskgen
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "config.toml"

    if template:
        template_path = _resolve_template(template)
        if template_path is None:
            click.secho(f"Template '{template}' not found.", fg="red")
            sys.exit(1)
        import shutil  # noqa: PLC0415

        shutil.copy(template_path, config_path)
    elif preset and preset in PRESETS:
        config_dict = {**PRESETS[preset], "preset": preset}
        with config_path.open("wb") as f:
            tomli_w.dump(config_dict, f)
    else:
        config_dict = {
            "agent": {
                "role": "Your agent role here",
                "description": "What this agent does end-to-end",
            },
            "toolset": [
                {
                    "name": "ToolName",
                    "description": "What this tool does",
                    "operations": ["read", "write"],
                },
            ],
            "scenario": {
                "name": "",
                "role_label": "",
                "conventions": "",
                "policies": [],
            },
            "generation": {
                "num_tasks": 10,
                "complexity": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
            },
            "pipeline": {
                "model": "claude-haiku-4-5",
            },
        }
        with config_path.open("wb") as f:
            tomli_w.dump(config_dict, f)

    click.echo()
    click.secho("Setup complete!", fg="green", bold=True)
    click.echo(f"  Config: {config_path}")
    click.echo()
    click.echo("Next steps:")
    click.echo("  1. Review and edit the generated config file")
    click.echo(f"  2. Validate: simlab tasks-gen validate {config_path}")
    click.echo(f"  3. Run:      simlab tasks-gen run --config {config_path}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@tasks_gen.command()
@click.argument("config_path", type=click.Path(exists=True))
@with_command_telemetry("tasks-gen validate", resolver=task_gen_capture_config)
def validate(config_path: str) -> None:
    r"""Validate a config.toml locally before submitting.

    Parses the TOML and checks it against the TaskGenRequest schema.

    \b
    Example:
      simlab tasks-gen validate ./taskgen/config.toml
    """
    try:
        request_kwargs = _load_request_inputs(config_path, None)
    except SystemExit:
        raise
    except Exception as e:
        click.secho(f"Config parsing failed: {e}", fg="red")
        sys.exit(1)

    request_kwargs.setdefault("num_tasks", 10)
    request_kwargs.setdefault("model", "claude-haiku-4-5")

    try:
        request = TaskGenRequest(**request_kwargs)
    except Exception as e:
        click.secho(f"Config validation failed: {e}", fg="red")
        sys.exit(1)

    click.secho("Schema: OK", fg="green")

    click.echo()
    click.secho("Config Summary:", fg="cyan", bold=True)
    if request.agent:
        click.echo(f"  Agent:       {request.agent.role}")
    click.echo(f"  Toolset:     {len(request.toolset)} tool definitions")
    click.echo(f"  Tasks:       {request.num_tasks}")
    click.echo(f"  Model:       {request.model}")
    if request.complexity:
        click.echo("  Complexity:  " + " ".join(f"{k}={v}" for k, v in request.complexity.items()))
    if request.scenario and request.scenario.name:
        click.echo(f"  Scenario:    {request.scenario.name}")
    if request.preset:
        click.echo(f"  Preset:      {request.preset}")
    click.echo()
    click.secho("Config is valid!", fg="green", bold=True)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@tasks_gen.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config TOML with toolset, agent, scenario, etc.",
)
@click.option(
    "--describe",
    type=str,
    default=None,
    help="Vague description of what to test",
)
@click.option(
    "--from-schema",
    "schema_path",
    type=click.Path(exists=True),
    default=None,
    help='Tool schema JSON (OpenAI format: {"tools": [...]}, or plain list of tool defs)',
)
@click.option(
    "--tools",
    "tools_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to MCP tool definitions JSON file (array of tool objects)",
)
@click.option(
    "--out",
    "output_dir",
    type=click.Path(),
    default="./generated-tasks",
    show_default=True,
    help="Output directory for the generated task bundle",
)
@click.option(
    "--num-tasks",
    type=int,
    default=None,
    help="Number of tasks to generate (default: 10, or from config)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model for task generation (default: claude-haiku-4-5, or from config)",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show detailed output for debugging",
)
@click.pass_context
@with_command_telemetry("tasks-gen run", resolver=task_gen_capture_config)
def run(
    ctx: click.Context,
    config_path: str | None,
    describe: str | None,
    schema_path: str | None,
    tools_path: str | None,
    output_dir: str,
    num_tasks: int | None,
    model: str | None,
    verbose: bool,
) -> None:
    """Submit a task generation request and stream progress until done.

    Provide either --config (TOML with full scenario config), --tools (JSON
    array of MCP tool definitions), or --from-schema (tool schema JSON).
    CLI flags override config values.
    """
    # 1. Load config TOML, tools JSON, or schema JSON
    if schema_path:
        with Path(schema_path).open() as f:
            tool_data = json.load(f)
        # Normalize to list of tool defs
        if isinstance(tool_data, dict) and "tools" in tool_data:
            tool_defs = [t.get("function", t) for t in tool_data["tools"]]
        elif isinstance(tool_data, list):
            tool_defs = tool_data
        else:
            tool_defs = [tool_data]

        # Validate that each tool def has at least a name
        for i, td in enumerate(tool_defs):
            if not isinstance(td, dict) or "name" not in td:
                raise click.BadParameter(
                    f"Tool #{i} must be a dict with a 'name' key, got: {td!r:.80}",
                    param_hint="--from-schema",
                )

        request_kwargs: dict[str, Any] = {
            "toolset": tool_defs,
            "num_tasks": num_tasks or 20,
            "model": model or "claude-haiku-4-5-20251001",
        }
    else:
        request_kwargs = _load_request_inputs(config_path, tools_path)

    # Inject description if provided via --describe
    if describe:
        request_kwargs["description"] = describe

    # CLI flags override config values.
    if num_tasks is not None:
        request_kwargs["num_tasks"] = num_tasks
    if model is not None:
        request_kwargs["model"] = model

    # Apply defaults for fields not set by config or flags.
    request_kwargs.setdefault("num_tasks", 10)
    request_kwargs.setdefault("model", "claude-haiku-4-5")

    # 2. Build request
    try:
        request = TaskGenRequest(**request_kwargs)
    except Exception as exc:
        emit_cli_event(
            "task_gen_run_failed",
            {
                "failure_stage": "build_request",
                "input_source": "config" if config_path else "tools",
                "verbose": verbose,
            },
        )
        click.secho(f"Invalid request parameters: {exc}", fg="red")
        sys.exit(1)

    # 3. Submit to server
    client = _scenario_manager_client_from_ctx(ctx)

    scenario_label = ""
    if request.scenario and request.scenario.name:
        scenario_label = f" for scenario: {request.scenario.name}"
    click.echo()
    click.secho(
        f"  Generating {request.num_tasks} tasks{scenario_label}",
        fg="cyan",
        bold=True,
    )
    click.echo()

    if verbose:
        click.echo(f"  Server:    {client.base_url}")
        click.echo(f"  Toolset:   {len(request.toolset)} tool definitions")
        click.echo(f"  Model:     {request.model}")
        if request.agent:
            click.echo(f"  Agent:     {request.agent.role}")
        if request.preset:
            click.echo(f"  Preset:    {request.preset}")
        click.echo()

    try:
        job = client.submit_task_gen(request)
    except ScenarioManagerApiError as exc:
        emit_cli_event(
            "task_gen_run_failed",
            {
                **task_gen_request_properties(request, config_path, tools_path, verbose),
                "failure_stage": "submit",
                "status_code": exc.status_code,
            },
        )
        _print_api_error(exc)
        sys.exit(1)

    if verbose:
        click.echo(f"  Job ID: {job.job_id}")

    # 4. Poll with live progress
    try:
        job = _poll_with_progress(client, job.job_id)
    except ScenarioManagerApiError as exc:
        emit_cli_event(
            "task_gen_run_failed",
            {
                **task_gen_request_properties(request, config_path, tools_path, verbose),
                "failure_stage": "poll",
                "status_code": exc.status_code,
            },
        )
        _print_api_error(exc)
        sys.exit(1)
    except KeyboardInterrupt:
        emit_cli_event(
            "task_gen_run_failed",
            {
                **task_gen_request_properties(request, config_path, tools_path, verbose),
                "failure_stage": "interrupted",
                "job_id_present": bool(job.job_id),
            },
        )
        click.echo()
        click.secho("  Interrupted. Job may still be running on the server.", fg="yellow")
        if verbose:
            click.echo(f"  Job ID: {job.job_id}")
        sys.exit(130)

    if job.status == "failed":
        emit_cli_event(
            "task_gen_run_failed",
            {
                **task_gen_request_properties(request, config_path, tools_path, verbose),
                "failure_stage": "job_failed",
            },
        )
        click.secho(f"  x  Failed: {job.error or 'unknown error'}", fg="red")
        sys.exit(1)

    # 5. Download results
    try:
        result = client.get_task_gen_results(job.job_id)
    except ScenarioManagerApiError as exc:
        emit_cli_event(
            "task_gen_run_failed",
            {
                **task_gen_request_properties(request, config_path, tools_path, verbose),
                "failure_stage": "download_results",
                "status_code": exc.status_code,
            },
        )
        _print_api_error(exc)
        sys.exit(1)

    # 6. Write to output directory
    out_path = Path(output_dir)
    _write_bundle(out_path, result)

    # Report verifier coverage
    _report_verifier_coverage(result)

    click.echo()
    task_count = len(result.tasks)
    if task_count < request.num_tasks:
        click.secho(
            f"  Done — {task_count} of {request.num_tasks} requested tasks"
            " survived quality filtering,",
            fg="yellow",
            bold=True,
        )
        click.secho(
            f"  written to {out_path.resolve()}/",
            fg="yellow",
            bold=True,
        )
        _print_filter_summary(result.filter_summary)
        click.echo()
        click.echo("  To get more tasks, try:")
        click.echo("    - Increase num_tasks in your config (e.g. 2-3x your target)")
        click.echo("    - Reduce the number of categories")
        click.echo("    - Set filter = false under [generation] to skip quality checks")
    else:
        click.secho(
            f"  Done — {task_count} tasks generated, written to {out_path.resolve()}/",
            fg="green",
            bold=True,
        )
    if verbose:
        click.echo(f"    Tasks:        {len(result.tasks)}")
        click.echo(f"    Instructions: {len(result.instructions)}")
        click.echo(f"    Rubrics:      {len(result.rubrics)}")
        click.echo(f"    Verifiers:    {len(result.verifiers)}")
    safe_out = shlex.quote(str(out_path.resolve()))
    click.echo()
    click.echo(
        click.style(f"Next: simlab tasks list --tasks-dir {safe_out}", bold=True),
        err=True,
    )
    click.echo(
        click.style(
            f"  Then: simlab tasks run --env <env_name> --tasks-dir {safe_out}"
            " --task <task_id> --agent-model <model>",
            dim=True,
        ),
        err=True,
    )
    emit_cli_event(
        "task_gen_run_completed",
        {
            **task_gen_request_properties(request, config_path, tools_path, verbose),
            "job_status": job.status,
            "generated_task_count": len(result.tasks),
            "generated_instruction_count": len(result.instructions),
            "generated_rubric_count": len(result.rubrics),
            "generated_verifier_count": len(result.verifiers),
            "generated_npc_count": len(result.npcs),
        },
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@tasks_gen.command()
@click.argument("job_id")
@click.option(
    "--api-key",
    "api_key",
    type=str,
    default=None,
    help="API key for authentication (overrides global config and COLLINEAR_API_KEY)",
)
@click.option(
    "--server-url",
    type=str,
    default=None,
    help=(
        "Scenario Manager API URL (overrides global config and COLLINEAR_SCENARIO_MANAGER_API_URL)"
    ),
)
@click.pass_context
@with_command_telemetry("tasks-gen status", resolver=task_gen_capture_config)
def status(
    ctx: click.Context,
    job_id: str,
    api_key: str | None,
    server_url: str | None,
) -> None:
    """Fetch the current status of a task generation job."""
    client = _scenario_manager_client_from_ctx(ctx, api_key=api_key, server_url=server_url)
    try:
        job = client.get_task_gen_status(job_id)
    except ScenarioManagerApiError as exc:
        _print_api_error(exc)
        sys.exit(1)

    click.echo()
    click.echo(f"  Job ID:   {job.job_id}")
    click.echo(f"  Status:   {job.status}")
    if job.progress:
        click.echo(f"  Progress: {job.progress}")
    if job.error:
        click.echo(f"  Error:    {job.error}")
    click.echo()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scenario_manager_client_from_ctx(
    ctx: click.Context,
    *,
    api_key: str | None = None,
    server_url: str | None = None,
) -> ScenarioManagerClient:
    resolved_url = resolve_scenario_manager_api_url(base_url=server_url, ctx=ctx)
    resolved_api_key = resolve_collinear_api_key(api_key, ctx=ctx)
    return ScenarioManagerClient(base_url=resolved_url, api_key=resolved_api_key)


def task_gen_request_properties(
    request: TaskGenRequest,
    config_path: str | None,
    tools_path: str | None,
    verbose: bool,
) -> dict[str, Any]:
    """Return sanitized telemetry properties for one task generation request."""
    _ = tools_path
    properties: dict[str, Any] = {
        "input_source": "config" if config_path else "tools",
        "tool_count": len(request.toolset),
        "requested_task_count": request.num_tasks,
        "has_scenario_context": bool(request.scenario),
        "verbose": verbose,
    }
    if request.preset:
        properties["preset"] = request.preset
    return properties


def _load_request_inputs(
    config_path: str | None,
    tools_path: str | None,
) -> dict[str, Any]:
    """Load request fields from --config TOML or --tools JSON.

    Returns a dict of kwargs suitable for ``TaskGenRequest(**kwargs)``.
    """
    if config_path is None and tools_path is None:
        click.secho("Provide either --config (TOML) or --tools (JSON).", fg="red")
        sys.exit(1)

    if config_path and tools_path:
        click.secho("Provide --config or --tools, not both.", fg="red")
        sys.exit(1)

    # --tools: simple JSON array of MCP tool definitions.
    if tools_path:
        tools_file = Path(tools_path)
        try:
            tools_data = json.loads(tools_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            click.secho(f"Invalid JSON in {tools_file}: {exc}", fg="red")
            sys.exit(1)

        if not isinstance(tools_data, list) or not tools_data:
            click.secho(
                f"Expected a non-empty JSON array of tool definitions in {tools_file}",
                fg="red",
            )
            sys.exit(1)
        return {"toolset": tools_data}

    # --config: full TOML with toolset, agent, scenario, etc.
    cfg_file = Path(config_path)  # type: ignore[arg-type]
    try:
        with cfg_file.open("rb") as f:
            cfg = tomllib.load(f)
    except Exception as exc:
        click.secho(f"Failed to read config TOML: {exc}", fg="red")
        sys.exit(1)

    tools_data = cfg.get("toolset", [])
    if not tools_data:
        click.secho(f"No [[toolset]] found in {cfg_file}", fg="red")
        sys.exit(1)

    kwargs: dict[str, Any] = {"toolset": tools_data}

    # Map TOML sections to request fields.
    if "agent" in cfg:
        kwargs["agent"] = cfg["agent"]
    if "scenario" in cfg:
        kwargs["scenario"] = cfg["scenario"]
    if "workspace" in cfg:
        kwargs["workspace"] = cfg["workspace"]
    if "categories" in cfg:
        kwargs["categories"] = cfg["categories"]
    if "workflows" in cfg:
        kwargs["workflows"] = cfg["workflows"]
    if "npcs" in cfg:
        kwargs["npcs"] = cfg["npcs"]
    if "preset" in cfg:
        kwargs["preset"] = cfg["preset"]

    # Generation section.
    gen = cfg.get("generation", {})
    if "num_tasks" in gen:
        kwargs["num_tasks"] = gen["num_tasks"]
    if "complexity" in gen:
        kwargs["complexity"] = gen["complexity"]
    if "diversity" in gen:
        kwargs["diversity"] = gen["diversity"]
    if "filter" in gen:
        kwargs["filter"] = gen["filter"]

    # Pipeline section — model.
    pipeline = cfg.get("pipeline", {})
    if "model" in pipeline:
        kwargs["model"] = pipeline["model"]

    return kwargs


def _poll_with_progress(
    client: ScenarioManagerClient,
    job_id: str,
) -> TaskGenJob:
    """Poll job status with live progress display."""
    completed_steps: list[str] = []
    last_progress = ""
    current_line_written = False

    while True:
        job = client.get_task_gen_status(job_id)
        progress = job.progress or ""

        if progress and progress != last_progress:
            # Mark previous in-progress step as done
            if current_line_written:
                # Overwrite the current "in progress" line with a checkmark
                click.echo("\r" + " " * 80 + "\r", nl=False)
                if last_progress and last_progress != "Preparing pipeline":
                    click.secho(f"  \u2713 {last_progress}", fg="green")
                    completed_steps.append(last_progress)

            last_progress = progress

            if progress == "Done":
                current_line_written = False
            else:
                # Show current step as in-progress
                click.echo(f"  \u25cb {progress}...", nl=False)
                current_line_written = True
                sys.stdout.flush()

        if job.status in ("completed", "failed"):
            if current_line_written:
                click.echo("\r" + " " * 80 + "\r", nl=False)
                if last_progress and last_progress != "Done":
                    if job.status == "completed":
                        click.secho(f"  \u2713 {last_progress}", fg="green")
                    else:
                        click.secho(f"  x  {last_progress}", fg="red")
            return job

        time.sleep(_POLL_INTERVAL_SECONDS)


def _report_verifier_coverage(result: TaskGenResult) -> None:
    """Warn when the generated bundle has tasks without verifier wiring."""
    task_count = len(result.tasks)
    if task_count == 0:
        return

    # Count failed verifier files (.FAILED.py suffix from older pipelines)
    failed_files = [f for f in result.verifiers if f.filename.endswith(".FAILED.py")]

    # Count tasks with verifier wiring
    wired = 0
    for bundle_file in result.tasks:
        if not bundle_file.filename.endswith(".json"):
            continue
        try:
            task_data = json.loads(bundle_file.content)
        except (json.JSONDecodeError, ValueError):
            continue
        has_verifier = (
            task_data.get("verifiers")
            or task_data.get("evaluators")
            or task_data.get("harbor_verifier")
        )
        if has_verifier:
            wired += 1

    missing = task_count - wired
    if missing == 0 and not failed_files:
        return

    parts: list[str] = []
    if missing:
        parts.append(f"{missing}/{task_count} task(s) have no verifier wiring")
    if failed_files:
        parts.append(f"{len(failed_files)} verifier(s) failed quality gates")
    click.echo()
    click.secho(
        "  Warning: " + "; ".join(parts) + ". Unverified tasks will not be scored.",
        fg="yellow",
    )


def _write_bundle(out_path: Path, result: TaskGenResult) -> None:
    """Write task bundle files to the output directory."""
    for subdir, files in [
        ("tasks", result.tasks),
        ("instructions", result.instructions),
        ("rubrics", result.rubrics),
        ("verifiers", result.verifiers),
        ("npcs", result.npcs),
    ]:
        dir_path = out_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        for bundle_file in files:
            safe_name = Path(bundle_file.filename).name
            if not safe_name:
                continue
            file_path = dir_path / safe_name
            file_path.write_text(bundle_file.content, encoding="utf-8")

    if result.skills_md:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "skills.md").write_text(result.skills_md, encoding="utf-8")


_CHECK_LABELS: dict[str, str] = {
    "required_npcs_covered": "NPC coverage",
    "message_info_secret_coverage": "NPC secret coverage",
    "seed_email_quality": "Seed email quality",
    "seed_calendar_quality": "Seed calendar quality",
    "seed_chat_channel_quality": "Seed chat channel quality",
    "task_quality": "Task quality",
}


def _print_filter_summary(summary: dict | None) -> None:
    """Print a concise breakdown of filtered tasks and which checks they failed."""
    if not summary:
        return

    filtered_tasks = summary.get("filtered_tasks", [])
    if not filtered_tasks:
        return

    click.echo()
    click.secho("  Filtered tasks:", bold=True)
    for entry in filtered_tasks:
        name = entry.get("display_name") or entry.get("file", "unknown")
        checks = entry.get("checks", {})
        failed = [_CHECK_LABELS.get(k, k) for k, v in checks.items() if v == 0]
        issues = entry.get("issues", [])
        reason = "; ".join(issues) if issues else ", ".join(failed)
        click.echo(f"    \u2717 {name}")
        if reason:
            click.echo(f"      {reason}")

    # Show stage-level funnel if available
    stage_stats = summary.get("stage_stats")
    total = summary.get("total")
    if stage_stats and total:
        click.echo()
        click.secho("  Filter funnel:", bold=True)
        click.echo(f"    Started with {total} tasks")
        for key in ("check1", "check2", "check3", "check4", "check5", "check6"):
            stage = stage_stats.get(key)
            if not stage:
                continue
            check_key = stage.get("check_key", key)
            label = _CHECK_LABELS.get(check_key, check_key)
            filtered = stage.get("filtered", 0)
            passed = stage.get("passed", 0)
            if filtered:
                click.echo(f"    {label}: {filtered} filtered, {passed} passed")


def _print_api_error(exc: ScenarioManagerApiError) -> None:
    """Print a user-friendly API error message."""
    if exc.status_code == 0:
        click.secho(f"Connection error: {exc.detail}", fg="red")
        click.echo("  Check that the server URL is correct and reachable.")
    elif exc.status_code == 401:
        click.secho("Authentication failed.", fg="red")
        click.echo(
            "  Get your API key at https://platform.collinear.ai (Developer Resources → API Keys)."
        )
        click.echo("  Then run: simlab auth login")
    elif exc.status_code == 404:
        click.secho(f"Not found: {exc.detail}", fg="red")
    else:
        click.secho(f"API error (HTTP {exc.status_code}): {exc.detail}", fg="red")
