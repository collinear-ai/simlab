"""Request/response schemas for the task-gen API."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class TaskGenAgentSpec(BaseModel):
    """Agent identity for task generation."""

    role: str = Field(..., description="Agent role, e.g. 'HR recruiting coordinator'")
    description: str = Field(..., description="What the agent handles end-to-end")


class TaskGenScenarioSpec(BaseModel):
    """Scenario context for task generation."""

    name: str = Field("", description="Scenario identifier, e.g. 'recruiting'")
    role_label: str = Field("", description="Human-readable role label")
    conventions: str = Field("", description="Free-text scenario conventions")
    policies: list[str] = Field(default_factory=list, description="Specific policy rules")


class TaskGenWorkspaceSpec(BaseModel):
    """Workspace branding for task generation."""

    email_domain: str = Field("example.com", description="Company email domain")
    agent_email: str = Field("agent@example.com", description="Agent's email address")


class TaskGenRequest(BaseModel):
    """POST /task-gen — submit toolset for server-side task generation."""

    # --- Scenario specification (client provides) ---
    toolset: list[dict[str, Any]] = Field(
        ..., min_length=1, description="MCP tool definitions (JSON from /tools endpoint)"
    )
    agent: TaskGenAgentSpec | None = Field(default=None, description="Agent role and description")
    scenario: TaskGenScenarioSpec | None = Field(
        default=None, description="Scenario conventions and policies"
    )
    categories: list[dict[str, str]] | None = Field(
        default=None,
        description='Task categories, e.g. [{"id": "scheduling", "label": "Scheduling"}]',
    )
    workflows: list[dict[str, Any]] | None = Field(
        default=None,
        description='Example workflows, e.g. [{"name": "...", "steps": ["...", "..."]}]',
    )
    npcs: list[dict[str, str]] | None = Field(
        default=None,
        description='NPC roles, e.g. [{"role": "...", "typical_asks": "..."}]',
    )
    workspace: TaskGenWorkspaceSpec | None = Field(
        default=None, description="Workspace branding (email domain, agent email)"
    )

    # --- Generation preferences ---
    num_tasks: int = Field(10, ge=1, le=200, description="Number of tasks to generate")
    model: str = Field("claude-sonnet-4-6", description="LLM model for generation")
    complexity: dict[str, float] | None = Field(
        default=None,
        description='Complexity distribution, e.g. {"easy": 0.3, "medium": 0.5, "hard": 0.2}',
    )
    diversity: dict[str, Any] | None = Field(
        default=None,
        description="Diversity config: variations list and dimensions",
    )
    filter: bool = Field(
        default=True, description="Enable quality filtering of generated tasks (step 6/7)"
    )
    preset: str | None = Field(
        default=None, description='Use a built-in preset, e.g. "recruiting" or "people_mgmt"'
    )


class TaskGenJob(BaseModel):
    """Status of a task-gen job."""

    job_id: str = Field(..., description="Unique job identifier")
    status: Literal["pending", "running", "completed", "failed"] = Field(
        ..., description="Current job status"
    )
    progress: str | None = Field(
        default=None, description="Human-readable progress, e.g. 'Step 4/10: formatting tasks'"
    )
    error: str | None = Field(default=None, description="Error message if status is failed")
    created_at: str | None = Field(default=None, description="ISO-8601 creation timestamp")
    completed_at: str | None = Field(default=None, description="ISO-8601 completion timestamp")


class TaskBundleFile(BaseModel):
    """A single file within a generated task bundle."""

    filename: str = Field(..., description="Relative path within the bundle")
    content: str = Field(..., description="File content (JSON string for .json, raw for .md)")


class TaskGenResult(BaseModel):
    """GET /task-gen/{job_id}/results — generated task bundle."""

    job_id: str = Field(..., description="Job that produced these results")
    tasks: list[TaskBundleFile] = Field(default_factory=list, description="Generated task JSONs")
    instructions: list[TaskBundleFile] = Field(
        default_factory=list, description="Generated instruction markdowns"
    )
    rubrics: list[TaskBundleFile] = Field(
        default_factory=list, description="Generated rubric files"
    )
    verifiers: list[TaskBundleFile] = Field(
        default_factory=list, description="Generated programmatic verifier files"
    )
    npcs: list[TaskBundleFile] = Field(
        default_factory=list, description="Generated NPC profile files"
    )
    skills_md: str | None = Field(default=None, description="Generated skills.md content")
    filter_summary: dict | None = Field(
        default=None,
        description="Quality filter summary (stage stats, filtered task details)",
    )


class ScenarioToolServer(BaseModel):
    """Tool server metadata used by scenario/task listing APIs."""

    name: str = Field(..., description="Tool server name")
    server_type: str | None = Field(default=None, description="Tool server type/category")
    tool_server_url: str | None = Field(default=None, description="Resolved tool server URL")


class ScenarioSummary(BaseModel):
    """Summary row returned by GET /scenarios."""

    scenario_id: str = Field(..., description="Scenario slug/id")
    name: str = Field(..., description="Human-readable scenario name")
    description: str | None = Field(default=None, description="Scenario description")
    num_tasks: int = Field(default=0, description="Task count")
    num_npcs: int = Field(default=0, description="NPC profile count")
    tool_servers: list[ScenarioToolServer] = Field(
        default_factory=list, description="Scenario tool servers"
    )
    scenario_guidance_md: str | None = Field(
        default=None,
        description="Scenario guidance markdown to materialize into environment/task prompts",
    )
    visibility: str | None = Field(default=None, description="Visibility marker")

    @field_validator("tool_servers", mode="before")
    @classmethod
    def _none_tool_servers_to_empty_list(cls, value: object) -> object:
        return [] if value is None else value


class ScenarioTask(BaseModel):
    """Task definition row returned by GET /scenarios/{id}/tasks."""

    task_id: str = Field(..., description="Task id")
    name: str = Field(default="", description="Display name")
    description: str = Field(default="", description="Task description/prompt")
    difficulty: str | None = Field(default=None, description="Task difficulty")
    category: str | None = Field(default=None, description="Task category")
    tool_servers: list[ScenarioToolServer] = Field(
        default_factory=list, description="Task tool servers"
    )
    verifier_modules: list[str] = Field(default_factory=list, description="Verifier module paths")
    seed_emails: list[dict[str, Any]] = Field(default_factory=list, description="Seed emails")
    seed_calendar_events: list[dict[str, Any]] = Field(
        default_factory=list, description="Seed calendar events"
    )
    npc_profiles: list[dict[str, Any]] = Field(default_factory=list, description="NPC profiles")

    @field_validator(
        "tool_servers",
        "verifier_modules",
        "seed_emails",
        "seed_calendar_events",
        "npc_profiles",
        mode="before",
    )
    @classmethod
    def _none_lists_to_empty_lists(cls, value: object) -> object:
        return [] if value is None else value


class ScenarioTasksResponse(BaseModel):
    """Envelope returned by GET /scenarios/{id}/tasks."""

    scenario_id: str = Field(..., description="Scenario slug/id")
    tasks: list[ScenarioTask] = Field(default_factory=list, description="Scenario tasks")
