"""Boundary validation for autoresearch proposals and scenario prompts."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

MAX_PROMPT_BYTES = 16_000
MAX_PROMPT_LINES = 400


class ProposalChanges(BaseModel):
    """The only supported change surface in v1."""

    model_config = {"extra": "forbid"}

    scenario_prompt: str = Field(alias="scenario_prompt", description="Full candidate prompt.")

    @field_validator("scenario_prompt")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("scenario_prompt must be a non-empty string")
        return value


class AutoresearchProposal(BaseModel):
    """Validated proposer output for a single iteration."""

    model_config = {"extra": "forbid"}

    version: str = Field(default="0.1")
    iteration: int = Field(ge=1)
    surface: str = Field(default="scenario_prompt")
    rationale: str = Field(default="")
    change_type: str | None = Field(default=None)
    changes: ProposalChanges

    @model_validator(mode="after")
    def _validate_surface(self) -> AutoresearchProposal:
        if self.surface != "scenario_prompt":
            raise ValueError("proposal.surface must be 'scenario_prompt' in v1")
        return self


def validate_scenario_prompt(
    prompt: str,
    *,
    required_headings: list[str] | None = None,
) -> None:
    """Validate a runtime scenario prompt against structural constraints."""
    if "\x00" in prompt:
        raise ValueError("scenario prompt contains null bytes")

    encoded = prompt.encode("utf-8", errors="strict")
    if len(encoded) > MAX_PROMPT_BYTES:
        raise ValueError(f"scenario prompt exceeds max size ({len(encoded)} > {MAX_PROMPT_BYTES})")

    line_count = len(prompt.splitlines())
    if line_count > MAX_PROMPT_LINES:
        raise ValueError(f"scenario prompt exceeds max lines ({line_count} > {MAX_PROMPT_LINES})")

    if required_headings:
        lines = {line.strip() for line in prompt.splitlines()}
        missing = [
            heading for heading in required_headings if heading.strip() and heading not in lines
        ]
        if missing:
            joined = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
            raise ValueError(f"scenario prompt is missing required headings: {joined}")


def parse_and_validate_proposal(
    payload: dict[str, object],
    *,
    required_headings: list[str] | None = None,
) -> AutoresearchProposal:
    """Parse and validate a proposer JSON payload."""
    proposal = AutoresearchProposal.model_validate(payload)
    validate_scenario_prompt(
        proposal.changes.scenario_prompt,
        required_headings=required_headings,
    )
    return proposal
