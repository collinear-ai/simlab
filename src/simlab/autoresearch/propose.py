"""LLM-backed proposer for v1 prompt-only autoresearch."""

from __future__ import annotations

import json
from textwrap import dedent

import litellm

from simlab.autoresearch.config import ModelSection


def propose_next_change(
    *,
    proposer: ModelSection,
    iteration: int,
    analysis: dict[str, object],
    current_prompt: str,
    prompt_required_headings: list[str],
) -> dict[str, object]:
    """Propose a single scenario prompt change as strict JSON."""
    litellm_model = proposer.model
    if proposer.provider and not proposer.model.startswith(f"{proposer.provider}/"):
        litellm_model = f"{proposer.provider}/{proposer.model}"

    system_prompt = dedent(
        """\
        You are a research agent proposing ONE bounded edit to a runtime scenario prompt.

        You must:
        - Propose exactly one change that targets a repeated failure pattern in the analysis.
        - Keep the task set, environment, model, and evaluation fixed.
        - Return strict JSON only, no markdown fences.

        Prohibited:
        - Mentioning verifiers, evaluators, reward model internals, or "gaming" the score.
        - Changing tools, tasks, evaluation, or anything besides the scenario prompt text.
        - Returning multiple unrelated edits.
        """
    )

    required_headings_text = "\n".join(f"- {h}" for h in prompt_required_headings) or "- (none)"
    user_prompt = dedent(
        f"""\
        # Current prompt

        {current_prompt}

        # Required headings

        The candidate prompt must keep these headings as exact lines:
        {required_headings_text}

        # Analysis JSON

        {json.dumps(analysis, indent=2)}

        # Output schema

        Respond with JSON using this exact schema:
        {{
          "version": "0.1",
          "iteration": {iteration},
          "surface": "scenario_prompt",
          "rationale": "one short paragraph",
          "change_type": "short_snake_case_label",
          "changes": {{
            "scenario_prompt": "full candidate prompt markdown"
          }}
        }}
        """
    )

    response = litellm.completion(
        model=litellm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        api_key=proposer.resolve_api_key(),
        base_url=proposer.resolve_base_url(),
    )
    raw_text = response.choices[0].message.content or ""
    payload = _extract_json(raw_text)
    if payload is None:
        raise ValueError(f"Could not parse JSON from proposer response: {raw_text[:500]}")
    return payload


def _extract_json(text: str) -> dict[str, object] | None:
    candidate = _strip_markdown_fence(text)
    if candidate.lstrip().lower().startswith("json\n"):
        candidate = candidate.lstrip()[4:].lstrip()
    parsed = _try_parse_json_object(candidate)
    if parsed is not None:
        return parsed

    snippet = _extract_first_json_object_snippet(candidate)
    if snippet is None:
        return None
    return _try_parse_json_object(snippet)


def _try_parse_json_object(text: str) -> dict[str, object] | None:
    """Parse JSON text into an object, with a small repair pass for common LLM mistakes."""
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    repaired = _escape_control_chars_in_json_strings(text)
    if repaired == text:
        return None
    try:
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _escape_control_chars_in_json_strings(text: str) -> str:
    r"""Escape raw control characters inside JSON string literals.

    Some models emit multi-line strings with literal newlines or tabs inside a
    quoted value, which is invalid JSON. This pass rewrites those characters to
    \\n / \\r / \\t escapes without touching JSON structure outside strings.
    """
    in_string = False
    escaping = False
    out: list[str] = []

    for ch in text:
        if in_string:
            if escaping:
                escaping = False
                out.append(ch)
                continue
            if ch == "\\":
                escaping = True
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ch == '"':
                in_string = False
            out.append(ch)
            continue

        if ch == '"':
            in_string = True
        out.append(ch)

    return "".join(out)


def _extract_first_json_object_snippet(text: str) -> str | None:
    """Extract the first balanced {...} JSON object snippet from text.

    This is a last-resort fallback for cases where a model wraps JSON in extra prose.
    It is intentionally strict: it tracks nesting and ignores braces inside strings.
    """
    start = -1
    depth = 0
    in_string = False
    escaping = False

    for idx, ch in enumerate(text):
        if in_string:
            if escaping:
                escaping = False
                continue
            if ch == "\\":
                escaping = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue

        if ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : idx + 1]

    return None


def _strip_markdown_fence(text: str) -> str:
    """Remove a surrounding ```fence``` block when present.

    The proposer is instructed to return strict JSON only. This exists as a
    defensive fallback for models that still wrap JSON in markdown code fences.
    """
    candidate = (text or "").strip()
    if not candidate.startswith("```"):
        return candidate

    newline = candidate.find("\n")
    if newline == -1:
        return ""
    candidate = candidate[newline + 1 :]

    closing = candidate.rfind("```")
    if closing != -1 and candidate[closing + 3 :].strip() == "":
        candidate = candidate[:closing]
    return candidate.strip()
