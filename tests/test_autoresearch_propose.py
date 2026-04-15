from __future__ import annotations

from simlab.autoresearch.propose import _extract_json


def test_extract_json_parses_strict_object() -> None:
    parsed = _extract_json('{"version":"0.1","iteration":1}')
    assert isinstance(parsed, dict)
    assert parsed.get("version") == "0.1"


def test_extract_json_supports_markdown_fence() -> None:
    parsed = _extract_json('```json\n{"version":"0.1","iteration":1}\n```')
    assert isinstance(parsed, dict)
    assert parsed.get("iteration") == 1


def test_extract_json_ignores_trailing_prose_objects() -> None:
    text = (
        "Here is the JSON:\n"
        '{"version":"0.1","iteration":1,"changes":{"scenario_prompt":"ok"}}\n'
        "And here is another brace block: {not json}.\n"
    )
    parsed = _extract_json(text)
    assert isinstance(parsed, dict)
    assert parsed.get("iteration") == 1


def test_extract_json_handles_braces_inside_strings() -> None:
    text = (
        "Result:\n"
        '{"version":"0.1","iteration":1,"changes":{"scenario_prompt":"x}y"}}'
        "\nTrailing brace } should not break extraction.\n"
    )
    parsed = _extract_json(text)
    assert isinstance(parsed, dict)
    assert parsed.get("iteration") == 1


def test_extract_json_repairs_newlines_inside_strings() -> None:
    text = '{"version":"0.1","iteration":1,"changes":{"scenario_prompt":"line 1\nline 2"}}'
    parsed = _extract_json(text)
    assert isinstance(parsed, dict)
    changes_obj = parsed.get("changes")
    assert isinstance(changes_obj, dict)
    assert changes_obj.get("scenario_prompt") == "line 1\nline 2"
