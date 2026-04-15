from __future__ import annotations

from simlab.runtime import rollout_runner


def test_extract_calendar_account_aliases_decodes_structured_content() -> None:
    resp = {
        "observation": {
            "is_error": False,
            "structured_content": {"accounts": [{"alias": "alex"}]},
        }
    }

    assert rollout_runner._extract_calendar_account_aliases(resp) == {"alex"}


def test_extract_calendar_account_aliases_parses_json_text() -> None:
    resp = {
        "observation": {
            "is_error": False,
            "text": '{"accounts": [{"alias": "pat"}]}',
        }
    }

    assert rollout_runner._extract_calendar_account_aliases(resp) == {"pat"}


def test_calendar_account_is_connected_reads_connected_flag() -> None:
    resp = {
        "observation": {
            "is_error": False,
            "structured_content": {"connected": True},
        }
    }

    assert rollout_runner._calendar_account_is_connected(resp) is True


def test_tool_response_is_error_allows_account_exists_error_code() -> None:
    resp = {
        "observation": {
            "is_error": True,
            "structured_content": {"error_code": "ACCOUNT_EXISTS"},
        }
    }

    assert (
        rollout_runner._tool_response_is_error(
            resp,
            allowed_error_codes={"ACCOUNT_EXISTS"},
        )
        is False
    )
