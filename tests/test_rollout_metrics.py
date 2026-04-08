from __future__ import annotations

from simlab.agents.rollout_metrics import extract_token_usage


def test_extract_token_usage_does_not_drop_zero_prompt_tokens() -> None:
    usage = {
        "prompt_tokens": 0,
        "input_tokens": 123,
        "completion_tokens": 1,
    }

    assert extract_token_usage(usage) == {
        "prompt_tokens": 0,
        "completion_tokens": 1,
        "total_tokens": 1,
    }


def test_extract_token_usage_does_not_drop_zero_completion_tokens() -> None:
    usage = {
        "prompt_tokens": 1,
        "completion_tokens": 0,
        "output_tokens": 999,
    }

    assert extract_token_usage(usage) == {
        "prompt_tokens": 1,
        "completion_tokens": 0,
        "total_tokens": 1,
    }
