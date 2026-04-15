from __future__ import annotations

import pytest
from simlab.autoresearch.objectives import is_better_result


def test_is_better_result_max_objective() -> None:
    best = {"objective_value": 0.2, "reward_model_score_mean": 0.4, "tool_error_rate": 0.2}
    cand = {"objective_value": 0.3, "reward_model_score_mean": 0.1, "tool_error_rate": 0.9}
    assert is_better_result(candidate=cand, best=best, objective_type="pass_rate") is True


def test_is_better_result_min_objective() -> None:
    best = {"objective_value": 0.2, "reward_model_score_mean": 0.4, "tool_error_rate": 0.2}
    cand = {"objective_value": 0.1, "reward_model_score_mean": 0.1, "tool_error_rate": 0.9}
    assert is_better_result(candidate=cand, best=best, objective_type="tool_error_rate") is True


def test_is_better_result_tie_breaks_on_reward_model_then_tool_errors() -> None:
    best = {"objective_value": 0.2, "reward_model_score_mean": 0.4, "tool_error_rate": 0.2}
    cand = {"objective_value": 0.2, "reward_model_score_mean": 0.5, "tool_error_rate": 0.9}
    assert is_better_result(candidate=cand, best=best, objective_type="pass_rate") is True

    cand2 = {"objective_value": 0.2, "reward_model_score_mean": 0.4, "tool_error_rate": 0.1}
    assert is_better_result(candidate=cand2, best=best, objective_type="pass_rate") is True


def test_is_better_requires_objective_value() -> None:
    with pytest.raises(ValueError):
        is_better_result(
            candidate={"objective_value": None},
            best={"objective_value": 0.1},
            objective_type="pass_rate",
        )
