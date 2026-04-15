from __future__ import annotations

from simlab.autoresearch.manager import should_stop_no_improvement


def test_should_stop_no_improvement_window_zero_disables_early_stop() -> None:
    assert should_stop_no_improvement(rejected_streak=0, window=0) is False
    assert should_stop_no_improvement(rejected_streak=1, window=0) is False


def test_should_stop_no_improvement_window_positive_counts_rejections() -> None:
    assert should_stop_no_improvement(rejected_streak=0, window=2) is False
    assert should_stop_no_improvement(rejected_streak=1, window=2) is False
    assert should_stop_no_improvement(rejected_streak=2, window=2) is True
