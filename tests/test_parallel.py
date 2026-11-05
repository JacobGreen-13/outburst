"""Ordered thread pools keep result order and stay within the worker cap."""

from __future__ import annotations

import time

import pytest

from outburst.parallel import preview_workers, run_ordered


def test_run_ordered_keeps_input_order_when_later_tasks_finish_first() -> None:
    seen: list[int] = []

    def work(index: int) -> int:
        if index == 0:
            time.sleep(0.05)
        return index

    results = run_ordered(5, work, workers=4, on_finished=lambda index, _value: seen.append(index))
    assert results == [0, 1, 2, 3, 4]
    assert seen == [0, 1, 2, 3, 4]


def test_preview_pool_stays_between_two_and_four_workers() -> None:
    assert preview_workers(1) == 1
    assert preview_workers(0) == 1
    assert 2 <= preview_workers(3) <= 3
    assert 2 <= preview_workers(12) <= 4


def test_cancel_runs_on_the_caller_before_the_pool_is_left_running() -> None:
    calls = {"count": 0}

    def cancel() -> None:
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("stop")

    def work(index: int) -> int:
        return index

    with pytest.raises(RuntimeError, match="stop"):
        run_ordered(6, work, workers=2, cancel=cancel)
