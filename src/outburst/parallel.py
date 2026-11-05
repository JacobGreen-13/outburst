"""Small thread pools for work that waits on disk or OpenCV.

The window stays on its own thread. These helpers run a bounded pool and
hand results back in input order, on the thread that asked for the work.
SQLite writes stay on that thread. Two to four workers is the cap for
preview scoring, because OpenCV already threads inside a measurement.
Keeper copies and XMP writes use two workers so they do not fight the disk.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TypeVar

PREVIEW_POOL_MIN = 2
PREVIEW_POOL_MAX = 4
SUBJECT_BATCH = 4
EXPORT_WORKERS = 2
XMP_WORKERS = 2

T = TypeVar("T")


def preview_workers(task_count: int) -> int:
    """How many preview tasks to run at once. One task needs no pool."""

    if task_count <= 1:
        return 1
    cpus = os.cpu_count() or PREVIEW_POOL_MIN
    return min(PREVIEW_POOL_MAX, max(PREVIEW_POOL_MIN, cpus), task_count)


def run_ordered(
    count: int,
    function: Callable[[int], T],
    *,
    workers: int,
    cancel: Callable[[], None] | None = None,
    on_finished: Callable[[int, T], None] | None = None,
) -> list[T]:
    """Run ``function(index)`` on a pool and return results in index order.

    At most ``workers`` calls are in flight. ``on_finished`` runs on this
    thread, in order, as soon as that prefix is done. A failure is raised
    when its index is reached, after earlier results have been delivered.
    ``cancel`` runs on this thread between completions and may raise.
    """

    if count <= 0:
        return []
    worker_count = max(1, min(int(workers), count))
    results: list[T] = []
    finished: dict[int, T | BaseException] = {}
    next_submit = 0
    next_done = 0
    pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="outburst")
    in_flight: dict[Future[T], int] = {}
    try:
        while next_done < count:
            if cancel is not None:
                cancel()
            while next_submit < count and len(in_flight) < worker_count:
                index = next_submit
                in_flight[pool.submit(function, index)] = index
                next_submit += 1
            if not in_flight:
                break
            done, _pending = wait(set(in_flight), timeout=0.25, return_when=FIRST_COMPLETED)
            for future in done:
                index = in_flight.pop(future)
                try:
                    finished[index] = future.result()
                except BaseException as exc:
                    finished[index] = exc
            while next_done in finished:
                item = finished.pop(next_done)
                if isinstance(item, BaseException):
                    raise item
                if on_finished is not None:
                    on_finished(next_done, item)
                results.append(item)
                next_done += 1
        if next_done != count:
            raise RuntimeError("Threaded work stopped before every item finished")
        return results
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
