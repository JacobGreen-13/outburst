"""Group photographs into bursts from capture timestamps.

``group_frames`` sorts by capture time and keeps neighbors together while the gap is strictly under the threshold. Frames with no clock stay in bursts of one, after the timed bursts. A gap equal to the threshold starts a new burst.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from outburst.models import Burst, Frame


def group_frames(frames: Sequence[Frame], threshold_seconds: float) -> list[Burst]:
    """Return bursts in capture order. The input sequence is not modified.

    Inputs: frames and a non-negative finite gap threshold in seconds.
    Output: bursts numbered burst-001 upward, timed frames first and each untimed frame alone after them.
    Uses: the scan pipeline when a folder is grouped or regrouped.
    Limitations: a missing capture time stays in its own burst, and a negative or non-finite threshold raises ValueError.
    """

    if not math.isfinite(threshold_seconds) or threshold_seconds < 0:
        raise ValueError("threshold_seconds must be a finite number that is zero or positive")

    timed = [frame for frame in frames if frame.captured_at is not None]
    untimed = [frame for frame in frames if frame.captured_at is None]
    timed.sort(key=lambda frame: (frame.captured_at, frame.path.name.lower(), str(frame.path)))
    untimed.sort(key=lambda frame: (frame.path.name.lower(), str(frame.path)))

    groups: list[list[Frame]] = []
    for frame in timed:
        if not groups:
            groups.append([frame])
            continue
        previous = groups[-1][-1]
        assert frame.captured_at is not None and previous.captured_at is not None
        gap = (frame.captured_at - previous.captured_at).total_seconds()
        if gap < threshold_seconds:
            groups[-1].append(frame)
        else:
            groups.append([frame])

    for frame in untimed:
        groups.append([frame])

    return [
        Burst(id=f"burst-{index:03d}", frames=tuple(group))
        for index, group in enumerate(groups, start=1)
    ]
