"""Group photographs into bursts from capture timestamps.

Frames are sorted by capture time. Consecutive frames stay in the same burst
while the gap between neighbors is strictly shorter than the threshold.
A one-second default matches typical wildlife bursts and starts a new burst
after a pause. Frames with no capture time stay alone so a missing clock
cannot glue unrelated pictures together.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from outburst.models import Burst, Frame


def group_frames(frames: Sequence[Frame], threshold_seconds: float) -> list[Burst]:
    """Return bursts in capture order. The input sequence is not modified."""

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
