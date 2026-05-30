"""Temporal window helpers for fixed-length video clips."""

from __future__ import annotations

import random


def select_temporal_window(frames, window_size, mode="last"):
    """Return a contiguous frame window using a deterministic or random policy."""

    window_size = int(window_size)
    if window_size <= 0 or len(frames) <= window_size:
        return frames

    mode = str(mode or "last").lower()
    if mode == "first":
        start = 0
    elif mode == "last":
        start = len(frames) - window_size
    elif mode == "random":
        start = random.randint(0, len(frames) - window_size)
    else:
        raise ValueError(f"Unsupported temporal window mode: {mode}")

    return frames[start : start + window_size]
