"""Pure duration -> sampling-plan logic. No I/O, fully unit-testable.

The frame-count policy and the centered-sampling formula both come directly
from the video-description integration spec this module implements --
tests/test_video.py is written against that spec's own table and examples,
not just against whatever this code happens to compute.
"""

from __future__ import annotations

CHUNK_SECONDS = 600.0          # 10 minutes
FRAMES_PER_CHUNK = 16


def frame_count_for(duration_seconds: float) -> int | None:
    """How many frames to sample for one VLM call. None means the video is
    too long for a single call -- the caller must use chunk_windows()
    instead (>10 minutes)."""
    if duration_seconds <= 10:
        return 4
    if duration_seconds <= 30:
        return 6
    if duration_seconds <= 60:
        return 8
    if duration_seconds <= 120:
        return 10
    if duration_seconds <= 300:
        return 12
    if duration_seconds <= 600:
        return 16
    return None


def sample_timestamps(duration_seconds: float, count: int) -> list[float]:
    """Evenly spaced, centered within each of `count` equal slices of the
    video (duration * (i + 0.5) / count) so the first/last sample isn't
    sitting on a potentially blank/black boundary frame, and sampling isn't
    just "first N frames" or fixed-fps."""
    if count <= 0:
        raise ValueError("count must be positive")
    return [duration_seconds * (i + 0.5) / count for i in range(count)]


def chunk_windows(duration_seconds: float,
                  chunk_seconds: float = CHUNK_SECONDS) -> list[tuple[float, float]]:
    """(start, end) windows covering the whole video, each at most
    `chunk_seconds` long. Only meant for videos frame_count_for() returned
    None for (>10 minutes) -- each window gets FRAMES_PER_CHUNK frames,
    sampled with sample_timestamps() over that window's own duration."""
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        windows.append((start, end))
        start = end
    return windows
