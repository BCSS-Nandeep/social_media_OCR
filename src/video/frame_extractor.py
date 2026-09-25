"""Grab frames from a video file at specific timestamps.

Uses cv2.VideoCapture -- already a project dependency for image handling,
and its backend already links against the same ffmpeg installed on the
deployment host, so no new dependency (PyAV, ffmpeg-python) is needed just
to seek and grab frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


class FrameExtractionError(RuntimeError):
    """The video could not be opened, or a frame could not be read/encoded
    at one of the requested timestamps."""


@dataclass
class Frame:
    timestamp: float
    jpeg_bytes: bytes


def extract_frames(path: Path, timestamps: list[float]) -> list[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FrameExtractionError(f"Could not open video: {path}")
    try:
        frames: list[Frame] = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(ts, 0.0) * 1000.0)
            ok, image = cap.read()
            if not ok or image is None:
                raise FrameExtractionError(f"Could not read frame at {ts:.2f}s.")
            ok, buf = cv2.imencode(".jpg", image)
            if not ok:
                raise FrameExtractionError(f"Could not encode frame at {ts:.2f}s.")
            frames.append(Frame(timestamp=ts, jpeg_bytes=buf.tobytes()))
        return frames
    finally:
        cap.release()
