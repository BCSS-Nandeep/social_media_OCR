"""Video metadata via ffprobe -- duration, dimensions, fps.

ffprobe (not cv2) is used here specifically: it's far more reliable across
odd containers/codecs for duration than cv2.VideoCapture's own frame-count
based estimate, and it's already installed on the deployment host alongside
ffmpeg. cv2 is still what actually grabs frames (frame_extractor.py) --
this module only asks "is this a video, and how long is it."
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ProbeError(RuntimeError):
    """ffprobe could not read this file, or it isn't a video ffprobe understands."""


@dataclass
class VideoMetadata:
    duration_seconds: float
    width: int | None
    height: int | None
    fps: float | None


def probe(path: Path, *, timeout: float) -> VideoMetadata:
    """Run ffprobe and extract duration (required) plus stream info
    (best-effort). Duration is the only field a caller can't proceed
    without -- a missing fps/width/height (unusual container) must not fail
    the request, per spec."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
    except FileNotFoundError as exc:
        raise ProbeError("ffprobe is not installed on this host.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"ffprobe timed out after {timeout}s.") from exc
    except subprocess.CalledProcessError as exc:
        raise ProbeError(f"Not a readable video file: {exc.stderr.strip()}") from exc

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned unparseable output.") from exc

    duration_raw = info.get("format", {}).get("duration")
    if duration_raw is None:
        raise ProbeError("Could not determine video duration.")
    try:
        duration = float(duration_raw)
    except ValueError as exc:
        raise ProbeError(f"Invalid duration reported by ffprobe: {duration_raw!r}") from exc
    if duration <= 0:
        raise ProbeError(f"Video reports non-positive duration: {duration}")

    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    width = height = fps = None
    if video_stream is not None:
        width = video_stream.get("width")
        height = video_stream.get("height")
        rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
        if rate and rate != "0/0":
            try:
                num, _, den = rate.partition("/")
                fps = float(num) / float(den) if den else float(num)
            except (ValueError, ZeroDivisionError):
                fps = None

    return VideoMetadata(duration_seconds=duration, width=width, height=height, fps=fps)
