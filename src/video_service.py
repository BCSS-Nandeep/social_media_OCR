"""Video -> chronological description.

Download is done by the caller (api.py, via the shared SSRF-safe
media_downloader) -- this module owns everything from raw video bytes
onward: metadata, frame sampling, extraction, and the VLM call(s), with
per-stage timing and guaranteed temp-file cleanup. Mirrors pipeline.py's
process_image() shape: one function, one result dict, timings broken out
per stage.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from functools import partial
from pathlib import Path

from .video import frame_extractor, metadata as video_metadata, sampler
from .vlm_client import VideoDescription, VLMClient, VLMError

log = logging.getLogger("ocr.video")


class VideoProcessingError(RuntimeError):
    """A controlled, user-facing failure -- bad input, ffprobe/extraction
    failure, or a VLM that never returned usable output. Always mapped to a
    422 in the {success, data, error} envelope, never a raw 500."""


async def _run_with_timeout(func, *, timeout: float):
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(loop.run_in_executor(None, func), timeout=timeout)


async def describe_video(
    video_bytes: bytes,
    *,
    vlm_client: VLMClient,
    max_duration_seconds: float,
    frame_extraction_timeout_seconds: float,
    download_time_seconds: float = 0.0,
) -> dict:
    timings: dict[str, float] = {"download": round(download_time_seconds, 3)}
    total_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="ocr-video-") as tmp:
        video_path = Path(tmp) / "input.mp4"
        video_path.write_bytes(video_bytes)

        t0 = time.perf_counter()
        try:
            meta = await _run_with_timeout(
                partial(video_metadata.probe, video_path, timeout=frame_extraction_timeout_seconds),
                timeout=frame_extraction_timeout_seconds)
        except video_metadata.ProbeError as exc:
            raise VideoProcessingError(f"Could not read video: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise VideoProcessingError("Timed out reading video metadata.") from exc
        timings["metadata"] = round(time.perf_counter() - t0, 3)

        if meta.duration_seconds > max_duration_seconds:
            raise VideoProcessingError(
                f"Video duration {meta.duration_seconds:.0f}s exceeds the "
                f"{max_duration_seconds:.0f}s limit.")

        frame_count = sampler.frame_count_for(meta.duration_seconds)
        if frame_count is not None:
            windows = [(0.0, meta.duration_seconds, frame_count)]
        else:
            windows = [(start, end, sampler.FRAMES_PER_CHUNK)
                      for start, end in sampler.chunk_windows(meta.duration_seconds)]

        frame_extraction_time = 0.0
        vlm_time = 0.0
        total_frames = 0
        chunk_results: list[tuple[float, float, VideoDescription]] = []

        for start, end, n_frames in windows:
            offsets = sampler.sample_timestamps(end - start, n_frames)
            timestamps = [start + off for off in offsets]

            t0 = time.perf_counter()
            try:
                frames = await _run_with_timeout(
                    partial(frame_extractor.extract_frames, video_path, timestamps),
                    timeout=frame_extraction_timeout_seconds)
            except frame_extractor.FrameExtractionError as exc:
                raise VideoProcessingError(f"Frame extraction failed: {exc}") from exc
            except asyncio.TimeoutError as exc:
                raise VideoProcessingError("Timed out extracting video frames.") from exc
            frame_extraction_time += time.perf_counter() - t0
            total_frames += len(frames)

            t0 = time.perf_counter()
            try:
                desc = await vlm_client.describe_frames(
                    [(f.timestamp, f.jpeg_bytes) for f in frames])
            except VLMError as exc:
                raise VideoProcessingError(str(exc)) from exc
            vlm_time += time.perf_counter() - t0

            chunk_results.append((start, end, desc))

        timings["frame_extraction"] = round(frame_extraction_time, 3)
        timings["vlm"] = round(vlm_time, 3)

        if len(chunk_results) == 1:
            _, _, only = chunk_results[0]
            description, summary = only.description, only.summary
            events, visible_text = only.events, only.visible_text
        else:
            # No extra "merge" VLM call -- bounded cost. Chunk summaries are
            # concatenated with their time range labeled; events/visible_text
            # from every chunk are combined as-is.
            description_parts: list[str] = []
            summary_parts: list[str] = []
            events = []
            visible_text = []
            for start, end, desc in chunk_results:
                description_parts.append(f"[{start:.0f}s-{end:.0f}s] {desc.description}")
                summary_parts.append(f"[{start:.0f}s-{end:.0f}s] {desc.summary}")
                events.extend(desc.events)
                visible_text.extend(desc.visible_text)
            description = "\n\n".join(description_parts)
            summary = " ".join(summary_parts)

        timings["total"] = round(time.perf_counter() - total_start, 3)

        return {
            "media_type": "video",
            "duration_seconds": round(meta.duration_seconds, 2),
            "frames_processed": total_frames,
            "chunks_processed": len(chunk_results),
            "description": description,
            "summary": summary,
            "events": events,
            "visible_text": visible_text,
            "video_meta": {"width": meta.width, "height": meta.height, "fps": meta.fps},
            "timing": timings,
        }
