"""Offline tests for the HTTP API layer. A fake engine stands in for
IndicOCR, so these run without model weights, a GPU, or network access --
and without ever running the real startup lifespan (which would try to
download the gated model)."""

from __future__ import annotations

import asyncio
import base64
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.api as api                       # noqa: E402
from src import media_downloader             # noqa: E402
from src.ocr_engine import OCREngine, TextBlock  # noqa: E402
from src.video import frame_extractor        # noqa: E402
from src.video import metadata as video_metadata  # noqa: E402
from src.vlm_client import VideoDescription  # noqa: E402


class FakeEngine(OCREngine):
    def warmup(self):
        pass

    def run(self, image):
        h, w = image.shape[:2]
        return [TextBlock("HELLO", 0.9, [0, 0, w, h], 0, "Paragraph", "Text")], "HELLO"


class FailingFakeEngine(FakeEngine):
    def run(self, image):
        raise ValueError("boom")


def _pool_of(*engines: OCREngine) -> asyncio.Queue:
    pool = asyncio.Queue()
    for engine in engines:
        pool.put_nowait(engine)
    return pool


def _sample_jpeg_bytes() -> bytes:
    return cv2.imencode(".jpg", np.full((80, 120, 3), 255, np.uint8))[1].tobytes()


class ExtractEndpointTests(unittest.TestCase):
    def setUp(self):
        api._pool = _pool_of(FakeEngine())  # skips the real lifespan / model download entirely
        self.client = TestClient(api.app)

    def test_extract_via_base64_returns_success_envelope(self):
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["full_text"], "HELLO")
        self.assertEqual(body["data"]["blocks"][0]["text"], "HELLO")

    def test_missing_both_sources_is_rejected(self):
        resp = self.client.post("/extract", json={})
        self.assertEqual(resp.status_code, 422)
        self.assertFalse(resp.json()["success"])

    def test_both_sources_given_is_rejected(self):
        b64 = base64.b64encode(_sample_jpeg_bytes()).decode()
        resp = self.client.post("/extract",
                                json={"image_base64": b64, "image_url": "http://x/y.jpg"})
        self.assertEqual(resp.status_code, 422)

    def test_invalid_base64_is_rejected(self):
        resp = self.client.post("/extract", json={"image_base64": "not valid base64!!"})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.json()["success"])

    def test_unsupported_url_scheme_is_rejected(self):
        resp = self.client.post("/extract", json={"image_url": "file:///etc/passwd"})
        self.assertEqual(resp.status_code, 400)

    def test_health_reports_model_loaded(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["workers_available"], 1)

    def test_worker_is_returned_to_pool_after_success(self):
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        self.client.post("/extract", json=payload)
        self.assertEqual(self.client.get("/health").json()["workers_available"], 1)

    def test_worker_is_returned_to_pool_after_failure(self):
        api._pool = _pool_of(FailingFakeEngine())
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(self.client.get("/health").json()["workers_available"], 1)

    def test_two_requests_share_a_two_worker_pool_without_failing(self):
        api._pool = _pool_of(FakeEngine(), FakeEngine())
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        for _ in range(2):
            resp = self.client.post("/extract", json=payload)
            self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.client.get("/health").json()["workers_available"], 2)

    def test_root_serves_html(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))
        self.assertIn("IndicOCR", resp.text)


class FakeVLMClient:
    """Stands in for a real vLLM server -- no HTTP call, no GPU."""

    async def describe_frames(self, frames):
        return VideoDescription(
            description="A person enters the frame and walks across the room.",
            summary="A person walks across the room.",
            events=[{"timestamp": frames[0][0], "description": "Person enters frame."}],
            visible_text=[],
        )

    async def health(self):
        return True


class VideoEndpointTests(unittest.TestCase):
    """These never touch a real vLLM server or ffmpeg -- video_metadata.probe
    and frame_extractor.extract_frames are mocked at the module level
    video_service.py imports them from, the same way FakeEngine stands in
    for OCREngine above."""

    def setUp(self):
        api._pool = _pool_of(FakeEngine())
        api._video_semaphore = asyncio.Semaphore(1)
        api._vlm_client = FakeVLMClient()
        self.client = TestClient(api.app)

    def test_video_url_and_image_base64_together_is_rejected(self):
        payload = {
            "video_url": "http://example.com/clip.mp4",
            "image_base64": base64.b64encode(_sample_jpeg_bytes()).decode(),
        }
        resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 422)
        self.assertFalse(resp.json()["success"])

    @mock.patch("src.video_service.frame_extractor.extract_frames")
    @mock.patch("src.video_service.video_metadata.probe")
    @mock.patch("src.api.media_downloader.fetch_url")
    def test_video_url_returns_chronological_description(self, mock_fetch, mock_probe, mock_extract):
        mock_fetch.return_value = media_downloader.FetchResult(
            data=b"fake-video-bytes", content_type="video/mp4")
        mock_probe.return_value = video_metadata.VideoMetadata(
            duration_seconds=20.0, width=640, height=360, fps=30.0)
        timestamps = [1.67, 5.0, 8.33, 11.67, 15.0, 18.33]  # frame_count_for(20) == 6
        mock_extract.return_value = [
            frame_extractor.Frame(timestamp=ts, jpeg_bytes=_sample_jpeg_bytes())
            for ts in timestamps
        ]

        resp = self.client.post("/extract", json={"video_url": "http://example.com/clip.mp4"})

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["media_type"], "video")
        self.assertEqual(body["data"]["frames_processed"], 6)
        self.assertEqual(body["data"]["chunks_processed"], 1)
        self.assertIn("description", body["data"])
        self.assertEqual(body["data"]["events"][0]["description"], "Person enters frame.")

    @mock.patch("src.api.media_downloader.fetch_url")
    def test_unsafe_video_url_is_rejected(self, mock_fetch):
        mock_fetch.side_effect = media_downloader.UnsafeURL("blocked: private address")
        resp = self.client.post(
            "/extract", json={"video_url": "http://169.254.169.254/metadata.mp4"})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.json()["success"])

    @mock.patch("src.api.media_downloader.fetch_url")
    def test_oversized_video_is_rejected(self, mock_fetch):
        mock_fetch.side_effect = media_downloader.PayloadTooLarge("too big")
        resp = self.client.post("/extract", json={"video_url": "http://example.com/huge.mp4"})
        self.assertEqual(resp.status_code, 413)
        self.assertFalse(resp.json()["success"])

    @mock.patch("src.video_service.video_metadata.probe")
    @mock.patch("src.api.media_downloader.fetch_url")
    def test_invalid_video_is_rejected(self, mock_fetch, mock_probe):
        mock_fetch.return_value = media_downloader.FetchResult(
            data=b"not-really-a-video", content_type="video/mp4")
        mock_probe.side_effect = video_metadata.ProbeError("moov atom not found")
        resp = self.client.post("/extract", json={"video_url": "http://example.com/bad.mp4"})
        self.assertEqual(resp.status_code, 422)
        self.assertFalse(resp.json()["success"])

    @mock.patch("src.video_service.video_metadata.probe")
    @mock.patch("src.api.media_downloader.fetch_url")
    def test_video_exceeding_max_duration_is_rejected(self, mock_fetch, mock_probe):
        mock_fetch.return_value = media_downloader.FetchResult(
            data=b"fake-video-bytes", content_type="video/mp4")
        mock_probe.return_value = video_metadata.VideoMetadata(
            duration_seconds=api.MAX_VIDEO_DURATION_SECONDS + 1, width=640, height=360, fps=30.0)
        resp = self.client.post("/extract", json={"video_url": "http://example.com/long.mp4"})
        self.assertEqual(resp.status_code, 422)
        self.assertFalse(resp.json()["success"])

    @mock.patch("src.video_service.frame_extractor.extract_frames")
    @mock.patch("src.video_service.video_metadata.probe")
    @mock.patch("src.api.media_downloader.fetch_url")
    def test_eleven_minute_video_is_processed_in_two_chunks(self, mock_fetch, mock_probe, mock_extract):
        mock_fetch.return_value = media_downloader.FetchResult(
            data=b"fake-video-bytes", content_type="video/mp4")
        mock_probe.return_value = video_metadata.VideoMetadata(
            duration_seconds=11 * 60.0, width=640, height=360, fps=30.0)
        mock_extract.return_value = [
            frame_extractor.Frame(timestamp=float(i), jpeg_bytes=_sample_jpeg_bytes())
            for i in range(16)
        ]

        resp = self.client.post("/extract", json={"video_url": "http://example.com/long.mp4"})

        self.assertEqual(resp.status_code, 200)
        body = resp.json()["data"]
        self.assertEqual(body["chunks_processed"], 2)
        self.assertEqual(mock_extract.call_count, 2)
        self.assertEqual(body["frames_processed"], 32)  # 16 frames/chunk x 2 chunks


class ImageOCRRegressionTests(unittest.TestCase):
    """Confirms the image path is unaffected by the video work sharing the
    same endpoint and the same media_downloader module."""

    def setUp(self):
        api._pool = _pool_of(FakeEngine())
        self.client = TestClient(api.app)

    def test_image_base64_still_works_exactly_as_before(self):
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["full_text"], "HELLO")

    def test_image_url_still_uses_the_shared_downloader(self):
        with mock.patch("src.api.media_downloader.fetch_url") as mock_fetch:
            mock_fetch.return_value = media_downloader.FetchResult(
                data=_sample_jpeg_bytes(), content_type="image/jpeg")
            resp = self.client.post("/extract", json={"image_url": "http://example.com/x.jpg"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_image_url_ssrf_to_private_ip_is_blocked(self):
        with mock.patch("src.api.media_downloader.fetch_url") as mock_fetch:
            mock_fetch.side_effect = media_downloader.UnsafeURL("blocked")
            resp = self.client.post(
                "/extract", json={"image_url": "http://169.254.169.254/latest/meta-data/"})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
