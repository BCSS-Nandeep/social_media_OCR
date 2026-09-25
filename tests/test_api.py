"""Offline tests for the HTTP API layer. Qwen (the default OCR/video engine)
and IndicOCR (the lazy fallback) are both faked here, so these run without
model weights, a GPU, vLLM, or network access -- and without ever running
the real startup lifespan."""

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

import src.api as api                        # noqa: E402
from src import media_downloader              # noqa: E402
from src.ocr_engine import OCREngine, TextBlock  # noqa: E402
from src.ocr_providers import IndicOCRProvider   # noqa: E402
from src.video import frame_extractor         # noqa: E402
from src.video import metadata as video_metadata  # noqa: E402
from src.vlm_client import VideoDescription   # noqa: E402


class FakeEngine(OCREngine):
    """Stands in for a real IndicOCR engine wherever the fallback pool would
    otherwise build one -- see IndicOCRFallbackTests, which patches
    src.ocr_providers.OCREngine with this class."""

    def warmup(self):
        pass

    def run(self, image):
        h, w = image.shape[:2]
        return [TextBlock("HELLO", 0.9, [0, 0, w, h], 0, "Paragraph", "Text")], "HELLO"


class FailingFakeEngine(FakeEngine):
    def run(self, image):
        raise ValueError("boom")


def _sample_jpeg_bytes() -> bytes:
    return cv2.imencode(".jpg", np.full((80, 120, 3), 255, np.uint8))[1].tobytes()


class FakeQwenProvider:
    """Stands in for QwenOCRProvider -- the default OCR path. No HTTP call,
    no GPU, no vLLM."""

    async def extract(self, image_bytes, filename, min_confidence=0.0):
        return {
            "engine": "qwen",
            "image": {"path": None, "name": filename, "width": 120, "height": 80},
            "settings": {"preprocessing": ["none"]},
            "summary": {
                "text_blocks_detected": 1, "mean_confidence": None, "min_confidence": None,
                "blocks_dropped_below_threshold": 0,
                "total_processing_time_sec": 0.01,
                "stage_times_sec": {"ocr": 0.01, "total": 0.01},
            },
            "full_text": "HELLO",
            "blocks": [{"index": 0, "order": 0, "label": "body", "type": "body",
                       "text": "HELLO", "confidence": None, "bbox_xyxy": None}],
            "error": None,
        }


class FailingFakeQwenProvider:
    async def extract(self, image_bytes, filename, min_confidence=0.0):
        raise RuntimeError("qwen unreachable (simulated)")


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


class ExtractEndpointTests(unittest.TestCase):
    """Default path: every image request goes to Qwen. IndicOCR is never
    touched here -- see IndicOCRFallbackTests for that path specifically."""

    def setUp(self):
        api._qwen_provider = FakeQwenProvider()
        api._indicocr_provider = IndicOCRProvider(1)   # fresh, unloaded, per test
        self.client = TestClient(api.app)

    def test_extract_via_base64_returns_success_envelope(self):
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["engine"], "qwen")
        self.assertEqual(body["data"]["full_text"], "HELLO")
        self.assertEqual(body["data"]["blocks"][0]["text"], "HELLO")

    def test_qwen_response_does_not_fabricate_confidence_or_bbox(self):
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        resp = self.client.post("/extract", json=payload)
        block = resp.json()["data"]["blocks"][0]
        self.assertIn("confidence", block)      # key present for compat...
        self.assertIsNone(block["confidence"])  # ...but null, never fabricated
        self.assertIn("bbox_xyxy", block)
        self.assertIsNone(block["bbox_xyxy"])
        summary = resp.json()["data"]["summary"]
        self.assertIsNone(summary["mean_confidence"])
        self.assertIsNone(summary["min_confidence"])

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

    def test_health_reports_qwen_as_default_and_indicocr_not_loaded(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["ocr_engine"], "qwen")
        self.assertFalse(body["indicocr_loaded"])

    def test_root_serves_html(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))
        self.assertIn("IndicOCR", resp.text)


class IndicOCRFallbackTests(unittest.TestCase):
    """Qwen fails in every test here -- these exist specifically to prove
    IndicOCR stays dormant unless INDICOCR_FALLBACK_ENABLED is set, and that
    it's built lazily (only on the call that actually needs it) rather than
    at startup."""

    def setUp(self):
        api._qwen_provider = FailingFakeQwenProvider()
        api._indicocr_provider = IndicOCRProvider(1)
        self.client = TestClient(api.app)

    def test_fallback_disabled_returns_422_and_indicocr_never_loads(self):
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        with mock.patch.object(api, "INDICOCR_FALLBACK_ENABLED", False):
            resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 422)
        self.assertFalse(resp.json()["success"])
        self.assertFalse(api._indicocr_provider.loaded)

    @mock.patch("src.ocr_providers.OCREngine", FakeEngine)
    def test_fallback_enabled_lazily_builds_indicocr_and_succeeds(self):
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        self.assertFalse(api._indicocr_provider.loaded)
        with mock.patch.object(api, "INDICOCR_FALLBACK_ENABLED", True):
            resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["engine"], "indicocr")
        self.assertEqual(body["data"]["full_text"], "HELLO")
        self.assertTrue(api._indicocr_provider.loaded)

    @mock.patch("src.ocr_providers.OCREngine", FailingFakeEngine)
    def test_both_qwen_and_indicocr_failing_returns_422(self):
        payload = {"image_base64": base64.b64encode(_sample_jpeg_bytes()).decode()}
        with mock.patch.object(api, "INDICOCR_FALLBACK_ENABLED", True):
            resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 422)
        self.assertFalse(resp.json()["success"])

    def test_indicocr_provider_constructor_does_no_gpu_work(self):
        """Constructing the provider (as happens at module import time) must
        never touch OCREngine -- only .extract() may."""
        with mock.patch("src.ocr_providers.OCREngine") as mock_engine_cls:
            IndicOCRProvider(1)
            mock_engine_cls.assert_not_called()


class VideoEndpointTests(unittest.TestCase):
    """These never touch a real vLLM server or ffmpeg -- video_metadata.probe
    and frame_extractor.extract_frames are mocked at the module level
    video_service.py imports them from."""

    def setUp(self):
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
    """Confirms the Qwen image path uses the shared, SSRF-safe downloader
    exactly like the video path does."""

    def setUp(self):
        api._qwen_provider = FakeQwenProvider()
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
