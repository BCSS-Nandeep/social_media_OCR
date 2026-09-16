"""Offline tests for the HTTP API layer. A fake engine stands in for
IndicOCR, so these run without model weights, a GPU, or network access --
and without ever running the real startup lifespan (which would try to
download the gated model)."""

from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.api as api                       # noqa: E402
from src.ocr_engine import OCREngine, TextBlock  # noqa: E402


class FakeEngine(OCREngine):
    def warmup(self):
        pass

    def run(self, image):
        h, w = image.shape[:2]
        return [TextBlock("HELLO", 0.9, [0, 0, w, h], 0, "Paragraph", "Text")], "HELLO"


def _sample_jpeg_bytes() -> bytes:
    return cv2.imencode(".jpg", np.full((80, 120, 3), 255, np.uint8))[1].tobytes()


class ExtractEndpointTests(unittest.TestCase):
    def setUp(self):
        api._engine = FakeEngine()  # skips the real lifespan / model download entirely
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
        self.assertEqual(resp.json(), {"status": "ok", "model_loaded": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
