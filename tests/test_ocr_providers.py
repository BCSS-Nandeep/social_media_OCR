"""Unit tests for src/ocr_providers.py, independent of the HTTP layer.

QwenOCRProvider is tested against a fake VLMClient (no HTTP, no GPU).
IndicOCRProvider is tested against a fake OCREngine (see FakeEngine) so
these never touch real model weights -- the one thing genuinely worth
proving here is that IndicOCRProvider does not build anything until
.extract() is actually called."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ocr_engine import OCREngine, TextBlock          # noqa: E402
from src.ocr_providers import IndicOCRProvider, QwenOCRProvider  # noqa: E402
from src.vlm_client import QwenOCRResult                 # noqa: E402


class FakeEngine(OCREngine):
    def warmup(self):
        pass

    def run(self, image):
        h, w = image.shape[:2]
        return [TextBlock("HELLO", 0.9, [0, 0, w, h], 0, "Paragraph", "Text")], "HELLO"


def _sample_jpeg_bytes() -> bytes:
    return cv2.imencode(".jpg", np.full((80, 120, 3), 255, np.uint8))[1].tobytes()


class FakeVLMClient:
    async def extract_text(self, image_bytes, media_type="image/jpeg"):
        return QwenOCRResult(
            full_text="HELLO WORLD",
            blocks=[{"text": "HELLO", "type": "headline"}, {"text": "WORLD", "type": "body"}],
        )


class QwenOCRProviderTests(unittest.TestCase):
    def test_extract_maps_qwen_output_into_the_compat_response_shape(self):
        provider = QwenOCRProvider(FakeVLMClient())
        data = asyncio.run(provider.extract(_sample_jpeg_bytes(), "test.jpg"))

        self.assertEqual(data["engine"], "qwen")
        self.assertEqual(data["full_text"], "HELLO WORLD")
        self.assertEqual(len(data["blocks"]), 2)
        self.assertEqual(data["blocks"][0]["text"], "HELLO")
        self.assertEqual(data["blocks"][0]["label"], "headline")
        # Never fabricated -- present as keys, null as values.
        self.assertIsNone(data["blocks"][0]["confidence"])
        self.assertIsNone(data["blocks"][0]["bbox_xyxy"])
        self.assertIsNone(data["summary"]["mean_confidence"])
        self.assertIsNone(data["summary"]["min_confidence"])
        self.assertEqual(data["summary"]["blocks_dropped_below_threshold"], 0)

    def test_image_dimensions_are_read_from_the_actual_bytes(self):
        provider = QwenOCRProvider(FakeVLMClient())
        data = asyncio.run(provider.extract(_sample_jpeg_bytes(), "test.jpg"))
        self.assertEqual(data["image"]["width"], 120)
        self.assertEqual(data["image"]["height"], 80)


class IndicOCRProviderTests(unittest.TestCase):
    def test_pool_is_not_built_at_construction(self):
        with mock.patch("src.ocr_providers.OCREngine") as mock_engine_cls:
            provider = IndicOCRProvider(2)
            self.assertFalse(provider.loaded)
            mock_engine_cls.assert_not_called()

    def test_pool_is_built_lazily_on_first_extract_call(self):
        with mock.patch("src.ocr_providers.OCREngine", FakeEngine):
            provider = IndicOCRProvider(1)
            self.assertFalse(provider.loaded)

            data = asyncio.run(provider.extract(_sample_jpeg_bytes(), "test.jpg"))

            self.assertTrue(provider.loaded)
            self.assertEqual(data["engine"], "indicocr")
            self.assertEqual(data["full_text"], "HELLO")

    def test_pool_is_built_only_once_across_repeated_calls(self):
        build_calls = []
        with mock.patch("src.ocr_providers.OCREngine", side_effect=lambda *a, **k: (
            build_calls.append(1) or FakeEngine())):
            provider = IndicOCRProvider(1)
            asyncio.run(provider.extract(_sample_jpeg_bytes(), "a.jpg"))
            asyncio.run(provider.extract(_sample_jpeg_bytes(), "b.jpg"))

        self.assertEqual(len(build_calls), 1)

    def test_indicocr_response_has_real_confidence_and_bbox(self):
        """Unlike the Qwen path, IndicOCR's own output is untouched -- real
        numbers, not null."""
        with mock.patch("src.ocr_providers.OCREngine", FakeEngine):
            provider = IndicOCRProvider(1)
            data = asyncio.run(provider.extract(_sample_jpeg_bytes(), "test.jpg"))

        self.assertIsNotNone(data["blocks"][0]["confidence"])
        self.assertIsNotNone(data["blocks"][0]["bbox_xyxy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
