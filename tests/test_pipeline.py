"""Offline tests for everything around the IndicOCR call.

These stub the parser, so they run without the model weights or huggingface_hub
installed:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.exporter import OCRResult, export, write_batch_summary  # noqa: E402
from src.ocr_engine import OCREngine, TextBlock                  # noqa: E402
from src.pipeline import process_image                           # noqa: E402
from src.preprocess import (PreprocessConfig, enhance_contrast,  # noqa: E402
                            fix_orientation, load_image, resize_for_ocr)

TELUGU = "శుభాకాంక్షలు"


def block(x0, y0, x1, y1, text="x", conf=0.9, order=0, label="Paragraph",
         block_type="Text") -> TextBlock:
    return TextBlock(text, conf, [x0, y0, x1, y1], order, label, block_type)


class TextBlockTests(unittest.TestCase):
    def test_box_corners_from_bbox(self):
        b = block(10, 20, 110, 60)
        self.assertEqual(b.box, [[10, 20], [110, 20], [110, 60], [10, 60]])

    def test_scaled_maps_bbox_back_to_original_space(self):
        scaled = block(0, 0, 100, 40).scaled(2.0)
        self.assertEqual(scaled.bbox_xyxy, [0.0, 0.0, 50.0, 20.0])

    def test_scaled_is_noop_at_factor_one(self):
        b = block(0, 0, 100, 40)
        self.assertIs(b.scaled(1.0), b)


class FakeParser:
    """Stands in for IndicOCR.parse(): returns fixed blocks for any page."""

    def parse(self, path):
        return {
            "blocks": [
                {"order": 0, "label": "Title", "type": "Title",
                 "bbox_xyxy": [0, 0, 300, 60], "conf": 0.95, "text": "BIG SALE"},
                {"order": 1, "label": "Paragraph", "type": "Text",
                 "bbox_xyxy": [0, 160, 300, 220], "conf": 0.55, "text": TELUGU},
                {"order": 2, "label": "Image", "type": "Picture",
                 "bbox_xyxy": [0, 260, 300, 400], "conf": 0.80, "text": ""},
            ],
            "markdown": f"BIG SALE\n\n{TELUGU}",
        }


class OCREngineTests(unittest.TestCase):
    def test_run_parses_blocks_in_order_and_returns_markdown(self):
        engine = OCREngine()
        engine._parser = FakeParser()
        blocks, markdown = engine.run(np.full((400, 300, 3), 255, np.uint8))
        self.assertEqual([b.text for b in blocks], ["BIG SALE", TELUGU, ""])
        self.assertEqual([b.order for b in blocks], [0, 1, 2])
        self.assertEqual(markdown, f"BIG SALE\n\n{TELUGU}")

    def test_untranscribed_blocks_keep_their_box_and_confidence(self):
        engine = OCREngine()
        engine._parser = FakeParser()
        blocks, _ = engine.run(np.full((400, 300, 3), 255, np.uint8))
        picture = blocks[2]
        self.assertEqual(picture.block_type, "Picture")
        self.assertEqual(picture.confidence, 0.80)


class PreprocessTests(unittest.TestCase):
    def test_upscales_small_image_and_reports_scale(self):
        out, scale = resize_for_ocr(np.zeros((200, 400, 3), np.uint8), 960, 2560)
        self.assertAlmostEqual(scale, 4.8, places=2)
        self.assertEqual(out.shape[0], 960)

    def test_caps_long_side(self):
        out, _ = resize_for_ocr(np.zeros((1000, 6000, 3), np.uint8), 960, 2560)
        self.assertLessEqual(max(out.shape[:2]), 2560)

    def test_contrast_and_deskew_preserve_shape(self):
        img = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
        self.assertEqual(enhance_contrast(img).shape, img.shape)
        self.assertEqual(fix_orientation(img)[0].shape, img.shape)

    def test_load_image_handles_non_ascii_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "పోస్టర్.png"
            cv2.imencode(".png", np.full((40, 60, 3), 128, np.uint8))[1].tofile(str(path))
            self.assertEqual(load_image(str(path)).shape, (40, 60, 3))


class FakeEngine(OCREngine):
    """Stands in for IndicOCR: returns fixed blocks scaled to the input image."""

    def __init__(self):
        super().__init__()

    def warmup(self):
        pass

    def run(self, image):
        h, w = image.shape[:2]
        blocks = [
            TextBlock("BIG SALE", 0.95, [0, 0, w // 2, h // 10], 0, "Title", "Title"),
            TextBlock(TELUGU, 0.55, [0, h // 5, w // 2, h // 3], 1, "Paragraph", "Text"),
        ]
        return blocks, f"BIG SALE\n\n{TELUGU}"


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.image = self.dir / "poster.png"
        cv2.imwrite(str(self.image), np.full((800, 600, 3), 240, np.uint8))
        self.addCleanup(self.tmp.cleanup)

    def test_process_and_export_all_formats(self):
        result, _ = process_image(self.image, FakeEngine(), PreprocessConfig())
        self.assertEqual(result.block_count, 2)
        self.assertEqual(result.image_size, (600, 800))
        self.assertIn("BIG SALE", result.text)
        self.assertIn(TELUGU, result.text)
        self.assertGreater(result.total_time, 0)

        written = export(result, self.dir / "out", ["txt", "json", "csv"])
        self.assertEqual([p.name for p in written],
                         ["poster.txt", "poster.json", "poster.csv"])

        payload = json.loads(written[1].read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["text_blocks_detected"], 2)
        self.assertEqual(payload["blocks"][0]["text"], "BIG SALE")
        self.assertEqual(payload["blocks"][0]["confidence"], 0.95)
        self.assertIn("total_processing_time_sec", payload["summary"])
        self.assertIn(TELUGU, written[0].read_text(encoding="utf-8"))
        self.assertIn(TELUGU, written[2].read_text(encoding="utf-8-sig"))

    def test_min_confidence_filters_and_counts_drops(self):
        result, _ = process_image(self.image, FakeEngine(), PreprocessConfig(),
                                  min_confidence=0.8)
        self.assertEqual(result.block_count, 1)
        self.assertEqual(result.dropped_low_confidence, 1)

    def test_preprocessing_scales_boxes_back_to_original_coordinates(self):
        cfg = PreprocessConfig(resize=True, min_side=1200, enhance_contrast=True)
        result, _ = process_image(self.image, FakeEngine(), cfg)
        self.assertTrue(any(s.startswith("resize") for s in result.preprocessing))
        self.assertLessEqual(max(b.x_max for b in result.blocks), 600)

    def test_batch_summary(self):
        result, _ = process_image(self.image, FakeEngine(), PreprocessConfig())
        path = write_batch_summary([result, result], self.dir / "summary.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["images_processed"], 2)
        self.assertEqual(payload["total_text_blocks"], 4)

    def test_failed_image_result_is_serialisable(self):
        failed = OCRResult("bad.jpg", (0, 0), [], "", [], {"total": 0.0},
                           error="ValueError: boom")
        payload = json.loads(json.dumps(failed.to_dict(), ensure_ascii=False))
        self.assertEqual(payload["summary"]["text_blocks_detected"], 0)
        self.assertEqual(payload["summary"]["mean_confidence"], 0.0)
        self.assertEqual(payload["error"], "ValueError: boom")


if __name__ == "__main__":
    unittest.main(verbosity=2)
