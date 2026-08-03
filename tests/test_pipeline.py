"""Offline tests for everything around the PaddleOCR call.

These stub the recogniser, so they run without paddlepaddle installed:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.exporter import OCRResult, export, write_batch_summary          # noqa: E402
from src.ocr_engine import (DETECTION_MODEL, RECOGNITION_MODELS,         # noqa: E402
                            OCREngine, TextBlock, merge_by_overlap)
from src.pipeline import process_image                                   # noqa: E402
from src.preprocess import (PreprocessConfig, enhance_contrast,          # noqa: E402
                            fix_orientation, load_image, resize_for_ocr)
from src.reading_order import group_into_lines, order_blocks, render_text  # noqa: E402

TELUGU = "శుభాకాంక్షలు"


def block(x0, y0, x1, y1, text="x", conf=0.9, lang="te") -> TextBlock:
    return TextBlock(text, conf, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], lang)


class ReadingOrderTests(unittest.TestCase):
    def test_groups_same_row_and_sorts_left_to_right(self):
        blocks = [block(300, 100, 400, 140, "WORLD"),
                  block(100, 100, 200, 140, "HELLO"),
                  block(100, 300, 400, 340, "SECOND LINE")]
        ordered, lines = order_blocks(blocks)
        self.assertEqual([b.text for b in ordered], ["HELLO", "WORLD", "SECOND LINE"])
        self.assertEqual(lines, [1, 1, 2])

    def test_small_caption_beside_headline_stays_separate(self):
        # A 20px caption sitting inside a 200px headline's vertical span must not
        # merge: overlap is measured against the *shorter* block's height.
        headline = block(50, 0, 900, 200, "HEADLINE")
        caption = block(50, 260, 300, 280, "caption")
        _, lines = order_blocks([headline, caption])
        self.assertEqual(len(set(lines)), 2)

    def test_render_text_one_output_line_per_visual_line(self):
        blocks = [block(100, 100, 200, 140, "HELLO"),
                  block(300, 100, 400, 140, TELUGU),
                  block(100, 300, 400, 340, "next")]
        ordered, lines = order_blocks(blocks)
        self.assertEqual(render_text(ordered, lines), f"HELLO {TELUGU}\nnext")

    def test_empty_input(self):
        self.assertEqual(group_into_lines([]), [])
        self.assertEqual(order_blocks([]), ([], []))
        self.assertEqual(render_text([], []), "")


class MergeTests(unittest.TestCase):
    def test_duplicate_region_keeps_higher_confidence_read(self):
        te = block(10, 10, 210, 60, "SALE", 0.62, "te")
        en = block(12, 11, 208, 59, "SALE", 0.97, "en")
        merged = merge_by_overlap([te, en])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].lang, "en")
        self.assertAlmostEqual(merged[0].confidence, 0.97)

    def test_distinct_regions_survive(self):
        merged = merge_by_overlap([block(0, 0, 100, 40, "a"), block(0, 200, 100, 240, "b")])
        self.assertEqual(len(merged), 2)

    def test_word_boxes_inside_a_line_box_do_not_duplicate(self):
        # The real failure this rule exists for: the `en` detector emits whole
        # lines, the `te` detector emits words. IoU("GRAND", "GRAND OPENING")
        # is only ~0.45, so an IoU rule would keep all three.
        line = block(72, 100, 870, 166, "GRAND OPENING", 0.967, "en")
        w1 = block(68, 100, 404, 167, "GRAND", 0.9992, "te")
        w2 = block(447, 103, 870, 165, "OPENING", 0.965, "te")
        merged = merge_by_overlap([line, w1, w2])
        self.assertEqual({b.lang for b in merged}, {"te"})
        self.assertEqual([b.text for b in sorted(merged, key=lambda b: b.x_min)],
                         ["GRAND", "OPENING"])

    def test_line_read_beats_fragments_when_fragments_are_weaker(self):
        line = block(66, 610, 674, 648, "50% OFF ON ALL ITEMS", 0.9498, "en")
        frags = [block(60, 608, 167, 654, "5000", 0.9063, "te"),
                 block(182, 604, 290, 656, "OF|", 0.8207, "te"),
                 block(307, 606, 387, 654, "ON", 0.9988, "te"),
                 block(399, 604, 498, 656, "ALL", 0.9957, "te"),
                 block(516, 609, 675, 652, "ITEMS", 0.9985, "te")]
        merged = merge_by_overlap([line, *frags])
        self.assertEqual([b.text for b in merged], ["50% OFF ON ALL ITEMS"])

    def test_short_high_confidence_fragment_does_not_outrank_a_long_read(self):
        # Character weighting: an unweighted mean would let the 2-char 1.00
        # fragment carry its whole segmentation past the correct full read.
        line = block(0, 0, 600, 50, "SPECIAL OFFER TODAY", 0.93, "en")
        frags = [block(0, 0, 60, 50, "SP", 1.00, "te"),
                 block(70, 0, 600, 50, "ECIAL OFFER TODAY", 0.80, "te")]
        merged = merge_by_overlap([line, *frags])
        self.assertEqual([b.text for b in merged], ["SPECIAL OFFER TODAY"])

    def test_region_seen_by_only_one_pass_is_kept_intact(self):
        # White-on-black button that the `en` detector missed entirely.
        only_te = [block(104, 1086, 259, 1125, "BOOK", 0.9997, "te"),
                   block(273, 1084, 398, 1125, "NOW", 0.9988, "te")]
        self.assertEqual(len(merge_by_overlap(only_te)), 2)


class ParserTests(unittest.TestCase):
    def test_v2_nested_list_format(self):
        raw = [[[[[0, 0], [100, 0], [100, 30], [0, 30]], ("OFFER", 0.98)]]]
        blocks = OCREngine._parse_v2(raw, "en")
        self.assertEqual((blocks[0].text, blocks[0].confidence, blocks[0].lang),
                         ("OFFER", 0.98, "en"))

    def test_v3_dict_format(self):
        raw = [{"rec_polys": [[[0, 0], [80, 0], [80, 24], [0, 24]]],
                "rec_texts": [TELUGU], "rec_scores": [0.88]}]
        blocks = OCREngine._parse_v3(raw, "te")
        self.assertEqual(blocks[0].text, TELUGU)
        self.assertEqual(blocks[0].box[1], [80.0, 0.0])

    def test_empty_page_is_not_an_error(self):
        self.assertEqual(OCREngine._parse_v2([None], "en"), [])
        self.assertEqual(OCREngine._parse_v3([], "en"), [])


class ModelSelectionTests(unittest.TestCase):
    """Guard the det/rec pairing -- its failure mode is silent, not loud.

    Naming a detection model makes PaddleOCR ignore `lang`. If the recognition
    model is not named in the same breath, it quietly falls back to the Chinese
    default and returns confident nonsense for Telugu.
    """

    def kwargs_for(self, lang, **engine_kwargs):
        engine = OCREngine([lang], **engine_kwargs)
        engine._api = "v3"
        captured = {}

        class FakePaddleOCR:
            def __init__(self, **kw):
                captured.update(kw)

        module = types.ModuleType("paddleocr")
        module.__version__ = "3.7.0"
        module.PaddleOCR = FakePaddleOCR
        with mock.patch.dict(sys.modules, {"paddleocr": module}):
            engine._build(lang)
        return captured

    def test_detector_and_recogniser_are_always_named_together(self):
        kw = self.kwargs_for("te")
        self.assertEqual(kw["text_detection_model_name"], DETECTION_MODEL)
        self.assertEqual(kw["text_recognition_model_name"],
                         RECOGNITION_MODELS["te"])
        self.assertNotIn("lang", kw)  # would be ignored anyway; do not imply otherwise

    def test_unmapped_language_keeps_paddleocr_lang_resolution(self):
        kw = self.kwargs_for("arabic")
        self.assertEqual(kw["lang"], "arabic")
        self.assertNotIn("text_detection_model_name", kw)
        self.assertNotIn("text_recognition_model_name", kw)

    def test_empty_det_model_opts_out_of_the_override(self):
        kw = self.kwargs_for("te", det_model="")
        self.assertEqual(kw["lang"], "te")
        self.assertNotIn("text_detection_model_name", kw)

    def test_explicit_det_model_is_honoured(self):
        kw = self.kwargs_for("te", det_model="PP-OCRv6_small_det")
        self.assertEqual(kw["text_detection_model_name"], "PP-OCRv6_small_det")

    def test_telugu_recogniser_is_not_a_v6_model(self):
        # PP-OCRv6 has no Telugu recogniser; pointing `te` at one would produce
        # confident garbage rather than an error.
        self.assertNotIn("v6", RECOGNITION_MODELS["te"])


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

    def test_box_coordinates_map_back_to_original_space(self):
        scaled = block(0, 0, 100, 40).scaled(2.0)
        self.assertEqual(scaled.box[2], [50.0, 20.0])

    def test_load_image_handles_non_ascii_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "పోస్టర్.png"
            cv2.imencode(".png", np.full((40, 60, 3), 128, np.uint8))[1].tofile(str(path))
            self.assertEqual(load_image(str(path)).shape, (40, 60, 3))


class FakeEngine(OCREngine):
    """Stands in for PaddleOCR: returns fixed blocks scaled to the input image."""

    def __init__(self):
        super().__init__(["te", "en"])

    def warmup(self):
        pass

    def _run_single(self, image, lang):
        h, w = image.shape[:2]
        return [TextBlock("BIG SALE", 0.95, [[0, 0], [w // 2, 0],
                                             [w // 2, h // 10], [0, h // 10]], lang),
                TextBlock(TELUGU, 0.55, [[0, h // 5], [w // 2, h // 5],
                                         [w // 2, h // 3], [0, h // 3]], lang)]


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.image = self.dir / "poster.png"
        cv2.imwrite(str(self.image), np.full((800, 600, 3), 240, np.uint8))
        self.addCleanup(self.tmp.cleanup)

    def test_process_and_export_all_formats(self):
        result, _ = process_image(self.image, FakeEngine(), PreprocessConfig())
        self.assertEqual(result.block_count, 2)          # te/en duplicates merged away
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
        failed = OCRResult("bad.jpg", (0, 0), [], [], "", ["te"], [], {"total": 0.0},
                           error="ValueError: boom")
        payload = json.loads(json.dumps(failed.to_dict(), ensure_ascii=False))
        self.assertEqual(payload["summary"]["text_blocks_detected"], 0)
        self.assertEqual(payload["summary"]["mean_confidence"], 0.0)
        self.assertEqual(payload["error"], "ValueError: boom")


if __name__ == "__main__":
    unittest.main(verbosity=2)
