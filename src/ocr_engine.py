"""PaddleOCR wrapper: detection -> angle classification -> recognition.

Two things this layer exists to hide:

1. PaddleOCR changed its public API in 3.0 (``.ocr(img, cls=True)`` returning
   nested lists became ``.predict(img)`` returning dicts). Both are handled.
2. No single PaddleOCR recognition model covers Telugu *and* Latin script well.
   The ``te`` model's dictionary does include ASCII, but English lines come back
   noticeably weaker than from the ``en`` model. ``--lang te+en`` runs both
   recognisers over the union of detected regions and keeps, per region, the
   higher-confidence read.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


# PP-OCRv6 (PaddleOCR 3.7, June 2026) covers Chinese, English, Japanese and the
# Latin-script languages -- there is NO Telugu recogniser for it, and none is
# planned in that set. Detection, however, is script-agnostic, so the v6
# detector can front the v5 Telugu recogniser.
#
# Measured on the six-poster ground-truth set, swapping only the detector:
#     PP-OCRv5_server_det (stock)  59.5% document accuracy, 5.9 s/image
#     PP-OCRv6_medium_det          68.0% document accuracy, 3.8 s/image
#     PP-OCRv6_small_det           58.3% document accuracy, 1.8 s/image
# Medium wins on accuracy *and* speed: it fragments far less (one poster went
# from 57 boxes to 28), so the recogniser sees whole lines instead of shards.
DETECTION_MODEL = "PP-OCRv6_medium_det"

# Recognition model per language. A language absent here falls back to
# PaddleOCR's own lang-based resolution.
RECOGNITION_MODELS = {
    "te": "te_PP-OCRv5_mobile_rec",   # newest Telugu model that exists
    "en": "PP-OCRv6_medium_rec",
    "ch": "PP-OCRv6_medium_rec",
    "japan": "PP-OCRv6_medium_rec",
    "chinese_cht": "PP-OCRv6_medium_rec",
}


def _major_version(module: Any) -> int:
    """Leading integer of ``module.__version__``; 2 if it cannot be read."""
    raw = str(getattr(module, "__version__", "2")).split(".")[0]
    try:
        return int(raw)
    except ValueError:
        return 2


@dataclass
class TextBlock:
    """One detected text region and its recognised content."""

    text: str
    confidence: float
    box: list[list[float]]      # 4 corner points, clockwise from top-left
    lang: str                   # which recognition model produced this read

    @property
    def x_min(self) -> float:
        return min(p[0] for p in self.box)

    @property
    def y_min(self) -> float:
        return min(p[1] for p in self.box)

    @property
    def x_max(self) -> float:
        return max(p[0] for p in self.box)

    @property
    def y_max(self) -> float:
        return max(p[1] for p in self.box)

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2

    def scaled(self, factor: float) -> "TextBlock":
        """Map the box back to original-image coordinates."""
        if factor == 1.0:
            return self
        return TextBlock(self.text, self.confidence,
                         [[p[0] / factor, p[1] / factor] for p in self.box], self.lang)

    def to_dict(self, index: int, line: int) -> dict[str, Any]:
        return {
            "index": index,
            "line": line,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "lang_model": self.lang,
            "bbox": [[round(p[0], 1), round(p[1], 1)] for p in self.box],
            "bbox_xyxy": [round(self.x_min, 1), round(self.y_min, 1),
                          round(self.x_max, 1), round(self.y_max, 1)],
        }


class OCREngine:
    """Lazily-built PaddleOCR instances, one per recognition language."""

    def __init__(self, langs: Sequence[str], use_gpu: bool = False,
                 use_angle_cls: bool = False, det_db_box_thresh: float = 0.5,
                 drop_score: float = 0.0, det_limit_side_len: int = 960,
                 det_db_unclip_ratio: float = 1.5, det_model: str | None = None,
                 paddle_extra: dict[str, Any] | None = None):
        self.langs = list(langs)
        self.use_gpu = use_gpu
        # Off by default. The textline orientation classifier is trained on
        # document scans; on poster art it misfires and flips upright lines
        # 180 degrees, after which recognition returns garbage. Measured on the
        # ground-truth set: 68.0% -> 84.9% document accuracy with it disabled,
        # and Telugu character accuracy 69.4% -> 92.2%. Detection is untouched
        # (identical block counts), so the damage is purely to the crops fed
        # to the recogniser.
        self.use_angle_cls = use_angle_cls
        self.det_db_box_thresh = det_db_box_thresh
        # PaddleOCR applies its own recognition-confidence filter (default 0.5)
        # *inside* .ocr(), discarding blocks before we ever see them -- which
        # would make the reported block count and confidence stats silently
        # incomplete. Default it to 0 and let --min-confidence be the only filter.
        self.drop_score = drop_score
        # Detection-resolution cap and box-inflation ratio. Measured on the
        # six-poster ground-truth set, neither moved accuracy outside +/-3%
        # noise (det 960/1600/2400 x unclip 1.5/2.0/2.5, plus score_mode and
        # dilation), so the defaults stay at PaddleOCR's own. Swapping the
        # *detector model* is what actually helped -- see DETECTION_MODEL.
        self.det_limit_side_len = det_limit_side_len
        self.det_db_unclip_ratio = det_db_unclip_ratio
        # None -> use DETECTION_MODEL; "" -> let PaddleOCR resolve from lang.
        self.det_model = DETECTION_MODEL if det_model is None else det_model
        # Version-specific extras passed straight to the PaddleOCR constructor
        # (e.g. det_db_score_mode="slow"). The caller owns name correctness.
        self.paddle_extra = dict(paddle_extra or {})
        self._models: dict[str, Any] = {}
        self._api = None  # "v3" or "v2", detected on first build

    # ------------------------------------------------------------------ build

    def _build(self, lang: str):
        try:
            import paddleocr
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "paddleocr is not installed. Create a Python 3.12 venv and run:\n"
                "    pip install -r requirements.txt"
            ) from exc

        # Dispatch on version, NOT on which kwargs are accepted: PaddleOCR 2.x
        # builds its config through argparse and silently swallows unknown
        # keywords, so a try/except probe would "succeed" with 3.x names while
        # quietly dropping the settings they carry (angle classifier, box
        # threshold) and then fail at inference time.
        self._api = "v3" if _major_version(paddleocr) >= 3 else "v2"

        if self._api == "v3":
            kwargs = {
                "lang": lang,
                "use_textline_orientation": self.use_angle_cls,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "text_det_box_thresh": self.det_db_box_thresh,
                "text_rec_score_thresh": self.drop_score,
                "text_det_limit_side_len": self.det_limit_side_len,
                "text_det_unclip_ratio": self.det_db_unclip_ratio,
            }
            # Naming an explicit detector makes PaddleOCR ignore `lang`
            # entirely, so the recogniser has to be named too or it silently
            # falls back to the Chinese default. Only override when we know
            # both halves; otherwise leave lang-based resolution alone.
            rec_model = RECOGNITION_MODELS.get(lang)
            if self.det_model and rec_model:
                kwargs.pop("lang")
                kwargs["text_detection_model_name"] = self.det_model
                kwargs["text_recognition_model_name"] = rec_model
        else:
            kwargs = {
                "lang": lang,
                "use_angle_cls": self.use_angle_cls,
                "use_gpu": self.use_gpu,
                "det_db_box_thresh": self.det_db_box_thresh,
                "drop_score": self.drop_score,
                "det_limit_side_len": self.det_limit_side_len,
                "det_db_unclip_ratio": self.det_db_unclip_ratio,
                "show_log": False,
            }

        kwargs.update(self.paddle_extra)
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised with actionable context
            raise RuntimeError(
                f"Could not initialise PaddleOCR for lang={lang!r} "
                f"(paddleocr {getattr(paddleocr, '__version__', '?')}). "
                "Check the language code is supported by this version "
                "(Telugu = 'te', English = 'en')."
            ) from exc

    def model(self, lang: str):
        if lang not in self._models:
            self._models[lang] = self._build(lang)
        return self._models[lang]

    def warmup(self) -> None:
        """Build every model + run one tiny inference so timings exclude load cost."""
        blank = np.full((64, 256, 3), 255, dtype=np.uint8)
        for lang in self.langs:
            try:
                self._run_single(blank, lang)
            except Exception:  # noqa: BLE001 - warmup must never be fatal
                pass

    # -------------------------------------------------------------- inference

    def _run_single(self, image: np.ndarray, lang: str) -> list[TextBlock]:
        model = self.model(lang)

        if self._api == "v3":
            raw = model.predict(image)
            return self._parse_v3(raw, lang)

        try:
            raw = model.ocr(image, cls=self.use_angle_cls)
        except TypeError:  # 2.9+ deprecated `cls` on some builds
            raw = model.ocr(image)
        return self._parse_v2(raw, lang)

    @staticmethod
    def _parse_v3(raw: Any, lang: str) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        for page in raw or []:
            data = page.get("res", page) if isinstance(page, dict) else getattr(page, "json", {})
            if not isinstance(data, dict):
                continue
            polys = data.get("rec_polys") or data.get("dt_polys") or []
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            for poly, text, score in zip(polys, texts, scores):
                box = [[float(p[0]), float(p[1])] for p in np.asarray(poly).reshape(-1, 2)]
                blocks.append(TextBlock(str(text), float(score), box, lang))
        return blocks

    @staticmethod
    def _parse_v2(raw: Any, lang: str) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        for page in raw or []:
            for line in page or []:
                box, (text, score) = line[0], line[1]
                blocks.append(TextBlock(str(text), float(score),
                                        [[float(p[0]), float(p[1])] for p in box], lang))
        return blocks

    def run(self, image: np.ndarray) -> tuple[list[TextBlock], dict[str, float]]:
        """Run every configured language pass. Returns (merged blocks, per-lang seconds)."""
        per_lang: dict[str, float] = {}
        collected: list[TextBlock] = []

        for lang in self.langs:
            start = time.perf_counter()
            blocks = self._run_single(image, lang)
            per_lang[lang] = time.perf_counter() - start
            collected.extend(blocks)

        if len(self.langs) > 1:
            collected = merge_by_overlap(collected)
        return collected, per_lang


# ------------------------------------------------------------------- merging

def _area(block: TextBlock) -> float:
    return (block.x_max - block.x_min) * (block.y_max - block.y_min)


def _overlap_ratio(a: TextBlock, b: TextBlock) -> float:
    """Intersection over the *smaller* box, not over the union.

    IoU is wrong here. Each language pass uses its own detector, and they
    segment differently: the `en` detector emits whole lines while the `te`
    multilingual detector emits words. "GRAND" sitting inside "GRAND OPENING"
    scores IoU ~0.45 -- below any sane threshold -- so an IoU rule keeps both
    and the word appears twice. Intersection-over-smaller correctly reports
    ~1.0 for containment.
    """
    inter_w = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    inter_h = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    smaller = min(_area(a), _area(b))
    return (inter_w * inter_h) / smaller if smaller > 0 else 0.0


def _cluster(blocks: list[TextBlock], threshold: float) -> list[list[TextBlock]]:
    """Union-find grouping of blocks that cover the same physical region."""
    parent = list(range(len(blocks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if _overlap_ratio(blocks[i], blocks[j]) >= threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    groups: dict[int, list[TextBlock]] = {}
    for index, block in enumerate(blocks):
        groups.setdefault(find(index), []).append(block)
    return list(groups.values())


def _segmentation_score(group: list[TextBlock]) -> float:
    """Character-weighted mean confidence of one detector's take on a region.

    Weighting by text length stops a single high-scoring two-character fragment
    from outranking a correct full-line read.
    """
    weights = [max(1, len(b.text)) for b in group]
    return sum(b.confidence * w for b, w in zip(group, weights)) / sum(weights)


def merge_by_overlap(blocks: list[TextBlock], overlap_threshold: float = 0.6) -> list[TextBlock]:
    """Reconcile the language passes into one non-duplicated set of blocks.

    Blocks covering the same region are clustered, then **one language's
    segmentation of that cluster is kept whole** -- the one with the higher
    character-weighted confidence. Choosing per-cluster rather than per-block
    matters: the passes disagree about where the boundaries are, so mixing
    their blocks would emit a line and its own constituent words side by side.
    Ties go to the coarser segmentation (fewer blocks).
    """
    merged: list[TextBlock] = []
    for cluster in _cluster(blocks, overlap_threshold):
        by_lang: dict[str, list[TextBlock]] = {}
        for block in cluster:
            by_lang.setdefault(block.lang, []).append(block)
        if len(by_lang) == 1:
            merged.extend(cluster)
            continue
        merged.extend(max(by_lang.values(),
                          key=lambda g: (_segmentation_score(g), -len(g))))
    return merged
