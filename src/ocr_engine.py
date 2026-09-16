"""IndicOCR wrapper: page image in -> reading-ordered layout blocks out.

IndicOCR (Bodhan AI / AI4Bharat, https://huggingface.co/bodhan-ai/indic-ocr)
is a local, two-stage model: IndicDocLayout detects and orders the page's
blocks, IndicBlockOCR transcribes them. Script is inferred from the image, so
there is no per-language pass, model pairing, or cross-language merge step to
manage -- the multi-recogniser machinery the PaddleOCR engine needed here is
gone because one model reads every supported script in a single call.

IndicOCR's own pipeline never sends Header/Footer/Diagram/Image/Chart/
Advertisement blocks to the recogniser (see FORCE_OCR_LABELS below) -- by
design, to avoid hallucinating text onto photos. On document scans that is
the right call. On social-media posters it routinely is not: headline text
is often baked straight into a photo or graphic, and the layout stage files
the whole region as "Image" without ever trying to read what's on it.
Measured on a real poster, a giant "Image" block that IndicOCR would have
left as text="" was the poster's entire headline -- "UCC WILL BE
IMPLEMENTED IN 21 STATES BEFORE 2029" plus a second banner -- while the
recogniser read it correctly the moment it was actually asked to. This
wrapper relabels those blocks before the recogniser sees them, then dedups
their output against whatever nearby blocks already captured on their own
(a photo-with-headline can span the whole page, paragraphs and all).

The gated HF repo requires a Hugging Face account with access granted
(request it at the model page) and either `hf auth login` or an `HF_TOKEN`
env var before the first download will succeed.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_REPO = "bodhan-ai/indic-ocr"

# Case-insensitive, matching IndicOCR's own idp_contract.OCR_SKIP_LABELS
# exactly -- these are the labels its pipeline never transcribes.
FORCE_OCR_LABELS = frozenset({"header", "footer", "diagram", "image", "chart", "advertisement"})

# What forced blocks get relabelled to before the recogniser sees them.
# Both fields matter: idp_crops.py skips cropping when *either* the label is
# in OCR_SKIP_LABELS *or* the type is in DROP_TYPES ({"Figure", "Picture"}),
# so changing only one leaves the block uncropped and the fix is a no-op.
_FORCE_LABEL, _FORCE_TYPE = "Paragraph", "Text"

# A forced block's crop often overlaps a block that was already transcribed
# normally (the giant "Image" region above literally contained two
# paragraphs IndicOCR had already read correctly on their own). Exact-match
# dedup misses near-duplicates -- the two recogniser passes over the same
# text can read it slightly differently -- so lines are compared fuzzily
# against every normally-transcribed block's text instead.
_DEDUP_SIMILARITY = 0.75


def _dedup_forced_text(text: str, known_lines: list[str]) -> str:
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(difflib.SequenceMatcher(None, line.lower(), known.lower()).ratio()
               >= _DEDUP_SIMILARITY for known in known_lines):
            continue
        kept.append(line)
    return "\n".join(kept)


def _merge_reads(base: str, extra: str) -> str:
    """Union two recognizer reads of the same block, keeping every line from
    `base` plus whatever `extra` adds that isn't already covered there."""
    if not extra:
        return base
    if not base:
        return extra
    kept = _dedup_forced_text(extra, base.splitlines())
    return f"{base}\n{kept}" if kept else base


# Forcing OCR onto a block IndicOCR's own pipeline would have left as a photo
# is exactly the case its docs call out as inviting hallucination -- and
# tested against a real poster, that is what happened: the recogniser
# returned "[Image of a uniform civil code (UCC) screen with a screen above
# it]" instead of transcribing. A whole line self-contained in one bracket
# pair, long enough to be a generated description rather than a short
# bracketed promo ("[LIMITED OFFER]" survives), or an explicit
# "image of"/"photo of" opener, is dropped from forced blocks only --
# legitimately transcribed blocks aren't at risk of this failure mode.
_CAPTION_OPENERS = ("image of", "photo of", "picture of", "screenshot of",
                    "this image", "the image", "an image of", "a photo of")


def _is_hallucinated_caption(line: str) -> bool:
    stripped = line.strip()
    wrapped = ((stripped.startswith("[") and stripped.endswith("]")) or
              (stripped.startswith("(") and stripped.endswith(")")))
    if wrapped and len(stripped.split()) > 4:
        return True
    return stripped.lower().startswith(_CAPTION_OPENERS)


def _strip_hallucinated_captions(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _is_hallucinated_caption(line))


@dataclass
class TextBlock:
    """One detected layout block and its transcription.

    `confidence` is IndicDocLayout's *detection* confidence -- IndicOCR does
    not expose a separate per-block transcription score. `label`/`block_type`
    always reflect IndicDocLayout's original classification, even for a
    block this wrapper force-transcribed against that classification's
    advice (see FORCE_OCR_LABELS) -- so a consumer can still tell "this text
    came out of what was detected as a photo."
    """

    text: str
    confidence: float
    bbox_xyxy: list[float]      # [x_min, y_min, x_max, y_max], pixels
    order: int                  # reading-order rank, 0-based, gap-free
    label: str                  # raw 37-class layout label, e.g. "Paragraph"
    block_type: str             # coarse category, e.g. "Text", "Table", "Picture"

    @property
    def x_min(self) -> float:
        return self.bbox_xyxy[0]

    @property
    def y_min(self) -> float:
        return self.bbox_xyxy[1]

    @property
    def x_max(self) -> float:
        return self.bbox_xyxy[2]

    @property
    def y_max(self) -> float:
        return self.bbox_xyxy[3]

    @property
    def box(self) -> list[list[float]]:
        """4 corner points, clockwise from top-left -- for the overlay drawer."""
        x0, y0, x1, y1 = self.bbox_xyxy
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    def scaled(self, factor: float) -> "TextBlock":
        """Map the box back to original-image coordinates."""
        if factor == 1.0:
            return self
        return TextBlock(self.text, self.confidence,
                         [v / factor for v in self.bbox_xyxy],
                         self.order, self.label, self.block_type)

    def to_dict(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "order": self.order,
            "label": self.label,
            "type": self.block_type,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(v, 1) for v in self.bbox_xyxy],
        }


class OCREngine:
    """Lazily-built IndicOCR layout + recognition stages."""

    def __init__(self, table_format: str = "html"):
        self.table_format = table_format
        self._layout_model = None
        self._ocr_model = None
        self._force_crop_config = None  # built in _build(); needs CropConfig
        self._page_result_cls = None    # ditto, needs PageResult

    def _build(self):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "huggingface_hub is not installed. Create a Python 3.12 venv and run:\n"
                "    pip install -r requirements.txt"
            ) from exc

        try:
            repo = snapshot_download(MODEL_REPO)
        except Exception as exc:  # noqa: BLE001 - re-raised with actionable context
            raise RuntimeError(
                f"Could not download {MODEL_REPO}. This repo is gated on Hugging "
                "Face: request access at https://huggingface.co/bodhan-ai/indic-ocr, "
                "then `hf auth login` or set HF_TOKEN before retrying."
            ) from exc

        if repo not in sys.path:
            sys.path.insert(0, repo)

        try:
            from idp_offline import IndicBlockOCR, IndicDocLayout
            from idp_types import CropConfig, PageResult, RecognizerConfig, TableFormat
        except ImportError:
            # The model ships its own installer, which reads the local driver
            # and picks matching CUDA wheels for torch/transformers -- running
            # it here is what the model card's own setup instructions do. It
            # installs into "whichever Python is active", so the venv this
            # process is running in has to be first on PATH, or its `pip`
            # calls resolve to the system Python and hit PEP 668's
            # externally-managed-environment guard instead.
            env = os.environ.copy()
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            subprocess.run(["bash", str(Path(repo) / "install.sh")],
                           check=True, cwd=repo, env=env)
            from idp_offline import IndicBlockOCR, IndicDocLayout
            from idp_types import CropConfig, PageResult, RecognizerConfig, TableFormat

        self._page_result_cls = PageResult

        layout_model = IndicDocLayout(f"{repo}/weights/layout")
        ocr_model = IndicBlockOCR(f"{repo}/weights/ocr",
                                  config=RecognizerConfig(table_format=TableFormat(self.table_format)))

        # A forced block (see FORCE_OCR_LABELS) is our hardest case by
        # construction: a photo with a headline baked in routinely also
        # carries much smaller embedded text -- a masthead, a book title --
        # that a normally-sized crop doesn't give the model enough pixels to
        # read. CropConfig.min_px_side is the vendor's own upscale trigger
        # (area below it gets scaled up before recognition); pointing it at
        # their own max_px_side asks for the model's full supported
        # resolution on these blocks instead of guessing an upscale factor.
        default_crop = CropConfig()
        self._force_crop_config = CropConfig(min_px_side=default_crop.max_px_side)

        return layout_model, ocr_model

    @property
    def _models(self):
        if self._layout_model is None:
            self._layout_model, self._ocr_model = self._build()
        return self._layout_model, self._ocr_model

    def warmup(self) -> None:
        """Build the model + run one tiny inference so timings exclude load cost."""
        blank = np.full((64, 256, 3), 255, dtype=np.uint8)
        try:
            self.run(blank)
        except Exception:  # noqa: BLE001 - warmup must never be fatal
            pass

    def run(self, image: np.ndarray) -> tuple[list[TextBlock], str]:
        """Run layout + recognition on an image array. Returns (ordered blocks, page markdown)."""
        layout_model, ocr_model = self._models

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, image)
            layout = layout_model.detect(tmp_path)

            original = {b.order: (b.label, b.type) for b in layout.blocks}
            forced_orders = {b.order for b in layout.blocks
                             if b.label.strip().lower() in FORCE_OCR_LABELS}
            for b in layout.blocks:
                if b.order in forced_orders:
                    b.label, b.type = _FORCE_LABEL, _FORCE_TYPE

            normal_layout = self._page_result_cls(
                image=layout.image, width=layout.width, height=layout.height,
                blocks=[b for b in layout.blocks if b.order not in forced_orders])
            forced_layout = self._page_result_cls(
                image=layout.image, width=layout.width, height=layout.height,
                blocks=[b for b in layout.blocks if b.order in forced_orders])

            page_blocks = list(ocr_model.run(tmp_path, normal_layout).blocks)
            if forced_layout.blocks:
                # Forced blocks are read twice, at two crop resolutions, and
                # merged -- not replaced. Measured on a real poster, the
                # vendor's document-tuned default resolution and the model's
                # own max resolution surfaced genuinely different text from
                # the same block (one had the price line, the other had the
                # masthead), so picking either loses real content.
                forced_default = ocr_model.run(tmp_path, forced_layout).blocks

                default_crop, ocr_model.crop = ocr_model.crop, self._force_crop_config
                try:
                    forced_highres = {b.order: b
                                      for b in ocr_model.run(tmp_path, forced_layout).blocks}
                finally:
                    ocr_model.crop = default_crop

                for b in forced_default:
                    hi = forced_highres.get(b.order)
                    b.text = _merge_reads(b.text or "", (hi.text or "") if hi else "")
                page_blocks += forced_default
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        known_lines = [line.strip()
                       for b in page_blocks if b.order not in forced_orders
                       for line in (b.text or "").splitlines() if line.strip()]

        blocks = []
        for b in sorted(page_blocks, key=lambda b: b.order):
            label, block_type = original[b.order]
            text = b.text or ""
            if b.order in forced_orders and text:
                text = _strip_hallucinated_captions(text)
                text = _dedup_forced_text(text, known_lines)
            blocks.append(TextBlock(text, float(b.conf), [float(v) for v in b.bbox_xyxy],
                                    int(b.order), label, block_type))

        markdown = "\n\n".join(b.text for b in blocks if b.text)
        return blocks, markdown
