"""IndicOCR wrapper: page image in -> reading-ordered layout blocks out.

IndicOCR (Bodhan AI / AI4Bharat, https://huggingface.co/bodhan-ai/indic-ocr)
is a local, two-stage model: IndicDocLayout detects and orders the page's
blocks, IndicBlockOCR transcribes them. Script is inferred from the image, so
there is no per-language pass, model pairing, or cross-language merge step to
manage -- the multi-recogniser machinery the PaddleOCR engine needed here is
gone because one model reads every supported script in a single call.

The gated HF repo requires a Hugging Face account with access granted
(request it at the model page) and either `hf auth login` or an `HF_TOKEN`
env var before the first download will succeed.
"""

from __future__ import annotations

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


@dataclass
class TextBlock:
    """One detected layout block and its transcription.

    `confidence` is IndicDocLayout's *detection* confidence -- IndicOCR does
    not expose a separate per-block transcription score. Pictorial and margin
    blocks (Image, Header, Footer, ...) carry a real box and confidence but
    `text == ""`; they were detected, just not sent to the recogniser.
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
    """Lazily-built IndicOCR parser."""

    def __init__(self, table_format: str = "html"):
        self.table_format = table_format
        self._parser = None

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
            from indic_ocr import IndicOCR
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
            from indic_ocr import IndicOCR

        return IndicOCR.from_pretrained(repo, table_format=self.table_format)

    @property
    def model(self):
        if self._parser is None:
            self._parser = self._build()
        return self._parser

    def warmup(self) -> None:
        """Build the model + run one tiny inference so timings exclude load cost."""
        blank = np.full((64, 256, 3), 255, dtype=np.uint8)
        try:
            self.run(blank)
        except Exception:  # noqa: BLE001 - warmup must never be fatal
            pass

    def run(self, image: np.ndarray) -> tuple[list[TextBlock], str]:
        """Run the parser on an image array. Returns (ordered blocks, page markdown)."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, image)
            page = self.model.parse(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        blocks = [
            TextBlock(b["text"], float(b["conf"]), [float(v) for v in b["bbox_xyxy"]],
                      int(b["order"]), b["label"], b["type"])
            for b in sorted(page["blocks"], key=lambda b: b["order"])
        ]
        return blocks, page["markdown"]
