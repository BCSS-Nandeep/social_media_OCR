"""Per-image script detection by probing a few text crops.

Detection is script-agnostic, so the boxes can be found once and then shown to
each candidate recogniser. The right script returns high-confidence readable
text; the wrong one returns low-confidence noise.

Why crops rather than whole images: running six *full* PaddleOCR pipelines per
image costs ~55 s and, because each PaddleOCR instance carries its own copy of
the detector, holding six at once segfaults the process. A standalone
recogniser is small enough to keep all six resident, and scoring six of them on
half a dozen crops is a few seconds rather than a minute.

Measured on four real posters (one Telugu, three Hindi), every image was
identified correctly with the winner above 0.95 and margins of 0.14-0.31:

    telugu=0.961  kannada=0.768  tamil=0.635 ...   -> telugu
    devanagari=0.971  kannada=0.662  latin=0.646 ... -> devanagari
"""

from __future__ import annotations

import cv2
import numpy as np

from .scripts import SCRIPT_MODELS

# Crops to probe. More crops means a steadier score but a slower probe; six
# was enough for a decisive margin on every test image.
DEFAULT_MAX_CROPS = 6

# Recognisers work at a fixed line height, so shrinking tall crops to it costs
# no accuracy and cuts probe time substantially.
PROBE_HEIGHT = 48

# Below this margin between the top two scripts, the detection is a coin flip
# and the caller should be told rather than quietly trusted.
CONFIDENT_MARGIN = 0.05


def warp_crop(image: np.ndarray, poly) -> np.ndarray | None:
    """Rectify one detected quadrilateral to an upright crop."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return None
    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if width < 8 or height < 8:
        return None

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(pts, dst),
                               (width, height))
    if height > PROBE_HEIGHT:
        scale = PROBE_HEIGHT / height
        crop = cv2.resize(crop, (max(8, int(width * scale)), PROBE_HEIGHT),
                          interpolation=cv2.INTER_AREA)
    return crop


def _as_dict(result) -> dict:
    return result if isinstance(result, dict) else getattr(result, "json", {})


class ScriptDetector:
    """Detects which script an image is written in. Models load on first use."""

    def __init__(self, detection_model: str, scripts: dict[str, str] | None = None,
                 max_crops: int = DEFAULT_MAX_CROPS):
        self.detection_model = detection_model
        self.scripts = dict(scripts or SCRIPT_MODELS)
        self.max_crops = max_crops
        self._det = None
        self._recs: dict[str, object] = {}

    # ------------------------------------------------------------------ models

    def _create(self, name: str):
        import paddlex
        return paddlex.create_model(name)

    @property
    def detector(self):
        if self._det is None:
            self._det = self._create(self.detection_model)
        return self._det

    def recogniser(self, script: str):
        if script not in self._recs:
            self._recs[script] = self._create(self.scripts[script])
        return self._recs[script]

    # ------------------------------------------------------------------- probe

    def crops(self, image: np.ndarray) -> list[np.ndarray]:
        """Largest detected regions, rectified. Largest first: big text is the
        most reliable script evidence and the least likely to be a stray mark."""
        result = _as_dict(next(iter(self.detector.predict([image])), None) or {})
        polys = result.get("dt_polys")
        if polys is None:
            return []

        scored = []
        for poly in polys:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] == 4:
                scored.append((cv2.contourArea(pts), poly))
        scored.sort(key=lambda kv: -kv[0])

        out = []
        for _, poly in scored:
            crop = warp_crop(image, poly)
            if crop is not None:
                out.append(crop)
            if len(out) >= self.max_crops:
                break
        return out

    def score(self, crops: list[np.ndarray]) -> dict[str, float]:
        """Character-weighted mean recognition confidence per script.

        Weighting by text length stops a confident two-character read from
        outranking a script that produced full, plausible lines.
        """
        scores: dict[str, float] = {}
        for script in self.scripts:
            try:
                outputs = list(self.recogniser(script).predict(crops))
            except Exception:  # noqa: BLE001 - one model must not sink detection
                scores[script] = 0.0
                continue

            weighted = weight = 0.0
            for output in outputs:
                data = _as_dict(output)
                text = (data.get("rec_text") or "").strip()
                if not text:
                    continue
                w = len(text)
                weighted += float(data.get("rec_score") or 0.0) * w
                weight += w
            scores[script] = weighted / weight if weight else 0.0
        return scores

    def detect(self, image: np.ndarray) -> tuple[str | None, dict[str, float]]:
        """Return (best script, all scores). Best is None if nothing was found."""
        crops = self.crops(image)
        if not crops:
            return None, {}
        scores = self.score(crops)
        if not scores:
            return None, {}
        return max(scores, key=lambda s: scores[s]), scores


def margin(scores: dict[str, float]) -> float:
    """Gap between the best and second-best script. Small means unreliable."""
    if len(scores) < 2:
        return 1.0
    ranked = sorted(scores.values(), reverse=True)
    return ranked[0] - ranked[1]
