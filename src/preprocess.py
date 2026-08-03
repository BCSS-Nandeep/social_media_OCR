"""Optional image preprocessing stages.

Every stage is off by default: PaddleOCR's own detector is already trained on
noisy in-the-wild imagery, and aggressive filtering usually *costs* recall on
poster art (gradients, thin display fonts, text over photos). Enable a stage
only when a specific image needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    resize: bool = False
    min_side: int = 960
    max_side: int = 2560
    denoise: bool = False
    enhance_contrast: bool = False
    sharpen: bool = False
    fix_orientation: bool = False

    applied: list[str] = field(default_factory=list)

    @property
    def any_enabled(self) -> bool:
        return (self.resize or self.denoise or self.enhance_contrast
                or self.sharpen or self.fix_orientation)


def load_image(path: str) -> np.ndarray:
    """Read an image as BGR, tolerating non-ASCII paths (cv2.imread cannot)."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Empty or unreadable file: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Not a decodable image: {path}")
    return image


def resize_for_ocr(image: np.ndarray, min_side: int, max_side: int) -> tuple[np.ndarray, float]:
    """Scale so the short side is >= min_side and the long side <= max_side.

    Returns the image and the scale factor applied, so boxes can be mapped back
    to original-image coordinates.
    """
    h, w = image.shape[:2]
    scale = 1.0
    if min(h, w) < min_side:
        scale = min_side / min(h, w)
    if max(h, w) * scale > max_side:
        scale = max_side / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return image, 1.0

    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=interp)
    return resized, scale


def denoise(image: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise. Mild settings: text strokes must survive."""
    return cv2.fastNlMeansDenoisingColored(image, None, h=5, hColor=5,
                                           templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB -- lifts low-contrast text without shifting hue."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Unsharp mask, mild. Recovers stroke edges softened by JPEG + rescaling.

    Kept gentle (amount 0.7, sigma 1.0) -- aggressive sharpening creates halo
    artifacts around strokes that the detector reads as extra edges.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image, 1.7, blurred, -0.7, 0)


def fix_orientation(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Deskew small rotations (< ~15 deg) estimated from dominant text edges.

    Uses the minimum-area rectangle over morphologically joined text blobs.
    Returns the corrected image and the angle applied (degrees, CCW positive).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(joined)
    if coords is None:
        return image, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.3 or abs(angle) > 15:
        return image, 0.0  # nothing worth fixing, or an unreliable estimate

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, float(angle)


def preprocess(image: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, float, list[str]]:
    """Run the enabled stages in order. Returns (image, scale, stage names)."""
    applied: list[str] = []
    scale = 1.0

    if config.fix_orientation:
        image, angle = fix_orientation(image)
        if angle:
            applied.append(f"orientation({angle:+.2f}deg)")

    if config.resize:
        image, scale = resize_for_ocr(image, config.min_side, config.max_side)
        if scale != 1.0:
            applied.append(f"resize(x{scale:.3f})")

    if config.denoise:
        image = denoise(image)
        applied.append("denoise")

    if config.enhance_contrast:
        image = enhance_contrast(image)
        applied.append("clahe")

    if config.sharpen:
        image = sharpen(image)
        applied.append("sharpen")

    return image, scale, applied
