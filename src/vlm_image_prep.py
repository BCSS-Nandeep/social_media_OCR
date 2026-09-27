"""Shared downscale step before any image reaches the VLM.

Qwen2.5-VL's image token count scales with pixel area (~ (H/28)*(W/28) for
this model's patch/merge size) -- an unresized 1920x1080 frame costs ~2700
tokens on its own, and the video path sends up to 16 frames in a single
request (sampler.FRAMES_PER_CHUNK), which blows past this model's 16384
token context on its own before the prompt text or the model's own output
get any room at all (observed directly: a 12-frame, 1920x1080 request hit
"decoder prompt length 33211 > max model length 16384").

960px on the longer side keeps 16 frames + prompt text comfortably under
budget (~660 tokens/frame -> ~10.6k for 16 frames, leaving headroom for
both the prompt text and the model's own JSON output) while staying well
above the resolution needed to read on-screen text -- this pipeline's real
content is news banners/headlines/captions, designed to be legible on a
phone screen, not fine print. Only ever downscales, never upscales, so a
source frame already at or below the cap is untouched.
"""

from __future__ import annotations

import cv2
import numpy as np

MAX_SIDE = 960


def downscale_for_vlm(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # INTER_AREA is the recommended cv2 interpolation for shrinking -- it
    # averages source pixels into each destination pixel rather than
    # sampling/interpolating between a few of them, which keeps text edges
    # cleaner than INTER_LINEAR at this kind of downscale ratio.
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
