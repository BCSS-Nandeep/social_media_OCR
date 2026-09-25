"""httpx client for vLLM's OpenAI-compatible chat completions API.

vLLM is a separate, already-installed long-running process serving
Qwen2.5-VL-7B-Instruct (its own Python, its own GPU allocation) -- this
module only ever talks to it over HTTP, the same way any OpenAI-API client
would. No model is loaded inside this process; the FastAPI app and vLLM are
two independent GPU-resident resources, coordinated only through this HTTP
call.

Two tasks share one model and one request/retry/parse loop
(_send_chat_json): image OCR (extract_text) and video description
(describe_frames). Each has its own prompt and its own response schema --
only the plumbing to get a validated JSON object back from the model is
shared.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Callable, TypeVar

import httpx

log = logging.getLogger("ocr.vlm")

T = TypeVar("T")

_VIDEO_PROMPT = """You are analyzing a sequence of frames extracted from the same video.

The frames are provided in chronological order.

Describe what can actually be observed across the video.

Requirements:

1. Maintain chronological order.
2. Describe people, actions, objects, locations, vehicles, events, and visible text when relevant.
3. Identify meaningful changes between frames.
4. Do not invent events that cannot be established from the frames.
5. Do not infer a person's identity unless it is clearly supported by visible information.
6. Do not infer intent, motive, emotion, or context that is not visually supported.
7. If something is uncertain, explicitly describe it as uncertain.
8. Avoid repeating the same observation for every frame.
9. Combine consecutive frames showing the same event into one coherent description.
10. Mention important text visible in the frames.
11. Preserve the chronological progression of events.
12. Focus on useful factual observations rather than generic descriptions such as "this is a video."
13. Do not hallucinate missing frames or events between sampled frames.

Return valid JSON only, with exactly this structure:

{
  "description": "A chronological description of the video.",
  "summary": "A concise overall summary.",
  "events": [
    {"timestamp": 0.0, "description": "..."}
  ],
  "visible_text": ["..."]
}
"""

_OCR_PROMPT = """Extract all text that is visibly present in this image.

Rules:

1. Return only text that can actually be read from the image.
2. Do not invent, infer, or complete missing text.
3. Do not describe people, objects, scenery, or photographs.
4. Extract text embedded inside photographs, posters, graphics, screenshots, banners, memes, and social-media cards.
5. Preserve the original language.
6. Preserve reading order as much as possible.
7. Support Telugu, Hindi, English, and mixed-language text.
8. Do not translate the extracted text.
9. If text is partially unreadable, do not guess it.
10. Do not generate captions or descriptions of the image.
11. Do not treat visual objects as text.
12. Extract visible text even when it is overlaid on a photograph.
13. Avoid duplicate text when the same text appears more than once.

Return valid JSON only, with exactly this structure:

{
  "full_text": "...",
  "blocks": [
    {"text": "...", "type": "headline|body|label|caption|other"}
  ]
}
"""

_RETRY_NUDGE = ("Your previous response was not valid JSON matching the required "
                "structure. Reply again with ONLY the JSON object, no other text.")


class VLMError(RuntimeError):
    """vLLM could not be reached, timed out, or never returned usable JSON."""


@dataclass
class VideoDescription:
    description: str
    summary: str
    events: list[dict]
    visible_text: list[str]


@dataclass
class QwenOCRResult:
    full_text: str
    blocks: list[dict]   # [{"text": str, "type": str}, ...]


def _extract_json(text: str) -> object:
    """The model occasionally wraps output in ```json fences despite the
    prompt asking for bare JSON -- strip that before parsing."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _validate_video_schema(payload: object) -> VideoDescription:
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    description = payload.get("description")
    summary = payload.get("summary")
    events = payload.get("events", [])
    visible_text = payload.get("visible_text", [])
    if not isinstance(description, str) or not isinstance(summary, str):
        raise ValueError("'description' and 'summary' must be strings")
    if not isinstance(events, list) or not isinstance(visible_text, list):
        raise ValueError("'events' and 'visible_text' must be lists")

    clean_events = []
    for event in events:
        if not isinstance(event, dict) or "timestamp" not in event or "description" not in event:
            raise ValueError("each event needs 'timestamp' and 'description'")
        clean_events.append({"timestamp": float(event["timestamp"]),
                             "description": str(event["description"])})

    return VideoDescription(description=description, summary=summary,
                            events=clean_events, visible_text=[str(t) for t in visible_text])


_VALID_BLOCK_TYPES = {"headline", "body", "label", "caption", "other"}


def _validate_ocr_schema(payload: object) -> QwenOCRResult:
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    full_text = payload.get("full_text")
    blocks = payload.get("blocks", [])
    if not isinstance(full_text, str):
        raise ValueError("'full_text' must be a string")
    if not isinstance(blocks, list):
        raise ValueError("'blocks' must be a list")

    clean_blocks = []
    for block in blocks:
        if not isinstance(block, dict) or "text" not in block:
            raise ValueError("each block needs 'text'")
        block_type = block.get("type", "other")
        if block_type not in _VALID_BLOCK_TYPES:
            block_type = "other"
        clean_blocks.append({"text": str(block["text"]), "type": block_type})

    return QwenOCRResult(full_text=full_text, blocks=clean_blocks)


class VLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "",
                timeout: float = 120.0, max_json_retries: int = 1):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_json_retries = max_json_retries

    async def _send_chat_json(self, content: list[dict], validate: Callable[[object], T]) -> T:
        """POST one chat-completions request, parse the reply as JSON, and
        validate it with `validate`. Retries up to `max_json_retries` times
        with a nudge back at the model if parsing/validation fails --
        network/HTTP errors are not retried, only bad model output is."""
        messages: list[dict] = [{"role": "user", "content": content}]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_json_retries + 1):
                text = ""
                try:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json={"model": self.model, "messages": messages, "temperature": 0.2},
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    text = body["choices"][0]["message"]["content"]
                    parsed = _extract_json(text)
                    return validate(parsed)
                except httpx.HTTPError as exc:
                    raise VLMError(f"vLLM request failed: {exc}") from exc
                except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                    last_error = exc
                    log.warning("VLM returned unusable output (attempt %d/%d): %s",
                               attempt + 1, self.max_json_retries + 1, exc)
                    if attempt < self.max_json_retries:
                        messages.append({"role": "assistant", "content": text})
                        messages.append({"role": "user", "content": _RETRY_NUDGE})

        raise VLMError(f"vLLM never returned valid JSON after "
                       f"{self.max_json_retries + 1} attempt(s): {last_error}")

    async def describe_frames(self, frames: list[tuple[float, bytes]]) -> VideoDescription:
        """`frames`: (timestamp_seconds, jpeg_bytes) pairs, chronological."""
        content: list[dict] = [{"type": "text", "text": _VIDEO_PROMPT}]
        for ts, jpeg in frames:
            content.append({"type": "text", "text": f"Frame -- timestamp: {ts:.2f} seconds"})
            b64 = base64.b64encode(jpeg).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return await self._send_chat_json(content, _validate_video_schema)

    async def extract_text(self, image_bytes: bytes, media_type: str = "image/jpeg") -> QwenOCRResult:
        """One image in, OCR text out. `media_type` should match the actual
        encoding of `image_bytes` (jpeg/png/webp/...) -- vLLM decodes the
        data URI itself, it doesn't re-derive the format."""
        b64 = base64.b64encode(image_bytes).decode()
        content: list[dict] = [
            {"type": "text", "text": _OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
        ]
        return await self._send_chat_json(content, _validate_ocr_schema)

    async def health(self) -> bool:
        """Cheap liveness check -- lists models, never runs inference."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/models")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
