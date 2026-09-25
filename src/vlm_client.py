"""httpx client for vLLM's OpenAI-compatible chat completions API.

vLLM is a separate, already-installed long-running process serving
Qwen2.5-VL-7B-Instruct (its own Python, its own GPU allocation) -- this
module only ever talks to it over HTTP, the same way any OpenAI-API client
would. No model is loaded inside this process; the FastAPI app and vLLM are
two independent GPU-resident resources, coordinated only through this HTTP
call.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger("ocr.vlm")

_PROMPT = """You are analyzing a sequence of frames extracted from the same video.

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


def _extract_json(text: str) -> object:
    """The model occasionally wraps output in ```json fences despite the
    prompt asking for bare JSON -- strip that before parsing."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _validate_schema(payload: object) -> VideoDescription:
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


class VLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "",
                timeout: float = 120.0, max_json_retries: int = 1):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_json_retries = max_json_retries

    async def describe_frames(self, frames: list[tuple[float, bytes]]) -> VideoDescription:
        """`frames`: (timestamp_seconds, jpeg_bytes) pairs, chronological."""
        content: list[dict] = [{"type": "text", "text": _PROMPT}]
        for ts, jpeg in frames:
            content.append({"type": "text", "text": f"Frame -- timestamp: {ts:.2f} seconds"})
            b64 = base64.b64encode(jpeg).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

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
                    return _validate_schema(parsed)
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

    async def health(self) -> bool:
        """Cheap liveness check -- lists models, never runs inference."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/models")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
