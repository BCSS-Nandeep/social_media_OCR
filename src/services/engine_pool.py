"""Pre-built OCR engines, checked out one thread at a time.

PaddleOCR predictors are **not thread-safe** — two threads calling `.predict()`
on the same instance race inside the native runtime. And in auto mode
`OCREngine` mutates its own state per image (the chosen script, releasing the
previous pipeline), so sharing one engine across a ThreadPoolExecutor is wrong
twice over.

Nor can each request build its own: constructing a pipeline costs seconds, and
this machine segfaults once too many are resident at the same time.

So: a fixed number of engines are built at startup and handed out one at a
time. Models load exactly once and are reused for every inference, concurrency
equals pool size, and memory is bounded no matter how many requests arrive.
"""

from __future__ import annotations

import logging
import queue
import threading
from contextlib import contextmanager
from typing import Iterator

import numpy as np

from ..config import settings
from ..ocr_engine import OCREngine
from ..scripts import resolve_models

log = logging.getLogger(__name__)


class EngineBusy(RuntimeError):
    """Every engine was in use for longer than the caller was willing to wait."""


def build_engine(lang: str) -> OCREngine:
    """One engine configured exactly like the CLI's defaults."""
    auto = lang.lower() == "auto"
    langs = (["auto"] if auto
             else [script for script, _ in resolve_models(lang.split("+"))])
    return OCREngine(
        langs,
        use_gpu=settings.use_gpu,
        auto_mode=auto,
        # Filter once, above this layer, so reported block counts stay honest.
        drop_score=0.0,
    )


def warm(engine: OCREngine) -> None:
    """Force model loading now rather than inside the first request.

    Failures here are logged, not raised: a cold engine still works, it is just
    slower on its first image, and refusing to start the service over that
    would be the worse outcome.
    """
    blank = np.full((64, 256, 3), 255, dtype=np.uint8)
    try:
        engine.warmup()
        if engine.auto_mode:
            # warmup() deliberately skips auto mode, so load the probe models
            # explicitly instead of paying for them on the first real request.
            engine.detector.detect(blank)
    except Exception:                                   # noqa: BLE001
        log.warning("warmup failed; models will load on first use", exc_info=True)


class EnginePool:
    """A fixed set of engines for one language spec."""

    def __init__(self, lang: str, size: int):
        self.lang = lang
        self.size = max(1, size)
        self._free: queue.LifoQueue[OCREngine] = queue.LifoQueue()
        self._built = 0
        self._lock = threading.Lock()

    def prime(self, warmup: bool = True) -> None:
        for index in range(self.size):
            engine = build_engine(self.lang)
            if warmup:
                warm(engine)
            self._free.put(engine)
            with self._lock:
                self._built += 1
            log.info("engine ready lang=%s (%d/%d)", self.lang, index + 1, self.size)

    @property
    def available(self) -> int:
        return self._free.qsize()

    @contextmanager
    def acquire(self, timeout: float | None = None) -> Iterator[OCREngine]:
        """Borrow an engine. Always returned, even if the caller raises."""
        try:
            engine = self._free.get(timeout=timeout if timeout is not None
                                    else settings.acquire_timeout)
        except queue.Empty as exc:
            raise EngineBusy(
                f"No OCR engine free for lang={self.lang!r} within "
                f"{settings.acquire_timeout:g}s. Raise OCR_POOL_SIZE or retry."
            ) from exc
        try:
            yield engine
        finally:
            self._free.put(engine)


class EngineRegistry:
    """Pools keyed by language spec, built on demand and then kept.

    Bounded by ``max_pools``: each pool pins several models in memory, so an
    unbounded registry would let request input drive memory growth.
    """

    def __init__(self) -> None:
        self._pools: dict[str, EnginePool] = {}
        self._lock = threading.Lock()

    def startup(self) -> None:
        lang = settings.default_lang
        log.info("loading OCR models lang=%s pool=%d gpu=%s",
                 lang, settings.pool_size, settings.use_gpu)
        self.pool(lang)
        log.info("OCR service ready")

    def pool(self, lang: str) -> EnginePool:
        key = lang.strip().lower()
        with self._lock:
            existing = self._pools.get(key)
            if existing is not None:
                return existing
            if len(self._pools) >= settings.max_pools:
                raise RuntimeError(
                    f"Refusing to load a {len(self._pools) + 1}th language pool "
                    f"({key!r}); OCR_MAX_POOLS={settings.max_pools}. Each pool "
                    f"pins its own models in memory. Loaded: "
                    f"{', '.join(sorted(self._pools))}."
                )
            pool = EnginePool(key, settings.pool_size)
            self._pools[key] = pool

        # Built outside the lock: loading models takes seconds and must not
        # block requests already using a different language.
        pool.prime(warmup=settings.warmup_on_startup)
        return pool

    @property
    def loaded(self) -> list[str]:
        with self._lock:
            return sorted(self._pools)

    def status(self) -> dict[str, int]:
        with self._lock:
            return {name: p.available for name, p in self._pools.items()}

    def shutdown(self) -> None:
        with self._lock:
            self._pools.clear()


registry = EngineRegistry()
