"""Windowed micro-batch. Tests call flush(); the server awaits submit()."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class Batcher(Generic[T, R]):
    """Collect items, then run `fn` once on the list.

    `flush` is the synchronous path used by tests and `--in-process` benches.
    `submit` is the async path: wait up to `window_ms` or until `max_batch`.
    """

    def __init__(self, fn: Callable[[list[T]], list[R]], max_batch: int = 16, window_ms: float = 8.0):
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self.fn = fn
        self.max_batch = max_batch
        self.window_ms = float(window_ms)
        self._queue: list[tuple[T, asyncio.Future]] = []
        self._flush_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def flush(self, items: Sequence[T]) -> list[R]:
        if not items:
            return []
        out = self.fn(list(items))
        if len(out) != len(items):
            raise RuntimeError(f"batch fn returned {len(out)} results for {len(items)} items")
        return out

    async def submit(self, item: T) -> R:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._lock:
            self._queue.append((item, fut))
            if len(self._queue) >= self.max_batch:
                self._run_flush()
            elif self._flush_task is None:
                self._flush_task = loop.create_task(self._wait_and_flush())
        return await fut

    async def _wait_and_flush(self) -> None:
        if self.window_ms > 0:
            await asyncio.sleep(self.window_ms / 1000.0)
        async with self._lock:
            self._run_flush()

    def _run_flush(self) -> None:
        pending = self._queue
        self._queue = []
        self._flush_task = None
        if not pending:
            return
        items = [item for item, _fut in pending]
        try:
            results = self.flush(items)
        except Exception as exc:
            for _item, fut in pending:
                if not fut.done():
                    fut.set_exception(exc)
            return
        for (_item, fut), result in zip(pending, results):
            if not fut.done():
                fut.set_result(result)


def timed_ms(fn: Callable[[], R]) -> tuple[R, float]:
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000.0
