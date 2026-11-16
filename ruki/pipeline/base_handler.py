"""Threaded pipeline stage, adapted from speech_to_speech.baseHandler.BaseHandler.

Each stage owns an input and an output queue and runs `process()` on its own thread. Everything
`process()` yields goes to the output queue. Put PIPELINE_END in the input queue (and/or set the
stop event) to stop a stage; it forwards PIPELINE_END downstream so the next stage stops too.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from queue import Empty, Queue
from typing import Any, Iterator

from .messages import PIPELINE_END, CancelScope

logger = logging.getLogger(__name__)


class BaseHandler:
    def __init__(
        self,
        stop_event: threading.Event,
        queue_in: Queue,
        queue_out: Queue | None,
        cancel_scope: CancelScope,
        **setup_kwargs: Any,
    ) -> None:
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.cancel_scope = cancel_scope
        self.pending: deque = deque()  # items taken from queue_in early (see take_queued) go back here
        self.setup(**setup_kwargs)

    # --- to override ---------------------------------------------------------------------------
    def setup(self, **kwargs: Any) -> None:
        pass

    def process(self, item: Any) -> Iterator[Any]:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass

    # --- helpers -------------------------------------------------------------------------------
    def is_stale(self, item: Any) -> bool:
        return self.cancel_scope.is_stale(getattr(item, "generation", None))

    def take_queued(self) -> list[Any]:
        """Everything queued right now, without waiting (for coalescing inputs). Callers put what they do not
        use back with `self.pending.extend(...)`; the run loop serves `pending` before the queue."""
        items = list(self.pending)
        self.pending.clear()
        while True:
            try:
                items.append(self.queue_in.get_nowait())
            except Empty:
                return items

    def run(self) -> None:
        name = self.__class__.__name__
        logger.debug("%s: thread started", name)
        try:
            while not self.stop_event.is_set():
                if self.pending:
                    item = self.pending.popleft()
                else:
                    try:
                        item = self.queue_in.get(timeout=0.1)
                    except Empty:
                        continue
                if isinstance(item, bytes) and item == PIPELINE_END:
                    break
                if self.is_stale(item):  # produced before a barge-in: drop it
                    continue
                try:
                    for out in self.process(item):
                        if self.queue_out is not None and out is not None:
                            self.queue_out.put(out)
                except Exception:
                    logger.exception("%s: error in process()", name)
        finally:
            try:
                self.cleanup()
            finally:
                if self.queue_out is not None:
                    self.queue_out.put(PIPELINE_END)
            logger.debug("%s: thread stopped", name)


def drain(q: Queue) -> None:
    """Drop everything currently queued (used on barge-in), keeping a pending PIPELINE_END."""
    stop = False
    while True:
        try:
            item = q.get_nowait()
        except Empty:
            break
        if isinstance(item, bytes) and item == PIPELINE_END:
            stop = True
    if stop:
        q.put(PIPELINE_END)
