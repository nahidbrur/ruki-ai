"""One conversation: four handler threads joined by queues, plus the state they share.

Mirrors speech_to_speech.s2s_pipeline (queues + ThreadManager), but one pipeline per WebSocket
session while the heavy models (ASR, TTS, LLM client) are loaded once and shared.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable

import numpy as np

from .config import Settings
from .llm.text import SentenceChunker
from .pipeline.base_handler import drain
from .pipeline.handlers import LLMHandler, STTHandler, TTSHandler, VADHandler
from .pipeline.messages import PIPELINE_END, CancelScope, EndOfResponse, TTSChunk, TurnTimings, UserText
from .vad.models import FRAME

logger = logging.getLogger(__name__)

TTS_RATE = 24000
PLAYBACK_MARGIN_S = 0.3  # client-side buffering on top of the audio length


@dataclass
class Models:
    asr: Any
    tts: Any
    llm: Any


@dataclass
class SessionOptions:
    voice: str
    allow_interruptions: bool
    tts_steps: int


class Session:
    def __init__(self, models: Models, settings: Settings, options: SessionOptions,
                 send: Callable[[dict | bytes], None]) -> None:
        self.settings = settings
        self.options = options
        self._send = send
        self.cancel_scope = CancelScope()
        self.stop_event = threading.Event()
        self._lock = threading.RLock()  # orders audio vs. interrupt messages, guards history

        self.history: list[dict] = []          # {"role", "text", "gen"}
        self._spoken: dict[int, list[str]] = {}
        self._carry: list[str] = []
        self._turn_id = 0
        self._busy_gen: int | None = None      # generation of the turn being answered
        self._playback_until = 0.0
        self._audio_started: set[int] = set()  # turn ids whose reply audio has started
        self._frame_buf = np.zeros(0, dtype=np.float32)

        self.q_audio: Queue = Queue()
        self.q_stt: Queue = Queue()
        self.q_llm: Queue = Queue()
        self.q_tts: Queue = Queue()
        common = dict(stop_event=self.stop_event, cancel_scope=self.cancel_scope)
        self.handlers = [
            VADHandler(queue_in=self.q_audio, queue_out=self.q_stt, session=self, **common),
            STTHandler(queue_in=self.q_stt, queue_out=self.q_llm, session=self, asr=models.asr, **common),
            LLMHandler(queue_in=self.q_llm, queue_out=self.q_tts, session=self, llm=models.llm, **common),
            TTSHandler(queue_in=self.q_tts, queue_out=None, session=self, tts=models.tts, **common),
        ]
        self.threads = [threading.Thread(target=h.run, name=type(h).__name__, daemon=True) for h in self.handlers]
        for t in self.threads:
            t.start()
        self.set_status("listening")

    # --- input --------------------------------------------------------------------------------
    def feed_audio(self, pcm: np.ndarray) -> None:
        """16 kHz mono float32 of any length; cut into the 512-sample frames the VAD expects."""
        buf = np.concatenate([self._frame_buf, pcm.astype(np.float32, copy=False)])
        n = len(buf) // FRAME * FRAME
        for i in range(0, n, FRAME):
            self.q_audio.put(buf[i:i + FRAME])
        self._frame_buf = buf[n:]

    def submit_text(self, text: str) -> None:
        """Typed input skips VAD and ASR."""
        text = text.strip()
        if not text:
            return
        if self.assistant_active():
            self.interrupt("text")
        gen = self.cancel_scope.generation
        self.begin_turn(gen)
        turn = self.next_turn_id()
        self.emit({"type": "user_text", "turn_id": turn, "text": text, "typed": True})
        self.q_llm.put(UserText(text=text, turn_id=turn, generation=gen, timings=TurnTimings()))

    def greet(self) -> None:
        """Speak the welcome message (straight to TTS, no LLM) and keep it in the history, so the LLM
        knows it has already greeted the user."""
        text = self.settings.welcome_message
        if not text:
            return
        gen = self.cancel_scope.generation
        self.begin_turn(gen)
        turn = self.next_turn_id()
        with self._lock:
            self.history.append({"role": "assistant", "text": text, "gen": gen})
        self.emit({"type": "assistant_delta", "turn_id": turn, "text": text, "welcome": True}, gen=gen)
        timings = TurnTimings()
        chunker = SentenceChunker(first_min_chars=self.settings.first_chunk_min_chars)
        for sentence in [*chunker.push(text), *chunker.flush()]:
            self.q_tts.put(TTSChunk(sentence, turn, gen, timings))
        self.q_tts.put(EndOfResponse(turn, gen, timings))

    def update_options(self, **changes: Any) -> None:
        for k, v in changes.items():
            if v is not None and hasattr(self.options, k):
                setattr(self.options, k, v)

    # --- state shared with the handlers ---------------------------------------------------------
    def next_turn_id(self) -> int:
        with self._lock:
            self._turn_id += 1
            return self._turn_id

    def assistant_active(self) -> bool:
        return (self._busy_gen == self.cancel_scope.generation
                or time.perf_counter() < self._playback_until + PLAYBACK_MARGIN_S)

    def begin_turn(self, gen: int) -> None:
        if not self.cancel_scope.is_stale(gen):
            self._busy_gen = gen

    def end_turn(self, gen: int) -> None:
        if self._busy_gen == gen:
            self._busy_gen = None

    def carry_over(self, text: str) -> None:
        with self._lock:
            self._carry.append(text)

    def add_user_message(self, text: str) -> list[dict]:
        """Append the user's words (merged with carried-over or unanswered words) and return the history."""
        with self._lock:
            if self._carry:
                text = " ".join(self._carry + [text])
                self._carry.clear()
            if self.history and self.history[-1]["role"] == "user":
                self.history[-1]["text"] += " " + text  # previous turn got no answer (interrupted early)
            else:
                self.history.append({"role": "user", "text": text, "gen": None})
            max_msgs = 2 * self.settings.max_history_turns
            if len(self.history) > max_msgs:
                del self.history[: len(self.history) - max_msgs]
                while self.history and self.history[0]["role"] != "user":
                    self.history.pop(0)
            return [dict(m) for m in self.history]

    def commit_assistant(self, gen: int, full_text: str) -> None:
        """Called when the LLM finishes or stops. If the reply was interrupted, keep only what was spoken."""
        with self._lock:
            text = full_text if not self.cancel_scope.is_stale(gen) else " ".join(self._spoken.get(gen, []))
            if text:
                self.history.append({"role": "assistant", "text": text, "gen": gen})

    def mark_spoken(self, gen: int, text: str) -> None:
        with self._lock:
            self._spoken.setdefault(gen, []).append(text)

    def interrupt(self, reason: str) -> None:
        with self._lock:
            old = self.cancel_scope.generation
            self.cancel_scope.cancel()
            self._busy_gen = None
            self._playback_until = 0.0
            drain(self.q_llm)
            drain(self.q_tts)
            if self.history and self.history[-1]["role"] == "assistant" and self.history[-1]["gen"] == old:
                spoken = " ".join(self._spoken.get(old, []))
                if spoken:
                    self.history[-1]["text"] = spoken
                else:
                    self.history.pop()
            self._spoken = {g: v for g, v in self._spoken.items() if g > old}
            logger.info("interrupted (%s)", reason)
            self._send({"type": "interrupt", "reason": reason})

    # --- output ---------------------------------------------------------------------------------
    def emit(self, msg: dict, gen: int | None = None) -> None:
        with self._lock:
            if gen is not None and self.cancel_scope.is_stale(gen):
                return
            self._send(msg)

    def set_status(self, state: str, gen: int | None = None) -> None:
        self.emit({"type": "status", "state": state}, gen=gen)

    def send_audio(self, gen: int, audio_24k: np.ndarray, chunk: TTSChunk) -> None:
        pcm = (np.clip(audio_24k, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        with self._lock:
            if self.cancel_scope.is_stale(gen):
                return
            now = time.perf_counter()
            if chunk.turn_id not in self._audio_started:
                self._audio_started.add(chunk.turn_id)
                chunk.timings.tts_first_audio = now
                self._send({"type": "status", "state": "speaking"})
                self._send({"type": "audio_start", "turn_id": chunk.turn_id, "sample_rate": TTS_RATE})
                self._send({"type": "metrics", "turn_id": chunk.turn_id, **chunk.timings.report()})
                logger.info("turn %d latency %s", chunk.turn_id, chunk.timings.report())
            self._send(pcm)
            # when the client will have played everything sent so far (it starts each piece on arrival at the latest)
            self._playback_until = max(self._playback_until, now) + len(audio_24k) / TTS_RATE

    def end_response(self, item: EndOfResponse) -> None:
        with self._lock:
            if self.cancel_scope.is_stale(item.generation):
                return
            self.end_turn(item.generation)
            self._send({"type": "assistant_done", "turn_id": item.turn_id, **item.timings.report()})
            if item.turn_id not in self._audio_started:
                self._send({"type": "status", "state": "listening"})

    # --- teardown -------------------------------------------------------------------------------
    def close(self) -> None:
        self.cancel_scope.cancel()  # stops any running LLM stream / TTS synthesis
        self.stop_event.set()
        self.q_audio.put(PIPELINE_END)
        for t in self.threads:
            t.join(timeout=5)
