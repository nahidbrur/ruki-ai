"""Messages passed between pipeline stages (adapted from speech_to_speech.pipeline.messages)."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

# Sentinel that stops a handler thread (same idea as speech-to-speech's PIPELINE_END).
PIPELINE_END = b"END"


class CancelScope:
    """Monotonic generation counter. Bumping it invalidates every in-flight response (barge-in).

    Each message carries the generation it was produced under; stages drop or abort work whose
    generation is no longer current (mirrors speech_to_speech.pipeline.cancel_scope).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def cancel(self) -> int:
        with self._lock:
            self._generation += 1
            return self._generation

    def is_stale(self, generation: int | None) -> bool:
        return generation is not None and generation != self._generation


@dataclass
class TurnTimings:
    """Wall-clock marks (time.perf_counter) of one conversational turn, for latency reporting."""

    speech_end: float = 0.0       # VAD decided the user stopped talking
    asr_done: float = 0.0
    llm_first_token: float = 0.0
    first_chunk: float = 0.0      # first text chunk handed to TTS
    tts_first_audio: float = 0.0  # first audio sent to the client

    def report(self) -> dict[str, int]:
        def ms(a: float, b: float) -> int:
            return int(round((b - a) * 1000)) if a and b else -1

        return {
            "asr_ms": ms(self.speech_end, self.asr_done),
            "llm_first_token_ms": ms(self.asr_done, self.llm_first_token),
            "first_chunk_ms": ms(self.asr_done, self.first_chunk),
            "tts_first_audio_ms": ms(self.first_chunk, self.tts_first_audio),
            "voice_to_voice_ms": ms(self.speech_end, self.tts_first_audio),
        }


@dataclass
class Utterance:
    """VAD -> STT: one finished user utterance (16 kHz mono float32)."""

    audio: np.ndarray
    turn_id: int
    generation: int
    timings: TurnTimings = field(default_factory=TurnTimings)


@dataclass
class UserText:
    """STT (or typed text) -> LLM."""

    text: str
    turn_id: int
    generation: int
    timings: TurnTimings = field(default_factory=TurnTimings)


@dataclass
class TTSChunk:
    """LLM -> TTS: a speakable piece of the reply (one or more whole sentences)."""

    text: str
    turn_id: int
    generation: int
    timings: TurnTimings


@dataclass
class EndOfResponse:
    """LLM -> TTS: no more chunks for this turn."""

    turn_id: int
    generation: int
    timings: TurnTimings
