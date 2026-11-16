"""Utterance segmentation from per-frame speech probabilities.

Port of speech_to_speech.VAD.vad_iterator.VADIterator (itself from snakers4/silero-vad) to numpy:
the model call is taken out (the caller passes the probability), so the same iterator works with
denoised audio and with the barge-in logic in VADHandler. Adds a maximum utterance length.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class VADIterator:
    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 600,
        speech_pad_ms: int = 200,
        max_speech_s: float = 25.0,
    ) -> None:
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
        self.max_speech_samples = int(sampling_rate * max_speech_s)
        self._pre: deque[np.ndarray] = deque()
        self.reset_states()

    def reset_states(self) -> None:
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.buffer: list[np.ndarray] = []
        self.buffer_samples = 0
        self.active_speech_samples = 0  # frames above threshold in the current utterance
        self.last_active_speech_samples = 0
        self._pre.clear()
        self._pre_samples = 0

    def _remember_pre_speech(self, x: np.ndarray) -> None:
        self._pre.append(x)
        self._pre_samples += len(x)
        while self._pre and self._pre_samples - len(self._pre[0]) >= self.speech_pad_samples:
            self._pre_samples -= len(self._pre.popleft())

    def _finish(self) -> np.ndarray:
        audio = np.concatenate(self.buffer) if self.buffer else np.zeros(0, np.float32)
        self.last_active_speech_samples = self.active_speech_samples
        self.triggered = False
        self.temp_end = 0
        self.buffer = []
        self.buffer_samples = 0
        self.active_speech_samples = 0
        return audio

    def __call__(self, x: np.ndarray, speech_prob: float) -> np.ndarray | None:
        """Feed one frame and its speech probability. Returns the utterance audio when one ends."""
        n = len(x)
        self.current_sample += n

        if not self.triggered:
            if speech_prob >= self.threshold:
                self.triggered = True
                self.buffer = list(self._pre) + [x]  # keep speech_pad_ms of lead-in
                self.buffer_samples = self._pre_samples + n
                self._pre.clear()
                self._pre_samples = 0
                self.active_speech_samples = n
            else:
                self._remember_pre_speech(x)
            return None

        self.buffer.append(x)
        self.buffer_samples += n
        if speech_prob >= self.threshold - 0.15:
            self.active_speech_samples += n
            self.temp_end = 0
        else:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end >= self.min_silence_samples:
                # Trim the trailing silence beyond the pad.
                extra = (self.current_sample - self.temp_end) - self.speech_pad_samples
                audio = self._finish()
                return audio[: len(audio) - extra] if extra > 0 else audio

        if self.buffer_samples >= self.max_speech_samples:
            return self._finish()  # very long monologue: hand it over anyway
        return None
