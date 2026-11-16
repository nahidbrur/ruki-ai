"""The four pipeline stages. Each runs on its own thread (see BaseHandler) and talks to its Session
for shared state: conversation history, barge-in, and sending events/audio to the client.

    VADHandler  : 512-sample frames -> Utterance        (GTCRN denoise, Silero VAD, barge-in)
    STTHandler  : Utterance -> UserText                 (Hima-ASR)
    LLMHandler  : UserText -> TTSChunk..., EndOfResponse (Gemini, streamed, cut into sentences)
    TTSHandler  : TTSChunk -> audio to the client       (OmniVoice, streamed)
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterator

import numpy as np

from ..llm.text import SentenceChunker
from ..vad.models import FRAME, SR, Denoiser, SileroVAD
from ..vad.vad_iterator import VADIterator
from .base_handler import BaseHandler
from .messages import EndOfResponse, TTSChunk, TurnTimings, Utterance, UserText

if TYPE_CHECKING:
    from ..session import Session

logger = logging.getLogger(__name__)


class VADHandler(BaseHandler):
    def setup(self, session: "Session") -> None:
        s = session.settings
        self.session = session
        self.vad = SileroVAD()
        self.denoiser = Denoiser() if s.denoise else None
        self.iterator = VADIterator(threshold=s.vad_threshold, sampling_rate=SR,
                                    min_silence_duration_ms=s.vad_min_silence_ms,
                                    speech_pad_ms=s.vad_speech_pad_ms, max_speech_s=s.vad_max_speech_s)
        self.min_speech_samples = int(SR * s.vad_min_speech_ms / 1000)
        self.barge_samples = int(SR * s.barge_in_min_ms / 1000)
        self.barge_threshold = s.barge_in_threshold
        self.barge_run = 0
        self.started_while_active = False
        self.barged = False

    def is_stale(self, item) -> bool:
        return False  # raw audio frames have no generation

    def process(self, frame: np.ndarray) -> Iterator[Utterance]:
        session = self.session
        x = self.denoiser(frame) if self.denoiser is not None else frame
        prob = self.vad(x)
        active = session.assistant_active()

        if active and not session.options.allow_interruptions:
            # Half duplex (like speech-to-speech's should_listen): ignore the mic while the assistant talks.
            if self.iterator.triggered:
                self.iterator.reset_states()
                session.emit({"type": "vad", "speaking": False})
            self.barge_run = 0
            return

        was_triggered = self.iterator.triggered
        utterance = self.iterator(x, prob)

        if not was_triggered and self.iterator.triggered:
            self.started_while_active = active
            self.barged = False
            self.barge_run = 0
            if not active:
                session.emit({"type": "vad", "speaking": True})

        # Barge-in: sustained, confident speech while the assistant is busy stops it.
        if active and self.iterator.triggered and not self.barged:
            self.barge_run = self.barge_run + len(x) if prob >= self.barge_threshold else 0
            if self.barge_run >= self.barge_samples:
                self.barged = True
                keep = self.barge_samples + self.iterator.speech_pad_samples
                buf = np.concatenate(self.iterator.buffer)[-keep:]  # drop what came before (likely echo)
                self.iterator.buffer, self.iterator.buffer_samples = [buf], len(buf)
                session.interrupt("barge-in")
                session.emit({"type": "vad", "speaking": True})

        if utterance is None:
            return
        echo_only = self.started_while_active and not self.barged
        self.started_while_active = self.barged = False
        if echo_only:
            return  # speech heard during the assistant's turn that never qualified as a barge-in
        session.emit({"type": "vad", "speaking": False})
        if self.iterator.last_active_speech_samples < self.min_speech_samples:
            session.set_status("listening")
            return  # a click or a cough
        timings = TurnTimings(speech_end=time.perf_counter())
        gen = self.cancel_scope.generation
        session.begin_turn(gen)
        yield Utterance(audio=utterance, turn_id=session.next_turn_id(), generation=gen, timings=timings)


class STTHandler(BaseHandler):
    def setup(self, session: "Session", asr) -> None:
        self.session = session
        self.asr = asr

    def is_stale(self, item) -> bool:
        return False  # always transcribe: words cut off by a barge-in are carried into the next turn

    def process(self, utt: Utterance) -> Iterator[UserText]:
        session = self.session
        session.set_status("transcribing", gen=utt.generation)
        t0 = time.perf_counter()
        text = self.asr.transcribe(utt.audio)
        utt.timings.asr_done = time.perf_counter()
        logger.info("ASR %.2fs audio in %.0f ms: %s", len(utt.audio) / SR, (utt.timings.asr_done - t0) * 1000, text)
        if sum(ch.isalnum() for ch in text) < 2:
            session.end_turn(utt.generation)
            session.set_status("listening")
            return
        session.emit({"type": "user_text", "turn_id": utt.turn_id, "text": text,
                      "audio_s": round(len(utt.audio) / SR, 2),
                      "asr_ms": int((utt.timings.asr_done - t0) * 1000)})
        if self.cancel_scope.is_stale(utt.generation):
            session.carry_over(text)  # the user kept talking (barge-in): prepend to their next turn
            return
        yield UserText(text=text, turn_id=utt.turn_id, generation=utt.generation, timings=utt.timings)


class LLMHandler(BaseHandler):
    def setup(self, session: "Session", llm) -> None:
        self.session = session
        self.llm = llm

    def process(self, item: UserText) -> Iterator[TTSChunk | EndOfResponse]:
        session, gen, timings = self.session, item.generation, item.timings
        if not timings.asr_done:
            timings.asr_done = time.perf_counter()  # typed input
        session.begin_turn(gen)
        session.set_status("thinking", gen=gen)
        history = session.add_user_message(item.text)

        def stale() -> bool:
            return self.cancel_scope.is_stale(gen) or self.stop_event.is_set()

        chunker = SentenceChunker(first_min_chars=session.settings.first_chunk_min_chars)
        parts: list[str] = []
        try:
            for delta in self.llm.stream(history, should_stop=stale):
                if not timings.llm_first_token:
                    timings.llm_first_token = time.perf_counter()
                parts.append(delta)
                session.emit({"type": "assistant_delta", "turn_id": item.turn_id, "text": delta}, gen=gen)
                for sentence in chunker.push(delta):
                    if not timings.first_chunk:
                        timings.first_chunk = time.perf_counter()
                    yield TTSChunk(sentence, item.turn_id, gen, timings)
            if stale():
                return
            for sentence in chunker.flush():
                if not timings.first_chunk:
                    timings.first_chunk = time.perf_counter()
                yield TTSChunk(sentence, item.turn_id, gen, timings)
        except Exception as e:
            logger.exception("Gemini request failed")
            session.emit({"type": "error", "message": f"LLM error: {e}"}, gen=gen)
        finally:
            session.commit_assistant(gen, "".join(parts).strip())
        if not stale():
            yield EndOfResponse(item.turn_id, gen, timings)


def soft_limit(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Leave the signal untouched below `knee`, squash peaks above it smoothly into (knee, 1)."""
    a = np.abs(x)
    over = a > knee
    if not over.any():
        return x
    y = x.copy()
    head = 1.0 - knee
    y[over] = np.sign(x[over]) * (knee + head * np.tanh((a[over] - knee) / head))
    return y


class TTSHandler(BaseHandler):
    def setup(self, session: "Session", tts) -> None:
        self.session = session
        self.tts = tts
        self.current_turn = None
        self.chunk_index = 0  # chunks synthesized so far in the current turn

    def _coalesce(self, first: TTSChunk) -> str:
        """Merge sentences that are already queued for the same turn (fewer, longer TTS calls are more
        efficient: each OmniVoice step has a fixed GPU cost). Never for a turn's first chunk: it must start fast."""
        if self.current_turn != (first.turn_id, first.generation):
            self.current_turn = (first.turn_id, first.generation)
            self.chunk_index = 0
            return first.text
        texts, rest = [first.text], []
        limit = self.session.settings.tts_max_chars
        for item in self.take_queued():
            if (not rest and isinstance(item, TTSChunk) and item.generation == first.generation
                    and sum(map(len, texts)) + len(item.text) <= limit):
                texts.append(item.text)
            else:
                rest.append(item)
        self.pending.extend(rest)
        return " ".join(texts)

    def process(self, item: TTSChunk | EndOfResponse) -> Iterator[None]:
        session = self.session
        if isinstance(item, EndOfResponse):
            session.end_response(item)
            return
        gen = item.generation
        text = self._coalesce(item)
        opts, s = session.options, session.settings
        # Ramp-up: a turn's first chunks are short and each step has a fixed GPU cost, so they synthesize slower
        # than real time; fewer steps there keep the start fast and the gap before the next chunk small.
        fast = self.chunk_index < s.tts_fast_chunks
        steps = min(opts.tts_steps, s.tts_first_steps) if fast else opts.tts_steps
        self.chunk_index += 1

        def stale() -> bool:
            return self.cancel_scope.is_stale(gen) or self.stop_event.is_set()

        t0 = time.perf_counter()
        n = 0
        for piece in self.tts.stream(text, lang=s.tts_language, voice=opts.voice, steps=steps,
                                     chunk_sec=s.tts_chunk_sec, should_stop=stale):
            if stale():
                break
            if n == 0:
                session.mark_spoken(gen, text)
            n += len(piece)
            session.send_audio(gen, soft_limit(piece), item)
        dt = time.perf_counter() - t0
        if n:
            logger.info("TTS %.2fs audio in %.2fs (RTF %.2f): %s", n / 24000, dt, dt / (n / 24000), text)
        return
        yield  # generator
