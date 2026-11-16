"""Hima-ASR (Prahaladha/hima-asr-ONNX, int8) with ONNX Runtime on CPU.

Trimmed from the model repo's `hima_onnx.py`: transcription only (no decision head, no Hub
download), local folder `models/asr/hima-asr-onnx`. Audio encoder (from Qwen3-ASR) + Gemma-3-270M
text writer with a KV cache, greedy decoding with loop cutting.

License: Apache-2.0 plus the Gemma Terms of Use for the text decoder (see the model folder's NOTICE).
"""
from __future__ import annotations

import json
import re
import threading
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort

RATE, CHUNK = 16000, 100
LANGUAGE_NAMES = {"as": "assamese", "bn": "bengali", "en": "english", "gu": "gujarati", "hi": "hindi",
                  "kn": "kannada", "ml": "malayalam", "mr": "marathi", "or": "odia", "pa": "punjabi",
                  "ta": "tamil", "te": "telugu"}
_JOINERS = {"‌", "‍"}
_TAG = re.compile(r"<[^>]*>|\[[^\]]*\]")


def clean_text(text: str) -> str:
    """NFC, drop tags/punctuation, keep letters, marks, digits and zero-width joiners (as upstream)."""
    text = _TAG.sub(" ", unicodedata.normalize("NFC", text or ""))
    out = []
    for i, ch in enumerate(text):
        kind = unicodedata.category(ch)[0]
        if ch in _JOINERS:
            out.append(ch)
        elif ch in "'’" and 0 < i < len(text) - 1 and text[i - 1].isalpha() and text[i + 1].isalpha():
            out.append("'")
        elif kind in "LMN":
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).lower().split())


def _loop_length(p: list[int]) -> int:
    for size in range(1, 9):
        times = 6 if size == 1 else 4
        if len(p) >= size * times and p[-size * times:] == p[-size:] * times:
            return size
    for size in range(9, 81):
        if len(p) >= size * 3 and p[-size * 3:] == p[-size:] * 3:
            return size
    return 0


def _cut_loop(p: list[int]) -> list[int]:
    size = _loop_length(p)
    if size == 0:
        return p
    unit = p[-size:]
    while len(p) >= 2 * size and p[-2 * size:] == unit * 2:
        p = p[:-size]
    return p


def _log_mel(x: np.ndarray, filters: np.ndarray) -> tuple[np.ndarray, int]:
    """Same maths as Qwen3ASRFeatureExtractor. Returns (mel [128, T] padded to a multiple of 100, valid frames)."""
    if len(x) < 8000:
        x = np.pad(x, (0, 8000 - len(x)))
    n_fft, hop = 400, 160
    xp = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    spec = np.fft.rfft(xp[idx] * win, axis=1)
    mag = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)[:-1].T
    log = np.log10(np.maximum(filters.T @ mag, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    log = ((log + 4.0) / 4.0).astype(np.float32)
    n_valid = log.shape[1]
    return np.pad(log, ((0, 0), (0, (-n_valid) % CHUNK))), n_valid


def _post(n: int) -> int:
    for _ in range(3):
        n = (n - 1) // 2 + 1 if n > 0 else 0
    return n


class HimaASR:
    """Thread-safe: ONNX Runtime sessions allow concurrent run() calls."""

    def __init__(self, model_dir: str | Path, threads: int | None = 6, language: str | None = "bn",
                 style: str = "native") -> None:
        from tokenizers import Tokenizer

        d = Path(model_dir)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / "onnx" / "encoder_model_int8.onnx"), so, providers=providers)
        self.dec = ort.InferenceSession(str(d / "onnx" / "decoder_model_int8.onnx"), so, providers=providers)
        self.embed = np.load(d / "embed.npy").astype(np.float32)   # kept rows, already scaled
        self.keep = np.array(json.load(open(d / "keep_ids.json")))
        self.row = {int(t): i for i, t in enumerate(self.keep)}
        self.sot = np.load(d / "start_of_text.npy").astype(np.float32)
        self.filters = np.load(d / "mel_filters.npy")
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.bos, self.eos = 2, 1
        self.language, self.style = language, style
        self._lang_rows = {}
        self._lock = threading.Lock()  # one decode at a time keeps CPU threads from oversubscribing

    def _language_rows(self, language: str) -> np.ndarray:
        key = (language, self.style)
        if key not in self._lang_rows:
            ids = self.tok.encode("<%s %s>" % (LANGUAGE_NAMES[language], self.style), add_special_tokens=False).ids
            self._lang_rows[key] = self.embed[[self.row[i] for i in ids]]
        return self._lang_rows[key]

    def _listen(self, x: np.ndarray) -> np.ndarray:
        mel, n_valid = _log_mel(x, self.filters)
        n = sum(_post(min(CHUNK, n_valid - i * CHUNK)) for i in range(mel.shape[1] // CHUNK))
        sound, _layers = self.enc.run(None, {"mel": mel[None], "n_frames": np.array(n, dtype=np.int64)})
        return sound[:n]

    def _step(self, x, pos, pk, pv):
        return self.dec.run(None, {"x": x, "pos0": np.array(pos, dtype=np.int64), "past_k": pk, "past_v": pv})

    def _write(self, sound: np.ndarray, language: str | None, max_pieces: int = 512) -> str:
        parts = [self.embed[[self.row[self.bos]]], sound, self.sot[None]]
        if language:
            parts.append(self._language_rows(language))
        x = np.concatenate(parts).astype(np.float32)
        pk = np.zeros((18, 1, 0, 256), np.float32)
        pv = pk
        pos, out = 0, []
        limit = min(max_pieces, int(1.5 * sound.shape[0]) + 8)
        for _ in range(limit + 1):
            logits, pk, pv, _hidden = self._step(x, pos, pk, pv)
            pos += x.shape[0]
            j = int(np.argmax(logits))
            tok = int(self.keep[j])
            if tok == self.eos:
                break
            out.append(tok)
            if len(out) >= limit:
                break
            if _loop_length(out) > 0:
                out = _cut_loop(out)
                break
            x = self.embed[[j]]
        return clean_text(self.tok.decode(out, skip_special_tokens=False))

    def transcribe(self, audio: np.ndarray, language: str | None = "default") -> str:
        """audio: float32 mono 16 kHz. language: ISO code ("bn"), None = auto-detect, "default" = constructor's."""
        lang = self.language if language == "default" else language
        x = np.ascontiguousarray(audio, dtype=np.float32)
        with self._lock:
            return self._write(self._listen(x), lang)
