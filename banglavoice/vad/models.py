"""Frame-level audio models, ONNX Runtime only (no torch).

- SileroVAD: Silero VAD v6 (MIT), speech probability per 32 ms frame at 16 kHz.
- Denoiser:  GTCRN streaming speech enhancement (MIT), ~3-8 % of one CPU core.

Both are taken from the Bangla ASR project (stream_asr.py) and run single-threaded so many
sessions can share the CPU with the ASR.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime

from ..config import MODELS

SR = 16000
FRAME = 512  # 32 ms, the frame size Silero VAD expects at 16 kHz
VAD_PATH = MODELS / "vad" / "silero_vad_v6.onnx"
GTCRN_PATH = MODELS / "denoise" / "gtcrn_simple.onnx"


def _single_thread_session(path: Path) -> onnxruntime.InferenceSession:
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 4
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=opts)


class SileroVAD:
    """Silero VAD v6, one 512-sample frame per call. Keeps its own recurrent state."""

    CONTEXT = 64  # samples of the previous frame prepended to each frame

    def __init__(self, path: Path = VAD_PATH) -> None:
        self.session = _single_thread_session(path)
        self.reset_states()

    def reset_states(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame])[None].astype(np.float32)
        prob, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = frame[-self.CONTEXT:]
        return float(np.ravel(prob)[0])


class Denoiser:
    """GTCRN streaming enhancement: 512-point STFT, 256-sample hop, sqrt-Hann windows; output lags 16 ms."""

    HOP = 256

    def __init__(self, path: Path = GTCRN_PATH) -> None:
        self.session = _single_thread_session(path)
        self.window = np.sqrt(np.hanning(2 * self.HOP + 1)[:-1]).astype(np.float32)  # periodic Hann
        self.caches = {
            "conv_cache": np.zeros((2, 1, 16, 16, 33), dtype=np.float32),
            "tra_cache": np.zeros((2, 3, 1, 1, 16), dtype=np.float32),
            "inter_cache": np.zeros((2, 1, 33, 16), dtype=np.float32),
        }
        self.prev_in = np.zeros(self.HOP, dtype=np.float32)
        self.prev_out = np.zeros(self.HOP, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(frame), self.HOP):
            hop = frame[i:i + self.HOP]
            spec = np.fft.rfft(np.concatenate([self.prev_in, hop]) * self.window)
            self.prev_in = hop
            mix = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)[None, :, None, :]
            enh, conv, tra, inter = self.session.run(None, {"mix": mix, **self.caches})
            self.caches = {"conv_cache": conv, "tra_cache": tra, "inter_cache": inter}
            y = np.fft.irfft(enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1], 2 * self.HOP) * self.window
            out.append(self.prev_out + y[:self.HOP])  # overlap-add
            self.prev_out = y[self.HOP:]
        return np.concatenate(out).astype(np.float32)
