"""OmniVoice TTS through omnivoice.cpp (libomnivoice.so, GGML + Vulkan), driven with ctypes.

Adapted from TTS/omnivoiceTTS/omnivoice_tts.py:
- the shared libraries are loaded from models/tts/lib (preloaded in dependency order, so the build
  folder's absolute RUNPATH is not needed);
- several reference voices are encoded once at start-up and kept, and each call picks one;
- one lock serialises GPU work, so several sessions can share one model context.

Output: float32 mono, 24 kHz.
"""
from __future__ import annotations

import contextlib
import ctypes as C
import logging
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
VOICE_EXTS = (".wav", ".mp3", ".flac", ".ogg")  # reference voice formats (read by soundfile/libsndfile)
OV_LOG_WARN = 2
_LIB_ORDER = ("libggml-base.so.0", "libggml-cpu.so.0", "libggml-vulkan.so.0", "libggml.so.0", "libomnivoice.so")


# --- ctypes mirror of omnivoice.cpp/src/omnivoice.h (ABI version 4) --------------------------------
class _Audio(C.Structure):
    _fields_ = [("samples", C.POINTER(C.c_float)), ("n_samples", C.c_int),
                ("sample_rate", C.c_int), ("channels", C.c_int)]


class _InitParams(C.Structure):
    _fields_ = [("abi_version", C.c_int), ("model_path", C.c_char_p), ("codec_path", C.c_char_p),
                ("use_fa", C.c_bool), ("clamp_fp16", C.c_bool), ("device", C.c_char_p)]


_CANCEL_CB = C.CFUNCTYPE(C.c_bool, C.c_void_p)
_CHUNK_CB = C.CFUNCTYPE(C.c_bool, C.POINTER(C.c_float), C.c_int, C.c_void_p)
_LOG_CB = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)


class _TTSParams(C.Structure):
    _fields_ = [
        ("abi_version", C.c_int),
        ("text", C.c_char_p), ("lang", C.c_char_p), ("instruct", C.c_char_p),
        ("T_override", C.c_int), ("chunk_duration_sec", C.c_float), ("chunk_threshold_sec", C.c_float),
        ("denoise", C.c_bool), ("preprocess_prompt", C.c_bool),
        ("mg_num_step", C.c_int), ("mg_guidance_scale", C.c_float), ("mg_t_shift", C.c_float),
        ("mg_layer_penalty_factor", C.c_float), ("mg_position_temperature", C.c_float),
        ("mg_class_temperature", C.c_float), ("mg_seed", C.c_uint64),
        ("ref_audio_tokens", C.POINTER(C.c_int32)), ("ref_T", C.c_int),
        ("ref_audio_24k", C.POINTER(C.c_float)), ("ref_n_samples", C.c_int), ("ref_text", C.c_char_p),
        ("dump_dir", C.c_char_p),
        ("cancel", _CANCEL_CB), ("cancel_user_data", C.c_void_p),
        ("on_chunk", _CHUNK_CB), ("on_chunk_user_data", C.c_void_p),
        ("postproc", C.c_bool),
    ]


class _VoiceRef(C.Structure):
    _fields_ = [("ref_codes", C.POINTER(C.c_int32)), ("ref_T", C.c_int), ("num_codebooks", C.c_int)]


def _load_lib(lib_dir: Path):
    lib = None
    for name in _LIB_ORDER:  # dependencies first; the loader then reuses them by soname
        lib = C.CDLL(str(lib_dir / name), mode=C.RTLD_GLOBAL)
    lib.ov_last_error.restype = C.c_char_p
    lib.ov_init_default_params.argtypes = [C.POINTER(_InitParams)]
    lib.ov_init.argtypes = [C.POINTER(_InitParams)]
    lib.ov_init.restype = C.c_void_p
    lib.ov_free.argtypes = [C.c_void_p]
    lib.ov_tts_default_params.argtypes = [C.POINTER(_TTSParams)]
    lib.ov_synthesize.argtypes = [C.c_void_p, C.POINTER(_TTSParams), C.POINTER(_Audio)]
    lib.ov_synthesize.restype = C.c_int
    lib.ov_audio_free.argtypes = [C.POINTER(_Audio)]
    lib.ov_extract_voice_ref.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int, C.POINTER(_VoiceRef)]
    lib.ov_extract_voice_ref.restype = C.c_int
    lib.ov_voice_ref_free.argtypes = [C.POINTER(_VoiceRef)]
    lib.ov_log_set.argtypes = [_LOG_CB, C.c_void_p]
    return lib


def load_audio_24k(path: str | Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


class OmniVoiceEngine:
    def __init__(self, tts_dir: str | Path, device: str = "Vulkan1", verbose: bool = False) -> None:
        tts_dir = Path(tts_dir)
        self._lib = _load_lib(tts_dir / "lib")
        self._verbose = verbose
        self._stderr_fd = os.dup(2)  # ggml sometimes writes straight to fd 2; _quiet() hides it

        def _log(level, msg, _user):
            if level >= OV_LOG_WARN or verbose:
                os.write(self._stderr_fd, msg + b"\n")

        self._log_cb = _LOG_CB(_log)  # keep a reference for the process lifetime
        self._lib.ov_log_set(self._log_cb, None)

        ip = _InitParams()
        self._lib.ov_init_default_params(C.byref(ip))
        ip.model_path = str(tts_dir / "omnivoice-base-Q8_0.gguf").encode()
        ip.codec_path = str(tts_dir / "omnivoice-tokenizer-Q8_0.gguf").encode()
        ip.device = device.encode() if device else None
        with self._quiet():
            self._ctx = self._lib.ov_init(C.byref(ip))
        if not self._ctx:
            raise RuntimeError(f"ov_init failed on {device}: {self._err()}")
        self.device = device
        self.voices: dict[str, tuple[_VoiceRef, bytes]] = {}
        self.gpu_lock = threading.Lock()
        self.synthesize("Hello.", steps=1)  # compile GPU pipelines now, not on the first request

    # --- internals -------------------------------------------------------------------------------
    def _err(self) -> str:
        e = self._lib.ov_last_error()
        return e.decode(errors="replace") if e else "unknown error"

    @contextlib.contextmanager
    def _quiet(self):
        if self._verbose:
            yield
            return
        sys.stderr.flush()
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(self._stderr_fd, 2)

    def _params(self, text, lang, voice, instruct, steps, seed, guidance_scale):
        p = _TTSParams()
        self._lib.ov_tts_default_params(C.byref(p))
        p.text = text.encode()
        p.lang = (lang or "").encode()
        p.mg_num_step = steps
        p.mg_seed = seed
        p.mg_guidance_scale = guidance_scale
        p.denoise = True
        if voice:
            ref, ref_text = self.voices[voice]
            p.ref_audio_tokens = ref.ref_codes
            p.ref_T = ref.ref_T
            p.ref_text = ref_text
        elif instruct:
            p.instruct = instruct.encode()
        return p

    # --- voices ----------------------------------------------------------------------------------
    def add_voice(self, name: str, ref_wav: str | Path, ref_text: str) -> float:
        """Encode a reference voice once (3-10 s of clean speech + exact transcript). Returns its length (s)."""
        audio = load_audio_24k(ref_wav)
        ref = _VoiceRef()
        with self.gpu_lock, self._quiet():
            st = self._lib.ov_extract_voice_ref(self._ctx, audio.ctypes.data_as(C.POINTER(C.c_float)),
                                                len(audio), C.byref(ref))
        if st != 0:
            raise RuntimeError(f"ov_extract_voice_ref failed for {name} ({st}): {self._err()}")
        if name in self.voices:
            self._lib.ov_voice_ref_free(C.byref(self.voices[name][0]))
        self.voices[name] = (ref, ref_text.strip().encode())
        return ref.ref_T / 25

    def load_voices_dir(self, voices_dir: str | Path) -> list[str]:
        """Every <name>.wav/.mp3/.flac/.ogg with a matching <name>.txt transcript becomes a voice."""
        for audio in sorted(Path(voices_dir).iterdir()):
            if audio.suffix.lower() not in VOICE_EXTS or audio.stem in self.voices:
                continue
            txt = audio.with_suffix(".txt")
            if not txt.is_file():
                logger.warning("voice %s has no transcript %s; skipped", audio.name, txt.name)
                continue
            secs = self.add_voice(audio.stem, audio, txt.read_text(encoding="utf-8"))
            logger.info("voice %-12s %.1f s reference", audio.stem, secs)
        return list(self.voices)

    # --- synthesis -------------------------------------------------------------------------------
    def synthesize(self, text: str, lang: str | None = None, voice: str | None = None, instruct: str | None = None,
                   steps: int = 16, seed: int = 42, guidance_scale: float = 2.0) -> np.ndarray:
        p = self._params(text, lang, voice, instruct, steps, seed, guidance_scale)
        out = _Audio()
        with self.gpu_lock, self._quiet():
            st = self._lib.ov_synthesize(self._ctx, C.byref(p), C.byref(out))
        try:
            if st != 0:
                raise RuntimeError(f"ov_synthesize failed ({st}): {self._err()}")
            return np.ctypeslib.as_array(out.samples, shape=(out.n_samples,)).copy()
        finally:
            self._lib.ov_audio_free(C.byref(out))

    def stream(self, text: str, lang: str | None = None, voice: str | None = None, instruct: str | None = None,
               steps: int = 16, seed: int = 42, guidance_scale: float = 2.0, chunk_sec: float = 4.0,
               should_stop: Callable[[], bool] | None = None) -> Iterator[np.ndarray]:
        """Yield 24 kHz pieces while the rest is still generated. The library splits `text` at punctuation into
        ~chunk_sec chunks. Breaking out of the loop, or `should_stop()` turning true, cancels the synthesis."""
        p = self._params(text, lang, voice, instruct, steps, seed, guidance_scale)
        p.chunk_duration_sec = chunk_sec
        p.chunk_threshold_sec = chunk_sec
        pieces: queue.Queue = queue.Queue()
        stop = threading.Event()

        def on_chunk(samples, n, _user):
            if n > 0:
                pieces.put(np.ctypeslib.as_array(samples, shape=(n,)).copy())
            return not (stop.is_set() or (should_stop is not None and should_stop()))  # False cancels

        p.on_chunk = _CHUNK_CB(on_chunk)

        def run():
            out = _Audio()
            try:
                with self.gpu_lock, self._quiet():
                    st = self._lib.ov_synthesize(self._ctx, C.byref(p), C.byref(out))
                cancelled = st == -5
                pieces.put(None if st == 0 or cancelled else RuntimeError(f"ov_synthesize failed ({st}): {self._err()}"))
            except BaseException as e:
                pieces.put(e)
            finally:
                self._lib.ov_audio_free(C.byref(out))

        worker = threading.Thread(target=run, daemon=True, name="omnivoice")
        worker.start()
        try:
            while True:
                item = pieces.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            worker.join()

    def close(self) -> None:
        for ref, _ in self.voices.values():
            self._lib.ov_voice_ref_free(C.byref(ref))
        self.voices.clear()
        if self._ctx:
            with self._quiet():
                self._lib.ov_free(self._ctx)
            self._ctx = None
