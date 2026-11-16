"""Settings, read from the environment (and `.env` in the project root)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv is optional; plain environment variables still work
    pass


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _api_keys() -> list[str]:
    """GEMINI_API_KEYS as a JSON list or comma-separated, plus GEMINI_API_KEY / GOOGLE_API_KEY."""
    import json

    keys: list[str] = []
    raw = _env("GEMINI_API_KEYS", "")
    if raw:
        try:
            parsed = json.loads(raw)
            keys += [str(k) for k in (parsed if isinstance(parsed, list) else [parsed])]
        except json.JSONDecodeError:
            keys += raw.split(",")
    keys += [_env("GEMINI_API_KEY", ""), _env("GOOGLE_API_KEY", "")]
    seen: list[str] = []
    for k in (k.strip().strip("'\"") for k in keys):
        if k and k not in seen:
            seen.append(k)
    return seen


DEFAULT_SYSTEM_PROMPT = (
    "তোমার নাম রুকি (Ruki)। তুমি একজন বন্ধুসুলভ বাংলা ভয়েস সহকারী। ব্যবহারকারী মুখে কথা বলছেন, "
    "আর তোমার উত্তর কণ্ঠে পড়ে শোনানো হবে।\n"
    "তুমি একজন মুসলিম। কথোপকথনের শুরুতে তুমি ‘আচ্ছালামু আলাইকুম ’ বলে সালাম দিয়েছ; প্রতিটি উত্তরে আবার সালাম দেবে না। "
    "কেউ সালাম দিলে ‘ওয়ালাইকুম আচ্ছালাম' বলে উত্তর দাও। কেউ নাম জানতে চাইলে বলো তোমার নাম রুকি।\n"
    "নিয়ম:\n"
    "- সবসময় শুদ্ধ, স্বাভাবিক কথ্য বাংলায় উত্তর দাও (ব্যবহারকারী অন্য ভাষায় বলতে বললে সেটা মানো)।\n"
    "- ব্যবহারকারীকে সম্মান দিয়ে ‘আপনি’ বলে সম্বোধন করো।\n"
    "- উত্তর ছোট রাখো: সাধারণত এক থেকে তিনটি বাক্য। বিস্তারিত চাইলে তবেই বেশি বলো।\n"
    "- প্রথম বাক্যটি খুব ছোট রাখো (দুই থেকে পাঁচ শব্দ), তারপর বাকি কথা বলো।\n"
    "- কোনো মার্কডাউন, তালিকা, বুলেট, ইমোজি, লিংক বা বিশেষ চিহ্ন ব্যবহার করবে না; শুধু বলার মতো সাধারণ বাক্য।\n"
    "- সংখ্যা, তারিখ, সময়, মুদ্রা ও সংক্ষেপ বাংলা শব্দে লিখে দাও, যেমন ‘১২৫’ না লিখে ‘একশো পঁচিশ’।\n"
    "- প্রতিটি বাক্য দাঁড়ি (।), প্রশ্নবোধক (?) বা বিস্ময়সূচক (!) চিহ্ন দিয়ে শেষ করো।\n"
    "- ব্যবহারকারীর কথা স্বয়ংক্রিয় স্পিচ রিকগনিশন থেকে আসে, তাই কিছু শব্দ ভুল থাকতে পারে; প্রসঙ্গ বুঝে উত্তর দাও, "
    "একেবারে বোঝা না গেলে ভদ্রভাবে আবার বলতে বলো।"
)

# Spoken by the assistant as soon as a conversation starts (empty WELCOME_MESSAGE = no greeting).
DEFAULT_WELCOME_MESSAGE = "আচ্ছালামু আলাইকুম! আমি রুকি। বলুন, আপনাকে কীভাবে সাহায্য করতে পারি?"


@dataclass
class Settings:
    # --- server ---
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))

    # --- LLM (Gemini) ---
    gemini_api_keys: list[str] = field(default_factory=_api_keys)  # tried in turn on quota/auth errors
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    # Gemini 3.x: minimal | low | medium | high (an unsupported level steps up to the next one).
    # Gemini 2.x: mapped to a thinking budget (minimal -> 0).
    gemini_thinking: str = field(default_factory=lambda: _env("GEMINI_THINKING", "minimal"))
    gemini_max_tokens: int = field(default_factory=lambda: int(_env("GEMINI_MAX_TOKENS", "512")))
    system_prompt: str = field(default_factory=lambda: _env("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    welcome_message: str = field(default_factory=lambda: os.environ.get("WELCOME_MESSAGE", DEFAULT_WELCOME_MESSAGE).strip())
    max_history_turns: int = field(default_factory=lambda: int(_env("MAX_HISTORY_TURNS", "12")))

    # --- ASR (Hima-ASR int8, ONNX Runtime on CPU) ---
    asr_dir: Path = field(default_factory=lambda: Path(_env("ASR_DIR", str(MODELS / "asr" / "hima-asr-onnx"))))
    asr_threads: int = field(default_factory=lambda: int(_env("ASR_THREADS", "6")))
    asr_language: str = field(default_factory=lambda: _env("ASR_LANGUAGE", "bn"))

    # --- VAD / turn taking ---
    vad_threshold: float = field(default_factory=lambda: float(_env("VAD_THRESHOLD", "0.5")))
    vad_min_silence_ms: int = field(default_factory=lambda: int(_env("VAD_MIN_SILENCE_MS", "600")))
    vad_min_speech_ms: int = field(default_factory=lambda: int(_env("VAD_MIN_SPEECH_MS", "300")))
    vad_speech_pad_ms: int = field(default_factory=lambda: int(_env("VAD_SPEECH_PAD_MS", "200")))
    vad_max_speech_s: float = field(default_factory=lambda: float(_env("VAD_MAX_SPEECH_S", "25")))
    denoise: bool = field(default_factory=lambda: _bool("DENOISE", True))
    # Barge-in: user speech while the assistant talks stops it. Needs echo cancellation (browser) or headphones.
    allow_interruptions: bool = field(default_factory=lambda: _bool("ALLOW_INTERRUPTIONS", True))
    barge_in_min_ms: int = field(default_factory=lambda: int(_env("BARGE_IN_MIN_MS", "350")))
    barge_in_threshold: float = field(default_factory=lambda: float(_env("BARGE_IN_THRESHOLD", "0.7")))

    # --- TTS (OmniVoice via omnivoice.cpp + Vulkan) ---
    tts_dir: Path = field(default_factory=lambda: Path(_env("TTS_DIR", str(MODELS / "tts"))))
    tts_device: str = field(default_factory=lambda: _env("TTS_DEVICE", "Vulkan1"))
    tts_steps: int = field(default_factory=lambda: int(_env("TTS_STEPS", "16")))
    # Each reply's first TTS_FAST_CHUNKS chunks use at most TTS_FIRST_STEPS steps: they decide when the user
    # hears anything, and whether playback has to pause before the third chunk is ready.
    tts_first_steps: int = field(default_factory=lambda: int(_env("TTS_FIRST_STEPS", "12")))
    tts_fast_chunks: int = field(default_factory=lambda: int(_env("TTS_FAST_CHUNKS", "2")))
    tts_chunk_sec: float = field(default_factory=lambda: float(_env("TTS_CHUNK_SEC", "4.0")))
    tts_language: str = field(default_factory=lambda: _env("TTS_LANGUAGE", "Bengali"))
    voices_dir: Path = field(default_factory=lambda: Path(_env("VOICES_DIR", str(ROOT / "voices"))))
    default_voice: str = field(default_factory=lambda: _env("DEFAULT_VOICE", "speaker_1"))
    # First LLM chunk may be cut at a comma once it is this long, so speech starts sooner.
    first_chunk_min_chars: int = field(default_factory=lambda: int(_env("FIRST_CHUNK_MIN_CHARS", "15")))
    # Later chunks: queued sentences are merged up to this many characters per TTS call.
    tts_max_chars: int = field(default_factory=lambda: int(_env("TTS_MAX_CHARS", "220")))

    # --- logging ---
    # DEBUG also shows the per-turn latency breakdown (voice-to-voice, ASR, LLM, TTS) in the web UI.
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    @property
    def debug(self) -> bool:
        return self.log_level.upper() == "DEBUG"
