"""Which language model answers: Gemini or Ollama, switchable at runtime from the web UI.

Every session talks to one LLMRouter; "Save and test" in the Settings dialog builds a new backend, checks that it
answers, then swaps it in (new messages use it at once). The choice is saved in llm_settings.json (it holds API keys:
kept out of git, readable by the owner only) and takes precedence over .env at the next start.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Iterator

from ..config import Settings

logger = logging.getLogger(__name__)
PROVIDERS = ("gemini", "ollama")


@dataclass
class LLMConfig:
    provider: str = "gemini"
    gemini_api_keys: list[str] = field(default_factory=list)
    gemini_model: str = ""
    ollama_url: str = ""
    ollama_model: str = ""

    @classmethod
    def load(cls, settings: Settings) -> "LLMConfig":
        cfg = cls(settings.llm_provider, list(settings.gemini_api_keys), settings.gemini_model,
                  settings.ollama_url, settings.ollama_model)
        path = settings.llm_settings_file
        if path.is_file():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                cfg = replace(cfg, **{k: v for k, v in saved.items() if k in cls.__dataclass_fields__})
                logger.info("LLM settings from %s (they override .env)", path.name)
            except Exception as e:
                logger.warning("ignoring %s: %s", path, e)
        if cfg.provider not in PROVIDERS:
            cfg.provider = "gemini"
        return cfg

    def save(self, settings: Settings) -> None:
        path = settings.llm_settings_file
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # API keys: owner only
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def label(self) -> str:
        return self.gemini_model if self.provider == "gemini" else f"{self.ollama_model} (Ollama)"

    def public(self) -> dict:
        """For the browser: everything except the API keys themselves."""
        return {"provider": self.provider,
                "gemini": {"model": self.gemini_model, "keys": [mask(k) for k in self.gemini_api_keys]},
                "ollama": {"url": self.ollama_url, "model": self.ollama_model}}


def mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}" if len(key) > 12 else "…"


def parse_keys(text: str) -> list[str]:
    keys: list[str] = []
    for k in text.replace(",", "\n").split():
        k = k.strip().strip("'\"[]")
        if k and k not in keys:
            keys.append(k)
    return keys


def build(cfg: LLMConfig, settings: Settings):
    if cfg.provider == "ollama":
        from .ollama import OllamaChat, normalize_url

        cfg.ollama_url = normalize_url(cfg.ollama_url)  # saved as http://host:port
        return OllamaChat(cfg.ollama_url, cfg.ollama_model, settings.system_prompt, settings.gemini_max_tokens,
                          keep_alive=settings.ollama_keep_alive, num_ctx=settings.ollama_num_ctx)
    from .gemini import GeminiChat

    if not cfg.gemini_model:
        raise ValueError("Enter a Gemini model name")
    return GeminiChat(cfg.gemini_api_keys, cfg.gemini_model, settings.system_prompt, settings.gemini_thinking,
                      settings.gemini_max_tokens)


def list_models(provider: str, settings: Settings, url: str = "", api_keys: list[str] | None = None) -> list[str]:
    if provider == "ollama":
        from .ollama import list_models as ollama_models

        return ollama_models(url)
    from .gemini import GeminiChat

    chat = GeminiChat(api_keys, "unused", settings.system_prompt)
    skip = ("tts", "image", "embedding", "aqa", "imagen", "veo", "live", "audio", "robotics", "computer-use")
    names = []
    for m in chat.client.models.list():
        name = m.name.removeprefix("models/")
        actions = getattr(m, "supported_actions", None) or []
        if (not actions or "generateContent" in actions) and name.startswith("gemini") and not any(s in name for s in skip):
            names.append(name)
    return sorted(names)


class LLMRouter:
    """What the sessions call. Forwards to the current backend; switch() replaces it for the next request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.config = LLMConfig.load(settings)
        self._backend = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Build the configured backend and check it. Failure is logged, not fatal: fix it in the web UI."""
        t = time.perf_counter()
        backend = None
        try:
            backend = build(self.config, self.settings)
            reply = backend.ping()
            logger.info("LLM   %s ok (%.2f s): %s", self.config.label(), time.perf_counter() - t, reply.strip()[:40])
        except Exception as e:
            # A built backend is kept even if its test failed (e.g. the Ollama machine is still booting).
            logger.error("LLM   %s is not working (%s). Fix it in the web UI: Settings > Language model.",
                         self.config.label() or self.config.provider, e)
        self._backend = backend

    def test_and_switch(self, cfg: LLMConfig) -> dict:
        """Build `cfg`, send one short request; on success make it current and save it. Raises on failure."""
        backend = build(cfg, self.settings)
        t = time.perf_counter()
        reply = backend.ping().strip()
        secs = time.perf_counter() - t
        if not reply:
            raise RuntimeError("the model answered with an empty reply")
        with self._lock:
            self._backend, self.config = backend, cfg
        cfg.save(self.settings)
        logger.info("LLM switched to %s (test reply in %.2f s)", cfg.label(), secs)
        return {"seconds": round(secs, 2), "reply": reply[:80]}

    def stream(self, history: list[dict], should_stop: Callable[[], bool] = lambda: False) -> Iterator[str]:
        backend = self._backend
        if backend is None:
            raise RuntimeError("No language model is set up. Open Settings > Language model.")
        yield from backend.stream(history, should_stop=should_stop)

    def label(self) -> str:
        return self.config.label()
