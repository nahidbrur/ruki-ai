"""Gemini chat with streamed output (google-genai SDK), with failover across several API keys."""
from __future__ import annotations

import logging
import threading
from typing import Callable, Iterator

from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)

_BUDGET = {"minimal": 0, "low": 512, "medium": 2048, "high": 8192}
_FAILOVER_CODES = {401, 403, 429}  # bad/blocked key or quota exhausted: try the next key


class GeminiChat:
    """Stateless client; the conversation history lives in the session and is passed in per call."""

    def __init__(self, api_keys: list[str] | str, model: str, system_prompt: str, thinking: str = "minimal",
                 max_output_tokens: int = 512) -> None:
        keys = [api_keys] if isinstance(api_keys, str) else list(api_keys)
        keys = [k for k in keys if k]
        if not keys:
            raise RuntimeError("No Gemini API key. Set GEMINI_API_KEYS (or GEMINI_API_KEY) in .env.")
        self.clients = [genai.Client(api_key=k) for k in keys]
        self._active = 0
        self._lock = threading.Lock()
        self.model = model
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self.thinking = thinking.lower()
        self._use_thinking_config = True

    @property
    def client(self) -> genai.Client:
        return self.clients[self._active]

    def _config(self) -> types.GenerateContentConfig:
        kwargs = dict(system_instruction=self.system_prompt, max_output_tokens=self.max_output_tokens)
        if self._use_thinking_config:
            if self.model.startswith("gemini-2"):
                # 2.5 Flash: budget 0 turns thinking off (the lowest latency); 2.5 Pro cannot go below 128.
                budget = _BUDGET.get(self.thinking, 0)
                if "pro" in self.model:
                    budget = max(budget, 128)
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
            else:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking.upper())
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _contents(history: list[dict]) -> list[types.Content]:
        return [types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["text"])])
                for m in history if m["text"]]

    def _open(self, contents):
        """Start a streamed request; returns (first chunk, iterator). Fails over to the next key on 401/403/429."""
        tried = 0
        while True:
            with self._lock:
                index = self._active
            try:
                response = iter(self.clients[index].models.generate_content_stream(
                    model=self.model, contents=contents, config=self._config()))
                return next(response, None), response  # request errors surface on the first chunk
            except errors.APIError as e:
                tried += 1
                if e.code in _FAILOVER_CODES and tried < len(self.clients):
                    with self._lock:
                        if self._active == index:
                            self._active = (index + 1) % len(self.clients)
                    logger.warning("Gemini key #%d failed (%s %s); switching to key #%d",
                                   index + 1, e.code, e.status, self._active + 1)
                    continue
                # Some models reject some thinking levels (3.5 Flash-Lite: no "minimal"): step up to the
                # next-lowest level, and drop the setting only if none is accepted.
                if self._use_thinking_config and e.code == 400 and "think" in str(e).lower():
                    order = list(_BUDGET)
                    nxt = order.index(self.thinking) + 1 if self.thinking in order else len(order)
                    if nxt < len(order) and not self.model.startswith("gemini-2"):
                        logger.warning("%s: thinking level %r not supported; using %r", self.model, self.thinking,
                                       order[nxt])
                        self.thinking = order[nxt]
                    else:
                        logger.warning("%s rejected the thinking config (%s); using the model default",
                                       self.model, e.message)
                        self._use_thinking_config = False
                    tried = 0
                    continue
                raise

    def stream(self, history: list[dict], should_stop: Callable[[], bool] = lambda: False) -> Iterator[str]:
        """Yield text deltas of the reply to `history` ([{"role": "user"|"assistant", "text": ...}])."""
        first, rest = self._open(self._contents(history))
        if first is None:
            return
        for chunk in _chain(first, rest):
            if should_stop():
                return
            text = _text_of(chunk)
            if text:
                yield text

    def ping(self) -> str:
        """One short request, to check the key and model at start-up."""
        return "".join(self.stream([{"role": "user", "text": "শুধু 'ঠিক আছে' লিখো।"}]))


def _chain(first, rest):
    yield first
    yield from rest


def _text_of(chunk) -> str:
    """Text parts only (skips thought summaries if a model returns them)."""
    try:
        parts = chunk.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        return ""
    return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
