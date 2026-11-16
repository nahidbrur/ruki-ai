"""Ollama chat with streamed output (its REST API, standard library only), e.g. on another machine in the LAN.

The Ollama machine must accept network connections: start it with OLLAMA_HOST=0.0.0.0 (the default is localhost).
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Callable, Iterator

logger = logging.getLogger(__name__)

DEFAULT_PORT = 11434


def normalize_url(url: str) -> str:
    """'192.168.1.20' -> 'http://192.168.1.20:11434'; keeps an explicit scheme and port."""
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("Enter the Ollama server address, e.g. http://192.168.1.20:11434")
    if "://" not in url:
        url = "http://" + url
    host = url.split("://", 1)[1].split("/", 1)[0]
    if ":" not in host.rsplit("]", 1)[-1]:  # no port (IPv6 literals keep theirs inside [..])
        url = url.replace(host, f"{host}:{DEFAULT_PORT}", 1)
    return url


def _error_text(e: Exception, url: str) -> str:
    if isinstance(e, urllib.error.HTTPError):
        try:
            return json.loads(e.read().decode("utf-8", "replace")).get("error") or f"HTTP {e.code}"
        except Exception:
            return f"HTTP {e.code}"
    if isinstance(e, urllib.error.URLError):
        return f"cannot reach Ollama at {url} ({e.reason})"
    return str(e)


def loaded_context(url: str, model: str, timeout: float = 5) -> int | None:
    """Context size the model is currently loaded with on the server (None if not loaded or unknown).

    A request with a different num_ctx makes Ollama reload the model (6-11 s for a 27B model), so requests reuse it.
    """
    try:
        with urllib.request.urlopen(f"{normalize_url(url)}/api/ps", timeout=timeout) as r:
            for m in json.load(r).get("models", []):
                if model in (m.get("name"), m.get("model")):
                    return m.get("context_length") or None
    except Exception:
        pass
    return None


def list_models(url: str, timeout: float = 10) -> list[str]:
    """Names of the chat models installed on the Ollama server (embedding-only models are left out)."""
    url = normalize_url(url)
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        raise RuntimeError(_error_text(e, url)) from None
    names = []
    for m in data.get("models", []):
        caps = m.get("capabilities")
        if caps is not None and "completion" not in caps:
            continue
        names.append(m["name"])
    return sorted(names)


class _ThinkFilter:
    """Drops a leading <think>...</think> block that some reasoning models put in the reply text."""

    def __init__(self) -> None:
        self.buf, self.state = "", "start"  # start -> (thinking ->) text

    def feed(self, s: str) -> str:
        if self.state == "text":
            return s
        self.buf += s
        if self.state == "start":
            head = self.buf.lstrip()
            if "<think>".startswith(head) and len(head) < len("<think>"):
                return ""  # could still become "<think>"
            if not head.startswith("<think>"):
                self.state, out, self.buf = "text", self.buf, ""
                return out
            self.state = "thinking"
        end = self.buf.find("</think>")
        if end < 0:
            return ""
        self.state, out, self.buf = "text", self.buf[end + len("</think>"):].lstrip(), ""
        return out


class OllamaChat:
    """Same interface as GeminiChat: stream(history) yields text deltas; the history lives in the session."""

    def __init__(self, url: str, model: str, system_prompt: str, max_output_tokens: int = 512,
                 keep_alive: str = "30m", num_ctx: int = 8192, timeout: float = 120) -> None:
        if not model:
            raise ValueError("Choose an Ollama model")
        self.url = normalize_url(url)
        self.model = model
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self.keep_alive = keep_alive  # keeps the model in memory between turns (loading it takes seconds)
        self.timeout = timeout
        self._send_think = True       # "think": false turns reasoning off on models that have it
        loaded = loaded_context(self.url, model)
        self.num_ctx = loaded or num_ctx  # match the loaded instance: a different size forces a reload
        logger.info("Ollama %s at %s: context %d%s", model, self.url, self.num_ctx,
                    " (as loaded on the server)" if loaded else "")

    def _open(self, messages: list[dict]):
        payload = {"model": self.model, "messages": messages, "stream": True, "keep_alive": self.keep_alive,
                   "options": {"num_predict": self.max_output_tokens, "num_ctx": self.num_ctx}}
        if self._send_think:
            payload["think"] = False
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as e:
            msg = _error_text(e, self.url)
            if self._send_think and "think" in msg.lower():  # an Ollama that does not know the option
                self._send_think = False
                return self._open(messages)
            raise RuntimeError(f"Ollama: {msg}") from None

    def stream(self, history: list[dict], should_stop: Callable[[], bool] = lambda: False) -> Iterator[str]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += [{"role": m["role"], "content": m["text"]} for m in history if m["text"]]
        think = _ThinkFilter()
        resp = self._open(messages)
        try:
            for raw in resp:  # one JSON object per line
                if should_stop():
                    return  # closing the connection stops the generation on the server
                line = raw.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(f"Ollama: {data['error']}")
                text = think.feed(data.get("message", {}).get("content", ""))
                if text:
                    yield text
                if data.get("done"):
                    return
        finally:
            resp.close()

    def ping(self) -> str:
        return "".join(self.stream([{"role": "user", "text": "শুধু 'ঠিক আছে' লিখো।"}]))
