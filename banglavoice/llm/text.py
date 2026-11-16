"""Turning streamed LLM text into speakable chunks for the TTS."""
from __future__ import annotations

import re

# Sentence ends: Bangla dari, double dari, ? ! an ASCII full stop followed by a space/end, or a newline.
_SENT_END = re.compile(r"[।॥?!]+[\"'”’)]*|(?<![0-9০-৯])\.(?=\s|$)[\"'”’)]*|\n+")
_SOFT_BREAK = re.compile(r"[,;:—–]\s")

_MARKDOWN = re.compile(r"(\*\*|__|`{1,3}|^#{1,6}\s*|^\s*[-*•]\s+|^\s*\d+[.)]\s+)", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
_SPACE = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """Strip markdown, links, emoji: the system prompt asks for plain text, this is the safety net."""
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    text = text.replace("*", "").replace("#", "")
    return _SPACE.sub(" ", text).strip()


def _has_speech(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


class SentenceChunker:
    """Feed streamed text; get back whole sentences as soon as they are complete.

    The very first chunk may also be cut at a comma once it holds `first_min_chars`, so the TTS can
    start earlier; later chunks always end at sentence punctuation (or at a newline).
    """

    def __init__(self, first_min_chars: int = 15) -> None:
        self.buf = ""
        self.first_min_chars = first_min_chars
        self.emitted = 0

    def push(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while True:
            m = _SENT_END.search(self.buf)
            if m is None and self.emitted == 0 and len(self.buf) >= self.first_min_chars:
                m = _SOFT_BREAK.search(self.buf, self.first_min_chars // 2)
            if m is None:
                break
            piece, self.buf = self.buf[: m.end()], self.buf[m.end():]
            piece = clean_for_speech(piece)
            if _has_speech(piece):
                out.append(piece)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        piece, self.buf = clean_for_speech(self.buf), ""
        if _has_speech(piece):
            self.emitted += 1
            return [piece]
        return []
