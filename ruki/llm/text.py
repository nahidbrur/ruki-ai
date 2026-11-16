"""Turning streamed LLM text into speakable chunks for the TTS."""
from __future__ import annotations

import re
from typing import Iterator

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


# Salam in a reply (আসসালামু আলাইকুম, ওয়ালাইকুম আসসালাম and spellings like আচ্ছালাম, আছলাম, আলায়কুম) and in the
# user's words (also typed in Latin letters: assalamualaikum, salam, slm, walaikum ...).
_SALAM = re.compile(r"সালাম|ছালাম|ছলাম|সলাম|লাইকুম|লায়কুম")
_USER_SALAM = re.compile(r"সালাম|ছালাম|ছলাম|সলাম|লাইকুম|লায়কুম|salam|salaam|\bslm\b|alaikum|alaykum|alaikom", re.IGNORECASE)
_CLAUSE_END = re.compile(r"[।॥!?,;\n]")


def user_gave_salam(text: str) -> bool:
    return bool(_USER_SALAM.search(text))


def strip_unprompted_salam(deltas: Iterator[str], user_text: str) -> Iterator[str]:
    """Drop the salam a model sometimes opens a reply with although the user did not greet with salam.

    Only the reply's opening clause(s) are checked: they are held back until the first clause ends, so the
    rest of the reply streams through unchanged. When the user did give salam, the reply passes as it is.
    """
    if user_gave_salam(user_text):
        yield from deltas
        return
    buf, passing, dropped = "", False, False
    for delta in deltas:
        if passing:
            yield delta
            continue
        buf += delta
        while (m := _CLAUSE_END.search(buf)) is not None:
            if _SALAM.search(buf[:m.end()]):
                buf = buf[m.end():].lstrip()  # drop the greeting clause; the next one may be another greeting
                dropped = True
                continue
            passing = True
            break
        if not passing and len(buf) > 80 and not _SALAM.search(buf):
            passing = True  # a long opening without punctuation and without salam
        if passing:
            buf = buf.lstrip() if dropped else buf
            if buf:
                yield buf
            buf = ""
    if not passing and buf.strip() and not _SALAM.search(buf):
        yield buf.lstrip() if dropped else buf


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
