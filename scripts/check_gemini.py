"""Check the Gemini key: list the models it can use and time one streamed Bangla reply.

    python scripts/check_gemini.py                 # uses GEMINI_MODEL from .env
    python scripts/check_gemini.py --model gemini-2.5-flash-lite
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ruki.config import Settings  # noqa: E402
from ruki.llm.gemini import GeminiChat  # noqa: E402


def main():
    s = Settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=s.gemini_model)
    ap.add_argument("--prompt", default="ঢাকার আবহাওয়া সাধারণত কেমন থাকে? সংক্ষেপে বলো।")
    args = ap.parse_args()

    chat = GeminiChat(s.gemini_api_keys, args.model, s.system_prompt, s.gemini_thinking, s.gemini_max_tokens)
    print("Models that support generateContent:")
    for m in chat.client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            print("  ", m.name.removeprefix("models/"))

    print(f"\n{args.model}, thinking={s.gemini_thinking}:")
    t0 = time.perf_counter()
    first = None
    for delta in chat.stream([{"role": "user", "text": args.prompt}]):
        first = first or time.perf_counter() - t0
        print(delta, end="", flush=True)
    print(f"\n\nfirst token {first * 1000:.0f} ms, total {(time.perf_counter() - t0) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
