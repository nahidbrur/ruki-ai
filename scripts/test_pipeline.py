"""End-to-end test without a browser: stream a WAV into a Session at real-time pace, as the microphone would,
and save the assistant's spoken reply.

    python scripts/test_pipeline.py question.wav                    # real Gemini (needs .env)
    python scripts/test_pipeline.py question.wav --fake-llm         # canned reply, tests ASR + VAD + TTS only
    python scripts/test_pipeline.py q.wav --barge-in q2.wav --barge-after 1.0   # interrupt the reply
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from banglavoice.config import Settings  # noqa: E402
from banglavoice.server import setup_logging  # noqa: E402
from banglavoice.session import Models, Session, SessionOptions  # noqa: E402


class FakeLLM:
    """Streams a fixed Bangla reply that quotes the user, word by word, like a fast LLM."""

    def stream(self, history, should_stop=lambda: False):
        heard = history[-1]["text"]
        reply = f"আপনি বলেছেন, {heard}। এটা একটা পরীক্ষামূলক উত্তর। আর কিছু জানতে চান?"
        time.sleep(0.3)
        for word in reply.split(" "):
            if should_stop():
                return
            yield word + " "
            time.sleep(0.02)


def load_16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--barge-in", help="second WAV, fed while the assistant is talking")
    ap.add_argument("--barge-after", type=float, default=1.0, help="seconds after the first reply audio")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "reply.wav"))
    ap.add_argument("--wait", type=float, default=30.0, help="max seconds to wait for the reply")
    args = ap.parse_args()

    s = Settings()
    setup_logging(s.log_level)
    from banglavoice.stt.hima import HimaASR
    from banglavoice.tts.omnivoice import OmniVoiceEngine

    if args.fake_llm:
        llm = FakeLLM()
    else:
        from banglavoice.llm.gemini import GeminiChat
        llm = GeminiChat(s.gemini_api_keys, s.gemini_model, s.system_prompt, s.gemini_thinking, s.gemini_max_tokens)
    asr = HimaASR(s.asr_dir, threads=s.asr_threads, language=s.asr_language)
    tts = OmniVoiceEngine(s.tts_dir, device=s.tts_device)
    tts.load_voices_dir(s.voices_dir)

    audio_out, events = [], []
    first_audio = threading.Event()
    done = threading.Event()

    def send(msg):
        if isinstance(msg, bytes):
            audio_out.append(np.frombuffer(msg, dtype="<i2").astype(np.float32) / 32768)
            first_audio.set()
            return
        events.append(msg)
        if msg["type"] in ("user_text", "metrics", "assistant_done", "interrupt", "error"):
            print("  <-", json.dumps(msg, ensure_ascii=False))
        if msg["type"] == "assistant_done":
            done.set()

    opts = SessionOptions(voice=args.voice or s.default_voice, allow_interruptions=True,
                          tts_steps=args.steps or s.tts_steps)
    session = Session(Models(asr=asr, tts=tts, llm=llm), s, opts, send)

    def feed(x, tail_s=1.5):
        x = np.concatenate([np.zeros(8000, np.float32), x, np.zeros(int(16000 * tail_s), np.float32)])
        for i in range(0, len(x), 512):  # real-time pace, 32 ms frames
            session.feed_audio(x[i:i + 512])
            time.sleep(512 / 16000)

    x = load_16k(args.wav)
    print(f"-> feeding {args.wav} ({len(x) / 16000:.1f} s) in real time")
    feeder = threading.Thread(target=feed, args=(x, 3.0))
    feeder.start()
    if args.barge_in:
        if not first_audio.wait(args.wait):
            print("no reply audio; cannot test barge-in")
        else:
            time.sleep(args.barge_after)
            feeder.join()
            done.clear()
            print(f"-> barge-in with {args.barge_in}")
            feed(load_16k(args.barge_in))
    done.wait(args.wait)
    feeder.join()
    # let the last pieces arrive
    time.sleep(0.5)
    session.close()
    tts.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wav = np.concatenate(audio_out) if audio_out else np.zeros(0, np.float32)
    sf.write(args.out, wav, 24000)
    replies = "".join(e["text"] for e in events if e["type"] == "assistant_delta")
    print(f"\nassistant text: {replies}\nreply audio: {len(wav) / 24000:.2f} s -> {args.out}")
    print("history:", json.dumps(session.history, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
