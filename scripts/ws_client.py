"""Talk to a running Ruki server over its WebSocket, like the browser does, using WAV files as the mic.

    python scripts/ws_client.py q1.wav q2.wav --url ws://localhost:8000/ws

Each WAV is streamed in real time (32 ms frames, 16 kHz float32), then the client waits for the reply to finish
playing before sending the next one. Replies are saved to outputs/ws_reply_<n>.wav.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent


def load_16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    async with websockets.connect(args.url, max_size=None) as ws:
        start = {"type": "start", "allow_interruptions": True}
        if args.voice:
            start["voice"] = args.voice
        if args.steps:
            start["tts_steps"] = args.steps
        await ws.send(json.dumps(start))
        audio: list[np.ndarray] = []
        arrivals: list[tuple[float, int]] = []  # (time, samples) of each audio message
        done = asyncio.Event()
        first_audio = [0.0]

        async def receiver():
            async for msg in ws:
                if isinstance(msg, bytes):
                    if not audio:
                        first_audio[0] = time.perf_counter()
                    audio.append(np.frombuffer(msg, dtype="<i2").astype(np.float32) / 32768)
                    arrivals.append((time.perf_counter(), len(msg) // 2))
                    continue
                m = json.loads(msg)
                if m["type"] == "assistant_delta":
                    print(m["text"], end="", flush=True)
                elif m["type"] in ("vad", "status"):
                    pass
                else:
                    print(("\n" if m["type"] == "assistant_done" else "") + "  <- " + json.dumps(m, ensure_ascii=False))
                if m["type"] == "assistant_done":
                    done.set()

        recv = asyncio.create_task(receiver())
        for n, wav in enumerate(args.wavs, 1):
            x = load_16k(wav)
            x = np.concatenate([np.zeros(8000, np.float32), x, np.zeros(16000 * 2, np.float32)])
            print(f"\n-> speaking {wav} ({len(x) / 16000:.1f} s incl. silence)")
            done.clear()
            audio.clear()
            arrivals.clear()
            for i in range(0, len(x), 512):
                await ws.send(x[i:i + 512].tobytes())
                await asyncio.sleep(512 / 16000)
            await asyncio.wait_for(done.wait(), 60)
            await asyncio.sleep(0.5)
            reply = np.concatenate(audio) if audio else np.zeros(0, np.float32)
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            out = Path(args.out_dir) / f"ws_reply_{n}.wav"
            sf.write(out, reply, 24000)
            # playback simulation: starts with the first audio, pauses whenever the buffer runs dry
            play_end, gaps = None, []
            for t, n in arrivals:
                if play_end is not None and t > play_end + 0.02:
                    gaps.append(t - play_end)
                play_end = max(play_end or t, t) + n / 24000
            print(f"   reply audio {len(reply) / 24000:.2f} s -> {out}; playback pauses: {len(gaps)} "
                  f"({', '.join(f'{g:.2f}s' for g in gaps) or 'none'})")
            # wait until the reply would have finished playing, or the next WAV counts as a barge-in
            if play_end is not None:
                await asyncio.sleep(max(0.0, play_end + 0.6 - time.perf_counter()))
        await ws.send(json.dumps({"type": "stop"}))
        recv.cancel()


if __name__ == "__main__":
    asyncio.run(main())
