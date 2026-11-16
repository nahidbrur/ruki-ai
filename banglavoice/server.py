"""FastAPI server: serves the web client and runs one pipeline Session per WebSocket.

WebSocket protocol (/ws)
  client -> server
    {"type": "start", "voice": "speaker_1", "allow_interruptions": true, "tts_steps": 16}   first message
    binary: 16 kHz mono float32 little-endian microphone audio, any length (512 samples is typical)
    {"type": "text", "text": "..."}              typed message (skips VAD + ASR)
    {"type": "config", "voice"?, "allow_interruptions"?, "tts_steps"?}
    {"type": "interrupt"}                        stop the assistant now
    {"type": "reset"}                            forget the conversation (and greet again)
  server -> client
    {"type": "ready", ...}, then the welcome message as an ordinary reply (assistant_delta + audio)
    {"type": "status", "state": listening|transcribing|thinking|speaking}
    {"type": "vad", "speaking": bool}
    {"type": "user_text", "turn_id", "text", "asr_ms"}
    {"type": "assistant_delta", "turn_id", "text"}   streamed LLM text
    {"type": "audio_start", "turn_id", "sample_rate": 24000}, then binary int16 PCM (24 kHz mono)
    {"type": "metrics" | "assistant_done", "turn_id", asr_ms, llm_first_token_ms, ..., voice_to_voice_ms}
    {"type": "interrupt"}   drop all buffered assistant audio
    {"type": "error", "message"}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, Settings
from .session import Models, Session, SessionOptions

logger = logging.getLogger("banglavoice")
WEB = ROOT / "web"


def setup_logging(level: str) -> None:
    # omnivoice.cpp output is silenced by pointing fd 2 at /dev/null during GPU calls. Give Python its own
    # copy of the real stderr first, so log lines from other threads are not lost meanwhile.
    sys.stderr = open(os.dup(2), "w", buffering=1, encoding="utf-8", errors="replace")
    logging.basicConfig(level=level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # one INFO line per Gemini request otherwise


def load_models(settings: Settings) -> Models:
    from .llm.gemini import GeminiChat
    from .stt.hima import HimaASR
    from .tts.omnivoice import OmniVoiceEngine

    t = time.perf_counter()
    llm = GeminiChat(settings.gemini_api_keys, settings.gemini_model, settings.system_prompt,
                     settings.gemini_thinking, settings.gemini_max_tokens)
    reply = llm.ping()
    logger.info("LLM   %s ok (%.2f s): %s", settings.gemini_model, time.perf_counter() - t, reply.strip()[:40])

    t = time.perf_counter()
    asr = HimaASR(settings.asr_dir, threads=settings.asr_threads, language=settings.asr_language or None)
    asr.transcribe(np.zeros(16000, np.float32))  # warm-up
    logger.info("ASR   Hima-ASR int8 loaded (%.2f s)", time.perf_counter() - t)

    t = time.perf_counter()
    tts = OmniVoiceEngine(settings.tts_dir, device=settings.tts_device)
    voices = tts.load_voices_dir(settings.voices_dir)
    if not voices:
        raise RuntimeError(f"no voices in {settings.voices_dir} (need <name>.wav + <name>.txt)")
    if settings.default_voice not in voices:
        logger.warning("DEFAULT_VOICE %r not found; using %r", settings.default_voice, voices[0])
        settings.default_voice = voices[0]
    logger.info("TTS   OmniVoice on %s, voices %s (%.2f s)", settings.tts_device, voices, time.perf_counter() - t)
    return Models(asr=asr, tts=tts, llm=llm)


def create_app(settings: Settings) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["models"] = await asyncio.to_thread(load_models, settings)
        logger.info("ready: open http://%s:%d", "localhost" if settings.host in ("127.0.0.1", "0.0.0.0") else settings.host,
                    settings.port)
        yield
        state["models"].tts.close()

    app = FastAPI(title="BanglaVoice", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/info")
    async def info():
        models: Models = state["models"]
        return JSONResponse({
            "voices": list(models.tts.voices), "default_voice": settings.default_voice,
            "allow_interruptions": settings.allow_interruptions, "tts_steps": settings.tts_steps,
            "debug": settings.debug,
            "llm": settings.gemini_model, "asr": "Hima-ASR int8 (ONNX, CPU)",
            "tts": f"OmniVoice Q8_0 (omnivoice.cpp, {settings.tts_device})",
        })

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        models: Models = state["models"]
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()

        def send(msg):  # called from handler threads
            loop.call_soon_threadsafe(out.put_nowait, msg)

        async def sender():
            while True:
                msg = await out.get()
                if msg is None:
                    return
                if isinstance(msg, bytes):
                    await ws.send_bytes(msg)
                else:
                    await ws.send_text(json.dumps(msg, ensure_ascii=False))

        session: Session | None = None
        sender_task = asyncio.create_task(sender())
        try:
            start = json.loads(await ws.receive_text())
            voice = start.get("voice") if start.get("voice") in models.tts.voices else settings.default_voice
            options = SessionOptions(
                voice=voice,
                allow_interruptions=bool(start.get("allow_interruptions", settings.allow_interruptions)),
                tts_steps=int(start.get("tts_steps") or settings.tts_steps),
            )
            session = await asyncio.to_thread(Session, models, settings, options, send)
            send({"type": "ready", "voice": options.voice, "allow_interruptions": options.allow_interruptions,
                  "tts_steps": options.tts_steps})
            logger.info("session started: %s", options)
            session.greet()
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    data = msg["bytes"]
                    session.feed_audio(np.frombuffer(data[: len(data) // 4 * 4], dtype="<f4"))
                elif msg.get("text"):
                    cmd = json.loads(msg["text"])
                    kind = cmd.get("type")
                    if kind == "text":
                        session.submit_text(str(cmd.get("text", "")))
                    elif kind == "config":
                        v = cmd.get("voice")
                        session.update_options(
                            voice=v if v in models.tts.voices else None,
                            allow_interruptions=cmd.get("allow_interruptions"),
                            tts_steps=int(cmd["tts_steps"]) if cmd.get("tts_steps") else None)
                    elif kind == "interrupt":
                        session.interrupt("client")
                    elif kind == "reset":
                        session.interrupt("reset")
                        with session._lock:
                            session.history.clear()
                        session.greet()
                    elif kind == "stop":
                        break
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("websocket session failed")
        finally:
            if session is not None:
                await asyncio.to_thread(session.close)
                logger.info("session closed")
            out.put_nowait(None)
            try:
                await sender_task
            except Exception:
                pass

    return app


def main() -> None:
    settings = Settings()
    ap = argparse.ArgumentParser(description="BanglaVoice real-time speech-to-speech server")
    ap.add_argument("--host", default=settings.host)
    ap.add_argument("--port", type=int, default=settings.port)
    ap.add_argument("--tts-device", default=settings.tts_device, help="Vulkan1 (RTX 2050), Vulkan0 (iGPU) or CPU")
    ap.add_argument("--tts-steps", type=int, default=settings.tts_steps, help="OmniVoice steps: 8 fast, 16, 32 best")
    ap.add_argument("--no-interruptions", action="store_true", help="half duplex: ignore the mic while speaking")
    ap.add_argument("--log-level", default=settings.log_level)
    args = ap.parse_args()
    settings.host, settings.port, settings.tts_device = args.host, args.port, args.tts_device
    settings.tts_steps, settings.log_level = args.tts_steps, args.log_level
    if args.no_interruptions:
        settings.allow_interruptions = False
    setup_logging(args.log_level)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning", ws_ping_interval=20)


if __name__ == "__main__":
    main()
