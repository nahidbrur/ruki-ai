"""FastAPI server: serves the web client and runs one pipeline Session per WebSocket.

WebSocket protocol (/ws)
  client -> server
    {"type": "start", "voice": "Ayesha", "allow_interruptions": true, "tts_steps": 16,
     "hour"?: 0-23, "history"?: [{"role": "user"|"assistant", "text"}]}   first message; with a history (the chat on screen
                                                           when the mic is turned on again) there is no greeting
    binary: 16 kHz mono float32 little-endian microphone audio, any length (512 samples is typical)
    {"type": "text", "text": "..."}              typed message (skips VAD + ASR)
    {"type": "config", "voice"?, "allow_interruptions"?, "tts_steps"?}
    {"type": "interrupt"}                        stop the assistant now
    {"type": "reset", "hour"?: 0-23}             forget the conversation (and greet again)
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
import base64
import io
import json
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, Settings
from .llm.manager import LLMRouter, list_models as list_llm_models, parse_keys
from .session import Models, Session, SessionOptions

logger = logging.getLogger("ruki")
WEB = ROOT / "web"
USER_VOICES = "user"            # voices/user/: voices cloned from the web UI
MAX_CLONE_S, MIN_CLONE_S = 20.0, 2.0


def setup_logging(level: str) -> None:
    # omnivoice.cpp output is silenced by pointing fd 2 at /dev/null during GPU calls. Give Python its own
    # copy of the real stderr first, so log lines from other threads are not lost meanwhile.
    sys.stderr = open(os.dup(2), "w", buffering=1, encoding="utf-8", errors="replace")
    logging.basicConfig(level=level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # one INFO line per Gemini request otherwise


def load_models(settings: Settings) -> Models:
    from .stt.hima import HimaASR
    from .tts.omnivoice import OmniVoiceEngine

    llm = LLMRouter(settings)  # Gemini or Ollama; switchable later from the web UI
    llm.start()

    t = time.perf_counter()
    asr = HimaASR(settings.asr_dir, threads=settings.asr_threads, language=settings.asr_language or None)
    asr.transcribe(np.zeros(16000, np.float32))  # warm-up
    logger.info("ASR   Hima-ASR int8 loaded (%.2f s)", time.perf_counter() - t)

    t = time.perf_counter()
    tts = OmniVoiceEngine(settings.tts_dir, device=settings.tts_device)
    voices = tts.load_voices_dir(settings.voices_dir)
    if (settings.voices_dir / USER_VOICES).is_dir():
        voices = tts.load_voices_dir(settings.voices_dir / USER_VOICES)
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

    app = FastAPI(title="Ruki", lifespan=lifespan)
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
            "llm": models.llm.label(), "asr": "Hima-ASR int8 (ONNX, CPU)",
            "tts": f"OmniVoice Q8_0 (omnivoice.cpp, {settings.tts_device})",
        })

    def _clip(body: dict) -> np.ndarray:
        """The uploaded reference clip (base64 WAV; the browser sends 24 kHz mono 16-bit) as 24 kHz float32."""
        import soundfile as sf
        from scipy.signal import resample_poly

        try:
            audio, sr = sf.read(io.BytesIO(base64.b64decode(body["audio"])), dtype="float32", always_2d=True)
        except Exception:
            raise HTTPException(400, "Could not read the audio. Upload a WAV, MP3, M4A or OGG file.")
        audio = audio.mean(1)
        if sr != 24000:
            g = np.gcd(sr, 24000)
            audio = resample_poly(audio, 24000 // g, sr // g).astype(np.float32)
        secs = len(audio) / 24000
        if not MIN_CLONE_S <= secs <= MAX_CLONE_S:
            raise HTTPException(400, f"The clip is {secs:.1f} s long; use {MIN_CLONE_S:.0f}-{MAX_CLONE_S:.0f} s of speech.")
        return audio

    @app.post("/api/transcribe")
    async def transcribe(body: dict = Body(...)):
        """Transcript of an uploaded reference clip, to pre-fill the voice-cloning form."""
        from scipy.signal import resample_poly

        audio = _clip(body)
        text = await asyncio.to_thread(state["models"].asr.transcribe, resample_poly(audio, 2, 3).astype(np.float32))
        return {"text": text.strip()}

    @app.post("/api/voices")
    async def add_voice(body: dict = Body(...)):
        """Clone a voice: save voices/user/<name>.wav + .txt and register it with the TTS."""
        import soundfile as sf

        models: Models = state["models"]
        transcript = str(body.get("transcript", "")).strip()
        if not transcript or len(transcript) > 600:
            raise HTTPException(400, "Enter the exact words spoken in the clip (up to 600 characters).")
        base = re.sub(r"[\s/\\.:]+", "_", str(body.get("name", "")).strip()).strip("_")[:40] or "my_voice"
        name, n = base, 2
        while name in models.tts.voices:  # never replace a voice another session may be using
            name, n = f"{base}_{n}", n + 1
        audio = _clip(body)
        folder = settings.voices_dir / USER_VOICES
        folder.mkdir(parents=True, exist_ok=True)
        wav = folder / f"{name}.wav"
        sf.write(wav, audio, 24000, subtype="PCM_16")
        (folder / f"{name}.txt").write_text(transcript + "\n", encoding="utf-8")
        try:
            secs = await asyncio.to_thread(models.tts.add_voice, name, wav, transcript)
        except Exception as e:
            wav.unlink(missing_ok=True)
            (folder / f"{name}.txt").unlink(missing_ok=True)
            logger.exception("voice cloning failed")
            raise HTTPException(500, f"Could not create the voice: {e}")
        logger.info("voice %s cloned from the web UI (%.1f s reference)", name, secs)
        return {"name": name, "voices": list(models.tts.voices)}

    # --- language model settings (Settings dialog) ---------------------------------------------------------
    @app.get("/api/llm")
    async def llm_settings():
        return state["models"].llm.config.public()

    @app.post("/api/llm/models")
    async def llm_models(body: dict = Body(...)):
        """Models the given Gemini key(s) or Ollama server offer (for the model picker)."""
        router: LLMRouter = state["models"].llm
        provider = body.get("provider")
        keys = parse_keys(str(body.get("api_keys") or "")) or router.config.gemini_api_keys
        try:
            names = await asyncio.to_thread(list_llm_models, provider, settings, str(body.get("url") or ""), keys)
        except Exception as e:
            raise HTTPException(400, f"Could not list the models: {e}")
        return {"models": names}

    @app.post("/api/llm")
    async def llm_save(body: dict = Body(...)):
        """Switch the language model after a test request succeeds; empty API keys keep the saved ones."""
        router: LLMRouter = state["models"].llm
        provider = body.get("provider")
        if provider not in ("gemini", "ollama"):
            raise HTTPException(400, "Choose Gemini or Ollama.")
        g, o = body.get("gemini") or {}, body.get("ollama") or {}
        cfg = replace(router.config, provider=provider,
                      gemini_api_keys=parse_keys(str(g.get("api_keys") or "")) or router.config.gemini_api_keys,
                      gemini_model=str(g.get("model") or router.config.gemini_model).strip(),
                      ollama_url=str(o.get("url") or "").strip() or router.config.ollama_url,
                      ollama_model=str(o.get("model") or "").strip() or router.config.ollama_model)
        if provider == "gemini" and not cfg.gemini_api_keys:
            raise HTTPException(400, "Enter at least one Gemini API key.")
        try:
            result = await asyncio.to_thread(router.test_and_switch, cfg)
        except Exception as e:
            raise HTTPException(400, f"{cfg.label()} did not answer: {e}")
        return {**result, "label": cfg.label(), "settings": cfg.public()}

    def _hour(msg: dict) -> int | None:
        """The user's local hour, sent by the browser with "start" and "reset"."""
        try:
            return int(msg["hour"])
        except (KeyError, TypeError, ValueError):
            return None

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
            restored = session.restore_history(start.get("history") or [])
            send({"type": "ready", "voice": options.voice, "allow_interruptions": options.allow_interruptions,
                  "tts_steps": options.tts_steps})
            logger.info("session started: %s, %d messages restored", options, restored)
            if not restored:
                session.greet(_hour(start))
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
                        session.greet(_hour(cmd))
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
    ap = argparse.ArgumentParser(description="Ruki real-time speech-to-speech server")
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
