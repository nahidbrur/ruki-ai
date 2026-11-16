// BanglaVoice web client: microphone -> WebSocket (16 kHz float32) and streamed 24 kHz PCM -> speakers.
"use strict";

const $ = (id) => document.getElementById(id);
const orb = $("orb"), stateLabel = $("stateLabel"), log = $("log"), empty = $("empty");
const voiceSel = $("voice"), stepsSel = $("steps"), interruptChk = $("interrupt");
const textIn = $("textIn"), sendBtn = $("send"), stopTalkBtn = $("stopTalk");

const LABELS = {
  idle: "Tap the microphone to start",
  connecting: "Connecting… (allow the microphone the first time)",
  listening: "Listening… go ahead",
  user: "You are speaking…",
  transcribing: "Understanding…",
  thinking: "Thinking…",
  speaking: "Ruki is speaking…",
};

let ws = null, micStream = null, micCtx = null, playCtx = null, player = null;
let serverState = "idle", userSpeaking = false, playing = false;
let replyOpen = false; // between audio_start and assistant_done: short gaps between chunks still count as speaking
let micLevel = 0, playLevel = 0;
const turns = new Map(); // turn_id -> {user, bot}
let currentBotTurn = null;
let debug = false; // LOG_LEVEL=DEBUG on the server: show the latency breakdown of each turn

// ---------------------------------------------------------------------------- UI helpers
function setState() {
  let s = serverState;
  if (!ws) s = "idle";
  else if (serverState === "connecting") s = "connecting";
  else if (userSpeaking) s = "user";
  else if (playing || replyOpen) s = "speaking";
  else if (s === "speaking") s = "listening"; // server finished sending and playback ran dry
  orb.dataset.state = s === "user" ? "listening" : s;
  stateLabel.textContent = LABELS[s] || s;
  stopTalkBtn.disabled = !(s === "speaking" || s === "thinking");
}

function scrollDown() { log.scrollTop = log.scrollHeight; }

const AVATARS = {
  bot: `<div class="avatar ruki" title="Ruki"><span>R</span></div>`,
  // default profile picture: the user has none
  user: `<div class="avatar user" title="You"><svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="12.5" r="6"/>`
    + `<path d="M4 33c0-7 5.4-11.5 12-11.5S28 26 28 33z"/></svg></div>`,
};

function addMsg(kind, text) {
  empty.style.display = "none";
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  el.innerHTML = `${AVATARS[kind.split(" ")[0]]}<div class="bubble"></div><div class="meta"></div>`;
  el.querySelector(".bubble").textContent = text;
  log.appendChild(el);
  scrollDown();
  return el;
}

function notice(text, err = false) {
  empty.style.display = "none";
  const el = document.createElement("div");
  el.className = "notice" + (err ? " err" : "");
  el.textContent = text;
  log.appendChild(el);
  scrollDown();
}

function turn(id) {
  if (!turns.has(id)) turns.set(id, {});
  return turns.get(id);
}

function botBubble(id) {
  const t = turn(id);
  if (!t.bot) {
    t.bot = addMsg("bot typing", "");
    currentBotTurn = id;
  }
  return t.bot;
}

function ms(v) { return v >= 0 ? (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${v} ms`) : "–"; }

function showMetrics(id, m) {
  if (!debug) return;
  const meta = botBubble(id).querySelector(".meta");
  meta.innerHTML = "";
  const parts = [
    ["Voice→voice", m.voice_to_voice_ms, "v2v"],
    ["ASR", m.asr_ms], ["LLM", m.llm_first_token_ms], ["TTS", m.tts_first_audio_ms],
  ];
  for (const [k, v, cls] of parts) {
    if (v === undefined || v < 0) continue;
    const span = document.createElement("span");
    if (cls) span.className = cls;
    span.textContent = `${k} ${ms(v)}`;
    meta.appendChild(span);
  }
}

// ---------------------------------------------------------------------------- server messages
function onMessage(ev) {
  if (ev.data instanceof ArrayBuffer) {
    const pcm = new Int16Array(ev.data);
    const f = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
    player && player.port.postMessage(f, [f.buffer]);
    return;
  }
  const m = JSON.parse(ev.data);
  switch (m.type) {
    case "ready":
      serverState = "listening";
      textIn.disabled = sendBtn.disabled = false;
      break;
    case "status":
      serverState = m.state;
      break;
    case "vad":
      userSpeaking = m.speaking;
      break;
    case "user_text": {
      const el = addMsg("user", m.text);
      if (debug && m.asr_ms !== undefined) el.querySelector(".meta").textContent = `ASR ${ms(m.asr_ms)} · ${m.audio_s} s of speech`;
      turn(m.turn_id).user = el;
      break;
    }
    case "assistant_delta": {
      const b = botBubble(m.turn_id).querySelector(".bubble");
      b.textContent += m.text;
      scrollDown();
      break;
    }
    case "audio_start":
      replyOpen = true;
      break;
    case "metrics":
      showMetrics(m.turn_id, m);
      break;
    case "assistant_done": {
      replyOpen = false;
      const t = turns.get(m.turn_id);
      if (t && t.bot) t.bot.classList.remove("typing");
      if (!(t && t.bot && t.bot.querySelector(".meta").childElementCount)) showMetrics(m.turn_id, m);
      break;
    }
    case "interrupt":
      player && player.port.postMessage({ type: "flush" });
      if (currentBotTurn !== null) {
        const t = turns.get(currentBotTurn);
        if (t && t.bot && (t.bot.classList.contains("typing") || playing)) {
          t.bot.classList.remove("typing");
          t.bot.classList.add("cut");
        }
      }
      playing = replyOpen = false;
      break;
    case "error":
      notice(m.message, true);
      break;
  }
  setState();
}

// ---------------------------------------------------------------------------- start / stop
async function start() {
  serverState = "connecting";
  ws = { readyState: 0 }; // placeholder so setState shows "connecting"
  setState();
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    micCtx = new AudioContext();
    await micCtx.audioWorklet.addModule("/static/mic-worklet.js");
    playCtx = new AudioContext({ sampleRate: 24000 });
    await playCtx.audioWorklet.addModule("/static/player-worklet.js");
    player = new AudioWorkletNode(playCtx, "player", { outputChannelCount: [1] });
    player.connect(playCtx.destination);
    player.port.onmessage = (e) => {
      if (e.data.type === "state") { playing = e.data.playing; setState(); }
      else if (e.data.type === "level") playLevel = e.data.rms;
    };

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/ws`);
    sock.binaryType = "arraybuffer";
    sock.onmessage = onMessage;
    sock.onclose = () => { if (ws === sock) { notice("Connection lost"); stop(); } };
    await new Promise((res, rej) => { sock.onopen = res; sock.onerror = () => rej(new Error("WebSocket failed")); });
    ws = sock;
    ws.send(JSON.stringify({
      type: "start", voice: voiceSel.value, tts_steps: Number(stepsSel.value), allow_interruptions: interruptChk.checked,
    }));

    const src = micCtx.createMediaStreamSource(micStream);
    const cap = new AudioWorkletNode(micCtx, "mic-capture");
    cap.port.onmessage = (e) => {
      const frame = new Float32Array(e.data);
      let s = 0;
      for (let i = 0; i < frame.length; i++) s += frame[i] * frame[i];
      micLevel = Math.sqrt(s / frame.length);
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
    };
    src.connect(cap);
  } catch (err) {
    notice(`Could not start: ${err.message}`, true);
    stop();
  }
}

function stop() {
  const sock = ws;
  ws = null;
  if (sock && sock.readyState === WebSocket.OPEN) {
    try { sock.send(JSON.stringify({ type: "stop" })); } catch (_) {}
    sock.close();
  }
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (micCtx) micCtx.close();
  if (playCtx) playCtx.close();
  micStream = micCtx = playCtx = player = null;
  serverState = "idle"; userSpeaking = playing = replyOpen = false;
  textIn.disabled = sendBtn.disabled = true;
  setState();
}

function sendConfig() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: "config", voice: voiceSel.value, tts_steps: Number(stepsSel.value), allow_interruptions: interruptChk.checked,
    }));
  }
  try { localStorage.setItem("bv-settings", JSON.stringify({ voice: voiceSel.value, steps: stepsSel.value, interrupt: interruptChk.checked })); } catch (_) {}
}

// ---------------------------------------------------------------------------- wiring
orb.addEventListener("click", () => (ws ? stop() : start()));
stopTalkBtn.addEventListener("click", () => ws && ws.readyState === 1 && ws.send(JSON.stringify({ type: "interrupt" })));
$("reset").addEventListener("click", () => {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "reset" }));
  turns.clear();
  currentBotTurn = null;
  log.querySelectorAll(".msg, .notice").forEach((el) => el.remove());
  empty.style.display = "";
});
[voiceSel, stepsSel, interruptChk].forEach((el) => el.addEventListener("change", sendConfig));
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = textIn.value.trim();
  if (!text || !ws || ws.readyState !== 1) return;
  ws.send(JSON.stringify({ type: "text", text }));
  textIn.value = "";
});

// ---------------------------------------------------------------------------- dark mode switch
const themeSwitch = $("themeSwitch");
const systemDark = matchMedia("(prefers-color-scheme: dark)");
const isDark = () => (document.documentElement.dataset.theme || (systemDark.matches ? "dark" : "light")) === "dark";
const syncThemeSwitch = () => themeSwitch.setAttribute("aria-checked", String(isDark()));
themeSwitch.addEventListener("click", () => {
  const theme = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("bv-theme", theme); } catch (_) {}
  syncThemeSwitch();
});
systemDark.addEventListener("change", syncThemeSwitch); // no saved choice: the page follows the system
syncThemeSwitch();

(function animate() {
  const lvl = playing ? playLevel * 4 : ws ? micLevel * 6 : 0;
  orb.style.setProperty("--level", Math.min(1, lvl).toFixed(3));
  requestAnimationFrame(animate);
})();

(async function init() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("bv-settings") || "{}"); } catch (_) {}
  try {
    const info = await (await fetch("/api/info")).json();
    debug = !!info.debug;
    if (debug) {
      $("chips").innerHTML = [["ASR", info.asr], ["LLM", info.llm], ["TTS", info.tts]]
        .map(([k, v]) => `<span class="chip"><b>${k}</b> ${v}</span>`).join("");
    }
    for (const v of info.voices) voiceSel.add(new Option(v.replace(/_/g, " "), v));
    voiceSel.value = info.voices.includes(saved.voice) ? saved.voice : info.default_voice;
    stepsSel.value = saved.steps || String(info.tts_steps);
    interruptChk.checked = saved.interrupt ?? info.allow_interruptions;
  } catch (err) {
    notice("Could not reach the server", true);
  }
  setState();
})();
