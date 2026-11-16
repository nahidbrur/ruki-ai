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

// The conversation shown in the chat, oldest first (sent when the mic is turned on again).
function chatHistory() {
  return [...log.querySelectorAll(".msg")]
    .map((el) => ({ role: el.classList.contains("user") ? "user" : "assistant", text: el.querySelector(".bubble").textContent.trim() }))
    .filter((m) => m.text);
}

// ---------------------------------------------------------------------------- start / stop
async function start() {
  // a new server session numbers its turns from 1 again: forget the old ids (the messages stay on screen)
  turns.clear();
  currentBotTurn = null;
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
      history: chatHistory(), // the chat still on screen: Ruki keeps its context (and does not greet again)
      hour: new Date().getHours(), // local time on this machine: good morning / afternoon / evening
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
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "reset", hour: new Date().getHours() }));
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

// ---------------------------------------------------------------------------- settings: voice cloning
const settingsDlg = $("settings"), cloneFile = $("cloneFile"), clonePreview = $("clonePreview");
const cloneText = $("cloneText"), cloneName = $("cloneName"), cloneStatus = $("cloneStatus"), cloneBtn = $("cloneBtn");
const CLONE_RATE = 24000, CLONE_MAX_S = 15;
let cloneAudio = null; // base64 WAV of the prepared clip

$("settingsBtn").addEventListener("click", () => settingsDlg.showModal());
$("settingsClose").addEventListener("click", () => settingsDlg.close());
settingsDlg.addEventListener("click", (e) => { if (e.target === settingsDlg) settingsDlg.close(); }); // backdrop

function cloneMsg(text, kind = "") { cloneStatus.textContent = text; cloneStatus.className = `clone-status ${kind}`; }
function updateCloneBtn() { cloneBtn.disabled = !(cloneAudio && cloneText.value.trim()); }
cloneText.addEventListener("input", updateCloneBtn);

// Any audio file the browser can decode -> 24 kHz mono, silence trimmed at both ends, at most CLONE_MAX_S.
async function prepareClip(file) {
  const ctx = new OfflineAudioContext(1, 1, CLONE_RATE);
  const buf = await ctx.decodeAudioData(await file.arrayBuffer());
  const x = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < x.length; i++) x[i] += ch[i] / buf.numberOfChannels;
  }
  let peak = 0;
  for (const v of x) peak = Math.max(peak, Math.abs(v));
  const thr = peak * 0.05, pad = Math.round(0.15 * CLONE_RATE);
  let a = 0, b = x.length - 1;
  while (a < b && Math.abs(x[a]) < thr) a++;
  while (b > a && Math.abs(x[b]) < thr) b--;
  a = Math.max(0, a - pad);
  b = Math.min(x.length, b + pad, a + CLONE_MAX_S * CLONE_RATE);
  return x.subarray(a, b);
}

function wavBlob(samples, rate) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o, s) => [...s].forEach((ch, i) => view.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVEfmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((v, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 32767, true));
  return new Blob([view], { type: "audio/wav" });
}

const toBase64 = (blob) => new Promise((res) => {
  const r = new FileReader();
  r.onload = () => res(String(r.result).split(",")[1]);
  r.readAsDataURL(blob);
});

async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `request failed (${r.status})`);
  return data;
}

cloneFile.addEventListener("change", async () => {
  const file = cloneFile.files[0];
  cloneAudio = null; clonePreview.hidden = true; cloneText.value = ""; updateCloneBtn();
  if (!file) return cloneMsg("");
  if (!cloneName.value) cloneName.value = file.name.replace(/\.[^.]+$/, "").slice(0, 40);
  try {
    cloneMsg("Reading the recording…");
    const clip = await prepareClip(file);
    const secs = clip.length / CLONE_RATE;
    if (secs < 2) throw new Error("the recording has less than 2 seconds of speech");
    const blob = wavBlob(clip, CLONE_RATE);
    clonePreview.src = URL.createObjectURL(blob);
    clonePreview.hidden = false;
    cloneAudio = await toBase64(blob);
    cloneMsg(`Transcribing ${secs.toFixed(1)} s of speech…`);
    const { text } = await postJSON("/api/transcribe", { audio: cloneAudio });
    if (!cloneText.value.trim()) cloneText.value = text;
    cloneMsg(secs > 10 ? "Check the words above. Tip: 3–10 s clips give faster replies." : "Check the words above, then create the voice.");
  } catch (err) {
    cloneMsg(`Could not use this file: ${err.message}`, "err");
  }
  updateCloneBtn();
});

cloneBtn.addEventListener("click", async () => {
  cloneBtn.disabled = true;
  cloneMsg("Creating the voice…");
  try {
    const { name } = await postJSON("/api/voices", { name: cloneName.value, transcript: cloneText.value, audio: cloneAudio });
    if (![...voiceSel.options].some((o) => o.value === name)) voiceSel.add(new Option(name.replace(/_/g, " "), name));
    voiceSel.value = name;
    sendConfig();
    cloneMsg(`Voice “${name.replace(/_/g, " ")}” created and selected.`, "ok");
    cloneFile.value = ""; cloneText.value = ""; cloneName.value = ""; cloneAudio = null; clonePreview.hidden = true;
  } catch (err) {
    cloneMsg(err.message, "err");
  }
  updateCloneBtn();
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
