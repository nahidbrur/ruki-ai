// Plays streamed assistant audio. Runs in an AudioContext at 24 kHz (the TTS rate), so no resampling.
// Messages in: Float32Array (samples to append) or {type: "flush"} (barge-in: drop everything now).
// Messages out: {type: "state", playing: bool} when playback starts / runs dry, {type: "level", rms}.
class Player extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];      // Float32Array chunks
    this.offset = 0;      // read position in queue[0]
    this.playing = false;
    this.levelAcc = 0;
    this.levelN = 0;
    this.port.onmessage = (e) => {
      if (e.data instanceof Float32Array) {
        this.queue.push(e.data);
      } else if (e.data && e.data.type === "flush") {
        this.queue = [];
        this.offset = 0;
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    let i = 0;
    while (i < out.length && this.queue.length) {
      const head = this.queue[0];
      const n = Math.min(out.length - i, head.length - this.offset);
      out.set(head.subarray(this.offset, this.offset + n), i);
      i += n;
      this.offset += n;
      if (this.offset >= head.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (i < out.length) out.fill(0, i);

    const playing = i > 0;
    if (playing !== this.playing) {
      this.playing = playing;
      this.port.postMessage({ type: "state", playing });
    }
    for (let k = 0; k < i; k++) this.levelAcc += out[k] * out[k];
    this.levelN += out.length;
    if (this.levelN >= 1200) { // ~50 ms
      this.port.postMessage({ type: "level", rms: Math.sqrt(this.levelAcc / this.levelN) });
      this.levelAcc = 0;
      this.levelN = 0;
    }
    return true;
  }
}

registerProcessor("player", Player);
