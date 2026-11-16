// Runs on the audio thread: low-pass + resample the microphone to 16 kHz and post 512-sample (32 ms) frames.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000;  // input samples per output sample
    this.t = 0;                      // position of the next output sample between prev (0) and current (1)
    this.prev = 0;
    this.lp = 0;
    // one-pole low-pass near 7 kHz so content above 8 kHz does not fold back when decimating
    this.alpha = 1 - Math.exp(-2 * Math.PI * 7000 / sampleRate);
    this.frame = new Float32Array(512);
    this.n = 0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.lp += this.alpha * (input[i] - this.lp);
      const x = this.lp;
      while (this.t <= 1) {
        this.frame[this.n++] = this.prev + (x - this.prev) * this.t;
        if (this.n === this.frame.length) {
          this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
          this.frame = new Float32Array(512);
          this.n = 0;
        }
        this.t += this.step;
      }
      this.t -= 1;
      this.prev = x;
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
