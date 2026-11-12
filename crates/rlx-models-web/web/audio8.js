// Audio8-TTS-Realistic-Air across Web Workers, each its own wasm32 instance:
// `slow` workers hold a run of the 24 slow layers each (the first embeds the
// rows, the last ends in the band head), one worker the fast decoder and the
// sampler, one the codec. Prompts are built here (the model's tokenizer and
// the processor's layout); a frame goes row → slow stages → frame maker → the
// next row. From the same seed the frames are the native runner's
// (`S2Model::generate`; `rlx_fish::audio8` tests the split against it).
//
//   import { Audio8Pipeline } from "./audio8.js";
//   const p = await Audio8Pipeline.create({ base: "/weights/tts/audio8-web", slow: 2 });
//   const { pcm, sampleRate } = await p.speak("Hello from the browser.");
//   p.terminate();
//
// `base` serves config.json, generation_config.json, tokenizer.json,
// model.safetensors (fetched by Range, each worker its own tensors) and the
// codec's export, codec_decoder.safetensors (+ codec_encoder.safetensors with
// `clone: true`), from `rlx-fish export-web`.
import init, { TextTokenizer, audio8_prompt, audio8_sentences } from "./pkg/rlx_models_web.js";

class Worker8 {
  constructor(url, label) {
    this.label = label;
    this.pending = new Map();
    this.next = 0;
    this.onProgress = () => {};
    this.onLog = () => {};
    this.onAudio = () => {};
    this.onFrame = () => {};
    this.waiting = null;
    // Time spent waiting on each kind of call, ms, and how many.
    this.time = {};
    this.worker = new Worker(url, { type: "module" });
    this.worker.onmessage = ({ data }) => {
      if (data.progress) return this.onProgress(data.progress);
      if (data.log) return this.onLog(`${this.label}: ${data.log}`);
      if (data.audio) return this.onAudio(data.audio, data.final);
      if (data.frame) return this.onFrame(data.frame);
      const p = this.pending.get(data.id);
      this.pending.delete(data.id);
      data.ok ? p.resolve(data) : p.reject(new Error(`${this.label}: ${data.error}`));
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(`${this.label}: ${e.message ?? "worker error"}`));
      this.pending.clear();
    };
  }

  call(op, args = {}, transfer = []) {
    const id = this.next++;
    this.waiting = { op, since: performance.now() };
    return new Promise((resolve, reject) => {
      const t0 = performance.now();
      const tally = () => {
        const t = (this.time[op] ??= { ms: 0, n: 0 });
        t.ms += performance.now() - t0;
        t.n += 1;
      };
      this.pending.set(id, {
        resolve: (v) => { tally(); this.waiting = null; resolve(v); },
        reject: (e) => { this.waiting = null; reject(e); },
      });
      this.worker.postMessage({ id, op, ...args }, transfer);
    });
  }
}

export class Audio8Pipeline {
  /**
   * @param {object} o
   * @param {string} o.base        URL of the model directory (see above)
   * @param {number} [o.slow=2]    workers for the slow layers (1 … 24)
   * @param {string} [o.backend="cpu"]  the slow stages' backend ("webgpu" or "cpu")
   * @param {string} [o.framesBackend]   the fast decoder's (default: `backend`)
   * @param {string} [o.codecBackend]    the codec's (default: `backend`)
   * @param {boolean} [o.clone=false]  load the codec's encoder, for `reference` voices
   * @param {boolean} [o.fastMath=false]  WebGPU: split-K decode matmuls and native
   *   `/`, `sqrt` (`set_gpu_fast_math`) — about 2x faster here, the same speech
   *   on the sentences checked; results no longer bit-identical to the CPU
   * @param {(msg: string) => void} [o.onProgress]
   */
  static async create({ base, slow = 2, backend = "cpu", framesBackend = backend, codecBackend = backend, clone = false, fastMath = false, settings = {}, onProgress = () => {},
    workerUrl = new URL("./audio8_worker.js", import.meta.url) }) {
    await init();
    const text = async (name) => {
      const r = await fetch(`${base}/${name}`);
      if (!r.ok) throw new Error(`${base}/${name}: HTTP ${r.status}`);
      return r;
    };
    const config = await (await text("config.json")).text();
    const generation = await (await text("generation_config.json")).json();
    const tokenizer = new TextTokenizer(new Uint8Array(await (await text("tokenizer.json")).arrayBuffer()));
    const p = new Audio8Pipeline(config, generation, tokenizer);
    p.slow = Array.from({ length: slow }, (_, i) => new Worker8(workerUrl, `slow ${i + 1}/${slow}`));
    p.frames = new Worker8(workerUrl, "frames");
    p.codec = new Worker8(workerUrl, "codec");
    const all = [...p.slow, p.frames, p.codec];
    all.forEach((w) => (w.onLog = onProgress));
    p.onProgress = onProgress;
    // Every 10 s, name any worker still on one call: where a hang is.
    p.watch = setInterval(() => {
      for (const w of all) {
        if (w.waiting && performance.now() - w.waiting.since > 10000) {
          onProgress(`still waiting on ${w.label}: ${w.waiting.op} (${Math.round((performance.now() - w.waiting.since) / 1000)} s)`);
        }
      }
    }, 10000);
    const load = async (w, args) => {
      const t0 = performance.now();
      const r = await w.call("load", { base, config, backend, clone, fastMath, settings, ...args });
      p.loaded.push({ worker: w.label, ms: Math.round(performance.now() - t0), fetched_mb: r.fetched_mb, memory_mb: r.memory_mb });
      onProgress(`${w.label} ready: ${r.fetched_mb} MB fetched, wasm memory ${r.memory_mb} MB, ${Math.round(performance.now() - t0)} ms`);
    };
    try {
      // One at a time: every worker's load-time peak at once can exceed the tab's memory.
      for (let i = 0; i < slow; i++) await load(p.slow[i], { role: "slow", index: i, count: slow });
      await load(p.frames, { role: "frames", backend: framesBackend });
      await load(p.codec, { role: "codec", backend: codecBackend });
      // The ring: slow 1 → … → slow N → frames → slow 1, and frames → codec.
      const hop = Array.from({ length: slow }, () => new MessageChannel()); // into slow i
      const back = new MessageChannel(); // last slow → frames
      const toCodec = new MessageChannel();
      await Promise.all(p.slow.map((w, i) => {
        const next = i + 1 < slow ? hop[i + 1].port1 : back.port1;
        return w.call("link", { prev: hop[i].port2, next }, [hop[i].port2, next]);
      }));
      await p.frames.call("link", { prev: back.port2, next: hop[0].port1, codec: toCodec.port1 },
        [back.port2, hop[0].port1, toCodec.port1]);
      await p.codec.call("link", { prev: toCodec.port2 }, [toCodec.port2]);
    } catch (e) {
      all.forEach((w) => w.worker.terminate());
      throw e;
    }
    return p;
  }

  constructor(config, generation, tokenizer) {
    this.configJson = config;
    this.config = JSON.parse(config);
    this.generation = generation;
    this.tokenizer = tokenizer;
    this.loaded = [];
  }

  get sampleRate() {
    return this.config.codec_sample_rate;
  }

  /** Codes `[10][frames]` (flattened) of a 44.1 kHz mono clip, for `speak({ reference })`. */
  async encode(pcm) {
    return (await this.codec.call("encode", { pcm })).out;
  }

  /**
   * Speak `text`. Options default to generation_config.json's.
   * `reference: { codes, text }` clones a voice (codes from `encode`).
   * `onAudio(pcm, sampleRate)` gets the speech as it is made: the codec is
   * causal, so decoding the frames so far gives the start of the final audio,
   * and each call hands over only what is new (decoded on the codec's worker
   * while generation goes on, `streamEvery` frames or more at a time).
   * `batch: n` (> 1) generates the text sentence by sentence, `n` at once:
   * one step of every worker serves the whole batch. Without `reference`
   * the first sentence is generated alone and its voice carries the rest;
   * sentence `i` samples from `seed + i`, the frames the native
   * `synthesize_batch` makes. Audio still arrives in text order.
   * Returns `{ pcm, sampleRate, codes, frames, finished, ms }`.
   */
  async speak(text, opts = {}) {
    if ((opts.batch ?? 1) > 1) {
      const sentences = audio8_sentences(text);
      if (sentences.length > 1) return this.#speakBatch(sentences, opts);
    }
    return this.#speakOne(text, opts);
  }

  async #speakOne(text, { maxFrames, temperature, topP, topK, seed = 0, reference, onFrame = () => {},
    onAudio = null, streamEvery = 32 } = {}) {
    const g = this.generation;
    const nc = this.config.num_codebooks;
    const width = 1 + nc;
    const rows = audio8_prompt(this.tokenizer, this.configJson, text, reference?.text, reference?.codes);
    const prompt = rows.length / width;
    const context = this.config.max_seq_len;
    if (prompt >= context) throw new Error(`prompt length ${prompt} must be smaller than ${context}`);
    const budget = Math.min(maxFrames ?? g.max_new_tokens, context - prompt);
    const t0 = performance.now();
    const { cap } = await this.slow[0].call("capacity", { prompt, maxNew: budget });
    await this.frames.call("restart", {
      temperature: temperature ?? g.temperature, topP: topP ?? g.top_p, topK: topK ?? g.top_k, seed,
    });
    await this.codec.call("reset", { streamEvery });
    let x = Uint32Array.from(rows);
    for (const s of this.slow) x = (await s.call("prefill", { input: x, cap }, [x.buffer])).out;
    const t1 = performance.now();
    this.onProgress?.(`prompt (${prompt} rows) done in ${Math.round(t1 - t0)} ms`);
    // From here the workers run the frames among themselves: the page only
    // hears each frame's count and the codec's audio.
    const pieces = [];
    const audioDone = new Promise((resolve) => {
      this.codec.onAudio = (pcm, final) => {
        if (pcm.length) {
          pieces.push(pcm);
          onAudio?.(pcm, this.sampleRate);
        }
        if (final) resolve();
      };
    });
    this.frames.onFrame = (n) => onFrame(n, budget);
    const run = await this.frames.call("run", { input: x, budget }, [x.buffer]);
    const t2 = performance.now();
    await audioDone;
    const t3 = performance.now();
    const pcm = new Float32Array(pieces.reduce((n, a) => n + a.length, 0));
    let at = 0;
    for (const a of pieces) { pcm.set(a, at); at += a.length; }
    // Codebook-major [10][frames], as the codec reads them.
    const n = run.frames;
    const codes = new Uint32Array(nc * n);
    for (let t = 0; t < n; t++) for (let i = 0; i < nc; i++) codes[i * n + t] = run.out[t * width + 1 + i];
    return {
      pcm, sampleRate: this.sampleRate, codes, frames: n, finished: n < budget,
      ms: { prompt: Math.round(t1 - t0), frames: Math.round(t2 - t1), codec: Math.round(t3 - t2) },
    };
  }

  async #speakBatch(sentences, { maxFrames, temperature, topP, topK, seed = 0, reference, onFrame = () => {},
    onAudio = null, streamEvery = 32, batch } = {}) {
    const g = this.generation;
    const nc = this.config.num_codebooks;
    const width = 1 + nc;
    const context = this.config.max_seq_len;
    const sampling = { temperature: temperature ?? g.temperature, topP: topP ?? g.top_p, topK: topK ?? g.top_k };
    const t0 = performance.now();
    await this.codec.call("reset", { streamEvery });
    const pieces = [];
    const audioDone = new Promise((resolve) => {
      this.codec.onAudio = (pcm, final) => {
        if (pcm.length) {
          pieces.push(pcm);
          onAudio?.(pcm, this.sampleRate);
        }
        if (final) resolve();
      };
    });
    const prompt = (t, r) => {
      const rows = audio8_prompt(this.tokenizer, this.configJson, t, r?.text, r?.codes);
      if (rows.length / width >= context) throw new Error(`prompt length ${rows.length / width} must be smaller than ${context}`);
      return rows;
    };
    const toCodes = (flat) => {
      const n = flat.length / width;
      const codes = new Uint32Array(nc * n);
      for (let t = 0; t < n; t++) for (let i = 0; i < nc; i++) codes[i * n + t] = flat[t * width + 1 + i];
      return codes;
    };
    const all = []; // rows, text order
    let made = 0;
    let finished = true;
    let rest = sentences;
    let first = 1; // seed offset of `rest[0]`
    const chunks = Math.ceil((reference ? sentences.length : sentences.length - 1) / batch);
    if (!reference) {
      // The voice: the first sentence alone, streamed as it is made.
      const rows = prompt(sentences[0]);
      const budget = Math.min(maxFrames ?? g.max_new_tokens, context - rows.length / width);
      const { cap } = await this.slow[0].call("capacity", { prompt: rows.length / width, maxNew: budget });
      await this.frames.call("restart", { ...sampling, seed });
      let x = Uint32Array.from(rows);
      for (const s of this.slow) x = (await s.call("prefill", { input: x, cap }, [x.buffer])).out;
      this.frames.onFrame = (n) => onFrame(n, budget);
      const run = await this.frames.call("run", { input: x, budget, end: chunks === 0 }, [x.buffer]);
      finished &&= run.frames < budget;
      made += run.frames;
      all.push(run.out);
      reference = { text: sentences[0], codes: toCodes(run.out) };
      rest = sentences.slice(1);
    } else {
      first = 0;
    }
    for (let c = 0; c < chunks; c++) {
      const part = rest.slice(c * batch, (c + 1) * batch);
      const prompts = part.map((t) => prompt(t, reference));
      const longest = Math.max(...prompts.map((r) => r.length / width));
      const budget = Math.min(maxFrames ?? g.max_new_tokens, context - 1 - longest);
      const { cap } = await this.slow[0].call("capacity", { prompt: longest, maxNew: budget });
      const seeds = part.map((_, i) => seed + first + c * batch + i);
      await this.frames.call("restartBatch", { ...sampling, seeds });
      for (const rows of prompts) {
        let x = Uint32Array.from(rows);
        for (const s of this.slow) x = (await s.call("queue", { input: x, cap }, [x.buffer])).out;
      }
      let x;
      for (const s of this.slow) x = (await s.call("join")).out;
      const base = made;
      this.frames.onFrame = (n) => onFrame(base + n, budget * part.length);
      const run = await this.frames.call("runBatch", { input: x, budget, n: part.length, end: c + 1 === chunks }, [x.buffer]);
      finished &&= run.counts.every((n) => n < budget);
      made += run.frames;
      all.push(run.out);
    }
    const t1 = performance.now();
    await audioDone;
    const t2 = performance.now();
    const pcm = new Float32Array(pieces.reduce((n, a) => n + a.length, 0));
    let at = 0;
    for (const a of pieces) { pcm.set(a, at); at += a.length; }
    const flat = new Uint32Array(made * width);
    at = 0;
    for (const a of all) { flat.set(a, at); at += a.length; }
    return {
      pcm, sampleRate: this.sampleRate, codes: toCodes(flat), frames: made, finished, sentences: sentences.length,
      ms: { prompt: 0, frames: Math.round(t1 - t0), codec: Math.round(t2 - t1) },
    };
  }

  /** Per worker and call: total ms and count since the last `resetTimes`. */
  times() {
    return Object.fromEntries([...this.slow, this.frames, this.codec].map((w) => [w.label,
      Object.fromEntries(Object.entries(w.time).map(([op, t]) => [op, { ms: Math.round(t.ms), n: t.n, avg: +(t.ms / t.n).toFixed(1) }]))]));
  }

  resetTimes() {
    [...this.slow, this.frames, this.codec].forEach((w) => (w.time = {}));
  }

  /** wasm memory of every worker, MB. */
  async memory() {
    const all = [...this.slow, this.frames, this.codec];
    return Promise.all(all.map(async (w) => ({ worker: w.label, memory_mb: (await w.call("memory")).memory_mb })));
  }

  terminate() {
    clearInterval(this.watch);
    [...this.slow, this.frames, this.codec].forEach((w) => w.worker.terminate());
  }
}

/** 16-bit mono WAV bytes of `pcm`. */
export function wavBytes(pcm, sampleRate) {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, pcm.length * 2, true);
  pcm.forEach((s, i) => v.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, s)) * 32767), true));
  return new Uint8Array(buf);
}
