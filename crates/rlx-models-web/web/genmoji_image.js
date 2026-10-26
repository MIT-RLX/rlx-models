// genmoji.html's image side: FLUX.1 [dev] + the Open Genmoji LoRA + Hyper-SD's
// 8-step LoRA (merged in), Q8_0, split across Web Workers — the files
// `rlx-genmoji --fast --export-web DIR` writes:
//   flux1-dev-genmoji-fast-q8_0.gguf  denoiser, 8 stages (each < 2 GiB: wasm32's
//                                     largest allocation), 2 blocks per graph
//   t5-v1_1-xxl-q8_0.gguf             T5 encoder, 4 stages
//   clip-vae.gguf                     CLIP pooled embedding + VAE decoder
//   t5-tokenizer.json, clip-tokenizer.json
// Each worker fetches only its own tensors by HTTP Range. The sampler (6
// Euler steps, guidance 5, seed 7, T5 at 128, 160 px) is `rlx-genmoji --fast`'s.
import init, { FluxTokenizers, flux1_noise, flux1_pack, flux1_schedule, flux1_unpack } from "./pkg/rlx_models_web.js";

const PX = 160;
const LATENT = PX / 8; // 20
const GRID = LATENT / 2; // 10 × 10 patches
const TXT_LEN = 128;
const CHANNELS = 16;

class Worker1 {
  constructor(url, onProgress) {
    this.w = new Worker(url, { type: "module" });
    this.pending = new Map();
    this.next = 0;
    this.w.onmessage = ({ data }) => {
      if (data.progress) return onProgress?.(data.progress.bytes);
      const p = this.pending.get(data.id);
      this.pending.delete(data.id);
      data.ok ? p.ok(data) : p.err(new Error(data.error));
    };
    this.w.onerror = (e) => {
      for (const p of this.pending.values()) p.err(new Error(e.message ?? "worker error"));
      this.pending.clear();
    };
  }

  call(op, args = {}) {
    return new Promise((ok, err) => {
      const id = this.next++;
      this.pending.set(id, { ok, err });
      this.w.postMessage({ id, op, ...args });
    });
  }

  terminate() {
    this.w.terminate();
  }
}

export class ImagePipeline {
  /**
   * @param {object} o
   * @param {string} o.base  directory URL holding the exported files
   * @param {number} [o.fluxWorkers=8] denoiser stages (fewer than 8 overflows wasm32)
   * @param {number} [o.t5Workers=4]
   * @param {(msg: string) => void} [o.onProgress]
   * @param {(fraction: number) => void} [o.onDownload]
   */
  static async create({ base, fluxWorkers = 8, t5Workers = 4, backend = "webgpu", group = 2,
    onProgress = () => {}, onDownload = () => {} }) {
    await init();
    const get = async (name) => {
      const r = await fetch(`${base}/${name}`);
      if (!r.ok) throw new Error(`${base}/${name}: HTTP ${r.status}`);
      return new Uint8Array(await r.arrayBuffer());
    };
    const tok = new FluxTokenizers(await get("t5-tokenizer.json"), await get("clip-tokenizer.json"));
    const sizes = {};
    for (const f of ["flux1-dev-genmoji-fast-q8_0.gguf", "t5-v1_1-xxl-q8_0.gguf", "clip-vae.gguf"]) {
      const r = await fetch(`${base}/${f}`, { method: "HEAD" });
      sizes[f] = Number(r.headers.get("content-length") ?? 0);
    }
    const total = Object.values(sizes).reduce((a, b) => a + b, 0) || 1;
    const got = [];
    const url = new URL("./genmoji_image_worker.js", import.meta.url);
    const make = () => {
      const i = got.length;
      got.push(0);
      return new Worker1(url, (bytes) => {
        got[i] = bytes;
        onDownload(got.reduce((a, b) => a + b, 0) / total);
      });
    };
    const p = new ImagePipeline();
    p.tok = tok;
    try {
      p.aux = make();
      const a = await p.aux.call("load", { role: "aux", url: `${base}/clip-vae.gguf`, backend, latentH: LATENT, latentW: LATENT });
      onProgress(`CLIP + VAE ready (${a.mb} MB, ${a.ms} ms)`);
      p.t5 = [];
      for (let i = 0; i < t5Workers; i++) {
        const w = make();
        const r = await w.call("load", { role: "t5", url: `${base}/t5-v1_1-xxl-q8_0.gguf`, index: i, count: t5Workers, backend, txtLen: TXT_LEN });
        p.t5.push(w);
        onProgress(`T5 stage ${i + 1}/${t5Workers} ready (${r.mb} MB, ${r.ms} ms, wasm ${r.memory_mb} MB)`);
      }
      p.flux = [];
      for (let i = 0; i < fluxWorkers; i++) {
        const w = make();
        const r = await w.call("load", { role: "flux", url: `${base}/flux1-dev-genmoji-fast-q8_0.gguf`, index: i, count: fluxWorkers,
          backend, txtLen: TXT_LEN, gridH: GRID, gridW: GRID, group });
        p.flux.push(w);
        onProgress(`denoiser stage ${i + 1}/${fluxWorkers} ready: ${r.parts.replaceAll(",", " ")} (${r.mb} MB, ${r.ms} ms)`);
      }
    } catch (e) {
      p.terminate();
      throw e;
    }
    return p;
  }

  /** Workers engaged. */
  get workers() {
    return 1 + this.t5.length + this.flux.length;
  }

  /**
   * Render `prompt` (an assembled FLUX prompt) at 160 px; resolves to RGB8
   * bytes. `onStep(i, n)` after each denoising step.
   */
  async generate(prompt, { steps = 6, guidance = 5.0, seed = 7, onStep = () => {} } = {}) {
    const t0 = performance.now();
    const ids = Uint32Array.from(this.tok.t5_ids(prompt, TXT_LEN));
    let x = null;
    for (const w of this.t5) x = (await w.call("t5", { ids, x })).out;
    const txt = x;
    const pooled = (await this.aux.call("pooled", { ids: Uint32Array.from(this.tok.clip_ids(prompt)) })).out;
    const t1 = performance.now();
    const noise = flux1_noise(CHANNELS * LATENT * LATENT, seed);
    let z = flux1_pack(noise, CHANNELS, LATENT, LATENT);
    const ts = flux1_schedule(steps, GRID * GRID, true);
    const stepMs = [];
    for (let i = 0; i < steps; i++) {
      const s0 = performance.now();
      let h = z;
      let vec = null;
      for (let k = 0; k < this.flux.length; k++) {
        const r = await this.flux[k].call("flux", { input: h, txt: k === 0 ? txt : null, vec, pooled, t: ts[i], guidance });
        h = r.out;
        vec = r.vec;
      }
      const dt = ts[i + 1] - ts[i];
      for (let j = 0; j < z.length; j++) z[j] += dt * h[j];
      stepMs.push(Math.round(performance.now() - s0));
      onStep(i + 1, steps);
    }
    const t2 = performance.now();
    const rgb = (await this.aux.call("decode", { latents: flux1_unpack(z, CHANNELS, LATENT, LATENT) })).out;
    const t3 = performance.now();
    this.stats = { text_ms: Math.round(t1 - t0), denoise_ms: Math.round(t2 - t1), step_ms: stepMs,
      decode_ms: Math.round(t3 - t2), total_ms: Math.round(t3 - t0) };
    return rgb;
  }

  terminate() {
    for (const w of [this.aux, ...(this.t5 ?? []), ...(this.flux ?? [])]) w?.terminate();
  }
}

/**
 * The sticker: `rgb` (`size`² RGB8 on white) upscaled to `out` px, then the
 * white background cut away — flood-filled from the border through
 * near-white pixels, with a soft edge. The vendor's learned matting (and
 * `rlx-genmoji`'s) does not run in a browser; this is the classic cut-out.
 */
export function sticker(rgb, size, out = 512) {
  const src = document.createElement("canvas");
  src.width = src.height = size;
  const sctx = src.getContext("2d");
  const id = sctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    id.data[4 * i] = rgb[3 * i];
    id.data[4 * i + 1] = rgb[3 * i + 1];
    id.data[4 * i + 2] = rgb[3 * i + 2];
    id.data[4 * i + 3] = 255;
  }
  sctx.putImageData(id, 0, 0);
  const dst = document.createElement("canvas");
  dst.width = dst.height = out;
  const ctx = dst.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, 0, 0, out, out);
  const img = ctx.getImageData(0, 0, out, out);
  const d = img.data;
  const n = out * out;
  // Distance from white: the largest channel shortfall.
  const dist = new Uint8Array(n);
  for (let i = 0; i < n; i++) dist[i] = 255 - Math.min(d[4 * i], d[4 * i + 1], d[4 * i + 2]);
  const LO = 10;
  const HI = 48;
  const bg = new Uint8Array(n);
  const stack = [];
  for (let x = 0; x < out; x++) stack.push(x, (out - 1) * out + x);
  for (let y = 0; y < out; y++) stack.push(y * out, y * out + out - 1);
  while (stack.length) {
    const i = stack.pop();
    if (bg[i] || dist[i] > HI) continue;
    bg[i] = 1;
    const x = i % out;
    const y = (i / out) | 0;
    if (x > 0) stack.push(i - 1);
    if (x < out - 1) stack.push(i + 1);
    if (y > 0) stack.push(i - out);
    if (y < out - 1) stack.push(i + out);
  }
  for (let i = 0; i < n; i++) {
    if (!bg[i]) continue;
    // Reached from the border: background, fading in over the edge band.
    const a = Math.min(1, Math.max(0, (dist[i] - LO) / (HI - LO)));
    d[4 * i + 3] = Math.round(255 * a);
    if (a > 0) {
      // Un-mix the white: C = a·F + (1 − a)·255.
      for (let c = 0; c < 3; c++) d[4 * i + c] = Math.max(0, Math.min(255, Math.round((d[4 * i + c] - (1 - a) * 255) / a)));
    }
  }
  ctx.putImageData(img, 0, 0);
  return dst;
}
