// One worker of an `Audio8Pipeline` (audio8.js), in its own wasm32 instance:
// a slow-layer stage, the frame maker (fast decoder + sampler) or the codec.
// A slow or frames worker fetches only its own tensors of model.safetensors
// with HTTP Range requests, straight into the buffer it keeps (Audio8Fetch).
import init, { Audio8Codec, Audio8Fetch, Audio8Frames, Audio8Slow, Files, init_webgpu, set_gpu_fast_math, set_setting } from "./pkg/rlx_models_web.js";

// Send this worker's errors and warnings (WebGPU validation errors arrive as
// console messages) to the page's log: a hang otherwise shows nothing there.
for (const level of ["error", "warn"]) {
  const orig = console[level].bind(console);
  console[level] = (...args) => {
    orig(...args);
    postMessage({ log: `${level}: ${args.map(String).join(" ").slice(0, 500)}` });
  };
}
self.addEventListener("error", (e) => postMessage({ log: `uncaught: ${e.message}` }));
self.addEventListener("unhandledrejection", (e) => postMessage({ log: `unhandled rejection: ${e.reason?.message ?? e.reason}` }));

let wasm = null;
let model = null;
let role = null;
// The ring of direct channels (`link`): a frame never passes through the
// page. A slow stage reads rows or hidden rows on `prev` and sends its output
// on `next`; the frames worker sends each row to the first stage and the codec
// and reads the last stage's output back; the codec reads rows and sends
// audio to the page.
const ports = { prev: null, next: null, codec: null };
let fromLast = null; // frames: resolves with the last stage's next output
// WebGPU cannot block on a read-back, so there every run is a `*_async` call.
let gpu = false;
const memoryMb = () => Math.round(wasm.memory.buffer.byteLength / 2 ** 20);

// Bytes [start, end) of `url` (a 206), or the whole file when the server
// ignores Range (a 200).
async function fetchRange(url, start, end) {
  const r = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  if (r.status !== 206) throw new Error(`${url}: the server ignores Range requests`);
  return bytes;
}

// This worker's tensors of a safetensors file, in pieces of at most 32 MB.
async function fetchTensors(url, config, role, index, count, onBytes) {
  let prefix = await fetchRange(url, 0, 1 << 20);
  let plan;
  for (let size = 1 << 20; ; size *= 4) {
    try {
      plan = new Audio8Fetch(prefix, config, role, index, count);
      break;
    } catch (e) {
      if (!/prefix too short/.test(String(e?.message ?? e)) || size > 1 << 28) throw e;
      prefix = await fetchRange(url, 0, size * 4);
    }
  }
  const r = plan.ranges();
  const PIECE = 32 << 20;
  let fetched = 0;
  for (let i = 0; i < r.length; i += 2) {
    for (let a = r[i]; a < r[i + 1]; a += PIECE) {
      const b = Math.min(a + PIECE, r[i + 1]);
      plan.add(a, await fetchRange(url, a, b));
      fetched += b - a;
      onBytes(fetched);
    }
  }
  return { plan, fetched };
}

const handlers = {
  async load({ role: r, base, config, index, count, backend = "cpu", clone = false, fastMath = false, settings = {} }) {
    wasm = await init();
    role = r;
    gpu = backend === "webgpu";
    if (gpu && !(await init_webgpu())) throw new Error("no WebGPU adapter in this worker");
    // wgpu's f32 matmuls beat its bf16 ones here (a slow step 98 → 62 ms, the
    // fast decoder 166 → 78 ms natively), at twice the weight memory.
    if (gpu) set_setting("RLX_WEIGHT_STORAGE", "f32");
    // Split-K decode matmuls (see `set_gpu_fast_math`), before any compile.
    // 16 columns a workgroup: the 896-wide layers here measured 82 → 62 ms a
    // frame against the default 32, the same speech.
    if (gpu && fastMath) {
      set_gpu_fast_math(true);
      if (!("RLX_WGPU_GEMV_COLS" in settings)) set_setting("RLX_WGPU_GEMV_COLS", "16");
    }
    for (const [k, v] of Object.entries(settings)) set_setting(k, String(v));
    const progress = (n) => postMessage({ progress: { fetched: n } });
    const t0 = performance.now();
    let fetched = 0;
    if (role === "codec") {
      const files = new Files();
      for (const name of clone ? ["codec_decoder.safetensors", "codec_encoder.safetensors"] : ["codec_decoder.safetensors"]) {
        const r = await fetch(`${base}/${name}`);
        if (!r.ok) throw new Error(`${base}/${name}: HTTP ${r.status} (run rlx-fish export-web)`);
        const bytes = new Uint8Array(await r.arrayBuffer());
        fetched += bytes.length;
        progress(fetched);
        files.add(name, bytes);
      }
      model = new Audio8Codec(files, backend);
    } else {
      const got = await fetchTensors(`${base}/model.safetensors`, config, role, index, count, progress);
      fetched = got.fetched;
      model = role === "slow"
        ? Audio8Slow.from_fetch(got.plan, config, index, count, backend)
        : Audio8Frames.from_fetch(got.plan, config, backend);
    }
    return { load_ms: Math.round(performance.now() - t0), fetched_mb: Math.round(fetched / 2 ** 20), memory_mb: memoryMb() };
  },
  async capacity({ prompt, maxNew }) {
    return { cap: model.capacity(prompt, maxNew) };
  },
  async prefill({ input, cap }) {
    return { out: gpu ? await model.prefill_async(input, cap) : Float32Array.from(model.prefill(input, cap)) };
  },
  async step({ input }) {
    return { out: gpu ? await model.step_async(input) : Float32Array.from(model.step(input)) };
  },
  // A batch of sentences: each queued (prefilled alone), joined, then
  // stepped together, one row per sentence a step.
  async queue({ input, cap }) {
    return { out: gpu ? await model.queue_async(input, cap) : Float32Array.from(model.queue(input, cap)) };
  },
  async join() {
    return { out: Float32Array.from(model.join()) };
  },
  async stepBatch({ input }) {
    return { out: gpu ? await model.step_batch_async(input) : Float32Array.from(model.step_batch(input)) };
  },
  async restartBatch({ temperature, topP, topK, seeds }) {
    model.restart_batch(temperature, topP, topK, Float64Array.from(seeds));
    return {};
  },
  async restart({ temperature, topP, topK, seed }) {
    model.restart(temperature, topP, topK, seed);
    return {};
  },
  async next({ input }) {
    const row = gpu ? await model.next_async(input) : model.next(input);
    return { out: row === undefined || row === null ? null : Uint32Array.from(row) };
  },
  async decode({ codes }) {
    const pcm = gpu ? await model.decode_async(codes) : Float32Array.from(model.decode(codes));
    return { out: pcm, sample_rate: model.sample_rate };
  },
  async decodeFrom({ codes, from }) {
    return { out: gpu ? await model.decode_from_async(codes, from) : Float32Array.from(model.decode_from(codes, from)) };
  },
  async encode({ pcm }) {
    return { out: gpu ? await model.encode_async(pcm) : Uint32Array.from(model.encode(pcm)) };
  },
  async memory() {
    return { memory_mb: memoryMb() };
  },
  // Wire this worker into the ring (see `ports`).
  async link({ prev, next, codec }) {
    Object.assign(ports, { prev: prev ?? null, next: next ?? null, codec: codec ?? null });
    if (role === "slow") {
      // A step arrives, runs here and leaves for the next worker, in order.
      let chain = Promise.resolve();
      prev.onmessage = ({ data }) => {
        chain = chain.then(async () => {
          const { out } = await (data.batch ? handlers.stepBatch : handlers.step)({ input: data.input });
          next.postMessage({ input: out, batch: data.batch }, [out.buffer]);
        }).catch((e) => postMessage({ log: `error: ${e?.message ?? e}` }));
      };
    } else if (role === "frames") {
      prev.onmessage = ({ data }) => fromLast?.(data.input);
    } else if (role === "codec") {
      prev.onmessage = ({ data }) => codecRow(data);
    }
    return {};
  },
  // frames: the whole utterance from the prompt's output `input`, up to
  // `budget` frames. Rows go to the first slow stage and to the codec.
  async run({ input, budget, end = true }) {
    let x = input;
    const rows = [];
    const t0 = performance.now();
    for (let step = 0; step < budget; step++) {
      const { out: row } = await handlers.next({ input: x });
      if (!row) break;
      rows.push(row);
      ports.codec.postMessage({ row: row.slice() });
      postMessage({ frame: rows.length });
      if (step + 1 < budget) {
        const back = new Promise((resolve) => (fromLast = resolve));
        const r = row.slice();
        ports.next.postMessage({ input: r }, [r.buffer]);
        x = await back;
      }
    }
    if (end) ports.codec.postMessage({ end: true });
    const flat = new Uint32Array(rows.length * (rows[0]?.length ?? 0));
    rows.forEach((r, i) => flat.set(r, i * r.length));
    return { out: flat, frames: rows.length, ms: Math.round(performance.now() - t0) };
  },
  // frames: `n` sentences together from the last stage's joined output,
  // up to `budget` frames each. The codec hears them in text order: the
  // earliest unfinished sentence's rows as they come, the others' once
  // every sentence before them has ended.
  async runBatch({ input, budget, n, end = true }) {
    let x = input;
    const rows = Array.from({ length: n }, () => []);
    const done = new Array(n).fill(false);
    let cur = 0;
    let made = 0;
    const toCodec = (row) => ports.codec.postMessage({ row: row.slice() });
    const advance = () => {
      while (cur < n && done[cur]) {
        cur += 1;
        if (cur < n) rows[cur].forEach(toCodec);
      }
    };
    const t0 = performance.now();
    for (let step = 0; step < budget; step++) {
      const out = gpu ? await model.next_batch_async(x) : model.next_batch(x);
      if (out === undefined || out === null) break;
      const width = out.length / n;
      for (let i = 0; i < n; i++) {
        if (done[i]) continue;
        if (model.ended(i)) {
          done[i] = true;
          continue;
        }
        const row = Uint32Array.from(out.subarray(i * width, (i + 1) * width));
        rows[i].push(row);
        made += 1;
        if (i === cur) toCodec(row);
      }
      advance();
      postMessage({ frame: made });
      if (step + 1 < budget) {
        const back = new Promise((resolve) => (fromLast = resolve));
        const r = Uint32Array.from(out);
        ports.next.postMessage({ input: r, batch: true }, [r.buffer]);
        x = await back;
      }
    }
    // Sentences cut by the budget count as ended.
    done.fill(true);
    advance();
    if (end) ports.codec.postMessage({ end: true });
    const width = rows.flat()[0]?.length ?? 0;
    const flat = new Uint32Array(made * width);
    rows.flat().forEach((r, i) => flat.set(r, i * width));
    return { out: flat, frames: made, counts: rows.map((r) => r.length), ms: Math.round(performance.now() - t0) };
  },
  // codec: forget the last utterance's frames.
  async reset({ streamEvery = 16 }) {
    Object.assign(codecState, { rows: [], sent: 0, streamed: 0, every: streamEvery, ended: false, done: false, busy: false });
    return {};
  },
};

// codec: frames as they arrive; the samples not sent yet are decoded whenever
// `every` new frames are in and no decode is running, the rest at the end.
const codecState = { rows: [], sent: 0, streamed: 0, every: 16, ended: false, done: false, busy: false };
function codecRow(data) {
  if (data.row) codecState.rows.push(data.row);
  if (data.end) codecState.ended = true;
  codecPump();
}
async function codecPump() {
  const s = codecState;
  if (s.busy) return;
  const pending = s.rows.length - s.sent;
  if (!(pending >= s.every || (s.ended && (pending > 0 || !s.done)))) return;
  s.busy = true;
  try {
    s.sent = s.rows.length;
    const n = s.sent;
    const nc = n ? s.rows[0].length - 1 : 0;
    let out = new Float32Array();
    if (n) {
      const codes = new Uint32Array(nc * n);
      for (let t = 0; t < n; t++) for (let i = 0; i < nc; i++) codes[i * n + t] = s.rows[t][1 + i];
      ({ out } = await handlers.decodeFrom({ codes, from: s.streamed }));
    }
    s.streamed += out.length;
    const final = s.ended && s.sent === s.rows.length;
    if (final) s.done = true;
    postMessage({ audio: out, final }, [out.buffer]);
  } catch (e) {
    postMessage({ log: `error: ${e?.message ?? e}` });
  } finally {
    s.busy = false;
  }
  codecPump();
}

onmessage = async ({ data }) => {
  const { id, op } = data;
  try {
    const res = await handlers[op](data);
    const transfer = ArrayBuffer.isView(res.out) ? [res.out.buffer] : [];
    postMessage({ id, ok: true, ...res }, transfer);
  } catch (e) {
    postMessage({ id, ok: false, error: String(e?.message ?? e) });
  }
};
