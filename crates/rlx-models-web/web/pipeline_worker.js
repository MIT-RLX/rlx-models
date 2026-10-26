// One stage of a `Qwen35Pipeline` (pipeline.js): loads only its share of the
// layers into this worker's own wasm instance and WebGPU device, then runs
// prefill / decode steps. Stages are chained by MessageChannel: a step comes
// from the page (first stage) or the previous stage's port, and its result
// goes straight to the next stage's port — hidden rows as transferred
// Float32Arrays, never through the page. The last stage answers the page.
import init, { Files, Qwen35Stage, Qwen35StageFetch, init_webgpu, set_setting } from "./pkg/rlx_models_web.js";

let stage = null;
let backend = "webgpu";
// Ports to the neighbouring stages (`link`); `next` is null on the last stage.
let next = null;
let wasm = null;
// This worker's wasm32 linear memory, MB (it only grows).
const memoryMb = () => Math.round(wasm.memory.buffer.byteLength / 2 ** 20);

// Bytes [start, end) of `url`, or null if the server answered with the whole
// file (no Range support) — then `whole` holds it.
async function fetchRange(url, start, end, onBytes = () => {}) {
  const r = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  if (r.status !== 206) return { whole: new Uint8Array(await r.arrayBuffer()) };
  // Streamed, so a long download (the Hugging Face Hub) can report progress.
  const out = new Uint8Array(end - start);
  let off = 0;
  const reader = r.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value.subarray(0, Math.max(0, Math.min(value.length, out.length - off))), off);
    off += value.length;
    onBytes(value.length);
  }
  return { bytes: out.subarray(0, Math.min(off, out.length)) };
}

// Download progress to the page, at most every 100 ms.
let lastReport = 0;
function reportProgress(fetched, total, force = false) {
  const now = performance.now();
  if (!force && now - lastReport < 100) return;
  lastReport = now;
  postMessage({ progress: { fetched, total } });
}

// The model file is read in fixed 32 MB cells aligned to absolute offsets and
// kept in the browser's Cache API, so a reload (any worker count, either mode)
// reads them back instead of downloading again. A cell that cannot be stored
// (quota, private mode) is simply not cached.
const CELL = 32 << 20;
const CACHE_NAME = "rlx-gguf-v1";
let cachePromise = null;
const openCache = () =>
  (cachePromise ??= (self.caches ? caches.open(CACHE_NAME) : Promise.resolve(null)).catch(() => null));
const cellKey = (url, k) => `${url}${url.includes("?") ? "&" : "?"}rlx-cell=${k}`;

// Cell `k` of `url`: from the cache, else downloaded (and stored). `whole` when
// the server ignored the Range request.
async function readCell(url, k, onBytes, stats) {
  const cache = await openCache();
  const key = cellKey(url, k);
  const hit = cache && (await cache.match(key));
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    stats.cached += bytes.length;
    onBytes(bytes.length);
    return { bytes };
  }
  const part = await fetchRange(url, k * CELL, (k + 1) * CELL, onBytes);
  if (part.whole) return part;
  stats.downloaded += part.bytes.length;
  if (cache) {
    // Copy: `part.bytes` may be a view on a larger buffer.
    await cache.put(key, new Response(part.bytes.slice())).catch(() => {});
  }
  return part;
}

// Only the parts of the GGUF this stage reads: the header (cells until it
// parses), then the cells holding the stage's tensor ranges.
async function fetchStageParts(url, index, count) {
  const stats = { cached: 0, downloaded: 0 };
  let fetched = 0;
  let total = CELL;
  const onBytes = (n) => reportProgress((fetched += n), total);
  const cells = new Map();
  const cell = async (k) => {
    if (!cells.has(k)) cells.set(k, await readCell(url, k, onBytes, stats));
    return cells.get(k);
  };
  let plan;
  for (let n = 1; ; n++) {
    const first = await cell(n - 1);
    if (first.whole) return { whole: first.whole, bytes: first.whole.length, stats };
    const parts = [];
    for (let k = 0; k < n; k++) parts.push((await cell(k)).bytes);
    const prefix = parts.length === 1 ? parts[0] : new Uint8Array(await new Blob(parts).arrayBuffer());
    try {
      plan = new Qwen35StageFetch(prefix, index, count);
      break;
    } catch (e) {
      // A header longer than the prefix: one more cell. Anything else is real.
      if (prefix.length < n * CELL || n >= 16 || !/header/.test(String(e?.message ?? e))) throw e;
    }
  }
  // The cells covering the stage's ranges; those read for the header count too.
  const r = plan.ranges();
  const need = new Set();
  for (let i = 0; i < r.length; i += 2) {
    for (let k = Math.floor(r[i] / CELL); k * CELL < r[i + 1]; k++) need.add(k);
  }
  total = fetched + [...need].filter((k) => !cells.has(k)).length * CELL;
  reportProgress(fetched, total, true);
  for (const k of [...need].sort((a, b) => a - b)) {
    const { bytes, whole } = await cell(k);
    if (whole) throw new Error(`${url}: server stopped honouring Range requests`);
    plan.add(k * CELL, bytes);
    cells.delete(k); // fed; let it go
  }
  return { plan, bytes: stats.cached + stats.downloaded, stats };
}

const handlers = {
  async load({ url, index, count, maxSeq, settings, backend: b, rows, prefill }) {
    wasm = await init();
    backend = b ?? "webgpu";
    if (backend === "webgpu" && !(await init_webgpu())) throw new Error("no WebGPU adapter in this worker");
    for (const [k, v] of Object.entries(settings ?? {})) set_setting(k, String(v));
    const parts = await fetchStageParts(url, index, count);
    const layers = parts.plan?.layers ? Array.from(parts.plan.layers()) : null;
    const t0 = performance.now();
    if (parts.whole) {
      const f = new Files();
      f.add("model.gguf", parts.whole);
      stage = new Qwen35Stage(f, backend, maxSeq, index, count, rows, prefill);
    } else {
      stage = Qwen35Stage.from_fetch(parts.plan, backend, maxSeq, rows, prefill);
    }
    return { build_ms: Math.round(performance.now() - t0), memory_mb: memoryMb(),
      fetched_mb: Math.round(parts.bytes / 2 ** 20), ranged: !parts.whole, layers,
      downloaded_mb: Math.round(parts.stats.downloaded / 2 ** 20), cached_mb: Math.round(parts.stats.cached / 2 ** 20) };
  },
  async prefill({ input }) {
    return { out: backend === "cpu" ? stage.prefill(input) : await stage.prefill_async(input) };
  },
  // Sampling for this stage's tokens (meaningful on the last stage): null =
  // greedy, else { temperature, topP, topK, presencePenalty, seed }.
  async sampling({ opts }) {
    if (!opts) stage.set_greedy();
    else stage.set_sampling(opts.temperature, opts.topP, opts.topK, opts.presencePenalty, opts.seed);
    return {};
  },
  async memory() {
    return { memory_mb: memoryMb() };
  },
  // Logits of `ids` as the next token after the last prefill (last stage).
  async scores({ ids }) {
    return { scores: Float32Array.from(stage.scores(Uint32Array.from(ids))) };
  },
  // Wire this stage to its neighbours: steps arrive on `prev` (except on the
  // first stage, which hears from the page), results leave on `next`.
  async link({ prev, next: n }) {
    next = n ?? null;
    if (prev) prev.onmessage = ({ data }) => runStep(data);
    return {};
  },
  async decode({ input }) {
    return { out: backend === "cpu" ? stage.decode(input) : await stage.decode_async(input) };
  },
  // A multi-row step (built with `rows` > 1): up to that many positions at
  // once. With `undo`, `rowsUndo` can cut it back afterwards.
  async rows({ input, undo, guesses }) {
    const g = guesses ? Uint32Array.from(guesses) : undefined;
    return { out: backend === "cpu" ? stage.rows(input, !!undo, g) : await stage.rows_async(input, !!undo, g) };
  },
  // Keep only the first `keep` rows of the last undoable step. Every stage
  // does this on its own: the rows it re-runs are a prefix of what it was given.
  async rowsUndo({ keep }) {
    if (backend === "cpu") stage.rows_undo(keep);
    else await stage.rows_undo_async(keep);
    return {};
  },
};

// One pipeline step: run this stage, then hand the output on — to the next
// stage, or to the page from the last one. Errors go to the page directly,
// under the step's id, so the caller hears about a failure in any stage.
async function runStep({ id, kind, input, undo, guesses }) {
  try {
    const out = await handlers[kind]({ input, undo, guesses });
    const transfer = ArrayBuffer.isView(out.out) ? [out.out.buffer] : [];
    if (next) next.postMessage({ id, kind, input: out.out, undo, guesses }, transfer);
    else postMessage({ id, ok: true, out: out.out }, transfer);
  } catch (e) {
    postMessage({ id, ok: false, error: String(e?.message ?? e) });
  }
}

onmessage = async ({ data }) => {
  const { id, op } = data;
  if (op === "run") return runStep(data);
  try {
    const res = await handlers[op](data);
    const transfer = res.out instanceof Float32Array ? [res.out.buffer] : [];
    postMessage({ id, ok: true, ...res }, transfer);
  } catch (e) {
    postMessage({ id, ok: false, error: String(e?.message ?? e) });
  }
};
