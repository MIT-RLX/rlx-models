// One FLUX.1 denoiser stage (flux_spike.html): fetch only this stage's tensors
// of the web GGUF (`rlx-genmoji --export-web`) by HTTP Range, compile them on
// WebGPU, run steps. The page chains the stages.
import init, { FluxStage, FluxStageFetch, init_webgpu } from "./pkg/rlx_models_web.js";

let wasm = null;
let stage = null;
const memoryMb = () => Math.round(wasm.memory.buffer.byteLength / 2 ** 20);

async function range(url, start, end) {
  const r = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
  if (r.status !== 206) throw new Error(`${url}: Range request answered ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

const handlers = {
  async load({ url, index, count, txtLen, gridH, gridW, backend, group }) {
    wasm = await init();
    if (backend === "webgpu" && !(await init_webgpu())) throw new Error("no WebGPU adapter");
    const t0 = performance.now();
    // The header: grow the prefix until it parses.
    let fetchPlan = null;
    for (let mb = 4; !fetchPlan; mb *= 2) {
      const prefix = await range(url, 0, mb << 20);
      try {
        fetchPlan = new FluxStageFetch(prefix, index, count);
      } catch (e) {
        if (mb >= 256 || !/header|truncat|eof|short/i.test(String(e?.message ?? e))) throw e;
      }
    }
    const r = fetchPlan.ranges();
    const PIECE = 64 << 20;
    let bytes = 0;
    for (let i = 0; i < r.length; i += 2) {
      for (let s = r[i]; s < r[i + 1]; s += PIECE) {
        const e = Math.min(s + PIECE, r[i + 1]);
        fetchPlan.add(s, await range(url, s, e));
        bytes += e - s;
        postMessage({ progress: { bytes } });
      }
    }
    const t1 = performance.now();
    const parts = fetchPlan.parts();
    stage = FluxStage.from_fetch(fetchPlan, backend, txtLen, gridH, gridW, group);
    return { parts, fetched_mb: Math.round(bytes / 2 ** 20), fetch_ms: Math.round(t1 - t0),
      build_ms: Math.round(performance.now() - t1), memory_mb: memoryMb(), first: stage.is_first() };
  },
  async step({ input, txt, vec, pooled, t, guidance }) {
    if (stage.is_first()) vec = stage.vec(t, guidance, pooled);
    const t0 = performance.now();
    const out = await stage.forward_async(input, txt ?? new Float32Array(0), vec);
    return { out, vec, ms: Math.round(performance.now() - t0), part_ms: Array.from(stage.part_ms(), Math.round) };
  },
};

onmessage = async ({ data }) => {
  const { id, op } = data;
  try {
    const res = await handlers[op](data);
    postMessage({ id, ok: true, ...res });
  } catch (e) {
    postMessage({ id, ok: false, error: String(e?.message ?? e) });
  }
};
