// One worker of genmoji.html's image pipeline (the files `rlx-genmoji
// --export-web` writes). Roles:
//   flux — a stage of the FLUX.1 denoiser (`FluxStage`), its tensors fetched
//          by HTTP Range from the denoiser GGUF;
//   t5   — a stage of the T5 encoder (`T5Stage`), the same way;
//   aux  — CLIP's pooled embedding and the VAE decoder (`AuxStage`, whole file).
import init, { AuxStage, Files, FluxStage, FluxStageFetch, T5Stage, T5StageFetch, init_webgpu } from "./pkg/rlx_models_web.js";

let wasm = null;
let model = null;
let role = null;
const memoryMb = () => Math.round(wasm.memory.buffer.byteLength / 2 ** 20);

async function range(url, start, end) {
  const r = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
  if (r.status !== 206) throw new Error(`${url}: Range request answered ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

// A stage plan from the file's header (prefix grown until it parses), filled
// with its byte ranges in 64 MB pieces.
async function fetchStage(url, make) {
  let plan = null;
  for (let mb = 4; !plan; mb *= 2) {
    const prefix = await range(url, 0, mb << 20);
    try {
      plan = make(prefix);
    } catch (e) {
      if (mb >= 256 || !/header|truncat|eof|short/i.test(String(e?.message ?? e))) throw e;
    }
  }
  const r = plan.ranges();
  const PIECE = 64 << 20;
  let bytes = 0;
  for (let i = 0; i < r.length; i += 2) {
    for (let s = r[i]; s < r[i + 1]; s += PIECE) {
      const e = Math.min(s + PIECE, r[i + 1]);
      plan.add(s, await range(url, s, e));
      bytes += e - s;
      postMessage({ progress: { bytes } });
    }
  }
  return { plan, bytes };
}

const handlers = {
  async load(a) {
    wasm = await init();
    if (a.backend === "webgpu" && !(await init_webgpu())) throw new Error("no WebGPU adapter in this worker");
    role = a.role;
    const t0 = performance.now();
    let bytes = 0;
    let parts = "";
    if (role === "flux") {
      const f = await fetchStage(a.url, (p) => new FluxStageFetch(p, a.index, a.count));
      parts = f.plan.parts();
      bytes = f.bytes;
      model = FluxStage.from_fetch(f.plan, a.backend, a.txtLen, a.gridH, a.gridW, a.group ?? 2);
    } else if (role === "t5") {
      const f = await fetchStage(a.url, (p) => new T5StageFetch(p, a.index, a.count));
      bytes = f.bytes;
      model = T5Stage.from_fetch(f.plan, a.backend, a.txtLen);
    } else if (role === "aux") {
      const r = await fetch(a.url);
      if (!r.ok) throw new Error(`${a.url}: HTTP ${r.status}`);
      const buf = new Uint8Array(await r.arrayBuffer());
      bytes = buf.length;
      const files = new Files();
      files.add("clip-vae.gguf", buf);
      model = new AuxStage(files, a.backend, a.latentH, a.latentW);
    } else {
      throw new Error(`unknown role ${role}`);
    }
    return { role, parts, mb: Math.round(bytes / 2 ** 20), ms: Math.round(performance.now() - t0), memory_mb: memoryMb() };
  },
  // flux: input = packed latents (first stage, with txt) or hidden rows.
  async flux({ input, txt, vec, pooled, t, guidance }) {
    if (model.is_first()) vec = model.vec(t, guidance, pooled);
    const out = await model.forward_async(input, txt ?? new Float32Array(0), vec);
    return { out, vec };
  },
  // t5: ids (first stage) or hidden rows.
  async t5({ ids, x }) {
    const input = model.is_first() ? model.embed(ids) : x;
    return { out: await model.forward_async(input) };
  },
  async pooled({ ids }) {
    return { out: await model.pooled_async(ids) };
  },
  async decode({ latents }) {
    return { out: await model.decode_async(latents) };
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
