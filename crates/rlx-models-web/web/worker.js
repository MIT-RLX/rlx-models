// Runs the wasm models off the main thread. Messages in:
//   {type: "load", id, weightsBase, backend} fetch the checkpoint, build the model
//   {type: "asr", pcm}                       Float32Array, 16 kHz mono
//   {type: "embed", texts, prompt}
//   {type: "gen", prompt, maxTokens}
// Messages out: {type: "progress"|"ready"|"result"|"error", …}
import init, { Files, Transcriber, Embedder, TextGenerator, Qwen35Generator, init_webgpu } from "./pkg/rlx_models_web.js";
import { MODELS } from "./models.js";

let model = null;
let backend = "cpu";
const ready = init();

async function fetchBytes(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const out = new Uint8Array(total || 0);
  if (!total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  let off = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, off);
    off += value.length;
    onBytes(value.length);
  }
  return out;
}

onmessage = async ({ data: msg }) => {
  try {
    await ready;
    if (msg.type === "load") {
      model?.free();
      model = null;
      const spec = MODELS[msg.id];
      backend = msg.backend;
      if (backend === "webgpu" && !(await init_webgpu())) throw new Error("no WebGPU adapter in this browser");
      const files = new Files();
      let got = 0;
      for (const name of spec.files) {
        const bytes = await fetchBytes(`${msg.weightsBase}/${spec.dir}/${name}`, (n) => {
          got += n;
          postMessage({ type: "progress", text: `fetching ${name}: ${(got / 1e6).toFixed(0)} MB` });
        });
        files.add(name, bytes);
      }
      postMessage({ type: "progress", text: `building ${spec.label}…` });
      const t0 = performance.now();
      model =
        spec.kind === "asr" ? new Transcriber(files, backend)
        : spec.kind === "embed" ? new Embedder(files, msg.id, "retrieval", backend)
        : spec.kind === "qwen35" ? new Qwen35Generator(files, backend, 512)
        : new TextGenerator(files, backend);
      postMessage({ type: "ready", text: `${spec.label} (${backend}) ready in ${((performance.now() - t0) / 1e3).toFixed(1)} s` });
    } else if (msg.type === "asr") {
      const t0 = performance.now();
      const text = await model.transcribe_async(msg.pcm);
      const secs = msg.pcm.length / 16000, dt = (performance.now() - t0) / 1e3;
      postMessage({ type: "result", text, stats: `${secs.toFixed(1)} s audio in ${dt.toFixed(1)} s` });
    } else if (msg.type === "embed") {
      const t0 = performance.now();
      const vs = [];
      for (const t of msg.texts) vs.push(await model.embed_async(t, msg.prompt));
      const sims = vs.map((a) => vs.map((b) => a.reduce((s, x, i) => s + x * b[i], 0)));
      postMessage({ type: "result", sims, stats: `${vs.length} texts in ${((performance.now() - t0) / 1e3).toFixed(1)} s` });
    } else if (msg.type === "gen") {
      const t0 = performance.now();
      const text = await model.generate_async(msg.prompt, msg.maxTokens);
      postMessage({ type: "result", text, stats: `${((performance.now() - t0) / 1e3).toFixed(1)} s` });
    }
  } catch (e) {
    postMessage({ type: "error", text: String(e?.message ?? e) });
  }
};
