// One of the workers `two_workers.html` starts: load a Qwen3.5 GGUF into this
// worker's own wasm instance and WebGPU device, generate, report timings.
import init, { Files, Qwen35Generator, init_webgpu } from "./pkg/rlx_models_web.js";

onmessage = async ({ data: { id, base, files, prompt, n } }) => {
  const now = () => performance.now();
  try {
    await init();
    if (!(await init_webgpu())) throw new Error("no WebGPU adapter in this worker");
    const f = new Files();
    for (const name of files) f.add(name, new Uint8Array(await (await fetch(`${base}/${name}`)).arrayBuffer()));
    const t0 = now();
    const gen = new Qwen35Generator(f, "webgpu", 256);
    const t1 = now();
    const text = await gen.generate_async(prompt, n);
    const t2 = now();
    const text2 = await gen.generate_async(prompt, n);
    const t3 = now();
    postMessage({ id, ok: true, build_ms: Math.round(t1 - t0), first_ms: Math.round(t2 - t1),
      warm_ms: Math.round(t3 - t2), tok_per_s: +(n / ((t3 - t2) / 1e3)).toFixed(1),
      same_text: text === text2, sample: text.slice(0, 80) });
  } catch (e) {
    postMessage({ id, ok: false, error: String(e?.message ?? e) });
  }
};
