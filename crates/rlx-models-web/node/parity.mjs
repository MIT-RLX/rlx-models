// The wasm build against the same references the native parity tests use
// (scripts/{lm_hf,embedding_hf,moonshine_streaming}_reference.py output):
//   node crates/rlx-models-web/node/parity.mjs <weights root> <reference dir> [only]
// With no `only`, each model runs in its own child process: wasm32 memory
// never shrinks, and after a 1.2 GB model the next large one can fail to find
// contiguous space in the same instance.
import { createRequire } from "node:module";
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, relative } from "node:path";

const require = createRequire(import.meta.url);
const web = require("../pkg-node/rlx_models_web.js");
const [root, refs, only] = process.argv.slice(2);
const MODELS = ["moonshine-streaming-tiny", "embeddinggemma", "jina-v5", "lfm2.5-230m"];
if (!only) {
  const { spawnSync } = await import("node:child_process");
  let bad = 0;
  for (const m of MODELS) {
    const r = spawnSync(process.execPath, [process.argv[1], root, refs, m], { stdio: "inherit" });
    if (r.status !== 0) bad++;
  }
  process.exit(bad ? 1 : 0);
}
const want = (m) => m === only;

function loadDir(dir) {
  const files = new web.Files();
  const walk = (d) => {
    for (const e of readdirSync(d)) {
      const p = join(d, e);
      if (e.startsWith(".")) continue;
      if (statSync(p).isDirectory()) walk(p);
      else if (!/\.(md|txt|ipynb|py|jinja)$|LICENSE/.test(e)) files.add(relative(dir, p), readFileSync(p));
    }
  };
  walk(dir);
  return files;
}

function wav16k(path) {
  const b = readFileSync(path);
  for (let off = 12; off < b.length; ) {
    const id = b.toString("ascii", off, off + 4), len = b.readUInt32LE(off + 4);
    if (id === "data") return Float32Array.from({ length: len / 2 }, (_, i) => b.readInt16LE(off + 8 + 2 * i) / 32768);
    off += 8 + len + (len & 1);
  }
  throw new Error("no data chunk");
}

let failed = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${detail}`);
  if (!ok) failed++;
};
const ref = (f) => JSON.parse(readFileSync(join(refs, f), "utf8"));
const have = (f) => existsSync(join(refs, f));

if (want("moonshine-streaming-tiny") && have("moonshine-streaming-tiny.ref.json")) {
  const r = ref("moonshine-streaming-tiny.ref.json");
  const asr = new web.Transcriber(loadDir(join(root, "asr/moonshine-streaming-tiny")));
  const text = asr.transcribe(wav16k(join(root, "../assets/harness/jfk_16k.wav")));
  asr.free(); // wasm memory never shrinks: release each model before the next
  check("moonshine-streaming-tiny", text.trim() === r.text.trim(), JSON.stringify(text));
}

for (const [model, dir, file, prompt, task] of [
  ["embeddinggemma", "embed/embeddinggemma-300m", "embeddinggemma.ref.json", "query", null],
  ["jina-v5", "embed/jina-embeddings-v5-text-nano", "jina-v5-nano-retrieval.ref.json", "query", "retrieval"],
]) {
  if (!want(model) || !have(file)) continue;
  const r = ref(file);
  const e = new web.Embedder(loadDir(join(root, dir)), model, task);
  let worst = 0;
  for (const row of r.rows) {
    const v = e.embed(row.text, prompt);
    row.embedding.forEach((x, i) => (worst = Math.max(worst, Math.abs(x - v[i]))));
  }
  e.free();
  check(model, worst < 1e-5, `max |wasm − HF| = ${worst.toExponential(2)} over ${r.rows.length} texts`);
}

if (want("lfm2.5-230m") && have("lfm25-230m.ref.json")) {
  const r = ref("lfm25-230m.ref.json");
  const g = new web.TextGenerator(loadDir(join(root, "lm/lfm2.5-230m")));
  const got = Array.from(g.generate_ids(Uint32Array.from(r.prompt_ids), r.greedy.length));
  g.free();
  const diff = got.findIndex((t, i) => t !== r.greedy[i]);
  const ok = diff === -1 || r.margins[diff] < 0.05;
  check("lfm2.5-230m", ok, diff === -1 ? `${got.length} tokens identical` : `first divergence at step ${diff} (reference margin ${r.margins[diff].toFixed(4)})`);
}

process.exit(failed ? 1 : 0);
