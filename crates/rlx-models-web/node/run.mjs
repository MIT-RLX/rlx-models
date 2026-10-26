// Run the wasm build under Node: `node crates/rlx-models-web/node/run.mjs <task> …`
//   asr   <checkpoint dir> <16 kHz PCM16 wav>
//   embed <checkpoint dir> embeddinggemma|jina-v5 <text>…
//   gen   <checkpoint dir> <prompt> [max_tokens]
import { createRequire } from "node:module";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const require = createRequire(import.meta.url);
const web = require("../pkg-node/rlx_models_web.js");

// Every file under `dir` except docs; names are relative, `/`-separated.
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

function readWav16k(path) {
  const b = readFileSync(path);
  let off = 12;
  while (off < b.length) {
    const id = b.toString("ascii", off, off + 4), len = b.readUInt32LE(off + 4);
    if (id === "data") {
      const n = len / 2, pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) pcm[i] = b.readInt16LE(off + 8 + 2 * i) / 32768;
      return pcm;
    }
    off += 8 + len + (len & 1);
  }
  throw new Error("no data chunk");
}

const [task, dir, ...rest] = process.argv.slice(2);
const t0 = performance.now();
const files = loadDir(dir);
const mb = (files.size / 1e6).toFixed(0);
if (task === "asr") {
  const asr = new web.Transcriber(files);
  const t1 = performance.now();
  const pcm = readWav16k(rest[0]);
  const text = asr.transcribe(pcm);
  const t2 = performance.now();
  asr.transcribe(pcm); // graphs compiled: steady-state cost
  const t3 = performance.now();
  console.log(text);
  const secs = pcm.length / 16000;
  console.error(`load ${mb} MB in ${((t1 - t0) / 1e3).toFixed(1)} s; ${secs.toFixed(1)} s audio: first ${((t2 - t1) / 1e3).toFixed(1)} s (incl. compile), again ${((t3 - t2) / 1e3).toFixed(1)} s = ${(secs / ((t3 - t2) / 1e3)).toFixed(2)}× real time`);
} else if (task === "embed") {
  const [model, ...texts] = rest;
  const e = new web.Embedder(files, model, "retrieval");
  const t1 = performance.now();
  const vs = texts.map((t) => e.embed(t, "document"));
  const t2 = performance.now();
  for (const v of vs) console.log(JSON.stringify(Array.from(v.slice(0, 4)).map((x) => +x.toFixed(6))));
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  if (vs.length > 1) console.log("cosine[0,1] =", dot(vs[0], vs[1]).toFixed(5));
  console.error(`load ${mb} MB in ${((t1 - t0) / 1e3).toFixed(1)} s; ${texts.length} texts in ${((t2 - t1) / 1e3).toFixed(1)} s`);
} else if (task === "gen") {
  const g = new web.TextGenerator(files);
  const t1 = performance.now();
  const out = g.generate(rest[0], Number(rest[1] ?? 16));
  const t2 = performance.now();
  console.log(out);
  console.error(`load ${mb} MB in ${((t1 - t0) / 1e3).toFixed(1)} s; generate in ${((t2 - t1) / 1e3).toFixed(1)} s`);
} else {
  throw new Error("task: asr | embed | gen");
}
