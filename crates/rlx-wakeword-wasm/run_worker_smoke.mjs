#!/usr/bin/env node
/**
 * Smoke the worker protocol under Node (worker_threads + --target web build).
 * Requires: just wakeword-wasm-web
 */
import { Worker } from "node:worker_threads";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { existsSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkgJs = join(__dirname, "web", "pkg-web", "rlx_wakeword_wasm.js");
if (!existsSync(pkgJs)) {
  console.error("missing web/pkg-web — run: just wakeword-wasm-web");
  process.exit(1);
}

const pkgHref = pathToFileURL(pkgJs).href;
const wasmPath = join(__dirname, "web", "pkg-web", "rlx_wakeword_wasm_bg.wasm");
const wasmHref = pathToFileURL(wasmPath).href;

const workerSrc = `
import { parentPort } from "node:worker_threads";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import init, {
  WakeSession, sample_rate, smoke_score, worker_safe,
} from ${JSON.stringify(pkgHref)};

let session = null;
let ready = null;
async function ensureInit() {
  if (!ready) {
    // Web target defaults to fetch(URL); in Node load bytes instead.
    const bytes = readFileSync(fileURLToPath(${JSON.stringify(wasmHref)}));
    ready = init({ module_or_path: bytes });
  }
  await ready;
}
function reply(id, payload) { parentPort.postMessage({ id, ...payload }); }

parentPort.on("message", async (msg) => {
  const id = msg.id;
  try {
    await ensureInit();
    switch (msg.cmd) {
      case "init":
        reply(id, { ok: true, kind: "ready", workerSafe: worker_safe(), sampleRate: sample_rate() });
        break;
      case "smoke":
        reply(id, { ok: true, score: smoke_score() });
        break;
      case "session_new":
        if (session) { session.free(); session = null; }
        session = new WakeSession(msg.n ?? 1, msg.hopMs ?? 40, msg.mode ?? "tern-all");
        reply(id, { ok: true, phraseCount: session.phrase_count, hopSamples: session.hop_samples, mode: session.mode });
        break;
      case "push_peak": {
        if (!session) throw new Error("no session");
        const peaks = session.push_peak(msg.pcm);
        reply(id, { ok: true, peaks: Array.from(peaks) });
        break;
      }
      default:
        throw new Error("unknown cmd " + msg.cmd);
    }
  } catch (e) {
    reply(id, { ok: false, error: String(e && e.message ? e.message : e) });
  }
});
`;

function call(worker, cmd, payload = {}) {
  const id = Math.random().toString(36).slice(2);
  return new Promise((resolve, reject) => {
    const onMsg = (msg) => {
      if (msg.id !== id) return;
      worker.off("message", onMsg);
      if (msg.ok) resolve(msg);
      else reject(new Error(msg.error || "fail"));
    };
    worker.on("message", onMsg);
    worker.postMessage({ id, cmd, ...payload });
  });
}

const worker = new Worker(workerSrc, { eval: true, type: "module" });
worker.on("error", (e) => {
  console.error(e);
  process.exit(1);
});

const readyMsg = await call(worker, "init");
console.log("init", { workerSafe: readyMsg.workerSafe, sampleRate: readyMsg.sampleRate });
if (!readyMsg.workerSafe) {
  console.error("expected worker_safe()");
  process.exit(1);
}

const smoke = await call(worker, "smoke");
console.log(`smoke_score=${smoke.score.toFixed(4)}`);

const sess = await call(worker, "session_new", { n: 2, hopMs: 40, mode: "tern-all" });
console.log("session", sess);

const pcm = new Float32Array(16000);
for (let i = 0; i < pcm.length; i++) {
  pcm[i] = Math.sin((i / 16000) * 440 * Math.PI * 2) * 0.25;
}
const peaks = await call(worker, "push_peak", { pcm });
console.log("peaks", peaks.peaks);

await worker.terminate();
console.log("worker smoke ok");
