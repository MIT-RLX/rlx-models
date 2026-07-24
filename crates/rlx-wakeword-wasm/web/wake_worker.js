/**
 * Dedicated module Worker for rlx-wakeword-wasm.
 *
 * Protocol (JSON-ish messages; PCM as transferable Float32Array):
 *
 *   → { id, cmd: "init" }
 *   ← { id, ok: true, kind: "ready", workerSafe, sampleRate }
 *
 *   → { id, cmd: "smoke" }
 *   ← { id, ok: true, score }
 *
 *   → { id, cmd: "bench", hopMs?, nMin?, nMax?, modes? }
 *   ← { id, ok: true, table }
 *
 *   → { id, cmd: "session_new", n?, hopMs?, mode? }
 *   ← { id, ok: true, phraseCount, hopSamples, mode }
 *
 *   → { id, cmd: "push", pcm: Float32Array }   // may transfer pcm.buffer
 *   ← { id, ok: true, scores: Float32Array, hops }
 *
 *   → { id, cmd: "push_peak", pcm: Float32Array }
 *   ← { id, ok: true, peaks: Float32Array }
 *
 *   → { id, cmd: "reset" }
 *   ← { id, ok: true }
 */

import init, {
  WakeSession,
  bench_multi_phrase,
  sample_rate,
  smoke_score,
  worker_safe,
} from "./pkg-web/rlx_wakeword_wasm.js";

let session = null;
let ready = null;

async function ensureInit() {
  if (!ready) {
    const wasmUrl = new URL("./pkg-web/rlx_wakeword_wasm_bg.wasm", import.meta.url);
    ready = init({ module_or_path: wasmUrl });
  }
  await ready;
}

function reply(id, payload, transfer = []) {
  self.postMessage({ id, ...payload }, transfer);
}

function fail(id, err) {
  reply(id, { ok: false, error: String(err && err.message ? err.message : err) });
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  const id = msg.id;
  try {
    await ensureInit();
    switch (msg.cmd) {
      case "init":
        reply(id, {
          ok: true,
          kind: "ready",
          workerSafe: worker_safe(),
          sampleRate: sample_rate(),
        });
        break;
      case "smoke":
        reply(id, { ok: true, score: smoke_score() });
        break;
      case "bench": {
        const table = bench_multi_phrase(
          msg.hopMs ?? 40,
          msg.nMin ?? 2,
          msg.nMax ?? 10,
          msg.modes ?? "f32,tern-fc,tern-all",
        );
        reply(id, { ok: true, table });
        break;
      }
      case "session_new": {
        if (session) {
          session.free();
          session = null;
        }
        session = new WakeSession(msg.n ?? 1, msg.hopMs ?? 40, msg.mode ?? "tern-all");
        reply(id, {
          ok: true,
          phraseCount: session.phrase_count,
          hopSamples: session.hop_samples,
          mode: session.mode,
        });
        break;
      }
      case "push": {
        if (!session) throw new Error("no session; send session_new first");
        const pcm = msg.pcm;
        if (!(pcm instanceof Float32Array)) throw new Error("pcm must be Float32Array");
        const scores = session.push(pcm);
        const n = session.phrase_count;
        const hops = n ? scores.length / n : 0;
        reply(id, { ok: true, scores, hops }, [scores.buffer]);
        break;
      }
      case "push_peak": {
        if (!session) throw new Error("no session; send session_new first");
        const pcm = msg.pcm;
        if (!(pcm instanceof Float32Array)) throw new Error("pcm must be Float32Array");
        const peaks = session.push_peak(pcm);
        reply(id, { ok: true, peaks }, [peaks.buffer]);
        break;
      }
      case "reset":
        if (session) session.reset();
        reply(id, { ok: true });
        break;
      default:
        throw new Error(`unknown cmd: ${msg.cmd}`);
    }
  } catch (err) {
    fail(id, err);
  }
};
