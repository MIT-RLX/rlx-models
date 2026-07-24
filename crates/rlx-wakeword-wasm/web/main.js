/** Main-thread UI for the wakeword module worker demo. */

const logEl = document.getElementById("log");
const btnInit = document.getElementById("btn-init");
const btnSmoke = document.getElementById("btn-smoke");
const btnSession = document.getElementById("btn-session");
const btnPush = document.getElementById("btn-push");
const btnBench = document.getElementById("btn-bench");

let worker = null;
let seq = 0;
const pending = new Map();

function log(msg, cls) {
  const line = typeof msg === "string" ? msg : JSON.stringify(msg, null, 2);
  logEl.textContent = line;
  logEl.className = cls || "";
}

function call(cmd, payload = {}, transfer = []) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, ...payload }, transfer);
  });
}

function tone(seconds, freqHz, amp, sampleRate) {
  const n = Math.floor(seconds * sampleRate);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.sin((i / sampleRate) * freqHz * Math.PI * 2) * amp;
  }
  return out;
}

function setReady(on) {
  for (const b of [btnSmoke, btnSession, btnPush, btnBench]) b.disabled = !on;
}

btnInit.onclick = () => {
  if (worker) worker.terminate();
  worker = new Worker(new URL("./wake_worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (ev) => {
    const msg = ev.data || {};
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.ok) p.resolve(msg);
    else p.reject(new Error(msg.error || "worker error"));
  };
  worker.onerror = (e) => log(`worker error: ${e.message}`, "err");

  call("init")
    .then((r) => {
      setReady(true);
      log(r, "ok");
    })
    .catch((e) => log(String(e), "err"));
};

btnSmoke.onclick = () => {
  call("smoke")
    .then((r) => log(`score=${r.score.toFixed(4)}`, "ok"))
    .catch((e) => log(String(e), "err"));
};

btnSession.onclick = () => {
  const n = Number(document.getElementById("n").value);
  const hopMs = Number(document.getElementById("hop").value);
  const mode = document.getElementById("mode").value;
  call("session_new", { n, hopMs, mode })
    .then((r) => log(r, "ok"))
    .catch((e) => log(String(e), "err"));
};

btnPush.onclick = async () => {
  try {
    const init = await call("init"); // noop-ish after first; ensures sampleRate
    const pcm = tone(1.0, 440, 0.25, init.sampleRate || 16000);
    const r = await call("push_peak", { pcm }, [pcm.buffer]);
    log({ peaks: [...r.peaks] }, "ok");
  } catch (e) {
    log(String(e), "err");
  }
};

btnBench.onclick = () => {
  log("bench running in worker…");
  call("bench", { hopMs: 40, nMin: 2, nMax: 4, modes: "f32,tern-all" })
    .then((r) => log(r.table, "ok"))
    .catch((e) => log(String(e), "err"));
};
