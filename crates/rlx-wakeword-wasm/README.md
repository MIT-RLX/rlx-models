# rlx-wakeword-wasm

`wasm-bindgen` bindings for [`rlx-wakeword-core`](../rlx-wakeword-core) — **Node**, **browser**, and **Web Worker**.

Not published to crates.io (`publish = false`); built with `wasm-pack` for apps and demos.

- No DOM / `window` (timing via `Date.now()`)
- `worker_safe() == true` — safe on a dedicated module worker
- Streaming `WakeSession` + multi-phrase / ternary bench

## Node bench

```bash
just wakeword-wasm-bench
```

## Web / Worker

```bash
just wakeword-wasm-web                 # → web/pkg-web
just wakeword-wasm-worker-smoke        # Node worker_threads protocol check
just wakeword-wasm-worker-serve        # http://127.0.0.1:8765/
```

Open the page → **Init worker** → quick check / session / push / bench.

### Worker protocol

See [`web/wake_worker.js`](web/wake_worker.js).

```js
const worker = new Worker(new URL("./wake_worker.js", import.meta.url), {
  type: "module",
});
worker.postMessage({ id: 1, cmd: "init" });
worker.postMessage({
  id: 2,
  cmd: "session_new",
  n: 2,
  hopMs: 40,
  mode: "tern-all", // f32 | tern-fc | tern-all
});
const pcm = new Float32Array(16000);
worker.postMessage({ id: 3, cmd: "push_peak", pcm }, [pcm.buffer]);
```

| `cmd` | Payload | Reply |
|-------|---------|--------|
| `init` | — | `workerSafe`, `sampleRate` |
| `smoke` / quick check | — | `score` |
| `bench` | `hopMs`, `nMin`, `nMax`, `modes` | `table` string |
| `session_new` | `n`, `hopMs`, `mode` | session meta |
| `push` / `push_peak` | `pcm: Float32Array` | scores / peaks |
| `reset` | — | ok |

## JS API (after `init`)

| Export | Role |
|--------|------|
| `WakeSession` | `new(n, hopMs, mode)`, `push`, `push_peak`, `reset` |
| `smoke_score()` | Finite score on synth tone (preflight) |
| `bench_multi_phrase(...)` | Latency / size table string |
| `worker_safe()` / `sample_rate()` | Capability / 16 kHz |
