# rlx-tts-bench

Unified TTS bench CLI across RLX model crates: short/long phrases, Whisper coverage, spectral parity vs CPU, output noise metrics, optional voice-clone (+ noisy-reference) scenarios, and a self-contained HTML report.

Per-crate `backend_matrix` examples remain the fast single-model path; this harness reuses the same load/synth APIs.

## Library

```rust,ignore
use rlx_tts_bench::prelude::*;

let models = select_models("fake");
let devices = filter_available(&parse_device_list("cpu")?);
```

See [`prelude`](src/prelude.rs) for the full re-export surface (adapters, suite, metrics, report, WAV helpers).

## Build

```bash
# Default: matrix-onnx adapters (chatterbox, supertonic, piper, …)
cargo build -p rlx-tts-bench --release

# Everything + Apple backends
cargo build -p rlx-tts-bench --release --features "all-models,apple-silicon"
```

| Feature | Models |
|---------|--------|
| `matrix-onnx` (default) | chatterbox, moss-nano, styletts2, piper, supertonic, luxtts, f5tts, soprano |
| `matrix-ar` | sesame, gepard, zonos, metavoice |
| `lm-tts` | orpheus, qwen3-tts, kyutai, kittentts |
| `all-models` | union |
| `apple-silicon` / `all-backends` | forward Metal/MLX/GPU/CUDA to deps |

`fake` is always available (synthetic sine) for CLI preflight without weights.

## Commands

```bash
# What can run (weights + clone support)
just tts-bench-list
# or: cargo run -p rlx-tts-bench -- list

# Quick local check
just tts-bench-quick

# Full matrix → HTML + Markdown backend tables (per model×device workers; survives abort/OOM/hang)
just tts-bench-apple run --models all --devices auto --phrases short,long \
  --whisper --spectral --noise --clone \
  --out-dir /tmp/tts-bench --html report.html --md BACKENDS.md

# Continue after a kill / crash
just tts-bench-apple run --models all --devices auto --phrases short,long \
  --whisper --spectral --noise --clone --resume --out-dir /tmp/tts-bench

# Narrow
cargo run -p rlx-tts-bench --release --features apple-silicon -- \
  run -m chatterbox,supertonic,luxtts -d cpu,metal \
  --phrases short --iters 1 --out-dir /tmp/quick
```

### Flags

| Flag | Meaning |
|------|---------|
| `-m` / `--models` | Comma ids or `all` (missing weights → `skipped`) |
| `-d` / `--devices` | `auto` or `cpu,metal,mlx,gpu,cuda,ane` |
| `--phrases` | `short`, `long` (override text with `--text-short` / `--text-long`) |
| `--whisper` | Greedy Whisper + word coverage (peak-normalize first) |
| `--spectral` | STFT / log-mel cosine vs CPU (or clean clone) |
| `--noise` | Peak, RMS, crest, spectral flatness, crude SNR |
| `--clone` | Clean clone + noisy-ref clone when adapter supports it |
| `--clone-ref` | Reference WAV (default: `assets/jfk/jfk_voice_clone.wav`) |
| `--iters` / `--warmup` | Timing iters (default 1/0; honors `RLX_ITERS`) |
| `--out-dir` | WAV + JSON + HTML + Markdown |
| `--md` | Backend matrices markdown (default `BACKENDS.md`) |
| `--fail-under-fox N` | Optional exit gate on short/plain fox word hits |
| `--no-isolate` | Disable per `(model, device)` workers (default: isolate on) |
| `--resume` | Skip cells already in `results.jsonl` |
| `--timeout-secs N` | Kill hung worker after N seconds (default 2400) |

## Weight layout

Adapters look under `weights/tts/<name>` (or crate `DEFAULT_LOCAL_DIR`) and known env vars (`RLX_SUPERTONIC_DIR`, `ORPHEUS_GGUF_PATH`, `RLX_QWEN3_TTS_DIR`, …). See `list` for resolved paths.

Whisper: `RLX_WHISPER_DIR` or `.cache/whisper-tiny` / `whisper-base.en` (`just fetch-whisper`).

## Report fields

- `results.jsonl` — one object per `(model, device, phrase, scenario)`
- `summary.json` — counts + per-model median RTF / Whisper coverage
- `report.html` — tables + RTF heatmap (inline SVG, no CDN)
- `BACKENDS.md` — model × device matrices (RTF, wall ms, cosine vs CPU, Whisper)
- `wav/` — per-row PCM

Scenarios: `plain`, `clone`, `clone_noisy_ref`.

After a run, copy `BACKENDS.md` into the repo root [`TTS_BACKENDS.md`](../../TTS_BACKENDS.md) (or merge the matrices section) for release notes.

## Publish

`publish = false` — harness crate, not shipped to crates.io.
