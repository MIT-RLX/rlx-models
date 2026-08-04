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
| `rlx-tts` | RLX TTS Gryphon native RLX (`.cache/rlx-tts-rlx`) |
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

## RLX TTS stress (≥1000 synthetic prompts)

Deterministic combinatorial English corpus → synthesize with RLX TTS → validate with Whisper
coverage/CER. Optionally synthesize the same lines with another RLX TTS (`--ref-model piper`)
for spectral cosine.

```bash
# Preflight (20 phrases)
just tts-rlx-tts-stress -- --n 20 --out-dir /tmp/rlx-tts-stress-pre

# Full 1000 + Whisper (resume-safe; WaveRNN is slow — expect a long run)
just tts-rlx-tts-stress -- --n 1000 --resume --out-dir /tmp/rlx-tts-stress

# With Piper reference audio + spectral
just tts-rlx-tts-stress -- --n 1000 --ref-model piper --spectral --resume

# Custom corpus file (plain lines or JSONL {"text":"..."})
cargo run -p rlx-tts-bench --release --features rlx-tts,matrix-onnx -- \
  stress --target rlx-tts --corpus-file prompts.txt --n 500 --whisper
```

Artifacts under `--out-dir`: `corpus.jsonl`, `stress_results.jsonl`, `stress_summary.json`, optional `wav/`.

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

After a run, fold `BACKENDS.md` into the Cross-backend parity & benchmarks section of the repo-root [`TTS.md`](../../TTS.md) for release notes.

## Publish

`publish = false` — harness crate, not shipped to crates.io.
