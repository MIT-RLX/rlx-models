# Cross-backend model harness

Runs each model on **every backend the host supports** and checks the output is
actually correct — semantic check (whisper coverage / WER / token sanity) **and**
cross-backend parity (cosine vs the CPU baseline). One command, same on any host.

- On the Mac it exercises `cpu, metal, mlx, wgpu, coreml`.
- On `msi` (Linux + RTX 3080 Ti) it exercises `cpu, wgpu, cuda, vulkan`.

The host decides — nothing platform-specific lives in the registry. A model stanza
only describes what's intrinsic to the model; the driver intersects the host's
backends with the crate's declared cargo features to pick what to build and run.

## Bonsai Metal↔CUDA decode taps

```bash
# Mac Metal taps:
scripts/matrix/bonsai_trace_compare.sh metal

# MSI CUDA taps (sync + build + scp):
scripts/matrix/bonsai_trace_compare.sh cuda

# First mismatched fingerprint:
scripts/matrix/bonsai_trace_compare.sh diff \
  /tmp/bonsai_metal_tap.jsonl /tmp/bonsai_cuda_tap.jsonl
```

Env: `RLX_QWEN35_DECODE_TRACE`, `RLX_QWEN35_TAP`, `RLX_CUDA_PATH_TRACE`, `RLX_CUDA_DUMP_NODES`.

## Run it

```bash
# Full curated Tier-1 matrix on msi (sync -> build -> run -> pull report back):
just matrix-remote

# One model (fast iteration):
just matrix-remote ONLY=qwen3-0.6b

# Everything, including Tier-2 (NC-licensed / big-VRAM / needs extra assets):
just matrix-remote TIER=all ALL=1

# A subset of backends:
just matrix-remote BACKENDS=cpu,cuda

# Run on THIS machine instead of msi (auto-detects local backends):
just matrix            # or: python3 scripts/matrix/run_matrix.py
```

Env knobs (also settable directly when calling `run_matrix.py`):
`TIER=1|2|all`  `ONLY=<name>[,<name>]`  `BACKENDS=cpu,wgpu,cuda,vulkan`
`ALL=1` (include Tier-2 / NC / gated)  `BUILD_TIMEOUT=<secs>`  `HF_TOKEN=<...>` (gated repos).

## Output

Written to `scripts/matrix/out/` (pulled back to the Mac by `matrix-remote`):

- `report.md` — a model × backend grid (✅ pass / ⚠️ warn / ❌ fail / 🧱 build-fail /
  📦 no-weights / · skipped) with the key metric per cell, plus a "needs attention" list.
- `results.json` — full per-(model,backend) record (status, metrics, ms, adapter).
- `artifacts/` — the produced WAVs / transcripts / token dumps for spot-checking.
- `build-logs/` — one cargo log per crate+featureset.

## How it decides "proper results"

| kind | signal | pass | warn | fail |
|---|---|---|---|---|
| tts | whisper coverage of the known sentence (cpu) + **cosine(gpu wav, cpu wav)** | cov≥pass & cos≥pass | either in warn band | silent wav or cos<warn |
| asr | WER vs reference | ≤ pass | ≤ warn | > warn or empty |
| lm  | non-degenerate tokens + cpu-vs-gpu prefix parity | non-degen & exact prefix | prefix drift after ≥1 tok | degenerate |
| vision | produced non-empty output (cosine parity: TODO) | produced | — | nothing |

Cross-backend **cosine vs CPU is the authoritative pass**; whisper-tiny is weak so TTS
coverage only ever *warns*, never fails on its own.

## Adding a model

Add a `[[models]]` stanza to `registry.toml`. It's platform-neutral — you do **not**
list backends or per-platform features. Fields: `name, package, kind, tier, bin,
template` (or an explicit `run` string), `features_base` (non-backend features like
`hf-download`/`espeak`/`tokenizer`), `weights.path`, `download.mode` (`flag|manual|none`),
`extra` (fixed args like `--data <dir> --voice X`), `validate` thresholds, `license`.
Use `skip_backends = ["vulkan"]` only for a genuine runtime gap; compile-time gaps are
detected automatically from the crate's `[features]`.

## Files

- `registry.toml` — the model catalogue (source of truth).
- `run_matrix.py` — driver: host detect → build once → weights → run per backend → validate → report. Stdlib only.
- `sync_to_msi.sh` — rsync both working trees (`rlx-models` + sibling `../rlx`) to msi; excludes `weights/`, `target/`, `.cache/`.
- `msi_run.sh` — from-Mac wrapper: sync → ssh run → pull report.
