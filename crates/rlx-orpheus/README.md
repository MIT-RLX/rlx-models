# rlx-orpheus

Orpheus TTS on RLX — [Canopy Orpheus](https://github.com/canopyai/Orpheus-TTS) speech LM (Llama-3B GGUF) + [SNAC 24 kHz](https://huggingface.co/hubertsiuzdak/snac_24khz) neural codec decoder.

## Quick start

```bash
just fetch-orpheus fetch-orpheus-snac
just export-orpheus-snac
export ORPHEUS_SNAC_PATH=/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors

just orpheus-demo                    # Metal LM + eager SNAC → /tmp/orpheus-demo.wav
just orpheus-coreml-demo             # Metal LM + native RLX Ane SNAC (macOS, --features coreml)
```

## Architecture

| Stage | Default (`--device metal`) | `--device coreml` |
|-------|---------------------------|-------------------|
| Llama-3B LM | Metal GGUF prefill + KV decode | Metal GGUF (same) |
| SNAC quantizer | CPU eager (safetensors) | CPU eager |
| SNAC conv decoder | CPU eager (ndarray) | Native RLX [`Device::Ane`](https://github.com/eugenehp/rlx) via `rlx-ir` |

CoreML is **native RLX** (no ONNX). The RVQ lookup stays on the host; only the conv decoder stack is compiled per latent-length bucket and cached (`ORPHEUS_SNAC_AOT`).

### Cargo features

| Feature | Enables |
|---------|---------|
| `llama` (default) | GGUF backbone + CLI |
| `codec` (default) | SNAC decoder (eager) |
| `coreml` | SNAC on `Device::Ane` (macOS; pulls `rlx-ir`, `rlx-runtime/coreml`) |
| `metal` / `mlx` / `cuda` / … | Forwarded to `rlx-llama32` for LM backends |
| `apple-silicon` | `metal` + `mlx` + `gpu` + `coreml` |
| `hf-download` | `--download` flags via `hf-hub` |

Build examples:

```bash
cargo build -p rlx-orpheus --release --features apple-silicon   # Metal LM + CoreML SNAC
cargo build -p rlx-orpheus --release --features all-backends    # every LM backend + coreml
```

## Audio samples

Bundled MP4 clips (mono AAC, 24 kHz) under [`assets/`](assets/):

| File | Description |
|------|-------------|
| [`sample_hi_eager.mp4`](assets/sample_hi_eager.mp4) | Golden SNAC codes → host eager decoder (~0.34 s) |
| [`sample_hi_coreml.mp4`](assets/sample_hi_coreml.mp4) | Same codes → native RLX CoreML SNAC |
| [`sample_hello_rlx.mp4`](assets/sample_hello_rlx.mp4) | “Hello from RLX.” |
| [`sample_longer.mp4`](assets/sample_longer.mp4) | “Hello from RLX. This is a longer test.” |

Regenerate:

```bash
just export-orpheus-audio-assets
# optional: refresh sentence MP4s from new synthesis WAVs
export ORPHEUS_SAMPLE_LONG_WAV=/tmp/orpheus-long.wav
export ORPHEUS_SAMPLE_LONGER_WAV=/tmp/orpheus-longer.wav
just export-orpheus-audio-assets
```

Requires `ffmpeg` on PATH. Hi clips only need exported SNAC weights; sentence clips need WAVs from `just orpheus-demo` (or any synthesis run).

## Weights

| Component | Source |
|-----------|--------|
| LM backbone | [unsloth/orpheus-3b-0.1-ft-GGUF](https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF) (`Q4_K_M` recommended) |
| SNAC decoder | [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) → [`scripts/export_snac_decoder.py`](../../scripts/export_snac_decoder.py) |

```bash
just fetch-orpheus              # GGUF → /tmp/rlx-weights/orpheus/
just fetch-orpheus-snac         # PyTorch checkpoint → /tmp/rlx-weights/snac/
just export-orpheus-snac        # safetensors + config JSON (+ parity fixtures)
just export-orpheus-snac -- --parity-layers   # also ref_stage_*.npy for layer tests
export ORPHEUS_SNAC_PATH=/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors
```

CLI download (`hf-download` feature): `--download`, `--download-orpheus [--quant Q4_K_M]`, `--download-snac`.

Manual SNAC export:

```bash
python3 -m venv .venv-orpheus && source .venv-orpheus/bin/activate
pip install snac torch safetensors
python3 scripts/export_snac_decoder.py --out /tmp/rlx-weights/snac
```

## CLI

```bash
just orpheus -- \
  --weights /tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf \
  --text "Hello from RLX Orpheus." \
  --voice tara \
  --device metal \
  --out /tmp/orpheus.wav
```

| `--device` | LM | SNAC |
|------------|-----|------|
| `auto` | Best GPU (Metal on Apple Silicon, else CUDA/ROCm/wgpu/Vulkan) | Eager CPU |
| `metal` / `cuda` / `rocm` / `gpu` / `wgpu` / `vulkan` | Selected RLX backend (KV decode on GPU) | Eager CPU |
| `coreml` | Metal (or best Apple GPU) | Native RLX `Device::Ane` |

CoreML needs `--features coreml` at build time:

```bash
cargo run -p rlx-orpheus --release --features "llama,coreml,metal" -- \
  --weights /tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf \
  --device coreml \
  --text "Hello from RLX on CoreML." \
  --voice tara \
  --out /tmp/orpheus-coreml.wav
```

Built-in voices: `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`.

Sampling defaults: temperature 0.6, top-p 0.8, repetition penalty 1.3, stop token 49158. Override via `--temperature`, `--top-p`, `--greedy`, etc.

### Voice cloning (pretrained checkpoint)

Finetune-prod GGUF (`just fetch-orpheus`) supports **named voices only**. Zero-shot clone needs a **pretrained** GGUF plus a reference JSON from SNAC encoding.

```bash
just orpheus-encode-ref /path/ref.wav "exact spoken transcript"
just orpheus-voice-clone -- \
  --weights /path/to/orpheus-3b-0.1-pretrained.Q4_K_M.gguf \
  --target-text "Hello from a cloned voice." \
  --out /tmp/orpheus_clone.wav
```

Convert [`canopylabs/orpheus-3b-0.1-pretrained`](https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained) to GGUF before use. Emotive tags: `<laugh>`, `<chuckle>`, `<sigh>`, … (upstream Orpheus docs).

## Library

```rust
use rlx_orpheus::{
    BackboneLoadOptions, GenerationConfig, OrpheusTts, resolve_orpheus_device,
};

let runtime = resolve_orpheus_device("metal")?;
let backbone_opts = BackboneLoadOptions::for_device(runtime.lm);
let mut tts = OrpheusTts::load_on_with(
    "/path/to/orpheus.gguf".as_ref(),
    "/path/to/snac_24khz_decoder.safetensors".as_ref(),
    runtime,
    backbone_opts,
)?;
tts.config = GenerationConfig { max_new_tokens: 1200 };
let out = tts.synthesize("Hello world.", Some("tara"))?;
// out.samples — mono f32 PCM at 24 kHz
```

SNAC-only decode (no LM):

```rust
use rlx_orpheus::{SnacBackend, SnacLoadOptions, decode_orpheus_codes};

let snac = SnacBackend::open(path, SnacLoadOptions { coreml: false })?;
let pcm = decode_orpheus_codes(&snac, &frame_codes)?;

// CoreML SNAC path (feature `coreml`, macOS):
let snac = SnacBackend::open(path, SnacLoadOptions { coreml: true })?;
```

Voice clone API: [`VoiceCloneReference::load_json`] + [`OrpheusTts::synthesize_voice_clone`].

Streaming: [`OrpheusTts::synthesize_stream`] — ~2048-sample PCM chunks after 4 SNAC frames; tail appended from full decode.

## Just recipes

| Recipe | Purpose |
|--------|---------|
| `just fetch-orpheus` | Download finetune GGUF (`Q4_K_M`) |
| `just fetch-orpheus-snac` | Download SNAC PyTorch weights |
| `just export-orpheus-snac` | Export safetensors from SNAC checkpoint |
| `just export-orpheus-audio-assets` | Regenerate bundled MP4 samples |
| `just orpheus -- …` | Run CLI (pass flags after `--`) |
| `just orpheus-demo` | Short Metal synthesis demo |
| `just orpheus-wgpu-demo` | wgpu LM synthesis (`--features gpu`) |
| `just orpheus-coreml-demo` | CoreML SNAC demo |
| `just orpheus-demos` | Batch demo WAVs |
| `just orpheus-encode-ref WAV TEXT` | Reference clip → clone JSON |
| `just orpheus-voice-clone` | Voice-clone example |
| `just test-orpheus-whisper` | SNAC golden → Whisper (fast) |
| `just test-orpheus-whisper-e2e` | Full LM → Whisper (`#[ignore]`) |
| `just test-orpheus-backends-whisper` | Per-backend + Whisper |
| `ORPHEUS_CODES_PARITY=1 cargo test -p rlx-orpheus --test backends_codes_parity --features all-backends --release` | Greedy SNAC codes vs CPU on every backend |
| `ORPHEUS_BACKENDS_BENCH=1 just test-orpheus-backends-whisper` | Whisper on all available accelerators |
| `just bench-orpheus` / `bench-orpheus-all-devices` | Timing + optional Whisper |

## Tests

```bash
# SNAC decoder vs Python reference (needs ref_noise_*.npy under ORPHEUS_SNAC_REF_DIR)
ORPHEUS_SNAC_PATH=/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors \
ORPHEUS_SNAC_REF_DIR=/tmp/rlx-weights/snac \
cargo test -p rlx-orpheus --features codec --lib decode_matches_reference_when_weights_available --release

# Native CoreML SNAC vs eager (macOS, feature coreml)
ORPHEUS_SNAC_PATH=/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors \
ORPHEUS_SNAC_REF_DIR=/tmp/rlx-weights/snac \
cargo test -p rlx-orpheus --features coreml --test coreml_snac_parity --release

just fetch-orpheus-snac fetch-whisper
just test-orpheus-whisper

# Per-backend LM + Whisper (includes wgpu when built with --features gpu)
just test-orpheus-backends-whisper named_voice_whisper_wgpu
```

## Environment

| Variable | Role |
|----------|------|
| `ORPHEUS_SNAC_PATH` | SNAC `snac_24khz_decoder.safetensors` (required) |
| `ORPHEUS_GGUF_PATH` | Default GGUF for tests / examples |
| `ORPHEUS_SNAC_REF_DIR` | Python parity fixtures (`ref_codes.json`, `ref_noise_*.npy`) |
| `ORPHEUS_SNAC_AOT` | AOT compile cache dir for CoreML SNAC graphs |
| `ORPHEUS_SNAC_COREML=1` | `decode_fixture` example: Ane SNAC path |
| `ORPHEUS_LOW_MEM=1` | CPU F32 LM prefill + decode (lower RAM) |
| `ORPHEUS_METAL_PREFILL` | `metal` \| `packed` \| `cpu` — Metal GGUF prefill mode |
| `ORPHEUS_COMPILE_SEQ_CAP` | Compile bucket cap (default tuned for synthesis) |
| `ORPHEUS_COMPILE_CACHE` | Prefill compile cache for bucket decode (`0` disables; default on when RAM allows) |
| `ORPHEUS_BUCKET_DECODE=1` | Faster multi-step LM decode when RAM allows |
| `ORPHEUS_MLX_KV=1` | Opt-in MLX LM KV decode |
| `ORPHEUS_MASK_LOGITS` | Restrict LM logits to SNAC slot (`1` force on, `0` off; default on for wgpu/Vulkan **native GPU decode** only) |
| `ORPHEUS_WHISPER_E2E=1` | Enable full LM e2e Whisper test |

## Notes

- **Memory:** default [`BackboneLoadOptions::for_device`] uses GPU prefill on Metal / CUDA / ROCm without eager F32 drain. wgpu/Vulkan use CPU GGUF parity. Avoid `ORPHEUS_PACKED_LM=1` (O(n²) recompute). `ORPHEUS_HIGH_MEM=1` raises compile cap to 256.
- **wgpu / Vulkan:** `--device gpu` selects the portable GPU backend; the 3B GGUF LM runs **CPU prefill + CPU decode** (GGUF parity, same sampling as Metal). SNAC stays CPU eager. Native wgpu LM decode is pending upstream KV parity (`ORPHEUS_MASK_LOGITS=1` only needed for that path).
- **MLX:** not selected by `--device auto`; set `ORPHEUS_MLX_KV=1` to enable KV decode on MLX.
- **Batch decode:** [`decode_orpheus_codes`] returns the full SNAC waveform; streaming uses 4-frame windows (2048-sample centers) and appends the remaining tail automatically.
- **Multi-utterance:** reuse one [`OrpheusTts`] session and call [`OrpheusTts::synthesize_batch`] (same backbone load as [`examples/batch_demos.rs`]).
- **Parity:** SNAC `z_q` + PCM vs Python within ~1e-3 when noise fixtures exist; per-decoder-stage parity with `export_snac_decoder.py --parity-layers`; CoreML SNAC vs eager within ~0.08 max abs diff (fp16 on ANE). LM: Metal `for_tts` greedy tokens match CPU synthesis reference (`metal_prefill_parity` tests).
- **Serving / throughput:** Orpheus is autoregressive (one SNAC slot token per LM step) — unlike chat LLMs there is no prefill/decode asymmetry to exploit for continuous batching today. [`rlx-serve`](../../crates/rlx-serve) continuous + fused batching targets [`Qwen3Generator::forward_batch_decode`](../../crates/rlx-qwen3/src/generator.rs); **`Llama32Generator` has no batched decode yet**, so multi-request TTS throughput is **one utterance per backbone lock** until batched Llama decode lands. Reuse [`OrpheusTts::synthesize_batch`] or a job queue for now.
- **Tensor parallel:** [`rlx-distributed`](../../crates/rlx-distributed) implements in-graph TP (Megatron MLP + head-sharded attention) for **Qwen3** blocks on the research path — **not wired for Llama-3B / Orpheus**. A 3B Orpheus model fits one Apple Silicon or consumer GPU; TP would help multi-node serving once Llama blocks expose the same collectives (see [distributed-inference-design.md](../../docs/distributed-inference-design.md)).
