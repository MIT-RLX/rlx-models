# rlx-tiny-tts

**TinyTTS** ([tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts)) — a MeloTTS / VITS2-style English text-to-speech model — running natively on RLX at 44.1 kHz.

**Distribution:** single [`tiny-tts.rlxp`](https://huggingface.co/eugenehp/tiny-tts-rlx) with nested
`graphs/*.rlxp` (hot weight tensors + `graph.json`). Runtime materializes the outer pack and lowers
each subgraph to rlx-ir HIR per utterance length. Monotonic-alignment + latent-sampling glue is in Rust.
Hub ships **no** `.onnx`.

Same bundle powers [`rlx-melotts`](../rlx-melotts) (`weights/tts/melotts` → this tree).

## Quick start

```bash
just fetch-tiny-tts          # eugenehp/tiny-tts-rlx → tiny-tts.rlxp
just tiny-tts-backends       # CPU / Metal / MLX cosine vs CPU
just export-tiny-tts-rlxp    # pack from local sources → nested graphs

cargo run -p rlx-tiny-tts --release --features apple-silicon -- \
  --data weights/tts/tiny-tts-rlx --text "The weather is nice today." \
  --device metal --out out.wav
# or: --data weights/tts/tiny-tts-rlx/tiny-tts.rlxp
```

`--data` is an RLX TinyTTS bundle: `config.json`, nested `graphs/{text_encoder,duration_predictor,flow,decoder}.rlxp`,
and a `frontend/` asset dir (or a packed `.rlxp`). Legacy local `onnx/*.onnx` still packs/loads for rebuilds.
With no `--device`, the bin picks the best available accelerator (Metal → MLX → wgpu, else CPU).

## Public API

```rust
use rlx_tiny_tts::{TinyTts, InferOpts, Device};

let model = TinyTts::load("weights/tts/tiny-tts-rlx")?;   // dir, .rlxp file, or any AssetSource
let opts = InferOpts::from_config(model.config());

// Full pipeline: raw text → 44.1 kHz mono waveform, every graph on `device`.
let wav = model.synthesize_on("Hello, world!", Device::Mlx, &opts)?;
// or `model.synthesize(text, &opts)` for the CPU backend.
println!("{} samples @ {} Hz", wav.samples.len(), wav.sample_rate);
```

`Wav { samples, sample_rate }` is the output; `text_to_ids` exposes the raw
`(phone, tone, lang)` ids. The English text frontend (CMUdict + g2p_en + tagger + BERT)
is reused byte-identically from [rlx-inflect-nano](../rlx-inflect-nano)
(re-exported as `rlx_tiny_tts::frontend`).

### Kernel variants (precision vs throughput)

`InferOpts.kernel` selects the backend kernel-variant policy, mirroring the
per-op kernel selection in `../rlx` (Metal `SgemmVariant`, CUDA TF32, CPU conv)
but as one option instead of raw `RLX_*` env vars:

```rust
use rlx_tiny_tts::KernelVariant;
let mut opts = InferOpts::from_config(model.config());
opts.kernel = KernelVariant::Precise;   // parity/precision kernels
```

| Variant | Metal | CPU | CUDA | Use |
|---|---|---|---|---|
| `Fast` (default) | cost-model SIMD matmul (e.g. `simd4x4`) | fast im2col conv | TF32 allowed | production |
| `Precise` | scalar fp32 `naive` (`RLX_METAL_PRECISE`) | exact conv | TF32 off (`RLX_CUDA_PARITY`) | bit-exact parity vs onnxruntime |
| `Inherit` | — | — | — | honor your own `RLX_*` env (e.g. a specific `RLX_METAL_SGEMM_VARIANT=mps\|tiled\|…`) |

Applied via `rlx_ir::env` **code overrides** (precedence over process env, read
by the backends at dispatch), so the same compiled graph runs fast or precise
kernels without recompiling. CLI: `--kernel fast|precise|inherit`; env:
`RLX_TTS_KERNEL`. The override is process-global (last-writer-wins across
concurrent models) — set one policy per process.

### Versatile loading (`AssetSource`)

`TinyTts::load` accepts anything convertible to an [`AssetSource`], so the same
bundle loads from a directory, a single packed file, memory, or a config — with
byte-identical output (see `examples/load_sources.rs`):

```rust
use rlx_tiny_tts::{TinyTts, AssetSource, SourceSpec};

TinyTts::load("weights/tts/tiny-tts-rlx")?;             // directory (auto-detected)
TinyTts::load("tiny-tts.rlxp")?;                        // single packed file (auto-detected)
TinyTts::load(AssetSource::pack_file("m.rlxp")?)?;
TinyTts::load(AssetSource::pack_bytes(bytes)?)?;        // in-memory pack (no disk)
```

Pack / in-memory sources materialize to a self-cleaning temp dir only when a
sub-loader needs a real path. Package a bundle into one distributable file with:

```bash
just export-tiny-tts-rlxp
# then: --data weights/tts/tiny-tts-rlx/tiny-tts.rlxp
```

Adopt the same loaders in any model crate with one line —
`rlx_core::asset_source::load_materialized(src, Self::load_from_dir)`.

## Backends

Subgraphs compile per device, so TinyTTS runs on every RLX backend:

| Feature | `--device` | Backend |
|---------|-----------|---------|
| `cpu` (default) | `cpu` | rlx-cpu (numeric reference) |
| `metal` | `metal` | Apple Metal |
| `mlx` | `mlx` | Apple MLX |
| `cuda` / `rocm` | `cuda` / `rocm` | NVIDIA / AMD |
| `gpu` / `vulkan` | `gpu` | wgpu (Metal / Vulkan / DX12) |
| `coreml` | `ane` / `coreml` | Apple CoreML — auto `RLX_COREML_UNITS=gpu` (Neural-Engine BNNS crashes on many TTS graphs) |

Convenience bundles: `apple-silicon`, `nvidia-gpu`, `amd-gpu`, `all-backends`.

## Examples

`keystone`, `flow_probe`, `tenc_probe`, `dec_probe`, `compile_all`, `debug_shapes`,
`pack_rlxp` under `examples/` exercise packing and individual subgraphs.
