# rlx-tiny-tts

**TinyTTS** ([tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts)) — a MeloTTS / VITS2-style English text-to-speech model — running natively on RLX at 44.1 kHz. The four exported ONNX subgraphs (`text_encoder`, `duration_predictor`, `flow`, `decoder`) are imported into rlx-ir HIR and compiled per device; the monotonic-alignment + latent-sampling glue is reimplemented in Rust.

## Quick start

```bash
cargo run -p rlx-tiny-tts --release -- \
  --data weights/tiny-tts-rlx --text "The weather is nice today." --out out.wav
# [--device cpu|metal|mlx|cuda|rocm|gpu] [--speaker MALE] [--speed 1.0] [--seed 1234]
```

`--data` is an RLX TinyTTS bundle (see `scripts/export_tiny_tts.py`): `config.json`,
`onnx/{text_encoder,duration_predictor,flow,decoder}.onnx`, and a `frontend/` asset dir.
With no `--device`, the bin picks the best available accelerator (Metal → MLX → wgpu, else CPU).

## Public API

```rust
use rlx_tiny_tts::{TinyTts, InferOpts, Device};

let model = TinyTts::load_from_dir(std::path::Path::new("weights/tiny-tts-rlx"))?;
let opts = InferOpts::from_config(model.config());

// Full pipeline: raw text → 44.1 kHz mono waveform, every graph on `device`.
let wav = model.synthesize_on("Hello, world!", Device::Metal, &opts)?;
// or `model.synthesize(text, &opts)` for the CPU backend.
println!("{} samples @ {} Hz", wav.samples.len(), wav.sample_rate);
```

`Wav { samples, sample_rate }` is the output; `text_to_ids` exposes the raw
`(phone, tone, lang)` ids. The English text frontend (CMUdict + g2p_en + tagger + BERT)
is reused byte-identically from [rlx-inflect-nano](../rlx-inflect-nano)
(re-exported as `rlx_tiny_tts::frontend`).

## Backends

The ONNX graphs compile per device, so TinyTTS runs on every RLX backend:

| Feature | `--device` | Backend |
|---------|-----------|---------|
| `cpu` (default) | `cpu` | rlx-cpu (numeric reference) |
| `metal` | `metal` | Apple Metal |
| `mlx` | `mlx` | Apple MLX |
| `cuda` / `rocm` | `cuda` / `rocm` | NVIDIA / AMD |
| `gpu` / `vulkan` | `gpu` | wgpu (Metal / Vulkan / DX12) |
| `coreml` | `ane` | Apple CoreML (ANE / GPU / CPU) |

Convenience bundles: `apple-silicon`, `nvidia-gpu`, `amd-gpu`, `all-backends`.

## Examples

`keystone`, `flow_probe`, `tenc_probe`, `dec_probe`, `compile_all`, `debug_shapes` under
`examples/` exercise individual subgraphs (single-stage parity / shape debugging).
