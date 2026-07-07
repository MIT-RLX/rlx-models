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

let model = TinyTts::load("weights/tiny-tts-rlx")?;   // dir, .rlxpack file, or any AssetSource
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

### Versatile loading (`AssetSource`)

`TinyTts::load` accepts anything convertible to an [`AssetSource`], so the same
bundle loads from a directory, a single packed file, memory, or a config — with
byte-identical output (see `examples/load_sources.rs`):

```rust
use rlx_tiny_tts::{TinyTts, AssetSource, SourceSpec};

TinyTts::load("weights/tiny-tts-rlx")?;             // directory (auto-detected)
TinyTts::load("tiny-tts.rlxpack")?;                 // single packed file (auto-detected)
TinyTts::load(AssetSource::pack_file("m.rlxpack")?)?;
TinyTts::load(AssetSource::pack_bytes(bytes)?)?;    // in-memory pack (no disk)
TinyTts::load(AssetSource::memory(name_to_bytes))?; // in-memory asset map
TinyTts::load_from_spec(&spec)?;                    // {"source":"pack","path":"…"} from JSON
```

`AssetSource` (in `rlx-core`) also takes a custom `AssetProvider` (HTTP cache,
zip, embedded VFS…). Directory sources load in place; every other source is
materialized to a self-cleaning temp dir only when a sub-loader needs a real
path. Package a bundle into one distributable file with:

```bash
cargo run -p rlx-tiny-tts --release -- --pack weights/tiny-tts-rlx --out tiny-tts.rlxpack
# then: --data tiny-tts.rlxpack  works anywhere --data <dir> did
```

Adopt the same loaders in any model crate with one line —
`rlx_core::asset_source::load_materialized(src, Self::load_from_dir)`.

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
