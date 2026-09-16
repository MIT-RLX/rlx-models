# rlx-moonshine

[Useful Sensors Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) English ASR for RLX.

- **Input:** raw 16 kHz mono PCM (Wav2Vec2-style conv frontend — **not** Whisper mel)
- **Arch:** encoder–decoder Transformer (`MoonshineForConditionalGeneration`) with RoPE (GPT-J interleaved, partial rotary)
- **Status:** full HIR enc–dec + safetensors load + greedy decode (Florence-style bucketed full-prefix decoder, host LM head)

## Quick start

```bash
# HuggingFace checkpoint dir with model*.safetensors + config.json + tokenizer.json
cargo run -p rlx-moonshine --release -- \
  --weights ~/.cache/huggingface/hub/.../moonshine-tiny \
  --wav audio16k.wav \
  --device cpu
```

```rust
use rlx_moonshine::{MoonshineConfig, MoonshineRunner};
use rlx_runtime::Device;

let mut runner = MoonshineRunner::builder()
    .weights("/path/to/moonshine-tiny")
    .device(Device::Cpu)
    .build()?;
let text = runner.transcribe(&pcm_f32_16k)?;
```

Or load explicitly:

```rust
use rlx_moonshine::{MoonshineConfig, MoonshineModel};

let cfg = MoonshineConfig::from_dir(dir).unwrap_or_else(|_| MoonshineConfig::tiny());
let mut model = MoonshineModel::load(dir, cfg, Device::Cpu)?;
let hidden = model.encode_pcm(&pcm)?;
let text = model.transcribe(&pcm)?;
```

## CLI

| Flag | Meaning |
|---|---|
| `--weights DIR\|FILE` | Safetensors dir or single file |
| `--wav PATH` | 16 kHz mono WAV (PCM16 or float32) |
| `--pcm PATH` | Raw little-endian `f32` samples @ 16 kHz |
| `--device` | `cpu` / `metal` / `mlx` / `cuda` / … |
| `--config` | Optional `config.json` override |

## Backend features

`metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, `coreml`, `all-backends`, `apple-silicon` — same pattern as `rlx-whisper`.
