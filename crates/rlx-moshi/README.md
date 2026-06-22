# rlx-moshi

Native Rust inference for [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — a real-time speech-to-speech foundation model. Uses [`rlx-mimi`](../rlx-mimi) for the Mimi codec and either eager CPU or Candle GPU (Metal/CUDA) for the 7B temporal + DepFormer stack.

## Architecture

| Component | Role |
|-----------|------|
| **Mimi** (`rlx-mimi`) | 12.5 Hz audio codec — encode user mic / decode Moshi speech |
| **Temporal transformer** | Helium-style 7B decoder over time (text + audio token embeddings) |
| **Depth transformer (DepFormer)** | Per-frame codebook autoregression (8 RVQ streams) |

### Variants

| CLI `--variant` | Voice | Mode | Codebooks |
|-----------------|-------|------|-----------|
| `moshiko-one-way` (default) | Moshiko (male) | One-way TTS | 8 generated |
| `moshiko` / `--duplex` | Moshiko | Full-duplex | 8 + 8 user |
| `moshika-one-way` | Moshika (female) | One-way TTS | 8 generated |
| `moshika` | Moshika | Full-duplex | 8 + 8 user |

Moshiko and Moshika share the same 7B architecture. [`MoshiVariant`](src/config.rs) picks the runtime codebook layout; [`MoshiCheckpoint`](src/checkpoint.rs) picks the quant preset; together they route to the correct `kyutai/moshiko-*` or `kyutai/moshika-*` HuggingFace repo.

## Quick start

```bash
# Mimi codec (~385 MB) + Moshi LM (~14 GB bf16)
just fetch-mimi
just fetch-moshi

# Q8 GGUF (~8 GB) — smaller on disk, CPU eager or GPU Candle
just fetch-moshi-q8
export RLX_MOSHI_CHECKPOINT=q8
export RLX_MOSHI_DIR=.cache/moshiko-q8

# One-way generation (CPU eager — slow; use Metal for production)
just moshi -- --prompt "Hello, I'm Moshi." --out-wav /tmp/moshi.wav --device cpu

# GPU path (~70 ms/frame on Apple Silicon with Q4 MLX or bf16)
just fetch-moshi-q4
just moshi -- --checkpoint q4 --device metal --prompt "Hello." --out-wav /tmp/moshi-metal.wav --max-steps 8

# Full-duplex: user WAV in → Moshi WAV out
just moshi -- --duplex --in-wav speech.wav --out-wav /tmp/moshi-reply.wav --max-steps 50 --device metal

# Moshika (female voice)
just fetch-moshika
just moshi -- --variant moshika-one-way --prompt "Hi." --out-wav /tmp/moshika.wav --device metal

# WebSocket duplex server (tokio + axum)
just moshi-ws -- --host 127.0.0.1 --port 8998 --device metal
```

## Checkpoints

There is no smaller Moshi **architecture** — only quantization variants:

| Preset | Env / CLI | HuggingFace (Moshiko) | Size | Backends |
|--------|-----------|----------------------|------|----------|
| `bf16` (default) | `RLX_MOSHI_CHECKPOINT=bf16` | [`kyutai/moshiko-candle-bf16`](https://huggingface.co/kyutai/moshiko-candle-bf16) | ~14 GB | CPU eager, Candle GPU |
| `q8` | `--checkpoint q8` | [`kyutai/moshiko-candle-q8`](https://huggingface.co/kyutai/moshiko-candle-q8) | ~8 GB | CPU eager (GGUF), Candle GPU |
| `q4` | `--checkpoint q4` | [`kyutai/moshiko-mlx-q4`](https://huggingface.co/kyutai/moshiko-mlx-q4) | ~4 GB | MLX Q4 → dequant → Candle / CPU |
| `q8-mlx` | `--checkpoint q8-mlx` | [`kyutai/moshiko-mlx-q8`](https://huggingface.co/kyutai/moshiko-mlx-q8) | ~7 GB | MLX Q8 → dequant → Candle / CPU |
| `mlx-bf16` | `--checkpoint mlx-bf16` | [`kyutai/moshiko-mlx-bf16`](https://huggingface.co/kyutai/moshiko-mlx-bf16) | ~14 GB | MLX key layout, BF16 tensors → Candle / CPU |

Pass `--variant moshika` or `--variant moshika-one-way` to use **Moshika** repos (`kyutai/moshika-*`) with the same presets.

### Default cache directories

Unless `RLX_MOSHI_DIR` is set, the CLI picks a voice + preset scoped path:

| Voice | Preset | Cache dir |
|-------|--------|-----------|
| Moshiko | bf16 | `.cache/moshiko` |
| Moshiko | q8 | `.cache/moshiko-q8` |
| Moshiko | q4 | `.cache/moshiko-mlx-q4` |
| Moshika | bf16 | `.cache/moshika` |
| Moshika | mlx-bf16 | `.cache/moshika-mlx-bf16` |

Fetch recipes:

```bash
just fetch-moshi              # Moshiko bf16
just fetch-moshi-q8           # Moshiko Q8 GGUF
just fetch-moshi-q4           # Moshiko MLX Q4
just fetch-moshi-mlx-bf16     # Moshiko MLX bf16
just fetch-moshika            # Moshika bf16
just fetch-moshika-q4         # Moshika MLX Q4
```

(`--device mlx` uses Apple GPU via Candle Metal; native rlx-mlx quantized matmul is future work.)

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `RLX_MOSHI_DIR` | voice/checkpoint cache | LM weights + tokenizer directory |
| `RLX_MOSHI_CHECKPOINT` | `bf16` | Preset: `bf16`, `q8`, `q4`, `q8-mlx`, `mlx-bf16` |
| `RLX_MOSHI_VOICE` | `moshiko` | Voice when variant unset: `moshiko`, `moshika` |
| `RLX_MIMI_DIR` | `.cache/mimi` | Mimi codec weights (CPU HF layout) |

## Backends

```bash
cargo build -p rlx-moshi --features "gpu-lm,compiled-lm,mlx-lm,metal"
cargo build -p rlx-mimi --features "gpu-codec,metal"
```

| Feature | Enables |
|---------|---------|
| `gpu-lm` | Kyutai `moshi` Candle backend (Metal/CUDA) |
| `mlx-lm` | MLX Q4/Q8/bf16 safetensors load (+ dequant when quantized) into Candle |
| `compiled-lm` | GPU path alias (native RLX Helium graph TBD) |
| `metal` / `cuda` / … | `rlx-runtime` device tags + Candle kernels |
| `hf-download` | `--fetch` and auto-download missing weights |
| `tokio` | `spawn_duplex_tokio` |
| `ws-server` | `examples/ws_server` (axum WebSocket) |

| Device | LM stack |
|--------|----------|
| `cpu` | Eager ndarray (bf16, Q8 GGUF dequant, MLX layout) |
| `metal` / `cuda` / `mlx` | Candle `moshi` (MLX checkpoints remapped; Q4/Q8 dequantized at load) |
| `wgpu` / `vulkan` | CPU fallback today (RLX compiled path TBD) |

## Mimi codec GPU

Session passes `--device` to both LM and Mimi. GPU Mimi uses Kyutai Candle weights from
`tokenizer-e351c8d8-checkpoint125.safetensors` (bundled in `kyutai/moshiko-*` / `kyutai/moshika-*` repos, or set
`RLX_MOSHI_DIR`). HF `kyutai/mimi` layout stays on CPU eager.

```bash
cargo build -p rlx-mimi --features "gpu-codec,metal"
RLX_MIMI_GPU_SMOKE=1 RLX_MOSHI_DIR=.cache/moshiko cargo test -p rlx-mimi --test gpu_codec_smoke --features "gpu-codec,metal"
```

## Streaming (channels + tokio)

Duplex streaming feeds **1920-sample** (80 ms @ 24 kHz) PCM chunks and receives Moshi output incrementally:

```rust
use rlx_moshi::{
    GenerationConfig, MoshiSession, MoshiVariant, StreamCommand, StreamEvent,
    spawn_duplex_stream,
};
use rlx_runtime::Device;

let session = MoshiSession::open_on(".cache/moshiko", ".cache/mimi", MoshiVariant::Moshiko, Device::Metal)?;
let handle = spawn_duplex_stream(session, "", GenerationConfig::default())?;

// worker thread emits StreamEvent::OutputPcm { samples, .. }
handle.cmd_tx.send(StreamCommand::Pcm(chunk))?;
handle.cmd_tx.send(StreamCommand::Finish)?;
```

## Library

```rust
use rlx_moshi::{
    MoshiCheckpoint, MoshiSession, MoshiVariant, MoshiVoice, GenerationConfig,
    default_moshi_dir_for,
};
use rlx_runtime::Device;

let variant = MoshiVariant::MoshikaOneWay;
let checkpoint = MoshiCheckpoint::Q4MlxSafetensors;
let moshi_dir = default_moshi_dir_for(variant, checkpoint); // .cache/moshika-mlx-q4

let mut session = MoshiSession::open_with_checkpoint(
    &moshi_dir,
    ".cache/mimi",
    variant,
    Device::Metal,
    checkpoint,
)?;
let result = session.generate_one_way("Hello!", &GenerationConfig::default())?;
```

## Tests

```bash
export RLX_MOSHI_DIR=.cache/moshiko
export RLX_MIMI_DIR=.cache/mimi
cargo test -p rlx-moshi --release

# Checkpoint routing (no weights)
cargo test -p rlx-moshi --test checkpoint_routing --release

just test-moshi-weights

# Q8 GGUF key coverage (needs model.q8.gguf)
RLX_MOSHI_Q8_GGUF=.cache/moshiko-q8/model.q8.gguf cargo test -p rlx-moshi --test gguf_keys --release

# MLX key mapping (needs model.q4.safetensors)
RLX_MOSHI_MLX_Q4=.cache/moshiko-mlx-q4/model.q4.safetensors cargo test -p rlx-moshi --test mlx_keys --release

# GPU quick check
RLX_MOSHI_GPU_SMOKE=1 cargo test -p rlx-moshi --test gpu_smoke --features "gpu-lm,compiled-lm,metal" --release

# Streaming duplex → WAV → Whisper (slow; needs weights)
RLX_MOSHI_STREAM_E2E=1 just test-moshi-stream-whisper
```
