# rlx-models

Concrete model graph builders + weight loaders for RLX — the "what actually runs" layer.

**[152 model families](MODELS.md)** across 15 categories — language & multimodal LLMs, vision, speech (ASR/TTS), audio codecs, and more — each a standalone crate under `crates/`. See **[MODELS.md](MODELS.md)** for the complete catalog with per-model backend support.

Standalone repo: [github.com/MIT-RLX/rlx-models](https://github.com/MIT-RLX/rlx-models). Clone next to [`rlx`](https://github.com/MIT-RLX/rlx):

```text
rlx-workspace/
  rlx/          # github.com/MIT-RLX/rlx
  rlx-models/   # github.com/MIT-RLX/rlx-models
  candle/       # optional, for parity-candle
```

```bash
git clone https://github.com/MIT-RLX/rlx.git
git clone https://github.com/MIT-RLX/rlx-models.git
cd rlx-models && cargo test -p rlx-models
```

The RLX monorepo lists `../rlx-models/crates/rlx-models` as a workspace member; you can also run `cd rlx && cargo test -p rlx-models` there.

Agent-oriented quick reference: [AGENTS.md](AGENTS.md).

## Contents

- [Model catalog](#model-catalog)
- [Architecture](#architecture)
- [Running models](#running-models)
- [What's here](#whats-here)
- [Text-to-speech (TTS)](#text-to-speech-tts)
- [Install](#install)
- [Quickstart — embeddings](#quickstart--embeddings)
- [High-level runner API](#high-level-runner-api)
- [Adding a new model](#adding-a-new-model)
- [Compile profiles](#compile-profiles-tier-1)
- [Qwen3](#qwen3)
- [MiniCPM5](#minicpm5)
- [Qwen3-TTS](#qwen3-tts)
- [Voxtral TTS](#voxtral-tts)
- [VAD (Earshot + Silero)](#vad-earshot--silero)
- [Wake-word](#wake-word)
- [Build and test](#build-and-test)
- [Publishing (crates.io)](#publishing-cratesio)
- [Status](#status)
- [Gotchas](#gotchas)
- [Per-crate READMEs](#per-crate-readmes)
- [License](#license)

## Model catalog

This workspace ships **152 model families** (one crate per architecture), plus 4 training crates and shared infrastructure. Highlights:

- **26 language models** — Qwen3 / 3.5 / 3.6, Llama 3.2 / 4, Gemma, DeepSeek-V3, GLM-4.x, MiniMax, Jamba, Mamba, Phi, gpt-oss, LLaDA2, MiniCPM5, TinyLlama, …
- **12 vision-language / omni** — Fara1.5, Qwen2.5-VL, Llama-3.2-Vision, Florence-2, LocateAnything, Kimi-K3, Inkling, …
- **9 vision** encoders / detection / segmentation — DINOv2 / v3, SigLIP 2, V-JEPA 2, SAM 1 / 2 / 3, Grounding DINO — plus **4 biomedical** and **3 embedding** models.
- **14 ASR** and **43 TTS / speech-LM** models, **11 neural audio codecs**, plus voice conversion, music generation, source separation, wake-word, VAD, OCR, and robotics (VLA).

See **[MODELS.md](MODELS.md)** for the complete list — every model, its crate, a one-line description, and the backends it supports (CPU · Metal · MLX · CUDA · ROCm · wgpu · Vulkan).

## Architecture

This repo is a **Cargo workspace**: one library crate per model family under `crates/`, plus shared infrastructure. The `rlx-models` package is a thin **facade** that re-exports historical paths (`rlx_models::qwen3`, `rlx_models::sam`, …).

```text
rlx-models/
├── Cargo.toml              # workspace members + [workspace.dependencies]
├── justfile                # shortcuts (optional)
├── crates/
│   ├── rlx-models-core/    # config, weight_map, flow_bridge (package `rlx-core`)
│   ├── rlx-ssm/            # SSM flow stages + custom ops (Mamba, LFM, …)
│   ├── rlx-cli/            # shared CLI + rlx-inspect
│   ├── rlx-<model>/        # one crate per family
│   └── rlx-models/         # facade + optional rlx-run multiplexer
└── crates/rlx-models/examples/   # integration templates
```

### Crates

The table below highlights shared infrastructure and a selection of model crates; for the **complete** model list (all 152 families, with backends) see **[MODELS.md](MODELS.md)**.

| Crate | Model / role |
|---|---|
| `rlx-models-core` (`rlx-core`) | config, `weight_map`, `weight_loader`, `flow_bridge`, `flow_util` |
| `rlx-ssm` | SSM flow stages (`MambaScanStage`, decode-step custom ops) |
| `rlx-mamba` | Mamba1 block + multi-backend driver |
| `rlx-bert` | BERT |
| `rlx-nomic` | NomicBERT |
| `rlx-vision` | NomicVision |
| `rlx-dinov2` | DINOv2 |
| [`rlx-dinov3`](crates/rlx-dinov3/README.md) | DINOv3 ViT (2-D axial RoPE, registers, LayerScale, optional gated MLP) |
| [`rlx-trellis2`](crates/rlx-trellis2/README.md) | TRELLIS.2-4B image→3D (flow DiTs + sparse VAEs + dual-grid mesh) |
| [`rlx-uni2`](crates/rlx-uni2/README.md) | UNI2-h pathology ViT-H/14 (packed SwiGLU + registers) |
| [`rlx-hoct`](crates/rlx-hoct/README.md) | HOCT cell tracking (regionprops → edge transformer → ILP) |
| `rlx-bioclip2` | BioCLIP-2 (OpenCLIP ViT-L-14) |
| [`rlx-siglip2`](crates/rlx-siglip2/README.md) | SigLIP 2 (fixed-resolution + NaFlex) |
| `rlx-embed` | embedding runtime |
| `rlx-sam` / `sam2` / `sam3` | SAM family |
| `rlx-sam-ir` | shared mask-decoder IR |
| `rlx-qwen3` | Qwen3 LM |
| `rlx-qwen35` | Qwen3.5 / 3.6 |
| [`rlx-fara`](crates/rlx-fara/README.md) | Microsoft Fara1.5 CUA ([Fara1.5-4B](https://huggingface.co/microsoft/Fara1.5-4B) / [9B](https://huggingface.co/microsoft/Fara1.5-9B); Qwen3.5 multimodal) |
| `rlx-llama32` | LLaMA 3.2 |
| `rlx-minicpm5` | MiniCPM5 (Llama-shaped; [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)) |
| [`rlx-nanbeige`](crates/rlx-nanbeige/README.md) | Nanbeige4.2 Looped Transformer ([Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)) |
| `rlx-tinyllama` | TinyLlama-1.1B (Llama-shaped; [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)) |
| [`rlx-inkling`](crates/rlx-inkling/README.md) | Inkling multimodal MoE ([thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling)); config + weight map + synth text forward |
| [`rlx-laguna`](crates/rlx-laguna/README.md) | Laguna MoE ([poolside/Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) / XS / [GGUF](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF)); `just fetch-laguna` → `.cache/laguna-s` (nested Unsloth quants; `--prefer Q4_K_M`); packed mmap generate (KV cache, optional Metal/MLX); `--tokenizer-dir` / OpenAI via [`rlx-openai`](crates/rlx-openai/README.md) |
| [`rlx-openai`](crates/rlx-openai/README.md) | Central OpenAI HTTP server (`RegistryBackend`); `--engine qwen3\|laguna\|qwen35\|…` |
| `rlx-gemma` | Gemma / Gemma 2 |
| `rlx-llada2` | LLaDA2 + TIDE offload |
| `rlx-flux2` | FLUX.2 |
| `rlx-vjepa2` | V-JEPA 2 |
| `rlx-wav2vec2-bert` | Wav2Vec2-BERT |
| `rlx-wav2vec2-asr` | Wav2Vec2 CTC forced alignment (WhisperX-style word timestamps) |
| `rlx-whisper` | OpenAI Whisper ASR (segment + word timestamps, optional diarization) |
| [`rlx-conformer-ctc`](crates/rlx-conformer-ctc/README.md) | NVIDIA NeMo Conformer-CTC (`stt_en_conformer_ctc_small`, native `.nemo`) |
| `rlx-diarize` | Speaker diarization (embedding + clustering) |
| [`rlx-fft`](crates/rlx-fft/README.md) | Learned butterfly FFT, Welch PSD, fast top-K spectral peaks |
| [`rlx-vad`](crates/rlx-vad/README.md) | Earshot + Silero VAD (embedded weights, 16 kHz) |
| [`rlx-wake`](crates/rlx-wake/README.md) | Shared wake-word API / mel / WakeCnn / ternary |
| [`rlx-wakeword-core`](crates/rlx-wakeword-core/README.md) | `no_std` mel + WakeCnn + ternary (embedded-ready) |
| [`rlx-wakeword`](crates/rlx-wakeword/README.md) | First-party event-streaming wakeword product |
| [`rlx-wakeword-wasm`](crates/rlx-wakeword-wasm/README.md) | WASM / Web Worker bindings (`wasm-pack`, not crates.io) |
| [`rlx-openwakeword`](crates/rlx-openwakeword/README.md) | Native openWakeWord (ONNX parity optional) |
| [`rlx-nanowakeword`](crates/rlx-nanowakeword/README.md) | Native nanowakeword CNN / lite |
| [`rlx-porcupine`](crates/rlx-porcupine/README.md) | Porcupine-style wake CNN on RLX (no `.ppn`) |
| [`rlx-voxrt`](crates/rlx-voxrt/README.md) | VoxRT-style wake CNN on RLX (no `.vxrt`) |
| `rlx-voxtral` | Mistral Voxtral speech LM |
| `rlx-voxtral-tts` | Voxtral-4B-TTS inference (codec + Ministral LM) |
| `rlx-voxtral-tts-train` | Native RLX voice-clone training (encoder + LoRA) |
| [`rlx-qwen3-tts`](crates/rlx-qwen3-tts/README.md) | Qwen3-TTS — voice clone + CustomVoice TTS, progressive streaming, duplex voice chat (Whisper + Qwen3 LM). [JFK samples + roundtrip audio](crates/rlx-qwen3-tts/examples/audio/) ship in the crate. |
| [`rlx-tts`](crates/rlx-tts/README.md) | RLX FastSpeech2 + WaveRNN TTS (Hub `rlx-tts.rlxp`) |
| `rlx-locateanything` | NVIDIA LocateAnything-3B VLM (grounding) |
| [`rlx-unlimited-ocr`](crates/rlx-unlimited-ocr/README.md) | Baidu Unlimited-OCR VLM (SAM+CLIP DeepEncoder + MoE LM) |
| [`rlx-ppocrv6`](crates/rlx-ppocrv6/README.md) | PP-OCRv6 tiny/small OCR — native HIR + safetensors (no runtime ONNX) |
| [`rlx-asr`](crates/rlx-asr/README.md) | RLX streaming Conformer ASR (`weights/asr/model.rlxp`); facade `streaming-asr` |
| `rlx-cli` | shared CLI helpers + `rlx-inspect` |
| `rlx-models` | facade (re-exports) + optional `rlx-run` multiplexer |

### How to depend

| Goal | Depend on |
|------|-----------|
| One model only (fast builds) | `rlx-qwen3`, `rlx-sam3`, … |
| Stable `rlx_models::qwen3` paths | `rlx-models` facade |
| CLI / inspect only | `rlx-cli` |

New code that only needs Qwen3 should depend on `rlx-qwen3` directly.

### Per-crate binaries

Each model crate with a CLI has `src/cli.rs` (`pub fn run`) and `src/bin/rlx-<name>.rs`. Shared flag parsing lives in `rlx-cli`.

**`rlx-run`** (in `rlx-models`) is an optional multiplexer over all built-in CLIs. Prefer per-crate binaries when you only need one family — they link less and compile faster.

**SAM unified runner:** `SamRunner` (SAM1/2/3) stays on the facade (`rlx-models/src/sam_runner.rs`) because `rlx-sam2` depends on `rlx-sam`. Per-arch CLIs are on `rlx-sam`, `rlx-sam2`, `rlx-sam3`.

Published `rlx*` crates (`rlx-runtime`, `rlx-flow`, …) are pinned at **0.2.8** in root `[workspace.dependencies]`; every crate uses `{ workspace = true }`. **Local dev** with a sibling `../rlx` checkout: `cp .cargo/config.toml.example .cargo/config.toml` (gitignored patches). **Publish / CI** uses crates.io only — no `.cargo/config.toml`, no `[patch.crates-io]` in committed `Cargo.toml`.

## Running models

### just (shortcuts)

Install [just](https://github.com/casey/just) (`brew install just`). From the repo root:

```sh
just                          # list recipes
just qwen3 -- --weights model.gguf --prompt-ids 1,2,3
just inspect weights/model.gguf
just weights-scan -- --query qwen --json
just qwen3-metal -- --weights model.gguf --device metal --prompt-ids 1,2,3
just fetch-minicpm5
just minicpm5 -- --weights /tmp/rlx-weights/MiniCPM5-1B/model-00000-of-00001.safetensors --device cpu --prompt-ids 1,42 --max-tokens 16
just minicpm5-chat "Hello from MiniCPM5"
just fetch-nanbeige
just nanbeige -- --weights /tmp/rlx-weights/Nanbeige4.2-3B --device cpu --prompt-ids 1,42 --max-tokens 16
just features=all-backends test-nanbeige-backends
just nanbeige-backend-matrix
just features=all-backends bench-nanbeige-backends
```

Pass model CLI flags after `--`. MiniCPM5 details: [crates/rlx-minicpm5/README.md](crates/rlx-minicpm5/README.md) and [MiniCPM5](#minicpm5). GPU backends: `just features=all-backends qwen3 -- --device metal`, `just qwen35-all-backends -- …`, or per-crate `qwen3-all-backends` / `qwen35-all-backends`.

### Per-crate binaries (recommended)

| Binary | Crate | Example |
|--------|-------|---------|
| `rlx-qwen3` | `rlx-qwen3` | `cargo run -p rlx-qwen3 --bin rlx-qwen3 --release -- --weights model.gguf --prompt-ids 1,2,3` |
| `rlx-qwen35` | `rlx-qwen35` | `cargo run -p rlx-qwen35 --bin rlx-qwen35 --release -- …` |
| `rlx-fara` | `rlx-fara` | `just fetch-fara-4b && just fara --model-dir .cache/fara/4b --image shot.png --goal "…" --device cpu` |
| `rlx-llama32` | `rlx-llama32` | `cargo run -p rlx-llama32 --bin rlx-llama32 --release -- …` |
| `rlx-minicpm5` | `rlx-minicpm5` | `cargo run -p rlx-minicpm5 --features tokenizer --release -- --weights …/model.safetensors --prompt-ids 1,42` |
| `rlx-tinyllama` | `rlx-tinyllama` | `cargo run -p rlx-tinyllama --features tokenizer --release -- --weights …/model.safetensors --prompt-ids 1,42` |
| `rlx-gemma` | `rlx-gemma` | `cargo run -p rlx-gemma --bin rlx-gemma --release -- --weights model.gguf --prompt-ids 1,2,3` |
| `rlx-dinov2` | `rlx-dinov2` | `cargo run -p rlx-dinov2 --bin rlx-dinov2 --release -- …` |
| `rlx-dinov3` | `rlx-dinov3` | `cargo run -p rlx-dinov3 --bin rlx-dinov3 --release -- --weights dinov3-vitb16.safetensors --variant vitb16 --image cat.jpg` ([docs](crates/rlx-dinov3/README.md)) |
| `rlx-uni2` | `rlx-uni2` | `cargo run -p rlx-uni2 --bin rlx-uni2 --release -- --weights uni2h.safetensors --image tile.png` ([docs](crates/rlx-uni2/README.md)) |
| `rlx-hoct` | `rlx-hoct` | `just fetch-hoct && just hoct -- track -m .cache/hoct/general_v0.safetensors --labels labels.raw -o /tmp/out` ([docs](crates/rlx-hoct/README.md)) |
| `rlx-bioclip2` | `rlx-bioclip2` | `cargo run -p rlx-bioclip2 --bin rlx-bioclip2 --release -- --model-dir weights/bioclip-2 --image photo.jpg --labels "cat,dog"` |
| `rlx-siglip2` | `rlx-siglip2` | `cargo run -p rlx-siglip2 --bin rlx-siglip2 --release -- --model-dir weights/siglip2-base-224 --image photo.jpg --labels "cat,dog"` |
| `rlx-vjepa2` | `rlx-vjepa2` | `cargo run -p rlx-vjepa2 --bin rlx-vjepa2 --release -- …` |
| `rlx-wav2vec2-bert` | `rlx-wav2vec2-bert` | `cargo run -p rlx-wav2vec2-bert --bin rlx-wav2vec2-bert --release -- …` |
| `rlx-whisper` | `rlx-whisper` | `cargo run -p rlx-whisper --bin rlx-whisper --release -- --weights model.safetensors --wav audio16k.wav` |
| `rlx-fft` | `rlx-fft` | `cargo run -p rlx-fft --release -- bench-welch-peaks --n-fft 256 --batch 32 --strategy auto` ([docs](crates/rlx-fft/README.md)) |
| `rlx-vad` | `rlx-vad` | `cargo run -p rlx-vad --release -- --backend silero --wav audio16k.wav` ([docs](crates/rlx-vad/README.md)) |
| `rlx-wakeword` | `rlx-wakeword` | `just wakeword-demo -- --wav clip.wav --hop-ms 40` |
| `rlx-openwakeword` | `rlx-openwakeword` | `just openwakeword-demo -- --wav clip.wav` |
| `rlx-nanowakeword` | `rlx-nanowakeword` | `just nanowakeword-demo -- --wav clip.wav` |
| `rlx-porcupine` | `rlx-porcupine` | `just porcupine-demo -- --wav clip.wav` |
| `rlx-voxrt` | `rlx-voxrt` | `just voxrt-demo -- --wav clip.wav` |
| `rlx-voxtral` | `rlx-voxtral` | `cargo run -p rlx-voxtral --bin rlx-voxtral --release -- --weights model_dir --wav audio16k.wav --transcribe` |
| `rlx-voxtral-tts` | `rlx-voxtral-tts` | `just voxtral-tts -- --model-dir DIR --text "Hello" --voice neutral_female -o out.wav` |
| `rlx-voxtral-tts-train` | `rlx-voxtral-tts-train` | `just voxtral-tts-train-production -- --model-dir DIR --wav-dir WAVS --device auto` |
| `rlx-locateanything` | `rlx-locateanything` | `cargo run -p rlx-locateanything --bin rlx-locateanything --release -- --model-dir DIR --dry` |
| `rlx-sam1` | `rlx-sam` | `cargo run -p rlx-sam --bin rlx-sam1 --release -- …` |
| `rlx-sam2` | `rlx-sam2` | `cargo run -p rlx-sam2 --bin rlx-sam2 --release -- …` |
| `rlx-sam3` | `rlx-sam3` | `cargo run -p rlx-sam3 --bin rlx-sam3 --release -- …` |
| `rlx-flux2` | `rlx-flux2` | `cargo run -p rlx-flux2 --bin rlx-flux2 --release -- …` |
| `rlx-flux2-serve` | `rlx-flux2` | JSON-lines server on stdin |
| `rlx-inspect` | `rlx-cli` | `cargo run -p rlx-cli --bin rlx-inspect -- model.gguf` |

Flags match the corresponding `rlx-run` subcommand (without the subcommand name).

### Multiplexer (`rlx-run`)

```sh
cargo run -p rlx-models --bin rlx-run --release --features metal -- \
    qwen3 --weights Qwen3-0.6B-Q4_K_M.gguf --device metal --prompt-ids 1,17,42

cargo run -p rlx-models --bin rlx-run -- inspect Qwen3-0.6B-Q4_K_M.gguf
```

`rlx-inspect` dumps format, tensor count, dtype histogram, GGUF metadata, MTP heads, and multi-`.gguf` dir hints (`--prefer Q4_K_M`).

**Local discovery:** `rlx-inspect scan` walks LM Studio, Ollama, Hugging Face hub (MLX/vLLM), Lemonade, and RLX local dirs on macOS/Linux/Windows (`weights/`, `.cache/`, `%TEMP%/rlx-weights` or `/tmp/rlx-weights`, `$RLX_WEIGHTS_DIR`, `$RLX_WEIGHTS_PATHS`). `rlx-inspect resolve <query>` picks one path (prefers `Q4_K_M`). Library: `rlx_core::weights_discover` / `rlx_models::{scan_weights, resolve_weight_query, …}` (see [crates/rlx-models-core/README.md](crates/rlx-models-core/README.md)). Short `--weights` names that are not existing paths also resolve via the same scanner when CLIs use `resolve_weights_cli`.

### Custom CLI

Downstream tools can register runners without forking `rlx-models`:

```rust
use rlx_cli::{dispatch, register_cli};

register_cli("my-model", "…", |args| { /* … */ });
dispatch(&argv)?;
```

See `crates/rlx-models/examples/register_custom_runner.rs`.

### Examples (facade)

Integration templates on the `rlx-models` package:

```sh
cargo run -p rlx-models --example run_qwen3_gguf --release -- [args]
just example-qwen3-gguf -- /path/to/model.gguf
```

| File | What it does |
|---|---|
| `run_qwen3_safetensors.rs` | Qwen3 from HF safetensors, builder API, streaming greedy decode |
| `run_qwen3_gguf.rs` | Same from `.gguf` (Q4_K_M / Q5_K_M / Q6_K), MTP head detection |
| `run_sam1.rs` | SAM 1 — encode image, prompt encoder + mask decoder |
| `run_sam2.rs` | SAM 2 — FPN + memory attention |
| `run_sam3.rs` | SAM 3 — text-conditioned detection + masks |
| `qwen3_gguf_inference.rs` | Detailed Qwen3 GGUF walk-through |
| `gguf_qwen3_probe.rs` | Validate `hf_to_gguf_name` against a real GGUF |
| `qwen3_matrix.rs` | (B, L, mode) × (CPU, Metal, MLX, wgpu) parity + perf vs candle |
| `minicpm5_download.rs` | Fetch [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) safetensors (`hf-download`) |
| `minicpm5_gguf_download.rs` | Fetch GGUF quants (Q4_K_M / Q8_0 / F16) |
| `run_minicpm5.rs` | `MiniCpm5Runner` prefill + greedy decode from safetensors |
| `minicpm5_forward_bench.rs` | Wall-clock prefill/decode across backends (real 1B weights) |
| `minicpm5_chat.py` | HF chat template → `rlx-minicpm5` (`just minicpm5-chat`) |

#### Qwen3-TTS samples

Audio and charts live in [`crates/rlx-qwen3-tts`](crates/rlx-qwen3-tts/README.md). **Duplex voice chat** (bundled question → JFK-clone reply):

<video controls preload="metadata" src="crates/rlx-qwen3-tts/examples/audio/voice_chat_question.mp4"></video>

<video controls preload="metadata" src="crates/rlx-qwen3-tts/examples/audio/voice_chat_reply.mp4"></video>

![Voice-chat roundtrip latency](crates/rlx-qwen3-tts/examples/charts/voice_chat_latency.svg)

Three **JFK voice-clone** clips (`ask_not`, `moon`, `rlx_intro`) — ECAPA cosine 0.95+, WER 0–3.8 %. Full metrics, streaming API, and `just voice-chat-demo`: [crate README](crates/rlx-qwen3-tts/README.md).

SAM examples synthesize a 1024×1024 RGB gradient — swap in `image::open(path)` for real images.

```sh
just fetch-minicpm5
just example run_minicpm5 --release
```

### Weight fetch (optional)

`docker/qwen3-fetch/` — container pulls HF checkpoints into `./weights`; host runs `cargo test` / benches natively.

```sh
just fetch-qwen3
# or: docker build -t rlx-qwen3-fetch docker/qwen3-fetch && …
just fetch-minicpm5
just fetch-minicpm5-gguf Q4_K_M
```

## What's here

- **`qwen3`** — Qwen3 decoder LM (GQA, QK-norm, RoPE, SwiGLU, tied embeddings). Safetensors + GGUF; optional `qwen3.rlx.toml`. See [Qwen3](#qwen3).
- **`qwen35`** — Qwen3.5 / 3.6 hybrid (gated DeltaNet + periodic attention + optional MTP). GGUF via `Qwen35Runner`; optional `qwen35.rlx.toml`. Parity: `examples/qwen35_compare.rs` with the llama.cpp reference script in `examples/`.
- **`gemma`** — Gemma / Gemma 2 / 3 / 4 (GQA, RoPE, GeGLU, tied weights, Gemma2 softcap). Safetensors + GGUF; optional `gemma.rlx.toml`. See [crates/rlx-gemma/README.md](crates/rlx-gemma/README.md). CLI: `rlx-gemma` / `rlx-run gemma`. Parity: `just test-gemma-parity gemma2_synthetic`; backends: `just features=all-backends test-gemma-backends`.
- **`bert`** — BERT graph builder (MiniLM, BGE, all-MiniLM-L6-v2).
- **`nomic`** — NomicBERT (RoPE + SwiGLU).
- **`vision`** — NomicVision-style encoders.
- **`dinov2`** — DINOv2 ViT (B/14, L/14, g/14).
- **`dinov3`** — [DINOv3](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) ViT (2-D axial RoPE, separate q/k/v/o, LayerScale, register tokens, optional gated GeGLU MLP). CLI: `rlx-dinov3`. Bit-exact vs HF on CPU/Metal/MLX/wgpu. Gated weights. See [crates/rlx-dinov3/README.md](crates/rlx-dinov3/README.md).
- **`trellis2`** — [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) image→3D (structure/shape/texture flow DiTs + sparse VAEs + dual-grid mesh). Host DiT/VAE path + `Trellis2Runner`; CLI: `rlx-trellis2` / `just trellis2`. See [crates/rlx-trellis2/README.md](crates/rlx-trellis2/README.md).
- **`uni2`** — [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) pathology ViT-H/14 (packed SwiGLU MLP, 8 register tokens, `no_embed_class`). CLI: `rlx-uni2`. Gated CC-BY-NC-ND weights. See [crates/rlx-uni2/README.md](crates/rlx-uni2/README.md).
- **`hoct`** — Higher-Order Cell Tracking Transformer ([arXiv:2607.11754](https://arxiv.org/abs/2607.11754)): regionprops → kNN graph → edge transformer → HiGHS ILP → CTC/GEFF. CLI: `rlx-hoct`. See [crates/rlx-hoct/README.md](crates/rlx-hoct/README.md).
- **`bioclip2`** — BioCLIP-2, an OpenCLIP ViT-L-14 (image + text towers → shared 768-d embeddings, zero-shot). Pure-Rust PIL preprocessing; 100% parity vs `open_clip` on CPU/Metal/MLX/wgpu. CLI: `rlx-bioclip2`. See [crates/rlx-bioclip2/README.md](crates/rlx-bioclip2/README.md).
- **`siglip2`** — [SigLIP 2](https://huggingface.co/blog/siglip2) sigmoid-loss image + text encoder (attention-pooling MAP head, `gelu_pytorch_tanh`, multilingual Gemma tokenizer, sigmoid zero-shot). Both the **fixed-resolution** (`siglip2-base-patch16-224`, …) and **NaFlex** (variable resolution/aspect ratio) families. Pure-Rust preprocessing; HuggingFace parity (cos = 1.0) on CPU/Metal/MLX/wgpu/CUDA. CLI: `rlx-siglip2`. See [crates/rlx-siglip2/README.md](crates/rlx-siglip2/README.md).
- **`sam`**, **`sam2`**, **`sam3`** — Segment Anything encoders + mask decoders. Optional `sam.rlx.toml` next to weights (reference: `crates/rlx-sam/src/sam.rlx.toml`).
- **`flux2`** — FLUX.2 rectified-flow denoiser. `rlx-flux2` CLI; presets `flux2_dev()`, `flux2_klein_4b()`, `flux2_klein_9b()`. VAE, CFG, img2img, LoRA, `hf-download`, `rlx-flux2-serve`. GPU backends via `rlx-models` features (`metal`, `cuda`, …).
- **`embed`** — `RlxEmbed`, registry, tokenizers, pooling. `from_pretrained` with `hf-download`.
- **`config`**, **`weight_loader`** — HF config parsing; `WeightMap` + `GgufLoader` (K-quants, MTP isolation).
- **`fft`** — Learned butterfly FFT, mel/Welch pipelines, IO-aware Welch peak picker (`AutoWelchPeaks`), fused `Op::WelchPeaks` on GPU. CLI: `rlx-fft`. See [crates/rlx-fft/README.md](crates/rlx-fft/README.md).
- **`mamba`** — Mamba1 SSM block (`rlx-mamba`); SSM via `rlx-ssm` + `SelectiveScan`. See [crates/rlx-mamba/README.md](crates/rlx-mamba/README.md).
- **`lfm`**, **`minimax`**, **`nemotron`** — hybrid runners using `rlx-ssm` decode-step stages.
- **`minicpm5`** — MiniCPM5 edge LMs (Llama-shaped 1B). Wraps `Llama32Runner`; safetensors + GGUF. See [MiniCPM5](#minicpm5) and [crates/rlx-minicpm5/README.md](crates/rlx-minicpm5/README.md).
- **`tinyllama`** — TinyLlama-1.1B (Llama-shaped; 2048 hidden, 22 layers). Wraps `Llama32Runner`; safetensors + GGUF. See [TinyLlama](#tinyllama) and [crates/rlx-tinyllama/README.md](crates/rlx-tinyllama/README.md).
- **`qwen3-tts`** — Qwen3-TTS Base (voice clone) + CustomVoice. ECAPA x-vector, 28-layer talker, 16-group code predictor, 12 Hz Mimi decode. [`VoiceClone`](crates/rlx-qwen3-tts/README.md#library-api) API, progressive streaming, and `bidirectional_voice_chat` (Whisper → Qwen3-0.6B → TTS). See [Qwen3-TTS](#qwen3-tts).
- **`voxtral-tts`** — Voxtral-4B-TTS native inference (Tekken tokenizer, codec decode, compiled LM). **`voxtral-tts-train`** — RLX autodiff training for reference-audio cloning (codec encoder + full attention LoRA). See [Voxtral TTS](#voxtral-tts).
- **`run`** — `Qwen3Runner`, `SamRunner`, … builders for one-call inference.

## Text-to-speech (TTS)

Nine inference crates cover lightweight edge models through multi‑billion‑parameter voice clones. Build GPU binaries with the same feature names as LMs (`metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, `all-backends`, `apple-silicon`). Training crates are listed separately below. Cross-backend Whisper/RTF notes: [TTS.md](TTS.md) (see the Cross-backend parity & benchmarks section).

### Models

| Model | Crate | Size | Rate | Weights | Voice modes | Streaming | CLI / `just` | Status |
|---|---|---:|---:|---|---|---|---|---|
| [Qwen3-TTS](crates/rlx-qwen3-tts/README.md) | `rlx-qwen3-tts` | 0.6B | 24 kHz | HF safetensors ([Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base), [CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)) | ECAPA clone + preset speakers | progressive + batched PCM | `just qwen3-tts`, `jfk_voice_clone` | production — duplex voice chat, HF parity tests |
| [Voxtral-4B-TTS](docker/voxtral-tts/README.md) | `rlx-voxtral-tts` | 4B | 24 kHz | HF safetensors ([Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)) | preset voices + reference clone (train encoder) | — | `just voxtral-tts` | production — native RLX codec + compiled LM |
| [KittenTTS mini](crates/rlx-kittentts/README.md) | `rlx-kittentts` | ~15M | 24 kHz | native [`kitten_tts_mini_rlx`](crates/kitten_tts_mini_rlx/README.md) bundle or optional ONNX ([KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8)) | named voices (Jasper, …); IPA or `--features espeak` text | — | `just kittentts`, `just fetch-kittentts` | production — native RLX graph default; optional ORT (`--features onnx`) |
| [RLX TTS](crates/rlx-tts/README.md) | `rlx-tts` | FastSpeech2 + WaveRNN h448 | 24 kHz | [`eugenehp/rlx-tts`](https://huggingface.co/eugenehp/rlx-tts) `rlx-tts.rlxp` | product path | — | `just fetch-rlx-tts` / `just tts-demo` | production — packed `.rlxp` + native RLX |
| [Orpheus](crates/rlx-orpheus/README.md) | `rlx-orpheus` | 3B | 24 kHz | GGUF LM + SNAC safetensors | 8 built-in + zero-shot clone (pretrained GGUF) | ~2k-sample PCM chunks | `just orpheus`, `just orpheus-demo` | production — Metal LM + eager or CoreML SNAC |
| [NeuTTS](crates/rlx-neutts/) | `rlx-neutts` | Nano / Air | 24 kHz | llama-tagged GGUF + NeuCodec safetensors | reference-audio clone | — | library API (`NeuTTS::load_with_decoder_on`) | production backbone + eager NeuCodec; no standalone CLI yet |
| [Kyutai TTS 1.6B](crates/rlx-kyutai-tts/README.md) | `rlx-kyutai-tts` | 1.6B | 24 kHz | HF safetensors ([tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)) | `speaker_wavs` cross-attn conditioning | planned | `just kyutai-tts` | production — DSM generate + Mimi; DepFormer eager |
| [Pocket TTS](crates/rlx-pocket-tts/README.md) | `rlx-pocket-tts` | ~100M | 24 kHz | safetensors ([ungated mirror](https://huggingface.co/Verylicious/pocket-tts-ungated)) | preset `audio_prompt` embeddings | flow LM (faster than realtime on CPU) | `cargo run -p rlx-pocket-tts --example generate --features hf-download` | production on CPU/Accelerate; optional RLX backends via `rlx` feature |
| [TinyTTS](crates/rlx-tiny-tts/) / [MeloTTS](crates/rlx-melotts/) | `rlx-tiny-tts` | VITS2 | 44.1 kHz | [`eugenehp/tiny-tts-rlx`](https://huggingface.co/eugenehp/tiny-tts-rlx) `tiny-tts.rlxp` | MALE / FEMALE | — | `just fetch-tiny-tts` / `just melotts-demo` | production — nested graphs; MeloTTS is an alias |
| [Soprano 1.1](crates/rlx-soprano/README.md) | `rlx-soprano` | ~80M | 32 kHz | [`eugenehp/soprano`](https://huggingface.co/eugenehp/soprano) `soprano.rlxp` | single voice | — | `just fetch-soprano` / `just soprano-demo` | production — nested backbone + Vocos; no Hub ONNX |
| [MOSS-TTS-Nano](crates/rlx-moss-nano/README.md) | `rlx-moss-nano` | 0.1B | 48 kHz stereo | [`eugenehp/moss-nano`](https://huggingface.co/eugenehp/moss-nano) `moss-nano.rlxp` | 18 builtin voices | — | `just fetch-moss-nano` / `just moss-nano` | production — hierarchical AR + codec; no Hub ONNX |
| [Inflect-Nano](crates/rlx-inflect-nano/README.md) | `rlx-inflect-nano` | ~4.6M | 24 kHz | exported safetensors bundle | single speaker | — | `rlx-inflect-nano --text "…"` | production — standalone Rust frontend; vocoder on RLX graph or CoreML (ORT) |

**Training (inference + finetune):**

| Crate | Target | Backends | Entry |
|---|---|---|---|
| `rlx-qwen3-tts-train` | Qwen3-TTS talker LoRA (JFK custom voice) | Metal, MLX | `just qwen3-tts-train-jfk-metal`, `just qwen3-tts-train-jfk-mlx` |
| `rlx-voxtral-tts-train` | Voxtral codec encoder + full-attention LoRA | all GPU backends | `just voxtral-tts-train-production`, [docker runbook](docker/voxtral-tts/README.md) |

### Notable features

| Model | Highlights |
|---|---|
| Qwen3-TTS | `VoiceClone` API, progressive streaming (`StreamMode::Progressive`), duplex voice chat (Whisper + Qwen3 LM), CustomVoice presets, optional `incremental-decode` / `speculative-decode` features |
| Voxtral-4B-TTS | Tekken tokenizer, compiled Ministral LM, reference WAV clone after native encoder training |
| KittenTTS | Smallest footprint; **native RLX default** (`kitten_tts_mini_rlx`); optional ORT via `--features onnx` / `native-fast` for Metal |
| Orpheus | Emotive tags (`<laugh>`, …), SNAC on Apple ANE (`--device coreml`), streaming decode, GGUF Q4_K_M |
| NeuTTS | On-device clone; GGUF backbone via `rlx-llama32`; optional `burn-gpu` NeuCodec |
| Kyutai TTS | Depth-multiplexed Helium + DepFormer, 32 codebooks, en/fr SPM; `KyutaiTtsSession::generate` → Mimi PCM |
| Pocket TTS | Kyutai FlowLM + Mimi decoder path; Whisper-validated; CPU-first |
| TinyTTS / MeloTTS | VITS2 at 44.1 kHz; nested `graphs/*.rlxp`; monotonic alignment in Rust glue |
| Soprano | ~80M Qwen3 AR + Vocos @ 32 kHz; nested `.rlxp`; Whisper-validated |
| MOSS-TTS-Nano | Hierarchical AR codec-LM @ 48 kHz stereo; 18 builtin voices; nested `.rlxp` |
| Inflect-Nano | FastSpeech-style acoustic + Snake HiFi-GAN vocoder; full G2P frontend in Rust |

### Backends

Legend: ✅ supported · ⚠️ partial (host fallback, ORT EP, or opt-in feature) · ❌ not wired

| Model | cpu | metal | mlx | cuda | rocm | wgpu | vulkan | Notes |
|---|---|---|---|---|---|---|---|---|
| Qwen3-TTS | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Progressive speech decode uses CPU on Metal/MLX (GPU prefix-length mismatch) |
| Voxtral-4B-TTS | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `--device` on all backends; compiled LM path |
| KittenTTS (native, default) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `kitten_tts_mini_rlx`; discrete wgpu/Vulkan wave-capped — [NATIVE.md](crates/rlx-kittentts/NATIVE.md) |
| KittenTTS (ONNX optional) | ✅ | ⚠️ | ❌ | ⚠️ | ⚠️ | ⚠️ | ❌ | `--features onnx`; ONNX Runtime execution providers |
| Orpheus LM | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ⚠️ | wgpu/Vulkan: CPU GGUF prefill+decode; MLX opt-in (`ORPHEUS_MLX_KV=1`) |
| Orpheus SNAC | ✅ | — | — | — | — | — | — | Eager CPU default; CoreML ANE with `--features coreml` |
| NeuTTS LM | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ⚠️ | Same routing as `rlx-llama32` / GGUF packed rules |
| NeuTTS codec | ✅ | — | — | — | — | ⚠️ | — | Eager ndarray; optional Burn wgpu (`burn-gpu`) |
| Kyutai TTS | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Temporal LM on RLX; DepFormer eager; Mimi decode |
| Pocket TTS | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ | Default Accelerate/ndarray CPU; enable `rlx` + backend features for GPU graph |
| TinyTTS | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `--device ane` with `--features coreml` |
| Inflect-Nano | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ❌ | Vocoder RLX graph; CoreML via static-shape ORT (`--features coreml`) |

Quick checks: `just test-qwen3-tts-parity`, `just test-kittentts-e2e`, `just test-orpheus-whisper`, `just test-voxtral-tts-codec`, `cargo test -p rlx-inflect-nano --features all-backends`.

Per-model runbooks: [Qwen3-TTS](#qwen3-tts), [Voxtral TTS](#voxtral-tts), and crate READMEs in [Per-crate READMEs](#per-crate-readmes).

## Install

```toml
[dependencies]
rlx-models = "0.2.8"
```

HF-hub download:

```toml
rlx-models = { version = "0.2.8", features = ["hf-download"] }
```

Workspace and published model crates are **0.2.8**, pinned to upstream **`rlx*`** **0.2.8** on crates.io (`[workspace.dependencies]`). Local sibling `../rlx`: `cp .cargo/config.toml.example .cargo/config.toml` (gitignored).

## Quickstart — embeddings

```rust
use rlx_models::embed::{Pooling, RlxEmbed};

let mut model = RlxEmbed::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
let hidden = model.forward(&[("input_ids", &ids), ("attention_mask", &mask)], 1, 16)?;
```

## High-level runner API

`rlx_models::run` exposes builder-style entry points (also `rlx::run` in the monorepo):

```rust
use rlx_models::run::{Qwen3Runner, Precision};
use rlx_runtime::Device;

let mut runner = Qwen3Runner::builder()
    .weights("Qwen3-0.6B-Q4_K_M.gguf")
    .device(Device::Metal)
    .max_seq(128)
    .precision(Precision::F32)
    .max_memory_gb(16.0)
    .stream(true)
    .use_mtp(false)
    .packed_weights(false)
    .build()?;

runner.generate(&prompt_ids, 32, |tok| print!("{tok} "))?;
```

**Packed weights** (large GGUF on limited RAM — CPU-only, memory-frugal, slower):

```rust,ignore
let mut runner = Qwen3Runner::builder()
    .weights("Qwen3-14B-Q4_K_M.gguf")
    .packed_weights(true)
    .max_seq(128)
    .build()?;
runner.generate(&prompt_ids, 16, |tok| print!(" {tok}"))?;
let logits = runner.predict_logits(&prompt_ids)?;
```

Format (`safetensors` vs `gguf`) is auto-detected. SAM uses `SamRunner::builder(SamArch::Sam2)`.

CLI equivalent:

```sh
just qwen3 -- --weights Qwen3-14B-Q4_K_M.gguf --packed --max-seq 128 --max-tokens 16 --prompt-ids 1,17,42
# or: cargo run -p rlx-qwen3 --bin rlx-qwen3 --release -- …
```

## Adding a new model

Borrowed from Max's four-file layout; each architecture is a workspace crate `crates/rlx-<name>/`.

### 1. Create the crate

Root `Cargo.toml`:

```toml
# [workspace.members]
"crates/rlx-myarch",

# [workspace.dependencies]
rlx-myarch = { path = "crates/rlx-myarch" }
```

Depend on `rlx-core`, `rlx-ir`, `rlx-flow`, `rlx-runtime` as needed.

### 2. Source layout

```text
crates/rlx-myarch/src/
├── lib.rs
├── arch.rs       # ArchSpec registration (optional)
├── config.rs     # HF config.json
├── weights.rs    # HF → RLX name map
├── builder.rs    # graph construction
├── flow.rs       # compile helpers (optional split)
└── cli.rs        # pub fn run(args: &[String])
```

`arch.rs` registers with `rlx_core::arch_registry`. `weights.rs` holds rename rules; `builder.rs` emits IR. Reference: `crates/rlx-qwen3`.

### 3. Facade re-export

In `crates/rlx-models/src/lib.rs`:

```rust
pub mod myarch {
    pub use rlx_myarch::*;
}
```

### 4. CLI (optional)

- `cli.rs` + `[[bin]] name = "rlx-myarch"`
- Register in `crates/rlx-models/src/bin/rlx_run.rs`: `register_cli("myarch", "…", rlx_myarch::cli::run)`
- Add a `just` recipe in `justfile` (optional)

### 5. High-level runner (optional)

Put `MyArchRunner` in the model crate; re-export from `crates/rlx-models/src/run.rs`.

Legacy flat modules (`rlx-bert`, `rlx-nomic`) stay as-is until they grow — use this layout for **new** architectures.

## Compile profiles (tier-1)

Compile through tier-1 profiles, not bare `Session::compile(graph)`:

| Model | Profile helper | Optional file next to weights |
|---|---|---|
| Qwen3 | `flow_util::compile_graph_qwen3_prefill_with_params` | `qwen3.rlx.toml` |
| Qwen3.5 | `compile_support::compile_qwen35_prefill` / `compile_qwen35_decode` | `qwen35.rlx.toml` |
| SAM / SAM3 | `flow_util::compile_graph_sam_with_params` | `sam.rlx.toml` |
| Encoders | `flow_util::compile_graph_encoder_with_params` | — |

Synthetic Qwen3.5 weights for CPU checks: `rlx_models::qwen35::synth` (`tiny_cfg`, `medium_cfg`, `bench_cfg`, …).

```sh
just test-quick
# cargo test -p rlx-models --test qwen35_forward_check --test compile_profile_quick_check
```

Real-GGUF / backend checks: set `QWEN35_GGUF_PATH` (LMs) or vision env vars (`SAM3_GGUF_PATH`, `DINOV2_GGUF_PATH`, `FLUX_GGUF_PATH`, `W2V_BERT_GGUF_PATH`). Drain: `cargo test -p rlx-models --test vision_gguf_load --release`. Compile quick check: `cargo test -p rlx-models --test vision_gguf_compile --release` (SAM3 also needs `VISION_GGUF_COMPILE=1`; W2V-BERT needs `RLX_W2V_BERT_DIR` with `config.json`). FLUX: `cargo test -p rlx-models --test flux2_gguf_runner_quick_check --release` (`FLUX_GGUF_PATH` / `FLUX_MODEL_ROOT`; optional `FLUX_VAE_DIR` for VAE encode). Q4_0 fused matmul: `cargo test -p rlx-models --test gguf_legacy_quant_matmul --release`; Metal parity: `GGUF_LEGACY_METAL_PARITY=1` with `--features metal`. Enable `metal` / `mlx` / `cuda` / `parity-llama` per test file where noted.

## Qwen3

Prefill + decode on all seven standard backends (CPU, Metal, MLX, CUDA, ROCm, WGPU, Vulkan). Enable matching features at build time (`cargo build -p rlx-qwen3 --features all-backends`). Synthetic checks: `just features=all-backends test-qwen3-backends`. Parity: 100% top-1 vs HF (`tests/qwen3_parity.rs`).

### Safetensors

```rust
use rlx_models::qwen3::{Qwen3Config, build_qwen3_graph_sized_last_logits};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::Device;

let cfg = Qwen3Config::from_file("weights/Qwen3-0.6B/config.json".as_ref())?;
let mut wm = WeightMap::from_file("weights/Qwen3-0.6B/model.safetensors")?;
let (graph, params) = build_qwen3_graph_sized_last_logits(&cfg, &mut wm, 1, 128, false)?;
let mut compiled = rlx_models::flow_util::compile_graph_qwen3_prefill_with_params(
    Device::Metal, graph, params,
)?;
```

### GGUF

```rust
use rlx_models::weight_loader::GgufLoader;
let mut wm = GgufLoader::from_file("Qwen3-0.6B-Q4_K_M.gguf")?;
// same compile + run as safetensors
```

Demo: `just example-qwen3-gguf -- path/to/model.gguf`. Verified vs `unsloth/Qwen3-0.6B-GGUF` (cosine ≈ 0.976 vs F32 safetensors on Q4_K_M).

**Directories with several `.gguf` files:** pass `ResolveWeightsOptions { prefer_gguf_substring: Some("Q4_K_M"), .. }` or `gguf_index: Some(0)` (see `rlx_core::gguf_support`). Multi-part split GGUF (`split.count` > 1) auto-merges when all shards sit in the same directory; otherwise `rlx-inspect` lists missing parts.

### Weights API (model-agnostic loader)

**`rlx_core::weights`** only handles paths, file formats, and drain policy. It does **not** know about Qwen, FLUX, BERT, etc.

```rust
use rlx_core::weights::{self, LoadOpts};

let (path, map) = weights::open_map("weights/")?;
let (path, map) = weights::open_map_with(LoadOpts::map().prefer_q4_k_m(), "weights/")?;
let loaded = weights::open_with(LoadOpts::loader(), "model.gguf")?; // packed take / MTP
```

**Model-specific policy** belongs in each runner:

```rust
use rlx_core::{load_weight_map, gguf_validate_arch, EMBED_GGUF_ARCHES, DINOV2_GGUF_ARCHES};

// One call: resolve path, validate arch on .gguf, drain to F32 map
let map = load_weight_map(path, DINOV2_GGUF_ARCHES)?;

// Or split validate + open (embed / custom drain policy)
gguf_validate_arch(&path, EMBED_GGUF_ARCHES)?;
let (_path, map) = weights::open_map(path)?;
```

| Layer | Responsibility |
|-------|----------------|
| `weights` / `weight_registry` | `.gguf` / `.safetensors`, resolve dir, custom extensions |
| `gguf_validate_arch`, `assert_gguf_family` | Optional arch guard in **your** crate |
| `register_gguf_tensor_resolver` | HF ↔ `blk.*` / prefix strip per checkpoint layout |
| `BertConfig::from_gguf`, `Flux2Config::from_gguf` | Hyperparameters from metadata |

**Inspect:** `rlx-inspect path [--prefer Q4_K_M] [--json]` — directory listing, split-part hints, runner suggestions. `rlx-inspect scan [--query …]` / `rlx-inspect resolve <query>` / `just weights-scan` — discover weights in local LLM app caches.

**CLI:** LM / FLUX binaries accept `--prefer-quant` and `--gguf-index` (via `rlx_cli::resolve_weights_cli`); default quant preference is `Q4_K_M` in multi-file dirs.

**Splits:** Multi-part GGUF (`split.count` > 1) auto-merges when all parts are in the same directory; otherwise `rlx-inspect` lists missing shards.

**Legacy quants:** `Q4_0` / `Q8_0` support packed `DequantMatMul` on **CPU** and **Metal** (fused MSL dequant+matmul, 32-element blocks). Set `RLX_DISABLE_METAL_DEQUANT_GPU=1` to force host dequant on Apple GPUs.

**Example:** `cargo run -p rlx-models --example custom_weight_format`

### Apple Silicon

Metal lowers to MPSGraph (per shape). Env toggles:

| env var | effect |
|---|---|
| `RLX_DISABLE_MPSGRAPH=1` | per-op Metal thunks |
| `RLX_DISABLE_MPSGRAPH_EXECUTABLE=1` | JIT MPSGraph |
| `RLX_MPSGRAPH_PARAM_CONST=1` | bake weights into executable |
| `RLX_QWEN3_F16_LM_HEAD=1` | F16 final matmul |
| `RLX_MPSGRAPH_TRACE=1` | print lowering blockers |

Harness: `examples/qwen3_matrix.rs`.

## MiniCPM5

[openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) — 1B Llama decoder (GQA, RoPE, SwiGLU). Implemented in `rlx-minicpm5` on top of `rlx-llama32` with HF `config.json` / GGUF arch checks. **Full runbook:** [crates/rlx-minicpm5/README.md](crates/rlx-minicpm5/README.md).

### Download

```sh
just fetch-minicpm5                              # safetensors → /tmp/rlx-weights/MiniCPM5-1B
just fetch-minicpm5-gguf Q4_K_M                  # GGUF → …/MiniCPM5-1B-GGUF
```

### CLI

Uses the same flags as `rlx-llama32` (`--weights`, `--device`, `--prompt-ids`, `--tokenizer`, `--packed`, `--max-seq`, `--max-tokens`, …). Build with `tokenizer` for decode:

```sh
W=/tmp/rlx-weights/MiniCPM5-1B/model-00000-of-00001.safetensors

just minicpm5 -- --weights "$W" --device cpu --prompt-ids 1,42,314 --max-tokens 16

# GGUF packed prefill (CPU + Metal native; MLX/wgpu/CUDA use CPU execution path today):
just minicpm5 -- --weights /tmp/rlx-weights/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  --packed --device metal --prompt-ids 1,42 --max-tokens 8
```

### Chat (HF template)

```sh
pip install transformers
just fetch-minicpm5
just minicpm5-chat "What is 2+2? Answer in one sentence."
```

`minicpm5_chat.py` tokenizes with the official template, then runs `rlx-minicpm5` (defaults to **CPU** for reliable KV decode on Apple Silicon).

### Library

```rust
use rlx_minicpm5::MiniCpm5Runner;
use rlx_runtime::Device;

let mut runner = MiniCpm5Runner::builder()
    .weights("/tmp/rlx-weights/MiniCPM5-1B/model-00000-of-00001.safetensors")
    .device(Device::Cpu)
    .max_seq(512)
    .build()?;
let logits = runner.predict_logits(&[1, 42, 314])?;
```

Example: `just example run_minicpm5 --release` (or `cargo run -p rlx-models --example run_minicpm5 --release`).

### Tests

| Command | What |
|---------|------|
| `just test-minicpm5-parity-full` | RLX vs PyTorch (safetensors, needs weights) |
| `just test-minicpm5-backends-all` | Synthetic 1B-shaped graph, all backends |
| `just test-minicpm5-gguf-backends` | Real Q4_K_M GGUF packed prefill |
| `../rlx/rig.sh test-minicpm5` | Remote rig: CPU + CUDA + WGPU on Windows/WSL (after `sync` + `sync-minicpm5-gguf`) |
| `just bench-minicpm5-real --device cpu` | Wall-clock prefill/decode on 1B weights |

Multiplexer: `cargo run -p rlx-models --bin rlx-run --features tokenizer -- minicpm5 --weights …`.

## TinyLlama

[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) — 1.1B Llama-2-style decoder (2048 hidden, 22 layers, vocab 32000). Implemented in `rlx-tinyllama` on top of `rlx-llama32`. **Full runbook:** [crates/rlx-tinyllama/README.md](crates/rlx-tinyllama/README.md).

### Download

```sh
just fetch-tinyllama                              # safetensors → /tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0
just fetch-tinyllama-gguf Q4_K_M                  # GGUF → …/TinyLlama-1.1B-GGUF (TheBloke)
```

### CLI

Same flags as `rlx-llama32`. Build with `tokenizer` for decode; pass `tokenizer.json` for text (GGUF SentencePiece alone is not enough):

```sh
W=/tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0/model-00001-of-00003.safetensors

just tinyllama -- --weights "$W" --device cpu --prompt-ids 1,42,314 --max-tokens 16

just tinyllama -- --weights /tmp/rlx-weights/TinyLlama-1.1B-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --packed --device metal --prompt-ids 1,42 --max-tokens 8
```

### Tests

| Command | What |
|---------|------|
| `just test-tinyllama-gguf-backends` | Real Q4_K_M GGUF packed prefill vs CPU |
| `just test-tinyllama-backends-all` | Synthetic 1.1B-shaped graph, all backends |
| `just test-tinyllama-real` | Safetensors config + runner build |

## Whisper

OpenAI Whisper ASR in `rlx-whisper` with native Rust segment timestamps, optional word alignment (DTW or Wav2Vec2 CTC), Silero VAD chunking, and speaker diarization — no Python runtime. Runbook: [crates/rlx-whisper/README.md](crates/rlx-whisper/README.md).

### Subtitles and word timestamps

```bash
just fetch-whisper fetch-whisper-bench
just whisper-subtitles   # JFK → SRT with segment + DTW word times
just bench-whisper-subtitles -- --device metal --modes timestamps+dtw --runs 3
just bench-whisper-subtitles-all-backends -- --modes timestamps+dtw
```

CLI flags on `rlx-whisper`:

| Flag | Purpose |
|------|---------|
| `--timestamps` | Parse `<\|M.SS\|>` tokens → structured `WhisperTranscript` |
| `--word-align dtw\|wav2vec2` | Word-level times (DTW default; Wav2Vec2 optional) |
| `--silero-vad` | Chunk long audio with `rlx-vad` Silero |
| `--diarize` | Speaker labels via `rlx-diarize` |
| `--max-region-batch N` | Batched VAD-region encode width |
| `--output PATH` | Write SRT, VTT, TSV, or JSON |

Library API:

```rust
use rlx_whisper::{WhisperPipeline, WhisperPipelineOpts, WordAlignMode, WhisperRunner};

let runner = WhisperRunner::builder()
    .weights("model.safetensors")
    .device(rlx_runtime::Device::Metal)
    .timestamps(true)
    .mel_frames_for_pcm(&pcm)
    .build()?;

let mut pipeline = WhisperPipeline::new(runner, WhisperPipelineOpts {
    word_align: WordAlignMode::Dtw,
    use_silero_vad: true,
    max_region_batch: 4,
    ..Default::default()
});
let transcript = pipeline.run(&pcm)?;
```

### Device routing (Metal)

| Stage | Device |
|-------|--------|
| Mel encoder | Metal |
| Cross / prefill / bucketed decode | CPU (parity gate) |
| DTW align-hidden | Metal |

On JFK + whisper-tiny (~11 s), `timestamps+dtw` is ~3.3 s total on Metal (RTF ~0.30) vs ~4.4 s on CPU; word alignment drops from ~580 ms to ~120 ms with the GPU align-hidden graph.

### Features

| Feature | Enables |
|---------|---------|
| `timestamps` (default) | Segment parse + SRT/VTT/JSON export |
| `word-dtw` | Cross-attention + DTW word alignment |
| `word-w2v` | `rlx-wav2vec2-asr` CTC forced alignment |
| `silero-vad` | Silero VAD chunking |
| `diarize` | `rlx-diarize` speaker labels |
| `whisper-subtitles` (`rlx-models`) | Full stack for examples/tests |

### Tests and benches

| Command | What |
|---------|------|
| `just test-whisper-timestamps` | Segment parse, DTW units, wav2vec2/diarize crates |
| `just test-whisper-e2e` | Greedy decode vs reference (needs weights) |
| `just bench-whisper-subtitles` | Pipeline latency (ASR / align / Silero / diarize) |
| `just bench-whisper-subtitles-all-backends` | Same bench on CPU, Metal, CUDA, MLX, … |

## LocateAnything

[NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) — MoonViT vision + `mlp1` projector + Qwen2.5-3B with MTP box decoding. Crate: `rlx-locateanything`; runbook: [crates/rlx-locateanything/README.md](crates/rlx-locateanything/README.md).

```bash
just fetch-locateanything
export RLX_LOCATEANYTHING_DIR=.cache/locateanything/LocateAnything-3B

just test-locateanything-checkpoint
just locateanything-demo   # bundled sample in rlx-locateanything/fixtures/sample.jpg
just locateanything -- --model-dir $RLX_LOCATEANYTHING_DIR \
  --image page.png --task ground-single --phrase "red backpack" \
  --generation-mode hybrid --device cpu
```

| Command | What |
|---------|------|
| `just test-locateanything-backends` | Synthetic projector + LM on all RLX backends |
| `just test-locateanything-moonvit-backends` | Compiled MoonViT on GPU backends |
| `just test-locateanything-parity` | Full tensor + MTP decode + RLX/HF processor prompts + tasks + slow/fast/hybrid `generate()` vs HF (28 tests; real JPEG fixture) |
| `just test-locateanything-parity-real` | Real-photo subset (`fixtures/sample.jpg`; `RLX_LOCATEANYTHING_IMAGE` optional) |
| `just locateanything-demo` | Quick ground on bundled sample (no `--image`) |
| `just bench-locateanything-backends` | E2E timing per backend; **one subprocess per backend** by default (avoids OOM). Single backend: `--device wgpu --no-isolate` |

Weights are HF safetensors only (770 tensors: vision / projector / `language_model.*`).

## Unlimited-OCR

[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) — SAM-ViT-B + CLIP-L/14 deep encoder, linear projector, DeepSeek-V2 MoE LM with ring-window decode. Crate: `rlx-unlimited-ocr`; runbook: [crates/rlx-unlimited-ocr/README.md](crates/rlx-unlimited-ocr/README.md). Vision stays host-f32; MoE LM compiles on RLX backends. Host pack defaults to **`--lm-precision auto`** (F32→F16→Q8→Q4 by RAM); Q8/Q4 stay packed in IR.

```bash
just fetch-unlimited-ocr
# optional: export RLX_UNLIMITED_OCR_DIR=/path/to/snapshot

just unlimited-ocr -- --image page.png --device auto
just unlimited-ocr -- --image page.png --mode base --lm-precision q8_0
just unlimited-ocr -- --images page1.png,page2.png
just unlimited-ocr -- --pdf document.pdf --max-tokens 8192
```

| Command | What |
|---------|------|
| `just fetch-unlimited-ocr` | Download checkpoint into Hugging Face cache |
| `just unlimited-ocr -- …` | Single / multi / PDF OCR CLI |
| `just unlimited-ocr-metal -- …` | Same with Metal feature build |
| `just test-unlimited-ocr-backends` | Tiny MoE compile+run (incl. Q8/Q4 packed IR) |
| `just test-unlimited-ocr-token-parity` | Full-checkpoint greedy tokens vs CPU (env-gated) |
| `just test-unlimited-ocr-parity` | Checkpoint inventory + tokenizer + greedy e2e (HF Python optional) |
| `just bench-unlimited-ocr-lm-precision` | Host MiB / latency / accuracy vs F32 |

## Qwen3-TTS

[Qwen3-TTS-12Hz-0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) — native Rust voice clone and CustomVoice synthesis in `rlx-qwen3-tts`. Full runbook: [crates/rlx-qwen3-tts/README.md](crates/rlx-qwen3-tts/README.md).

### Download and clone

```bash
just fetch-qwen3-tts-base
export RLX_QWEN3_TTS_DIR=.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base

cargo build -p rlx-qwen3-tts --release --features apple-silicon --bin jfk_voice_clone
./target/release/jfk_voice_clone \
  --model-dir $RLX_QWEN3_TTS_DIR \
  --ref-wav assets/jfk/jfk_voice_clone.wav \
  --target-text "Hello from native Rust TTS." \
  --out-wav /tmp/hello.wav --device metal
```

### Duplex voice chat

Mic WAV → Whisper → Qwen3-0.6B → progressive TTS (JFK clone). Bundled roundtrip audio under `crates/rlx-qwen3-tts/examples/audio/`.

```bash
just fetch-qwen3 && just fetch-whisper-base   # LM + ASR weights
just voice-chat-demo                          # → /tmp/voice_chat_roundtrip/
```

`--turbo` preloads all models, streams LM tokens, and uses batched TTS by default (`--streaming-tts` for progressive partial-decode). Measured stop-speaking → first audio ≈ **5.1 s** on Apple Silicon (see [voice_chat_latency.svg](crates/rlx-qwen3-tts/examples/charts/voice_chat_latency.svg)).

### Streaming API

[`VoiceClone::generate_stream`](crates/rlx-qwen3-tts/README.md#live-streaming-api) supports `StreamMode::Batched` (lossless chunking of full `generate()`) and `StreamMode::Progressive` (codec frames decoded during AR). Progressive speech decode uses CPU on Metal/MLX (GPU prefix-length mismatch); CUDA and other backends use GPU speech decode when available.

### Tests

| Command | What |
|---------|------|
| `just test-qwen3-tts-parity` | Codec frames + speech decode vs reference (`RLX_QWEN3_TTS_DIR`) |
| `just features=all-backends test-qwen3-tts-backends` | Talker prefill/decode per backend |
| `just features=all-backends test-qwen3-tts-streaming` | Streaming PCM parity (batched + progressive) |
| `just qwen3-tts-vivian-demo` | CustomVoice preset speaker → `/tmp/vivian-demo.wav` |

Env: `RLX_QWEN3_TTS_CP_EAGER=1` / `RLX_QWEN3_TTS_SPEECH_EAGER=1` force CPU paths; `RLX_QWEN3_TTS_TIMING=1` prints stage breakdown.

## Voxtral TTS

[Mistral Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) — native Rust inference in `rlx-voxtral-tts`, voice-clone training in `rlx-voxtral-tts-train`. Full runbook: [docker/voxtral-tts/README.md](docker/voxtral-tts/README.md).

### Download and synthesize

```bash
just fetch-voxtral-tts
export RLX_VOXTRAL_TTS_DIR=.cache/voxtral/Voxtral-4B-TTS-2603

just voxtral-tts-prepare-voices
just voxtral-tts -- --model-dir $RLX_VOXTRAL_TTS_DIR \
  --text "Hello world" --voice neutral_female -o out.wav
```

### Voice cloning (native RLX training)

Public checkpoints omit the codec **encoder**. Train it in RLX, inject into `consolidated.safetensors`, then synthesize from a reference WAV:

```bash
# Optional manifest (transcript field improves ASR auxiliary loss):
just voxtral-tts-train-manifest -- --wav-dir ./wavs --out ./wavs/manifest.json

PRODUCTION=1 just features=all-backends voxtral-tts-train-production -- \
  --model-dir $RLX_VOXTRAL_TTS_DIR --wav-dir ./wavs \
  --manifest ./wavs/manifest.json --out-dir ./out/train --device auto

just voxtral-tts -- --model-dir $RLX_VOXTRAL_TTS_DIR \
  --reference-wav ./ref.wav --text "Hello from my voice" -o cloned.wav
```

Periodic checkpoints during long runs: `CHECKPOINT_EVERY=500`. Resume: `--resume-weights ./out/train/encoder/encoder_step_5000.safetensors --resume-step 5000`. Rig validation: `RLX_VOXTRAL_TTS_TRAIN_RIG=1 RLX_VOXTRAL_TTS_REF_WAV=./ref.wav just test-voxtral-tts-train-synthesize-rig` (reports mel similarity).

### Tests

| Command | What |
|---------|------|
| `just test-voxtral-tts-train` | Train crate unit + integration tests |
| `just test-voxtral-tts-train-backends` | Encoder/LoRA backward compile on all GPU backends |
| `just test-voxtral-tts-codec` | Codec round-trip |
| `just test-voxtral-tts-native-parity` | Native vs Docker reference export |

## AEC (acoustic echo cancellation)

[`rlx-aec`](crates/rlx-aec/README.md) — pure Rust **16 kHz** FDAF-NLMS (`rlx-fft` / `rustfft`) + optional per-bin RLX residual mask. Pre-ASR front-end for duplex voice chat.

```sh
cargo run -p rlx-aec --release -- \
  --mic-wav echoed_mic.wav --ref-wav speaker_ref.wav --out-wav cleaned.wav
just test-aec
just bench-aec
just bench-aec-parity   # Rust + Python NLMS baseline → /tmp/aec_compare.csv
```

Voice chat: `cargo run -p rlx-qwen3-tts --example bidirectional_voice_chat -- … --aec` feeds TTS playback into the far-end reference ring.

## VAD (Earshot + Silero)

[`rlx-vad`](crates/rlx-vad/README.md) — 16 kHz voice activity detection with **embedded weights** (no ONNX Runtime):

- **Earshot** — `weights/earshot_weights.bin` (~75 KiB)
- **Silero** — `weights/silero_vad_16k.safetensors` (~920 KiB), exported from official `silero_vad.onnx` 16 kHz branch

```sh
cargo run -p rlx-vad --release -- --backend silero --wav audio16k.wav
cargo run -p rlx-vad --example jfk_bench --release
cargo test -p rlx-vad
```

Regenerate Silero embed: `python3 scripts/export_silero_onnx_weights.py …` (see crate README). The Hugging Face file named `silero_vad_16k.safetensors` is a different (8 kHz) graph — do not substitute it.

Shared loader: `rlx_core::embedded_safetensors::EmbeddedSafetensors`.

## Wake-word

First-party product [`rlx-wakeword`](crates/rlx-wakeword/README.md): event API (20–40 ms hops), multi-phrase train/pack, optional VAD + speaker-id, ternary weights for bake TQ2 / fused kernels. Shared math in [`rlx-wake`](crates/rlx-wake/README.md) / [`rlx-wakeword-core`](crates/rlx-wakeword-core/README.md). WASM / Web Worker: [`rlx-wakeword-wasm`](crates/rlx-wakeword-wasm/README.md) (local `wasm-pack` only). Compat engines remain for parity.

| Crate | Notes |
|-------|-------|
| [`rlx-wakeword`](crates/rlx-wakeword/README.md) | Product session + train/pack |
| [`rlx-wakeword-core`](crates/rlx-wakeword-core/README.md) | `no_std` mel + CNN + ternary |
| [`rlx-wakeword-wasm`](crates/rlx-wakeword-wasm/README.md) | Node / browser / Web Worker (not on crates.io) |
| [`rlx-wake`](crates/rlx-wake/README.md) | Shared API / train CNN / ternary helpers |
| [`rlx-openwakeword`](crates/rlx-openwakeword/README.md) | Mel → embedding → phrase head |
| [`rlx-nanowakeword`](crates/rlx-nanowakeword/README.md) | Native CNN / lite |
| [`rlx-porcupine`](crates/rlx-porcupine/README.md) | Porcupine-style wake CNN |
| [`rlx-voxrt`](crates/rlx-voxrt/README.md) | VoxRT-style wake CNN |

```sh
just wakeword-train -- --synth --phrase hey_rlx --out-dir /tmp/wake_bundle --ternary
just features=all-backends wakeword-demo -- --wav clip.wav --bundle /tmp/wake_bundle --hop-ms 40 --device all
just wakeword-multi-bench
just wakeword-wasm-bench
just test-wakeword
just test-wakeword-backends
```

## Build and test

```sh
just check
just test
just build

cargo build -p rlx-models
cargo test  -p rlx-models
cargo test  -p rlx-models --features parity-candle
```

burnembed (`/Users/Shared/burnembed`) re-exports `rlx_models::embed` with `--features rlx`.

### Publishing (crates.io)

Prerequisite: upstream **`rlx*`** **0.2.14** published from the [RLX](https://github.com/MIT-RLX/rlx) repo. Verify registry resolution without a local patch:

```sh
rm -f .cargo/config.toml
cargo tree -p rlx-models-core -i rlx-runtime   # expect v0.2.14, no path source
```

Pre-flight (same gates as `scripts/publish.sh` / `just lint` / CI):

```sh
just fmt-check
just lint
cargo test --workspace --release --exclude kitten_tts_mini_rlx
```

Local auto-gate on commit: `just install-hooks` (sets `core.hooksPath=.githooks`). Cursor agents also rustfmt on edit and re-run the scoped gate on turn end.

Dry-run packaging (no upload):

```sh
just publish-list
just publish-dry-run
```

Real publish (tier order — `rlx-models-core` first, facade `rlx-models` last; needs `cargo login` or `CARGO_REGISTRY_TOKEN`):

```sh
./scripts/publish.sh --yes
```

See `scripts/publish.sh --help` for `--start-crate`, rate limits, and resume options.

### Real-weight integration tests

```sh
just fetch-real-weights              # downloads ~1.5 GB of small Q4_K_M GGUFs (idempotent)
just test-real-weights               # config + compat + chat-template across 4 families (~2 s/suite)
just test-real-weights-inference     # adds end-to-end forward inference (slow on CPU)
just test-net-hf                     # live HuggingFace Hub compat check (RLX_NET_TESTS=1)
```

Covers SmolLM2 135M (`llama`), Qwen 2.5 0.5B (`qwen2`), Gemma 3 270M (`gemma3` — currently `KnownUnimplemented(M2)`), and Llama 3.2 1B (`llama` + Llama-3 RoPE scaling). The inference path verifies the full `Llama32Runner`/`Qwen3Runner` packed-decode pipeline against real downloaded GGUFs.

### Auto-dispatch + compatibility check

```sh
rlx-run check <path-or-hf-repo>      # `SUPPORTED`, `KnownUnimplemented(<milestone>)`, `MissingMetadata`, or `Unknown`
rlx-run check <path> --json          # machine-readable verdict
rlx-run auto <weights> [args...]     # sniffs arch, dispatches to the right runner
```

Programmatic: [`rlx_models::run::check_path`](crates/rlx-cli/src/compat.rs), [`check_hf_repo`](crates/rlx-cli/src/compat.rs) (requires `compat-net` feature), [`auto_dispatch`](crates/rlx-cli/src/auto_dispatch.rs), [`ChatTemplate::from_gguf`](crates/rlx-cli/src/chat.rs). Implements the same load-time-field predicate llama.cpp uses (`general.architecture` + `<arch>.context_length` + `<arch>.embedding_length` + `<arch>.block_count` + `tokenizer.ggml.{model,tokens}`).

## Status

### Weights on Hugging Face (RLX packs)

Local staging lives under gitignored `weights/` (`vision/`, `lm/`, `tts/`, `asr/`, `audio/`). Each publishable directory maps to one Hub model under [`eugenehp`](https://huggingface.co/eugenehp).

| Hub primary | Examples |
|-------------|---------|
| Nested / outer **`.rlxp`** | `eugenehp/tiny-tts-rlx`, `soprano`, `moss-nano`, `rlx-tts`, `rlx-asr` |
| ONNX / safetensors / GGUF | Other TTS and vision redistribs (kokoro, chatterbox, bioclip, …) |

Native RLXP Hub repos ship **no** pack-time `.onnx`. Bake locally, then:

```bash
python3 scripts/prepare_weights_hf.py --cleanup          # cards / LICENSE / .gitattributes
python3 scripts/publish_weights_hf.py --dry-run          # list jobs
python3 scripts/publish_weights_hf.py --only moss-nano,soprano,tiny-tts-rlx,rlx-tts,rlx-asr
```

Asset helpers: [`rlx-assets`](crates/rlx-assets/README.md) (`native-pack` feature).

### Weights and parity

**rlx GGUF** = this repo can load `.gguf` through `GgufLoader` and the family runner. **GGUF on HF** = models on the Hub tagged `library:gguf` (counts are approximate; use the search link to browse).

| family | safetensors | rlx GGUF | GGUF on Hugging Face | parity |
|---|---|---|---|---|
| `bert`, `nomic`, `vision` (`embed`) | yes | yes (`bert`, `nomic-bert`, …) | **yes** — [minilm](https://huggingface.co/models?library=gguf&search=minilm) (~128), [bge](https://huggingface.co/models?library=gguf&search=bge) (~247), [nomic](https://huggingface.co/models?library=gguf&search=nomic) (~60); e.g. [nomic-embed-text-v1.5-GGUF](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) (`nomic-bert`), [bge-small-en-v1.5-gguf](https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf). Vision embed: no GGUF sibling. | production (safetensors) |
| `dinov2` | yes | yes (`dinov2`; F32 drain or K-quant/Q4_0/Q8_0 packed `DequantMatMul` when quant tensors present) | **no** for `facebook/dinov2-*` — [dinov2](https://huggingface.co/models?library=gguf&search=dinov2) (0). Community converters (dinov2.cpp) use `dinov2` arch; tensor names must match HF/candle keys. | production |
| `sam`, `sam2`, `sam3` | yes | yes (`sam` / `mobile-sam` / `sam2` F32 drain). **SAM3**: F32 drain or K-quant via fused CPU `gguf_matmul` (ViT, text, detector host+IR, seg cross-attn/mask/scoring, 1×1 inst/sem `DequantMatMul` IR); 3×3 pixel conv stays packed at load (one-time dequant cache on host, materialize for tier-1 IR compile) | **SAM1 ViT-H / SAM2**: no official Hub GGUF — [segment+anything](https://huggingface.co/models?library=gguf&search=segment+anything) (0), [sam2.1](https://huggingface.co/models?library=gguf&search=sam2.1) (0). **MobileSAM**: [mobilesam](https://huggingface.co/models?library=gguf&search=mobilesam) (2), e.g. [Acly/MobileSAM-GGUF](https://huggingface.co/Acly/MobileSAM-GGUF) (`mobile-sam`). **SAM3**: [sam3](https://huggingface.co/models?library=gguf&search=sam3) (1) — [rob-laz/sam3-gguf](https://huggingface.co/rob-laz/sam3-gguf) (`sam3`). Beware [TheBloke/SAM-GGUF](https://huggingface.co/TheBloke/SAM-GGUF) — 7B **chat LM** (`llama`), not Segment Anything. | production (encoder + mask path) |
| `qwen3` | yes | yes (Q4_K_M / Q5_K_M / Q6_K) | **yes** — [qwen3](https://huggingface.co/models?library=gguf&search=qwen3) (many); e.g. `unsloth/Qwen3-*-GGUF` | top-1 vs HF (`parity-candle` + weights) |
| `qwen35` | — | yes | **yes** — same hub space; e.g. `unsloth/Qwen3.5-*-GGUF`, `unsloth/Qwen3.6-27B-MTP-GGUF` (VLM: text GGUF + CLIP `mmproj-*.gguf`; hybrid gated-DeltaNet + periodic attention, MTP head; Q3_K_S text generation coherent on CPU/Metal, verified vs llama.cpp; `just fetch-qwen36-27b`) | coherent vs llama.cpp (`QWEN35_GGUF_PATH` / `parity-llama`) |
| `llama32` | yes | yes | **yes** — [llama-3.2](https://huggingface.co/models?library=gguf&search=llama-3.2) (~5k) | vs llama.cpp when `LLAMA32_GGUF_PATH` |
| `minicpm5` | yes | yes (`llama`) | **yes** — [MiniCPM5-1B-GGUF](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) (Q4_K_M / Q8_0 / F16) | vs PyTorch (`minicpm5_parity`); `rlx-minicpm5` 0.2.6 on `rlx-llama32` 0.2.6; GGUF packed CPU/Metal |
| `tinyllama` | yes | yes (`llama`) | **yes** — [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) (Q4_K_M / Q8_0 / Q6_K) | real-weight GGUF packed prefill vs CPU (`tinyllama_backend_gguf_check`); wraps `rlx-llama32` |
| `llada2` | yes | — | **preview** — [llada2](https://huggingface.co/models?library=gguf&search=llada2) (1): [LLaDA2.0-mini-preview-GGUF](https://huggingface.co/wsbagnsv1/LLaDA2.0-mini-preview-GGUF) (`llada2`) | vs PyTorch when `LLADA2_MODEL_DIR` |
| `flux2` | yes (BFL / NVFP4 safetensors) | yes (denoiser `.gguf`, `architecture: flux`; K-quant GGUF uses packed `DequantMatMul`; `Flux2Runner` + VAE/TE safetensors) | **yes** — [flux2](https://huggingface.co/models?library=gguf&search=flux2) (~53); e.g. [unsloth/FLUX.2-klein-9B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF), [city96/FLUX.2-dev-gguf](https://huggingface.co/city96/FLUX.2-dev-gguf) | GGUF = denoiser only; VAE + Qwen3 TE still safetensors dirs |
| `vjepa2` | yes | yes (`vjepa2` / `vjepa`, F32 drain) | **no** Hub GGUF yet — [vjepa](https://huggingface.co/models?library=gguf&search=vjepa) (0) | synthetic + optional weight checks |
| `wav2vec2-bert` | yes | yes (`w2v-bert` / `wav2vec2`, F32 drain) | **no** for Seamless W2V-BERT — [w2v-bert](https://huggingface.co/models?library=gguf&search=w2v-bert) (0). Classic ASR: [wav2vec2](https://huggingface.co/models?library=gguf&search=wav2vec2) (~7), e.g. `cstr/wav2vec2-*-GGUF` (`wav2vec2` arch; keys may not match W2V-BERT) | vs HF when `RLX_W2V_BERT_DIR` + python reference |

To discover GGUF on the Hub: open [Models → library GGUF](https://huggingface.co/models?library=gguf) and add a **search term** matching the family (`qwen3`, `bge`, `flux2`, …). Check the model card **Architecture** field — many repos share a name but are unrelated LMs.

### Backends

Every model family targets the same standard backends: **CPU, Metal, MLX, CUDA, ROCm, WGPU (`gpu`), Vulkan**. SAM also accepts **`tpu`**. Policy lives in [`rlx_core::device_capabilities`](crates/rlx-core/src/device_capabilities.rs); runners call `validate_standard_device` (or `validate_sam_device`) at build time.

Enable GPU at compile time with matching features on `rlx-models` or any model crate, e.g. `cargo build -p rlx-qwen3 --features all-backends` or `cargo run -p rlx-models --features metal --bin rlx-run -- qwen3 …`. Per-crate binaries (`rlx-qwen3`, `rlx-sam3`, …) expose the same feature names. CLI: `cpu`, `metal`/`mps`, `mlx`, `cuda`, `rocm`/`hip`, `gpu`/`wgpu`, `vulkan`.

Legend: ✅ supported · ⚠️ partial (host fallback or open runtime gap) · ❌ not supported

| family | cpu | metal | mlx | cuda | rocm | wgpu | vulkan | notes |
|---|---|---|---|---|---|---|---|---|
| `embed` (`bert`, `nomic`, `vision`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | [`RlxEmbed::from_dir_on`](crates/rlx-embed/src/runtime.rs); `from_dir` defaults to CPU |
| `dinov2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | [`DinoV2Runner`](crates/rlx-dinov2/src/runner.rs) `--device` |
| `uni2` | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ | [`Uni2Runner`](crates/rlx-uni2/src/runner.rs) `--device`; **CPU/Metal/MLX/wgpu bit-exact vs timm (cosine 1.0)**; wgpu correct via auto no-reuse workaround (open rlx-wgpu executor reuse bug); CUDA/ROCm/Vulkan untested (no local hw) |
| `bioclip2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | [`BioClip2Runner`](crates/rlx-bioclip2/src/runner.rs) `--device`; 100% open_clip parity verified on cpu/metal/mlx/wgpu |
| `sam`, `sam2`, `sam3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | SAM v1 also accepts `tpu`; CPU/Metal/MLX most exercised in CI |
| `qwen3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | packed GGUF: CPU + Metal native; MLX/wgpu/CUDA prefill via CPU path (`rlx_core::packed_gguf_*`); MTP decode not wired |
| `qwen35` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `--device` on all backends; some ops use host GDN/dequant on GPU; MoE offload may keep experts on host |
| `llama32` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `rlx-llama32` 0.2.6: Metal decode guard + packed GGUF helpers; same packed rules as Qwen3 |
| `minicpm5` | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Wraps `rlx-llama32`; safetensors decode on CPU/Metal; GGUF `--packed` parity on CPU/Metal (MLX/wgpu tests use CPU prefill path) |
| `tinyllama` | ✅ | ✅ | ✅ | ⚠️ | — | ⚠️ | — | Wraps `rlx-llama32`; GGUF `--packed` parity CPU/Metal/MLX; CUDA prefill on CPU; wgpu buffer limits on some GPUs |
| `llada2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | MoE predictive expert offload on all standard backends (GPU uses resident experts + host fallback) |
| `flux2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Full pipeline; text encoder compiled on Metal/MLX by default, host once on CUDA/ROCm/WGPU/Vulkan |
| `vjepa2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Runner `--device` |
| `wav2vec2-bert` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | [`Wav2Vec2BertRunner`](crates/rlx-wav2vec2-bert/src/runner.rs) `--device` |

Multi-tenant serving (paged KV, continuous batching) lives in `rlx_runtime::paged_kv`; `qwen3::generator` is single-stream.

## Gotchas

- Safetensors names ≠ IR `Param` names — `weight_map.rs` renames; GGUF uses `GgufLoader`.
- **GGUF LMs** (`qwen3`, `qwen35`, `llama32`, `minicpm5`, `tinyllama`): pass a `.gguf` file or a directory with one `.gguf` / `model.safetensors`. Wrong-family files get a redirect (`rlx_core::assert_gguf_family`). Shared helpers: `resolve_weights_file`, `WeightFormat::resolve`, `open_loader_resolved`. MiniCPM5 and TinyLlama expect `general.architecture = llama` and HF `model_type = llama`; TinyLlama also checks 2048×22 dims.
- **Packed GGUF prefill** (`--packed`, K-quant): use `rlx_core::{packed_gguf_compile_guard, compile_options_for_packed_gguf_prefill_with_profile, packed_gguf_execution_device}` in `rlx-llama32`, `rlx-qwen3`, `rlx-gemma`, and `rlx-minicpm5`. Metal sets `RLX_DISABLE_MPSGRAPH=1` during compile; MLX uses `RLX_MLX_MODE=lazy` (host GGUF dequant); wgpu/CUDA/ROCm disable fusion and may run prefill on CPU until upstream GPU parity.
- **GGUF elsewhere on HF** (embed, FLUX, SAM3, …) does not imply rlx support — see [Weights and parity](#weights-and-parity) column *GGUF on Hugging Face*.
- **GGUF shapes** are innermost-first labels; byte layout matches safetensors row-major — do not transpose in `take`.
- Unsupported GGUF quants (Q1_0, Q2_K, IQ*, …) error cleanly.
- **27B GGUF on Mac**: F32 dequant ≈ 108 GB; needs Metal `Op::DequantMatMul` to stay packed (~13.5 GB).
- Pooling in `embed::pooling`.
- New arch: new crate under `crates/`, facade hook, optional parity test.

## Per-crate READMEs

Model-specific runbooks live next to each crate. Agent quick reference: [AGENTS.md](AGENTS.md).

| Crate | README |
|-------|--------|
| `rlx-fft` | [crates/rlx-fft/README.md](crates/rlx-fft/README.md) |
| `rlx-qwen3-tts` | [crates/rlx-qwen3-tts/README.md](crates/rlx-qwen3-tts/README.md) |
| `rlx-kittentts` | [crates/rlx-kittentts/README.md](crates/rlx-kittentts/README.md) |
| `rlx-tts` | [crates/rlx-tts/README.md](crates/rlx-tts/README.md) |
| `rlx-orpheus` | [crates/rlx-orpheus/README.md](crates/rlx-orpheus/README.md) |
| `rlx-kyutai-tts` | [crates/rlx-kyutai-tts/README.md](crates/rlx-kyutai-tts/README.md) |
| `rlx-pocket-tts` | [crates/rlx-pocket-tts/README.md](crates/rlx-pocket-tts/README.md) |
| `rlx-inflect-nano` | [crates/rlx-inflect-nano/README.md](crates/rlx-inflect-nano/README.md) |
| `kitten_tts_mini_rlx` | [crates/kitten_tts_mini_rlx/README.md](crates/kitten_tts_mini_rlx/README.md) |
| `rlx-gemma` | [crates/rlx-gemma/README.md](crates/rlx-gemma/README.md) |
| `rlx-minicpm5` | [crates/rlx-minicpm5/README.md](crates/rlx-minicpm5/README.md) |
| `rlx-tinyllama` | [crates/rlx-tinyllama/README.md](crates/rlx-tinyllama/README.md) |
| `rlx-llama32` | [crates/rlx-llama32/README.md](crates/rlx-llama32/README.md) |
| `rlx-locateanything` | [crates/rlx-locateanything/README.md](crates/rlx-locateanything/README.md) |
| `rlx-fara` | [crates/rlx-fara/README.md](crates/rlx-fara/README.md) |
| `rlx-ppocrv6` | [crates/rlx-ppocrv6/README.md](crates/rlx-ppocrv6/README.md) |
| `rlx-trellis2` | [crates/rlx-trellis2/README.md](crates/rlx-trellis2/README.md) |
| `rlx-vad` | [crates/rlx-vad/README.md](crates/rlx-vad/README.md) |
| `rlx-wake` | [crates/rlx-wake/README.md](crates/rlx-wake/README.md) |
| `rlx-wakeword-core` | [crates/rlx-wakeword-core/README.md](crates/rlx-wakeword-core/README.md) |
| `rlx-wakeword` | [crates/rlx-wakeword/README.md](crates/rlx-wakeword/README.md) |
| `rlx-wakeword-wasm` | [crates/rlx-wakeword-wasm/README.md](crates/rlx-wakeword-wasm/README.md) |
| `rlx-openwakeword` | [crates/rlx-openwakeword/README.md](crates/rlx-openwakeword/README.md) |
| `rlx-nanowakeword` | [crates/rlx-nanowakeword/README.md](crates/rlx-nanowakeword/README.md) |
| `rlx-porcupine` | [crates/rlx-porcupine/README.md](crates/rlx-porcupine/README.md) |
| `rlx-voxrt` | [crates/rlx-voxrt/README.md](crates/rlx-voxrt/README.md) |
| `rlx-mamba` | [crates/rlx-mamba/README.md](crates/rlx-mamba/README.md) |
| `rlx-ssm` | [crates/rlx-ssm/README.md](crates/rlx-ssm/README.md) |
| `rlx-models-core` (`rlx-core`) | [crates/rlx-models-core/README.md](crates/rlx-models-core/README.md) |
| `rlx-clinicalbert` | [crates/rlx-clinicalbert/README.md](crates/rlx-clinicalbert/README.md) |
| `rlx-onnx-import` | [crates/rlx-onnx-import/README.md](crates/rlx-onnx-import/README.md) |
| `rlx-onnx-decompose` | [crates/rlx-onnx-decompose/README.md](crates/rlx-onnx-decompose/README.md) |
| Voxtral TTS training | [docker/voxtral-tts/README.md](docker/voxtral-tts/README.md) |

Crates without a dedicated README are documented in [What's here](#whats-here) and the facade examples under `crates/rlx-models/examples/`.

## License

GPL-3.0-only.
