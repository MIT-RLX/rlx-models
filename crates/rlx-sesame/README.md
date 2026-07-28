# rlx-sesame — Sesame CSM-1B

Conversational TTS ([`sesame/csm-1b`](https://huggingface.co/sesame/csm-1b), Apache-2.0)
on RLX: **Llama-3.2-1B backbone** (16×2048) predicts the first Mimi codebook;
a **4-layer depth decoder** (1024-d) fills the remaining 31 codebooks; **Kyutai
Mimi** decodes to 24 kHz mono.

## Status

| stage | status |
|-------|--------|
| HF transformers weights (`model.safetensors`) | ✅ load (skip embedded codec) |
| Llama-3.2 tokenizer + `[speaker]text` packing | ✅ |
| Eager backbone + depth AR | ✅ CPU |
| Mimi encode/decode via `rlx-mimi` | ✅ `--device` |
| Context WAV continuity | ✅ `--context` |
| Compiled Llama32Flow backbone | follow-on |

Ungated mirror for fetch: [`unsloth/csm-1b`](https://huggingface.co/unsloth/csm-1b)
(same transformers layout as gated `sesame/csm-1b`).

## Setup

```bash
just fetch-sesame   # → weights/tts/sesame/
just fetch-mimi     # → .cache/mimi/
```

## Run

```bash
just sesame TEXT="The quick brown fox jumps over the lazy dog." DEVICE=cpu

cargo run -p rlx-sesame --release -- \
  --model-dir weights/tts/sesame \
  --mimi-dir .cache/mimi \
  --text "Hello from Sesame." \
  --device metal \
  --output /tmp/sesame.wav
```

LM AR is eager CPU in this arc; Mimi runs on `--device` (`cpu` / `metal` / `mlx` / `cuda` / …).

## Whisper check

```bash
just sesame-whisper
just sesame-backends   # Mimi on cpu/metal/mlx/gpu/… + Whisper ≥5/6
```

Needs Whisper Tiny under `.cache/whisper-tiny` and CSM + Mimi weights.
The harness forces Whisper `language=en` (Tiny otherwise mis-detects language on CSM audio).

Validated: fox seed 42 → Whisper **6/6**; long paragraph seed 42 → **15/15**.
Cross-backend Mimi: `just sesame-backends` / `just sesame-backends-long`.
CUDA on the NVIDIA box (`just sesame-validate-cuda`): fox cpu+cuda both **6/6**, cos **1.000**,
Mimi decode ~596 ms CUDA vs ~3103 ms CPU (LM frames cached on CPU); long **15/15**.

## Architecture

```text
Text (+ optional context wav)
  → Llama BPE frames / Mimi encode
  → sum(audio codebook embeds + text embed) per position
  → Backbone 16× Llama (GQA 32/8, RoPE θ=500k, llama3 scale)
  → c0 = lm_head(h)
  → Depth 4× Llama: codes 1..31
  → Mimi decode @ 24 kHz
```

## Library

```rust
use rlx_sesame::{GenerateOpts, SesameSession};
use rlx_runtime::Device;

let mut session = SesameSession::open_on(
    "weights/tts/sesame",
    ".cache/mimi",
    Device::Cpu,
)?;
let out = session.synthesize("Hello.", &GenerateOpts::default())?;
```

## License

Apache-2.0 (model + this crate).
