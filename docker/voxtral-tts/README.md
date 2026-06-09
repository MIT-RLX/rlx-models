# Voxtral-4B-TTS Docker helpers

Native inference, tokenization, and voice prep live in **`rlx-voxtral-tts`** (Rust only).

Docker images here are **optional** — used only for **vLLM-Omni parity export** (`export-codes`, full wav reference).

## Native workflow (no Docker, no Python)

```bash
just fetch-voxtral-tts
export RLX_VOXTRAL_TTS_DIR=.cache/voxtral/Voxtral-4B-TTS-2603

just voxtral-tts-prepare-voices
just voxtral-tts -- --model-dir $RLX_VOXTRAL_TTS_DIR \
  --text "Hello world" --voice neutral_female -o out.wav
```

Tokenize only:

```bash
just voxtral-tts-tokenize
# or:
just voxtral-tts -- --model-dir $RLX_VOXTRAL_TTS_DIR \
  --text "Hello" --voice neutral_female \
  --write-prompt-tokens .cache/voxtral/tts/prompt_tokens.txt --tokenize-only
```

## Docker images (parity only)

| Image | Dockerfile | Purpose |
|-------|------------|---------|
| `rlx-voxtral-tts-ref:gpu` | `Dockerfile.ref` | vLLM-Omni export codes / full wav (GPU) |

Legacy `Dockerfile.tools` (Python tokenize / convert-voices) is superseded by native Rust CLI flags `--tokenize-only` and `--convert-voices`.

## Parity vs vLLM (codec decode)

```bash
just voxtral-tts-docker-ref-build   # once, needs GPU host
just test-voxtral-tts-parity
```

## Reference-audio cloning

Public checkpoints omit the codec **encoder**; preset `voice_embedding/` voices work out of the box after `just voxtral-tts-prepare-voices`. Reference WAV cloning needs encoder weights in `consolidated.safetensors`.

Native RLX training (no Python):

```bash
export RLX_VOXTRAL_TTS_DIR=.cache/voxtral/Voxtral-4B-TTS-2603
just features=all-backends voxtral-tts-train-encoder -- \
  --model-dir $RLX_VOXTRAL_TTS_DIR --wav-dir ./wavs --out-dir ./out/encoder --device auto
just voxtral-tts-train-encoder -- \
  --model-dir $RLX_VOXTRAL_TTS_DIR --wav-dir ./wavs --out-dir ./out/encoder --device cpu
LOW_VRAM=1 just voxtral-tts-train-encoder-low-vram -- \
  --model-dir $RLX_VOXTRAL_TTS_DIR --wav-dir ./wavs --out-dir ./out/encoder --device auto
just features=all-backends voxtral-tts-train-lora -- \
  --model-dir $RLX_VOXTRAL_TTS_DIR --reference-wav-dir ./wavs --out-dir ./out/lora --device auto
just voxtral-tts-inject-encoder -- \
  --model-dir $RLX_VOXTRAL_TTS_DIR --encoder-weights ./out/encoder/best_encoder.safetensors
```

**Production pipeline** (encoder → full attention LoRA → inject):

```bash
# Manifest with transcripts improves ASR auxiliary loss (optional field per file):
# { "sample_rate": 24000, "files": [{ "path": "a.wav", "duration_sec": 3.2, "transcript": "Hello world" }] }

PRODUCTION=1 just voxtral-tts-train-production -- \
  --model-dir $RLX_VOXTRAL_TTS_DIR --wav-dir ./wavs --out-dir ./out/train --device auto

# Or step-by-step:
PRODUCTION=1 just features=all-backends voxtral-tts-train-all -- \
  --model-dir $RLX_VOXTRAL_TTS_DIR --wav-dir ./wavs --manifest ./wavs/manifest.json

# Optional Whisper CER in encoder ASR loss:
USE_WHISPER_ASR=1 WHISPER_MODEL_DIR=/path/to/whisper-tiny just voxtral-tts-train-encoder -- ...

# Rig on real weights + reference synthesis:
RLX_VOXTRAL_TTS_TRAIN_RIG=1 RLX_VOXTRAL_TTS_REF_WAV=./ref.wav just test-voxtral-tts-train-synthesize-rig
```

`--device` accepts `auto` (first available GPU: cuda → metal → mlx → rocm → wgpu → vulkan), `cpu`, or any name in `cpu|metal|mps|mlx|cuda|rocm|hip|gpu|wgpu|vulkan`. Build GPU backends with `just features=all-backends` or `--features metal|cuda|…`. Backward graphs are lowered for Metal/wgpu/Vulkan (conv + activation backward → primitives); hybrid CPU backward remains as fallback when a backend still lacks an op. Set `RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU=1` to force CPU backward only.

See [`voice_clone.rs`](../crates/rlx-voxtral-tts/src/voice_clone.rs) and [`rlx-voxtral-tts-train`](../crates/rlx-voxtral-tts-train/).

## Manual docker (vLLM reference)

```bash
bash docker/voxtral-tts/run-ref.sh build
RLX_VOXTRAL_TTS_DIR=$RLX_VOXTRAL_TTS_DIR bash docker/voxtral-tts/run-ref.sh export-codes
```
