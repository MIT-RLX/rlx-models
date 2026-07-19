# rlx-chatterbox

[ChatterBox](https://github.com/resemble-ai/chatterbox) — Resemble AI's
zero-shot voice-cloning TTS for RLX (**MIT**, 24 kHz).

A 0.5B-Llama **T3** (text→speech-token) backbone + **S3Gen** flow vocoder.
**Default path is native RLX** (ONNX graphs → rlx-ir → compile → run; no ONNX
Runtime at inference). Optional `--features onnx` keeps the ORT reference.

## Backend status (fox pangram, greedy, Whisper ≥5/6)

| backend | status | notes |
|---------|--------|-------|
| **CPU** | ✅ 6/6 | reference |
| **MLX** | ✅ 6/6 | cos 1.0 vs CPU; speech_encoder on CPU |
| **wgpu** | ✅ 6/6 | cos 1.0 vs CPU; speech_encoder on CPU |
| **Metal** | ✅ | hand-authored T3 LM (ONNX LM zeros on Metal); S3Gen on CPU |
| **CUDA** | ✅ 6/6 | msi: cos 1.000 vs self; ~8.8 s fox; `dit`/`exec` Cuda |

## Setup

```bash
huggingface-cli download synath/chatterbox-ONNX --local-dir weights/tts/chatterbox
# or: just fetch-chatterbox
```

Uses the **fp16** language-model variant (`onnx/language_model_fp16.onnx` +
`.onnx_data`). Metal uses `native/t3_lm.safetensors` automatically.

## Usage

```bash
just chatterbox
just chatterbox-whisper
just chatterbox-backends
```

`--exaggeration` (0.5), `--temperature` (0.8), `--seed`, `--device`, `--greedy`.

## Notes

- Speech-encoder import needs scalar **initializer** Gather indices to drop the
  gathered axis (fixed in `rlx-onnx-import`).
- `speech_encoder` runs on CPU when the session device is MLX/Metal/wgpu (one-shot
  conditioning); MLX/wgpu keep embed + LM + S3Gen on-device.
- Metal keeps S3Gen/HiFT on CPU and drives T3 via the hand-authored `rlx-llama32`
  graph (`native/t3_lm.safetensors`). Override with `RLX_CB_ONNX_LM=1` /
  `RLX_CB_NATIVE_LM=1` / `RLX_CB_NATIVE_LM_KV=1`.
- Output PCM is onset-polished (drop HiFT startup click + leading hole, short fade-in).
