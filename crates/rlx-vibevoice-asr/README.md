# rlx-vibevoice-asr

Native RLX port of Microsoft **VibeVoice-ASR**:

| Variant | Hub | Weights |
|---------|-----|---------|
| **Streaming-7B** | [VibeVoice-ASR-Streaming-7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B) | BF16 safetensors (~17 GB) |
| **BitNet** | [VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) | I8_S VAE + I2_S LM GGUFs |

## Streaming-7B pipeline

```
24 kHz mono (normalize_audio=false for this checkpoint)
  → encode_then_split (default for files): one VAE pass → feature chunks
     or split_then_encode: per-segment VAE (live mic)
  → acoustic ConvNeXt VAE (GELU) → SpeechConnector → [T, 3584]
  → semantic ConvNeXt VAE (GELU) → SpeechConnector → [T, 3584]
  → element-wise sum
  → Qwen2.5-7B KV streaming:
       prompt prefill
       per chunk: [<|object_ref_start|> | feats | <|object_ref_end|>]
       greedy until <|text_chunk_end|>
```

**Speed / accuracy defaults:** acoustic mean (deterministic), VAE graphs cached by
padded length, LM weights dequantized once into RAM, intermediate speech frames
skip `lm_head` (~25× fewer vocab matmuls per chunk), file path uses one encode.

All RLX backends: `cpu`, `metal`, `mlx`, `cuda`, `rocm`, `gpu` (wgpu), `vulkan`, `coreml`/`ane`.

```bash
just features=all-backends test-vibevoice-asr-backends
```

Synthetic VAE encode (BitNet ReLU + Streaming GELU) runs on every available
device and checks CPU agreement (`max|Δ| < 2e-3`). Unavailable backends skip.

## Streaming CLI

```bash
just fetch-vibevoice-asr-streaming
just vibevoice-asr-streaming -- --audio clip.wav --device metal
# or:
cargo run -p rlx-vibevoice-asr --release --features tokenizer,apple-silicon -- \
    --model-dir .cache/vibevoice-asr-streaming-7b \
    --audio clip.wav --device metal --context-info "Microsoft,VibeVoice"
# live-style per-chunk encode:
#   --encode split_then_encode
# stage timing:
#   RLX_VIBEVOICE_ASR_TIMING=1 …
```

Env: `RLX_VIBEVOICE_ASR_STREAMING_DIR` (default `.cache/vibevoice-asr-streaming-7b`),
`RLX_VIBEVOICE_ASR_ENCODE=split` for mic path, `RLX_VIBEVOICE_ASR_TIMING=1` for RTF logs.

## BitNet GGUF CLI

```bash
cargo run -p rlx-vibevoice-asr --features tokenizer --release -- \
    --vae  vibeasr-vae-encoder-i8_s.gguf \
    --lm   vibeasr-lm-i2_s-embed-q6_k.gguf \
    --audio input.wav
```

BitNet path: ReLU FFN in the VAE (I8_S reference), packed `I2_S`→`Q2_0` LM by default (`VIBEASR_DENSE=1` for dense f32).

## Status

- Streaming-7B: safetensors load, GELU VAE, embeds prefill / KV-continue / decode, chunked generate
- BitNet: GGUF load, unit tests; e2e numeric validation still pending real weights
- Backends: `just features=all-backends test-vibevoice-asr-backends` (CPU+Metal+MLX+wgpu+Vulkan+CoreML verified; CUDA/ROCm skip when absent)
- Env-gated Streaming e2e: `RLX_VIBEVOICE_ASR_STREAMING_DIR` + short wav
