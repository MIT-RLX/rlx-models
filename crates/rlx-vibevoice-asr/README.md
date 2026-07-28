# rlx-vibevoice-asr

Native RLX port of **[microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet)** —
a CPU-first speech-recognition model that loads Microsoft's shipped GGUFs directly.

## Pipeline

```
24 kHz mono audio ─▶ RMS normalize (−25 dBFS) ─▶ pad to 3200
   ├─▶ acoustic ConvNeXt VAE encoder (I8_S) ─▶ SpeechConnector ─▶ [T, 1536]
   └─▶ semantic ConvNeXt VAE encoder (I8_S) ─▶ SpeechConnector ─▶ [T, 1536]
              (element-wise sum) ─▶ speech features
                     │
   Qwen2.5 chat prompt with N=ceil(samples/3200) <|speech_pad|> rows
                     │  (speech rows overwritten by the summed features)
                     ▼
   BitNet Qwen2-1.5B decoder (I2_S ternary projections + Q6_K token embeddings
   + F16 lm_head, 28 layers, GQA 12/2, RoPE θ=1e6) ─▶ greedy decode ─▶ text
```

## BitNet GGUF support

The shipped quantization types are Microsoft's own, added to the RLX framework
(`rlx-gguf`) during this port:

- **`I2_S` (ggml type 36)** — 2-bit ternary. 128-element blocks (32 bytes each);
  code ∈ {0,1,2} → `(code − 1)·scale`; one per-tensor f32 scale at byte offset
  `n/4`. Verified byte-for-byte against the shipped weights.
- **`I8_S` (ggml type 37)** — symmetric int8, `w = int8·scale`, per-tensor scale
  at offset `n`.

The LM projections are dequantized to f32 on load today (correctness-first);
transcoding them to rlx's packed `TQ2_0` DequantMatMul path (numerically exact
for ternary) is the tracked follow-up for the full BitNet memory win.

## CLI

```bash
cargo run -p rlx-vibevoice-asr --features cpu,tokenizer --release -- \
    --vae  vibeasr-vae-encoder-i8_s.gguf \
    --lm   vibeasr-lm-i2_s-embed-q6_k.gguf \
    --audio input.wav \
    --tokenizer tokenizer.json    # defaults to a sibling of --lm
```

Add `--json` for the segment-JSON prompt (Start/End/Speaker/Content). Backends:
`--features metal|mlx|cuda|gpu|vulkan` (CPU is the reference / fastest-to-set-up path).

## Status

- ✅ Framework `I2_S`/`I8_S` dequant (unit-tested + validated on real GGUF bytes)
- ✅ Audio front-end, prompt/tokenizer, ConvNeXt VAE encoder graph (all backends), LM wiring
- ✅ Compiles; 13 unit tests pass
- ⏳ End-to-end numeric validation against the real 1.77 GB GGUFs
- ⏳ Packed `TQ2_0` LM path (BitNet memory win)
