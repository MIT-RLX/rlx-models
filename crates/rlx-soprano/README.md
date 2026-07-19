# rlx-soprano — Soprano 1.1

Native (ort-free) Vocos path for [Soprano 1.1](https://github.com/ekwek1/soprano) (~80M, Apache-2.0) via
[`KevinAHM/soprano-1.1-onnx`](https://huggingface.co/KevinAHM/soprano-1.1-onnx):
Qwen3 AR backbone ONNX + 32 kHz decoder ONNX, imported with `rlx-onnx-import`.

## Pipeline

```text
Text → HF tokenizer → backbone AR → 512-d latents → Vocos decoder → PCM 32 kHz
```

Prompt format matches the web demo: `[STOP][TEXT]…[START]`.

## Setup

```bash
just fetch-soprano   # → weights/tts/soprano/
just soprano-demo Hi. cpu
```

Weights under `weights/tts/soprano/`:

- `tokenizer.json`
- `onnx/soprano_backbone_kv_fp32.onnx`
- `onnx/soprano_decoder_fp32.onnx` (+ `.onnx.data`)

## CLI

```bash
cargo run -p rlx-soprano --release --features apple-silicon -- \
  --text "The quick brown fox jumps over the lazy dog." \
  --device metal \
  --output /tmp/soprano.wav
```

## Validation

```bash
# Full native text → Whisper (fox pangram). Brand/name: just soprano-whisper-brand
just soprano-whisper
just soprano-whisper-brand

# ORT backbone AR latents → native Vocos → Whisper (isolates Vocos)
# Needs: python3, onnxruntime, tokenizers, numpy
cargo test -p rlx-soprano --release --test native_whisper_roundtrip -- --nocapture

just soprano-matrix   # CPU / Metal / MLX / wgpu / CoreML when available
```

Short `"Hello from Soprano."` is often transcribed as `"Suprano"` by Whisper-tiny
(ORT and RLX alike). Prefer `"Hello from the Soprano model."`, or rely on the
harness’s light edit-distance match for proper-noun slips.

## Status

| Stage | Status |
|-------|--------|
| Decoder (Vocos) | ORT PCM parity on CPU/Metal; MLX uses native rank-1 Scatter reshape |
| Backbone AR | Full-prefix recompute; greedy tokens match ORT on CPU/Metal/MLX |
| Whisper bar | Full native fox 100%; brand phrase via `soprano-whisper-brand` |

## Notes

- Backbone ONNX→HIR mis-broadcasts when `past_sequence_length > 1`, so the RLX runner recomputes the full prefix with empty past. Sequences with `n>32` bucket to 128.
- Decoder registers ONNX `ScatterElements` at `open()`.
- Reference streaming: [`soprano-web-onnx`](https://github.com/KevinAHM/soprano-web-onnx).

## License

Model: Apache-2.0. This crate follows the workspace license (GPL-3.0).
