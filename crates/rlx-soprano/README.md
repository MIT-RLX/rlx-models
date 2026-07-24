# rlx-soprano — Soprano 1.1

Native (ort-free) Vocos path for [Soprano 1.1](https://github.com/ekwek1/soprano) (~80M, Apache-2.0).

**Distribution:** single [`soprano.rlxp`](https://huggingface.co/eugenehp/soprano) with
nested `graphs/*.rlxp` (hot tensors + `graph.json`). Runtime materializes and lowers
to HIR (no ONNX Runtime; **no `.onnx` on Hub**). Pack-time source:
[`KevinAHM/soprano-1.1-onnx`](https://huggingface.co/KevinAHM/soprano-1.1-onnx).

## Pipeline

```text
Text → HF tokenizer → backbone AR → 512-d latents → Vocos decoder → PCM 32 kHz
```

Prompt format matches the web demo: `[STOP][TEXT]…[START]`. Long prompts are
sentence/word-chunked to stay within the sequence limit.

## Setup

```bash
just fetch-soprano            # eugenehp/soprano soprano.rlxp
just export-soprano-rlxp      # pack from local onnx/ sources → nested graphs
just soprano-demo Hi. cpu
```

Packed layout embeds:

- `tokenizer.json`
- `graphs/soprano_backbone_kv_fp32.rlxp`
- `graphs/soprano_decoder_fp32.rlxp`

## CLI

```bash
cargo run -p rlx-soprano --release --features apple-silicon -- \
  --text "The quick brown fox jumps over the lazy dog." \
  --device metal \
  --output /tmp/soprano.wav
```

`--pack-rlxp PATH` packs a loose `--model-dir` into `soprano.rlxp`.
`--pack-gguf PATH` remains for legacy `soprano.gguf`.

## Validation

```bash
just soprano-whisper
just soprano-whisper-brand
cargo test -p rlx-soprano --release --test native_whisper_roundtrip -- --nocapture
just soprano-matrix
```

## Status

| Check | Result |
|-------|--------|
| Whisper fox | Full native ~100% |
| Brand phrase | `just soprano-whisper-brand` |
| Backend matrix | CPU / Metal / MLX / wgpu / CoreML when available |

## Links

- Reference streaming: [`soprano-web-onnx`](https://github.com/KevinAHM/soprano-web-onnx).
