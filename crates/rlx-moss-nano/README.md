# rlx-moss-nano

[MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano) — OpenMOSS's
0.1B multilingual **hierarchical autoregressive** codec-LM TTS for RLX
(**Apache-2.0**, 48 kHz stereo, en/zh/ja).

**Default path is native RLX** (ONNX graphs → rlx-ir → compile → run; no ONNX
Runtime at inference). Optional `--features onnx` keeps the ORT reference.

Pipeline:

- **global** 12-layer transformer (`prefill`, re-run on the growing padded sequence)
- fused **local** graph (`fixed_sampled_frame`) → 16 audio-codebook tokens / frame
- **MOSS-Audio-Tokenizer** (`decode_full`) → 48 kHz stereo

Voice cloning uses 18 builtin voices (pre-computed reference codes in the
manifest — no reference audio needed). The local sampler is pinned to CPU so the
code stream (hence the waveform) stays bit-identical across backends.

## Backend status (fox pangram, Trump, Whisper ≥5/6)

| backend | status | notes |
|---------|--------|-------|
| **CPU** | ✅ | reference |
| **Metal** | ✅ | prefill + sampler on CPU; codec on Metal |
| **MLX** | ✅ | sampler on CPU; cos 1.0 vs CPU |
| **wgpu** | ✅ | prefill + sampler on CPU; codec on wgpu |
| **CUDA** | ✅ | msi: prefill on CUDA by default; codec on CPU (`RLX_MOSS_CODEC_DEVICE=gpu` to force) |

## Setup

```bash
# LM (prefill / local + external .data + tokenizer + manifest)
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX --local-dir weights/tts/moss-nano
# codec → weights/tts/moss-nano/codec/
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX --local-dir weights/tts/moss-nano/codec
# BPE tokenizer.model → tokenizer.json (pure-Rust loadable)
python crates/rlx-moss-nano/scripts/convert_tokenizer.py weights/tts/moss-nano
```

## Usage

```bash
just moss-nano
just moss-nano-whisper
just moss-nano-backends
```

```bash
cargo run -p rlx-moss-nano --release --features apple-silicon -- \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice Trump --device metal --out /tmp/moss.wav
```

`--seed`, `--max-frames`, `--device cpu|metal|mlx|cuda|gpu`, `--list-voices`.

## Notes

- **Tokenizer**: convert `tokenizer.model` (SentencePiece BPE) to `tokenizer.json`
  for the pure-Rust `tokenizers` crate. Do **not** link the C++ `sentencepiece`
  crate when using the optional `onnx` feature — it clashes with ORT's protobuf.
- Override sampler device with `RLX_MOSS_SAMPLER_DEVICE=gpu` (may break cross-backend
  token identity).
- Output PCM is pause-polished by default: lead/trail trim + internal holes
  ≥150 ms clamped to 100 ms (`--max-pause-ms`, `--no-tighten`).
