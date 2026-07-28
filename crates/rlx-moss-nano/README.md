# rlx-moss-nano

[MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano) — OpenMOSS's
0.1B multilingual **hierarchical autoregressive** codec-LM TTS for RLX
(**Apache-2.0**, 48 kHz stereo, en/zh/ja).

**Distribution:** single [`moss-nano.rlxp`](https://huggingface.co/eugenehp/moss-nano)
with nested native `graphs/*.rlxp` (hot tensors + `graph.json`). Runtime materializes
the outer pack and lowers each subgraph to HIR (no ONNX Runtime, no `.onnx` on Hub).

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
| **CUDA** | ✅ | NVIDIA: prefill on CUDA by default; codec on CPU (`RLX_MOSS_CODEC_DEVICE=gpu` to force) |

## Setup

```bash
just fetch-moss-nano          # eugenehp/moss-nano moss-nano.rlxp
just export-moss-nano-rlxp    # pack from local ONNX tree → nested graphs
```

Loose ONNX (dev / pack source only — not published):

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX --local-dir weights/tts/moss-nano
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX --local-dir weights/tts/moss-nano/codec
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

`--seed`, `--max-frames`, `--device cpu|metal|mlx|cuda|gpu`, `--list-voices`,
`--pack-rlxp` (and legacy `--pack-gguf`).

## Notes

- **Tokenizer**: convert `tokenizer.model` (SentencePiece BPE) to `tokenizer.json`
  for the pure-Rust `tokenizers` crate when packing from loose ONNX.
- Override sampler device with `RLX_MOSS_SAMPLER_DEVICE=gpu` (may break cross-backend
  token identity).
- Output PCM is pause-polished by default: lead/trail trim + internal holes
  ≥150 ms clamped to 100 ms (`--max-pause-ms`, `--no-tighten`).
