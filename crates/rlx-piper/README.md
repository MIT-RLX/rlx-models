# rlx-piper

[Piper](https://github.com/OHF-Voice/piper1-gpl) VITS text-to-speech for RLX —
small, fast, single-ONNX voices with an espeak-ng phoneme frontend.

> The Piper **voices** ([`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices))
> are **MIT**-licensed; only the reference Python runtime is GPL.

Each voice is a single VITS ONNX (`input` / `input_lengths` / `scales` →
`output`) plus a `<voice>.onnx.json` config (sample rate, espeak voice,
`phoneme_id_map`, inference scales). This crate reuses the bundled espeak-ng
phonemizer + ONNX Runtime EP selector from `rlx-kittentts`.

## Quick start

```bash
# Download a voice (e.g. en_US-lessac-medium) into weights/tts/piper/
#   <voice>.onnx and <voice>.onnx.json

cargo run -p rlx-piper --bin rlx-piper -- \
    --text "The quick brown fox jumps over the lazy dog." --out out.wav
```

`--length <F>` (>1 slower), `--device cpu|metal|mlx|cuda|gpu`, `--data <dir>`.

## Tokenization

Text → espeak-ng phonemes → `phoneme_id_map`, wrapped `^ … $` with a `_` pad
after every phoneme (Piper's convention).

## Backends

Runs the VITS ONNX on ONNX Runtime (CPU, plus CoreML / CUDA / DirectML via
`metal`/`mlx`/`cuda`/`gpu`).

## Note

Uses the bundled espeak-ng, which can differ slightly from Piper's reference
espeak build, so a few vowels may shift; output stays intelligible.
