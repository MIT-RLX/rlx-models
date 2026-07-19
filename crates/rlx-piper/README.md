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

Default **native** path needs `rlx-split/` beside the voice ONNX (one-time:
`python crates/rlx-piper/scripts/split_piper.py …`). For ONNX Runtime instead,
rebuild with `--features onnx`.

`--length <F>` (>1 slower), `--device cpu|metal|mlx|cuda|gpu`, `--data <dir>`.

## Tokenization

Text → espeak-ng phonemes → `phoneme_id_map`, wrapped `^ … $` with a `_` pad
after every phoneme (Piper's convention).

## Backends

| Feature | Path | Devices |
|---------|------|---------|
| **`native`** (default) | **native RLX (ort-free)** | **cpu / metal / mlx / wgpu** |
| **`coreml`** | **native RLX (ort-free)** | **CoreML / Neural Engine (`--device ane`)** |
| `onnx` | ONNX Runtime (optional) | CPU (CoreML/CUDA/DirectML unstable for this VITS graph) |

All five native Apple backends verified at cross-backend parity (cosine 1.0 vs CPU;
MLX bit-identical) and whisper round-trip validated end-to-end. **CUDA** (msi):
cosine **1.000** vs CPU, RTF ≈11.5× on fox; Whisper coverage can be low on the
bundled espeak path (same on CPU — not a CUDA divergence).

### Native rlx-ir multi-backend (`native`)

The `native` feature runs piper entirely on the RLX compiler — no ONNX Runtime.
The monolithic VITS graph is **graph-split** into two fixed-shape subgraphs, and
the **stochastic duration predictor** (a rational-quadratic-spline coupling flow
whose boolean-mask indexing no static-shape importer can rank) is reimplemented
directly in Rust (`src/sdp.rs`) — where the spline's inside/outside split is a
trivial per-element branch. Validated against onnxruntime (conditioning path
bit-exact, durations within ±1 at rare bin-boundary phonemes) and whisper-checked
end-to-end.

```text
enc_p.onnx   [input, input_lengths] → m_p [1,192,T], logs_p [1,192,T], dp_in [1,192,T]
  ── Rust StochasticDurationPredictor (sdp.rs): dp_in → durations[T] ──
  ── Rust length regulator + z_p = m_p' + noise·exp(logs_p')·noise_scale ──
flow_dec.onnx [z_p [1,192,T'], y_mask [1,1,T']] → waveform [1,1,1,T'·hop]
```

```bash
# one-time: split the monolithic voice into the native bundle (rlx-split/)
python crates/rlx-piper/scripts/split_piper.py \
  weights/tts/piper/en_US-lessac-medium.onnx weights/tts/piper/rlx-split

cargo run -p rlx-piper --no-default-features --features native,espeak \
  --example native_synthesize -- --dir weights/tts/piper \
  --text "Hello from native piper." --out out.wav [--device cpu|metal|mlx|gpu]
```

`NativeVits::load(dir, device)` is a drop-in for the ort `Piper` with the same
`synthesize` / `synthesize_phonemes` entry points.

## Note

Uses the bundled espeak-ng, which can differ slightly from Piper's reference
espeak build, so a few vowels may shift; output stays intelligible.
