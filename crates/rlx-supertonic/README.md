# rlx-supertonic

Supertonic-3 text-to-speech for RLX — a multilingual (31-language)
**flow-matching latent TTS** (~99M params, 44.1 kHz, 10 preset voices).

Four ONNX subgraphs are chained with a small Rust glue that mirrors the
reference `supertonic-py` pipeline:

1. `duration_predictor` → a **scalar total duration** (seconds)
2. `text_encoder` → text embedding `[1, 256, T]`
3. sample `noisy_latent ~ N(0, I)` of shape `[1, 144, L]`, `L = ceil(dur·sr / 3072)`
4. **flow-matching ODE loop** (default 8 steps; the vector estimator integrates
   internally, so the caller just feeds `xt` back with the step index)
5. `vocoder` → 44.1 kHz waveform, trimmed to `dur·sr` samples

The tokenizer is pure char/unicode (`unicode_indexer.json`, `id = table[ord(c)]`)
with `<lang>…</lang>` wrapping — **no phonemizer**, which is how it covers 31
languages cheaply.

## Quick start

```bash
# Download the ONNX bundle + voices (needs the hf-download feature)
cargo run -p rlx-supertonic --features hf-download --bin rlx-supertonic -- --download

# Synthesize
cargo run -p rlx-supertonic --bin rlx-supertonic -- \
    --text "Hello from Supertonic." --voice F1 --lang en --out out.wav

cargo run -p rlx-supertonic --bin rlx-supertonic -- --list-voices
```

Set `RLX_SUPERTONIC_DIR` or pass `--data <dir>` (default `weights/tts/supertonic-3`).

## Model directory layout

```text
config.json
onnx/{duration_predictor,text_encoder,vector_estimator,vocoder}.onnx
onnx/tts.json
onnx/unicode_indexer.json
voice_styles/{F1..F5,M1..M5}.json
```

Weights: [`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3) (OpenRAIL-M).

## Backends

Default path is **native RLX** (import each ONNX subgraph via `rlx-onnx-import`).
Optional `--features onnx` keeps the ORT reference.

Cross-backend matrix (`examples/backend_matrix.rs`): Apple backends bit-identical
vs CPU (whisper 1.00). **CUDA** (NVIDIA): RTF ≈2.4×, cos ≈0.965 vs CPU, whisper 1.00.

## Notes

- `--steps` trades quality for speed (5–12 typical; 8 default).
- `--speed` > 1.0 speaks faster (default 1.05, matching the reference).
- Voice names encode gender: `F*` female, `M*` male.
