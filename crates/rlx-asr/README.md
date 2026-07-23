# rlx-asr

Native [RLX](https://github.com/MIT-RLX/rlx) streaming Conformer ASR:

```text
audio → 80-mel frontend → energy VAD → Conformer encoder → CTC beam → native AED → text
```

## Weights — single GGUF

```text
weights/asr/
  model.gguf       # units, silence fbank, etiquette, TP FSTs, encoder, decoder, codebook, LS
  manifest.json    # optional listing
```

Pack from a private source tree (not published):

```bash
export RLX_ASR_PACK_SRC=.cache/asr   # or your dump
just asr-pack-gguf                   # → weights/asr/model.gguf
just asr-weights-sync                # prune to GGUF-only + manifest
```

Env: `RLX_ASR_DIR` (default `weights/asr`), `RLX_ASR_GGUF`, `RLX_ASR_TIMING=1`.

## Run

```bash
just asr-check
cargo run -p rlx-asr --release -- transcribe --wav clip.wav
# Folded CTC e2e (Python, same GGUF):
just asr-e2e-native -- --wav clip.wav
```

Facade: `rlx-models` feature `streaming-asr` → `rlx_models::streaming_asr`.

## Status

| Stage | Rust | Notes |
|-------|------|--------|
| Frontend / VAD / CTC beam / Hammer FSTs / AED | yes | Loads from `model.gguf` |
| Folded encoder → CTC | Python e2e | `tools/e2e_native_whole.py` |
| Native Conformer graph | stub | Shaped outputs for pipeline wiring |

## Features

Backend flags mirror other model crates (`metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, `coreml`, `all-backends`, `apple-silicon`).
