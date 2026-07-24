# rlx-asr

Native [RLX](https://github.com/MIT-RLX/rlx) streaming Conformer ASR:

```text
audio → 80-mel frontend → energy VAD → Conformer encoder → CTC beam → native AED → text
```

## Weights — single `.rlxp`

```text
weights/asr/
  model.rlxp       # primary Hub pack (units, silence fbank, etiquette, TP, encoder, …)
  model.gguf       # legacy local pack (still loads)
  manifest.json    # optional listing
```

Hub: [`eugenehp/rlx-asr`](https://huggingface.co/eugenehp/rlx-asr) ships **`model.rlxp` only**.

```bash
just fetch-rlx-asr                   # → weights/asr/model.rlxp
```

Pack from a private source tree:

```bash
export RLX_ASR_PACK_SRC=.cache/asr   # or your dump
cargo run -p rlx-asr --release --bin rlx-asr-pack-gguf -- --rlxp
# → weights/asr/model.rlxp
cargo run -p rlx-asr --release --bin rlx-asr-pack-gguf --   # legacy GGUF
just asr-weights-sync                # prune to pack-only + manifest
```

Env: `RLX_ASR_DIR` (default `weights/asr`), `RLX_ASR_GGUF`, `RLX_ASR_TIMING=1`.

## Run

```bash
just asr-check
cargo run -p rlx-asr --release -- transcribe --wav clip.wav
# Folded CTC e2e (Python, same pack):
just asr-e2e-native -- --wav clip.wav
```

Facade: `rlx-models` feature `streaming-asr` → `rlx_models::streaming_asr`.

## Status

| Stage | Rust | Notes |
|-------|------|--------|
| Frontend / VAD / CTC beam / Hammer FSTs / AED | yes | Loads from `model.rlxp` (or legacy GGUF) |
| Folded encoder → CTC | Python e2e | `tools/e2e_native_whole.py` |
| Native Conformer graph | stub | Shaped outputs for pipeline wiring |

## Features / backends

Backend flags mirror other model crates (`metal`, `mlx`, `cuda`, `rocm`, `gpu`,
`vulkan`, `coreml`, `all-backends`, `apple-silicon`).

| Host | Typical devices |
|------|-----------------|
| Apple Silicon | CPU, Metal, MLX, wgpu, CoreML |
| Windows / Linux + NVIDIA | CPU, CUDA, wgpu, Vulkan |
