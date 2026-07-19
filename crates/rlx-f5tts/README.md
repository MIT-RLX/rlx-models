# rlx-f5tts

F5-TTS voice-cloning text-to-speech for RLX — a **flow-matching DiT** (330M,
24 kHz) that clones a voice from a short reference clip + transcript.

> **License:** the F5-TTS weights are **CC-BY-NC-4.0 (non-commercial)**; the code
> is MIT. Use accordingly.

**Default path is native RLX** (ONNX graphs → rlx-ir → compile → run; no ONNX
Runtime). Optional `--features onnx` keeps the ORT reference (`F5Tts` /
`examples/ort_clone`).

Three subgraphs:

1. `preprocess(audio, text_ids, max_duration)` → noise + RoPE + CFG conditioning
2. **NFE denoising loop** (default 32 in `InferOpts`; demos use **16**): DiT does CFG + ODE step internally
3. `decode(latent, ref_len)` → 24 kHz audio (Vocos + ISTFT folded in)

## Backend status (fox pangram, NFE=32, Whisper ≥4/6)

| backend | status | notes |
|---------|--------|-------|
| **CPU** | ✅ | 6/6 fox; speech/HF ≈22 dB (PyTorch-class) |
| **Metal** | ✅ on-device | DiT on Metal; fox 6/6 @ NFE=32 (NFE=8 under-denoises on CPU too) |
| **MLX** | ✅ on-device | fox 6/6, traj cos≈1.0 vs CPU |
| **wgpu** | ✅ on-device | Apple `--device gpu` still routes DiT to Metal by default. True wgpu DiT (`RLX_F5_WGPU_DIT=1`): traj mad≈1e-8 vs CPU, fox **6/6** @ NFE=32 (Transpose(Param) bind fixed in `rlx-wgpu`). |
| **CUDA** | ✅ on-device | RoPE `ScatterNd` via `force_indices_f32` |
| **CoreML** | ✅ on-device | `Device::Ane`; MIL `scatter_nd`; `RLX_COREML_UNITS=gpu` via `resolve_tts_device` |
| **Vulkan** | ✅ wired | `--device vulkan` on `all-backends`; DiT stays on-device |



Matrix WAVs: `tmp/f5tts_wavs/matrix_{cpu,cuda,metal,mlx,wgpu}.wav`.

## Setup

Download the ONNX export ([`huggingfacess/F5-TTS-ONNX`](https://huggingface.co/huggingfacess/F5-TTS-ONNX))
and the vocab ([`SWivid/F5-TTS`](https://huggingface.co/SWivid/F5-TTS) →
`F5TTS_v1_Base/vocab.txt`) into `weights/tts/f5tts/`:
`F5_Preprocess.onnx`, `F5_Transformer.onnx`, `F5_Decode.onnx`, `vocab.txt`.

```bash
just fetch-f5tts
```

## Usage

```bash
just f5tts                 # writes /tmp/f5tts.wav
just f5tts-whisper         # writes + Whisper-gates tmp/f5tts_wavs/validated.wav
just f5tts-backends        # matrix + per-backend WAVs under tmp/f5tts_wavs/
```

`just f5tts-whisper` **fails** unless Whisper hears ≥4/6 fox words. That is the
speech bar — do not treat other files in `tmp/f5tts_wavs/` (Metal noise, NFE=8,
truncated refs) as validated.

```bash
cargo run -p rlx-f5tts --release --features apple-silicon -- \
  --ref-wav crates/rlx-f5tts/tests/fixtures/prompt.wav \
  --ref-text "Hello from Kokoro. This is a test of speech synthesis in Rust." \
  --text "The quick brown fox jumps over the lazy dog." \
  --nfe 16 --device metal --out /tmp/f5tts.wav
```

`--nfe` (default 32; lower = faster), `--speed` (1.0), `--device`.

## Notes

- DiT runs on the requested device. Opt out to CPU with `RLX_F5_CPU_DIT=1`.
  Metal keeps output-ancestor arena pin when the graph has ScatterNd (F5 RoPE)
  even above the 4 GiB MPS cliff — unpinning saved almost nothing and could
  corrupt the ODE. Apple `--device gpu` routes DiT to Metal unless
  `RLX_F5_WGPU_DIT=1`. True wgpu DiT matches Metal/CPU after the
  `Transpose(Param)` bind-window fix in `rlx-wgpu`.

- DiT import fuses `Op::AdaLayerNorm` / `Op::GatedResidual` (uniform ONNX fills
  specialized before fusion; F5 Reshape broadcast peeled like Expand).
- NFE defaults to **32** (export schedule). The DiT loop runs **NFE−1** Euler
  steps (`delta_t` length); demos should keep `--nfe 32`. NFE=16 under-denoises
  this ONNX bundle and sounds hissy.
- Reference audio is silence-trimmed + 50 ms padded, and ref text is normalized
  to end with `". "` — same as official `f5_tts` before duration estimation.
- Decode may keep the full padded length after onnx-import; native post-crops the
  reference hop prefix so WAVs match ORT (gen-only).
- Preprocess noise is stochastic — native↔PyTorch cosine is not a parity metric;
  Whisper fox coverage + speech/HF ratio are.
