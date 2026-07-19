# StyleTTS2 (Kokoro-82M)

`rlx-styletts2` runs Kokoro-82M. **Default path is native** graph-split RLX
(decoder on the requested device; duration/prosody encoder on ORT CPU unless
`RLX_KOKORO_NATIVE_ENC=1`). Force the monolithic onnxruntime graph with
`RLX_STYLETTS2_ORT=1`.

## Backend status (fox pangram, Whisper ≥5/6)

| backend | status | notes |
|---------|--------|-------|
| **CPU** | ✅ 6/6 | hybrid ORT enc + RLX dec; full-native enc also 6/6 |
| **Metal** | ✅ 6/6 | cos≈0.99 vs CPU (`disable_mpsgraph` on decoder) |
| **MLX** | ✅ 6/6 | cos≈0.998; Lazy fallback for large decoder graph |
| **wgpu** | ✅ 6/6 | cos≈0.99 vs CPU |
| **CoreML** | ✅ | `--device coreml` / `ane`; `RLX_COREML_UNITS=gpu` via `resolve_tts_device` |
| **Vulkan** | ✅ wired | `--device vulkan` (`all-backends`); availability host-dependent |

## Backends

| Mode | Env | Executes on |
|------|-----|-------------|
| Native (default) | — | RLX decoder on `--device`; ORT CPU encoder (or RLX with `RLX_KOKORO_NATIVE_ENC=1`) |
| Monolithic ORT | `RLX_STYLETTS2_ORT=1` | onnxruntime EP for the requested device |
| Legacy ORT | `RLX_STYLETTS2_NATIVE=0` | same as `RLX_STYLETTS2_ORT=1` |

## Setup

```bash
just fetch-kokoro
# once (native path): python crates/rlx-kokoro/scripts/split_kokoro.py weights/tts/kokoro-82m
```

## CLI / validation

```bash
just styletts2
just styletts2-whisper
just styletts2-backends
RLX_KOKORO_NATIVE_ENC=1 just styletts2-whisper   # full RLX encoder
```

## See also

- [`rlx-kokoro`](../rlx-kokoro) — `Kokoro` / `NativeKokoro`
- [StyleTTS2 paper](https://arxiv.org/abs/2306.07691)
