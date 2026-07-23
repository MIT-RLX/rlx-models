# StyleTTS2 (Kokoro-82M)

`rlx-styletts2` runs Kokoro-82M over the **ort-free native graph-split RLX
path**: the decoder runs on the requested device and the duration/prosody
encoder runs on CPU (set `RLX_KOKORO_ENC_DEVICE=gpu` to move it onto the
requested device).

## Backend status (fox pangram, Whisper ≥5/6)

| backend | status | notes |
|---------|--------|-------|
| **CPU** | ✅ 6/6 | native RLX encoder + decoder |
| **Metal** | ✅ 6/6 | cos≈0.99 vs CPU (`disable_mpsgraph` on decoder) |
| **MLX** | ✅ 6/6 | cos≈0.998; Lazy fallback for large decoder graph |
| **wgpu** | ✅ 6/6 | cos≈0.99 vs CPU |
| **CoreML** | ✅ | `--device coreml` / `ane`; `RLX_COREML_UNITS=gpu` via `resolve_tts_device` |
| **Vulkan** | ✅ wired | `--device vulkan` (`all-backends`); availability host-dependent |

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
```

## See also

- [`rlx-kokoro`](../rlx-kokoro) — `NativeKokoro`
- [StyleTTS2 paper](https://arxiv.org/abs/2306.07691)
