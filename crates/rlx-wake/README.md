# rlx-wake

Shared building blocks for wake-word / keyword activation crates:

- `WakeEngine` streaming trait (`push_pcm` → score / fire)
- 16 kHz mono WAV helpers + device selection (`all-backends`)
- Mel frontend (OWW-compatible hop / `(x/10)+2` norm)
- Compact `WakeCnn` (nanowakeword / porcupine / voxrt weight layout)
- **RLX-only training** (`rlx-wake-train`) — no PyTorch train loops
- **Ternary** helpers (`TernaryOpts`, `WakeCnnWeights::ternarize`) for bake TQ2 / fused kernels

## Product vs shared

| Crate | Role |
|-------|------|
| [`rlx-wakeword`](../rlx-wakeword) | **Product** — event API, multi-phrase train/pack, VAD, speaker-id |
| [`rlx-wakeword-core`](../rlx-wakeword-core) | `no_std` mel + CNN (+ ternary fused path) |
| [`rlx-wakeword-wasm`](../rlx-wakeword-wasm) | WASM / Web Worker bindings (not on crates.io) |
| This crate (`rlx-wake`) | Shared API, train CNN, device/bench helpers |

Prefer `rlx-wakeword` / `TrainBuilder` for new wake phrases. Use `rlx-wake-train` when targeting compat engine weight files directly.

## Custom wake words (train in RLX)

```bash
# Synth corpus (quick check)
cargo run -p rlx-wake --bin rlx-wake-train --release -- synth --out-dir .cache/wake_train/demo

# Train WakeCnn → safetensors
cargo run -p rlx-wake --bin rlx-wake-train --release -- cnn \
  --pos .cache/wake_train/demo/positives \
  --neg .cache/wake_train/demo/negatives \
  --keyword "hey rlx" \
  --out crates/rlx-nanowakeword/weights/hey_rlx.safetensors \
  --epochs 50

just nanowakeword-demo -- --wav clip.wav \
  --weights crates/rlx-nanowakeword/weights/hey_rlx.safetensors \
  --keyword "hey rlx"
```

Product path (bundle + session):

```bash
just wakeword-train -- --synth --phrase hey_rlx --out-dir /tmp/wake_bundle --ternary
just wakeword-demo -- --wav clip.wav --bundle /tmp/wake_bundle --hop-ms 40
```

## Compat engines

| Crate | Role |
|-------|------|
| [`rlx-openwakeword`](../rlx-openwakeword) | Mel → embedding → phrase head |
| [`rlx-nanowakeword`](../rlx-nanowakeword) | Native CNN / lite |
| [`rlx-porcupine`](../rlx-porcupine) | Porcupine-style CNN |
| [`rlx-voxrt`](../rlx-voxrt) | VoxRT-style CNN |

ONNX Runtime is never used for training. Optional `onnx` on engine crates is parity-only.

## Backend checks

Cross-backend score trajectories are bit-exact: numerics use `rlx-cpu` BLAS; `--device all` validates every RLX backend in the build.

```bash
just test-wake-backends
just features=all-backends wake-all-backends -- --wav clip.wav
just features=all-backends wake-train-cnn -- --synth --keyword "hey rlx" \
  --out /tmp/wake_model.safetensors --device all --epochs 20
just bench-wake
just wake-cuda-msi   # CUDA on ssh msi
```
