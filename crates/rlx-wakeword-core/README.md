# rlx-wakeword-core

`no_std` + `alloc` mel frontend and `WakeCnn` for the first-party wakeword product.

No RLX runtime dependency — suitable for MCU / FPGA ports and WASM (via [`rlx-wakeword-wasm`](../rlx-wakeword-wasm)).

## Contents

| Module | Role |
|--------|------|
| `mel` | 16 kHz mel frontend (OWW-compatible hop / norm) |
| `pcen` | Per-channel energy normalization over the mel features (stateful, streaming) |
| `cnn` | Lite / mid / full `WakeCnn` + `WakeCnnWeights`, streaming, K outputs |
| `ops` | Pure f32 conv / GEMV / pool (no BLAS) |
| `ternary` | Exact `{−1,0,+1}` quantize, trit pack, fused add/sub kernels |
| `pack` | `.rlxw` header (`RLXW` magic; dtype 0=f32, 2=ternary) |

Weight field layout matches [`rlx-wake`](../rlx-wake) (`conv1..3`, `res{i}`, `attn`, `fc1`, `fc2`, `cfg`).

## Network options (`WakeCnnConfig`)

| Field | Effect |
|-------|--------|
| `pool_chunks` | The last layer averaged in this many ordered stretches of the window (1: one average) |
| `res_blocks` | Residual blocks after the third layer, dilated 1, 2, 4, …: each output hears ~1.3 s with 4 instead of 90 ms |
| `attn` | Attention pooling (bounded scores, softmax over time), appended to the stretch averages |
| `pcen` | PCEN on the input frames; its smoother is carried across pushes |

`WakeCnn` streams: when the window slides by a 40 ms hop, the first three layers recompute only the
positions that changed; residual blocks and pooling run over the third layer's output. Scores are
bit-identical to scoring the window from scratch (`tests/streaming.rs`). `score_window` scores the
current window on demand (a verifier fed with `append_mel_frames`).

## Ternary

```rust
use rlx_wakeword_core::{TernaryOpts, WakeCnnConfig, WakeCnnWeights};

let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
let stats = w.ternarize(TernaryOpts::fc_only()); // FC MatMul → bake TQ2-eligible
// Inference auto-uses fused gemv/conv when tensors are exact ternary.
```

- `TernaryOpts::fc_only()` — default; MatMul-shaped weights for `rlx-bake` TQ2
- `TernaryOpts::all_weights()` — also ternarize conv kernels
- `pack_trits` / `unpack_trits` — 2 bits per weight for embedded packs

## Features

| Feature | Default | Role |
|---------|---------|------|
| `std` | yes | Host tests / floats via std libm |

```bash
cargo test -p rlx-wakeword-core --release
cargo build -p rlx-wakeword-core --target wasm32-unknown-unknown --release
```

## Product crate

Streaming session, train, VAD, speaker-id: [`rlx-wakeword`](../rlx-wakeword).
