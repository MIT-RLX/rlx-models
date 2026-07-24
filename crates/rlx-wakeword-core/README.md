# rlx-wakeword-core

`no_std` + `alloc` mel frontend and `WakeCnn` for the first-party wakeword product.

No RLX runtime dependency — suitable for MCU / FPGA ports and WASM (via [`rlx-wakeword-wasm`](../rlx-wakeword-wasm)).

## Contents

| Module | Role |
|--------|------|
| `mel` | 16 kHz mel frontend (OWW-compatible hop / norm) |
| `cnn` | Lite / full `WakeCnn` + `WakeCnnWeights` |
| `ops` | Pure f32 conv / GEMV / pool (no BLAS) |
| `ternary` | Exact `{−1,0,+1}` quantize, trit pack, fused add/sub kernels |
| `pack` | `.rlxw` header (`RLXW` magic; dtype 0=f32, 2=ternary) |

Weight field layout matches [`rlx-wake`](../rlx-wake) (`conv1..3`, `fc1`, `fc2`, `cfg`).

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
