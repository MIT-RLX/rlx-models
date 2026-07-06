# rlx-vjepa2

**V-JEPA 2** — Meta's self-supervised video Vision Transformer (ViT-G and friends) — for RLX. Ships the encoder, predictor, and attentive pooler, with compiled IR graph paths for all three when a device is set on the runner.

## Quick start

```bash
cargo run -p rlx-vjepa2 --bin rlx-vjepa2 --release -- --help
```

## Public API

```rust
// The crate exposes the encoder / predictor / pooler builders and an
// IR graph flow (see `builder`, `encoder`, `predictor`, `pooler`, `flow`).
// Configure via `config`, then build + run on a device.
```

Modules: `config`, `builder`, `encoder`, `predictor`, `pooler`, `layers`, `flow`, `cli`.

## How it fits

- Core graph layer: [rlx-flow](../../rlx) / [rlx-ir](../../rlx).
- Sibling image backbone: [rlx-dinov2](../rlx-dinov2).
