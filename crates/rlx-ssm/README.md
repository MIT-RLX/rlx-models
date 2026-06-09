# rlx-ssm

State-space model (SSM) **flow stages** and **custom IR ops** for hybrid / Mamba runners in the `rlx-models` workspace.

Depends on sibling [`rlx`](https://github.com/MIT-RLX/rlx) (`rlx-flow`, `rlx-ir`, `rlx-cpu`). Consumers compile graphs with `rlx-runtime` on CPU, CUDA, or wgpu.

## Stages

| Stage | Role | Lowers to |
|-------|------|-----------|
| [`MambaScanStage`](src/stages.rs) | Mamba1 prefill selective scan + D-skip | `Op::SelectiveScan` (+ softplus on `dt_raw`) |
| [`Mamba1StepStage`](src/stages.rs) | Mamba1 single-token decode with state carry | `Op::Custom` `mamba1_step` |
| [`Mamba2StepStage`](src/stages.rs) | Mamba2 decode step (packed `y \| state_out`) | `Op::Custom` `mamba2_step` |
| [`LfmSsmStepStage`](src/stages.rs) | LFM SSM decode step | `Op::Custom` `lfm_ssm_step` |
| [`LightningAttentionStepStage`](src/stages.rs) | Lightning-attention decode step | `Op::Custom` `lightning_attention_step` |
| [`LightningAttentionStage`](src/stages.rs) | Prefill placeholder (requires caller wiring) | — |

Weight keys for Mamba1 scan use [`MambaScanWeightKeys`](src/stages.rs) (`blk.N.A_log`, `blk.N.D`).

## Registration

Call once before compile / run:

```rust
rlx_ssm::register_ir_ops();
```

This registers IR shape rules and CPU reference kernels for the custom ops. GPU backends pick up `Op::SelectiveScan` via `rlx-runtime` when enabled.

## Consumers

| Crate | Usage |
|-------|--------|
| `rlx-mamba` | `MambaScanStage` + `Mamba1StepStage` in `scan.rs` |
| `rlx-lfm` | `LfmSsmStepStage` decode flow |
| `rlx-minimax` | Lightning + SSM step stages |
| `rlx-nemotron` | `Mamba2StepStage` hybrid layers |

## Tests

Integration tests live under `crates/rlx-models/tests/` (`ssm_mamba_quick_check`, `ssm_step_quick_check`, family runner tests). This crate has no standalone test target.

```bash
cargo test -p rlx-mamba
cargo test -p rlx-models --test ssm_mamba_quick_check
```

## See also

- Main repo [README](../../README.md#per-crate-readmes)
- [rlx-mamba/README.md](../rlx-mamba/README.md) — Mamba1 block driver
- [AGENTS.md](../../AGENTS.md) — SSM and Mamba test recipes
