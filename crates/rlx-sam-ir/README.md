# rlx-sam-ir

Shared RLX IR builders for the **SAM v1 / SAM 2 mask decoders**. This is a small internal library — no CLI, no weight loader, no model. It emits the compiled sub-graphs that both segmentation crates run inside their mask-decoder loop, so the two-way transformer, hypernetwork matmul, and MLP logic lives in exactly one place.

## Modules

| Module | Builds |
|---|---|
| `twoway_transformer_ir` | the two-way (tokens ↔ image) transformer stack → [`TwoWayTransformerCompiled`] |
| `mask_hyper_matmul_ir` | the hypernetwork mask-token × upscaled-embedding matmul |
| `mask_prompt_ir` | prompt / dense-embedding mask helper graph |
| `mlp_relu_ir` | ReLU MLP head (hyper MLPs, IoU / object-score heads) |

## Public API

```rust
use rlx_sam_ir::TwoWayTransformerCompiled;
// also: rlx_sam_ir::mask_hyper_matmul_ir::MaskHyperMatmulCompiled,
//       rlx_sam_ir::{mask_prompt_ir, mlp_relu_ir};

// Compiled once per shape on a device, then `.run(...)` each mask-decoder call.
let mut hyper = rlx_sam_ir::mask_hyper_matmul_ir::MaskHyperMatmulCompiled::compile(
    num_mask_tokens,
    /*chan=*/ transformer_dim / 8,
    /*grid=*/ prompt_grid,
    rlx_runtime::Device::Cpu,
)?;
# anyhow::Ok(())
```

Each `*Compiled` type wraps a device-compiled `Graph`; construct it with `compile(...)` and reuse it across frames (RLX compiles for static shapes, so recompile on a new shape). Consumers pass the extracted weight structs in and get flat `Vec<f32>` outputs back — see how [rlx-sam](../rlx-sam) and [rlx-sam2](../rlx-sam2) wire these into `mask_decoder_forward`.

## How it fits

| Consumer | Uses rlx-sam-ir for |
|---|---|
| [rlx-sam](../rlx-sam) | SAM v1 mask decoder |
| [rlx-sam2](../rlx-sam2) | SAM 2 mask decoder (`TwoWayTransformerCompiled`, hyper matmul) |

## Tests

```bash
cargo test -p rlx-sam-ir
```

Numerical parity is exercised by the consumer crates' decoder tests (e.g. `cargo test -p rlx-sam2`), which compile these graphs and compare against their host references.
