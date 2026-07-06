# rlx-quant-calib

Calibration / learned **post-training quantization** for RLX. Host-side algorithms that turn an FP weight (plus a little calibration data) into low-bit weights with lower error than naïve round-to-nearest. The products are plain quantized tensors; running them is the job of the first-class `QMatMul` / `DequantMatMul` ops, so a quantized model stays backend-portable.

## Algorithms

| Module | Method |
|--------|--------|
| [`quant`](src/quant.rs) | Group-wise symmetric RTN — the baseline every learned method must beat (`quantize_rtn`, `dequantize`, `mse`) |
| [`awq`](src/awq.rs) | **AWQ** — activation-aware per-channel scaling (`awq_quantize`, `awq_effective_weight`) |
| [`gptq`](src/gptq.rs) | **GPTQ** — Hessian-based optimal-brain quantization with error feedback (`gptq_quantize`) |
| [`dynamic`](src/dynamic.rs) | Per-layer bit allocation by sensitivity (`dynamic_bit_allocation`, `rtn_sensitivity`) |
| [`bitnet`](src/bitnet.rs) | **BitNet b1.58** ternary `{-1,0,1}` weights + int8 activation quant (`quantize_bitnet`, `pack_ternary`) |

DWQ (distill-to-FP) lives in [`rlx-tune`](../rlx-tune) since it trains. [`gguf_sink`](src/gguf_sink.rs) serializes any of these into a loadable GGUF v3 checkpoint so calibrated weights become a shippable artifact.

## Public API

```rust
use rlx_quant_calib::{quantize_rtn, awq_quantize, dequantize, mse};

// out×in FP weight, quantize to 4-bit with group size 128.
let (out, inn, bits, gs) = (256, 512, 4, 128);

// Baseline round-to-nearest.
let rtn = quantize_rtn(&w, out, inn, bits, gs);

// Activation-aware: `act_scale` is a per-input-channel importance vector
// gathered from calibration activations.
let (awq, scale) = awq_quantize(&w, out, inn, &act_scale, bits, gs);

// Compare reconstruction error against the FP weight.
println!("rtn mse {}  awq mse {}", mse(&w, &dequantize(&rtn)), mse(&w, &dequantize(&awq)));
# anyhow::Ok(())
```

`GroupQuant { q, scales, out, inn, bits, group_size }` is the shared output type. Serialize to GGUF:

```rust
use rlx_quant_calib::{group_quant_bytes, write_gguf_file, SinkTensor};
// group_quant_bytes(&gq, ggml_type) → packed block bytes; write_gguf_file(...) writes the checkpoint.
```

## How it fits

- Consumed by [`rlx-tune`](../rlx-tune) (DWQ, training-time flows) and any model crate that ships quantized GGUF weights.
- Depends only on [`rlx-gguf`](../rlx-gguf) for the serialization sink — no runtime/backend deps, so calibration is pure CPU host code.
