// RLX models — calibration quantization.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Calibration / learned post-training quantization for RLX.
//!
//! Host-side algorithms that turn an FP weight (+ a little calibration data)
//! into low-bit weights with lower error than naïve round-to-nearest:
//!   - [`quant`] — the shared group-wise symmetric quantizer (the baseline).
//!   - [`awq`] — activation-aware per-channel scaling.
//!   - [`gptq`] — Hessian-based optimal-brain quantization with error feedback.
//!   - [`dynamic`] — per-layer bit allocation by sensitivity.
//!   - [`bitnet`] — BitNet b1.58 ternary (`{-1,0,1}`) weight + int8 activation quant.
//!
//! The products are plain quantized weights; running them is the job of the
//! first-class `QMatMul` / `DequantMatMul` ops, so a quantized model is
//! backend-portable. DWQ (distill-to-FP) lives in `rlx-tune` since it trains.
//!
//! [`gguf_sink`] serializes any of these into a **loadable** GGUF v3 checkpoint
//! so the calibrated weights become a shippable artifact, not just in-memory
//! tensors.

pub mod awq;
pub mod bitnet;
pub mod dynamic;
pub mod gguf_sink;
pub mod gptq;
pub mod quant;

pub use awq::{awq_effective_weight, awq_quantize};
pub use bitnet::{
    TernaryQuant, dequantize_bitnet, pack_ternary, quantize_activations_int8, quantize_bitnet,
    unpack_ternary,
};
pub use dynamic::{dynamic_bit_allocation, rtn_sensitivity};
pub use gguf_sink::{Encoding, SinkTensor, encode_bytes, group_quant_bytes, write_gguf_file};
pub use gptq::gptq_quantize;
pub use quant::{GroupQuant, dequantize, matmul_wt, mse, quantize_rtn};
