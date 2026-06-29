// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! DWQ — Distilled Weight Quantization (heal-via-adapter).
//!
//! Quantize a weight, freeze it, then train a small LoRA adapter on calibration
//! inputs to **recover** the full-precision behavior — the adapter learns the
//! quantization residual. Built entirely on the [`crate::inject_lora`] +
//! [`crate::train`] machinery and the host quantizer in `rlx-quant-calib`. The
//! product is a healed effective weight that runs as a plain dense (or
//! re-quantized) layer with much lower error than the quantized layer alone.

use crate::trainer::{Adam, ParamSlot, train};
use crate::{FuseMode, LoraSpec, inject_lora};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_quant_calib::{dequantize, quantize_rtn};
use std::collections::HashMap;

/// Outcome of a DWQ heal on one linear layer.
#[derive(Debug, Clone)]
pub struct DwqResult {
    /// Healed effective weight `[in, out]` = `W_q + A·B`.
    pub healed_weight: Vec<f32>,
    /// Output MSE of the quantized layer alone vs the FP teacher.
    pub quant_error: f32,
    /// Output MSE of the healed (quant + adapter) layer vs the FP teacher.
    pub dwq_error: f32,
}

fn matmul_io(x: &[f32], w: &[f32], samples: usize, inn: usize, out: usize) -> Vec<f32> {
    let mut y = vec![0f32; samples * out];
    for s in 0..samples {
        for o in 0..out {
            let mut acc = 0.0;
            for i in 0..inn {
                acc += x[s * inn + i] * w[i * out + o];
            }
            y[s * out + o] = acc;
        }
    }
    y
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len().max(1) as f32
}

/// DWQ-heal a single linear layer `w_fp [in, out]`: quantize it to `bits`,
/// inject a rank-`rank` LoRA on the frozen quantized base, and train the
/// adapter (Adam, `steps`) on calibration `x [samples, in]` to match the FP
/// teacher's output `x · w_fp`. Returns the healed effective weight and the
/// quant-only / healed output errors.
#[allow(clippy::too_many_arguments)]
pub fn dwq_heal_linear(
    w_fp: &[f32],
    x: &[f32],
    inn: usize,
    out: usize,
    samples: usize,
    bits: u32,
    rank: usize,
    steps: usize,
    lr: f32,
) -> anyhow::Result<DwqResult> {
    // FP teacher output, and the quantized base.
    let target = matmul_io(x, w_fp, samples, inn, out);
    let q = quantize_rtn(w_fp, inn, out, bits, out); // [in, out], group along out
    let w_q = dequantize(&q);
    let quant_error = mse(&target, &matmul_io(x, &w_q, samples, inn, out));

    // Graph: y = x · Wq ; loss = MSE(y, target). Inject LoRA on "wq".
    let f = DType::F32;
    let mut g = Graph::new("dwq_heal");
    let xin = g.input("x", Shape::new(&[samples, inn], f));
    let wq = g.param("wq", Shape::new(&[inn, out], f));
    let y = g.matmul(xin, wq, Shape::new(&[samples, out], f));
    let t = g.input("t", Shape::new(&[samples, out], f));
    let diff = g.sub(y, t);
    let sq = g.mul(diff, diff);
    let flat = g.reshape_(sq, vec![(samples * out) as i64]);
    let loss = g.mean(flat, vec![0], false);
    g.set_outputs(vec![loss]);

    let spec = LoraSpec::new(rank, rank as f32, vec!["wq".into()]); // scale = 1
    let (graph, adapters) = inject_lora(&g, &spec, FuseMode::Unfused);

    // A random (deterministic), B = 0; base Wq frozen.
    let a_name = adapters[0].name.clone();
    let b_name = adapters[1].name.clone();
    let mut a_init = vec![0f32; inn * rank];
    let mut seed = 0x9e37_79b9u32;
    for v in a_init.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((seed >> 8) as f32 / u32::MAX as f32 - 0.5) * 0.1;
    }
    let mut params = HashMap::new();
    params.insert("wq".to_string(), w_q.clone());
    params.insert(a_name.clone(), a_init);
    params.insert(b_name.clone(), vec![0f32; rank * out]);

    let wrt = vec![
        ParamSlot {
            name: a_name.clone(),
            node: adapters[0].node,
        },
        ParamSlot {
            name: b_name.clone(),
            node: adapters[1].node,
        },
    ];
    let inputs = vec![
        ("x".to_string(), x.to_vec()),
        ("t".to_string(), target.clone()),
    ];
    let mut opt = Adam::new(lr);
    train(graph, &wrt, &mut params, &inputs, &mut opt, steps, None)?;

    // Healed effective weight Wq + A·B, and its output error.
    let a = &params[&a_name];
    let b = &params[&b_name];
    let mut healed = w_q.clone();
    for i in 0..inn {
        for o in 0..out {
            let mut acc = 0.0;
            for k in 0..rank {
                acc += a[i * rank + k] * b[k * out + o];
            }
            healed[i * out + o] += acc;
        }
    }
    let dwq_error = mse(&target, &matmul_io(x, &healed, samples, inn, out));

    Ok(DwqResult {
        healed_weight: healed,
        quant_error,
        dwq_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 2.0
            })
            .collect()
    }

    #[test]
    fn dwq_heals_quantization_error() {
        // Aggressive 3-bit quant, then a rank-`out` LoRA can represent the full
        // residual → the healed layer recovers most of the FP accuracy.
        let (inn, out, rank) = (8usize, 4usize, 4usize);
        let samples = 32usize;
        let w_fp = pseudo(inn * out, 1);
        let x = pseudo(samples * inn, 2);

        let res = dwq_heal_linear(&w_fp, &x, inn, out, samples, 3, rank, 400, 0.02).unwrap();

        assert!(res.quant_error > 1e-5, "quant should introduce error");
        assert!(
            res.dwq_error < res.quant_error * 0.25,
            "DWQ {} should heal quant {}",
            res.dwq_error,
            res.quant_error
        );
    }
}
