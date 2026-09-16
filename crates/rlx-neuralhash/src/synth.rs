// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! A synthetic Espresso container that mirrors the real NeuralHash network's
//! **op set and geometry** with deterministic pseudo-random weights.
//!
//! Apple's `NeuralHashv3b` weights live inside `Vision.framework` and are not
//! redistributable, so backend validation on non-Apple hardware (CUDA, ROCm)
//! runs this instead. It is not a smaller toy: it uses the same input and
//! output contract (`[1, 3, 360, 360]` → 128 floats) and every layer form the
//! shipping model contains —
//!
//! * strided 3×3 and depthwise 3×3 / 5×5 convolutions with SAME padding
//!   (including the asymmetric pad SAME produces on even-sized inputs),
//!   1×1 VALID convolutions, biased and unbiased, with and without `fused_relu`
//! * `batchnorm` with `training_instancenorm` (runtime per-`(N, C)` statistics)
//! * the explicit hard-swish / hard-sigmoid chain: `+3` → `clamp(0, 6)` → `×1/6` → `×x`
//! * squeeze-excite: global average pool → 1×1 convs → broadcast multiply
//! * a residual add
//! * two `inner_product` heads, the first with `has_relu`
//!
//! so a backend that miscompiles anything the real model needs fails here too.

use anyhow::Result;
use half::f16;
use std::collections::HashMap;

use crate::espresso::{BlobShape, EspressoNet};
use crate::hash::HASH_BITS;
use crate::seed::{EMBED_DIM, SeedMatrix};

/// Deterministic weights: an LCG, so any failure reproduces exactly.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
    /// Instance-norm parameters in Espresso's interleaved `[gamma, beta, mean,
    /// variance]` per-channel layout, with the placeholder statistics the real
    /// container carries.
    fn norm_params(&mut self, c: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(4 * c);
        for _ in 0..c {
            v.push(1.0 + self.next_f32() * 0.4); // gamma
            v.push(self.next_f32() * 0.3); // beta
            v.push(0.0); // mean placeholder
            v.push(1.0); // variance placeholder
        }
        v
    }
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| f16::from_f32(*x).to_le_bytes())
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Pack blobs into Espresso's `.weights` container layout.
fn container(entries: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let mut out = (entries.len() as u64).to_le_bytes().to_vec();
    for (idx, data) in entries {
        out.extend(idx.to_le_bytes());
        out.extend((data.len() as u64).to_le_bytes());
    }
    for (_, data) in entries {
        out.extend(data);
    }
    out
}

/// The layer list. Blob indices line up with [`synthetic_net`]'s weight table.
const NET_JSON: &str = r#"{
  "storage": "synthetic.espresso.weights",
  "format_version": 200,
  "transform_params": { "image": { "scale": 0.007843137718737125,
                                   "bias_r": -1, "bias_g": -1, "bias_b": -1,
                                   "bias_a": 0, "center_mean": 0, "is_network_bgr": 0 } },
  "layers": [
    {"type":"convolution","name":"stem/Conv2D","top":"stem","bottom":"image",
     "C":16,"K":3,"Nx":3,"Ny":3,"n_groups":1,"stride_x":2,"stride_y":2,
     "pad_mode":1,"has_biases":0,"n_parallel":1,"blob_weights_f16":0},
    {"type":"batchnorm","name":"stem/InstanceNorm","top":"stem_n","bottom":"stem",
     "C":16,"training":1,"training_instancenorm":1,"training_eps":9.999999974752427e-07,
     "training_momentum":0.9,"blob_batchnorm_params":1},

    {"type":"elementwise","name":"stem/add","top":"stem_a","bottom":"stem_n","operation":2,"alpha":3,"beta":0},
    {"type":"elementwise","name":"stem/Relu6","top":"stem_c","bottom":"stem_a","operation":119,"alpha":0,"beta":6},
    {"type":"elementwise","name":"stem/mul","top":"stem_s","bottom":"stem_c","operation":3,"alpha":0.1666666716337204,"beta":0},
    {"type":"elementwise","name":"stem/mul_1","top":"stem_h","bottom":"stem_n,stem_s","operation":1,"alpha":1,"beta":0},

    {"type":"convolution","name":"dw3/depthwise","top":"dw3","bottom":"stem_h",
     "C":16,"K":16,"Nx":3,"Ny":3,"n_groups":16,"stride_x":2,"stride_y":2,
     "pad_mode":1,"has_biases":0,"n_parallel":1,"blob_weights_f16":2},
    {"type":"batchnorm","name":"dw3/InstanceNorm","top":"dw3_n","bottom":"dw3",
     "C":16,"training":1,"training_instancenorm":1,"training_eps":9.999999974752427e-07,
     "training_momentum":0.9,"blob_batchnorm_params":3},
    {"type":"activation","name":"dw3/Relu","top":"dw3_r","bottom":"dw3_n","mode":0,"beta":0},

    {"type":"pool","name":"se/gap/Mean","top":"se_gap","bottom":"dw3_r",
     "avg_or_max":0,"is_global":1,"average_count_exclude_padding":1,
     "size_x":90,"size_y":90,"stride_x":1,"stride_y":1,"pad_mode":0},
    {"type":"convolution","name":"se/Conv","top":"se1","bottom":"se_gap",
     "C":8,"K":16,"Nx":1,"Ny":1,"n_groups":1,"pad_mode":2,"has_biases":1,
     "fused_relu":1,"n_parallel":1,"blob_weights_f16":4,"blob_biases":5},
    {"type":"convolution","name":"se/Conv_1","top":"se2","bottom":"se1",
     "C":16,"K":8,"Nx":1,"Ny":1,"n_groups":1,"pad_mode":2,"has_biases":1,
     "n_parallel":1,"blob_weights_f16":6,"blob_biases":7},
    {"type":"elementwise","name":"se/add","top":"se_a","bottom":"se2","operation":2,"alpha":3,"beta":0},
    {"type":"elementwise","name":"se/Relu6","top":"se_c","bottom":"se_a","operation":119,"alpha":0,"beta":6},
    {"type":"elementwise","name":"se/mul","top":"se_g","bottom":"se_c","operation":3,"alpha":0.1666666716337204,"beta":0},
    {"type":"elementwise","name":"se/scale","top":"se_out","bottom":"dw3_r,se_g","operation":1,"alpha":1,"beta":0,"nd_mode":true},

    {"type":"convolution","name":"compress/Conv2D","top":"cmp","bottom":"se_out",
     "C":24,"K":16,"Nx":1,"Ny":1,"n_groups":1,"pad_mode":1,"has_biases":0,
     "n_parallel":1,"blob_weights_f16":8},
    {"type":"batchnorm","name":"compress/InstanceNorm","top":"cmp_n","bottom":"cmp",
     "C":24,"training":1,"training_instancenorm":1,"training_eps":9.999999974752427e-07,
     "training_momentum":0.9,"blob_batchnorm_params":9},

    {"type":"convolution","name":"dw5/depthwise","top":"dw5","bottom":"cmp_n",
     "C":24,"K":24,"Nx":5,"Ny":5,"n_groups":24,"stride_x":2,"stride_y":2,
     "pad_mode":1,"has_biases":0,"n_parallel":1,"blob_weights_f16":10},
    {"type":"batchnorm","name":"dw5/InstanceNorm","top":"dw5_n","bottom":"dw5",
     "C":24,"training":1,"training_instancenorm":1,"training_eps":9.999999974752427e-07,
     "training_momentum":0.9,"blob_batchnorm_params":11},
    {"type":"convolution","name":"proj/Conv2D","top":"proj","bottom":"dw5_n",
     "C":24,"K":24,"Nx":1,"Ny":1,"n_groups":1,"pad_mode":1,"has_biases":0,
     "n_parallel":1,"blob_weights_f16":12},
    {"type":"elementwise","name":"res/add","top":"res","bottom":"proj,dw5_n","operation":0,"alpha":1,"beta":0},

    {"type":"pool","name":"Mean","top":"gap","bottom":"res",
     "avg_or_max":0,"is_global":1,"average_count_exclude_padding":1,
     "size_x":45,"size_y":45,"stride_x":1,"stride_y":1,"pad_mode":0},
    {"type":"inner_product","name":"fc1/MatMul","top":"fc1","bottom":"gap",
     "nB":24,"nC":64,"has_biases":1,"has_relu":1,"has_tanh":0,"has_prelu":0,
     "blob_weights_f16":13,"blob_biases":14},
    {"type":"inner_product","name":"leaf/logits","top":"leaf/logits","bottom":"fc1",
     "nB":64,"nC":128,"has_biases":1,"has_relu":0,"has_tanh":0,"has_prelu":0,
     "blob_weights_f16":15,"blob_biases":16,"attributes":{"is_output":1}}
  ]
}"#;

/// Build the synthetic container. `seed` selects the weight draw.
pub fn synthetic_net(seed: u64) -> Result<EspressoNet> {
    let mut r = Lcg::new(seed);
    let weights = container(&[
        (0, f16_bytes(&r.vec(16 * 3 * 3 * 3, 0.35))), // stem [16,3,3,3]
        (1, f32_bytes(&r.norm_params(16))),
        (2, f16_bytes(&r.vec(16 * 3 * 3, 0.45))), // dw3 [16,1,3,3]
        (3, f32_bytes(&r.norm_params(16))),
        (4, f16_bytes(&r.vec(8 * 16, 0.5))), // se fc1 [8,16,1,1]
        (5, f32_bytes(&r.vec(8, 0.1))),
        (6, f16_bytes(&r.vec(16 * 8, 0.5))), // se fc2 [16,8,1,1]
        (7, f32_bytes(&r.vec(16, 0.1))),
        (8, f16_bytes(&r.vec(24 * 16, 0.4))), // compress [24,16,1,1]
        (9, f32_bytes(&r.norm_params(24))),
        (10, f16_bytes(&r.vec(24 * 5 * 5, 0.3))), // dw5 [24,1,5,5]
        (11, f32_bytes(&r.norm_params(24))),
        (12, f16_bytes(&r.vec(24 * 24, 0.4))), // proj [24,24,1,1]
        (13, f16_bytes(&r.vec(64 * 24, 0.5))), // fc1 [nC=64, nB=24]
        (14, f32_bytes(&r.vec(64, 0.15))),
        (15, f16_bytes(&r.vec(128 * 64, 0.5))), // fc2 [nC=128, nB=64]
        (16, f32_bytes(&r.vec(128, 0.2))),
    ]);
    let mut shapes = HashMap::new();
    shapes.insert(
        "image".to_string(),
        BlobShape {
            n: 1,
            c: 3,
            h: crate::INPUT_SIZE,
            w: crate::INPUT_SIZE,
        },
    );
    EspressoNet::from_parts(NET_JSON, &weights, shapes)
}

/// A deterministic, non-degenerate `[96, 128]` output projection.
pub fn synthetic_seed(seed: u64) -> SeedMatrix {
    let mut r = Lcg::new(seed ^ 0x5EED);
    SeedMatrix::from_rows(r.vec(HASH_BITS * EMBED_DIM, 1.0)).expect("seed dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{Act, NeuralHashSpec, OpKind};

    #[test]
    fn synthetic_net_matches_the_neuralhash_contract() {
        let net = synthetic_net(7).unwrap();
        let spec = NeuralHashSpec::from_espresso(&net).unwrap();
        spec.validate_neuralhash_io().unwrap();
        assert_eq!(spec.input_shape, [1, 3, 360, 360]);
        assert_eq!(spec.output, "leaf/logits");
    }

    /// The point of this fixture is op coverage — assert it really covers the
    /// forms the shipping model uses, so it cannot silently drift into a toy.
    /// Coverage is asserted on the *unfused* spec: the point is that the
    /// fixture contains every literal Espresso layer form the shipping model
    /// uses. Fusion then collapses some of them, which
    /// `fusion_rewrites_the_hard_activation_chains` checks separately.
    #[test]
    fn synthetic_net_covers_every_real_op_form() {
        let net = synthetic_net(7).unwrap();
        let spec = NeuralHashSpec::from_espresso_unfused(&net).unwrap();
        let (mut conv, mut dw, mut k5, mut strided, mut fused, mut biased) =
            (false, false, false, false, false, false);
        let (mut norm, mut affine, mut clamp, mut add, mut mul, mut bcast) =
            (false, false, false, false, false, false);
        let (mut relu, mut pool, mut ip, mut ip_act) = (false, false, false, false);
        for o in &spec.ops {
            match &o.kind {
                OpKind::Conv {
                    groups,
                    kernel,
                    stride,
                    act,
                    bias,
                    ..
                } => {
                    conv = true;
                    dw |= *groups > 1;
                    k5 |= kernel[0] == 5;
                    strided |= stride[0] == 2;
                    fused |= act.is_some();
                    biased |= bias.is_some();
                }
                OpKind::InstanceNorm { .. } => norm = true,
                OpKind::Affine { .. } => affine = true,
                OpKind::Clamp { .. } => clamp = true,
                OpKind::Elementwise { kind } => match kind {
                    crate::spec::EwKind::Add => add = true,
                    crate::spec::EwKind::Mul => {
                        mul = true;
                        // A broadcast multiply has inputs of differing extent.
                        let a = spec.ops.iter().find(|p| p.out == o.ins[0]).map(|p| p.shape);
                        let b = spec.ops.iter().find(|p| p.out == o.ins[1]).map(|p| p.shape);
                        if let (Some(a), Some(b)) = (a, b) {
                            bcast |= a[2] != b[2];
                        }
                    }
                },
                OpKind::Activation { .. } => relu = true,
                OpKind::Pool { global, .. } => pool |= *global,
                OpKind::InnerProduct { act, .. } => {
                    ip = true;
                    ip_act |= act.is_some();
                }
                _ => {}
            }
        }
        for (ok, what) in [
            (conv, "convolution"),
            (dw, "depthwise convolution"),
            (k5, "5x5 kernel"),
            (strided, "stride 2"),
            (fused, "fused activation"),
            (biased, "conv bias"),
            (norm, "instance norm"),
            (affine, "scalar affine"),
            (clamp, "clamp"),
            (add, "residual add"),
            (mul, "elementwise multiply"),
            (bcast, "broadcast multiply (squeeze-excite)"),
            (relu, "standalone relu"),
            (pool, "global average pool"),
            (ip, "inner product"),
            (ip_act, "inner product activation"),
        ] {
            assert!(ok, "synthetic fixture no longer covers: {what}");
        }
    }

    /// Fusion must find both shapes of chain: the hard-swish (multiplied by
    /// its own input) and the squeeze-excite gate (hard-sigmoid + a separate
    /// broadcast multiply).
    #[test]
    fn fusion_rewrites_the_hard_activation_chains() {
        let net = synthetic_net(7).unwrap();
        let raw = NeuralHashSpec::from_espresso_unfused(&net).unwrap();
        let fused = NeuralHashSpec::from_espresso(&net).unwrap();
        assert!(
            fused.ops.len() < raw.ops.len(),
            "fusion did not shrink the spec ({} vs {})",
            fused.ops.len(),
            raw.ops.len()
        );
        let acts: Vec<Act> = fused
            .ops
            .iter()
            .filter_map(|o| match &o.kind {
                OpKind::Activation { act } => Some(*act),
                _ => None,
            })
            .collect();
        assert!(
            acts.contains(&Act::HardSwish),
            "no hard-swish fused: {acts:?}"
        );
        assert!(
            acts.contains(&Act::HardSigmoid),
            "no hard-sigmoid fused: {acts:?}"
        );
        // The raw chain ops must be gone.
        assert!(
            !fused
                .ops
                .iter()
                .any(|o| matches!(&o.kind, OpKind::Clamp { max, .. } if (*max - 6.0).abs() < 1e-6)),
            "a relu6 clamp survived fusion"
        );
    }
}
