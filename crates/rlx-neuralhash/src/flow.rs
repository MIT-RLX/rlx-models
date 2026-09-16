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

//! [`NeuralHashSpec`] → rlx-ir graph.
//!
//! Every op is a native rlx op, so the descriptor network runs on any rlx
//! backend (CPU / Metal / MLX / CUDA / ROCm / wgpu / Vulkan) with no ONNX
//! runtime anywhere in the path.
//!
//! Where rlx-ir has a fused op, it is used rather than composed: instance norm
//! becomes one `GroupNorm` (one group per channel) instead of a nine-op
//! mean/variance chain, and hard-swish / hard-sigmoid become single
//! activations. Only `l2norm` is still composed
//! (`x / sqrt(sum(x², axis=1) + eps)`); the shipping model does not use it.
//!
//! Broadcasting is left to `shape::binary_shape` rather than materialized with
//! `Expand`: a conv bias is added as `[1, C, 1, 1]`, not expanded to the full
//! feature map first.
//!
//! Asymmetric convolution / pooling padding is materialized as an explicit
//! zero [`Op::Pad`] before the op, because rlx's `Conv2d` carries a single
//! symmetric `[pad_h, pad_w]`. Silently using the top/left pad for both sides
//! would shift every feature map by a pixel.

use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::vision_ops_ir::nchw_shape;
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op, PadMode, ReduceOp};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

use crate::spec::{Act, EwKind, NeuralHashSpec, OpDef, OpKind, PoolKind};

/// Build the descriptor network with CPU-appropriate lowering.
pub fn build_graph(
    spec: &NeuralHashSpec,
    weights: &mut WeightMap,
) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    build_graph_with(spec, weights, LoweringOpts::for_device(Device::Cpu))
}

/// Build the descriptor network. Returns the lowered graph plus its host params.
pub fn build_graph_with(
    spec: &NeuralHashSpec,
    weights: &mut WeightMap,
    opts: LoweringOpts,
) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    let mut b = Builder::new(&spec.name, opts);
    let mut node: HashMap<String, HirNodeId> = HashMap::new();
    let mut shape: HashMap<String, [usize; 4]> = HashMap::new();

    let x = b.hir_mut().input(
        spec.input.clone(),
        Shape::new(&spec.input_shape, DType::F32),
    );
    node.insert(spec.input.clone(), x);
    shape.insert(spec.input.clone(), spec.input_shape);

    for (i, op) in spec.ops.iter().enumerate() {
        let id = b
            .emit(op, &node, &shape, weights)
            .with_context(|| format!("op {i}/{} {:?}", spec.ops.len(), op.name))?;
        node.insert(op.out.clone(), id);
        shape.insert(op.out.clone(), op.shape);
    }

    let out = *node
        .get(&spec.output)
        .ok_or_else(|| anyhow!("spec output {:?} is produced by no op", spec.output))?;
    b.hir_mut().set_outputs(vec![out]);
    b.finish()
}

/// Lowering choices that trade graph size for backend compatibility.
///
/// The fast paths are the default. Each can be turned off independently so a
/// backend that mis-lowers one fused op can be bisected — and worked around —
/// without giving up the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoweringOpts {
    /// Emit instance norm as a single `GroupNorm` (one group per channel)
    /// instead of the mean/variance chain.
    pub fused_norm: bool,
    /// Let binary ops broadcast `[1, C, 1, 1]` operands instead of
    /// materializing them with `Expand`.
    pub implicit_broadcast: bool,
}

impl Default for LoweringOpts {
    fn default() -> Self {
        Self {
            fused_norm: true,
            implicit_broadcast: true,
        }
    }
}

impl LoweringOpts {
    /// Defaults for `device`, then the environment overrides.
    ///
    /// Both fast paths are on everywhere. wgpu used to need the decomposed
    /// norm — `rlx-wgpu` lowered `Op::GroupNorm` to `Step::GroupNormHost`, a
    /// device→host→device round-trip of the whole arena per norm, which was
    /// both ruinous (35 norms per forward) and wrong in a chain. That is fixed
    /// upstream by a native WGSL `group_norm` kernel, so no backend is carved
    /// out here any more.
    ///
    /// `RLX_NEURALHASH_NO_FUSED_NORM=1` / `RLX_NEURALHASH_NO_BROADCAST=1` force
    /// the fallbacks, for bisecting a future divergence.
    pub fn for_device(device: Device) -> Self {
        let _ = device;
        let mut o = Self::default();
        let off = |k: &str| std::env::var_os(k).is_some();
        if off("RLX_NEURALHASH_NO_FUSED_NORM") {
            o.fused_norm = false;
        }
        if off("RLX_NEURALHASH_NO_BROADCAST") {
            o.implicit_broadcast = false;
        }
        o
    }
}

struct Builder {
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
    /// Counter for synthesized constants, so names stay unique.
    consts: usize,
    opts: LoweringOpts,
}

impl Builder {
    fn new(name: &str, opts: LoweringOpts) -> Self {
        Self {
            hir: HirModule::new(name),
            params: HashMap::new(),
            consts: 0,
            opts,
        }
    }

    fn hir_mut(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    fn load(&mut self, wm: &mut WeightMap, key: &str, want: &[usize]) -> Result<HirNodeId> {
        let (data, shape) = wm
            .take(key)
            .with_context(|| format!("missing weight {key}"))?;
        ensure!(
            shape == want,
            "weight {key} has shape {shape:?}, the graph needs {want:?}"
        );
        let id = self.hir_mut().param(key, Shape::new(&shape, DType::F32));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    /// A constant tensor filled with `v`, broadcastable over NCHW.
    fn splat(&mut self, tag: &str, v: f32) -> HirNodeId {
        self.consts += 1;
        let name = format!("const.{tag}.{}", self.consts);
        let id = self
            .hir_mut()
            .param(&name, Shape::new(&[1, 1, 1, 1], DType::F32));
        self.params.insert(name, vec![v]);
        id
    }

    fn zeros(&mut self, tag: &str, n: usize) -> HirNodeId {
        self.consts += 1;
        let name = format!("const.{tag}.{}", self.consts);
        let id = self.hir_mut().param(&name, Shape::new(&[n], DType::F32));
        self.params.insert(name, vec![0.0; n]);
        id
    }

    fn finish(self) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
        rlx_core::flow_util::graph_from_hir(self.hir, self.params)
    }

    fn emit(
        &mut self,
        op: &OpDef,
        node: &HashMap<String, HirNodeId>,
        shapes: &HashMap<String, [usize; 4]>,
        wm: &mut WeightMap,
    ) -> Result<HirNodeId> {
        let get = |name: &str| -> Result<HirNodeId> {
            node.get(name)
                .copied()
                .ok_or_else(|| anyhow!("input blob {name:?} has not been produced yet"))
        };
        let get_shape = |name: &str| -> Result<[usize; 4]> {
            shapes
                .get(name)
                .copied()
                .ok_or_else(|| anyhow!("input blob {name:?} has no recorded shape"))
        };
        ensure!(!op.ins.is_empty(), "op {:?} has no inputs", op.name);
        let [n, out_c, out_h, out_w] = op.shape;

        match &op.kind {
            OpKind::Conv {
                weight,
                bias,
                kernel,
                stride,
                pad,
                groups,
                act,
            } => {
                let x = get(&op.ins[0])?;
                let [_, in_c, in_h, in_w] = get_shape(&op.ins[0])?;
                let w = self.load(wm, weight, &[out_c, in_c / groups, kernel[0], kernel[1]])?;
                let b = match bias {
                    Some(k) => self.load(wm, k, &[out_c])?,
                    None => self.zeros(&op.name, out_c),
                };
                // Fold asymmetric pads into an explicit Pad; keep the symmetric
                // remainder on the conv itself.
                let (x, in_h, in_w, sym) =
                    self.explicit_pad(x, n, in_c, in_h, in_w, *pad, &op.name)?;
                let _ = (in_h, in_w);

                let dt = DType::F32;
                let out_shape = nchw_shape(n, out_c, out_h, out_w, dt);
                let y = self
                    .hir_mut()
                    .conv2d(x, w, *kernel, *stride, sym, *groups, out_shape);
                let y = self.add_bias(y, b, op.shape);
                Ok(match act {
                    Some(a) => self.act(*a, y, op.shape),
                    None => y,
                })
            }

            OpKind::Activation { act } => {
                let x = get(&op.ins[0])?;
                Ok(self.act(*act, x, op.shape))
            }

            OpKind::Pool {
                kind,
                kernel,
                stride,
                pad,
                global,
            } => {
                let x = get(&op.ins[0])?;
                let [_, in_c, in_h, in_w] = get_shape(&op.ins[0])?;
                if *global {
                    // Exact mean over H·W — no kernel/padding edge cases, and it
                    // is what an Espresso global average pool computes.
                    ensure!(
                        matches!(kind, PoolKind::Avg),
                        "global max pooling is not supported ({:?})",
                        op.name
                    );
                    return Ok(self.hir_mut().mean(x, vec![2, 3], true));
                }
                let (x, _, _, sym) = self.explicit_pad(x, n, in_c, in_h, in_w, *pad, &op.name)?;
                let dt = DType::F32;
                let out_shape = nchw_shape(n, out_c, out_h, out_w, dt);
                Ok(self.hir_mut().add_node(
                    Op::Pool {
                        kind: match kind {
                            PoolKind::Max => ReduceOp::Max,
                            PoolKind::Avg => ReduceOp::Mean,
                        },
                        kernel_size: kernel.to_vec(),
                        stride: stride.to_vec(),
                        padding: sym.to_vec(),
                    },
                    vec![x],
                    out_shape,
                ))
            }

            OpKind::Elementwise { kind } => {
                ensure!(
                    op.ins.len() >= 2,
                    "elementwise {:?} needs 2+ inputs",
                    op.name
                );
                let mut acc: Option<HirNodeId> = None;
                for name in &op.ins {
                    // Squeeze-excite feeds [N,C,1,1] against [N,C,H,W]; the
                    // binary op broadcasts, so no Expand is materialized.
                    let id = get(name)?;
                    let id = self.broadcast_to(id, get_shape(name)?, op.shape);
                    acc = Some(match acc {
                        None => id,
                        Some(a) => match kind {
                            EwKind::Add => self.hir_mut().add(a, id),
                            EwKind::Mul => self.hir_mut().mul(a, id),
                        },
                    });
                }
                Ok(acc.unwrap())
            }

            OpKind::InnerProduct { weight, bias, act } => {
                let x = get(&op.ins[0])?;
                let [_, c, h, w] = get_shape(&op.ins[0])?;
                let in_dim = c * h * w;
                let wid = self.load(wm, weight, &[in_dim, out_c])?;
                let flat = self.hir_mut().reshape_(x, vec![n as i64, in_dim as i64]);
                let y = self.hir_mut().mm(flat, wid);
                let y = match bias {
                    Some(k) => {
                        let b = self.load(wm, k, &[out_c])?;
                        self.hir_mut().add(y, b)
                    }
                    None => y,
                };
                // Back to NCHW so downstream ops see a uniform rank.
                let y = self
                    .hir_mut()
                    .reshape_(y, vec![n as i64, out_c as i64, 1, 1]);
                Ok(match act {
                    Some(a) => self.act(*a, y, op.shape),
                    None => y,
                })
            }

            OpKind::Affine { alpha, beta } => {
                let x = get(&op.ins[0])?;
                let mut y = x;
                if *alpha != 1.0 {
                    let a = self.splat(&op.name, *alpha);
                    y = self.hir_mut().mul(y, a);
                }
                if *beta != 0.0 {
                    let b = self.splat(&op.name, *beta);
                    y = self.hir_mut().add(y, b);
                }
                Ok(y)
            }

            OpKind::Clamp { min, max } => {
                let x = get(&op.ins[0])?;
                let s = Shape::new(&op.shape, DType::F32);
                Ok(self.clamp(x, *min, *max, s))
            }

            OpKind::InstanceNorm { scale, shift, eps } => {
                // Instance norm is exactly `GroupNorm` with one group per
                // channel — each group is `1 × H × W`, so the statistics are
                // the per-`(N, C)` spatial mean/variance Espresso wants.
                //
                // Emitting the single fused op instead of the
                // mean/sub/mul/mean/add/rsqrt/mul/mul/add chain removes eight
                // full-tensor passes per norm, and there are 35 of them.
                let x = get(&op.ins[0])?;
                let g = self.load(wm, scale, &[out_c])?;
                let b = self.load(wm, shift, &[out_c])?;
                if self.opts.fused_norm {
                    let out_shape = Shape::new(&op.shape, DType::F32);
                    return Ok(self.hir_mut().add_node(
                        Op::GroupNorm {
                            num_groups: out_c,
                            eps: *eps,
                        },
                        vec![x, g, b],
                        out_shape,
                    ));
                }
                // Fallback: the explicit statistics chain.
                let mean = self.hir_mut().mean(x, vec![2, 3], true);
                let centered = self.hir_mut().sub(x, mean);
                let sq = self.hir_mut().mul(centered, centered);
                let var = self.hir_mut().mean(sq, vec![2, 3], true);
                let e = self.splat(&op.name, *eps);
                let var = self.hir_mut().add(var, e);
                let stat_shape = Shape::new(&[n, out_c, 1, 1], DType::F32);
                let inv = self
                    .hir_mut()
                    .activation(Activation::Rsqrt, var, stat_shape);
                let normed = self.hir_mut().mul(centered, inv);
                let g4 = self.hir_mut().reshape_(g, vec![1, out_c as i64, 1, 1]);
                let g4 = self.broadcast_to(g4, [1, out_c, 1, 1], op.shape);
                let scaled = self.hir_mut().mul(normed, g4);
                let b4 = self.hir_mut().reshape_(b, vec![1, out_c as i64, 1, 1]);
                let b4 = self.broadcast_to(b4, [1, out_c, 1, 1], op.shape);
                Ok(self.hir_mut().add(scaled, b4))
            }

            OpKind::Concat => {
                let ids: Result<Vec<HirNodeId>> = op.ins.iter().map(|k| get(k)).collect();
                Ok(self.hir_mut().concat_(ids?, 1))
            }

            OpKind::Reshape => {
                let x = get(&op.ins[0])?;
                let src = get_shape(&op.ins[0])?;
                if src == op.shape {
                    return Ok(x);
                }
                Ok(self
                    .hir_mut()
                    .reshape_(x, vec![n as i64, out_c as i64, out_h as i64, out_w as i64]))
            }

            OpKind::L2Norm { eps } => {
                let x = get(&op.ins[0])?;
                let sq = self.hir_mut().mul(x, x);
                let sum = self.hir_mut().sum(sq, vec![1], true);
                let e = self.splat(&op.name, *eps);
                let sum = self.hir_mut().add(sum, e);
                let shape = Shape::new(&[n, 1, out_h, out_w], DType::F32);
                let norm = self.hir_mut().activation(Activation::Sqrt, sum, shape);
                Ok(self.hir_mut().div(x, norm))
            }
        }
    }

    /// Split `[t, b, l, r]` into an explicit zero-pad for the asymmetric part
    /// plus the symmetric `[pad_h, pad_w]` the conv/pool op can carry itself.
    ///
    /// Returns `(x, padded_h, padded_w, symmetric_pad)`.
    fn explicit_pad(
        &mut self,
        x: HirNodeId,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        pad: [usize; 4],
        tag: &str,
    ) -> Result<(HirNodeId, usize, usize, [usize; 2])> {
        let [t, bo, l, r] = pad;
        if t == bo && l == r {
            return Ok((x, h, w, [t, l]));
        }
        // Pad the asymmetric excess (the extra row/column on one side) up front,
        // then let the op apply the symmetric minimum on both sides.
        let (min_v, extra_t, extra_b) = (t.min(bo), t - t.min(bo), bo - t.min(bo));
        let (min_h, extra_l, extra_r) = (l.min(r), l - l.min(r), r - l.min(r));
        let padded_h = h + extra_t + extra_b;
        let padded_w = w + extra_l + extra_r;
        ensure!(
            padded_h > 0 && padded_w > 0,
            "{tag}: degenerate padding {pad:?}"
        );
        let out_shape = nchw_shape(n, c, padded_h, padded_w, DType::F32);
        let padded = self.hir_mut().add_node(
            Op::Pad {
                pads: vec![[0, 0], [0, 0], [extra_t, extra_b], [extra_l, extra_r]],
                mode: PadMode::Constant(0.0),
            },
            vec![x],
            out_shape,
        );
        Ok((padded, padded_h, padded_w, [min_v, min_h]))
    }

    /// Add a `[C]` bias to NCHW activations.
    ///
    /// rlx's binary ops broadcast (`shape::binary_shape`), so a `[1, C, 1, 1]`
    /// reshape is enough — materializing the bias to the full `[N, C, H, W]`
    /// with an `Expand` first would write a whole feature map per conv, and
    /// there are 54 of them.
    fn add_bias(&mut self, y: HirNodeId, bias: HirNodeId, dst: [usize; 4]) -> HirNodeId {
        let out_c = dst[1];
        let b4 = self.hir_mut().reshape_(bias, vec![1, out_c as i64, 1, 1]);
        let b4 = self.broadcast_to(b4, [1, out_c, 1, 1], dst);
        self.hir_mut().add(y, b4)
    }

    /// Bring `src`-shaped `id` up to `dst`.
    ///
    /// With `implicit_broadcast` the binary op does it (no tensor is
    /// materialized); otherwise an explicit `Expand` is emitted for backends
    /// that need the operands to match exactly.
    fn broadcast_to(&mut self, id: HirNodeId, src: [usize; 4], dst: [usize; 4]) -> HirNodeId {
        if src == dst || self.opts.implicit_broadcast {
            return id;
        }
        let out_shape = Shape::new(&dst, DType::F32);
        self.hir_mut().add_node(
            Op::Expand {
                target_shape: dst.iter().map(|d| *d as i64).collect(),
            },
            vec![id],
            out_shape,
        )
    }

    /// Emit an activation, composing the ones rlx-ir has no single op for.
    fn act(&mut self, a: Act, x: HirNodeId, shape: [usize; 4]) -> HirNodeId {
        let s = Shape::new(&shape, DType::F32);
        match a {
            Act::Relu => self.hir_mut().relu(x),
            Act::Relu6 => self.clamp(x, 0.0, 6.0, s),
            Act::Sigmoid => self.hir_mut().activation(Activation::Sigmoid, x, s),
            Act::Tanh => self.hir_mut().activation(Activation::Tanh, x, s),
            // rlx-ir has both natively; `crate::spec::fuse_hard_activations`
            // rewrites Espresso's four-op chains onto them.
            Act::HardSigmoid => self.hir_mut().activation(Activation::HardSigmoid, x, s),
            Act::HardSwish => self.hir_mut().activation(Activation::HardSwish, x, s),
            Act::Linear { alpha, beta } => {
                let a = self.splat("alpha", alpha);
                let y = self.hir_mut().mul(x, a);
                if beta == 0.0 {
                    y
                } else {
                    let b = self.splat("beta", beta);
                    self.hir_mut().add(y, b)
                }
            }
        }
    }

    fn clamp(&mut self, x: HirNodeId, min: f32, max: f32, shape: Shape) -> HirNodeId {
        self.hir_mut()
            .add_node(Op::Clamp { min, max }, vec![x], shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::espresso::{BlobShape, EspressoNet};
    use half::f16;
    use rlx_runtime::Session;

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
    fn f16b(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| f16::from_f32(*x).to_le_bytes())
            .collect()
    }
    fn f32b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn build_and_run(net: &EspressoNet, input: &[f32]) -> Vec<Vec<f32>> {
        let spec = NeuralHashSpec::from_espresso(net).unwrap();
        let mut wm = crate::weights::from_espresso(net).unwrap();
        let (graph, params) = build_graph(&spec, &mut wm).unwrap();
        let mut c = Session::new(Device::Cpu).compile(graph);
        rlx_core::flow_util::attach_built_params(&mut c, params, &[]);
        c.run(&[(spec.input.as_str(), input)])
    }

    fn shapes(pairs: &[(&str, [usize; 4])]) -> HashMap<String, BlobShape> {
        pairs
            .iter()
            .map(|(k, [n, c, h, w])| {
                (
                    (*k).to_string(),
                    BlobShape {
                        n: *n,
                        c: *c,
                        h: *h,
                        w: *w,
                    },
                )
            })
            .collect()
    }

    /// 1×1 conv over a 1×1×2×2 input: a pure, hand-checkable matmul.
    #[test]
    fn pointwise_conv_matches_hand_computation() {
        let net_json = r#"{"layers": [
          {"type":"input","name":"i","top":"i","bottom":""},
          {"type":"convolution","name":"c","top":"o","bottom":"i",
           "C":2,"K":2,"Nx":1,"Ny":1,"has_biases":1,"blob_weights_f16":0,"blob_biases":1}
        ]}"#;
        // W = [[1, 2], [0, -1]] (out, in); b = [0.5, -0.5]
        let weights = container(&[(0, f16b(&[1.0, 2.0, 0.0, -1.0])), (1, f32b(&[0.5, -0.5]))]);
        let net =
            EspressoNet::from_parts(net_json, &weights, shapes(&[("i", [1, 2, 1, 1])])).unwrap();
        // x = [3, 5] (channel 0 = 3, channel 1 = 5)
        let out = build_and_run(&net, &[3.0, 5.0]);
        // y0 = 1*3 + 2*5 + 0.5 = 13.5 ; y1 = 0*3 + -1*5 - 0.5 = -5.5
        assert_eq!(out[0].len(), 2);
        assert!((out[0][0] - 13.5).abs() < 1e-4, "{:?}", out[0]);
        assert!((out[0][1] + 5.5).abs() < 1e-4, "{:?}", out[0]);
    }

    /// Depthwise 3×3 with SAME padding — checks grouping and the pad mapping.
    #[test]
    fn depthwise_same_padded_conv_preserves_geometry() {
        let net_json = r#"{"layers": [
          {"type":"input","name":"i","top":"i","bottom":""},
          {"type":"convolution","name":"dw","top":"o","bottom":"i",
           "C":2,"K":2,"Nx":3,"Ny":3,"n_groups":2,"pad_mode":1,"has_biases":0,
           "blob_weights_f16":0}
        ]}"#;
        // Both channels: identity kernel (centre tap = 1).
        // [C=2, K=1 (per group), 3, 3]
        let mut k = vec![0f32; 2 * 3 * 3];
        k[4] = 1.0;
        k[9 + 4] = 1.0;
        let weights = container(&[(0, f16b(&k))]);
        let net =
            EspressoNet::from_parts(net_json, &weights, shapes(&[("i", [1, 2, 4, 4])])).unwrap();
        let x: Vec<f32> = (0..2 * 4 * 4).map(|i| i as f32).collect();
        let out = build_and_run(&net, &x);
        // Identity kernel + SAME padding → output equals input.
        assert_eq!(out[0].len(), x.len());
        for (a, b) in out[0].iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-3, "{:?} vs {:?}", out[0], x);
        }
    }

    /// Residual add + global average pool + inner product: the MobileNet tail.
    #[test]
    fn residual_gap_and_head() {
        let net_json = r#"{"layers": [
          {"type":"input","name":"i","top":"i","bottom":""},
          {"type":"convolution","name":"c","top":"a","bottom":"i",
           "C":2,"K":2,"Nx":1,"Ny":1,"has_biases":0,"blob_weights_f16":0},
          {"type":"elementwise","name":"res","top":"r","bottom":"a,i","operation":0},
          {"type":"pool","name":"gap","top":"p","bottom":"r","avg_or_max":0,"is_global":1},
          {"type":"inner_product","name":"head","top":"e","bottom":"p","nB":2,"nC":2,
           "blob_weights_f16":1,"blob_biases":2}
        ]}"#;
        // conv = identity → r = 2·i
        let weights = container(&[
            (0, f16b(&[1.0, 0.0, 0.0, 1.0])),
            (1, f16b(&[1.0, 0.0, 0.0, 1.0])), // ip weight [nC=2, nB=2] identity
            (2, f32b(&[0.0, 1.0])),
        ]);
        let net =
            EspressoNet::from_parts(net_json, &weights, shapes(&[("i", [1, 2, 2, 2])])).unwrap();
        // channel 0 = [1,1,1,1] (mean 1), channel 1 = [2,2,2,2] (mean 2)
        let x = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let out = build_and_run(&net, &x);
        // r = 2x → GAP = [2, 4] → identity head + bias [0, 1] = [2, 5]
        assert_eq!(out[0].len(), 2);
        assert!((out[0][0] - 2.0).abs() < 1e-4, "{:?}", out[0]);
        assert!((out[0][1] - 5.0).abs() < 1e-4, "{:?}", out[0]);
    }

    /// Squeeze-excite: `[N,C,1,1]` gate multiplied against `[N,C,H,W]`.
    #[test]
    fn squeeze_excite_broadcast_multiply() {
        let net_json = r#"{"layers": [
          {"type":"input","name":"i","top":"i","bottom":""},
          {"type":"pool","name":"gap","top":"g","bottom":"i","avg_or_max":0,"is_global":1},
          {"type":"elementwise","name":"se","top":"o","bottom":"i,g","operation":1}
        ]}"#;
        let net =
            EspressoNet::from_parts(net_json, &container(&[]), shapes(&[("i", [1, 1, 2, 2])]))
                .unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0]; // mean 2.5
        let out = build_and_run(&net, &x);
        for (o, i) in out[0].iter().zip(x.iter()) {
            assert!((o - i * 2.5).abs() < 1e-4, "{:?}", out[0]);
        }
    }

    /// Fused activations must match their closed forms.
    #[test]
    fn fused_activations() {
        for (flag, f) in [
            ("fused_relu", (|v: f32| v.max(0.0)) as fn(f32) -> f32),
            ("fused_relu6", |v: f32| v.clamp(0.0, 6.0)),
            ("fused_hard_swish", |v: f32| {
                v * (v / 6.0 + 0.5).clamp(0.0, 1.0)
            }),
        ] {
            let net_json = format!(
                r#"{{"layers": [
                  {{"type":"input","name":"i","top":"i","bottom":""}},
                  {{"type":"convolution","name":"c","top":"o","bottom":"i",
                    "C":1,"K":1,"Nx":1,"Ny":1,"has_biases":0,"blob_weights_f16":0,"{flag}":1}}
                ]}}"#
            );
            let weights = container(&[(0, f16b(&[1.0]))]);
            let net = EspressoNet::from_parts(&net_json, &weights, shapes(&[("i", [1, 1, 1, 5])]))
                .unwrap();
            let x = vec![-8.0, -2.0, 0.0, 3.0, 9.0];
            let out = build_and_run(&net, &x);
            for (o, v) in out[0].iter().zip(x.iter()) {
                assert!(
                    (o - f(*v)).abs() < 1e-3,
                    "{flag}: got {o} want {} for {v}",
                    f(*v)
                );
            }
        }
    }

    /// Asymmetric padding must not collapse to the top/left value.
    #[test]
    fn asymmetric_pad_is_materialized() {
        let net_json = r#"{"layers": [
          {"type":"input","name":"i","top":"i","bottom":""},
          {"type":"convolution","name":"c","top":"o","bottom":"i",
           "C":1,"K":1,"Nx":2,"Ny":1,"n_groups":1,"stride_x":1,"stride_y":1,
           "pad_t":0,"pad_b":0,"pad_l":0,"pad_r":1,"has_biases":0,"blob_weights_f16":0}
        ]}"#;
        // kernel [1, 1] over width: y[x] = i[x] + i[x+1], last tap sees the pad.
        let weights = container(&[(0, f16b(&[1.0, 1.0]))]);
        let net =
            EspressoNet::from_parts(net_json, &weights, shapes(&[("i", [1, 1, 1, 3])])).unwrap();
        let out = build_and_run(&net, &[1.0, 2.0, 4.0]);
        // width stays 3: [1+2, 2+4, 4+0]
        assert_eq!(out[0].len(), 3);
        assert!((out[0][0] - 3.0).abs() < 1e-4, "{:?}", out[0]);
        assert!((out[0][1] - 6.0).abs() < 1e-4, "{:?}", out[0]);
        assert!((out[0][2] - 4.0).abs() < 1e-4, "{:?}", out[0]);
    }

    #[test]
    fn missing_weight_names_the_tensor() {
        let net_json = r#"{"layers": [
          {"type":"input","name":"i","top":"i","bottom":""},
          {"type":"convolution","name":"c","top":"o","bottom":"i",
           "C":1,"K":1,"Nx":1,"Ny":1,"has_biases":0,"blob_weights_f16":0}
        ]}"#;
        let weights = container(&[(0, f16b(&[1.0]))]);
        let net =
            EspressoNet::from_parts(net_json, &weights, shapes(&[("i", [1, 1, 1, 1])])).unwrap();
        let spec = NeuralHashSpec::from_espresso(&net).unwrap();
        let mut empty = WeightMap::from_tensors(HashMap::new());
        let e = format!("{:#}", build_graph(&spec, &mut empty).unwrap_err());
        assert!(e.contains("c.weight"), "{e}");
    }
}
