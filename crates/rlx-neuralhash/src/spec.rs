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

//! The native NeuralHash architecture description.
//!
//! A [`NeuralHashSpec`] is a topologically-ordered list of rlx-level ops with
//! resolved activation shapes — the same "recipe" shape `rlx-ocr2` uses for its
//! 265-op detector. It is the crate's architecture source of truth:
//! [`crate::flow`] turns one into an rlx-ir graph, and nothing downstream ever
//! looks at a vendor container again.
//!
//! A spec is produced once by [`NeuralHashSpec::from_espresso`] from the
//! vendor's `.espresso.net` layer list, and serializes to JSON so the derived
//! architecture can be pinned in-repo (`--export-spec`) and reviewed.
//!
//! # Unsupported layers are hard errors
//!
//! Espresso carries ~40 layer types; NeuralHash uses a MobileNetV3-shaped
//! subset. Anything outside the mapping below aborts the build naming the
//! layer and its attributes, because a skipped layer would still yield 128
//! plausible floats and therefore a plausible — but wrong — hash.

use anyhow::{Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::espresso::{BlobShape, EspressoLayer, EspressoNet};

/// Pointwise activation, applied either fused onto a conv or standalone.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Act {
    Relu,
    /// `min(max(x, 0), 6)`.
    Relu6,
    Sigmoid,
    Tanh,
    /// `clamp(x / 6 + 0.5, 0, 1)` — MobileNetV3's `hardSigmoid`.
    HardSigmoid,
    /// `x * hard_sigmoid(x)` — MobileNetV3's `hardSwish`.
    HardSwish,
    /// `alpha * x + beta`.
    Linear {
        alpha: f32,
        beta: f32,
    },
}

/// Pooling reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolKind {
    Max,
    Avg,
}

/// Binary elementwise reduction over 2+ inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EwKind {
    Add,
    Mul,
}

/// One rlx-level op.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum OpKind {
    /// Conv2d, optionally grouped/depthwise, optionally biased, with an
    /// optional fused activation. `pad` is `[top, bottom, left, right]`.
    Conv {
        weight: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bias: Option<String>,
        kernel: [usize; 2],
        stride: [usize; 2],
        pad: [usize; 4],
        groups: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        act: Option<Act>,
    },
    /// Standalone activation.
    Activation { act: Act },
    /// `[top, bottom, left, right]` padded pooling.
    Pool {
        kind: PoolKind,
        kernel: [usize; 2],
        stride: [usize; 2],
        pad: [usize; 4],
        global: bool,
    },
    /// Elementwise combine of every input.
    Elementwise { kind: EwKind },
    /// Single-input affine `x * alpha + beta` (Espresso `elementwise`
    /// operations 2 and 3, which carry the scalar in `alpha` / `beta`).
    Affine { alpha: f32, beta: f32 },
    /// Single-input `clamp(x, min, max)` (Espresso `elementwise` operation 119
    /// — the `Relu6` in a hard-swish / hard-sigmoid chain).
    Clamp { min: f32, max: f32 },
    /// Per-channel instance normalization: normalize over `H×W` for each
    /// `(N, C)`, then scale and shift.
    ///
    /// Espresso spells this `batchnorm` with `training_instancenorm = 1`, which
    /// means the statistics come from the *input at runtime* — the stored mean
    /// and variance are placeholders and must not be used.
    InstanceNorm {
        scale: String,
        shift: String,
        eps: f32,
    },
    /// Fully-connected over a flattened `[N, C·H·W]`. `weight` is `[in, out]`.
    InnerProduct {
        weight: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bias: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        act: Option<Act>,
    },
    /// Channel-axis concatenation.
    Concat,
    /// Shape-only rewire (`reshape`, `flatten`, `copy`, `pass_through`).
    Reshape,
    /// L2 normalization over the channel axis.
    L2Norm { eps: f32 },
}

/// One op plus its resolved NCHW output shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpDef {
    pub name: String,
    #[serde(flatten)]
    pub kind: OpKind,
    #[serde(rename = "in")]
    pub ins: Vec<String>,
    pub out: String,
    /// `[N, C, H, W]` of `out`.
    pub shape: [usize; 4],
}

/// A complete, self-contained NeuralHash architecture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralHashSpec {
    pub name: String,
    /// Graph input blob name.
    pub input: String,
    /// `[N, C, H, W]` of the input (`[1, 3, 360, 360]` for NeuralHash).
    pub input_shape: [usize; 4],
    /// Graph output blob name.
    pub output: String,
    pub ops: Vec<OpDef>,
}

impl NeuralHashSpec {
    /// Derive the native spec from Apple's Espresso layer list, fusing the
    /// hard-swish / hard-sigmoid chains onto single activation ops.
    pub fn from_espresso(net: &EspressoNet) -> Result<Self> {
        let mut spec = Deriver::new(net).run()?;
        spec.ops = fuse_hard_activations(std::mem::take(&mut spec.ops), &spec.output);
        Ok(spec)
    }

    /// Derive without the fusion pass — the literal Espresso op sequence.
    ///
    /// Only useful for checking that fusion is value-preserving; the fused and
    /// unfused specs must produce identical hashes.
    pub fn from_espresso_unfused(net: &EspressoNet) -> Result<Self> {
        Deriver::new(net).run()
    }

    /// Parse a previously exported spec.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| anyhow!("parsing neuralhash spec: {e}"))
    }

    /// Serialize for `--export-spec`.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow!("serializing neuralhash spec: {e}"))
    }

    /// Descriptor width — the channel count of the output blob, flattened.
    pub fn output_dim(&self) -> usize {
        self.ops
            .iter()
            .find(|o| o.out == self.output)
            .map(|o| o.shape.iter().skip(1).product())
            .unwrap_or(0)
    }

    /// Names of every weight tensor the spec references, in graph order.
    pub fn weight_names(&self) -> Vec<String> {
        let mut v = Vec::new();
        for o in &self.ops {
            match &o.kind {
                OpKind::Conv { weight, bias, .. } | OpKind::InnerProduct { weight, bias, .. } => {
                    v.push(weight.clone());
                    if let Some(b) = bias {
                        v.push(b.clone());
                    }
                }
                OpKind::InstanceNorm { scale, shift, .. } => {
                    v.push(scale.clone());
                    v.push(shift.clone());
                }
                _ => {}
            }
        }
        v
    }

    /// Check the spec matches what the NeuralHash pipeline requires:
    /// `[1, 3, 360, 360]` in, 128 floats out.
    pub fn validate_neuralhash_io(&self) -> Result<()> {
        ensure!(
            self.input_shape == [1, 3, crate::INPUT_SIZE, crate::INPUT_SIZE],
            "neuralhash: spec input is {:?}, expected [1, 3, {}, {}]",
            self.input_shape,
            crate::INPUT_SIZE,
            crate::INPUT_SIZE
        );
        let dim = self.output_dim();
        ensure!(
            dim == crate::EMBED_DIM,
            "neuralhash: spec output {:?} is {dim} floats, expected {}",
            self.output,
            crate::EMBED_DIM
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Espresso → spec
// ---------------------------------------------------------------------------

struct Deriver<'a> {
    net: &'a EspressoNet,
    shapes: HashMap<String, [usize; 4]>,
    ops: Vec<OpDef>,
    input: Option<String>,
    input_shape: [usize; 4],
    /// Blob name → tensor name counter, so Espresso's duplicate layer names
    /// (every LayerNorm is called `instancenorm_test`) stay distinct.
    used_names: HashMap<String, usize>,
}

impl<'a> Deriver<'a> {
    fn new(net: &'a EspressoNet) -> Self {
        Self {
            net,
            shapes: HashMap::new(),
            ops: Vec::new(),
            input: None,
            input_shape: [0; 4],
            used_names: HashMap::new(),
        }
    }

    fn run(mut self) -> Result<NeuralHashSpec> {
        self.seed_input()?;
        for (i, l) in self.net.layers.iter().enumerate() {
            self.layer(i, l)
                .map_err(|e| annotate(e, i, l, self.net.layers.len()))?;
        }
        let input = self
            .input
            .clone()
            .ok_or_else(|| anyhow!("espresso net has no input blob"))?;
        // Espresso marks the graph output with `attributes.is_output`; fall back
        // to the last op for nets that omit it.
        let flagged = self.net.layers.iter().find(|l| {
            l.attrs
                .get("attributes")
                .and_then(|a| a.get("is_output"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
                != 0
        });
        let output = match flagged.and_then(|l| l.top.first().cloned()) {
            Some(o) => o,
            None => self
                .ops
                .last()
                .map(|o| o.out.clone())
                .ok_or_else(|| anyhow!("espresso net produced no ops"))?,
        };
        ensure!(
            self.ops.iter().any(|o| o.out == output),
            "espresso net declares output blob {output:?}, which no op produces"
        );
        Ok(NeuralHashSpec {
            name: "neuralhash".into(),
            input,
            input_shape: self.input_shape,
            output,
            ops: self.ops,
        })
    }

    /// Establish the graph input.
    ///
    /// Older containers carry an explicit `input` layer; the shipping macOS net does
    /// not, so the input is the one blob that layers consume but none produces.
    /// Its geometry comes from the `.shape` sidecar.
    fn seed_input(&mut self) -> Result<()> {
        if self.net.layers.iter().any(|l| is_input_layer(&l.kind)) {
            return Ok(()); // handled by `input_layer` during the walk
        }
        let produced: std::collections::HashSet<&str> = self
            .net
            .layers
            .iter()
            .flat_map(|l| l.top.iter().map(|s| s.as_str()))
            .collect();
        let mut free: Vec<&str> = Vec::new();
        for l in &self.net.layers {
            for b in &l.bottom {
                if !produced.contains(b.as_str()) && !free.contains(&b.as_str()) {
                    free.push(b.as_str());
                }
            }
        }
        let name = match free.as_slice() {
            [one] => (*one).to_string(),
            [] => bail!("espresso net has no free input blob (every bottom is produced)"),
            many => bail!(
                "espresso net has {} free input blobs {many:?}; this port expects exactly one",
                many.len()
            ),
        };
        let shape = self.sidecar(&name).ok_or_else(|| {
            anyhow!(
                "input blob {name:?} has no entry in the .espresso.shape sidecar, \
                 so its geometry is unknown"
            )
        })?;
        self.shapes.insert(name.clone(), shape);
        self.input = Some(name);
        self.input_shape = shape;
        Ok(())
    }

    /// A unique, stable tensor-name prefix for `layer`.
    fn tensor_base(&mut self, layer: &EspressoLayer) -> String {
        tensor_base_for(&mut self.used_names, &layer.name)
    }

    fn shape_of(&self, blob: &str) -> Result<[usize; 4]> {
        self.shapes.get(blob).copied().ok_or_else(|| {
            anyhow!("input blob {blob:?} has no known shape (produced by no layer?)")
        })
    }

    /// The `.shape` sidecar's geometry for `blob`, if the container had one.
    fn sidecar(&self, blob: &str) -> Option<[usize; 4]> {
        self.net
            .shapes
            .get(blob)
            .map(|BlobShape { n, c, h, w }| [(*n).max(1), *c, *h, *w])
    }

    /// Record `out`'s shape, cross-checking the sidecar when present.
    fn set_shape(&mut self, layer: &EspressoLayer, out: &str, computed: [usize; 4]) -> Result<()> {
        if let Some(declared) = self.sidecar(out) {
            // The sidecar is authoritative; a mismatch means our stride/pad
            // reading is wrong and every downstream shape would be too.
            ensure!(
                declared == computed,
                "layer {:?} ({}): computed output shape {computed:?} for blob {out:?} \
                 but the .shape sidecar declares {declared:?} — padding/stride mapping is wrong",
                layer.name,
                layer.kind
            );
        }
        self.shapes.insert(out.to_string(), computed);
        Ok(())
    }

    fn single_top(&self, l: &EspressoLayer) -> Result<String> {
        match l.top.len() {
            1 => Ok(l.top[0].clone()),
            n => bail!("expected exactly 1 `top`, got {n}: {:?}", l.top),
        }
    }

    fn layer(&mut self, _idx: usize, l: &EspressoLayer) -> Result<()> {
        match l.kind.as_str() {
            k if is_input_layer(k) => self.input_layer(l),
            "convolution" => self.conv(l),
            "pool" | "pooling" => self.pool(l),
            "elementwise" | "eltwise" | "add" | "sum" | "product" | "multiply" => {
                self.elementwise(l)
            }
            "inner_product" | "innerproduct" | "fully_connected" => self.inner_product(l),
            "concat" | "concatenate" => self.concat(l),
            "reshape" | "flatten" | "copy" | "pass_through" | "identity" | "squeeze" => {
                self.rewire(l)
            }
            "l2norm" | "l2_normalize" | "normalize" => self.l2norm(l),
            "batchnorm" | "instancenorm" | "instancenorm_1d" => self.norm(l),
            "activation" | "relu" | "relu6" | "sigmoid" | "tanh" | "hard_sigmoid"
            | "hard_swish" | "hardswish" | "hardsigmoid" | "clip" | "scale" => self.activation(l),
            other => bail!(
                "unsupported Espresso layer type {other:?}. \
                 Attributes: {:?}. Extend `crate::spec` with a mapping for this type — \
                 skipping it would silently corrupt every hash.",
                sorted_keys(l)
            ),
        }
    }

    fn input_layer(&mut self, l: &EspressoLayer) -> Result<()> {
        let top = self.single_top(l)?;
        let shape = self
            .sidecar(&top)
            .or_else(|| {
                let c = l.int(&["k", "K", "C"])? as usize;
                let h = l.int(&["h", "H", "Ny"])? as usize;
                let w = l.int(&["w", "W", "Nx"])? as usize;
                Some([1, c, h, w])
            })
            .ok_or_else(|| {
                anyhow!(
                    "input layer {:?} has no shape: the .espresso.shape sidecar is missing and \
                     the layer carries no k/h/w fields",
                    l.name
                )
            })?;
        self.shapes.insert(top.clone(), shape);
        if self.input.is_none() {
            self.input = Some(top);
            self.input_shape = shape;
        }
        Ok(())
    }

    fn conv(&mut self, l: &EspressoLayer) -> Result<()> {
        ensure!(
            l.bottom.len() == 1,
            "convolution takes 1 input, got {:?}",
            l.bottom
        );
        let top = self.single_top(l)?;
        let [n, in_c, in_h, in_w] = self.shape_of(&l.bottom[0])?;

        let out_c = l.req_int(&["C", "n_output_channels", "outputChannels"])? as usize;
        let k_in = l.req_int(&["K", "n_input_channels", "inputChannels"])? as usize;
        let kw = l.int_or(&["Nx", "kernel_x", "size_x", "kernelWidth"], 1) as usize;
        let kh = l.int_or(&["Ny", "kernel_y", "size_y", "kernelHeight"], 1) as usize;
        let groups = l.int_or(&["n_groups", "groups", "nGroups"], 1).max(1) as usize;
        let sx = l.int_or(&["stride_x", "strideWidth", "stride"], 1).max(1) as usize;
        let sy = l.int_or(&["stride_y", "strideHeight", "stride"], 1).max(1) as usize;
        let pad = self.padding(l, [kh, kw], [sy, sx], [in_h, in_w])?;

        ensure!(
            out_c.is_multiple_of(groups) && in_c.is_multiple_of(groups),
            "convolution {:?}: groups={groups} does not divide in_c={in_c} / out_c={out_c}",
            l.name
        );
        // Espresso's `K` is the layer's TOTAL input channel count, not the
        // per-group count: a depthwise 3x3 over 16 channels is written
        // `C=16, K=16, n_groups=16` and stores a `[16, 1, 3, 3]` kernel.
        ensure!(
            k_in == in_c,
            "convolution {:?}: declared K={k_in} but the input carries {in_c} channels",
            l.name
        );
        // `n_parallel` splits the layer across compute units; anything but 1
        // would change the weight blob layout.
        let n_parallel = l.int_or(&["n_parallel"], 1);
        ensure!(
            n_parallel == 1,
            "convolution {:?}: n_parallel={n_parallel} is not supported",
            l.name
        );
        // A conv that folds its own batch norm would need the norm params too.
        ensure!(
            !l.flag(&["has_batch_norm"]),
            "convolution {:?}: has_batch_norm is set, but no batch-norm blob mapping exists",
            l.name
        );

        let out_h = out_dim(in_h, kh, sy, pad[0] + pad[1])?;
        let out_w = out_dim(in_w, kw, sx, pad[2] + pad[3])?;

        let base = self.tensor_base(l);
        let weight = format!("{base}.weight");
        let has_bias =
            l.flag(&["has_biases", "hasBiases"]) || l.blob(&["blob_biases", "b_f32"]).is_some();
        let bias = has_bias.then(|| format!("{base}.bias"));

        self.push(
            l,
            OpDef {
                name: base,
                kind: OpKind::Conv {
                    weight,
                    bias,
                    kernel: [kh, kw],
                    stride: [sy, sx],
                    pad,
                    groups,
                    act: fused_act(l)?,
                },
                ins: l.bottom.clone(),
                out: top,
                shape: [n, out_c, out_h, out_w],
            },
        )
    }

    fn pool(&mut self, l: &EspressoLayer) -> Result<()> {
        ensure!(
            l.bottom.len() == 1,
            "pool takes 1 input, got {:?}",
            l.bottom
        );
        let top = self.single_top(l)?;
        let [n, c, in_h, in_w] = self.shape_of(&l.bottom[0])?;
        let global = l.flag(&["is_global", "global_pooling", "globalPooling"]);

        // Espresso encodes the reduction as `avg_or_max`: **0 = average,
        // 1 = max**. Verified against the shipping NeuralHash net, whose ten
        // pools are all `avg_or_max = 0` and are named `.../gap/Mean` — global
        // *average* pooling for the squeeze-excite blocks and the final
        // descriptor pool. They also set `average_count_exclude_padding`, which
        // is meaningless for a max pool.
        let kind = match l.int(&["avg_or_max", "avgOrMax", "pool_type", "mode"]) {
            Some(0) => PoolKind::Avg,
            Some(1) => PoolKind::Max,
            None => bail!(
                "pool {:?} has no `avg_or_max` field; attributes: {:?}",
                l.name,
                sorted_keys(l)
            ),
            Some(other) => bail!(
                "pool {:?}: unknown `avg_or_max` value {other} (expected 0 = average, 1 = max)",
                l.name
            ),
        };

        let (kernel, stride, pad, out_h, out_w) = if global {
            ([in_h, in_w], [1, 1], [0; 4], 1, 1)
        } else {
            let kw = l.int_or(&["size_x", "kernel_x", "Nx"], 1).max(1) as usize;
            let kh = l.int_or(&["size_y", "kernel_y", "Ny"], 1).max(1) as usize;
            let sx = l.int_or(&["stride_x", "stride"], 1).max(1) as usize;
            let sy = l.int_or(&["stride_y", "stride"], 1).max(1) as usize;
            let pad = self.padding(l, [kh, kw], [sy, sx], [in_h, in_w])?;
            let oh = out_dim(in_h, kh, sy, pad[0] + pad[1])?;
            let ow = out_dim(in_w, kw, sx, pad[2] + pad[3])?;
            ([kh, kw], [sy, sx], pad, oh, ow)
        };

        let name = self.tensor_base(l);
        self.push(
            l,
            OpDef {
                name,
                kind: OpKind::Pool {
                    kind,
                    kernel,
                    stride,
                    pad,
                    global,
                },
                ins: l.bottom.clone(),
                out: top,
                shape: [n, c, out_h, out_w],
            },
        )
    }

    /// Espresso's `elementwise` covers both binary combines and single-input
    /// scalar arithmetic, discriminated by `operation`:
    ///
    /// | `operation` | inputs | meaning                     |
    /// |-------------|--------|-----------------------------|
    /// | 0           | 2      | `a + b` (residual add)      |
    /// | 1           | 2      | `a * b` (SE gate, hard-swish) |
    /// | 2           | 1      | `x + alpha`                 |
    /// | 3           | 1      | `x * alpha`                 |
    /// | 119         | 1      | `clamp(x, alpha, beta)`     |
    ///
    /// The shipping net builds every hard-swish out of 2 → 119 → 3 → 1
    /// (`x * clamp(x + 3, 0, 6) / 6`) rather than using a fused activation, so
    /// these scalar forms are not optional.
    fn elementwise(&mut self, l: &EspressoLayer) -> Result<()> {
        let top = self.single_top(l)?;
        let alpha = l.attrs.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let beta = l.attrs.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let op = match l.kind.as_str() {
            "add" | "sum" => 0,
            "product" | "multiply" => 1,
            _ => l.int(&["operation", "mode", "op"]).unwrap_or(0),
        };
        ensure!(
            !l.flag(&["fused_relu"]),
            "elementwise {:?}: fused_relu is set but not modelled",
            l.name
        );

        let (kind, ins) = match (op, l.bottom.len()) {
            (0, 2..) => {
                ensure!(
                    alpha == 1.0,
                    "elementwise add {:?}: alpha={alpha} scaling is not modelled",
                    l.name
                );
                (OpKind::Elementwise { kind: EwKind::Add }, l.bottom.clone())
            }
            (1, 2..) => {
                ensure!(
                    alpha == 1.0,
                    "elementwise mul {:?}: alpha={alpha} scaling is not modelled",
                    l.name
                );
                (OpKind::Elementwise { kind: EwKind::Mul }, l.bottom.clone())
            }
            (2, 1) => (
                OpKind::Affine {
                    alpha: 1.0,
                    beta: alpha,
                },
                l.bottom.clone(),
            ),
            (3, 1) => (OpKind::Affine { alpha, beta: 0.0 }, l.bottom.clone()),
            (119, 1) => (
                OpKind::Clamp {
                    min: alpha,
                    max: beta,
                },
                l.bottom.clone(),
            ),
            (op, n) => bail!(
                "elementwise {:?}: unhandled operation={op} with {n} input(s) \
                 (alpha={alpha}, beta={beta}); attributes: {:?}",
                l.name,
                sorted_keys(l)
            ),
        };

        // Broadcasting: squeeze-excite multiplies [N,C,H,W] by [N,C,1,1] (that
        // is what `nd_mode` marks). The result takes the largest extent.
        let mut shape = self.shape_of(&ins[0])?;
        for b in &ins[1..] {
            let s = self.shape_of(b)?;
            ensure!(
                s[1] == shape[1],
                "elementwise {:?}: channel mismatch {} vs {}",
                l.name,
                shape[1],
                s[1]
            );
            shape[2] = shape[2].max(s[2]);
            shape[3] = shape[3].max(s[3]);
        }

        let name = self.tensor_base(l);
        self.push(
            l,
            OpDef {
                name,
                kind,
                ins,
                out: top,
                shape,
            },
        )
    }

    /// Espresso `batchnorm`. With `training_instancenorm = 1` — which is what
    /// NeuralHash ships — the stored mean/variance are placeholders and the
    /// statistics are computed per `(N, C)` over `H×W` at runtime.
    fn norm(&mut self, l: &EspressoLayer) -> Result<()> {
        ensure!(
            l.bottom.len() == 1,
            "{} takes 1 input, got {:?}",
            l.kind,
            l.bottom
        );
        let top = self.single_top(l)?;
        let shape = self.shape_of(&l.bottom[0])?;
        let c = l.req_int(&["C"])? as usize;
        ensure!(
            c == shape[1],
            "norm {:?}: declared C={c} but the input carries {} channels",
            l.name,
            shape[1]
        );
        ensure!(
            l.flag(&["training_instancenorm"]),
            "norm {:?}: only instance normalization is supported \
             (training_instancenorm is unset, so this layer wants running \
              batch statistics that are not modelled)",
            l.name
        );
        let eps = l
            .attrs
            .get("training_eps")
            .or_else(|| l.attrs.get("epsilon"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;

        let base = self.tensor_base(l);
        self.push(
            l,
            OpDef {
                name: base.clone(),
                kind: OpKind::InstanceNorm {
                    scale: format!("{base}.scale"),
                    shift: format!("{base}.shift"),
                    eps,
                },
                ins: l.bottom.clone(),
                out: top,
                shape,
            },
        )
    }

    fn inner_product(&mut self, l: &EspressoLayer) -> Result<()> {
        ensure!(
            l.bottom.len() == 1,
            "inner_product takes 1 input, got {:?}",
            l.bottom
        );
        let top = self.single_top(l)?;
        let [n, c, h, w] = self.shape_of(&l.bottom[0])?;
        let n_in = l.req_int(&["nB", "n_input", "inputChannels"])? as usize;
        let n_out = l.req_int(&["nC", "n_output", "outputChannels"])? as usize;
        ensure!(
            n_in == c * h * w,
            "inner_product {:?}: declared nB={n_in} but the input flattens to {} ({c}·{h}·{w})",
            l.name,
            c * h * w
        );

        ensure!(
            !l.flag(&["has_prelu"]),
            "inner_product {:?}: has_prelu is set but PReLU slopes are not mapped",
            l.name
        );
        let act = if l.flag(&["has_relu"]) {
            Some(Act::Relu)
        } else if l.flag(&["has_tanh"]) {
            Some(Act::Tanh)
        } else {
            None
        };

        let base = self.tensor_base(l);
        let weight = format!("{base}.weight");
        let bias = l
            .blob(&["blob_biases", "b_f32"])
            .map(|_| format!("{base}.bias"));
        self.push(
            l,
            OpDef {
                name: base,
                kind: OpKind::InnerProduct { weight, bias, act },
                ins: l.bottom.clone(),
                out: top,
                shape: [n, n_out, 1, 1],
            },
        )
    }

    fn concat(&mut self, l: &EspressoLayer) -> Result<()> {
        ensure!(
            l.bottom.len() >= 2,
            "concat takes 2+ inputs, got {:?}",
            l.bottom
        );
        let top = self.single_top(l)?;
        let mut shape = self.shape_of(&l.bottom[0])?;
        let mut c = shape[1];
        for b in &l.bottom[1..] {
            let s = self.shape_of(b)?;
            ensure!(
                s[2] == shape[2] && s[3] == shape[3],
                "concat {:?}: spatial mismatch {:?} vs {:?}",
                l.name,
                shape,
                s
            );
            c += s[1];
        }
        shape[1] = c;
        let name = self.tensor_base(l);
        self.push(
            l,
            OpDef {
                name,
                kind: OpKind::Concat,
                ins: l.bottom.clone(),
                out: top,
                shape,
            },
        )
    }

    fn rewire(&mut self, l: &EspressoLayer) -> Result<()> {
        ensure!(
            l.bottom.len() == 1,
            "{} takes 1 input, got {:?}",
            l.kind,
            l.bottom
        );
        let top = self.single_top(l)?;
        let src = self.shape_of(&l.bottom[0])?;
        // Prefer the sidecar (a reshape's target geometry is not derivable from
        // the input alone); otherwise the op is a pure alias.
        let shape = self.sidecar(&top).unwrap_or(match l.kind.as_str() {
            "flatten" | "squeeze" => [src[0], src[1] * src[2] * src[3], 1, 1],
            _ => src,
        });
        ensure!(
            shape.iter().product::<usize>() == src.iter().product::<usize>(),
            "{} {:?}: cannot reshape {src:?} into {shape:?} (element count differs)",
            l.kind,
            l.name
        );
        let name = self.tensor_base(l);
        self.push(
            l,
            OpDef {
                name,
                kind: OpKind::Reshape,
                ins: l.bottom.clone(),
                out: top,
                shape,
            },
        )
    }

    fn l2norm(&mut self, l: &EspressoLayer) -> Result<()> {
        ensure!(
            l.bottom.len() == 1,
            "l2norm takes 1 input, got {:?}",
            l.bottom
        );
        let top = self.single_top(l)?;
        let shape = self.shape_of(&l.bottom[0])?;
        let eps = l
            .attrs
            .get("epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-12) as f32;
        let name = self.tensor_base(l);
        self.push(
            l,
            OpDef {
                name,
                kind: OpKind::L2Norm { eps },
                ins: l.bottom.clone(),
                out: top,
                shape,
            },
        )
    }

    fn activation(&mut self, l: &EspressoLayer) -> Result<()> {
        ensure!(
            l.bottom.len() == 1,
            "{} takes 1 input, got {:?}",
            l.kind,
            l.bottom
        );
        let top = self.single_top(l)?;
        let shape = self.shape_of(&l.bottom[0])?;
        let act = standalone_act(l)?;
        let name = self.tensor_base(l);
        self.push(
            l,
            OpDef {
                name,
                kind: OpKind::Activation { act },
                ins: l.bottom.clone(),
                out: top,
                shape,
            },
        )
    }

    /// `[top, bottom, left, right]` padding.
    ///
    /// `pad_mode` wins over the explicit `pad_t`/`pad_b`/`pad_l`/`pad_r` fields.
    /// That ordering matters: the shipping NeuralHash net writes all four
    /// explicit pads as **0** on every convolution while relying on
    /// `pad_mode = 1` (SAME), so preferring the explicit values would silently
    /// turn every 3×3 and 5×5 conv into a VALID conv and shrink the whole
    /// feature pyramid.
    ///
    /// Observed modes: `1` = SAME (output is `ceil(in / stride)`),
    /// `2` = VALID (no padding). Espresso also uses `0` for "explicit pads".
    /// SAME is resolved to concrete pads here so the spec stays fully explicit.
    fn padding(
        &self,
        l: &EspressoLayer,
        kernel: [usize; 2],
        stride: [usize; 2],
        input: [usize; 2],
    ) -> Result<[usize; 4]> {
        let explicit = || -> [usize; 4] {
            [
                l.int_or(&["pad_t"], 0).max(0) as usize,
                l.int_or(&["pad_b"], 0).max(0) as usize,
                l.int_or(&["pad_l"], 0).max(0) as usize,
                l.int_or(&["pad_r"], 0).max(0) as usize,
            ]
        };
        // Only zero-fill padding is modelled; a non-zero pad value would need a
        // different constant in the emitted `Pad`.
        let fill = l.int_or(&["pad_fill_mode"], 0);
        let pad_value = l
            .attrs
            .get("pad_value")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        ensure!(
            fill == 0 && pad_value == 0.0,
            "layer {:?}: pad_fill_mode={fill} pad_value={pad_value} — only zero padding is supported",
            l.name
        );

        match l.int(&["pad_mode", "padding_mode", "padMode"]) {
            // SAME: pad so the output is ceil(in / stride), extra pixel last.
            Some(1) => {
                let same = |inp: usize, k: usize, s: usize| -> (usize, usize) {
                    let out = inp.div_ceil(s);
                    let need = (out - 1) * s + k;
                    let total = need.saturating_sub(inp);
                    (total / 2, total - total / 2)
                };
                let (t, b) = same(input[0], kernel[0], stride[0]);
                let (le, r) = same(input[1], kernel[1], stride[1]);
                Ok([t, b, le, r])
            }
            // VALID.
            Some(2) => Ok([0; 4]),
            // Explicit pads (or none declared at all).
            Some(0) | None => {
                if l.attrs.contains_key("pad_x") || l.attrs.contains_key("pad_y") {
                    let py = l.int_or(&["pad_y"], 0).max(0) as usize;
                    let px = l.int_or(&["pad_x"], 0).max(0) as usize;
                    return Ok([py, py, px, px]);
                }
                Ok(explicit())
            }
            Some(other) => bail!(
                "layer {:?}: unknown `pad_mode` {other} \
                 (expected 0 = explicit pads, 1 = same, 2 = valid)",
                l.name
            ),
        }
    }

    fn push(&mut self, l: &EspressoLayer, op: OpDef) -> Result<()> {
        self.set_shape(l, &op.out, op.shape)?;
        self.ops.push(op);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fusion: hard-swish / hard-sigmoid chains
// ---------------------------------------------------------------------------

/// Espresso never emits a fused hard-swish. It spells every one out as four
/// elementwise layers:
///
/// ```text
///   a = x + 3
///   b = clamp(a, 0, 6)
///   c = b · 1/6            ← hard_sigmoid(x)
///   y = x · c              ← hard_swish(x), when the multiplicand is x itself
/// ```
///
/// The shipping model contains 28 such chains: 19 close with `x · c`
/// (hard-swish) and 9 multiply a *different* tensor — the squeeze-excite gate,
/// which is `hard_sigmoid(gate) · features`.
///
/// rlx-ir has both activations natively, so this collapses 4 ops → 1 (or 2 for
/// a gate). It is value-preserving by construction: the closed forms are the
/// same expression, and `spec::tests` plus the real-model parity test pin the
/// hashes together.
///
/// Chains are only fused when every intermediate is consumed exactly once, so
/// a graph that also reads `b` or `c` elsewhere is left alone.
fn fuse_hard_activations(ops: Vec<OpDef>, graph_output: &str) -> Vec<OpDef> {
    let mut uses: HashMap<&str, usize> = HashMap::new();
    for o in &ops {
        for i in &o.ins {
            *uses.entry(i.as_str()).or_default() += 1;
        }
    }
    // The graph output escapes, so it can never be treated as a dead
    // intermediate.
    *uses.entry(graph_output).or_default() += 1;
    let single_use = |blob: &str| uses.get(blob).copied().unwrap_or(0) == 1;

    let approx = |a: f32, b: f32| (a - b).abs() <= 1e-6 * b.abs().max(1.0);

    let mut out: Vec<OpDef> = Vec::with_capacity(ops.len());
    let mut i = 0usize;
    while i < ops.len() {
        // a = x + 3
        let plus3 = matches!(&ops[i].kind,
            OpKind::Affine { alpha, beta } if approx(*alpha, 1.0) && approx(*beta, 3.0));
        let chain = plus3
            && i + 2 < ops.len()
            && matches!(&ops[i + 1].kind,
                OpKind::Clamp { min, max } if approx(*min, 0.0) && approx(*max, 6.0))
            && matches!(&ops[i + 2].kind,
                OpKind::Affine { alpha, beta } if approx(*alpha, 1.0 / 6.0) && approx(*beta, 0.0))
            // …and they really are chained, each intermediate used once.
            && ops[i + 1].ins == [ops[i].out.clone()]
            && ops[i + 2].ins == [ops[i + 1].out.clone()]
            && single_use(&ops[i].out)
            && single_use(&ops[i + 1].out);

        if !chain {
            out.push(ops[i].clone());
            i += 1;
            continue;
        }

        let x = ops[i].ins[0].clone();
        let gate_out = ops[i + 2].out.clone();

        // Does the next op multiply the chain's own input by the gate? Then the
        // whole thing is hard_swish(x).
        let swish = i + 3 < ops.len()
            && matches!(&ops[i + 3].kind, OpKind::Elementwise { kind: EwKind::Mul })
            && ops[i + 3].ins.len() == 2
            && ops[i + 3].ins.contains(&x)
            && ops[i + 3].ins.contains(&gate_out)
            && single_use(&gate_out);

        if swish {
            let last = &ops[i + 3];
            out.push(OpDef {
                name: last.name.clone(),
                kind: OpKind::Activation {
                    act: Act::HardSwish,
                },
                ins: vec![x],
                out: last.out.clone(),
                shape: last.shape,
            });
            i += 4;
        } else {
            let last = &ops[i + 2];
            out.push(OpDef {
                name: last.name.clone(),
                kind: OpKind::Activation {
                    act: Act::HardSigmoid,
                },
                ins: vec![x],
                out: last.out.clone(),
                shape: last.shape,
            });
            i += 3;
        }
    }
    out
}

/// A unique, stable tensor-name prefix for an Espresso layer name.
///
/// Espresso reuses layer names freely, so repeats get a `#n` suffix. The
/// deriver calls this exactly once per non-`input` layer, in order;
/// [`crate::weights`] replays the same sequence to name the tensors it
/// extracts, and `spec_and_weights_agree_on_names` pins the two together.
pub(crate) fn tensor_base_for(used: &mut HashMap<String, usize>, name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| match c {
            '/' | ' ' | ':' | '@' => '.',
            c => c,
        })
        .collect();
    let n = used.entry(sanitized.clone()).or_insert(0);
    let base = if *n == 0 {
        sanitized.clone()
    } else {
        format!("{sanitized}#{n}")
    };
    *n += 1;
    base
}

/// Espresso layer types that carry no weights and produce no spec op.
pub(crate) fn is_input_layer(kind: &str) -> bool {
    matches!(kind, "input" | "input_layer")
}

/// Activation fused into a convolution by the Espresso optimizer.
fn fused_act(l: &EspressoLayer) -> Result<Option<Act>> {
    if l.flag(&["fused_relu"]) {
        return Ok(Some(Act::Relu));
    }
    if l.flag(&["fused_relu6"]) {
        return Ok(Some(Act::Relu6));
    }
    if l.flag(&["fused_tanh"]) {
        return Ok(Some(Act::Tanh));
    }
    if l.flag(&["fused_sigmoid"]) {
        return Ok(Some(Act::Sigmoid));
    }
    if l.flag(&["fused_hard_swish", "fused_hardswish"]) {
        return Ok(Some(Act::HardSwish));
    }
    Ok(None)
}

/// Activation for a standalone activation layer.
fn standalone_act(l: &EspressoLayer) -> Result<Act> {
    match l.kind.as_str() {
        "relu" => return Ok(Act::Relu),
        "relu6" => return Ok(Act::Relu6),
        "sigmoid" => return Ok(Act::Sigmoid),
        "tanh" => return Ok(Act::Tanh),
        "hard_sigmoid" | "hardsigmoid" => return Ok(Act::HardSigmoid),
        "hard_swish" | "hardswish" => return Ok(Act::HardSwish),
        "clip" => return Ok(Act::Relu6),
        "scale" => {
            let alpha = l.attrs.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let beta = l.attrs.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            return Ok(Act::Linear { alpha, beta });
        }
        _ => {}
    }
    // `type: "activation"` carries the function in a `mode` / `nonlinearity` field.
    if let Some(s) = l
        .attrs
        .get("nonlinearity")
        .or_else(|| l.attrs.get("mode"))
        .and_then(|v| v.as_str())
    {
        return match s.to_ascii_lowercase().as_str() {
            "relu" => Ok(Act::Relu),
            "relu6" => Ok(Act::Relu6),
            "sigmoid" => Ok(Act::Sigmoid),
            "tanh" => Ok(Act::Tanh),
            "hard_sigmoid" | "hardsigmoid" => Ok(Act::HardSigmoid),
            "hard_swish" | "hardswish" => Ok(Act::HardSwish),
            other => bail!("activation {:?}: unknown nonlinearity {other:?}", l.name),
        };
    }
    // Numeric `mode`. Mode 0 is ReLU: every `mode: 0` layer in the shipping
    // NeuralHash net is literally named `.../Relu`.
    if let Some(mode) = l.int(&["mode"]) {
        let beta = l.attrs.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.0);
        return match mode {
            0 => {
                ensure!(
                    beta == 0.0,
                    "activation {:?}: mode 0 (relu) with beta={beta} is not a plain ReLU",
                    l.name
                );
                Ok(Act::Relu)
            }
            other => bail!(
                "activation {:?}: unmapped numeric mode {other} (beta={beta}); \
                 attributes: {:?}",
                l.name,
                sorted_keys(l)
            ),
        };
    }
    bail!(
        "activation layer {:?} does not name its function; attributes: {:?}",
        l.name,
        sorted_keys(l)
    )
}

fn out_dim(input: usize, kernel: usize, stride: usize, pad_total: usize) -> Result<usize> {
    let padded = input + pad_total;
    ensure!(
        padded >= kernel,
        "kernel {kernel} exceeds the padded input extent {padded}"
    );
    Ok((padded - kernel) / stride + 1)
}

fn sorted_keys(l: &EspressoLayer) -> Vec<&str> {
    let mut k: Vec<&str> = l.attrs.keys().map(|s| s.as_str()).collect();
    k.sort_unstable();
    k
}

fn annotate(e: anyhow::Error, idx: usize, l: &EspressoLayer, total: usize) -> anyhow::Error {
    e.context(format!(
        "espresso layer {idx}/{total} {:?} (type {})",
        l.name, l.kind
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn net(layers_json: &str, shapes: &str) -> EspressoNet {
        let weights = {
            let mut v = 0u64.to_le_bytes().to_vec();
            v.truncate(8);
            v
        };
        let shapes = if shapes.is_empty() {
            HashMap::new()
        } else {
            let v: serde_json::Value = serde_json::from_str(shapes).unwrap();
            v.as_object()
                .unwrap()
                .iter()
                .map(|(k, s)| {
                    let g = |f: &str| s.get(f).and_then(|x| x.as_u64()).unwrap_or(1) as usize;
                    (
                        k.clone(),
                        BlobShape {
                            n: g("n"),
                            c: g("k"),
                            h: g("h"),
                            w: g("w"),
                        },
                    )
                })
                .collect()
        };
        EspressoNet::from_parts(
            &format!(r#"{{"format_version": 200, "layers": {layers_json}}}"#),
            &weights,
            shapes,
        )
        .unwrap()
    }

    #[test]
    fn derives_a_strided_padded_conv() {
        let n = net(
            r#"[
              {"type":"input","name":"image","top":"image","bottom":""},
              {"type":"convolution","name":"stem","top":"s","bottom":"image",
               "C":16,"K":3,"Nx":3,"Ny":3,"n_groups":1,"stride_x":2,"stride_y":2,
               "pad_t":1,"pad_b":1,"pad_l":1,"pad_r":1,"has_biases":1,
               "fused_relu":1,"blob_weights_f16":0,"blob_biases":1}
            ]"#,
            r#"{"image": {"n":1,"k":3,"h":360,"w":360}}"#,
        );
        let spec = NeuralHashSpec::from_espresso(&n).unwrap();
        assert_eq!(spec.input, "image");
        assert_eq!(spec.input_shape, [1, 3, 360, 360]);
        assert_eq!(spec.ops.len(), 1);
        // (360 + 2 - 3) / 2 + 1 = 180
        assert_eq!(spec.ops[0].shape, [1, 16, 180, 180]);
        match &spec.ops[0].kind {
            OpKind::Conv {
                act,
                pad,
                groups,
                bias,
                ..
            } => {
                assert_eq!(*act, Some(Act::Relu));
                assert_eq!(*pad, [1, 1, 1, 1]);
                assert_eq!(*groups, 1);
                assert_eq!(bias.as_deref(), Some("stem.bias"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn depthwise_conv_k_is_total_input_channels() {
        let n = net(
            r#"[
              {"type":"input","name":"i","top":"i","bottom":""},
              {"type":"convolution","name":"dw","top":"o","bottom":"i",
               "C":16,"K":16,"Nx":3,"Ny":3,"n_groups":16,"pad_mode":1,"has_biases":0,
               "blob_weights_f16":0}
            ]"#,
            r#"{"i": {"n":1,"k":16,"h":32,"w":32}}"#,
        );
        let spec = NeuralHashSpec::from_espresso(&n).unwrap();
        // pad_mode SAME with stride 1 → output keeps 32×32.
        assert_eq!(spec.ops[0].shape, [1, 16, 32, 32]);
        match &spec.ops[0].kind {
            OpKind::Conv {
                groups, pad, bias, ..
            } => {
                assert_eq!(*groups, 16);
                assert_eq!(*pad, [1, 1, 1, 1]);
                assert!(bias.is_none());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn wrong_declared_input_channels_is_caught() {
        let n = net(
            r#"[
              {"type":"input","name":"i","top":"i","bottom":""},
              {"type":"convolution","name":"c","top":"o","bottom":"i",
               "C":8,"K":7,"Nx":1,"Ny":1,"n_groups":1,"blob_weights_f16":0}
            ]"#,
            r#"{"i": {"n":1,"k":3,"h":8,"w":8}}"#,
        );
        let e = format!("{:#}", NeuralHashSpec::from_espresso(&n).unwrap_err());
        assert!(e.contains("declared K=7"), "{e}");
    }

    #[test]
    fn sidecar_disagreement_aborts() {
        // Declare a sidecar shape that our stride reading cannot produce.
        let n = net(
            r#"[
              {"type":"input","name":"i","top":"i","bottom":""},
              {"type":"convolution","name":"c","top":"o","bottom":"i",
               "C":8,"K":3,"Nx":3,"Ny":3,"n_groups":1,"stride_x":1,"stride_y":1,
               "blob_weights_f16":0}
            ]"#,
            r#"{"i": {"n":1,"k":3,"h":8,"w":8}, "o": {"n":1,"k":8,"h":8,"w":8}}"#,
        );
        let e = format!("{:#}", NeuralHashSpec::from_espresso(&n).unwrap_err());
        assert!(e.contains("sidecar declares"), "{e}");
    }

    #[test]
    fn global_pool_and_inner_product_tail() {
        let n = net(
            r#"[
              {"type":"input","name":"i","top":"i","bottom":""},
              {"type":"pool","name":"gap","top":"p","bottom":"i","avg_or_max":0,"is_global":1},
              {"type":"inner_product","name":"head","top":"e","bottom":"p","nB":64,"nC":128,
               "blob_weights_f16":0,"blob_biases":1}
            ]"#,
            r#"{"i": {"n":1,"k":64,"h":12,"w":12}}"#,
        );
        let spec = NeuralHashSpec::from_espresso(&n).unwrap();
        assert_eq!(spec.ops[0].shape, [1, 64, 1, 1]);
        assert_eq!(spec.ops[1].shape, [1, 128, 1, 1]);
        assert_eq!(spec.output, "e");
        assert_eq!(spec.output_dim(), 128);
        assert_eq!(
            spec.weight_names(),
            vec!["head.weight".to_string(), "head.bias".to_string()]
        );
    }

    #[test]
    fn squeeze_excite_broadcast_multiply() {
        let n = net(
            r#"[
              {"type":"input","name":"i","top":"i","bottom":""},
              {"type":"pool","name":"gap","top":"g","bottom":"i","avg_or_max":0,"is_global":1},
              {"type":"elementwise","name":"scale","top":"o","bottom":"i,g","operation":1}
            ]"#,
            r#"{"i": {"n":1,"k":32,"h":16,"w":16}}"#,
        );
        let spec = NeuralHashSpec::from_espresso(&n).unwrap();
        // The `input` layer produces no op, so pool is 0 and the multiply is 1.
        assert_eq!(spec.ops.len(), 2);
        assert_eq!(spec.ops[1].shape, [1, 32, 16, 16]);
        assert_eq!(spec.ops[1].kind, OpKind::Elementwise { kind: EwKind::Mul });
    }

    #[test]
    fn unsupported_layer_names_itself() {
        let n = net(
            r#"[
              {"type":"input","name":"i","top":"i","bottom":""},
              {"type":"lstm","name":"weird","top":"o","bottom":"i","some_attr":3}
            ]"#,
            r#"{"i": {"n":1,"k":3,"h":8,"w":8}}"#,
        );
        let e = format!("{:#}", NeuralHashSpec::from_espresso(&n).unwrap_err());
        assert!(
            e.contains("unsupported Espresso layer type \"lstm\""),
            "{e}"
        );
        assert!(e.contains("some_attr"), "{e}");
        assert!(e.contains("weird"), "{e}");
    }

    #[test]
    fn unknown_pool_mode_is_rejected_not_defaulted() {
        let n = net(
            r#"[
              {"type":"input","name":"i","top":"i","bottom":""},
              {"type":"pool","name":"p","top":"o","bottom":"i","avg_or_max":7,"is_global":1}
            ]"#,
            r#"{"i": {"n":1,"k":3,"h":8,"w":8}}"#,
        );
        let e = format!("{:#}", NeuralHashSpec::from_espresso(&n).unwrap_err());
        assert!(e.contains("avg_or_max"), "{e}");
    }

    #[test]
    fn duplicate_layer_names_get_distinct_tensors() {
        let n = net(
            r#"[
              {"type":"input","name":"i","top":"i","bottom":""},
              {"type":"convolution","name":"conv","top":"a","bottom":"i",
               "C":4,"K":3,"Nx":1,"Ny":1,"blob_weights_f16":0},
              {"type":"convolution","name":"conv","top":"b","bottom":"a",
               "C":4,"K":4,"Nx":1,"Ny":1,"blob_weights_f16":1}
            ]"#,
            r#"{"i": {"n":1,"k":3,"h":8,"w":8}}"#,
        );
        let spec = NeuralHashSpec::from_espresso(&n).unwrap();
        assert_eq!(spec.weight_names(), vec!["conv.weight", "conv#1.weight"]);
    }

    #[test]
    fn spec_json_roundtrips() {
        let n = net(
            r#"[
              {"type":"input","name":"i","top":"i","bottom":""},
              {"type":"convolution","name":"c","top":"o","bottom":"i",
               "C":4,"K":3,"Nx":1,"Ny":1,"has_biases":1,"blob_weights_f16":0,"blob_biases":1}
            ]"#,
            r#"{"i": {"n":1,"k":3,"h":8,"w":8}}"#,
        );
        let spec = NeuralHashSpec::from_espresso(&n).unwrap();
        let back = NeuralHashSpec::from_json(&spec.to_json().unwrap()).unwrap();
        assert_eq!(spec, back);
    }
}
