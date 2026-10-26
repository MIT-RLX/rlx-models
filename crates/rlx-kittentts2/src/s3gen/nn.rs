// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Weight store and graph-building helpers shared by the S3Gen graphs.
//!
//! Layouts: sequence tensors are `[T, C]` for linears and attention; 1-D
//! convolutions run on `[1, C, T, 1]` (length in H, kernel `[k, 1]`), the form
//! every backend's `Conv` lowering agrees on. Padding is always explicit
//! (concat of zeros / reflected columns) rather than the conv's own `padding`
//! argument, since backends disagree on which axis that pads in this layout.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{Activation, BinaryOp, Op, ReduceOp};
use rlx_ir::{DType, HirGraphExt, Shape};
use safetensors::{Dtype, SafeTensors};

pub const F32: DType = DType::F32;

/// The S3Gen tensors this crate runs (`flow.*`, `mel2wav.*`), as f32, with
/// `weight_norm` parametrizations folded into plain `weight`s.
/// How a BitNet-style quantised layer treats its input (`BitLinear` /
/// `BitConv1d` in the student decoder): optional `SimpleRMSNorm`, then
/// optional per-token int8 fake-quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitCfg {
    pub norm_input: bool,
    pub act_quant: bool,
}

pub struct S3GenWeights {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
    /// Weight keys of BitNet layers, and how their inputs are prepared.
    quantised: std::collections::HashSet<String>,
    bit: Option<BitCfg>,
    /// Every key a builder has read — so a test can prove no checkpoint
    /// tensor (a forgotten bias, say) is silently left out of the graphs.
    touched: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl S3GenWeights {
    /// Read `s3gen_meanflow.safetensors` (chatterbox-turbo), keeping only the
    /// flow and vocoder — the S3 tokenizer and CAM++ encoder are for cloning.
    pub fn open(path: &Path) -> Result<Self> {
        let w = Self::open_prefixes(path, &["flow.", "mel2wav."])?;
        ensure!(
            w.tensors
                .keys()
                .any(|k| k.starts_with("flow.decoder.estimator.time_embed_mixer")),
            "{} is not the meanflow S3Gen (no time_embed_mixer)",
            path.display()
        );
        Ok(w)
    }

    /// Read the f32 tensors under any of `prefixes`, folding weight norm.
    pub fn open_prefixes(path: &Path, prefixes: &[&str]) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes).context("parse S3Gen safetensors")?;
        let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for name in st.names() {
            if !prefixes.iter().any(|p| name.starts_with(p)) {
                continue;
            }
            let v = st.tensor(name)?;
            if v.dtype() != Dtype::F32 {
                // `num_batches_tracked` counters (I64) carry no inference state.
                continue;
            }
            let data = v
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            raw.insert(name.to_string(), (data, v.shape().to_vec()));
        }
        ensure!(
            !raw.is_empty(),
            "{}: no tensors under {prefixes:?}",
            path.display()
        );
        Ok(Self::from_tensors(fold_weight_norm(raw)?))
    }

    /// Every key under `prefix`, sorted.
    pub fn keys_with_prefix(&self, prefix: &str) -> Vec<String> {
        let mut v: Vec<String> = self
            .tensors
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        v.sort();
        v
    }

    pub fn from_tensors(tensors: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self {
            tensors,
            quantised: Default::default(),
            bit: None,
            touched: Default::default(),
        }
    }

    /// Mark `keys` (weight names) as BitNet layers whose inputs get `bit`.
    pub fn with_bit_layers(mut self, keys: impl IntoIterator<Item = String>, bit: BitCfg) -> Self {
        self.quantised.extend(keys);
        self.bit = Some(bit);
        self
    }

    /// The input treatment of the layer owning `weight_key`, if quantised.
    pub fn bit_layer(&self, weight_key: &str) -> Option<BitCfg> {
        self.bit.filter(|_| self.quantised.contains(weight_key))
    }

    /// Checkpoint tensors no graph or host step has read yet.
    pub fn untouched(&self) -> Vec<String> {
        let t = self.touched.lock().unwrap();
        let mut v: Vec<String> = self
            .tensors
            .keys()
            .filter(|k| !t.contains(*k))
            .cloned()
            .collect();
        v.sort();
        v
    }

    pub fn get(&self, name: &str) -> Result<&[f32]> {
        self.touched.lock().unwrap().insert(name.to_string());
        Ok(&self
            .tensors
            .get(name)
            .with_context(|| format!("S3Gen weight {name} missing"))?
            .0)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self
            .tensors
            .get(name)
            .with_context(|| format!("S3Gen weight {name} missing"))?
            .1)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// `w = g · v / ‖v‖` with the norm over every axis but 0 — torch's
/// `parametrizations.weight.original{0,1}` (`weight_norm(dim=0)`).
fn fold_weight_norm(
    mut raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let gs: Vec<String> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".parametrizations.weight.original0"))
        .map(str::to_string)
        .collect();
    for base in gs {
        let (g, _) = raw
            .remove(&format!("{base}.parametrizations.weight.original0"))
            .unwrap();
        let (v, shape) = raw
            .remove(&format!("{base}.parametrizations.weight.original1"))
            .with_context(|| format!("{base}: weight-norm direction missing"))?;
        let rows = shape[0];
        ensure!(g.len() == rows, "{base}: {} gains for {rows} rows", g.len());
        let per = v.len() / rows;
        let mut w = v;
        for (r, row) in w.chunks_exact_mut(per).enumerate() {
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let s = g[r] / n;
            row.iter_mut().for_each(|x| *x *= s);
        }
        raw.insert(format!("{base}.weight"), (w, shape));
    }
    Ok(raw)
}

/// HIR builder that interns every constant once under its checkpoint key.
pub struct Ctx<'a, 'b> {
    pub g: &'a mut HirMut<'b>,
    params: Vec<(String, Vec<f32>)>,
    interned: HashMap<String, HirNodeId>,
}

impl<'a, 'b> Ctx<'a, 'b> {
    pub fn new(g: &'a mut HirMut<'b>) -> Self {
        Self {
            g,
            params: Vec::new(),
            interned: HashMap::new(),
        }
    }

    pub fn into_params(self) -> Vec<(String, Vec<f32>)> {
        self.params
    }

    /// Intern `data` as a param under `key` (one node per key).
    pub fn param(
        &mut self,
        key: &str,
        shape: &[usize],
        data: impl FnOnce() -> Vec<f32>,
    ) -> HirNodeId {
        if let Some(&id) = self.interned.get(key) {
            return id;
        }
        let data = data();
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "param {key} shape {shape:?}"
        );
        let name = format!("p{}", self.params.len());
        let id = self.g.param(name.clone(), Shape::new(shape, F32));
        self.params.push((name, data));
        self.interned.insert(key.to_string(), id);
        id
    }

    /// A scalar broadcast to `shape`.
    pub fn scalar(&mut self, v: f32, shape: &[usize]) -> HirNodeId {
        let s = self.param(&format!("scalar:{v:e}"), &[1], || vec![v]);
        self.g.expand_(s, shape.iter().map(|&d| d as i64).collect())
    }

    pub fn act(&mut self, a: Activation, x: HirNodeId) -> HirNodeId {
        let s = self.g.shape(x).clone();
        self.g.activation(a, x, s)
    }

    /// `max(x, slope · x)`.
    pub fn leaky_relu(&mut self, x: HirNodeId, slope: f32) -> HirNodeId {
        let dims = dims(self, x);
        let a = self.scalar(slope, &dims);
        let ax = self.g.mul(x, a);
        let s = self.g.shape(x).clone();
        self.g.add_node(Op::Binary(BinaryOp::Max), vec![x, ax], s)
    }

    /// `y = x·Wᵀ + b` over the last axis of `x` (`[rows, in]` or `[1, T, in]`),
    /// with the checkpoint's `{key}.weight` (`[out, in]`) and optional bias.
    pub fn linear(&mut self, w: &S3GenWeights, x: HirNodeId, key: &str) -> Result<HirNodeId> {
        let shape = w.shape(&format!("{key}.weight"))?.to_vec();
        let [out, inp] = shape[..] else {
            bail!("{key}.weight has shape {shape:?}");
        };
        let wt = self.param(&format!("{key}.weight^T"), &[inp, out], || {
            rlx_core::weight_map::transpose_2d(w.get(&format!("{key}.weight")).unwrap(), out, inp)
        });
        let xd = dims(self, x);
        let rows: usize = xd[..xd.len() - 1].iter().product();
        ensure!(
            xd[xd.len() - 1] == inp,
            "{key}: input width {} != {inp}",
            xd[xd.len() - 1]
        );
        let x2 = self.g.reshape_(x, vec![rows as i64, inp as i64]);
        let mut y = self.g.mm(x2, wt);
        let bkey = format!("{key}.bias");
        if w.has(&bkey) {
            let b = self.param(&bkey, &[1, out], || w.get(&bkey).unwrap().to_vec());
            let be = self.g.expand_(b, vec![rows as i64, out as i64]);
            y = self.g.add(y, be);
        }
        let mut od: Vec<i64> = xd[..xd.len() - 1].iter().map(|&d| d as i64).collect();
        od.push(out as i64);
        Ok(self.g.reshape_(y, od))
    }

    /// LayerNorm over the last axis with `{key}.weight` / `{key}.bias`.
    pub fn layer_norm(
        &mut self,
        w: &S3GenWeights,
        x: HirNodeId,
        key: &str,
        eps: f32,
    ) -> Result<HirNodeId> {
        let n = *dims(self, x).last().unwrap();
        let gk = format!("{key}.weight");
        let bk = format!("{key}.bias");
        ensure!(w.get(&gk)?.len() == n, "{gk}: width mismatch");
        let g = self.param(&gk, &[n], || w.get(&gk).unwrap().to_vec());
        let b = self.param(&bk, &[n], || w.get(&bk).unwrap().to_vec());
        Ok(self.g.ln(x, g, b, eps))
    }

    /// `[1, C, T, 1]` → `[T, C]`.
    pub fn nct_to_tc(&mut self, x: HirNodeId) -> HirNodeId {
        let d = dims(self, x);
        let (c, t) = (d[1], d[2]);
        let r = self.g.reshape_(x, vec![c as i64, t as i64]);
        self.g.transpose_(r, vec![1, 0])
    }

    /// `[T, C]` → `[1, C, T, 1]`.
    pub fn tc_to_nct(&mut self, x: HirNodeId) -> HirNodeId {
        let d = dims(self, x);
        let (t, c) = (d[0], d[1]);
        let tr = self.g.transpose_(x, vec![1, 0]);
        self.g.reshape_(tr, vec![1, c as i64, t as i64, 1])
    }

    /// Zero-pad the length axis of `[1, C, T, 1]` by `(left, right)`.
    pub fn pad_len(&mut self, x: HirNodeId, left: usize, right: usize) -> HirNodeId {
        if left == 0 && right == 0 {
            return x;
        }
        let d = dims(self, x);
        let c = d[1];
        let mut parts = Vec::new();
        if left > 0 {
            parts.push(
                self.param(&format!("zeros:{c}x{left}"), &[1, c, left, 1], || {
                    vec![0.0; c * left]
                }),
            );
        }
        parts.push(x);
        if right > 0 {
            parts.push(
                self.param(&format!("zeros:{c}x{right}"), &[1, c, right, 1], || {
                    vec![0.0; c * right]
                }),
            );
        }
        self.g.concat_(parts, 2)
    }

    /// Reflect-pad the length axis of `[1, C, T, 1]` on the left by `n`
    /// (`ReflectionPad1d((n, 0))`: columns `n, n-1, …, 1`).
    pub fn reflect_pad_left(&mut self, x: HirNodeId, n: usize) -> HirNodeId {
        let mut parts: Vec<HirNodeId> = (1..=n).rev().map(|i| self.g.narrow_(x, 2, i, 1)).collect();
        parts.push(x);
        self.g.concat_(parts, 2)
    }

    /// `Conv1d` with weight `{key}.weight` `[C_out, C_in, K]` on `[1, C_in, T, 1]`,
    /// explicit zero padding `(left, right)`.
    pub fn conv1d(
        &mut self,
        w: &S3GenWeights,
        x: HirNodeId,
        key: &str,
        pad: (usize, usize),
        stride: usize,
        dilation: usize,
    ) -> Result<HirNodeId> {
        let wk = format!("{key}.weight");
        let shape = w.shape(&wk)?.to_vec();
        let [c_out, c_in, k] = shape[..] else {
            bail!("{wk} has shape {shape:?}");
        };
        let wid = self.param(&wk, &[c_out, c_in, k, 1], || w.get(&wk).unwrap().to_vec());
        self.conv_with(
            w,
            x,
            wid,
            &format!("{key}.bias"),
            [c_out, c_in, k],
            pad,
            stride,
            dilation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_with(
        &mut self,
        w: &S3GenWeights,
        x: HirNodeId,
        weight: HirNodeId,
        bias_key: &str,
        [c_out, c_in, k]: [usize; 3],
        pad: (usize, usize),
        stride: usize,
        dilation: usize,
    ) -> Result<HirNodeId> {
        let d = dims(self, x);
        ensure!(
            d[1] == c_in,
            "conv over {} channels, weight wants {c_in}",
            d[1]
        );
        let xp = self.pad_len(x, pad.0, pad.1);
        let t = d[2] + pad.0 + pad.1;
        let span = dilation * (k - 1) + 1;
        ensure!(
            t >= span,
            "conv input of {t} shorter than its {span}-tap span"
        );
        let t_out = (t - span) / stride + 1;
        let mut y = self.g.add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![stride, 1],
                padding: vec![0, 0],
                dilation: vec![dilation, 1],
                groups: 1,
            },
            vec![xp, weight],
            Shape::new(&[1, c_out, t_out, 1], F32),
        );
        if w.has(bias_key) {
            let b = self.param(bias_key, &[1, c_out, 1, 1], || {
                w.get(bias_key).unwrap().to_vec()
            });
            let be = self.g.expand_(b, vec![1, c_out as i64, t_out as i64, 1]);
            y = self.g.add(y, be);
        }
        Ok(y)
    }

    /// `ConvTranspose1d(stride, padding)` with torch weight `[C_in, C_out, K]`,
    /// as zero-insertion + a stride-1 conv with the flipped kernel.
    pub fn conv_transpose1d(
        &mut self,
        w: &S3GenWeights,
        x: HirNodeId,
        key: &str,
        stride: usize,
        padding: usize,
    ) -> Result<HirNodeId> {
        let wk = format!("{key}.weight");
        let shape = w.shape(&wk)?.to_vec();
        let [c_in, c_out, k] = shape[..] else {
            bail!("{wk} has shape {shape:?}");
        };
        ensure!(padding < k, "{key}: padding {padding} ≥ kernel {k}");
        let wid = self.param(&format!("{wk}^flip"), &[c_out, c_in, k, 1], || {
            let src = w.get(&wk).unwrap();
            let mut out = vec![0f32; c_out * c_in * k];
            for i in 0..c_in {
                for o in 0..c_out {
                    for j in 0..k {
                        out[(o * c_in + i) * k + j] = src[(i * c_out + o) * k + (k - 1 - j)];
                    }
                }
            }
            out
        });
        let d = dims(self, x);
        let t = d[2];
        let up = if stride > 1 {
            let z = self.param(
                &format!("zeros:{c_in}x{t}x{}", stride - 1),
                &[1, c_in, t, stride - 1],
                || vec![0.0; c_in * t * (stride - 1)],
            );
            let x4 = self.g.reshape_(x, vec![1, c_in as i64, t as i64, 1]);
            let cat = self.g.concat_(vec![x4, z], 3);
            let r = self
                .g
                .reshape_(cat, vec![1, c_in as i64, (t * stride) as i64, 1]);
            self.g.narrow_(r, 2, 0, (t - 1) * stride + 1)
        } else {
            x
        };
        let p = k - 1 - padding;
        self.conv_with(
            w,
            up,
            wid,
            &format!("{key}.bias"),
            [c_out, c_in, k],
            (p, p),
            1,
            1,
        )
    }
}

impl Ctx<'_, '_> {
    /// `BitLinear` / `BitConv1d` input: `SimpleRMSNorm` (L2-normalize ×
    /// `dim^-½`) if configured, then per-token int8 fake-quantization
    /// `clamp(round(x · s), −128, 127) / s` with `s = 127 / max(|x|, 1e-5)`,
    /// both over `axis`.
    pub fn bit_input(&mut self, x: HirNodeId, axis: usize, bit: BitCfg) -> HirNodeId {
        let d = dims(self, x);
        let full: Vec<i64> = d.iter().map(|&v| v as i64).collect();
        let mut red = d.clone();
        red[axis] = 1;
        let reduce = |ctx: &mut Self, op: ReduceOp, v: HirNodeId| {
            ctx.g.add_node(
                Op::Reduce {
                    op,
                    axes: vec![axis],
                    keep_dim: true,
                },
                vec![v],
                Shape::new(&red, F32),
            )
        };
        let mut x = x;
        if bit.norm_input {
            let sq = self.g.mul(x, x);
            let ss = reduce(self, ReduceOp::Sum, sq);
            let n = self.g.sqrt(ss);
            let n = self.g.add_node(
                Op::Clamp {
                    min: 1e-12,
                    max: f32::INFINITY,
                },
                vec![n],
                Shape::new(&red, F32),
            );
            let n = self.g.expand_(n, full.clone());
            let k = self.scalar((d[axis] as f32).powf(-0.5), &d);
            let y = self.g.div(x, n);
            x = self.g.mul(y, k);
        }
        if bit.act_quant {
            let a = self.act(Activation::Abs, x);
            let m = reduce(self, ReduceOp::Max, a);
            let m = self.g.add_node(
                Op::Clamp {
                    min: 1e-5,
                    max: f32::INFINITY,
                },
                vec![m],
                Shape::new(&red, F32),
            );
            let c127 = self.scalar(127.0, &red);
            let s = self.g.div(c127, m);
            let s = self.g.expand_(s, full);
            let q = self.g.mul(x, s);
            let q = self.act(Activation::Round, q);
            let q = self.g.add_node(
                Op::Clamp {
                    min: -128.0,
                    max: 127.0,
                },
                vec![q],
                Shape::new(&d, F32),
            );
            x = self.g.div(q, s);
        }
        x
    }
}

/// Host twin of [`Ctx::bit_input`] for one vector.
pub fn bit_input_host(x: &[f32], bit: BitCfg) -> Vec<f32> {
    let mut v = x.to_vec();
    if bit.norm_input {
        let n = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
        let k = (v.len() as f32).powf(-0.5);
        v.iter_mut().for_each(|a| *a = *a / n * k);
    }
    if bit.act_quant {
        let s = 127.0 / v.iter().fold(0f32, |m, a| m.max(a.abs())).max(1e-5);
        v.iter_mut()
            .for_each(|a| *a = (*a * s).round_ties_even().clamp(-128.0, 127.0) / s);
    }
    v
}

/// Static dims of a node.
pub fn dims(ctx: &Ctx, x: HirNodeId) -> Vec<usize> {
    let s = ctx.g.shape(x);
    (0..s.rank()).map(|i| s.dim(i).unwrap_static()).collect()
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;
    use rlx_ir::hir::HirModule;
    use rlx_runtime::{Device, Session};

    /// Build a graph with one `x` input, run it on CPU, return the output.
    pub fn run(
        w: &S3GenWeights,
        x_shape: &[usize],
        x: &[f32],
        build: impl FnOnce(&mut Ctx, &S3GenWeights, HirNodeId) -> HirNodeId,
    ) -> Vec<f32> {
        let mut hir = HirModule::new("t");
        let mut g = HirMut::new(&mut hir);
        let mut ctx = Ctx::new(&mut g);
        let xi = ctx.g.input("x", Shape::new(x_shape, F32));
        let out = build(&mut ctx, w, xi);
        let params = ctx.into_params();
        hir.set_outputs(vec![out]);
        let graph = rlx_ir::Graph::from_hir(hir).unwrap();
        let mut c = Session::new(Device::Cpu).compile(graph);
        for (n, d) in &params {
            c.set_param(n, d);
        }
        c.run(&[("x", x)]).into_iter().next().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::run;
    use super::*;

    fn store(entries: &[(&str, Vec<f32>, Vec<usize>)]) -> S3GenWeights {
        S3GenWeights::from_tensors(
            entries
                .iter()
                .map(|(k, v, s)| (k.to_string(), (v.clone(), s.clone())))
                .collect(),
        )
    }

    /// Direct `ConvTranspose1d` reference: y[o, t·s + j − p] += x[i, t]·W[i, o, j].
    fn conv_t_ref(
        x: &[f32],
        c_in: usize,
        t: usize,
        w: &[f32],
        c_out: usize,
        k: usize,
        s: usize,
        p: usize,
    ) -> Vec<f32> {
        let full = (t - 1) * s + k;
        let mut y = vec![0f32; c_out * full];
        for i in 0..c_in {
            for ti in 0..t {
                for o in 0..c_out {
                    for j in 0..k {
                        y[o * full + ti * s + j] += x[i * t + ti] * w[(i * c_out + o) * k + j];
                    }
                }
            }
        }
        let out_len = full - 2 * p;
        (0..c_out)
            .flat_map(|o| y[o * full + p..o * full + p + out_len].to_vec())
            .collect()
    }

    #[test]
    fn conv_transpose_matches_the_direct_definition() {
        let (c_in, c_out, k, s, p, t) = (3, 2, 7, 3, 2, 5);
        let w: Vec<f32> = (0..c_in * c_out * k)
            .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.1)
            .collect();
        let x: Vec<f32> = (0..c_in * t)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
            .collect();
        let st = store(&[("u.weight", w.clone(), vec![c_in, c_out, k])]);
        let got = run(&st, &[1, c_in, t, 1], &x, |ctx, w, x| {
            ctx.conv_transpose1d(w, x, "u", s, p).unwrap()
        });
        let want = conv_t_ref(&x, c_in, t, &w, c_out, k, s, p);
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn weight_norm_folds_per_output_row() {
        let raw: HashMap<_, _> = [
            (
                "c.parametrizations.weight.original0".to_string(),
                (vec![2.0, 3.0], vec![2, 1, 1]),
            ),
            (
                "c.parametrizations.weight.original1".to_string(),
                (vec![3.0, 4.0, 0.0, 5.0], vec![2, 1, 2]),
            ),
        ]
        .into_iter()
        .collect();
        let f = fold_weight_norm(raw).unwrap();
        assert_eq!(f["c.weight"].0, vec![1.2, 1.6, 0.0, 3.0]);
    }

    #[test]
    fn reflect_pad_and_leaky_relu() {
        let st = store(&[]);
        let got = run(&st, &[1, 1, 4, 1], &[1.0, -2.0, 3.0, -4.0], |ctx, _, x| {
            let p = ctx.reflect_pad_left(x, 1);
            ctx.leaky_relu(p, 0.1)
        });
        assert_eq!(got, vec![-0.2, 1.0, -0.2, 3.0, -0.4]);
    }
}
