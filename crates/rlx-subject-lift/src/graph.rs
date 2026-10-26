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

//! Vendor network → RLX graph for the op set the vendor's subject-lifting network uses.
//!
//! The format stores layers in execution order, each naming its input and output
//! blobs; the builder walks them once, mapping blob name → HIR node + NCHW
//! shape. Weights are per-output-channel quantized u8 (`W_U8`, with f32
//! `per_ch_qscale` / `per_ch_qbias`: `w = q·scale + bias`), dequantized to f32
//! once at build time.
//!
//! Supported (everything `subject_lifting_gen1_rev5_*` contains): convolution
//! (any kernel / stride / groups, fused ReLU), activation `mode=3` (sigmoid),
//! elementwise `op 0` (add) / `1` (multiply) / `119` (clip, ReLU6 here),
//! global average pool, ×2 upsample, channel concat. Anything else is an error
//! naming the layer, not a silent skip.

use anyhow::{Context, Result, bail, ensure};
use rlx_flow::{BuiltModel, Emit, MapWeights, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_neuralhash::netfile::{NetFile, NetLayer};
use std::collections::HashMap;

/// How the graph realizes the format's `upsample` (settled by parity against
/// the Vision framework; see the crate docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Upsample {
    Nearest,
    Bilinear,
}

#[derive(Clone, Copy)]
struct Blob {
    id: HirNodeId,
    c: usize,
    h: usize,
    w: usize,
}

fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Dequantized kernel `[C_out, C_in/groups, kH, kW]` and bias `[C_out]`.
pub fn conv_weights(net: &NetFile, l: &NetLayer) -> Result<(Vec<f32>, Vec<f32>)> {
    let c_out = l.req_int(&["C"])? as usize;
    let w = l
        .attrs
        .get("weights")
        .and_then(|w| w.as_object())
        .context("conv without weights")?;
    let idx = |k: &str| w.get(k).and_then(|v| v.as_u64());
    let kernel = match (idx("W_U8"), idx("per_ch_qscale"), idx("per_ch_qbias")) {
        (Some(q), Some(s), Some(b)) => {
            let q = net.blob_bytes(q)?;
            let (s, b) = (f32s(net.blob_bytes(s)?), f32s(net.blob_bytes(b)?));
            ensure!(
                s.len() == c_out && b.len() == c_out,
                "{}: per-channel params != C",
                l.name
            );
            let per = q.len() / c_out;
            q.iter()
                .enumerate()
                .map(|(i, &v)| v as f32 * s[i / per] + b[i / per])
                .collect()
        }
        _ => match l.blob(&["blob_weights", "W"]) {
            Some(i) => net.blob_f16(i)?,
            None => bail!("{}: unsupported weight encoding {w:?}", l.name),
        },
    };
    let bias = if l.flag(&["has_biases"]) {
        f32s(
            net.blob_bytes(
                l.blob(&["blob_biases"])
                    .context("has_biases without blob_biases")?,
            )?,
        )
    } else {
        vec![0.0; c_out]
    };
    Ok((kernel, bias))
}

struct Builder<'e, 'a> {
    emit: &'e mut Emit<'a>,
    blobs: HashMap<String, Blob>,
    n: usize,
}

impl Builder<'_, '_> {
    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.emit.hir())
    }

    fn param(&mut self, data: Vec<f32>, dims: &[usize]) -> HirNodeId {
        self.n += 1;
        let name = format!("lift.p{}", self.n);
        self.emit
            .synth_param(&name, data, Shape::new(dims, DType::F32))
    }

    fn get(&self, name: &str) -> Result<Blob> {
        self.blobs
            .get(name)
            .copied()
            .with_context(|| format!("blob {name:?} used before it is produced"))
    }

    /// `b` broadcast to `a`'s shape (squeeze-excite gates are `[1, C, 1, 1]`).
    fn broadcast(&mut self, b: Blob, a: Blob) -> HirNodeId {
        if (b.h, b.w) == (a.h, a.w) && b.c == a.c {
            return b.id;
        }
        self.g()
            .expand_(b.id, vec![1, a.c as i64, a.h as i64, a.w as i64])
    }

    fn layer(
        &mut self,
        l: &NetLayer,
        weights: Option<&(Vec<f32>, Vec<f32>)>,
        up: Upsample,
    ) -> Result<Blob> {
        let ins: Vec<Blob> = l
            .bottom
            .iter()
            .map(|b| self.get(b))
            .collect::<Result<_>>()?;
        let x = ins.first().copied().context("layer without inputs")?;
        Ok(match l.kind.as_str() {
            "convolution" => {
                let c_out = l.req_int(&["C"])? as usize;
                let groups = l.int_or(&["n_groups"], 1) as usize;
                let (kx, ky) = (l.int_or(&["Nx"], 1) as usize, l.int_or(&["Ny"], 1) as usize);
                let (sx, sy) = (
                    l.int_or(&["stride_x"], 1) as usize,
                    l.int_or(&["stride_y"], 1) as usize,
                );
                let (pl, pr, pt, pb) = (
                    l.int_or(&["pad_l"], 0) as usize,
                    l.int_or(&["pad_r"], 0) as usize,
                    l.int_or(&["pad_t"], 0) as usize,
                    l.int_or(&["pad_b"], 0) as usize,
                );
                ensure!(pl == pr && pt == pb, "{}: asymmetric padding", l.name);
                ensure!(
                    l.int_or(&["pad_mode"], 0) == 0,
                    "{}: only zero padding",
                    l.name
                );
                let (kernel, bias) = weights.cloned().context("conv weights not prepared")?;
                ensure!(
                    kernel.len() == c_out * (x.c / groups) * ky * kx,
                    "{}: kernel size",
                    l.name
                );
                let (oh, ow) = ((x.h + 2 * pt - ky) / sy + 1, (x.w + 2 * pl - kx) / sx + 1);
                let w = self.param(kernel, &[c_out, x.c / groups, ky, kx]);
                let b = self.param(bias, &[1, c_out, 1, 1]);
                let relu = l.flag(&["fused_relu"]);
                let mut g = self.g();
                let y = g.conv2d(
                    x.id,
                    w,
                    [ky, kx],
                    [sy, sx],
                    [pt, pl],
                    groups,
                    Shape::new(&[1, c_out, oh, ow], DType::F32),
                );
                let b = g.expand_(b, vec![1, c_out as i64, oh as i64, ow as i64]);
                let y = g.add(y, b);
                let y = if relu { g.relu(y) } else { y };
                Blob {
                    id: y,
                    c: c_out,
                    h: oh,
                    w: ow,
                }
            }
            "activation" => match l.int_or(&["mode"], -1) {
                3 => Blob {
                    id: self.g().activation(
                        rlx_ir::op::Activation::Sigmoid,
                        x.id,
                        Shape::new(&[1, x.c, x.h, x.w], DType::F32),
                    ),
                    ..x
                },
                m => bail!("{}: activation mode {m} not supported", l.name),
            },
            "elementwise" => match (l.int_or(&["operation"], -1), ins.len()) {
                (0, 2) | (1, 2) => {
                    let (a, b) = if ins[0].h * ins[0].w >= ins[1].h * ins[1].w {
                        (ins[0], ins[1])
                    } else {
                        (ins[1], ins[0])
                    };
                    let bb = self.broadcast(b, a);
                    let mul = l.int_or(&["operation"], 0) == 1;
                    let mut g = self.g();
                    Blob {
                        id: if mul {
                            g.mul(a.id, bb)
                        } else {
                            g.add(a.id, bb)
                        },
                        ..a
                    }
                }
                (119, 1) => {
                    // Clip to [alpha, beta]; this net only uses [0, 6] (ReLU6).
                    let lo = l.attrs.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let hi = l.attrs.get("beta").and_then(|v| v.as_f64()).unwrap_or(6.0) as f32;
                    ensure!(lo == 0.0, "{}: clip lower bound {lo} unsupported", l.name);
                    let six = self.param(vec![hi], &[1, 1, 1, 1]);
                    let mut g = self.g();
                    let six = g.expand_(six, vec![1, x.c as i64, x.h as i64, x.w as i64]);
                    let r = g.relu(x.id);
                    let over = g.sub(r, six);
                    let over = g.relu(over);
                    Blob {
                        id: g.sub(r, over),
                        ..x
                    }
                }
                (op, n) => bail!(
                    "{}: elementwise op {op} with {n} inputs not supported",
                    l.name
                ),
            },
            "pool" => {
                ensure!(
                    l.int_or(&["is_global"], 0) == 1 && l.int_or(&["avg_or_max"], 0) == 0,
                    "{}: only global average pooling",
                    l.name
                );
                Blob {
                    id: self.g().mean(x.id, vec![2, 3], true),
                    h: 1,
                    w: 1,
                    ..x
                }
            }
            "upsample" => {
                let s = l.int_or(&["scaling_factor_x"], 2) as usize;
                ensure!(
                    s == 2 && l.int_or(&["scaling_factor_y"], 2) == 2,
                    "{}: only ×2 upsample",
                    l.name
                );
                let id = match up {
                    Upsample::Nearest => self.g().resize_nearest_2x(x.id),
                    Upsample::Bilinear => self.g().resize_bilinear2d(
                        x.id,
                        2 * x.h,
                        2 * x.w,
                        l.flag(&["align_corners"]),
                    ),
                };
                Blob {
                    id,
                    h: 2 * x.h,
                    w: 2 * x.w,
                    ..x
                }
            }
            "concat" => {
                let c = ins.iter().map(|b| b.c).sum();
                Blob {
                    id: self.g().concat_(ins.iter().map(|b| b.id).collect(), 1),
                    c,
                    ..x
                }
            }
            k => bail!("{}: network layer kind {k:?} not supported", l.name),
        })
    }
}

/// Build the graph: input `input` `[1, 3, side, side]`, output the
/// flattened `outputs` concatenated in order (each squeezed to 1-D).
pub fn build(
    net: &NetFile,
    input: &str,
    side: usize,
    outputs: &[&str],
    up: Upsample,
) -> Result<BuiltModel> {
    // Everything the plugin needs, owned: layers plus their dequantized weights.
    let layers: Vec<(NetLayer, Option<(Vec<f32>, Vec<f32>)>)> = net
        .layers
        .iter()
        .map(|l| -> Result<_> {
            let w = if l.kind == "convolution" {
                Some(conv_weights(net, l)?)
            } else {
                None
            };
            Ok((l.clone(), w))
        })
        .collect::<Result<_>>()?;
    let input = input.to_string();
    let outs: Vec<String> = outputs.iter().map(|s| s.to_string()).collect();
    ModelFlow::new("subject_lift")
        .input(&input, Shape::new(&[1, 3, side, side], DType::F32))
        .plugin_named("espresso", move |emit, _| {
            let x = emit.flow_input(&input)?.hir_id();
            let mut b = Builder {
                emit,
                blobs: HashMap::new(),
                n: 0,
            };
            b.blobs.insert(
                input.clone(),
                Blob {
                    id: x,
                    c: 3,
                    h: side,
                    w: side,
                },
            );
            for (l, w) in &layers {
                let out = b
                    .layer(l, w.as_ref(), up)
                    .with_context(|| format!("layer {} ({})", l.name, l.kind))?;
                for t in &l.top {
                    b.blobs.insert(t.clone(), out);
                }
            }
            let mut flat = Vec::new();
            let mut n = 0;
            for o in &outs {
                let blob = b.get(o)?;
                let len = blob.c * blob.h * blob.w;
                n += len;
                flat.push(b.g().reshape_(blob.id, vec![len as i64]));
            }
            let out = if flat.len() == 1 {
                flat[0]
            } else {
                b.g().concat_(flat, 0)
            };
            Ok(Some(b.emit.wrap(out, Shape::new(&[n], DType::F32))))
        })
        .output("heads")
        .build(&mut MapWeights::default())
}
