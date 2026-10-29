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

//! A host Whisper text decoder for beam search: one token per beam per
//! step against a per-beam self-attention cache and a shared cross-attention
//! cache. Decoding a handful of beams is bound by streaming the decoder's
//! weights (≈0.9 B parameters), so each linear layer reads every weight row
//! once for all beams, across threads; nothing is recompiled as the
//! sequence grows and beams are reordered by moving cache rows.
//!
//! Weights come straight from the HuggingFace `model.safetensors`
//! (`model.decoder.*`, f16 widened to f32). The layer math is
//! `WhisperDecoderLayer`'s: pre-LN self-attention, cross-attention, GELU
//! (erf) MLP; tied output projection.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::path::Path;

struct Lin {
    w: Vec<f32>, // [out, in]
    b: Option<Vec<f32>>,
    out: usize,
    inp: usize,
}

struct Ln {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct Layer {
    sa_ln: Ln,
    /// `q`, `k`, `v` stacked: one pass over the layer input.
    qkv: Lin,
    o: Lin,
    ca_ln: Ln,
    cq: Lin,
    co: Lin,
    ff_ln: Ln,
    fc1: Lin,
    fc2: Lin,
}

pub struct FastDecoder {
    embed: Vec<f32>, // [vocab, d]
    pos: Vec<f32>,   // [max_pos, d]
    layers: Vec<Layer>,
    ln: Ln,
    pub d: usize,
    pub heads: usize,
    pub vocab: usize,
    pub max_pos: usize,
}

/// Per-beam self-attention K/V for every layer: `[layer][beam] -> [t * d]`.
#[derive(Clone)]
pub struct SelfCache {
    pub k: Vec<Vec<Vec<f32>>>,
    pub v: Vec<Vec<Vec<f32>>>,
    pub len: usize,
}

impl SelfCache {
    pub fn new(layers: usize, beams: usize) -> Self {
        SelfCache {
            k: vec![vec![Vec::new(); beams]; layers],
            v: vec![vec![Vec::new(); beams]; layers],
            len: 0,
        }
    }
    /// Beam `b` takes beam `parents[b]`'s history.
    pub fn reorder(&mut self, parents: &[usize]) {
        if parents.iter().enumerate().all(|(b, &p)| b == p) {
            return;
        }
        for l in 0..self.k.len() {
            let k: Vec<Vec<f32>> = parents.iter().map(|&p| self.k[l][p].clone()).collect();
            let v: Vec<Vec<f32>> = parents.iter().map(|&p| self.v[l][p].clone()).collect();
            self.k[l] = k;
            self.v[l] = v;
        }
    }
}

fn f32_tensor(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
    let data = t.data();
    let v = match t.dtype() {
        safetensors::Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        d => bail!("{name}: dtype {d:?}"),
    };
    Ok((v, t.shape().to_vec()))
}

/// Four weight rows against `B` beams: each input chunk is loaded once for
/// the four rows, so the loop is bound by its fused multiply-adds.
#[inline(always)]
fn rows4<const B: usize>(w: &[f32], x: &[f32], inp: usize, out: &mut [f32]) {
    assert!(w.len() >= 4 * inp && x.len() >= B * inp && out.len() >= 4 * B && inp % 4 == 0);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: the assert bounds every 4-lane load below.
    unsafe {
        use core::arch::aarch64::*;
        let mut acc = [[vdupq_n_f32(0.0); B]; 4];
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let mut k = 0;
        while k < inp {
            let wv = [
                vld1q_f32(wp.add(k)),
                vld1q_f32(wp.add(inp + k)),
                vld1q_f32(wp.add(2 * inp + k)),
                vld1q_f32(wp.add(3 * inp + k)),
            ];
            for b in 0..B {
                let xv = vld1q_f32(xp.add(b * inp + k));
                for r in 0..4 {
                    acc[r][b] = vfmaq_f32(acc[r][b], wv[r], xv);
                }
            }
            k += 4;
        }
        for r in 0..4 {
            for b in 0..B {
                out[r * B + b] = vaddvq_f32(acc[r][b]);
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for r in 0..4 {
        for b in 0..B {
            let (wr, xb) = (&w[r * inp..(r + 1) * inp], &x[b * inp..(b + 1) * inp]);
            out[r * B + b] = wr.iter().zip(xb).fold(0f32, |a, (p, q)| p.mul_add(*q, a));
        }
    }
}

fn linear_rows<const B: usize>(w: &[f32], x: &[f32], inp: usize, out_rows: usize) -> Vec<f32> {
    let mut yt = vec![0f32; out_rows * B]; // [out, B]
    const ROWS: usize = 32;
    yt.par_chunks_mut(ROWS * B)
        .enumerate()
        .for_each(|(blk, dst)| {
            let r0 = blk * ROWS;
            for (g, d) in dst.chunks_mut(4 * B).enumerate() {
                let r = r0 + 4 * g;
                rows4::<B>(&w[r * inp..(r + 4) * inp], x, inp, d);
            }
        });
    yt
}

/// `y[b] = W x[b] + bias` for `beams` rows of `x`, each weight row read once.
fn linear(l: &Lin, x: &[f32], beams: usize) -> Vec<f32> {
    let (inp, out) = (l.inp, l.out);
    assert!(out % 4 == 0 && inp % 4 == 0);
    let yt = match beams {
        1 => linear_rows::<1>(&l.w, x, inp, out),
        2 => linear_rows::<2>(&l.w, x, inp, out),
        3 => linear_rows::<3>(&l.w, x, inp, out),
        4 => linear_rows::<4>(&l.w, x, inp, out),
        5 => linear_rows::<5>(&l.w, x, inp, out),
        6 => linear_rows::<6>(&l.w, x, inp, out),
        7 => linear_rows::<7>(&l.w, x, inp, out),
        8 => linear_rows::<8>(&l.w, x, inp, out),
        _ => panic!("FastDecoder: at most 8 beams"),
    };
    let mut y = vec![0f32; beams * out];
    for o in 0..out {
        let bias = l.b.as_ref().map_or(0.0, |b| b[o]);
        for b in 0..beams {
            y[b * out + o] = yt[o * beams + b] + bias;
        }
    }
    y
}

fn layer_norm(ln: &Ln, x: &[f32], d: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (xr, yr) in x.chunks(d).zip(y.chunks_mut(d)) {
        let mean = xr.iter().sum::<f32>() / d as f32;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..d {
            yr[i] = (xr[i] - mean) * inv * ln.w[i] + ln.b[i];
        }
    }
    y
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len() && a.len() % 16 == 0);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: equal lengths, a multiple of 16, bound every load.
    unsafe {
        use core::arch::aarch64::*;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i < a.len() {
            for j in 0..4 {
                acc[j] = vfmaq_f32(
                    acc[j],
                    vld1q_f32(pa.add(i + 4 * j)),
                    vld1q_f32(pb.add(i + 4 * j)),
                );
            }
            i += 16;
        }
        vaddvq_f32(vaddq_f32(
            vaddq_f32(acc[0], acc[1]),
            vaddq_f32(acc[2], acc[3]),
        ))
    }
    #[cfg(not(target_arch = "aarch64"))]
    a.iter().zip(b).fold(0f32, |s, (x, y)| x.mul_add(*y, s))
}

/// Cephes `expf` for `x <= 0` (softmax after the max is subtracted):
/// Cody–Waite reduction by ln 2, degree-5 polynomial; ~1 ulp, branch-free
/// so a row of scores vectorizes (libm's scalar `expf` dominated the
/// cross-attention cost: 7 500 calls per head per layer per step).
#[inline(always)]
fn expf_nonpos(x: f32) -> f32 {
    let x = x.max(-87.3);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = n.mul_add(2.121_944_4e-4, n.mul_add(-0.693_359_4, x));
    let mut p = 1.987_569_1e-4f32;
    p = p.mul_add(r, 1.398_199_9e-3);
    p = p.mul_add(r, 8.333_452e-3);
    p = p.mul_add(r, 4.166_579_6e-2);
    p = p.mul_add(r, 1.666_666_5e-1);
    p = p.mul_add(r, 5.000_000_1e-1);
    let e = (p * r).mul_add(r, r) + 1.0;
    e * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// `o += p * v` over equal-length slices (a multiple of 4).
#[inline(always)]
fn axpy(o: &mut [f32], p: f32, v: &[f32]) {
    assert!(o.len() == v.len() && o.len() % 4 == 0);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: equal lengths, a multiple of 4, bound every access.
    unsafe {
        use core::arch::aarch64::*;
        let (po, pv) = (o.as_mut_ptr(), v.as_ptr());
        let pp = vdupq_n_f32(p);
        let mut i = 0;
        while i < o.len() {
            vst1q_f32(
                po.add(i),
                vfmaq_f32(vld1q_f32(po.add(i)), pp, vld1q_f32(pv.add(i))),
            );
            i += 4;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (x, y) in o.iter_mut().zip(v) {
        *x = p.mul_add(*y, *x);
    }
}

/// Per-head attention for one query per beam. `keys[b]`/`vals[b]` are
/// `[t, d]`; when every beam shares the same buffers (cross-attention) each
/// key/value row is read once for all beams.
fn attend(q: &[f32], keys: &[&[f32]], vals: &[&[f32]], d: usize, heads: usize, out: &mut [f32]) {
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let beams = keys.len();
    let shared = keys
        .windows(2)
        .all(|w| std::ptr::eq(w[0].as_ptr(), w[1].as_ptr()));
    let per_head: Vec<Vec<f32>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let mut o = vec![0f32; beams * hd];
            let groups: Vec<Vec<usize>> = if shared {
                vec![(0..beams).collect()]
            } else {
                (0..beams).map(|b| vec![b]).collect()
            };
            for g in groups {
                let (k, v) = (keys[g[0]], vals[g[0]]);
                let t = k.len() / d;
                let mut sc = vec![0f32; g.len() * t];
                for j in 0..t {
                    let kr = &k[j * d + h * hd..j * d + (h + 1) * hd];
                    for (gi, &b) in g.iter().enumerate() {
                        sc[gi * t + j] = dot(&q[b * d + h * hd..b * d + (h + 1) * hd], kr) * scale;
                    }
                }
                for (gi, &b) in g.iter().enumerate() {
                    let row = &mut sc[gi * t..(gi + 1) * t];
                    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut z = 0f32;
                    for x in row.iter_mut() {
                        *x = (*x - m).exp();
                        z += *x;
                    }
                    let inv = 1.0 / z;
                    let ob = &mut o[b * hd..(b + 1) * hd];
                    for j in 0..t {
                        axpy(ob, row[j] * inv, &v[j * d + h * hd..j * d + (h + 1) * hd]);
                    }
                }
            }
            o
        })
        .collect();
    for (h, o) in per_head.iter().enumerate() {
        for b in 0..beams {
            out[b * d + h * hd..b * d + (h + 1) * hd].copy_from_slice(&o[b * hd..(b + 1) * hd]);
        }
    }
}

/// A chunk's cross-attention keys/values laid out head-major
/// (`[layer][head][t][hd]`), built once per chunk: every head's keys are
/// one contiguous block, read once per step for all beams.
pub struct CrossKv {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
}

fn attend_cross_b<const B: usize>(
    q: &[f32],
    kv: (&[f32], &[f32]),
    t: usize,
    d: usize,
    heads: usize,
    out: &mut [f32],
) {
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let main = t - t % 4;
    let per_head: Vec<Vec<f32>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let kh = &kv.0[h * t * hd..(h + 1) * t * hd];
            let vh = &kv.1[h * t * hd..(h + 1) * t * hd];
            let mut qh = vec![0f32; B * hd];
            for b in 0..B {
                qh[b * hd..(b + 1) * hd].copy_from_slice(&q[b * d + h * hd..b * d + (h + 1) * hd]);
            }
            // scores [t][B]: four keys x B beams per kernel call.
            let mut sc = vec![0f32; t * B];
            let mut j = 0;
            while j < main {
                rows4::<B>(
                    &kh[j * hd..(j + 4) * hd],
                    &qh,
                    hd,
                    &mut sc[j * B..(j + 4) * B],
                );
                j += 4;
            }
            for j in main..t {
                for b in 0..B {
                    sc[j * B + b] = dot(&kh[j * hd..(j + 1) * hd], &qh[b * hd..(b + 1) * hd]);
                }
            }
            let mut o = vec![0f32; B * hd];
            for b in 0..B {
                let mut m = f32::NEG_INFINITY;
                for j in 0..t {
                    let x = sc[j * B + b] * scale;
                    sc[j * B + b] = x;
                    m = m.max(x);
                }
                let mut z = 0f32;
                for j in 0..t {
                    let e = expf_nonpos(sc[j * B + b] - m);
                    sc[j * B + b] = e;
                    z += e;
                }
                let inv = 1.0 / z;
                for j in 0..t {
                    sc[j * B + b] *= inv;
                }
            }
            // out[b] = sum_j p[b][j] * V[j]: each value row loaded once for all beams.
            for j in 0..t {
                let vr = &vh[j * hd..(j + 1) * hd];
                for b in 0..B {
                    axpy(&mut o[b * hd..(b + 1) * hd], sc[j * B + b], vr);
                }
            }
            o
        })
        .collect();
    for (h, o) in per_head.iter().enumerate() {
        for b in 0..B {
            out[b * d + h * hd..b * d + (h + 1) * hd].copy_from_slice(&o[b * hd..(b + 1) * hd]);
        }
    }
}

fn attend_cross(
    q: &[f32],
    kv: (&[f32], &[f32]),
    t: usize,
    d: usize,
    heads: usize,
    beams: usize,
    out: &mut [f32],
) {
    match beams {
        1 => attend_cross_b::<1>(q, kv, t, d, heads, out),
        2 => attend_cross_b::<2>(q, kv, t, d, heads, out),
        3 => attend_cross_b::<3>(q, kv, t, d, heads, out),
        4 => attend_cross_b::<4>(q, kv, t, d, heads, out),
        5 => attend_cross_b::<5>(q, kv, t, d, heads, out),
        6 => attend_cross_b::<6>(q, kv, t, d, heads, out),
        7 => attend_cross_b::<7>(q, kv, t, d, heads, out),
        8 => attend_cross_b::<8>(q, kv, t, d, heads, out),
        _ => panic!("FastDecoder: at most 8 beams"),
    }
}

impl FastDecoder {
    pub fn load(path: &Path, heads: usize) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)?;
        let p = "model.decoder";
        let lin = |name: &str, bias: bool| -> Result<Lin> {
            let (w, s) = f32_tensor(&st, &format!("{name}.weight"))?;
            let b = if bias {
                Some(f32_tensor(&st, &format!("{name}.bias"))?.0)
            } else {
                None
            };
            Ok(Lin {
                w,
                b,
                out: s[0],
                inp: s[1],
            })
        };
        let ln = |name: &str| -> Result<Ln> {
            Ok(Ln {
                w: f32_tensor(&st, &format!("{name}.weight"))?.0,
                b: f32_tensor(&st, &format!("{name}.bias"))?.0,
            })
        };
        let (embed, es) = f32_tensor(&st, &format!("{p}.embed_tokens.weight"))?;
        let (pos, ps) = f32_tensor(&st, &format!("{p}.embed_positions.weight"))?;
        let mut layers = Vec::new();
        let mut i = 0;
        while st.tensor(&format!("{p}.layers.{i}.fc1.weight")).is_ok() {
            let l = format!("{p}.layers.{i}");
            layers.push(Layer {
                sa_ln: ln(&format!("{l}.self_attn_layer_norm"))?,
                qkv: {
                    let (q, k, v) = (
                        lin(&format!("{l}.self_attn.q_proj"), true)?,
                        lin(&format!("{l}.self_attn.k_proj"), false)?,
                        lin(&format!("{l}.self_attn.v_proj"), true)?,
                    );
                    let mut w = q.w;
                    w.extend(k.w);
                    w.extend(v.w);
                    let mut b = q.b.unwrap();
                    b.extend(std::iter::repeat_n(0.0, k.out));
                    b.extend(v.b.unwrap());
                    Lin {
                        w,
                        b: Some(b),
                        out: q.out + k.out + v.out,
                        inp: q.inp,
                    }
                },
                o: lin(&format!("{l}.self_attn.out_proj"), true)?,
                ca_ln: ln(&format!("{l}.encoder_attn_layer_norm"))?,
                cq: lin(&format!("{l}.encoder_attn.q_proj"), true)?,
                co: lin(&format!("{l}.encoder_attn.out_proj"), true)?,
                ff_ln: ln(&format!("{l}.final_layer_norm"))?,
                fc1: lin(&format!("{l}.fc1"), true)?,
                fc2: lin(&format!("{l}.fc2"), true)?,
            });
            i += 1;
        }
        Ok(FastDecoder {
            embed,
            pos,
            layers,
            ln: ln(&format!("{p}.layer_norm"))?,
            d: es[1],
            heads,
            vocab: es[0],
            max_pos: ps[0],
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Re-lay a chunk's cross K/V (`[t, d]` per layer, token-major, as
    /// `WhisperRunner::cross_cache_batch(enc, 1)` returns them) head-major.
    pub fn prepare_cross(&self, k: &[Vec<f32>], v: &[Vec<f32>]) -> CrossKv {
        let (d, hd) = (self.d, self.d / self.heads);
        let t = k[0].len() / d;
        let relay = |src: &Vec<f32>| -> Vec<f32> {
            let mut out = vec![0f32; src.len()];
            for h in 0..self.heads {
                for j in 0..t {
                    out[(h * t + j) * hd..(h * t + j + 1) * hd]
                        .copy_from_slice(&src[j * d + h * hd..j * d + (h + 1) * hd]);
                }
            }
            out
        };
        CrossKv {
            k: k.par_iter().map(relay).collect(),
            v: v.par_iter().map(relay).collect(),
            t,
        }
    }

    /// One decoder step: `tokens[b]` at position `cache.len`, appending to
    /// the self cache. `cross_k/v[layer]` are `[enc_seq * d]` (shared by all
    /// beams). Returns `[beams, vocab]` logits.
    pub fn step(&self, tokens: &[u32], cache: &mut SelfCache, cross: &CrossKv) -> Vec<f32> {
        let (d, beams, t) = (self.d, tokens.len(), cache.len);
        let mut x = vec![0f32; beams * d];
        for (b, &tok) in tokens.iter().enumerate() {
            let e = &self.embed[tok as usize * d..(tok as usize + 1) * d];
            let p = &self.pos[t * d..(t + 1) * d];
            for i in 0..d {
                x[b * d + i] = e[i] + p[i];
            }
        }
        let mut att = vec![0f32; beams * d];
        let prof = std::env::var_os("FASTDEC_PROF").is_some();
        let (mut t_lin, mut t_self, mut t_cross, mut t_misc) = (0f64, 0f64, 0f64, 0f64);
        let tick = || std::time::Instant::now();
        for (li, l) in self.layers.iter().enumerate() {
            let t0 = tick();
            let h = layer_norm(&l.sa_ln, &x, d);
            let qkv = linear(&l.qkv, &h, beams);
            let mut q = vec![0f32; beams * d];
            for b in 0..beams {
                let r = &qkv[b * 3 * d..(b + 1) * 3 * d];
                q[b * d..(b + 1) * d].copy_from_slice(&r[..d]);
                cache.k[li][b].extend_from_slice(&r[d..2 * d]);
                cache.v[li][b].extend_from_slice(&r[2 * d..]);
            }
            let t1 = tick();
            let ks: Vec<&[f32]> = cache.k[li].iter().map(|v| v.as_slice()).collect();
            let vs: Vec<&[f32]> = cache.v[li].iter().map(|v| v.as_slice()).collect();
            attend(&q, &ks, &vs, d, self.heads, &mut att);
            let t2 = tick();
            let o = linear(&l.o, &att, beams);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            let h = layer_norm(&l.ca_ln, &x, d);
            let q = linear(&l.cq, &h, beams);
            let t3 = tick();
            attend_cross(
                &q,
                (&cross.k[li], &cross.v[li]),
                cross.t,
                d,
                self.heads,
                beams,
                &mut att,
            );
            let t4 = tick();
            let o = linear(&l.co, &att, beams);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            let h = layer_norm(&l.ff_ln, &x, d);
            let mut f = linear(&l.fc1, &h, beams);
            f.par_iter_mut().for_each(|z| {
                *z = 0.5 * *z * (1.0 + libm::erff(*z * std::f32::consts::FRAC_1_SQRT_2))
            });
            let o = linear(&l.fc2, &f, beams);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
            let t5 = tick();
            if prof {
                t_self += (t2 - t1).as_secs_f64();
                t_cross += (t4 - t3).as_secs_f64();
                t_lin +=
                    (t5 - t0).as_secs_f64() - (t2 - t1).as_secs_f64() - (t4 - t3).as_secs_f64();
            }
        }
        let t6 = tick();
        cache.len += 1;
        let h = layer_norm(&self.ln, &x, d);
        let lg = logits_tied(&self.embed, &h, beams, self.vocab, d);
        if prof {
            t_misc += t6.elapsed().as_secs_f64();
            eprintln!(
                "step t={t}: linear+ln {:.1} self {:.1} cross {:.1} logits {:.1} ms",
                t_lin * 1e3,
                t_self * 1e3,
                t_cross * 1e3,
                t_misc * 1e3
            );
        }
        lg
    }
}

/// `h @ embed^T` for each beam.
fn logits_tied(embed: &[f32], h: &[f32], beams: usize, vocab: usize, d: usize) -> Vec<f32> {
    let l = Lin {
        w: Vec::new(),
        b: None,
        out: vocab,
        inp: d,
    };
    let _ = l;
    let yt = match beams {
        1 => linear_rows::<1>(embed, h, d, vocab - vocab % 4),
        5 => linear_rows::<5>(embed, h, d, vocab - vocab % 4),
        2 => linear_rows::<2>(embed, h, d, vocab - vocab % 4),
        3 => linear_rows::<3>(embed, h, d, vocab - vocab % 4),
        4 => linear_rows::<4>(embed, h, d, vocab - vocab % 4),
        6 => linear_rows::<6>(embed, h, d, vocab - vocab % 4),
        7 => linear_rows::<7>(embed, h, d, vocab - vocab % 4),
        8 => linear_rows::<8>(embed, h, d, vocab - vocab % 4),
        _ => panic!("FastDecoder: at most 8 beams"),
    };
    let main = vocab - vocab % 4;
    let mut y = vec![0f32; beams * vocab];
    for o in 0..main {
        for b in 0..beams {
            y[b * vocab + o] = yt[o * beams + b];
        }
    }
    for o in main..vocab {
        let row = &embed[o * d..(o + 1) * d];
        for b in 0..beams {
            y[b * vocab + o] = row
                .iter()
                .zip(&h[b * d..(b + 1) * d])
                .fold(0f32, |a, (p, q)| p.mul_add(*q, a));
        }
    }
    y
}

/// Diagnostics: time `reps` calls of one `out x inp` linear over 5 beams.
#[doc(hidden)]
pub fn bench_linear(out: usize, inp: usize, reps: usize) -> f64 {
    let l = Lin {
        w: vec![0.01; out * inp],
        b: Some(vec![0.0; out]),
        out,
        inp,
    };
    let x = vec![0.5f32; 5 * inp];
    let t = std::time::Instant::now();
    let mut s = 0f32;
    for _ in 0..reps {
        s += linear(&l, &x, 5)[0];
    }
    std::hint::black_box(s);
    t.elapsed().as_secs_f64() / reps as f64
}

#[cfg(test)]
mod tests {
    /// Against libm over the softmax range: within 2 ulp.
    #[test]
    fn expf_nonpos_is_within_two_ulp() {
        let mut worst = 0u32;
        let mut x = 0f32;
        while x > -87.0 {
            let (a, b) = (super::expf_nonpos(x), x.exp());
            worst = worst.max((a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32);
            x -= 1.3e-3;
        }
        assert!(worst <= 2, "{worst} ulp");
    }
}
