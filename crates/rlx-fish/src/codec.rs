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

//! The fish codec (`codec.pth`, shared by S2-Pro and Drama 3): a causal
//! 44.1 kHz DAC whose RVQ sits behind a 4× temporal downsample with a
//! transformer on each side.
//!
//! ```text
//! decode: codes [10, T] ─ VQ lookup (host) ─ post transformer (8 layers,
//!         causal, window 128) ─ 2 × (ConvTranspose k2 s2 + ConvNeXt) ─
//!         DAC decoder (1536 ch, rates 8·8·4·2, causal) ─ tanh ─ [2048·T]
//! encode: pcm ─ DAC encoder (rates 2·4·8·8, 4-layer transformer, causal) ─
//!         2 × (Conv k2 s2 + ConvNeXt) ─ pre transformer ─ RVQ search (host)
//! ```
//!
//! Code 0 is the semantic codebook (4096 entries), codes 1..=9 the residual
//! ones (1024 each); every codebook is 8-dimensional and projected to 1024.
//! The decoder transformer the config declares is never registered by the
//! reference `DecoderBlock` and the checkpoint has no weights for it.
//!
//! Layout: everything runs in rows (`[1, T, C]`); convs are sums of matmuls
//! over shifted views (see the conv half below).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, NodeId, PadMode, RopeStyle, Shape};
use rlx_runtime::{CompiledGraph, Device};
use rlx_torch_ckpt::PtModel;

use crate::nn::{self, AttnSpec, BlockSpec, Builder, Keys, TensorSource};

/// Output sample rate.
pub const SAMPLE_RATE: u32 = 44_100;
/// Waveform samples per code frame (512 hop × 4 quantizer downsample).
pub const FRAME_SAMPLES: usize = 2048;
/// Codebooks per frame: one semantic + nine residual.
pub const NUM_CODEBOOKS: usize = 10;
/// Semantic codebook entries (code 0).
pub const SEMANTIC_SIZE: usize = 4096;
/// Residual codebook entries (codes 1..).
pub const RESIDUAL_SIZE: usize = 1024;

const LATENT: usize = 1024;
const CODE_DIM: usize = 8;
/// Frames per compiled decode/encode graph are rounded up to this. The codec
/// is causal end to end, so zero frames appended after the real ones cannot
/// change them; they are trimmed from the output.
const FRAME_BUCKET: usize = 32;

const F32: DType = DType::F32;

/// One codebook: `in_proj` (encode) and `out_proj` (both directions).
struct Codebook {
    entries: Vec<f32>,
    size: usize,
    in_w: Vec<f32>,
    in_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

impl Codebook {
    fn load(src: &PtModel, prefix: &str) -> Result<Self> {
        let (entries, shape) = get(src, &format!("{prefix}.codebook.weight"))?;
        ensure!(
            shape.len() == 2 && shape[1] == CODE_DIM,
            "{prefix}: codebook {shape:?}"
        );
        let (in_w, _) = folded_conv(src, &format!("{prefix}.in_proj"))?;
        let (out_w, _) = folded_conv(src, &format!("{prefix}.out_proj"))?;
        Ok(Self {
            size: shape[0],
            entries,
            in_w,
            in_b: get(src, &format!("{prefix}.in_proj.bias"))?.0,
            out_w,
            out_b: get(src, &format!("{prefix}.out_proj.bias"))?.0,
        })
    }

    /// `out_proj(entries[code])` added into `z[t·LATENT ..]` (row layout).
    fn add_decoded(&self, code: usize, z: &mut [f32]) {
        let e = &self.entries[code * CODE_DIM..(code + 1) * CODE_DIM];
        for (o, slot) in z.iter_mut().enumerate() {
            let w = &self.out_w[o * CODE_DIM..(o + 1) * CODE_DIM];
            *slot += self.out_b[o] + w.iter().zip(e).map(|(a, b)| a * b).sum::<f32>();
        }
    }

    /// Nearest entry by cosine similarity of the projected residual, exactly
    /// as `VectorQuantize.decode_latents` scores it.
    fn nearest(&self, residual: &[f32]) -> usize {
        let mut e = [0f32; CODE_DIM];
        for (d, slot) in e.iter_mut().enumerate() {
            let w = &self.in_w[d * LATENT..(d + 1) * LATENT];
            *slot = self.in_b[d] + w.iter().zip(residual).map(|(a, b)| a * b).sum::<f32>();
        }
        l2_normalize(&mut e);
        let e2: f32 = e.iter().map(|v| v * v).sum();
        let mut best = (f32::INFINITY, 0usize);
        for i in 0..self.size {
            let mut c = [0f32; CODE_DIM];
            c.copy_from_slice(&self.entries[i * CODE_DIM..(i + 1) * CODE_DIM]);
            l2_normalize(&mut c);
            let dot: f32 = e.iter().zip(&c).map(|(a, b)| a * b).sum();
            let c2: f32 = c.iter().map(|v| v * v).sum();
            let dist = e2 - 2.0 * dot + c2;
            if dist < best.0 {
                best = (dist, i);
            }
        }
        best.1
    }
}

/// A checkpoint tensor as `(f32 data, shape)`.
fn get(src: &PtModel, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    TensorSource::tensor(src, key)
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v {
        *x /= n;
    }
}

/// A conv weight with weight-norm folded in: `g · v / ‖v‖`, the norm taken
/// over every axis but the first (torch `weight_norm(dim=0)`). Reads either
/// parametrization spelling, or a plain `.weight`. Returns `(w, shape)`.
fn folded_conv(src: &PtModel, prefix: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let (g_key, v_key) = if src
        .shape_of(&format!("{prefix}.parametrizations.weight.original0"))
        .is_some()
    {
        (
            format!("{prefix}.parametrizations.weight.original0"),
            format!("{prefix}.parametrizations.weight.original1"),
        )
    } else if src.shape_of(&format!("{prefix}.weight_g")).is_some() {
        (format!("{prefix}.weight_g"), format!("{prefix}.weight_v"))
    } else {
        return get(src, &format!("{prefix}.weight"));
    };
    let (g, _) = get(src, &g_key)?;
    let (mut v, shape) = get(src, &v_key)?;
    let per = v.len() / shape[0];
    ensure!(
        g.len() == shape[0],
        "{prefix}: weight_norm g {} vs {shape:?}",
        g.len()
    );
    for (row, gain) in v.chunks_mut(per).zip(&g) {
        let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let s = gain / n;
        for x in row {
            *x *= s;
        }
    }
    Ok((v, shape))
}

/// The codec: host codebooks plus compiled decode/encode graphs per bucket.
pub struct FishCodec {
    ckpt: PtModel,
    device: Device,
    semantic: Codebook,
    residual: Vec<Codebook>,
    decoders: HashMap<usize, CompiledGraph>,
    encoders: HashMap<usize, CompiledGraph>,
}

impl FishCodec {
    /// Open `codec.pth` (a file, or a model directory holding one).
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let file = if path.is_dir() {
            path.join("codec.pth")
        } else {
            path.to_path_buf()
        };
        let ckpt = PtModel::open(&file).with_context(|| format!("open {}", file.display()))?;
        let semantic = Codebook::load(&ckpt, "quantizer.semantic_quantizer.quantizers.0")?;
        ensure!(
            semantic.size == SEMANTIC_SIZE,
            "semantic codebook has {} entries",
            semantic.size
        );
        let residual = (0..NUM_CODEBOOKS - 1)
            .map(|i| Codebook::load(&ckpt, &format!("quantizer.quantizer.quantizers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            ckpt,
            device,
            semantic,
            residual,
            decoders: HashMap::new(),
            encoders: HashMap::new(),
        })
    }

    /// Codes `[NUM_CODEBOOKS][frames]` (codebook-major, like the reference's
    /// `[10, T]`) → mono f32 PCM at [`SAMPLE_RATE`], `2048 · frames` samples.
    pub fn decode(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let z = self.dequantize(codes)?;
        let frames = codes[0].len();
        self.decode_latent(&z, frames)
    }

    /// The summed codebook vectors `[frames, 1024]` (row layout) — what
    /// `DownsampleResidualVectorQuantize.decode` hands its post transformer.
    pub fn dequantize(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        ensure!(
            codes.len() == NUM_CODEBOOKS,
            "need {NUM_CODEBOOKS} codebooks, got {}",
            codes.len()
        );
        let frames = codes[0].len();
        ensure!(
            codes.iter().all(|c| c.len() == frames),
            "ragged code matrix"
        );
        let mut z = vec![0f32; frames * LATENT];
        for t in 0..frames {
            let row = &mut z[t * LATENT..(t + 1) * LATENT];
            // The reference clamps out-of-range codes rather than failing.
            let c0 = (codes[0][t] as usize).min(SEMANTIC_SIZE - 1);
            self.semantic.add_decoded(c0, row);
            for (cb, book) in self.residual.iter().enumerate() {
                let c = (codes[cb + 1][t] as usize).min(RESIDUAL_SIZE - 1);
                book.add_decoded(c, row);
            }
        }
        Ok(z)
    }

    /// Every codebook's `out_proj(entry) + bias` rows, the semantic book's
    /// then each residual one's: `[4096 + 9 · 1024, 1024]` — a code's row
    /// is what [`Self::dequantize`] adds for it.
    #[cfg(feature = "export")]
    pub(crate) fn decoded_rows(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for book in std::iter::once(&self.semantic).chain(&self.residual) {
            for code in 0..book.size {
                let mut row = vec![0f32; LATENT];
                book.add_decoded(code, &mut row);
                out.extend(row);
            }
        }
        out
    }

    /// The checkpoint the decoder graphs read.
    #[cfg(feature = "export")]
    pub(crate) fn checkpoint(&self) -> &PtModel {
        &self.ckpt
    }

    /// Decode a `[frames, 1024]` latent.
    pub fn decode_latent(&mut self, z: &[f32], frames: usize) -> Result<Vec<f32>> {
        ensure!(
            frames > 0 && z.len() == frames * LATENT,
            "latent is {} values for {frames} frames",
            z.len()
        );
        let bucket = frames.div_ceil(FRAME_BUCKET) * FRAME_BUCKET;
        if !self.decoders.contains_key(&bucket) {
            let t = std::time::Instant::now();
            let (b, out) = build_decoder(&self.ckpt, bucket)?;
            if rlx_ir::env::flag("RLX_FISH_VERBOSE") {
                eprintln!(
                    "codec: build decoder for {bucket} frames {:.1} s",
                    t.elapsed().as_secs_f32()
                );
            }
            let g = compile(self.device, b, out, &self.ckpt)?;
            self.decoders.insert(bucket, g);
        }
        let mut input = vec![0f32; bucket * LATENT];
        input[..z.len()].copy_from_slice(z);
        let g = self.decoders.get_mut(&bucket).expect("decoder graph");
        let mut outs = g.run(&[("z", &input)]);
        let mut wav = outs.swap_remove(0);
        wav.truncate(frames * FRAME_SAMPLES);
        Ok(wav)
    }

    /// Mono f32 PCM at [`SAMPLE_RATE`] → codes `[NUM_CODEBOOKS][frames]`,
    /// `frames = ceil(len / 2048)`.
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
        let latent = self.encode_latent(pcm)?;
        let frames = latent.len() / LATENT;
        let mut codes: Vec<Vec<u32>> = (0..NUM_CODEBOOKS)
            .map(|_| Vec::with_capacity(frames))
            .collect();
        let mut decoded = vec![0f32; LATENT];
        for t in 0..frames {
            let mut residual = latent[t * LATENT..(t + 1) * LATENT].to_vec();
            let books = std::iter::once(&self.semantic).chain(&self.residual);
            for (cb, book) in books.enumerate() {
                let c = book.nearest(&residual);
                codes[cb].push(c as u32);
                decoded.fill(0.0);
                book.add_decoded(c, &mut decoded);
                for (r, d) in residual.iter_mut().zip(&decoded) {
                    *r -= d;
                }
            }
        }
        Ok(codes)
    }

    /// The pre-quantizer latent `[frames, 1024]` for `pcm`.
    pub fn encode_latent(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        ensure!(!pcm.is_empty(), "empty audio");
        let frames = pcm.len().div_ceil(FRAME_SAMPLES);
        let bucket = frames.div_ceil(FRAME_BUCKET) * FRAME_BUCKET;
        if !self.encoders.contains_key(&bucket) {
            let (b, out) = build_encoder(&self.ckpt, bucket * FRAME_SAMPLES)?;
            let g = compile(self.device, b, out, &self.ckpt)?;
            self.encoders.insert(bucket, g);
        }
        let mut input = vec![0f32; bucket * FRAME_SAMPLES];
        input[..pcm.len()].copy_from_slice(pcm);
        let g = self.encoders.get_mut(&bucket).expect("encoder graph");
        let mut outs = g.run(&[("pcm", &input)]);
        let mut latent = outs.swap_remove(0);
        latent.truncate(frames * LATENT);
        Ok(latent)
    }

    /// Drop every compiled graph (they are rebuilt on demand).
    pub fn release(&mut self) {
        self.decoders.clear();
        self.encoders.clear();
    }
}

fn compile(device: Device, mut b: Builder, out: NodeId, src: &PtModel) -> Result<CompiledGraph> {
    b.g.set_outputs(vec![out]);
    let profile = rlx_flow::CompileProfile::default();
    let name = b.g.name.clone();
    let t = std::time::Instant::now();
    let mut g = rlx_core::flow_bridge::compile_graph_with_profile(device, b.g, &profile)?;
    let t_compile = t.elapsed();
    let t = std::time::Instant::now();
    nn::upload(&mut g, b.params, src)?;
    if rlx_ir::env::flag("RLX_FISH_VERBOSE") {
        eprintln!(
            "codec: {name}: compile {:.1} s, weights {:.1} s",
            t_compile.as_secs_f32(),
            t.elapsed().as_secs_f32()
        );
    }
    Ok(g)
}

// ── transformer half (rows) ─────────────────────────────────────────────

/// The codec's `WindowLimitedTransformer`: causal, `window` keys per query,
/// LayerScale, RoPE from bf16 tables, a final RMSNorm. `x: [1, t, dim]`.
fn transformer(
    b: &mut Builder,
    prefix: &str,
    x: NodeId,
    t: usize,
    layers: usize,
    dim: usize,
    window: Option<usize>,
) -> NodeId {
    let spec = BlockSpec {
        dim,
        ffn: dim * 3,
        eps: 1e-5,
        attn: AttnSpec {
            n_head: dim / 64,
            n_kv: dim / 64,
            head_dim: 64,
            qk_norm: false,
            rope_dim: 64,
            rope_style: RopeStyle::GptJ,
            output_gate: false,
        },
        layer_scale: true,
        zero_centered: false,
    };
    let positions: Vec<usize> = (0..t).collect();
    let (cos, sin) = nn::rope_tables(&positions, 64, 10_000.0, true);
    let cos = b.data(&format!("{prefix}rope_cos.{t}"), cos, &[t, 32]);
    let sin = b.data(&format!("{prefix}rope_sin.{t}"), sin, &[t, 32]);
    // `make_window_limited_mask`: key j is visible to query i when
    // i - window < j <= i, i.e. `window` keys including itself.
    let mask = match window {
        Some(w) if w < t => MaskKind::SlidingWindow(w - 1),
        _ => MaskKind::Causal,
    };
    let mut h = x;
    for l in 0..layers {
        h = nn::block(
            b,
            &spec,
            &format!("{prefix}layers.{l}."),
            h,
            t,
            cos,
            sin,
            Keys::Own(mask),
        )
        .x;
    }
    b.rms_norm(h, &format!("{prefix}norm.weight"), dim, 1e-5, false)
}

/// `ConvNeXtBlock` in rows: causal depthwise k7, LayerNorm, MLP ×4 with exact
/// GELU, LayerScale, residual.
fn convnext(
    b: &mut Builder,
    src: &PtModel,
    prefix: &str,
    x: NodeId,
    t: usize,
    c: usize,
) -> Result<NodeId> {
    let (w, shape) = get(src, &format!("{prefix}.dwconv.conv.weight"))?;
    ensure!(shape == [c, 1, 7], "{prefix}: dwconv {shape:?}");
    let xp =
        b.g.pad_(x, vec![[0, 0], [6, 0], [0, 0]], PadMode::Constant(0.0));
    let mut acc = None;
    for j in 0..7 {
        let tap: Vec<f32> = (0..c).map(|ch| w[ch * 7 + j]).collect();
        let tw = b.data(&format!("{prefix}.dwconv.tap{j}"), tap, &[c]);
        let s = b.g.narrow_(xp, 1, j, t);
        let m = b.g.mul(s, tw);
        acc = Some(match acc {
            None => m,
            Some(a) => b.g.add(a, m),
        });
    }
    let bias = b.weight(
        &format!("{prefix}.dwconv.conv.bias"),
        &[c],
        nn::TensorOp::Raw,
    );
    let y = b.g.add(acc.expect("taps"), bias);
    let gamma = b.weight(&format!("{prefix}.norm.weight"), &[c], nn::TensorOp::Raw);
    let beta = b.weight(&format!("{prefix}.norm.bias"), &[c], nn::TensorOp::Raw);
    let y = b.g.ln(y, gamma, beta, 1e-6);
    let y = b.linear_bias(y, &format!("{prefix}.pwconv1"), c, 4 * c);
    let y = b.g.gelu(y);
    let y = b.linear_bias(y, &format!("{prefix}.pwconv2"), 4 * c, c);
    let ls = b.weight(&format!("{prefix}.gamma"), &[c], nn::TensorOp::Raw);
    let y = b.g.mul(y, ls);
    Ok(b.g.add(x, y))
}

// ── conv half (rows) ────────────────────────────────────────────────────
//
// Every conv is a sum of matmuls over shifted views of `[1, T, C]` rows: a
// k-tap conv is `Σ_j x[t + j·d] · W_j` with `W_j = w[:, :, j]ᵀ`. Matmuls run
// at full speed on every backend, where direct 1-D conv kernels may not (the
// Metal conv on this stack ran ~10× under the CPU). One matmul per tap, not
// an im2col concat: at 44.1 kHz an im2col of the 96-channel stage is 2 GB.

/// `[cin, cout]` tap `j` of a conv weight `[cout, cin, k]`.
fn conv_tap(w: &[f32], c_out: usize, c_in: usize, k: usize, j: usize) -> Vec<f32> {
    let mut t = vec![0f32; c_in * c_out];
    for o in 0..c_out {
        for i in 0..c_in {
            t[i * c_out + o] = w[(o * c_in + i) * k + j];
        }
    }
    t
}

fn add_bias(b: &mut Builder, src: &PtModel, prefix: &str, y: NodeId, c: usize) -> Result<NodeId> {
    let (bias, _) = get(src, &format!("{prefix}.bias"))?;
    ensure!(
        bias.len() == c,
        "{prefix}.bias: {} for {c} channels",
        bias.len()
    );
    let bn = b.data(&format!("{prefix}.b"), bias, &[c]);
    Ok(b.g.add(y, bn))
}

/// `Σ_j narrow(x, j·step) @ W_j` over `taps` with `rows` output rows.
fn tap_sum(b: &mut Builder, x: NodeId, taps: Vec<(usize, NodeId)>, rows: usize) -> NodeId {
    let mut acc = None;
    for (off, w) in taps {
        let v = b.g.narrow_(x, 1, off, rows);
        let m = b.g.mm(v, w);
        acc = Some(match acc {
            None => m,
            Some(a) => b.g.add(a, m),
        });
    }
    acc.expect("at least one tap")
}

/// Causal `Conv1d` on rows `[1, t, c_in]`. Stride 1 (any dilation), or
/// `k = 2·stride` (the encoder's downsamplers). Returns `(y, c_out, t_out)`.
fn causal_conv(
    b: &mut Builder,
    src: &PtModel,
    prefix: &str,
    x: NodeId,
    t: usize,
    stride: usize,
    dilation: usize,
) -> Result<(NodeId, usize, usize)> {
    let (w, shape) = folded_conv(src, prefix)?;
    let (c_out, c_in, k) = (shape[0], shape[1], shape[2]);
    let pad = (k - 1) * dilation + 1 - stride;
    let xp = if pad > 0 {
        b.g.pad_(x, vec![[0, 0], [pad, 0], [0, 0]], PadMode::Constant(0.0))
    } else {
        x
    };
    if stride == 1 {
        let taps = (0..k)
            .map(|j| {
                let wj = b.data(
                    &format!("{prefix}.tap{j}"),
                    conv_tap(&w, c_out, c_in, k, j),
                    &[c_in, c_out],
                );
                (j * dilation, wj)
            })
            .collect();
        let y = tap_sum(b, xp, taps, t);
        return Ok((add_bias(b, src, prefix, y, c_out)?, c_out, t));
    }
    // `k = 2s`, left pad `s`: output `t'` reads padded samples
    // `[t'·s, t'·s + 2s)` — rows `t'` and `t' + 1` of the padded input
    // regrouped `s` samples per row.
    let s = stride;
    ensure!(
        k == 2 * s && dilation == 1 && t.is_multiple_of(s),
        "{prefix}: strided conv k {k} s {s} on {t}"
    );
    let t_out = t / s;
    let r =
        b.g.reshape_(xp, vec![1, (t_out + 1) as i64, (s * c_in) as i64]);
    let half = |b: &mut Builder, first: usize, name: &str| {
        let mut m = vec![0f32; s * c_in * c_out];
        for j in 0..s {
            let tap = conv_tap(&w, c_out, c_in, k, first + j);
            m[j * c_in * c_out..(j + 1) * c_in * c_out].copy_from_slice(&tap);
        }
        b.data(&format!("{prefix}.{name}"), m, &[s * c_in, c_out])
    };
    let wa = half(b, 0, "lo");
    let wb = half(b, s, "hi");
    let y = tap_sum(b, r, vec![(0, wa), (1, wb)], t_out);
    Ok((add_bias(b, src, prefix, y, c_out)?, c_out, t_out))
}

/// Causal `ConvTranspose1d` with `k = 2·stride` on rows, keeping the first
/// `t · stride` outputs (`CausalTransConvNet` trims the right `k - stride`).
/// Output `t·s + p` is `x[t]·w[:, :, p] + x[t-1]·w[:, :, p+s]`: two matmuls
/// onto all `s` phases at once, then a reshape interleaves them.
fn causal_conv_transpose(
    b: &mut Builder,
    src: &PtModel,
    prefix: &str,
    x: NodeId,
    t: usize,
    stride: usize,
) -> Result<(NodeId, usize, usize)> {
    let (w, shape) = folded_conv(src, prefix)?;
    let (c_in, c_out, k) = (shape[0], shape[1], shape[2]);
    let s = stride;
    ensure!(k == 2 * s, "{prefix}: kernel {k} is not 2·stride {s}");
    let phases = |b: &mut Builder, first: usize, name: &str| {
        let mut m = vec![0f32; c_in * s * c_out];
        for i in 0..c_in {
            for p in 0..s {
                for o in 0..c_out {
                    m[i * s * c_out + p * c_out + o] = w[(i * c_out + o) * k + first + p];
                }
            }
        }
        b.data(&format!("{prefix}.{name}"), m, &[c_in, s * c_out])
    };
    let w_cur = phases(b, 0, "cur");
    let w_prev = phases(b, s, "prev");
    let xp =
        b.g.pad_(x, vec![[0, 0], [1, 0], [0, 0]], PadMode::Constant(0.0));
    let y = tap_sum(b, xp, vec![(1, w_cur), (0, w_prev)], t);
    let t_out = t * s;
    let y = b.g.reshape_(y, vec![1, t_out as i64, c_out as i64]);
    Ok((add_bias(b, src, prefix, y, c_out)?, c_out, t_out))
}

/// Snake: `x + sin(α·x)² / (α + 1e-9)`, per channel (last axis).
fn snake(
    b: &mut Builder,
    src: &PtModel,
    key: &str,
    x: NodeId,
    c: usize,
    t: usize,
) -> Result<NodeId> {
    let (alpha, shape) = get(src, key)?;
    ensure!(alpha.len() == c, "{key}: {shape:?} for {c} channels");
    let inv: Vec<f32> = alpha.iter().map(|a| 1.0 / (a + 1e-9)).collect();
    let a = b.data(&format!("{key}.a"), alpha, &[c]);
    let ia = b.data(&format!("{key}.ia"), inv, &[c]);
    let ax = b.g.mul(x, a);
    let s =
        b.g.activation(rlx_ir::op::Activation::Sin, ax, Shape::new(&[1, t, c], F32));
    let s2 = b.g.mul(s, s);
    let sc = b.g.mul(s2, ia);
    Ok(b.g.add(x, sc))
}

/// `ResidualUnit(causal)`: snake → conv k7 (dilated) → snake → conv k1, + x.
fn residual_unit(
    b: &mut Builder,
    src: &PtModel,
    prefix: &str,
    x: NodeId,
    c: usize,
    t: usize,
    dilation: usize,
) -> Result<NodeId> {
    let h = snake(b, src, &format!("{prefix}.block.0.alpha"), x, c, t)?;
    let (h, _, _) = causal_conv(b, src, &format!("{prefix}.block.1.conv"), h, t, 1, dilation)?;
    let h = snake(b, src, &format!("{prefix}.block.2.alpha"), h, c, t)?;
    let (h, _, _) = causal_conv(b, src, &format!("{prefix}.block.3.conv"), h, t, 1, 1)?;
    Ok(b.g.add(x, h))
}

pub(crate) fn build_decoder(src: &PtModel, frames: usize) -> Result<(Builder, NodeId)> {
    let mut b = Builder::new("fish_codec_decode");
    let z = b.input("z", &[1, frames, LATENT]);
    let mut h = transformer(
        &mut b,
        "quantizer.post_module.",
        z,
        frames,
        8,
        LATENT,
        Some(128),
    );

    // Upsample: ConvTranspose(k2, s2) in rows is one matmul onto both output
    // phases, `[t, C] @ [C, 2C]`, then a reshape that interleaves them.
    let mut t = frames;
    for i in 0..2 {
        let prefix = format!("quantizer.upsample.{i}");
        let (w, shape) = get(src, &format!("{prefix}.0.conv.weight"))?;
        ensure!(shape == [LATENT, LATENT, 2], "{prefix}: {shape:?}");
        let mut wp = vec![0f32; LATENT * 2 * LATENT];
        for ic in 0..LATENT {
            for oc in 0..LATENT {
                for p in 0..2 {
                    wp[ic * 2 * LATENT + p * LATENT + oc] = w[(ic * LATENT + oc) * 2 + p];
                }
            }
        }
        let wn = b.data(&format!("{prefix}.0.w"), wp, &[LATENT, 2 * LATENT]);
        let y = b.g.mm(h, wn);
        t *= 2;
        let y = b.g.reshape_(y, vec![1, t as i64, LATENT as i64]);
        let bias = b.weight(
            &format!("{prefix}.0.conv.bias"),
            &[LATENT],
            nn::TensorOp::Raw,
        );
        let y = b.g.add(y, bias);
        h = convnext(&mut b, src, &format!("{prefix}.1"), y, t, LATENT)?;
    }

    let (mut x, mut c, mut t) = causal_conv(&mut b, src, "decoder.model.0.conv", h, t, 1, 1)?;
    for (i, stride) in [8usize, 8, 4, 2].into_iter().enumerate() {
        let p = format!("decoder.model.{}.block", i + 1);
        let s = snake(&mut b, src, &format!("{p}.0.alpha"), x, c, t)?;
        (x, c, t) = causal_conv_transpose(&mut b, src, &format!("{p}.1.conv"), s, t, stride)?;
        for (j, dil) in [1usize, 3, 9].into_iter().enumerate() {
            x = residual_unit(&mut b, src, &format!("{p}.{}", j + 2), x, c, t, dil)?;
        }
    }
    let x = snake(&mut b, src, "decoder.model.5.alpha", x, c, t)?;
    let (x, _, _) = causal_conv(&mut b, src, "decoder.model.6.conv", x, t, 1, 1)?;
    let out = b.g.tanh(x);
    Ok((b, out))
}

fn build_encoder(src: &PtModel, samples: usize) -> Result<(Builder, NodeId)> {
    if !samples.is_multiple_of(FRAME_SAMPLES) {
        bail!("encoder input must be a multiple of {FRAME_SAMPLES} samples");
    }
    let mut b = Builder::new("fish_codec_encode");
    let pcm = b.input("pcm", &[1, samples, 1]);
    let (mut x, mut c, mut t) =
        causal_conv(&mut b, src, "encoder.block.0.conv", pcm, samples, 1, 1)?;
    for (i, stride) in [2usize, 4, 8, 8].into_iter().enumerate() {
        let p = format!("encoder.block.{}.block", i + 1);
        for (j, dil) in [1usize, 3, 9].into_iter().enumerate() {
            x = residual_unit(&mut b, src, &format!("{p}.{j}"), x, c, t, dil)?;
        }
        x = snake(&mut b, src, &format!("{p}.3.alpha"), x, c, t)?;
        (x, c, t) = causal_conv(&mut b, src, &format!("{p}.4.conv"), x, t, stride, 1)?;
        if i == 3 {
            x = transformer(&mut b, &format!("{p}.5."), x, t, 4, c, Some(512));
        }
    }
    let x = snake(&mut b, src, "encoder.block.5.alpha", x, c, t)?;
    let (x, c, t) = causal_conv(&mut b, src, "encoder.block.6.conv", x, t, 1, 1)?;
    ensure!(c == LATENT, "encoder latent has {c} channels");

    // Downsample: causal Conv(k2, s2) in rows is `[t/2, 2C] @ [2C, C]`.
    let mut h = x;
    let mut t = t;
    for i in 0..2 {
        let prefix = format!("quantizer.downsample.{i}");
        let (w, shape) = get(src, &format!("{prefix}.0.conv.weight"))?;
        ensure!(shape == [LATENT, LATENT, 2], "{prefix}: {shape:?}");
        let mut wp = vec![0f32; 2 * LATENT * LATENT];
        for oc in 0..LATENT {
            for ic in 0..LATENT {
                for k in 0..2 {
                    wp[(k * LATENT + ic) * LATENT + oc] = w[(oc * LATENT + ic) * 2 + k];
                }
            }
        }
        t /= 2;
        let r = b.g.reshape_(h, vec![1, t as i64, (2 * LATENT) as i64]);
        let wn = b.data(&format!("{prefix}.0.w"), wp, &[2 * LATENT, LATENT]);
        let y = b.g.mm(r, wn);
        let bias = b.weight(
            &format!("{prefix}.0.conv.bias"),
            &[LATENT],
            nn::TensorOp::Raw,
        );
        let y = b.g.add(y, bias);
        h = convnext(&mut b, src, &format!("{prefix}.1"), y, t, LATENT)?;
    }
    let out = transformer(&mut b, "quantizer.pre_module.", h, t, 8, LATENT, Some(128));
    Ok((b, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_geometry() {
        assert_eq!(FRAME_SAMPLES, 512 * 4);
        assert_eq!(SAMPLE_RATE as f32 / FRAME_SAMPLES as f32, 21.533203);
    }
}
