// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// FACodecDecoder as an rlx-runtime graph running natively on every backend:
// timbre AdaIN (LayerNorm-over-channels + per-channel affine) → WNConv1d →
// 4× DecoderBlock (anti-aliased SnakeBeta + transposed conv + 3 ResidualUnit
// MRF) → SnakeBeta → WNConv1d → Tanh. The anti-aliased activation is BigVGAN's
// upsample(2×)→snake→downsample(2×) with a shared kaiser-sinc FIR.

use crate::model::*;
use anyhow::Result;
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, ConvTranspose1d, PadMode, compile, finish_graph,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;

/// Named tensors `(name, data)` produced alongside a built graph.
type NamedTensors = Vec<(String, Vec<f32>)>;

/// Tile a per-tap FIR `[k]` into a depthwise conv weight `[c, 1, k]`, optionally
/// pre-scaled (used to fold the `*ratio` gain of the transposed-conv upsampler).
fn tile_filter(filter: &[f32], c: usize, scale: f32) -> Vec<f32> {
    let mut w = Vec::with_capacity(c * filter.len());
    for _ in 0..c {
        w.extend(filter.iter().map(|&v| v * scale));
    }
    w
}

/// Anti-aliased SnakeBeta (BigVGAN `Activation1d`): 2× upsample (replicate-pad +
/// depthwise transposed conv, gain `ratio`), SnakeBeta, 2× downsample (replicate
/// -pad + strided depthwise conv). Length is preserved.
fn aa_snake(
    ag: &mut AudioGraph,
    x: HirNodeId,
    c: usize,
    t: usize,
    act: &SnakeW,
    filter: &[f32],
) -> (HirNodeId, usize) {
    // Upsample 2×: F.pad(replicate, 5,5) → 2·conv_transpose1d(stride2, groups=c) → crop[15:-15].
    let up_w = tile_filter(filter, c, 2.0);
    let (xp, tp) = ag.pad_len(x, c, t, 5, 5, PadMode::Replicate);
    let (up, _, t_up) = ag.conv_transpose1d(
        xp,
        tp,
        &ConvTranspose1d {
            weight: &up_w,
            bias: None,
            c_in: c,
            c_out_per_group: 1,
            k: 12,
            stride: 2,
            groups: c,
            trim_left: 15,
            trim_right: 15,
        },
    );
    debug_assert_eq!(t_up, 2 * t);

    let s = ag.snake_beta(up, c, t_up, &act.alpha, &act.beta);

    // Downsample 2×: F.pad(replicate, 5,6) → conv1d(stride2, groups=c).
    let down_w = tile_filter(filter, c, 1.0);
    let (down, _, t_down) = ag.conv1d(
        s,
        t_up,
        &Conv1d {
            weight: &down_w,
            bias: None,
            c_out: c,
            c_in: c,
            k: 12,
            stride: 2,
            dilation: 1,
            groups: c,
            pad_left: 5,
            pad_right: 6,
            pad_mode: PadMode::Replicate,
        },
    );
    (down, t_down)
}

fn conv_k(ag: &mut AudioGraph, x: HirNodeId, t: usize, w: &ConvW, dilation: usize) -> HirNodeId {
    let pad = ((w.k - 1) / 2) * dilation; // symmetric "same" pad
    let (y, _, _) = ag.conv1d(
        x,
        t,
        &Conv1d {
            weight: &w.weight,
            bias: Some(&w.bias),
            c_out: w.c_out,
            c_in: w.c_in,
            k: w.k,
            stride: 1,
            dilation,
            groups: 1,
            pad_left: pad,
            pad_right: pad,
            pad_mode: PadMode::Constant,
        },
    );
    y
}

fn residual_unit(
    ag: &mut AudioGraph,
    x: HirNodeId,
    c: usize,
    t: usize,
    u: &ResidualUnitW,
    filter: &[f32],
) -> HirNodeId {
    let (h, _) = aa_snake(ag, x, c, t, &u.act0, filter);
    let h = conv_k(ag, h, t, &u.conv1, u.dilation);
    let (h, _) = aa_snake(ag, h, c, t, &u.act1, filter);
    let h = conv_k(ag, h, t, &u.conv3, 1);
    ag.add(x, h)
}

fn decoder_block(
    ag: &mut AudioGraph,
    x: HirNodeId,
    c_in: usize,
    t: usize,
    b: &DecoderBlockW,
    filter: &[f32],
) -> (HirNodeId, usize) {
    let (h, _) = aa_snake(ag, x, c_in, t, &b.act0, filter);
    // WNConvTranspose1d(stride): padding = stride//2 + stride%2, output_padding = stride%2.
    let stride = b.convt.stride;
    let p = stride / 2 + stride % 2;
    let op = stride % 2;
    let (mut h, c_out, t_out) = ag.conv_transpose1d(
        h,
        t,
        &ConvTranspose1d {
            weight: &b.convt.weight,
            bias: Some(&b.convt.bias),
            c_in: b.convt.c_in,
            c_out_per_group: b.convt.c_out,
            k: b.convt.k,
            stride,
            groups: 1,
            trim_left: p,
            trim_right: p - op,
        },
    );
    for u in &b.units {
        h = residual_unit(ag, h, c_out, t_out, u, filter);
    }
    (h, t_out)
}

/// emb `[256, T]` (channel-major) + per-speaker `gamma`/`beta` `[256]` → wav `[T_out]`.
pub fn build_decode_graph(
    w: &FacodecWeights,
    gamma: &[f32],
    beta: &[f32],
    t: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("facodec_decode");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let emb = ag.input("emb", &[IN_CH, t]); // [C, T]
    let x_ct = ag.reshape(emb, vec![1, IN_CH as i64, t as i64, 1]);

    // Timbre AdaIN: LayerNorm over channels (no affine) then per-channel scale+shift.
    let xtc = ag.ct_to_tc(x_ct, IN_CH, t); // [T, C]
    let ones = vec![1.0f32; IN_CH];
    let zeros = vec![0.0f32; IN_CH];
    let ln = ag.layer_norm(xtc, IN_CH, &ones, &zeros, LN_EPS);
    let scaled = ag.scale_rows(ln, t, IN_CH, gamma);
    let beta_p = ag.param(beta.to_vec(), &[1, IN_CH]);
    let beta_e = ag.expand(beta_p, &[t, IN_CH]);
    let cond = ag.add(scaled, beta_e);
    let mut x = ag.tc_to_ct(cond, t, IN_CH); // [1, C, T, 1]

    x = conv_k(&mut ag, x, t, &w.conv0, 1);

    let mut cur_t = t;
    let mut cur_c = INIT_CH;
    for b in &w.blocks {
        let (nx, nt) = decoder_block(&mut ag, x, cur_c, cur_t, b, &w.filter);
        x = nx;
        cur_t = nt;
        cur_c >>= 1;
    }

    let (x, _) = aa_snake(&mut ag, x, cur_c, cur_t, &w.act_final, &w.filter);
    let x = conv_k(&mut ag, x, cur_t, &w.conv_final, 1);
    let out = ag.tanh(x); // [1, 1, T_out, 1]

    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params, cur_t))
}

pub struct FacodecDecoderGraph {
    compiled: CompiledAudioGraph,
    pub t: usize,
    pub t_out: usize,
}

impl FacodecDecoderGraph {
    /// Compile for a fixed frame count `t` and a per-speaker timbre `(gamma, beta)`
    /// (precomputed on the host from the speaker embedding via [`FacodecWeights::timbre_affine`]).
    pub fn compile_for(
        device: Device,
        w: &FacodecWeights,
        gamma: &[f32],
        beta: &[f32],
        t: usize,
    ) -> Result<Self> {
        let (graph, params, t_out) = build_decode_graph(w, gamma, beta, t)?;
        let compiled = compile(device, graph, params, 1, t_out, "emb");
        Ok(Self { compiled, t, t_out })
    }

    /// `emb` `[256, T]` channel-major → waveform `[T_out]`.
    pub fn run(&mut self, emb: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(emb)?.0)
    }
}
