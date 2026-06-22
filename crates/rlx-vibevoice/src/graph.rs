// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// VibeVoice acoustic σ-VAE decoder as an rlx-runtime graph running natively on
// every backend: causal stem conv → 7 ConvNeXt stages interleaved with 6 causal
// transposed-conv upsamplers → causal head conv. Each ConvNeXt block is
// RMSNorm → depthwise causal conv → layer-scale → +res, then RMSNorm → FFN
// (Linear→GELU→Linear) → layer-scale → +res.

use crate::model::*;
use anyhow::Result;
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, ConvTranspose1d, PadMode, compile, finish_graph,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;

/// Causal conv: left-pad `(k-1)` zeros (stride 1, dilation 1).
fn causal_conv(ag: &mut AudioGraph, x: HirNodeId, t: usize, w: &ConvW, groups: usize) -> HirNodeId {
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
            dilation: 1,
            groups,
            pad_left: w.k - 1,
            pad_right: 0,
            pad_mode: PadMode::Constant,
        },
    );
    y
}

/// Per-channel layer-scale of a `[1,C,T,1]` tensor by `gamma[C]`.
fn layer_scale(ag: &mut AudioGraph, x: HirNodeId, c: usize, t: usize, gamma: &[f32]) -> HirNodeId {
    let g = ag.param(gamma.to_vec(), &[1, c, 1, 1]);
    let ge = ag.expand(g, &[1, c, t, 1]);
    ag.mul(x, ge)
}

fn block(ag: &mut AudioGraph, x: HirNodeId, t: usize, b: &BlockW) -> HirNodeId {
    let c = b.dim;
    // mixer: RMSNorm(over channels) → depthwise causal conv → layer-scale → +res
    let xtc = ag.ct_to_tc(x, c, t);
    let n = ag.rms_norm(xtc, c, &b.norm_w, EPS);
    let n_ct = ag.tc_to_ct(n, t, c);
    let m = causal_conv(ag, n_ct, t, &b.mixer, c); // depthwise groups = c
    let m = layer_scale(ag, m, c, t, &b.gamma);
    let x = ag.add(x, m);

    // ffn: RMSNorm → Linear→GELU→Linear (in [T,C]) → layer-scale → +res
    let xtc2 = ag.ct_to_tc(x, c, t);
    let n2 = ag.rms_norm(xtc2, c, &b.ffn_norm_w, EPS);
    let h = ag.linear_bias(n2, t, c, 4 * c, &b.l1_w, &b.l1_b);
    let h = ag.gelu(h);
    let h = ag.linear_bias(h, t, 4 * c, c, &b.l2_w, &b.l2_b);
    let h = ag.scale_rows(h, t, c, &b.ffn_gamma);
    let h_ct = ag.tc_to_ct(h, t, c);
    ag.add(x, h_ct)
}

/// latent `[64, T]` (channel-major) → wav `[T_out]`.
pub fn build_decode_graph(
    w: &VibeWeights,
    t: usize,
) -> Result<(rlx_ir::Graph, Vec<(String, Vec<f32>)>, usize)> {
    let mut hir = HirModule::new("vibevoice_decode");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let lat = ag.input("latent", &[DIM_IN, t]);
    let mut x = ag.reshape(lat, vec![1, DIM_IN as i64, t as i64, 1]);

    x = causal_conv(&mut ag, x, t, &w.stem, 1); // stem (upsample_layers[0])
    let mut cur_t = t;
    for s in &w.stages[0] {
        x = block(&mut ag, x, cur_t, s);
    }
    for (i, up) in w.ups.iter().enumerate() {
        // causal transposed conv: trim_left=0, trim_right=k-stride → out = L·stride
        let (ux, _, t_up) = ag.conv_transpose1d(
            x,
            cur_t,
            &ConvTranspose1d {
                weight: &up.weight,
                bias: Some(&up.bias),
                c_in: up.c_in,
                c_out_per_group: up.c_out,
                k: up.k,
                stride: up.stride,
                groups: 1,
                trim_left: 0,
                trim_right: up.k - up.stride,
            },
        );
        cur_t = t_up;
        x = ux;
        for s in &w.stages[i + 1] {
            x = block(&mut ag, x, cur_t, s);
        }
    }
    let out = causal_conv(&mut ag, x, cur_t, &w.head, 1); // [1, 1, T_out, 1]

    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params, cur_t))
}

pub struct VibeDecoderGraph {
    compiled: CompiledAudioGraph,
    pub t: usize,
    pub t_out: usize,
}

impl VibeDecoderGraph {
    pub fn compile_for(device: Device, w: &VibeWeights, t: usize) -> Result<Self> {
        let (graph, params, t_out) = build_decode_graph(w, t)?;
        let compiled = compile(device, graph, params, 1, t_out, "latent");
        Ok(Self { compiled, t, t_out })
    }

    /// `latent` `[64, T]` channel-major → waveform `[T_out]`.
    pub fn run(&mut self, latent: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(latent)?.0)
    }
}
