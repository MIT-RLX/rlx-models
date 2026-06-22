// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// NanoCodec CausalHiFiGANDecoder as an rlx-runtime graph running natively on
// every backend: pre-conv → 5× (half_snake → grouped causal transposed conv →
// HiFiGAN residual layer) → half_snake → post-conv → clamp[-1,1]. All convs are
// causal (left-padded). Group-FSQ dequant runs on the host.

use crate::model::*;
use anyhow::Result;
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, ConvTranspose1d, PadMode, compile, finish_graph,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;

/// Named tensors `(name, data)` produced alongside a built graph.
type NamedTensors = Vec<(String, Vec<f32>)>;

/// Causal conv: left-pad by `(k-1)*dilation` zeros, no right pad.
fn causal_conv(
    ag: &mut AudioGraph,
    x: HirNodeId,
    t: usize,
    w: &ConvW,
    dilation: usize,
) -> (HirNodeId, usize) {
    let pad = (w.k - 1) * dilation;
    let (y, _, t_out) = ag.conv1d(
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
            pad_right: 0,
            pad_mode: PadMode::Constant,
        },
    );
    (y, t_out)
}

fn residual_block(
    ag: &mut AudioGraph,
    x: HirNodeId,
    c: usize,
    t: usize,
    b: &ResidualBlockW,
) -> HirNodeId {
    let h = ag.half_snake(x, c, t, &b.act0);
    let (h, _) = causal_conv(ag, h, t, &b.input_conv, b.dilation);
    let h = ag.half_snake(h, c, t, &b.act1);
    let (h, _) = causal_conv(ag, h, t, &b.skip_conv, 1);
    ag.add(x, h)
}

/// HiFiGANResLayer: mean over the 3 kernel-size blocks; each block chains its
/// 3 dilation residual blocks sequentially.
fn res_layer(
    ag: &mut AudioGraph,
    x: HirNodeId,
    c: usize,
    t: usize,
    res: &[Vec<ResidualBlockW>],
) -> HirNodeId {
    let mut acc: Option<HirNodeId> = None;
    for blocks in res {
        let mut h = x;
        for b in blocks {
            h = residual_block(ag, h, c, t, b);
        }
        acc = Some(match acc {
            None => h,
            Some(a) => ag.add(a, h),
        });
    }
    let sum = acc.unwrap();
    let inv = ag.scalar(1.0 / res.len() as f32, &[1, c, t, 1]);
    ag.mul(sum, inv)
}

/// latent `[16, T]` (channel-major) → wav `[T_out]`.
pub fn build_decode_graph(
    w: &NanoWeights,
    t: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("nanocodec_decode");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let lat = ag.input("latent", &[INPUT_DIM, t]); // [C, T]
    let mut x = ag.reshape(lat, vec![1, INPUT_DIM as i64, t as i64, 1]);

    let (px, mut cur_t) = causal_conv(&mut ag, x, t, &w.pre_conv, 1);
    x = px;

    for s in &w.stages {
        let xa = ag.half_snake(x, s.c_in, cur_t, &s.act);
        // grouped causal transposed conv: groups = c_out, trim_left=0, trim_right=k-stride.
        let (ux, _, t_up) = ag.conv_transpose1d(
            xa,
            cur_t,
            &ConvTranspose1d {
                weight: &s.up_weight,
                bias: Some(&s.up_bias),
                c_in: s.c_in,
                c_out_per_group: 1,
                k: s.k,
                stride: s.stride,
                groups: s.c_out,
                trim_left: 0,
                trim_right: s.k - s.stride,
            },
        );
        cur_t = t_up;
        x = res_layer(&mut ag, ux, s.c_out, cur_t, &s.res);
    }

    let last_c = w.stages.last().unwrap().c_out;
    let x = ag.half_snake(x, last_c, cur_t, &w.post_act);
    let (x, _) = causal_conv(&mut ag, x, cur_t, &w.post_conv, 1);
    let out = ag.clamp(x, 1, cur_t, -1.0, 1.0); // [1, 1, T_out, 1]

    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params, cur_t))
}

pub struct NanoDecoderGraph {
    compiled: CompiledAudioGraph,
    pub t: usize,
    pub t_out: usize,
}

impl NanoDecoderGraph {
    pub fn compile_for(device: Device, w: &NanoWeights, t: usize) -> Result<Self> {
        let (graph, params, t_out) = build_decode_graph(w, t)?;
        let compiled = compile(device, graph, params, 1, t_out, "latent");
        Ok(Self { compiled, t, t_out })
    }

    /// `latent` `[16, T]` channel-major → waveform `[T_out]`.
    pub fn run(&mut self, latent: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(latent)?.0)
    }

    /// Decode from Group-FSQ `codes[NUM_GROUPS][T]` → waveform.
    pub fn decode_codes(&mut self, codes: &[Vec<i64>]) -> Result<Vec<f32>> {
        let latent = fsq_decode(codes, self.t);
        self.run(&latent)
    }
}
