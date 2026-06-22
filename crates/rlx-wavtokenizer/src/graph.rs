// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// WavTokenizer Vocos decoder backbone as an rlx-runtime graph (runs on every
// backend). Produces the magnitude/phase head output `[T, 1282]`; the ISTFT is
// host-side (see istft.rs).

use crate::model::*;
use anyhow::Result;
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, PadMode, compile, finish_graph,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;

fn conv_sym(ag: &mut AudioGraph, x: HirNodeId, t: usize, w: &ConvW, pad: usize) -> HirNodeId {
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
            groups: w.groups,
            pad_left: pad,
            pad_right: pad,
            pad_mode: PadMode::Constant,
        },
    );
    y
}

fn resnet(ag: &mut AudioGraph, x: HirNodeId, t: usize, r: &ResnetBlockW) -> HirNodeId {
    let h = ag.group_norm(x, DIM, &r.norm1_w, &r.norm1_b, GN_GROUPS, EPS);
    let h = ag.silu(h);
    let h = conv_sym(ag, h, t, &r.conv1, 1);
    let h = ag.group_norm(h, DIM, &r.norm2_w, &r.norm2_b, GN_GROUPS, EPS);
    let h = ag.silu(h);
    let h = conv_sym(ag, h, t, &r.conv2, 1);
    ag.add(x, h)
}

fn attn(ag: &mut AudioGraph, x: HirNodeId, t: usize, a: &AttnBlockW) -> HirNodeId {
    let h = ag.group_norm(x, DIM, &a.norm_w, &a.norm_b, GN_GROUPS, EPS);
    let q = conv_sym(ag, h, t, &a.q, 0);
    let k = conv_sym(ag, h, t, &a.k, 0);
    let v = conv_sym(ag, h, t, &a.v, 0);
    let qc = ag.reshape(q, vec![DIM as i64, t as i64]); // [C, T]
    let kc = ag.reshape(k, vec![DIM as i64, t as i64]);
    let vc = ag.reshape(v, vec![DIM as i64, t as i64]);
    let qt = ag.transpose(qc, vec![1, 0]); // [T, C]
    let scores = ag.matmul(qt, kc); // [T, T] : scores[a,b] = Σc q[c,a]k[c,b]
    let sc = ag.scalar(1.0 / (DIM as f32).sqrt(), &[t, t]);
    let scores = ag.mul(scores, sc);
    let attn = ag.softmax(scores, -1); // softmax over keys (last axis)
    let attnt = ag.transpose(attn, vec![1, 0]); // [T, T]
    let out = ag.matmul(vc, attnt); // [C, T] : out[c,j] = Σi v[c,i] attn[j,i]
    let out4 = ag.reshape(out, vec![1, DIM as i64, t as i64, 1]);
    let proj = conv_sym(ag, out4, t, &a.proj_out, 0);
    ag.add(x, proj)
}

fn convnext(ag: &mut AudioGraph, x: HirNodeId, t: usize, b: &ConvNextW) -> HirNodeId {
    let h = conv_sym(ag, x, t, &b.dwconv, 3); // depthwise
    let htc = ag.ct_to_tc(h, DIM, t); // [T, C]
    let htc = ag.layer_norm(htc, DIM, &b.norm.scale, &b.norm.shift, LN_EPS); // AdaLN(bw=0) = LN affine
    let htc = ag.linear_bias(htc, t, DIM, INTERMEDIATE, &b.pwconv1_w, &b.pwconv1_b);
    let htc = ag.gelu(htc);
    let htc = ag.linear_bias(htc, t, INTERMEDIATE, DIM, &b.pwconv2_w, &b.pwconv2_b);
    let htc = ag.scale_rows(htc, t, DIM, &b.gamma);
    let hx = ag.tc_to_ct(htc, t, DIM);
    ag.add(x, hx)
}

/// Build the decoder graph. Input `feats` `[1, 512, t, 1]`; output `[t, 1282]`
/// (head magnitude/phase, pre-ISTFT).
pub fn build_decode_graph(
    w: &WavtokWeights,
    t: usize,
) -> Result<(rlx_ir::Graph, Vec<(String, Vec<f32>)>, usize)> {
    let bb = &w.backbone;
    let mut hir = HirModule::new("wavtok_decode");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let feats = ag.input("feats", &[1, INPUT_CH, t, 1]);
    let mut x = conv_sym(&mut ag, feats, t, &bb.embed, 3); // [1,768,t,1]

    x = resnet(&mut ag, x, t, &bb.resnets[0]);
    x = resnet(&mut ag, x, t, &bb.resnets[1]);
    x = attn(&mut ag, x, t, &bb.attn);
    x = resnet(&mut ag, x, t, &bb.resnets[2]);
    x = resnet(&mut ag, x, t, &bb.resnets[3]);
    x = ag.group_norm(x, DIM, &bb.posnet_gn_w, &bb.posnet_gn_b, GN_GROUPS, EPS);

    let xtc = ag.ct_to_tc(x, DIM, t);
    let xtc = ag.layer_norm(xtc, DIM, &bb.norm.scale, &bb.norm.shift, LN_EPS);
    x = ag.tc_to_ct(xtc, t, DIM);

    for cb in &bb.convnext {
        x = convnext(&mut ag, x, t, cb);
    }

    let xtc = ag.ct_to_tc(x, DIM, t);
    let xtc = ag.layer_norm(xtc, DIM, &bb.final_ln_w, &bb.final_ln_b, LN_EPS);
    let out = ag.linear_bias(
        xtc,
        t,
        DIM,
        2 * (N_FFT / 2 + 1),
        &w.head.out_w,
        &w.head.out_b,
    ); // [t, 1282]

    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params, 2 * (N_FFT / 2 + 1)))
}

pub struct WavtokDecoderGraph {
    compiled: CompiledAudioGraph,
    pub t: usize,
    pub cols: usize,
}

impl WavtokDecoderGraph {
    pub fn compile_for(device: Device, w: &WavtokWeights, t: usize) -> Result<Self> {
        let (graph, params, cols) = build_decode_graph(w, t)?;
        let compiled = compile(device, graph, params, t, cols, "feats");
        Ok(Self { compiled, t, cols })
    }

    /// Run with `feats` `[512, t]` row-major → head output `[t, 1282]` row-major.
    pub fn run(&mut self, feats: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(feats)?.0)
    }
}
