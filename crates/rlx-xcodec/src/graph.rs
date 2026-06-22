// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// XCodec2 CodecDecoderVocos backbone as an rlx-runtime graph: embed conv →
// prior_net (resnet) → 12× RoFormer transformer (RMSNorm + rotary MHA +
// SiLU-MLP) → post_net (resnet) → final LayerNorm → ISTFT head (host ISTFT).

use crate::model::*;
use anyhow::Result;
use rlx_core::audio_ops_ir::{
    Attention, AudioGraph, CompiledAudioGraph, Conv1d, PadMode, compile, finish_graph, rope_tables,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;

/// Named tensors `(name, data)` produced alongside a built graph.
type NamedTensors = Vec<(String, Vec<f32>)>;

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
            groups: 1,
            pad_left: pad,
            pad_right: pad,
            pad_mode: PadMode::Constant,
        },
    );
    y
}

fn resnet(ag: &mut AudioGraph, x: HirNodeId, t: usize, r: &ResnetW) -> HirNodeId {
    let h = ag.group_norm(x, DIM, &r.norm1_w, &r.norm1_b, GN_GROUPS, EPS);
    let h = ag.silu(h);
    let h = conv_sym(ag, h, t, &r.conv1, 1);
    let h = ag.group_norm(h, DIM, &r.norm2_w, &r.norm2_b, GN_GROUPS, EPS);
    let h = ag.silu(h);
    let h = conv_sym(ag, h, t, &r.conv2, 1);
    ag.add(x, h)
}

fn transformer(
    ag: &mut AudioGraph,
    x: HirNodeId,
    t: usize,
    l: &TransformerW,
    cos: &[f32],
    sin: &[f32],
) -> HirNodeId {
    let n = ag.rms_norm(x, DIM, &l.att_norm, EPS);
    let att = ag.attention(
        n,
        t,
        &Attention {
            q_w: &l.q_w,
            k_w: &l.k_w,
            v_w: &l.v_w,
            o_w: &l.o_w,
            num_heads: HEADS,
            head_dim: HEAD_DIM,
            scaling: 1.0 / (HEAD_DIM as f32).sqrt(),
        },
        None, // RoPE is a no-op here (rotates by head index → cancels in qᵀk)
        None,
        None,
    );
    let _ = (cos, sin);
    let x = ag.add(x, att);
    let n2 = ag.rms_norm(x, DIM, &l.ffn_norm, EPS);
    let m = ag.linear(n2, DIM, 4 * DIM, &l.fc1);
    let m = ag.silu(m);
    let m = ag.linear(m, 4 * DIM, DIM, &l.fc2);
    ag.add(x, m)
}

/// emb `[1, T, 1024]` (time-major) → head output `[T, 1282]`.
pub fn build_decode_graph(w: &XcodecWeights, t: usize) -> Result<(rlx_ir::Graph, NamedTensors)> {
    let half = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / ROPE_THETA.powf(2.0 * i as f32 / HEAD_DIM as f32))
        .collect();
    let (cos, sin) = rope_tables(&inv_freq, t);

    let mut hir = HirModule::new("xcodec_decode");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let emb = ag.input("emb", &[t, DIM]); // [T, C]
    let mut x = ag.tc_to_ct(emb, t, DIM); // [1, C, T, 1]
    x = conv_sym(&mut ag, x, t, &w.embed, 3);
    for r in &w.prior {
        x = resnet(&mut ag, x, t, r);
    }
    let mut xtc = ag.ct_to_tc(x, DIM, t); // [T, C]
    for l in &w.transformers {
        xtc = transformer(&mut ag, xtc, t, l, &cos, &sin);
    }
    x = ag.tc_to_ct(xtc, t, DIM);
    for r in &w.post {
        x = resnet(&mut ag, x, t, r);
    }
    let xtc = ag.ct_to_tc(x, DIM, t);
    let xtc = ag.layer_norm(xtc, DIM, &w.final_ln_w, &w.final_ln_b, EPS);
    let out = ag.linear_bias(xtc, t, DIM, 2 * (N_FFT / 2 + 1), &w.out_w, &w.out_b); // [T, 1282]

    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params))
}

pub struct XcodecDecoderGraph {
    compiled: CompiledAudioGraph,
    pub t: usize,
}

impl XcodecDecoderGraph {
    pub fn compile_for(device: Device, w: &XcodecWeights, t: usize) -> Result<Self> {
        let (graph, params) = build_decode_graph(w, t)?;
        let compiled = compile(device, graph, params, t, 2 * (N_FFT / 2 + 1), "emb");
        Ok(Self { compiled, t })
    }

    /// `emb` `[T, 1024]` row-major → head output `[T, 1282]`.
    pub fn run(&mut self, emb: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(emb)?.0)
    }
}
