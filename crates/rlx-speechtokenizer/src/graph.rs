// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SpeechTokenizer SEANet conv stacks as rlx-runtime graphs (NON-causal reflect
// padding). LSTM bottlenecks (encoder bidirectional, decoder unidirectional) +
// euclidean RVQ run on the host.

use crate::model::{ConvW, DecoderW, EncoderW, ResnetW};
use anyhow::Result;
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, ConvTranspose1d, PadMode, compile, finish_graph,
    noncausal_pad,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;

/// Named tensors `(name, data)` produced alongside a built graph.
type NamedTensors = Vec<(String, Vec<f32>)>;

fn nc_conv(ag: &mut AudioGraph, x: HirNodeId, x_t: usize, w: &ConvW) -> (HirNodeId, usize, usize) {
    let (pl, pr) = noncausal_pad(x_t, w.k, w.stride, w.dilation);
    let mode = if pl == 0 && pr == 0 {
        PadMode::Constant
    } else {
        PadMode::Reflect
    };
    ag.conv1d(
        x,
        x_t,
        &Conv1d {
            weight: &w.weight,
            bias: Some(&w.bias),
            c_out: w.c_out,
            c_in: w.c_in,
            k: w.k,
            stride: w.stride,
            dilation: w.dilation,
            groups: 1,
            pad_left: pl,
            pad_right: pr,
            pad_mode: mode,
        },
    )
}

fn nc_transpose(
    ag: &mut AudioGraph,
    x: HirNodeId,
    x_t: usize,
    w: &ConvW,
) -> (HirNodeId, usize, usize) {
    let padding_total = w.k - w.stride;
    let trim_right = padding_total / 2;
    let trim_left = padding_total - trim_right;
    ag.conv_transpose1d(
        x,
        x_t,
        &ConvTranspose1d {
            weight: &w.weight,
            bias: Some(&w.bias),
            c_in: w.c_in,
            c_out_per_group: w.c_out,
            k: w.k,
            stride: w.stride,
            groups: 1,
            trim_left,
            trim_right,
        },
    )
}

fn resnet(ag: &mut AudioGraph, x: HirNodeId, dim: usize, t: usize, r: &ResnetW) -> HirNodeId {
    let (sc, _, _) = nc_conv(ag, x, t, &r.shortcut);
    let h = ag.elu(x, dim, t);
    let (h, hc, ht) = nc_conv(ag, h, t, &r.conv1);
    let h = ag.elu(h, hc, ht);
    let (h, _, _) = nc_conv(ag, h, ht, &r.conv2);
    ag.add(sc, h)
}

/// PCM `[1,1,in_len,1]` → pre-LSTM `[1, dim, t, 1]`.
pub fn build_enc_pre(
    enc: &EncoderW,
    in_len: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize, usize)> {
    let mut hir = HirModule::new("st_enc_pre");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);
    let x = ag.input("pcm", &[1, 1, in_len, 1]);
    let (mut h, mut c, mut t) = nc_conv(&mut ag, x, in_len, &enc.stem);
    for stage in &enc.stages {
        h = resnet(&mut ag, h, c, t, &stage.resnet);
        h = ag.elu(h, c, t);
        let (hn, cn, tn) = nc_conv(&mut ag, h, t, &stage.downsample);
        h = hn;
        c = cn;
        t = tn;
    }
    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, h)?, params, c, t))
}

/// Bidirectional-LSTM output `[1, 2*dim, in_t, 1]` → latent `[1, dim, in_t, 1]`
/// (ELU → final conv 2*dim→dim).
pub fn build_enc_post(
    enc: &EncoderW,
    two_dim: usize,
    in_t: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("st_enc_post");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);
    let z = ag.input("z", &[1, two_dim, in_t, 1]);
    let h = ag.elu(z, two_dim, in_t);
    let (out, oc, _t) = nc_conv(&mut ag, h, in_t, &enc.final_conv);
    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params, oc))
}

/// Quantized latent `[1, dim, in_t, 1]` → pre-LSTM `[1, dim, in_t, 1]`.
pub fn build_dec_pre(dec: &DecoderW, in_t: usize) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("st_dec_pre");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);
    let x = ag.input("zq", &[1, dec.conv0.c_in, in_t, 1]);
    let (out, oc, _t) = nc_conv(&mut ag, x, in_t, &dec.conv0);
    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params, oc))
}

/// Post-LSTM `[1, dim, in_t, 1]` → waveform `[1, 1, out_len, 1]`.
pub fn build_dec_post(
    dec: &DecoderW,
    dim: usize,
    in_t: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("st_dec_post");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);
    let z = ag.input("z", &[1, dim, in_t, 1]);
    let (mut h, mut c, mut t) = (z, dim, in_t);
    for stage in &dec.stages {
        h = ag.elu(h, c, t);
        let (hn, cn, tn) = nc_transpose(&mut ag, h, t, &stage.transpose);
        h = resnet(&mut ag, hn, cn, tn, &stage.resnet);
        c = cn;
        t = tn;
    }
    h = ag.elu(h, c, t);
    let (out, _oc, out_t) = nc_conv(&mut ag, h, t, &dec.final_conv);
    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params, out_t))
}

/// Generic single-input compiled graph runner.
pub struct StGraph {
    compiled: CompiledAudioGraph,
    pub out_c: usize,
    pub out_t: usize,
}

impl StGraph {
    pub fn new(
        device: Device,
        graph: rlx_ir::Graph,
        params: NamedTensors,
        out_c: usize,
        out_t: usize,
        input: &str,
    ) -> Self {
        Self {
            compiled: compile(device, graph, params, out_c, out_t, input),
            out_c,
            out_t,
        }
    }
    pub fn run(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(x)?.0)
    }
}
