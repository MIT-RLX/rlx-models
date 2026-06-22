// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// EnCodec SEANet conv stacks as rlx-runtime graphs on the shared audio_ops_ir
// foundation. The encoder is split around the host LSTM bottleneck: a pre-LSTM
// graph (stem + downsampling stages) and a post-LSTM graph (ELU + final conv).
// All convs are causal (EnCodec `use_causal_conv=True`).

use crate::model::{ConvW, DecoderW, EncoderW, ResnetW};
use anyhow::Result;
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, ConvTranspose1d, PadMode, causal_pad, compile,
    finish_graph,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;

/// Named tensors `(name, data)` produced alongside a built graph.
type NamedTensors = Vec<(String, Vec<f32>)>;

fn causal_conv(
    ag: &mut AudioGraph,
    x: HirNodeId,
    x_t: usize,
    w: &ConvW,
) -> (HirNodeId, usize, usize) {
    let (pl, pr) = causal_pad(x_t, w.k, w.stride, w.dilation);
    // EnCodec pads causally with reflect mode (falls back to no-op when pad==0).
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

/// EnCodec resnet block: `shortcut(x) + [ELU → conv1(k3) → ELU → conv2(k1)](x)`.
/// Length-preserving (causal). Returns `(node, dim)`.
fn resnet(ag: &mut AudioGraph, x: HirNodeId, dim: usize, t: usize, r: &ResnetW) -> HirNodeId {
    let (sc, _, _) = causal_conv(ag, x, t, &r.shortcut);
    let h = ag.elu(x, dim, t);
    let (h, hc, ht) = causal_conv(ag, h, t, &r.conv1);
    let h = ag.elu(h, hc, ht);
    let (h, _, _) = causal_conv(ag, h, ht, &r.conv2);
    ag.add(sc, h)
}

/// PCM `[1,1,in_len,1]` → pre-LSTM latent `[1, lstm_dim, t, 1]`.
pub fn build_pre_lstm_graph(
    enc: &EncoderW,
    in_len: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize, usize)> {
    let mut hir = HirModule::new("encodec_pre_lstm");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let x = ag.input("pcm", &[1, 1, in_len, 1]);
    let (mut h, mut c, mut t) = causal_conv(&mut ag, x, in_len, &enc.stem);
    for stage in &enc.stages {
        h = resnet(&mut ag, h, c, t, &stage.resnet);
        h = ag.elu(h, c, t);
        let (hn, cn, tn) = causal_conv(&mut ag, h, t, &stage.downsample);
        h = hn;
        c = cn;
        t = tn;
    }
    let params = std::mem::take(&mut ag.params);
    let graph = finish_graph(hir, h)?;
    Ok((graph, params, c, t))
}

/// Post-LSTM latent `[1, lstm_dim, in_t, 1]` → encoder output `[1, hidden, in_t, 1]`.
pub fn build_post_lstm_graph(
    enc: &EncoderW,
    lstm_dim: usize,
    in_t: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("encodec_post_lstm");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let z = ag.input("z", &[1, lstm_dim, in_t, 1]);
    let h = ag.elu(z, lstm_dim, in_t);
    let (out, oc, _ot) = causal_conv(&mut ag, h, in_t, &enc.final_conv);
    let params = std::mem::take(&mut ag.params);
    let graph = finish_graph(hir, out)?;
    Ok((graph, params, oc))
}

/// EnCodec causal transposed conv: trim_left=0, trim_right = k - stride.
fn causal_transpose(
    ag: &mut AudioGraph,
    x: HirNodeId,
    x_t: usize,
    w: &ConvW,
) -> (HirNodeId, usize, usize) {
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
            trim_left: 0,
            trim_right: w.k - w.stride,
        },
    )
}

/// Quantized latent `[1, hidden, in_t, 1]` → pre-LSTM `[1, lstm_dim, in_t, 1]`.
pub fn build_decode_pre_lstm_graph(
    dec: &DecoderW,
    in_t: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("encodec_dec_pre");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);
    let x = ag.input("zq", &[1, dec.conv0.c_in, in_t, 1]);
    let (out, oc, _t) = causal_conv(&mut ag, x, in_t, &dec.conv0);
    let params = std::mem::take(&mut ag.params);
    let graph = finish_graph(hir, out)?;
    Ok((graph, params, oc))
}

/// Post-LSTM `[1, lstm_dim, in_t, 1]` → waveform `[1, 1, out_len, 1]`.
pub fn build_decode_post_lstm_graph(
    dec: &DecoderW,
    lstm_dim: usize,
    in_t: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("encodec_dec_post");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);
    let z = ag.input("z", &[1, lstm_dim, in_t, 1]);
    let (mut h, mut c, mut t) = (z, lstm_dim, in_t);
    for stage in &dec.stages {
        h = ag.elu(h, c, t);
        let (hn, cn, tn) = causal_transpose(&mut ag, h, t, &stage.transpose);
        h = resnet(&mut ag, hn, cn, tn, &stage.resnet);
        c = cn;
        t = tn;
    }
    h = ag.elu(h, c, t);
    let (out, _oc, out_t) = causal_conv(&mut ag, h, t, &dec.final_conv);
    let params = std::mem::take(&mut ag.params);
    let graph = finish_graph(hir, out)?;
    Ok((graph, params, out_t))
}

pub struct DecodePreLstmGraph {
    compiled: CompiledAudioGraph,
    pub out_c: usize,
    pub out_t: usize,
}

impl DecodePreLstmGraph {
    pub fn compile_for(device: Device, dec: &DecoderW, in_t: usize) -> Result<Self> {
        let (graph, params, out_c) = build_decode_pre_lstm_graph(dec, in_t)?;
        let compiled = compile(device, graph, params, out_c, in_t, "zq");
        Ok(Self {
            compiled,
            out_c,
            out_t: in_t,
        })
    }
    pub fn run(&mut self, zq: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(zq)?.0)
    }
}

pub struct DecodePostLstmGraph {
    compiled: CompiledAudioGraph,
    pub out_len: usize,
}

impl DecodePostLstmGraph {
    pub fn compile_for(
        device: Device,
        dec: &DecoderW,
        lstm_dim: usize,
        in_t: usize,
    ) -> Result<Self> {
        let (graph, params, out_len) = build_decode_post_lstm_graph(dec, lstm_dim, in_t)?;
        let compiled = compile(device, graph, params, 1, out_len, "z");
        Ok(Self { compiled, out_len })
    }
    pub fn run(&mut self, z: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(z)?.0)
    }
}

pub struct PreLstmGraph {
    compiled: CompiledAudioGraph,
    pub out_c: usize,
    pub out_t: usize,
}

impl PreLstmGraph {
    pub fn compile_for(device: Device, enc: &EncoderW, in_len: usize) -> Result<Self> {
        let (graph, params, out_c, out_t) = build_pre_lstm_graph(enc, in_len)?;
        let compiled = compile(device, graph, params, out_c, out_t, "pcm");
        Ok(Self {
            compiled,
            out_c,
            out_t,
        })
    }

    pub fn run(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(pcm)?.0)
    }
}

pub struct PostLstmGraph {
    compiled: CompiledAudioGraph,
    pub out_c: usize,
    pub out_t: usize,
}

impl PostLstmGraph {
    pub fn compile_for(
        device: Device,
        enc: &EncoderW,
        lstm_dim: usize,
        in_t: usize,
    ) -> Result<Self> {
        let (graph, params, out_c) = build_post_lstm_graph(enc, lstm_dim, in_t)?;
        let compiled = compile(device, graph, params, out_c, in_t, "z");
        Ok(Self {
            compiled,
            out_c,
            out_t: in_t,
        })
    }

    pub fn run(&mut self, z: &[f32]) -> Result<Vec<f32>> {
        Ok(self.compiled.run(z)?.0)
    }
}
