// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SNAC decoder as an rlx-runtime graph on the shared audio_ops_ir foundation —
// runs natively on every backend (cpu/metal/mlx/cuda/rocm/wgpu/vulkan).

use crate::model::{EncoderW, ResidualUnitW, SnacWeights};
use anyhow::Result;
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, ConvTranspose1d, PadMode, compile, finish_graph,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;

/// Named tensors `(name, data)` produced alongside a built graph.
type NamedTensors = Vec<(String, Vec<f32>)>;

/// Decode graph build result: `(graph, params, t_wav, noise_specs)`.
type DecodeGraph = (rlx_ir::Graph, NamedTensors, usize, Vec<(String, usize)>);

fn conv_same<'w>(
    weight: &'w [f32],
    bias: Option<&'w [f32]>,
    c_out: usize,
    c_in: usize,
    k: usize,
    dil: usize,
    groups: usize,
) -> Conv1d<'w> {
    let pad = (dil * (k - 1)) / 2;
    Conv1d {
        weight,
        bias,
        c_out,
        c_in,
        k,
        stride: 1,
        dilation: dil,
        groups,
        pad_left: pad,
        pad_right: pad,
        pad_mode: PadMode::Constant,
    }
}

fn residual_unit(
    ag: &mut AudioGraph,
    x: HirNodeId,
    t: usize,
    ru: &ResidualUnitW,
) -> (HirNodeId, usize) {
    let h = ag.snake(x, ru.dim, t, &ru.snake1_alpha);
    let (h, _, ht) = ag.conv1d(
        h,
        t,
        &conv_same(
            &ru.conv1_w,
            Some(&ru.conv1_b),
            ru.dim,
            ru.dim,
            7,
            ru.conv1_dilation,
            ru.groups,
        ),
    );
    let h = ag.snake(h, ru.dim, ht, &ru.snake2_alpha);
    let (h, _, ht) = ag.conv1d(
        h,
        ht,
        &conv_same(&ru.conv2_w, Some(&ru.conv2_b), ru.dim, ru.dim, 1, 1, 1),
    );
    // SNAC residual conv stack preserves length → direct add.
    (ag.add(x, h), ht)
}

/// Build the decode graph for a fixed latent length. Inputs: `z` `[1, latent,
/// t_latent, 1]` plus one `noise_i` `[1, 1, t_i, 1]` per decoder block. Output:
/// waveform `[1, 1, t_wav, 1]`. Returns `(graph, params, t_wav, noise_specs)`.
pub fn build_decode_graph(w: &SnacWeights, t_latent: usize) -> Result<DecodeGraph> {
    let latent = w.latent();
    let mut hir = HirModule::new("snac_decode");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let z = ag.input("z", &[1, latent, t_latent, 1]);
    // init: depthwise k=7 → pointwise k=1
    // dw conv's channel count is immediately overwritten by the pw conv below; ignore it.
    let (mut h, _, mut t) = ag.conv1d(
        z,
        t_latent,
        &Conv1d {
            weight: &w.init_dw_w,
            bias: Some(&w.init_dw_b),
            c_out: latent,
            c_in: latent,
            k: 7,
            stride: 1,
            dilation: 1,
            groups: latent,
            pad_left: 3,
            pad_right: 3,
            pad_mode: PadMode::Constant,
        },
    );
    let mut c;
    (h, c, t) = ag.conv1d(
        h,
        t,
        &Conv1d {
            weight: &w.init_pw_w,
            bias: Some(&w.init_pw_b),
            c_out: w.config.decoder_dim,
            c_in: latent,
            k: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            pad_left: 0,
            pad_right: 0,
            pad_mode: PadMode::Constant,
        },
    );

    let mut noise_specs: Vec<(String, usize)> = Vec::new();
    for (bi, block) in w.blocks.iter().enumerate() {
        h = ag.snake(h, c, t, &block.snake_alpha);
        let padding = block.stride.div_ceil(2);
        let op = block.stride % 2;
        (h, c, t) = ag.conv_transpose1d(
            h,
            t,
            &ConvTranspose1d {
                weight: &block.upsample_w,
                bias: Some(&block.upsample_b),
                c_in: block.in_dim,
                c_out_per_group: block.out_dim,
                k: 2 * block.stride,
                stride: block.stride,
                groups: 1,
                trim_left: padding,
                trim_right: padding - op,
            },
        );
        if w.config.noise {
            let name = format!("noise_{bi}");
            let nin = ag.input(&name, &[1, 1, t, 1]);
            noise_specs.push((name, t));
            let (nh, _, _) = ag.conv1d(
                h,
                t,
                &Conv1d {
                    weight: &block.noise_w,
                    bias: None,
                    c_out: c,
                    c_in: c,
                    k: 1,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    pad_left: 0,
                    pad_right: 0,
                    pad_mode: PadMode::Constant,
                },
            );
            let ne = ag.broadcast_to_channels(nin, c, t);
            let prod = ag.mul(nh, ne);
            h = ag.add(h, prod);
        }
        for ru in &block.residual_units {
            let (hn, tn) = residual_unit(&mut ag, h, t, ru);
            h = hn;
            t = tn;
        }
    }

    h = ag.snake(h, c, t, &w.final_snake_alpha);
    let final_dim = w.final_dim();
    let (conv_out, _, t_wav) = ag.conv1d(
        h,
        t,
        &Conv1d {
            weight: &w.final_conv_w,
            bias: Some(&w.final_conv_b),
            c_out: 1,
            c_in: final_dim,
            k: 7,
            stride: 1,
            dilation: 1,
            groups: 1,
            pad_left: 3,
            pad_right: 3,
            pad_mode: PadMode::Constant,
        },
    );
    // SNAC decoder ends with Tanh.
    let out = ag.tanh(conv_out);

    let params = std::mem::take(&mut ag.params);
    let graph = finish_graph(hir, out)?;
    Ok((graph, params, t_wav, noise_specs))
}

/// Build the encoder graph for a fixed PCM length. Input `pcm` `[1, 1, in_len,
/// 1]` → latent `[1, latent, t, 1]`. Returns `(graph, params, latent, t)`.
pub fn build_encode_graph(
    enc: &EncoderW,
    latent_dim: usize,
    encoder_dim: usize,
    in_len: usize,
) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("snac_encode");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let x = ag.input("pcm", &[1, 1, in_len, 1]);
    // stem: 1 → encoder_dim, k=7 same-pad
    let (mut h, mut c, mut t) = ag.conv1d(
        x,
        in_len,
        &conv_same(&enc.stem_w, Some(&enc.stem_b), encoder_dim, 1, 7, 1, 1),
    );

    for block in &enc.blocks {
        for ru in &block.residual_units {
            let (hn, tn) = residual_unit(&mut ag, h, t, ru);
            h = hn;
            t = tn;
        }
        h = ag.snake(h, c, t, &block.snake_alpha);
        // downsample: strided conv k=2*stride, pad=ceil(stride/2) symmetric
        let pad = block.stride.div_ceil(2);
        let (hn, cn, tn) = ag.conv1d(
            h,
            t,
            &Conv1d {
                weight: &block.downsample_w,
                bias: Some(&block.downsample_b),
                c_out: block.output_dim,
                c_in: block.input_dim,
                k: 2 * block.stride,
                stride: block.stride,
                dilation: 1,
                groups: 1,
                pad_left: pad,
                pad_right: pad,
                pad_mode: PadMode::Constant,
            },
        );
        h = hn;
        c = cn;
        t = tn;
    }

    // final depthwise conv: latent → latent, k=7 same-pad
    let (out, _, _t) = ag.conv1d(
        h,
        t,
        &conv_same(
            &enc.final_w,
            Some(&enc.final_b),
            latent_dim,
            latent_dim,
            7,
            1,
            enc.final_groups,
        ),
    );
    let _ = c;
    let params = std::mem::take(&mut ag.params);
    let graph = finish_graph(hir, out)?;
    Ok((graph, params, t))
}

/// A compiled SNAC encoder for one device, keyed by PCM length.
pub struct SnacEncoderGraph {
    compiled: CompiledAudioGraph,
    t_latent: usize,
    latent_dim: usize,
}

impl SnacEncoderGraph {
    pub fn compile_for(device: Device, w: &SnacWeights, in_len: usize) -> Result<Self> {
        let enc = w
            .encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("encoder weights not loaded"))?;
        let latent = w.latent();
        let (graph, params, t_latent) =
            build_encode_graph(enc, latent, w.config.encoder_dim, in_len)?;
        let compiled = compile(device, graph, params, latent, t_latent, "pcm");
        Ok(Self {
            compiled,
            t_latent,
            latent_dim: latent,
        })
    }

    /// Run on mono PCM (`[in_len]`). Returns the latent `[latent, t]` row-major.
    pub fn run(&mut self, pcm: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        let (flat, _) = self.compiled.run(pcm)?;
        Ok((flat, self.latent_dim, self.t_latent))
    }
}

/// A compiled SNAC decoder for one device, keyed by latent length.
pub struct SnacDecoderGraph {
    compiled: CompiledAudioGraph,
    noise_specs: Vec<(String, usize)>,
    t_wav: usize,
}

impl SnacDecoderGraph {
    pub fn compile_for(device: Device, w: &SnacWeights, t_latent: usize) -> Result<Self> {
        let (graph, params, t_wav, noise_specs) = build_decode_graph(w, t_latent)?;
        let compiled = compile(device, graph, params, 1, t_wav, "z");
        Ok(Self {
            compiled,
            noise_specs,
            t_wav,
        })
    }

    pub fn noise_lengths(&self) -> Vec<usize> {
        self.noise_specs.iter().map(|(_, t)| *t).collect()
    }

    /// Run with quantized latent `z_q` (`[latent, t_latent]` row-major) and one
    /// noise plane per decoder block. Returns the mono waveform.
    pub fn run(&mut self, z_q: &[f32], noise: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut feed: Vec<(&str, &[f32])> = Vec::with_capacity(1 + self.noise_specs.len());
        feed.push(("z", z_q));
        for (i, (name, _t)) in self.noise_specs.iter().enumerate() {
            feed.push((name.as_str(), noise[i].as_slice()));
        }
        let (flat, _) = self.compiled.run_many(&feed)?;
        Ok(flat)
    }

    pub fn out_len(&self) -> usize {
        self.t_wav
    }
}
