// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// The ConvNeXt VAE encoder as an rlx-runtime graph (runs natively on every
// backend). It is the mirror of the rlx-vibevoice decoder: causal strided
// down-sampling convs interleaved with ConvNeXt stages, a causal head conv, and
// the SpeechConnector (Linear → RMSNorm → Linear) projecting the VAE latent to
// the LM hidden size. Input: waveform `[T]` (T a multiple of 3200). Output:
// speech features `[n_frames, connector_dim]` (row-major).

use anyhow::Result;
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, PadMode, compile, finish_graph,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;

use crate::config::{DOWNSAMPLE_STRIDES, VAE_EPS};
use crate::weights::{BlockW, ConvW, VaeEncoderWeights};

/// Causal conv with a given stride: left-pad `k - stride` zeros, no right pad.
fn strided_causal_conv(
    ag: &mut AudioGraph,
    x: HirNodeId,
    t: usize,
    w: &ConvW,
    stride: usize,
    groups: usize,
) -> (HirNodeId, usize) {
    // For a depthwise conv the GGUF weight is `[c_out, 1, k]`, so `w.c_in` is the
    // per-group count (1), not the tensor's input-channel count. conv1d needs the
    // real input channels (= c_out for depthwise) so `c_in / groups == 1`.
    let c_in = if groups > 1 { w.c_out } else { w.c_in };
    let (y, _c, t_out) = ag.conv1d(
        x,
        t,
        &Conv1d {
            weight: &w.weight,
            bias: Some(&w.bias),
            c_out: w.c_out,
            c_in,
            k: w.k,
            stride,
            dilation: 1,
            groups,
            pad_left: w.k - stride,
            pad_right: 0,
            pad_mode: PadMode::Constant,
        },
    );
    (y, t_out)
}

/// Pointwise (1×1) conv on a `[1,C,T,1]` tensor = a per-timestep Linear over
/// channels, keeping channel-major layout (no transpose). `w` is torch
/// `[c_out, c_in]` row-major.
fn conv1x1(
    ag: &mut AudioGraph,
    x: HirNodeId,
    t: usize,
    w: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
) -> HirNodeId {
    let (y, _, _) = ag.conv1d(
        x,
        t,
        &Conv1d {
            weight: w,
            bias: Some(bias),
            c_out,
            c_in,
            k: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            pad_left: 0,
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

/// One ConvNeXt block: RMSNorm → depthwise causal conv → layer-scale → +res;
/// RMSNorm → FFN(Linear→GELU→Linear) → layer-scale → +res.
fn block(ag: &mut AudioGraph, x: HirNodeId, t: usize, b: &BlockW) -> HirNodeId {
    let c = b.dim;
    let inter = b.l1_w.len() / c; // 4·c (ffn_expansion)

    // mixer: RMSNorm(channels) → depthwise causal conv → layer-scale → +res.
    // All ops stay in channel-major `[1,C,T,1]` (no transpose — see rms_norm_ch).
    let n = ag.rms_norm_ch(x, c, t, &b.norm_w, VAE_EPS);
    let (m, _) = strided_causal_conv(ag, n, t, &b.mixer, 1, c); // depthwise groups = c
    let m = layer_scale(ag, m, c, t, &b.gamma);
    let x = ag.add(x, m);

    // ffn: RMSNorm → 1×1 conv (C→4C) → ReLU → 1×1 conv (4C→C) → layer-scale → +res.
    // The shipped VAE weights are I8_S, whose reference kernel (VibeASR.cpp
    // `ggml_nn_linear_relu`) applies ReLU — not the F32 checkpoint's GELU.
    let n2 = ag.rms_norm_ch(x, c, t, &b.ffn_norm_w, VAE_EPS);
    let h = conv1x1(ag, n2, t, &b.l1_w, &b.l1_b, c, inter);
    let h = ag.relu(h);
    let h = conv1x1(ag, h, t, &b.l2_w, &b.l2_b, inter, c);
    let h = layer_scale(ag, h, c, t, &b.ffn_gamma);
    ag.add(x, h)
}

/// Build the encoder **body** graph (waveform → VAE latent, before the
/// connector) for a padded input length `t`. Returns
/// `(graph, params, n_frames, vae_dim)`. Output is the latent `[n_frames, vae_dim]`
/// (row-major) — the point where HF applies `(latent + bias) * scale`.
pub fn build_latent_graph(
    w: &VaeEncoderWeights,
    t: usize,
) -> Result<(rlx_ir::Graph, Vec<(String, Vec<f32>)>, usize, usize)> {
    let mut hir = HirModule::new("vibevoice_encode_body");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let audio = ag.input("audio", &[1, t]);
    let mut x = ag.reshape(audio, vec![1, 1, t as i64, 1]);

    let mut cur_t = t;
    for (i, ds) in w.downsamples.iter().enumerate() {
        let stride = DOWNSAMPLE_STRIDES[i];
        let (y, t_out) = strided_causal_conv(&mut ag, x, cur_t, ds, stride, 1);
        x = y;
        cur_t = t_out;
        for b in &w.stages[i] {
            x = block(&mut ag, x, cur_t, b);
        }
    }
    // head conv (stride 1, causal) → latent, in [T', vae_dim] row-major.
    let (h, t_head) = strided_causal_conv(&mut ag, x, cur_t, &w.head, 1, 1);
    cur_t = t_head;
    let vae_dim = w.head.c_out;
    let latent = ag.ct_to_tc(h, vae_dim, cur_t);

    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, latent)?, params, cur_t, vae_dim))
}

/// Build the SpeechConnector graph: latent `[n_frames, vae_dim]` → `[n_frames, out_dim]`.
pub fn build_connector_graph(
    c: &crate::weights::ConnectorW,
    n_frames: usize,
) -> Result<(rlx_ir::Graph, Vec<(String, Vec<f32>)>, usize)> {
    let mut hir = HirModule::new("vibevoice_connector");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let x = ag.input("latent", &[n_frames, c.in_dim]);
    let h1 = ag.linear_bias(x, n_frames, c.in_dim, c.out_dim, &c.fc1_w, &c.fc1_b);
    let nrm = ag.rms_norm(h1, c.out_dim, &c.norm_w, VAE_EPS);
    let out = ag.linear_bias(nrm, n_frames, c.out_dim, c.out_dim, &c.fc2_w, &c.fc2_b);

    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params, c.out_dim))
}

/// A compiled VAE encoder for a fixed input length: body (→ latent), an optional
/// HF latent normalization `(latent − mean)/std` over the whole clip, then the
/// SpeechConnector.
pub struct VaeEncoderGraph {
    body: CompiledAudioGraph,
    connector: CompiledAudioGraph,
    pub padded_len: usize,
    pub n_frames: usize,
    pub vae_dim: usize,
    pub connector_dim: usize,
    normalize: bool,
}

impl VaeEncoderGraph {
    /// Compile for a specific padded input length (must be a multiple of 3200).
    pub fn compile_for(device: Device, w: &VaeEncoderWeights, padded_len: usize) -> Result<Self> {
        let (bg, bp, n_frames, vae_dim) = build_latent_graph(w, padded_len)?;
        let body = compile(device, bg, bp, vae_dim, n_frames, "audio");
        let (cg, cp, connector_dim) = build_connector_graph(&w.connector, n_frames)?;
        let connector = compile(device, cg, cp, connector_dim, n_frames, "latent");
        // HF normalizes the latent before the connector; disable with VIBEASR_NO_NORM=1.
        let normalize = std::env::var("VIBEASR_NO_NORM").is_err();
        Ok(Self {
            body,
            connector,
            padded_len,
            n_frames,
            vae_dim,
            connector_dim,
            normalize,
        })
    }

    /// Encode a padded waveform `[padded_len]` → features `[n_frames, connector_dim]`.
    pub fn run(&mut self, audio_padded: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            audio_padded.len() == self.padded_len,
            "audio length {} != compiled length {}",
            audio_padded.len(),
            self.padded_len
        );
        let mut latent = self.body.run(audio_padded)?.0;
        if self.normalize {
            // audio_features = (latent + bias) * scale, with
            // bias = −mean(latent), scale = 1/std(latent) over the whole clip.
            let n = latent.len().max(1) as f32;
            let mean = latent.iter().sum::<f32>() / n;
            let var = latent.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv_std = 1.0 / (var.sqrt() + 1e-8);
            for v in latent.iter_mut() {
                *v = (*v - mean) * inv_std;
            }
        }
        Ok(self.connector.run(&latent)?.0)
    }
}

/// Zero-pad a waveform up to a multiple of `compress_ratio`.
pub fn pad_to_multiple(samples: &[f32], compress_ratio: usize) -> Vec<f32> {
    let padded = samples.len().div_ceil(compress_ratio) * compress_ratio;
    let mut v = samples.to_vec();
    v.resize(padded, 0.0);
    v
}
