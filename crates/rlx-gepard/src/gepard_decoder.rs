// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard HiFi-GAN decoder — `nano_dec_1.89kbps.safetensors`.
//!
//! Mirrors [`rlx_nanocodec::NanoDecoderGraph`] but:
//! - Loads from Gepard's key naming: `pre_conv.*`, `stage{i}.*`, `s{i}.rb{ki}_{di}.*`
//! - 32 input channels (vs NanoCodec's 16)
//! - Up-rates [8, 8, 4, 2, 2] → 1024 samples/frame at 22 050 Hz
//! - FSQ dequant with Gepard levels [8, 7, 6, 6] × 8 groups
//!
//! The same `AudioGraph` IR ops as NanoDecoderGraph are used so execution is
//! device-accelerated (CPU / Metal / MLX / etc.).

use anyhow::{Context, Result};
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, ConvTranspose1d, PadMode, compile, finish_graph,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_nanocodec::model::{ConvW, NanoWeights, ResidualBlockW, StageW};
use rlx_runtime::Device;
use safetensors::SafeTensors;

use crate::codec_ops::{FSQ_LEVELS, NUM_CHANNELS};

/// Samples per codec frame = 8 × 8 × 4 × 2 × 2.
pub const SAMPLES_PER_FRAME: usize = 1024;

const GEPARD_UP_RATES: [usize; 5] = [8, 8, 4, 2, 2];

// ── weight loading ────────────────────────────────────────────────────────────

fn tf32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    use safetensors::tensor::Dtype;
    let t = st
        .tensor(name)
        .with_context(|| format!("missing key: {name}"))?;
    let raw = t.data();
    match t.dtype() {
        Dtype::F32 => Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        Dtype::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|b| {
                let bits = u32::from(u16::from_le_bytes([b[0], b[1]])) << 16;
                f32::from_bits(bits)
            })
            .collect()),
        dt => anyhow::bail!("{name}: unsupported dtype {dt:?}"),
    }
}

fn conv_from_wb(w: Vec<f32>, b: Vec<f32>, c_out: usize, c_in: usize, k: usize) -> ConvW {
    ConvW {
        weight: w,
        bias: b,
        c_out,
        c_in,
        k,
    }
}

/// Load `nano_dec_1.89kbps.safetensors` into `NanoWeights`.
/// Key format: `pre_conv.weight/bias`, `stage{i}.act/up_weight/up_bias`,
/// `s{i}.rb{ki}_{di}.act0/act1/ic_w/ic_b/sc_w/sc_b`.
pub fn load_gepard_nano_weights(bytes: &[u8]) -> Result<NanoWeights> {
    let st = SafeTensors::deserialize(bytes).context("parse gepard decoder")?;

    let pre_conv = conv_from_wb(
        tf32(&st, "pre_conv.weight")?,
        tf32(&st, "pre_conv.bias")?,
        864,
        NUM_CHANNELS,
        7,
    );

    let mut stages = Vec::new();
    let mut in_ch = 864usize;

    for (si, &stride) in GEPARD_UP_RATES.iter().enumerate() {
        let out_ch = in_ch / 2;
        let k = stride * 2;

        let kernels = [3usize, 7, 11];
        let dilations = [1usize, 3, 5];
        let mut res = Vec::new();
        for (ki, &kk) in kernels.iter().enumerate() {
            let mut blocks = Vec::new();
            for (di, &dil) in dilations.iter().enumerate() {
                let bp = format!("s{si}.rb{ki}_{di}");
                blocks.push(ResidualBlockW {
                    act0: tf32(&st, &format!("{bp}.act0"))?,
                    input_conv: conv_from_wb(
                        tf32(&st, &format!("{bp}.ic_w"))?,
                        tf32(&st, &format!("{bp}.ic_b"))?,
                        out_ch,
                        out_ch,
                        kk,
                    ),
                    dilation: dil,
                    act1: tf32(&st, &format!("{bp}.act1"))?,
                    skip_conv: conv_from_wb(
                        tf32(&st, &format!("{bp}.sc_w"))?,
                        tf32(&st, &format!("{bp}.sc_b"))?,
                        out_ch,
                        out_ch,
                        kk,
                    ),
                });
            }
            res.push(blocks);
        }

        stages.push(StageW {
            act: tf32(&st, &format!("stage{si}.act"))?,
            up_weight: tf32(&st, &format!("stage{si}.up_weight"))?,
            up_bias: tf32(&st, &format!("stage{si}.up_bias"))?,
            c_in: in_ch,
            c_out: out_ch,
            k,
            stride,
            res,
        });
        in_ch = out_ch;
    }

    let post_act = tf32(&st, "post_act")?;
    let post_conv = conv_from_wb(
        tf32(&st, "post_conv.weight")?,
        tf32(&st, "post_conv.bias")?,
        1,
        in_ch,
        3,
    );

    Ok(NanoWeights {
        pre_conv,
        stages,
        post_act,
        post_conv,
    })
}

// ── FSQ dequantization ────────────────────────────────────────────────────────

/// Dequantize Gepard FSQ frames to `[NUM_CHANNELS, T]` channel-major.
/// Official formula (`gepard/model/codec_ops.py`): `(x - L//2) / (L//2)`.
pub fn gepard_fsq_decode(frames: &[Vec<u32>]) -> Vec<f32> {
    let t = frames.len();
    let mut out = vec![0.0f32; NUM_CHANNELS * t];
    for (fi, frame) in frames.iter().enumerate() {
        for ch in 0..NUM_CHANNELS {
            let l = FSQ_LEVELS[ch % FSQ_LEVELS.len()] as i32;
            let half = (l / 2).max(1) as f32;
            let code = (frame[ch] as i32).clamp(0, l - 1) as f32;
            out[ch * t + fi] = (code - half) / half;
        }
    }
    out
}

// ── IR graph (mirrors nanocodec/graph.rs, input_dim from w.pre_conv.c_in) ─────

type NamedTensors = Vec<(String, Vec<f32>)>;

fn causal_conv_ir(
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

fn residual_block_ir(
    ag: &mut AudioGraph,
    x: HirNodeId,
    c: usize,
    t: usize,
    b: &ResidualBlockW,
) -> HirNodeId {
    let h = ag.half_snake(x, c, t, &b.act0);
    let (h, _) = causal_conv_ir(ag, h, t, &b.input_conv, b.dilation);
    let h = ag.half_snake(h, c, t, &b.act1);
    let (h, _) = causal_conv_ir(ag, h, t, &b.skip_conv, 1);
    ag.add(x, h)
}

fn res_layer_ir(
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
            h = residual_block_ir(ag, h, c, t, b);
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

fn build_gepard_graph(w: &NanoWeights, t: usize) -> Result<(rlx_ir::Graph, NamedTensors, usize)> {
    let c_in = w.pre_conv.c_in; // 32

    let mut hir = HirModule::new("gepard_decode");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);

    let lat = ag.input("latent", &[c_in, t]);
    let mut x = ag.reshape(lat, vec![1, c_in as i64, t as i64, 1]);
    let (px, mut cur_t) = causal_conv_ir(&mut ag, x, t, &w.pre_conv, 1);
    x = px;

    for s in &w.stages {
        let xa = ag.half_snake(x, s.c_in, cur_t, &s.act);
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
        x = res_layer_ir(&mut ag, ux, s.c_out, cur_t, &s.res);
    }

    let last_c = w.stages.last().unwrap().c_out;
    let x = ag.half_snake(x, last_c, cur_t, &w.post_act);
    let (x, _) = causal_conv_ir(&mut ag, x, cur_t, &w.post_conv, 1);
    let out = ag.clamp(x, 1, cur_t, -1.0, 1.0);

    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params, cur_t))
}

// ── public API ────────────────────────────────────────────────────────────────

pub struct GepardDecoderGraph {
    compiled: CompiledAudioGraph,
    pub t: usize,
    pub t_out: usize,
}

impl GepardDecoderGraph {
    pub fn compile_for(device: Device, w: &NanoWeights, t: usize) -> Result<Self> {
        let (graph, params, t_out) = build_gepard_graph(w, t)?;
        let compiled = compile(device, graph, params, 1, t_out, "latent");
        Ok(Self { compiled, t, t_out })
    }

    /// Decode Gepard FSQ frames to PCM.
    pub fn decode(&mut self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        let latent = gepard_fsq_decode(frames);
        Ok(self.compiled.run(&latent)?.0)
    }
}

/// High-level Gepard decoder: stores weights and compiles an IR graph on each call.
///
/// Compilation is O(1) in weight size and fast for typical frame counts.
pub struct GepardDecoder {
    weights: NanoWeights,
    device: Device,
}

impl GepardDecoder {
    /// Load from `nano_dec_1.89kbps.safetensors` bytes (CPU decode).
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        Self::from_safetensors_on(bytes, Device::Cpu)
    }

    pub fn from_safetensors_on(bytes: &[u8], device: Device) -> Result<Self> {
        let weights = load_gepard_nano_weights(bytes)?;
        Ok(Self { weights, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Decode codec `frames` → PCM f32 in [-1, 1].
    /// Compiles the RLX IR graph for the exact frame count on each call.
    pub fn decode(&self, frames: &[Vec<u32>]) -> Vec<f32> {
        let t = frames.len();
        if t == 0 {
            return vec![];
        }
        match GepardDecoderGraph::compile_for(self.device, &self.weights, t) {
            Ok(mut g) => g.decode(frames).unwrap_or_else(|e| {
                eprintln!("[gepard_decoder] decode failed: {e}");
                vec![0.0; t * SAMPLES_PER_FRAME]
            }),
            Err(e) => {
                eprintln!("[gepard_decoder] compile failed: {e}");
                vec![0.0; t * SAMPLES_PER_FRAME]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn decoder_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../weights/tts/gepard/nano_dec_1.89kbps.safetensors")
    }

    #[test]
    fn load_weights_smoke() {
        let path = decoder_path();
        if !path.is_file() {
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        let w = load_gepard_nano_weights(&bytes).expect("load gepard weights");
        assert_eq!(w.pre_conv.c_in, NUM_CHANNELS);
        assert_eq!(w.stages.len(), 5);
        assert_eq!(w.stages[0].stride, 8);
        assert_eq!(w.stages[2].stride, 4);
    }

    #[test]
    fn decode_smoke() {
        let path = decoder_path();
        if !path.is_file() {
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        let w = load_gepard_nano_weights(&bytes).unwrap();
        let frames: Vec<Vec<u32>> = (0..4).map(|_| vec![3u32; NUM_CHANNELS]).collect();
        let mut g = GepardDecoderGraph::compile_for(Device::Cpu, &w, 4).unwrap();
        let audio = g.decode(&frames).unwrap();
        assert_eq!(audio.len(), 4 * SAMPLES_PER_FRAME);
        for s in &audio {
            assert!(s.abs() <= 1.0, "sample {s} out of range");
        }
    }
}
