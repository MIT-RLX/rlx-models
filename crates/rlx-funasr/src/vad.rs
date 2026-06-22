// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! **FSMN-VAD** — voice-activity detection.
//!
//! A Deep-FSMN classifier (affine projections + stacked causal depthwise
//! memory blocks) runs on the selected RLX device and emits per-frame
//! silence/speech posteriors; the host state machine turns those into
//! `[start_ms, end_ms]` speech segments using the silence-duration thresholds
//! from `fsmn_vad_streaming/model.py`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::built_from_hir;
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirNodeId};
use rlx_runtime::Device;

use crate::cache::GraphCache;
use crate::config::FsmnVadConfig;
use crate::frontend::WavFrontend;
use crate::sanm::Graph;
use crate::weights::RefSource;

/// A loaded FSMN-VAD model.
pub struct FsmnVad {
    cfg: FsmnVadConfig,
    weights: WeightMap,
    frontend: WavFrontend,
    device: Device,
    cache: GraphCache,
}

impl FsmnVad {
    /// Open an FSMN-VAD model directory.
    pub fn open(dir: &Path, device: Device) -> Result<Self> {
        let cfg = FsmnVadConfig::from_dir(dir)?;
        let weights = crate::weights::load_dir(dir)?;
        let cmvn = crate::frontend::load_configured_cmvn(dir);
        let frontend = WavFrontend::new(cfg.frontend.clone(), cmvn);
        Ok(Self {
            cfg,
            weights,
            frontend,
            device,
            cache: GraphCache::new(4),
        })
    }

    /// Construct from an in-memory config + weights (used by tests).
    pub fn from_parts(cfg: FsmnVadConfig, weights: WeightMap, device: Device) -> Self {
        let frontend = WavFrontend::new(cfg.frontend.clone(), None);
        Self {
            cfg,
            weights,
            frontend,
            device,
            cache: GraphCache::new(4),
        }
    }

    /// The model configuration.
    pub fn config(&self) -> &FsmnVadConfig {
        &self.cfg
    }

    /// Run the FSMN encoder over features `[t, input_dim]`; returns per-frame
    /// posteriors `[t, output_dim]`.
    pub fn run_logits(&self, feats: &[f32], t: usize) -> Result<Vec<f32>> {
        let in_dim = self.cfg.input_dim;
        ensure!(feats.len() == t * in_dim, "feature length mismatch");
        let cfg = &self.cfg;
        let weights = &self.weights;
        let build = || -> anyhow::Result<rlx_flow::BuiltModel> {
            let mut params = HashMap::new();
            let mut hir = HirModule::new("fsmn_vad").with_fusion_policy(FusionPolicy::Direct);
            {
                let mut src = RefSource(weights);
                let mut g = Graph::new(&mut hir, &mut params, &mut src);
                let x = g.input("feats", &[1, t, in_dim]);
                let out = build_fsmn(&mut g, x, cfg, t)?;
                g.set_output(out);
            }
            built_from_hir(hir, params)
        };
        self.cache
            .run(t as u64, self.device, build, &[("feats", feats)])?
            .into_iter()
            .next()
            .context("fsmn-vad produced no output")
    }

    /// Detect speech segments in mono 16 kHz PCM. Returns `[start_ms, end_ms]`.
    pub fn segments(&self, pcm: &[f32]) -> Result<Vec<(f32, f32)>> {
        let feats = self.frontend.extract(pcm);
        if feats.n_frames == 0 {
            return Ok(Vec::new());
        }
        ensure!(
            feats.feat_dim == self.cfg.input_dim,
            "frontend dim {} != vad input {}",
            feats.feat_dim,
            self.cfg.input_dim
        );
        let logits = self.run_logits(&feats.data, feats.n_frames)?;
        let sil = sil_probs(
            &logits,
            feats.n_frames,
            self.cfg.output_dim,
            self.cfg.sil_pdf_id,
        );
        let segs = decide_segments(
            &sil,
            self.cfg.frontend.frame_shift_ms,
            self.cfg.max_end_silence_ms,
            self.cfg.speech_noise_thres,
        );
        Ok(split_long(segs, self.cfg.max_single_segment_ms))
    }
}

/// Build the Deep-FSMN encoder graph → per-frame posteriors `[1, t, output_dim]`.
fn build_fsmn(g: &mut Graph, x: HirNodeId, cfg: &FsmnVadConfig, t: usize) -> Result<HirNodeId> {
    let mut h = g.linear(
        x,
        "encoder.in_linear1.linear.weight",
        Some("encoder.in_linear1.linear.bias"),
        cfg.input_affine_dim,
    )?;
    h = g.linear(
        h,
        "encoder.in_linear2.linear.weight",
        Some("encoder.in_linear2.linear.bias"),
        cfg.linear_dim,
    )?;
    h = g.g().relu(h);
    for i in 0..cfg.fsmn_layers {
        let p = format!("encoder.fsmn.{i}");
        // LinearTransform: linear_dim → proj_dim (no bias)
        h = g.linear(h, &format!("{p}.linear.linear.weight"), None, cfg.proj_dim)?;
        // FSMN memory block (causal depthwise conv + residual)
        h = g.vad_fsmn(
            h,
            cfg.proj_dim,
            cfg.lorder,
            cfg.lstride,
            t,
            &format!("{p}.fsmn_block.conv_left.weight"),
        )?;
        // AffineTransform: proj_dim → linear_dim
        h = g.linear(
            h,
            &format!("{p}.affine.linear.weight"),
            Some(&format!("{p}.affine.linear.bias")),
            cfg.linear_dim,
        )?;
        h = g.g().relu(h);
    }
    h = g.linear(
        h,
        "encoder.out_linear1.linear.weight",
        Some("encoder.out_linear1.linear.bias"),
        cfg.output_affine_dim,
    )?;
    g.linear(
        h,
        "encoder.out_linear2.linear.weight",
        Some("encoder.out_linear2.linear.bias"),
        cfg.output_dim,
    )
}

/// Per-frame silence probability (softmax over the posteriors, take `sil_pdf`).
fn sil_probs(logits: &[f32], t: usize, dim: usize, sil_pdf: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t];
    for ti in 0..t {
        let row = &logits[ti * dim..(ti + 1) * dim];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &v in row {
            sum += (v - max).exp();
        }
        let p = (row[sil_pdf] - max).exp() / sum;
        out[ti] = p;
    }
    out
}

/// Silence-duration state machine → `[start_ms, end_ms]` segments.
fn decide_segments(sil: &[f32], shift_ms: f32, max_end_sil_ms: f32, thres: f32) -> Vec<(f32, f32)> {
    let mut segs = Vec::new();
    let mut in_speech = false;
    let mut seg_start = 0usize;
    let mut sil_run = 0usize;
    let max_end = (max_end_sil_ms / shift_ms).ceil() as usize;
    for (t, &p) in sil.iter().enumerate() {
        // SenseVoice/Paraformer-style frame decision: speech iff (1-p) >= p + thres
        let is_speech = (1.0 - p) >= p + thres;
        if is_speech {
            if !in_speech {
                in_speech = true;
                seg_start = t;
            }
            sil_run = 0;
        } else if in_speech {
            sil_run += 1;
            if sil_run >= max_end {
                let seg_end = t + 1 - sil_run;
                segs.push((seg_start as f32 * shift_ms, seg_end as f32 * shift_ms));
                in_speech = false;
            }
        }
    }
    if in_speech {
        segs.push((seg_start as f32 * shift_ms, sil.len() as f32 * shift_ms));
    }
    segs
}

/// Cap segment length at `max_ms` (FunASR `max_single_segment_time`).
fn split_long(segs: Vec<(f32, f32)>, max_ms: f32) -> Vec<(f32, f32)> {
    if max_ms <= 0.0 {
        return segs;
    }
    let mut out = Vec::new();
    for (mut a, b) in segs {
        while b - a > max_ms {
            out.push((a, a + max_ms));
            a += max_ms;
        }
        out.push((a, b));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_from_sil_pattern() {
        // 10ms frames; thres 0.6 → speech iff p<=0.2. max_end 50ms → 5 frames.
        // sil pattern: 5 speech, 6 sil, 5 speech
        let mut sil = vec![0.0f32; 5];
        sil.extend(vec![1.0f32; 6]);
        sil.extend(vec![0.0f32; 5]);
        let segs = decide_segments(&sil, 10.0, 50.0, 0.6);
        assert_eq!(segs.len(), 2);
        assert!((segs[0].0 - 0.0).abs() < 1e-3);
        assert!((segs[0].1 - 50.0).abs() < 1e-3); // ends at frame 5
    }
}
