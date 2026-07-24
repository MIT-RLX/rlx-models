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

//! Compact wake CNN used by nanowakeword, porcupine, and voxrt.

use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::Path;

use crate::ops::{conv1d_nchw, gemv_bias, global_mean_pool_chw, relu, sigmoid};
use crate::ternary::{TernaryOpts, TernaryStats, is_ternary_f32, ternarize_inplace};
use crate::weights_io::{load_f32_map, save_f32_map};

/// Lite-style config (~12k–400k params depending on channels).
#[derive(Debug, Clone)]
pub struct WakeCnnConfig {
    pub n_mels: usize,
    pub c1: usize,
    pub c2: usize,
    pub c3: usize,
    pub k: usize,
    pub hidden: usize,
}

impl WakeCnnConfig {
    pub fn lite() -> Self {
        Self {
            n_mels: 32,
            c1: 16,
            c2: 32,
            c3: 32,
            k: 3,
            hidden: 64,
        }
    }

    pub fn full() -> Self {
        Self {
            n_mels: 32,
            c1: 64,
            c2: 128,
            c3: 128,
            k: 3,
            hidden: 256,
        }
    }
}

#[derive(Clone)]
pub struct WakeCnnWeights {
    pub cfg: WakeCnnConfig,
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    pub conv2_w: Vec<f32>,
    pub conv2_b: Vec<f32>,
    pub conv3_w: Vec<f32>,
    pub conv3_b: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

impl WakeCnnWeights {
    /// Deterministic stub for CI (small weights so scores stay mid-range on silence).
    pub fn stub(cfg: WakeCnnConfig) -> Self {
        let mut rng = 0xC0FFEEu64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / u32::MAX as f32) * 0.02 - 0.01
        };
        let fill = |n: usize, f: &mut dyn FnMut() -> f32| -> Vec<f32> {
            (0..n).map(|_| f()).collect()
        };
        let k = cfg.k;
        Self {
            conv1_w: fill(cfg.c1 * cfg.n_mels * k, &mut next),
            conv1_b: vec![0.0; cfg.c1],
            conv2_w: fill(cfg.c2 * cfg.c1 * k, &mut next),
            conv2_b: vec![0.0; cfg.c2],
            conv3_w: fill(cfg.c3 * cfg.c2 * k, &mut next),
            conv3_b: vec![0.0; cfg.c3],
            fc1_w: fill(cfg.hidden * cfg.c3, &mut next),
            fc1_b: vec![0.0; cfg.hidden],
            fc2_w: fill(cfg.hidden, &mut next),
            fc2_b: vec![-2.0], // bias toward low score on stub
            cfg,
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut map = HashMap::new();
        map.insert("conv1.weight".into(), self.conv1_w.clone());
        map.insert("conv1.bias".into(), self.conv1_b.clone());
        map.insert("conv2.weight".into(), self.conv2_w.clone());
        map.insert("conv2.bias".into(), self.conv2_b.clone());
        map.insert("conv3.weight".into(), self.conv3_w.clone());
        map.insert("conv3.bias".into(), self.conv3_b.clone());
        map.insert("fc1.weight".into(), self.fc1_w.clone());
        map.insert("fc1.bias".into(), self.fc1_b.clone());
        map.insert("fc2.weight".into(), self.fc2_w.clone());
        map.insert("fc2.bias".into(), self.fc2_b.clone());
        map.insert(
            "cfg".into(),
            vec![
                self.cfg.n_mels as f32,
                self.cfg.c1 as f32,
                self.cfg.c2 as f32,
                self.cfg.c3 as f32,
                self.cfg.k as f32,
                self.cfg.hidden as f32,
            ],
        );
        save_f32_map(path, &map)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let map = load_f32_map(path)?;
        let cfg_v = map
            .get("cfg")
            .ok_or_else(|| anyhow::anyhow!("missing cfg tensor"))?;
        if cfg_v.len() < 6 {
            bail!("cfg tensor too short");
        }
        let cfg = WakeCnnConfig {
            n_mels: cfg_v[0] as usize,
            c1: cfg_v[1] as usize,
            c2: cfg_v[2] as usize,
            c3: cfg_v[3] as usize,
            k: cfg_v[4] as usize,
            hidden: cfg_v[5] as usize,
        };
        Ok(Self {
            cfg,
            conv1_w: map["conv1.weight"].clone(),
            conv1_b: map["conv1.bias"].clone(),
            conv2_w: map["conv2.weight"].clone(),
            conv2_b: map["conv2.bias"].clone(),
            conv3_w: map["conv3.weight"].clone(),
            conv3_b: map["conv3.bias"].clone(),
            fc1_w: map["fc1.weight"].clone(),
            fc1_b: map["fc1.bias"].clone(),
            fc2_w: map["fc2.weight"].clone(),
            fc2_b: map["fc2.bias"].clone(),
        })
    }

    /// Exact `{−1,0,+1}` on selected tensors (biases stay f32) for bake TQ2 / fused kernels.
    pub fn ternarize(&mut self, opts: TernaryOpts) -> TernaryStats {
        let mut stats = TernaryStats::default();
        let mut apply = |w: &mut Vec<f32>| {
            ternarize_inplace(w, opts.keep_frac);
            stats.tensors += 1;
            stats.elems += w.len();
            stats.nonzero += w.iter().filter(|&&v| v != 0.0).count();
        };
        if opts.conv {
            apply(&mut self.conv1_w);
            apply(&mut self.conv2_w);
            apply(&mut self.conv3_w);
        }
        if opts.fc {
            apply(&mut self.fc1_w);
            apply(&mut self.fc2_w);
        }
        stats
    }

    pub fn fc_ternary(&self) -> bool {
        is_ternary_f32(&self.fc1_w) && is_ternary_f32(&self.fc2_w)
    }
}

pub struct WakeCnn {
    weights: WakeCnnWeights,
    mel_buf: Vec<f32>,
    /// Frames of mel kept for the CNN window.
    window_frames: usize,
}

impl WakeCnn {
    pub fn new(weights: WakeCnnWeights) -> Self {
        Self {
            // ~1.3 s of mel frames at hop 160 (~8 frames per 80 ms chunk → keep 40)
            window_frames: 40,
            weights,
            mel_buf: Vec::new(),
        }
    }

    pub fn weights(&self) -> &WakeCnnWeights {
        &self.weights
    }

    pub fn reset(&mut self) {
        self.mel_buf.clear();
    }

    /// Append mel frames `[n_frames * n_mels]` (frame-major) and score.
    pub fn push_mel_frames(&mut self, frames: &[f32]) -> f32 {
        let n_mels = self.weights.cfg.n_mels;
        if frames.is_empty() || !frames.len().is_multiple_of(n_mels) {
            return 0.0;
        }
        self.mel_buf.extend_from_slice(frames);
        let max_len = self.window_frames * n_mels;
        if self.mel_buf.len() > max_len {
            let drop = self.mel_buf.len() - max_len;
            self.mel_buf.drain(..drop);
        }
        if self.mel_buf.len() < n_mels * 8 {
            return 0.0;
        }
        self.forward_buffer()
    }

    fn forward_buffer(&self) -> f32 {
        let cfg = &self.weights.cfg;
        let n_mels = cfg.n_mels;
        let t = self.mel_buf.len() / n_mels;
        // Treat as [1, t] over mean-pooled mel energy per frame, then expand:
        // Build channel-major x: [n_mels, t]
        let mut x = vec![0.0f32; n_mels * t];
        for ti in 0..t {
            for m in 0..n_mels {
                x[m * t + ti] = self.mel_buf[ti * n_mels + m];
            }
        }
        // Conv1: in_ch = n_mels
        let mut y1 = vec![0.0f32; cfg.c1 * t];
        let t1 = conv1d_nchw(
            &x,
            n_mels,
            t,
            &self.weights.conv1_w,
            cfg.c1,
            cfg.k,
            1,
            cfg.k / 2,
            Some(&self.weights.conv1_b),
            &mut y1,
        );
        for v in &mut y1[..cfg.c1 * t1] {
            *v = relu(*v);
        }
        let mut y2 = vec![0.0f32; cfg.c2 * t1];
        let t2 = conv1d_nchw(
            &y1[..cfg.c1 * t1],
            cfg.c1,
            t1,
            &self.weights.conv2_w,
            cfg.c2,
            cfg.k,
            2,
            cfg.k / 2,
            Some(&self.weights.conv2_b),
            &mut y2,
        );
        for v in &mut y2[..cfg.c2 * t2] {
            *v = relu(*v);
        }
        let mut y3 = vec![0.0f32; cfg.c3 * t2];
        let t3 = conv1d_nchw(
            &y2[..cfg.c2 * t2],
            cfg.c2,
            t2,
            &self.weights.conv3_w,
            cfg.c3,
            cfg.k,
            2,
            cfg.k / 2,
            Some(&self.weights.conv3_b),
            &mut y3,
        );
        for v in &mut y3[..cfg.c3 * t3] {
            *v = relu(*v);
        }
        let mut pooled = vec![0.0f32; cfg.c3];
        global_mean_pool_chw(&y3[..cfg.c3 * t3], cfg.c3, t3, &mut pooled);
        let mut h = vec![0.0f32; cfg.hidden];
        gemv_bias(
            cfg.hidden,
            cfg.c3,
            &self.weights.fc1_w,
            &pooled,
            &self.weights.fc1_b,
            &mut h,
        );
        for v in &mut h {
            *v = relu(*v);
        }
        let mut logit = [0.0f32];
        gemv_bias(
            1,
            cfg.hidden,
            &self.weights.fc2_w,
            &h,
            &self.weights.fc2_b,
            &mut logit,
        );
        sigmoid(logit[0])
    }
}
