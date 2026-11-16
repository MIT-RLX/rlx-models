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
    /// See `rlx_wakeword_core::WakeCnnConfig::pool_chunks` (1: one average over the window).
    pub pool_chunks: usize,
    /// See `rlx_wakeword_core::WakeCnnConfig`: dilated residual blocks after the third layer,
    /// attention pooling, the PCEN front end. Run such weights with `rlx_wakeword_core`'s
    /// engine; [`WakeCnn`] here is the plain network only.
    pub res_blocks: usize,
    pub attn: bool,
    pub pcen: bool,
    /// Input = log-mel and PCEN side by side (`2·n_mels` channels), see the core crate.
    pub dual: bool,
    /// Causal, unpadded convolutions: a fully streaming network (see the core crate).
    pub causal: bool,
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
            pool_chunks: 1,
            res_blocks: 0,
            attn: false,
            pcen: false,
            dual: false,
            causal: false,
        }
    }

    /// Between the two: twice lite's channels (about 4x its multiply-adds).
    pub fn mid() -> Self {
        Self {
            n_mels: 32,
            c1: 32,
            c2: 64,
            c3: 64,
            k: 3,
            hidden: 128,
            pool_chunks: 1,
            res_blocks: 0,
            attn: false,
            pcen: false,
            dual: false,
            causal: false,
        }
    }

    /// The convolutions' zero padding each side: `k / 2`, none when `causal`.
    pub fn pad(&self) -> usize {
        if self.causal { 0 } else { self.k / 2 }
    }

    /// The first layer's input channels: `n_mels`, or twice that with `dual`.
    pub fn in_ch(&self) -> usize {
        self.n_mels * (1 + usize::from(self.dual))
    }

    /// The pooled vector's length (the first dense layer's input).
    pub fn pooled_len(&self) -> usize {
        self.c3 * (self.pool_chunks.max(1) + usize::from(self.attn))
    }

    /// `lite`, `mid` or `full`.
    pub fn named(name: &str) -> Option<Self> {
        match name {
            "lite" => Some(Self::lite()),
            "mid" => Some(Self::mid()),
            "full" => Some(Self::full()),
            _ => None,
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
            pool_chunks: 1,
            res_blocks: 0,
            attn: false,
            pcen: false,
            dual: false,
            causal: false,
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
    /// Residual blocks (`cfg.res_blocks`): `[c3, c3, k]` weights and `[c3]` biases each.
    pub res_w: Vec<Vec<f32>>,
    pub res_b: Vec<Vec<f32>>,
    /// Attention pooling (`cfg.attn`): `[c3]` score weights, `[1]` bias.
    pub attn_w: Vec<f32>,
    pub attn_b: Vec<f32>,
}

impl WakeCnnWeights {
    /// Deterministic stub for CI (small weights so scores stay mid-range on silence).
    pub fn stub(cfg: WakeCnnConfig) -> Self {
        let mut rng = 0xC0FFEEu64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) * 0.02 - 0.01
        };
        let fill =
            |n: usize, f: &mut dyn FnMut() -> f32| -> Vec<f32> { (0..n).map(|_| f()).collect() };
        let k = cfg.k;
        Self {
            conv1_w: fill(cfg.c1 * cfg.in_ch() * k, &mut next),
            conv1_b: vec![0.0; cfg.c1],
            conv2_w: fill(cfg.c2 * cfg.c1 * k, &mut next),
            conv2_b: vec![0.0; cfg.c2],
            conv3_w: fill(cfg.c3 * cfg.c2 * k, &mut next),
            conv3_b: vec![0.0; cfg.c3],
            fc1_w: fill(cfg.hidden * cfg.pooled_len(), &mut next),
            fc1_b: vec![0.0; cfg.hidden],
            fc2_w: fill(cfg.hidden, &mut next),
            fc2_b: vec![-2.0], // bias toward low score on stub
            res_w: (0..cfg.res_blocks)
                .map(|_| fill(cfg.c3 * cfg.c3 * k, &mut next))
                .collect(),
            res_b: (0..cfg.res_blocks).map(|_| vec![0.0; cfg.c3]).collect(),
            attn_w: if cfg.attn {
                fill(cfg.c3, &mut next)
            } else {
                Vec::new()
            },
            attn_b: if cfg.attn { vec![0.0] } else { Vec::new() },
            cfg,
        }
    }

    /// Weights to start training from: He (Kaiming) uniform, `±sqrt(6 / fan_in)` per layer,
    /// zero biases. [`Self::stub`]'s ±0.01 everywhere shrinks the signal ~100x per layer, so
    /// after five layers the output no longer depends on the input and SGD only learns the
    /// final bias: every trained phrase answered one class (or 0.5) to everything.
    pub fn he_init(cfg: WakeCnnConfig, seed: u64) -> Self {
        let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut unit = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let mut fill = |n: usize, fan_in: usize| -> Vec<f32> {
            let a = (6.0 / fan_in.max(1) as f32).sqrt();
            (0..n).map(|_| unit() * a).collect()
        };
        let k = cfg.k;
        Self {
            conv1_w: fill(cfg.c1 * cfg.in_ch() * k, cfg.in_ch() * k),
            conv1_b: vec![0.0; cfg.c1],
            conv2_w: fill(cfg.c2 * cfg.c1 * k, cfg.c1 * k),
            conv2_b: vec![0.0; cfg.c2],
            conv3_w: fill(cfg.c3 * cfg.c2 * k, cfg.c2 * k),
            conv3_b: vec![0.0; cfg.c3],
            fc1_w: fill(cfg.hidden * cfg.pooled_len(), cfg.pooled_len()),
            fc1_b: vec![0.0; cfg.hidden],
            fc2_w: fill(cfg.hidden, cfg.hidden),
            fc2_b: vec![0.0],
            // Residual branches start small: each block begins close to the identity.
            res_w: (0..cfg.res_blocks)
                .map(|_| {
                    fill(cfg.c3 * cfg.c3 * k, cfg.c3 * k)
                        .into_iter()
                        .map(|v| v * 0.1)
                        .collect()
                })
                .collect(),
            res_b: (0..cfg.res_blocks).map(|_| vec![0.0; cfg.c3]).collect(),
            attn_w: if cfg.attn {
                fill(cfg.c3, cfg.c3)
            } else {
                Vec::new()
            },
            attn_b: if cfg.attn { vec![0.0] } else { Vec::new() },
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
        for (i, (w, b)) in self.res_w.iter().zip(&self.res_b).enumerate() {
            map.insert(format!("res{i}.weight"), w.clone());
            map.insert(format!("res{i}.bias"), b.clone());
        }
        if self.cfg.attn {
            map.insert("attn.weight".into(), self.attn_w.clone());
            map.insert("attn.bias".into(), self.attn_b.clone());
        }
        map.insert(
            "cfg".into(),
            vec![
                self.cfg.n_mels as f32,
                self.cfg.c1 as f32,
                self.cfg.c2 as f32,
                self.cfg.c3 as f32,
                self.cfg.k as f32,
                self.cfg.hidden as f32,
                self.cfg.pool_chunks.max(1) as f32,
                self.cfg.res_blocks as f32,
                f32::from(u8::from(self.cfg.attn)),
                f32::from(u8::from(self.cfg.pcen)),
                f32::from(u8::from(self.cfg.dual)),
                f32::from(u8::from(self.cfg.causal)),
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
            // Absent in older files: one average over the window.
            pool_chunks: cfg_v.get(6).map_or(1, |&v| (v as usize).max(1)),
            res_blocks: cfg_v.get(7).map_or(0, |&v| v as usize),
            attn: cfg_v.get(8).is_some_and(|&v| v != 0.0),
            pcen: cfg_v.get(9).is_some_and(|&v| v != 0.0),
            dual: cfg_v.get(10).is_some_and(|&v| v != 0.0),
            causal: cfg_v.get(11).is_some_and(|&v| v != 0.0),
        };
        let get = |k: &str| -> Result<Vec<f32>> {
            map.get(k)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing {k}"))
        };
        Ok(Self {
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
            res_w: (0..cfg.res_blocks)
                .map(|i| get(&format!("res{i}.weight")))
                .collect::<Result<_>>()?,
            res_b: (0..cfg.res_blocks)
                .map(|i| get(&format!("res{i}.bias")))
                .collect::<Result<_>>()?,
            attn_w: if cfg.attn {
                get("attn.weight")?
            } else {
                Vec::new()
            },
            attn_b: if cfg.attn {
                get("attn.bias")?
            } else {
                Vec::new()
            },
            cfg,
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
        let c = &weights.cfg;
        assert!(
            c.res_blocks == 0 && !c.attn && !c.pcen && !c.dual && !c.causal,
            "residual blocks, attention pooling and PCEN run in rlx_wakeword_core::WakeCnn"
        );
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
        let chunks = cfg.pool_chunks.max(1);
        let mut pooled = vec![0.0f32; cfg.c3 * chunks];
        if chunks == 1 {
            global_mean_pool_chw(&y3[..cfg.c3 * t3], cfg.c3, t3, &mut pooled);
        } else {
            rlx_wakeword_core::cnn::pool_chunks(
                &y3[..cfg.c3 * t3],
                cfg.c3,
                t3,
                chunks,
                &mut pooled,
            );
        }
        let mut h = vec![0.0f32; cfg.hidden];
        gemv_bias(
            cfg.hidden,
            cfg.c3 * chunks,
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
