// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Speech embedding CNN: mel window `[76, n_mels]` → 96-d vector.

use anyhow::Result;
use rlx_wake::ops::{conv2d_nchw, gemv_bias, global_mean_pool_chw, relu};
use rlx_wake::weights_io::{load_f32_map, save_f32_map};
use std::collections::HashMap;
use std::path::Path;

pub const EMBED_DIM: usize = 96;
pub const MEL_WINDOW: usize = 76;
pub const MEL_STEP: usize = 8;

#[derive(Clone)]
pub struct EmbeddingWeights {
    pub n_mels: usize,
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    pub conv2_w: Vec<f32>,
    pub conv2_b: Vec<f32>,
    pub conv3_w: Vec<f32>,
    pub conv3_b: Vec<f32>,
    pub fc_w: Vec<f32>,
    pub fc_b: Vec<f32>,
    pub c1: usize,
    pub c2: usize,
    pub c3: usize,
}

impl EmbeddingWeights {
    pub fn stub(n_mels: usize) -> Self {
        let mut rng = 0xEABEDDu64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / u32::MAX as f32) * 0.02 - 0.01
        };
        let c1 = 32;
        let c2 = 64;
        let c3 = 64;
        let mut fill = |n: usize| -> Vec<f32> { (0..n).map(|_| next()).collect() };
        Self {
            n_mels,
            conv1_w: fill(c1 * 1 * 3 * 3),
            conv1_b: vec![0.0; c1],
            conv2_w: fill(c2 * c1 * 3 * 3),
            conv2_b: vec![0.0; c2],
            conv3_w: fill(c3 * c2 * 3 * 3),
            conv3_b: vec![0.0; c3],
            fc_w: fill(EMBED_DIM * c3),
            fc_b: vec![0.0; EMBED_DIM],
            c1,
            c2,
            c3,
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut map = HashMap::new();
        map.insert("embed.cfg".into(), vec![self.n_mels as f32, self.c1 as f32, self.c2 as f32, self.c3 as f32]);
        map.insert("embed.conv1.weight".into(), self.conv1_w.clone());
        map.insert("embed.conv1.bias".into(), self.conv1_b.clone());
        map.insert("embed.conv2.weight".into(), self.conv2_w.clone());
        map.insert("embed.conv2.bias".into(), self.conv2_b.clone());
        map.insert("embed.conv3.weight".into(), self.conv3_w.clone());
        map.insert("embed.conv3.bias".into(), self.conv3_b.clone());
        map.insert("embed.fc.weight".into(), self.fc_w.clone());
        map.insert("embed.fc.bias".into(), self.fc_b.clone());
        save_f32_map(path, &map)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let map = load_f32_map(path)?;
        let cfg = &map["embed.cfg"];
        Ok(Self {
            n_mels: cfg[0] as usize,
            c1: cfg[1] as usize,
            c2: cfg[2] as usize,
            c3: cfg[3] as usize,
            conv1_w: map["embed.conv1.weight"].clone(),
            conv1_b: map["embed.conv1.bias"].clone(),
            conv2_w: map["embed.conv2.weight"].clone(),
            conv2_b: map["embed.conv2.bias"].clone(),
            conv3_w: map["embed.conv3.weight"].clone(),
            conv3_b: map["embed.conv3.bias"].clone(),
            fc_w: map["embed.fc.weight"].clone(),
            fc_b: map["embed.fc.bias"].clone(),
        })
    }
}

pub struct EmbeddingNet {
    w: EmbeddingWeights,
}

impl EmbeddingNet {
    pub fn new(w: EmbeddingWeights) -> Self {
        Self { w }
    }

    /// `mel` is frame-major `[MEL_WINDOW * n_mels]`.
    pub fn forward(&self, mel: &[f32]) -> [f32; EMBED_DIM] {
        let h = MEL_WINDOW;
        let w = self.w.n_mels;
        debug_assert_eq!(mel.len(), h * w);
        // x: [1, h, w]
        let x = mel.to_vec();
        let mut y1 = vec![0.0f32; self.w.c1 * h * w];
        let (h1, w1) = conv2d_nchw(
            &x,
            1,
            h,
            w,
            &self.w.conv1_w,
            self.w.c1,
            3,
            3,
            1,
            1,
            1,
            1,
            Some(&self.w.conv1_b),
            &mut y1,
        );
        for v in &mut y1[..self.w.c1 * h1 * w1] {
            *v = relu(*v);
        }
        let mut y2 = vec![0.0f32; self.w.c2 * h1 * w1];
        let (h2, w2) = conv2d_nchw(
            &y1[..self.w.c1 * h1 * w1],
            self.w.c1,
            h1,
            w1,
            &self.w.conv2_w,
            self.w.c2,
            3,
            3,
            2,
            2,
            1,
            1,
            Some(&self.w.conv2_b),
            &mut y2,
        );
        for v in &mut y2[..self.w.c2 * h2 * w2] {
            *v = relu(*v);
        }
        let mut y3 = vec![0.0f32; self.w.c3 * h2 * w2];
        let (h3, w3) = conv2d_nchw(
            &y2[..self.w.c2 * h2 * w2],
            self.w.c2,
            h2,
            w2,
            &self.w.conv3_w,
            self.w.c3,
            3,
            3,
            2,
            2,
            1,
            1,
            Some(&self.w.conv3_b),
            &mut y3,
        );
        for v in &mut y3[..self.w.c3 * h3 * w3] {
            *v = relu(*v);
        }
        let mut pooled = vec![0.0f32; self.w.c3];
        global_mean_pool_chw(&y3[..self.w.c3 * h3 * w3], self.w.c3, h3 * w3, &mut pooled);
        let mut out = [0.0f32; EMBED_DIM];
        gemv_bias(
            EMBED_DIM,
            self.w.c3,
            &self.w.fc_w,
            &pooled,
            &self.w.fc_b,
            &mut out,
        );
        out
    }
}
