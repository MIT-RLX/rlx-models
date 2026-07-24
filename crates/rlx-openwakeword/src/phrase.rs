// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Phrase classifier head: last 16 embeddings → score.

use anyhow::Result;
use rlx_wake::ops::{gemv_bias, relu, sigmoid};
use rlx_wake::weights_io::{load_f32_map, save_f32_map};
use std::collections::HashMap;
use std::path::Path;

use crate::embedding::EMBED_DIM;

pub const EMBED_HISTORY: usize = 16;

#[derive(Clone)]
pub struct PhraseWeights {
    pub hidden: usize,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    pub keyword: String,
}

impl PhraseWeights {
    pub fn stub(keyword: &str) -> Self {
        let hidden = 64;
        let in_dim = EMBED_HISTORY * EMBED_DIM;
        let mut rng = 0xA11CEu64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / u32::MAX as f32) * 0.02 - 0.01
        };
        Self {
            hidden,
            fc1_w: (0..hidden * in_dim).map(|_| next()).collect(),
            fc1_b: vec![0.0; hidden],
            fc2_w: (0..hidden).map(|_| next()).collect(),
            fc2_b: vec![-2.5],
            keyword: keyword.into(),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut map = HashMap::new();
        map.insert("phrase.hidden".into(), vec![self.hidden as f32]);
        map.insert("phrase.fc1.weight".into(), self.fc1_w.clone());
        map.insert("phrase.fc1.bias".into(), self.fc1_b.clone());
        map.insert("phrase.fc2.weight".into(), self.fc2_w.clone());
        map.insert("phrase.fc2.bias".into(), self.fc2_b.clone());
        save_f32_map(path, &map)
    }

    pub fn load(path: &Path, keyword: &str) -> Result<Self> {
        let map = load_f32_map(path)?;
        Ok(Self {
            hidden: map["phrase.hidden"][0] as usize,
            fc1_w: map["phrase.fc1.weight"].clone(),
            fc1_b: map["phrase.fc1.bias"].clone(),
            fc2_w: map["phrase.fc2.weight"].clone(),
            fc2_b: map["phrase.fc2.bias"].clone(),
            keyword: keyword.into(),
        })
    }
}

pub fn score_phrase(w: &PhraseWeights, embeds: &[[f32; EMBED_DIM]; EMBED_HISTORY]) -> f32 {
    let mut flat = vec![0.0f32; EMBED_HISTORY * EMBED_DIM];
    for (i, e) in embeds.iter().enumerate() {
        flat[i * EMBED_DIM..(i + 1) * EMBED_DIM].copy_from_slice(e);
    }
    let mut h = vec![0.0f32; w.hidden];
    gemv_bias(
        w.hidden,
        EMBED_HISTORY * EMBED_DIM,
        &w.fc1_w,
        &flat,
        &w.fc1_b,
        &mut h,
    );
    for v in &mut h {
        *v = relu(*v);
    }
    let mut logit = [0.0f32];
    gemv_bias(1, w.hidden, &w.fc2_w, &h, &w.fc2_b, &mut logit);
    sigmoid(logit[0])
}
