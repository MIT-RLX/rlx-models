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

//! Text generation — slow (AR), fast (MTP), and hybrid modes.

use crate::config::LocateAnythingConfig;
use crate::embed::argmax_token;

/// Inference decoding strategy (HF `generation_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenerationMode {
    /// MTP only — parallel box blocks.
    Fast,
    /// Autoregressive one token at a time.
    Slow,
    /// MTP with AR fallback on uncertain boxes (default).
    #[default]
    Hybrid,
}

impl GenerationMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fast" => Some(Self::Fast),
            "slow" => Some(Self::Slow),
            "hybrid" => Some(Self::Hybrid),
            _ => None,
        }
    }
}

/// Special token ids used during MTP / box decoding.
#[derive(Debug, Clone)]
pub struct TokenIds {
    pub box_start: u32,
    pub box_end: u32,
    pub coord_start: u32,
    pub coord_end: u32,
    pub ref_start: u32,
    pub ref_end: u32,
    pub none_token: u32,
    pub null_token: u32,
    pub switch_token: u32,
    pub text_mask: u32,
    pub im_end: u32,
}

impl TokenIds {
    pub fn from_config(cfg: &LocateAnythingConfig) -> Self {
        Self {
            box_start: cfg.box_start_token_id,
            box_end: cfg.box_end_token_id,
            coord_start: cfg.coord_start_token_id,
            coord_end: cfg.coord_end_token_id,
            ref_start: cfg.ref_start_token_id,
            ref_end: cfg.ref_end_token_id,
            none_token: cfg.none_token_id,
            null_token: cfg.text_config.null_token_id.unwrap_or(152_678),
            switch_token: cfg.text_config.switch_token_id.unwrap_or(152_679),
            text_mask: cfg.text_config.text_mask_token_id.unwrap_or(151_676),
            im_end: cfg.text_config.eos_token_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SampleOpts {
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub max_new_tokens: usize,
    pub mode: GenerationMode,
}

impl Default for SampleOpts {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.1,
            max_new_tokens: 2048,
            mode: GenerationMode::Hybrid,
        }
    }
}

/// Greedy or temperature-scaled sample from a single logits row `[vocab]`.
pub fn sample_token(logits: &[f32], opts: &SampleOpts, history: &[u32]) -> u32 {
    debug_assert!(!logits.is_empty());
    let mut scores: Vec<f32> = logits.to_vec();
    if opts.repetition_penalty != 1.0 {
        for &tok in history {
            let i = tok as usize;
            if i < scores.len() {
                if scores[i] > 0.0 {
                    scores[i] /= opts.repetition_penalty;
                } else {
                    scores[i] *= opts.repetition_penalty;
                }
            }
        }
    }
    if opts.temperature > 0.0 {
        for s in &mut scores {
            *s /= opts.temperature;
        }
        sample_stochastic(&scores, opts.top_p)
    } else {
        argmax_token(&scores)
    }
}

fn sample_stochastic(logits: &[f32], top_p: f32) -> u32 {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0f32; logits.len()];
    let mut sum = 0f32;
    for &i in &idx {
        let p = (logits[i] - max).exp();
        probs[i] = p;
        sum += p;
    }
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    if top_p < 1.0 {
        let mut cum = 0f32;
        for &i in &idx {
            cum += probs[i];
            if cum > top_p {
                for j in idx.iter().position(|&x| x == i).unwrap() + 1..idx.len() {
                    probs[idx[j]] = 0.0;
                }
                break;
            }
        }
    }
    let r: f32 = rand_uniform();
    let mut c = 0f32;
    for (i, &p) in probs.iter().enumerate() {
        c += p;
        if r <= c {
            return i as u32;
        }
    }
    argmax_token(logits)
}

fn rand_uniform() -> f32 {
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    SystemTime::now().hash(&mut h);
    (h.finish() % 10_000) as f32 / 10_000.0
}

/// After sampling an MTP block, decide whether to continue in MTP or switch to AR (hybrid).
pub fn hybrid_continue_mtp(out_type: &str, mode: GenerationMode) -> bool {
    match mode {
        GenerationMode::Fast => true,
        GenerationMode::Slow => false,
        GenerationMode::Hybrid => !matches!(out_type, "error_box"),
    }
}

/// Classify a sampled AR token for hybrid mode switching.
pub fn classify_ar_token(tok: u32, ids: &TokenIds) -> &'static str {
    if tok == ids.im_end {
        "im_end"
    } else if tok == ids.box_end {
        "box_end_ar"
    } else if (ids.coord_start..=ids.coord_end).contains(&tok) || tok == ids.none_token {
        "coord_ar"
    } else {
        "continue_ar"
    }
}
