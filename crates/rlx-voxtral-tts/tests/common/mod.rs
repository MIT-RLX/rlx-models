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

//! Shared helpers for compiled LM integration tests.

#![allow(dead_code)]

use ndarray::{Array1, Array2};
use rlx_runtime::Device;
use rlx_voxtral_tts::load::WeightSnapshot;
use rlx_voxtral_tts::{CompiledMinistralLm, MinistralLm, VoxtralTtsConfig, VoxtralTtsWeightStore};
use std::path::{Path, PathBuf};

pub const PREFILL_SEQ: usize = 4;
pub const DECODE_STEPS: usize = 2;

pub fn model_dir() -> Option<PathBuf> {
    let env = std::env::var("RLX_VOXTRAL_TTS_DIR").ok().map(PathBuf::from);
    let mut candidates = env.into_iter().chain([
        PathBuf::from(".cache/voxtral/Voxtral-4B-TTS-2603"),
        PathBuf::from("../../.cache/voxtral/Voxtral-4B-TTS-2603"),
    ]);
    candidates.find(|p| p.join("consolidated.safetensors").is_file())
}

pub fn seeded_embeds(seed: u64, seq: usize, hidden: usize) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((seq, hidden));
    let mut state = seed;
    for t in 0..seq {
        for h in 0..hidden {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            out[[t, h]] = ((state >> 32) as f32 / u32::MAX as f32) * 0.02 - 0.01;
        }
    }
    out
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / na.sqrt() / nb.sqrt()) as f32
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

pub fn load_backbone(
    model_dir: &Path,
) -> (VoxtralTtsConfig, VoxtralTtsWeightStore, WeightSnapshot) {
    let cfg = VoxtralTtsConfig::from_model_dir(model_dir).expect("config");
    let store = VoxtralTtsWeightStore::open(model_dir).expect("weights");
    let tensors = store
        .tensor_snapshot_for_backbone()
        .expect("backbone snapshot");
    (cfg, store, tensors)
}

pub fn eager_trace(
    tensors: &WeightSnapshot,
    cfg: &rlx_voxtral_tts::config::TextConfig,
    hidden: usize,
) -> Vec<Array1<f32>> {
    let prefill = seeded_embeds(42, PREFILL_SEQ, hidden);
    let decode: Vec<Array2<f32>> = (0..DECODE_STEPS)
        .map(|i| seeded_embeds(100 + i as u64, 1, hidden))
        .collect();
    let mut lm = MinistralLm::from_tensors(tensors, cfg).expect("eager lm");
    let mut out = Vec::with_capacity(1 + decode.len());
    let h = lm.forward(prefill.view()).expect("eager prefill");
    out.push(lm.last_hidden(&h));
    for step in decode {
        let h = lm.forward(step.view()).expect("eager decode");
        out.push(lm.last_hidden(&h));
    }
    out
}

pub fn compiled_trace(
    store: &VoxtralTtsWeightStore,
    cfg: &rlx_voxtral_tts::config::TextConfig,
    device: Device,
    hidden: usize,
) -> Vec<Array1<f32>> {
    let prefill = seeded_embeds(42, PREFILL_SEQ, hidden);
    let decode: Vec<Array2<f32>> = (0..DECODE_STEPS)
        .map(|i| seeded_embeds(100 + i as u64, 1, hidden))
        .collect();
    let mut lm = CompiledMinistralLm::open(store, cfg, device, None, None).expect("compiled lm");
    let mut out = Vec::with_capacity(1 + decode.len());
    let h = lm.forward(prefill.view()).expect("compiled prefill");
    out.push(lm.last_hidden(&h));
    for step in decode {
        let h = lm.forward(step.view()).expect("compiled decode");
        out.push(lm.last_hidden(&h));
    }
    out
}

pub fn l2(v: &Array1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
