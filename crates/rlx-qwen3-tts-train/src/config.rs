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

//! JFK / custom-voice LoRA training knobs.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct JfkLoraConfig {
    pub model_dir: PathBuf,
    pub train_jsonl: PathBuf,
    pub out_dir: PathBuf,
    pub device: Option<String>,
    pub speaker: String,
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub rank: usize,
    pub lr: f64,
    pub max_seq: usize,
    pub n_layers: usize,
    pub grad_accum: usize,
    pub max_clips: usize,
    pub cache_path: Option<PathBuf>,
    pub verbose: bool,
}

impl Default for JfkLoraConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base"),
            train_jsonl: PathBuf::from(".cache/qwen3-tts/jfk/train_with_codes.jsonl"),
            out_dir: PathBuf::from(".cache/qwen3-tts/jfk-checkpoint-mlx"),
            device: Some("mlx".into()),
            speaker: "jfk".into(),
            epochs: 3,
            steps_per_epoch: 20,
            rank: 8,
            lr: 1e-4,
            max_seq: 128,
            n_layers: 4,
            grad_accum: 4,
            max_clips: 0,
            cache_path: None,
            verbose: true,
        }
    }
}

pub fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
