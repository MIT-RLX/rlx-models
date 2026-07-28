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

//! MiniMax-M3 (MSA — MiniMax Sparse Attention) — the `minimax_m3_vl` /
//! `MiniMaxM3SparseForCausalLM` architecture.
//!
//! A mixed dense/sparse 128-expert MoE decoder on a GQA backbone with per-head
//! Gemma QK-norm, partial NeoX RoPE, SwiGLU-OAI experts, and **MSA** block-sparse
//! attention (a lightning indexer selects, per query, the top-k key blocks the
//! main attention may see). See [`flow::build_m3_text_flow`].

pub mod attention;
pub mod cli;
pub mod config;
pub mod decode;
pub mod flow;
pub mod indexer;
pub mod mlp;
pub mod moe;
pub mod ops;
pub mod preprocess;
pub mod runner;
pub mod vision;
pub mod vl_runner;
pub mod weights;

pub use cli::cli_run;
pub use config::{M3VisionConfig, MiniMaxM3Config, SparseAttnConfig};
pub use flow::{build_m3_text_embeds_flow, build_m3_text_flow};
pub use preprocess::M3ImagePreprocessor;
pub use runner::MiniMaxM3Runner;
pub use vision::{build_m3_projector_flow, build_m3_vision_flow, vision_rope_tables};
pub use vl_runner::{ImageInput, MiniMaxM3VlRunner};

/// Graph-input key for the RoPE cosine table (`[seq, n_rot/2]`).
pub const ROPE_COS: &str = "rope_cos";
/// Graph-input key for the RoPE sine table (`[seq, n_rot/2]`).
pub const ROPE_SIN: &str = "rope_sin";

/// Build the M3 partial-RoPE cos/sin tables `[seq, n_rot/2]` (NeoX / HF layout).
/// `inv_freq[j] = theta^(-2j / n_rot)`, `angle[p,j] = p · inv_freq[j]`.
pub fn rope_tables(seq: usize, n_rot: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = n_rot / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for p in 0..seq {
        for j in 0..half {
            let inv_freq = theta.powf(-(2.0 * j as f64) / n_rot as f64);
            let a = p as f64 * inv_freq;
            cos[p * half + j] = a.cos() as f32;
            sin[p * half + j] = a.sin() as f32;
        }
    }
    (cos, sin)
}

/// Single-position RoPE row `[n_rot/2]` for absolute position `pos` — the decode
/// step's cos/sin tables (`[1, n_rot/2]`).
pub fn rope_row(pos: usize, n_rot: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = n_rot / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for j in 0..half {
        let inv_freq = theta.powf(-(2.0 * j as f64) / n_rot as f64);
        let a = pos as f64 * inv_freq;
        cos[j] = a.cos() as f32;
        sin[j] = a.sin() as f32;
    }
    (cos, sin)
}
