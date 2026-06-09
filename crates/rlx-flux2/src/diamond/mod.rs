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

//! Diamond Maps reward alignment for FLUX.2 (arXiv:2602.05993).
//!
//! Inference-time reward alignment without retraining the base model:
//!
//! - **Glass** — multi-step GLASS posterior (`mc_samples × inner_steps` extra forwards)
//! - **Weighted** — renoising + x0 lookahead (`mc_samples` extra forwards per step)
//! - **Dps** — denoiser baseline `r(D_t(x))` (one reward eval per guided step)
//!
//! Dual-time denoiser averages embed(t) and embed(t′), optionally from separate weight
//! tensors (`time_guidance_embed_target` / diffusers `second_embedder`). Enable with
//! `--dual-time-embedder` or `--lora` (auto). Flow-map LoRA: [`FLOW_MAP_LORA_HF_REPO`].

pub mod decode_reward;
pub mod dps_guidance;
pub mod flow_map;
pub mod glass_sampler;
pub mod guidance_pipeline;
pub mod params;
pub mod weighted_guidance;

pub use decode_reward::{HybridLatentDecodeReward, decoded_blueness_reward, hybrid_reward};
pub use dps_guidance::{apply_dps_guidance_step, dps_reward_grad, flux_x0_estimate};
pub use flow_map::{FlowMapPrediction, flow_map_predict, forward_noise_dual};
pub use glass_sampler::{FluxGlassReference, glass_posterior_sample};
pub use guidance_pipeline::sample_rectified_flow_diamond;
pub use params::{
    DiamondGuidanceParams, DiamondMethod, FLOW_MAP_LORA_HF_REPO, FLOW_MAP_LORA_HF_WEIGHT,
};
pub use weighted_guidance::{apply_weighted_guidance_step, weighted_diamond_grad};
