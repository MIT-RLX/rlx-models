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

//! Diamond Maps guidance parameters for FLUX.2 sampling.

/// HuggingFace repo for the reference FLUX.1-dev flow-map LoRA (weighted Diamond Maps).
pub const FLOW_MAP_LORA_HF_REPO: &str = "gabeguofanclub/flux-1-dev-flowmap-lsd";

/// Default weight file inside [`FLOW_MAP_LORA_HF_REPO`] (see reference `weighted_diamond_maps`).
pub const FLOW_MAP_LORA_HF_WEIGHT: &str = concat!(
    "01-12-26/runs/res_512_steps_50k_rank_64_lr_1e-4/checkpoint-43000/",
    "pytorch_lora_weights.safetensors"
);

/// Which inference-time reward alignment path to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiamondMethod {
    /// Multi-step GLASS posterior + value gradient (no flow-map weights).
    #[default]
    Glass,
    /// Renoise + flow-map-style x0 lookahead (single-timestep denoiser; no dual-time LoRA).
    Weighted,
    /// Denoiser approximation V_t ≈ r(D_t(x_t)) (fast baseline).
    Dps,
}

impl DiamondMethod {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "glass" => Some(Self::Glass),
            "weighted" | "weighted_diamond" => Some(Self::Weighted),
            "dps" | "flow_map" | "fmtt" => Some(Self::Dps),
            _ => None,
        }
    }
}

/// Inference-time reward alignment settings (no base-model retraining).
#[derive(Debug, Clone)]
pub struct DiamondGuidanceParams {
    pub method: DiamondMethod,
    /// Monte Carlo particles for value / gradient estimation.
    pub mc_samples: usize,
    /// Inner GLASS ODE steps per particle (`Glass` only).
    pub inner_steps: usize,
    /// Last N outer denoising steps that apply reward guidance.
    pub guidance_steps: usize,
    /// Multiplier on reward before softmax.
    pub reward_scale: f32,
    /// Max |b_t| for FLUX guidance coefficient.
    pub max_guidance_b: f32,
    /// SNR factor for weighted renoising time t′ (`Weighted` only).
    pub snr_factor: f32,
    /// Include Gaussian likelihood term in weighted gradient.
    pub include_likelihood: bool,
    /// Include score correction in weighted gradient.
    pub include_score: bool,
    /// Softmax logits use full weighting (likelihood + score + reward).
    pub include_weights: bool,
    /// Temperature on particle logits when `include_weights`.
    pub weight_temperature: f32,
    /// Scale combined guidance vector before Euler step.
    pub gradient_norm_scale: f32,
    /// Use dual-time flow-map x0 for weighted particles (`Weighted` only).
    pub use_flow_map: bool,
    /// Evaluate reward on VAE-decoded RGB when VAE is loaded.
    pub decode_reward: bool,
    /// RNG seed offset for particle noise.
    pub seed: u64,
}

impl Default for DiamondGuidanceParams {
    fn default() -> Self {
        Self {
            method: DiamondMethod::Glass,
            mc_samples: 4,
            inner_steps: 10,
            guidance_steps: 5,
            reward_scale: 1.0,
            max_guidance_b: 20.0,
            snr_factor: 5.0,
            include_likelihood: true,
            include_score: true,
            include_weights: false,
            weight_temperature: 1.0,
            gradient_norm_scale: 1.0,
            use_flow_map: true,
            decode_reward: false,
            seed: 0,
        }
    }
}
