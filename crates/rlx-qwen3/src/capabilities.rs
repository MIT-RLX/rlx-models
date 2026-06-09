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

//! Backend capability checks for Qwen3 graphs.
//!
//! Every standard RLX device (`cpu`, `metal`, `mlx`, `cuda`, `rocm`, `gpu`,
//! `vulkan`) is accepted when the matching `rlx-runtime` feature is enabled at
//! build time. Packed GGUF (`packed_weights`) uses `Op::DequantMatMul` on the
//! selected device (host fallback for some legacy Q4_0/Q8_0 schemes on GPU).

use crate::config::Qwen3Config;
use anyhow::Result;
use rlx_core::validate_standard_device;
pub use rlx_core::{STANDARD_DEVICE_NAMES, STANDARD_DEVICES};
use rlx_runtime::Device;

/// Validate that `device` is in the workspace standard backend set.
pub fn validate_device(cfg: &Qwen3Config, device: Device, packed_weights: bool) -> Result<()> {
    let _ = (cfg, packed_weights);
    validate_standard_device("qwen3", device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::STANDARD_DEVICES;

    fn tiny_cfg() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            qk_norm: true,
            sliding_window: None,
            max_window_layers: usize::MAX,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    #[test]
    fn all_standard_backends_allowed() {
        let cfg = tiny_cfg();
        for dev in STANDARD_DEVICES {
            validate_device(&cfg, *dev, false).unwrap();
            validate_device(&cfg, *dev, true).unwrap();
        }
    }
}
