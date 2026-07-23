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

//! Backend capability checks for qwen35 graphs.

use crate::config::Qwen35Config;
use anyhow::Result;
use rlx_core::validate_lm_device;
pub use rlx_core::{STANDARD_DEVICE_NAMES, STANDARD_DEVICES};
use rlx_runtime::Device;

/// Validate that `device` is in the workspace standard backend set (CPU, Metal, MLX,
/// CUDA, ROCm, WGPU, Vulkan). Build with `all-backends` on `rlx-qwen35` to link every
/// native runtime backend into the `rlx-qwen35` binary.
pub fn validate_device(cfg: &Qwen35Config, device: Device, packed_weights: bool) -> Result<()> {
    let _ = (cfg, packed_weights);
    validate_lm_device("qwen35", device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Qwen35Config;
    use rlx_core::STANDARD_DEVICES;

    fn tiny_cfg() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            nextn_predict_layers: 0,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            key_length: 4,
            value_length: 4,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_dim_count: 4,
            rope_dim_sections: vec![],
            mrope_interleaved: false,
            rms_norm_offset: false,
            full_attention_interval: 3,
            ssm_conv_kernel: 4,
            ssm_group_count: 2,
            ssm_inner_size: 8,
            ssm_state_size: 4,
            ssm_time_step_rank: 2,
            tie_word_embeddings: true,

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
