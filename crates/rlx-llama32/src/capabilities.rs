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

//! Backend capability checks for llama32 graphs.
//!
//! All portable GPU backends support the standard Llama decoder IR plus
//! packed GGUF K-quant (`Op::DequantMatMul`).

use crate::config::Llama32Config;
use anyhow::Result;
use rlx_core::validate_standard_device;
pub use rlx_core::{STANDARD_DEVICE_NAMES, STANDARD_DEVICES};
use rlx_runtime::Device;

/// Validate that `device` can run a llama32 graph with the given options.
pub fn validate_device(cfg: &Llama32Config, device: Device, packed_weights: bool) -> Result<()> {
    let _ = (cfg, packed_weights);
    validate_standard_device("llama32", device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Llama32Config;
    use rlx_core::STANDARD_DEVICES;

    fn tiny_cfg() -> Llama32Config {
        Llama32Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: None,
            rope_scaling: None,
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
