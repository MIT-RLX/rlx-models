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

//! Checkpoint tensor-name helpers for FireRedAudio safetensors (no `thinker.` prefix).

pub const PREFIX_AUDIO: &str = "audio_encoder.";
pub const PREFIX_BACKBONE: &str = "backbone_llm.";
pub const PREFIX_RED_VAE: &str = "red_vae.";
pub const PREFIX_PATCH_ENCODER: &str = "patch_encoder.";
pub const PREFIX_DIT: &str = "dit.";

pub const KEY_EMBED_TOKENS: &str = "backbone_llm.model.language_model.embed_tokens.weight";
pub const KEY_LM_HEAD: &str = "backbone_llm.lm_head.weight";

/// HF weight-name helpers for the understanding audio encoder.
pub struct AudioWeightPrefix;

impl AudioWeightPrefix {
    pub const CONV1_W: &'static str = "audio_encoder.conv1.weight";
    pub const CONV1_B: &'static str = "audio_encoder.conv1.bias";
    pub const CONV2_W: &'static str = "audio_encoder.conv2.weight";
    pub const CONV2_B: &'static str = "audio_encoder.conv2.bias";

    pub const ADAPTER_CONV3_W: &'static str = "audio_encoder.adapter.conv3.weight";
    pub const ADAPTER_CONV3_B: &'static str = "audio_encoder.adapter.conv3.bias";
    pub const ADAPTER_CONV4_W: &'static str = "audio_encoder.adapter.conv4.weight";
    pub const ADAPTER_CONV4_B: &'static str = "audio_encoder.adapter.conv4.bias";
    pub const ADAPTER_LN_W: &'static str = "audio_encoder.adapter.layer_norm.weight";
    pub const ADAPTER_LN_B: &'static str = "audio_encoder.adapter.layer_norm.bias";
    pub const ADAPTER_LINEAR1_W: &'static str = "audio_encoder.adapter.linear1.weight";
    pub const ADAPTER_LINEAR1_B: &'static str = "audio_encoder.adapter.linear1.bias";
    pub const ADAPTER_LINEAR2_W: &'static str = "audio_encoder.adapter.linear2.weight";
    pub const ADAPTER_LINEAR2_B: &'static str = "audio_encoder.adapter.linear2.bias";

    pub fn audio_layer(i: usize, suffix: &str) -> String {
        format!("audio_encoder.layers.{i}.{suffix}")
    }
}

/// Convenience alias matching the call pattern in the `encoder` module.
pub fn audio_layer(i: usize, suffix: &str) -> String {
    AudioWeightPrefix::audio_layer(i, suffix)
}
