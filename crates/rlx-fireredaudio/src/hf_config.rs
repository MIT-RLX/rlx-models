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

//! Load FireRedAudio `config.json` and map the nested backbone to [`Qwen35Config`].

use crate::config::{
    AudioEncoderConfig, BackboneConfig, DitConfig, FireRedAudioConfig, PatchEncoderConfig,
    RedVaeConfig, SpecialTokens,
};
use anyhow::{Context, Result, bail};
use rlx_qwen35::Qwen35Config;
use serde_json::Value;
use std::path::Path;

impl FireRedAudioConfig {
    /// Parse the released HF `FireRedAudio/config.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading FireRedAudio config {path:?}"))?;
        Self::from_json(&raw)
    }

    pub fn from_json(data: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(data).context("parsing FireRedAudio config.json")?;
        Self::from_value(&v)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if !model_type.is_empty() && model_type != "firered_audio" {
            bail!("expected model_type firered_audio, got {model_type:?}");
        }

        let audio_encoder = parse_audio_encoder(v.get("audio_encoder_config"))?;
        let backbone = parse_backbone(v.get("backbone_config"))?;
        let red_vae = parse_red_vae(v.get("red_vae_config"))?;
        let patch_encoder = parse_patch(v.get("patch_encoder_config"))?;
        let dit = parse_dit(v.get("dit_config"))?;
        let tokens = SpecialTokens {
            sosp_idx: u32_field(v, "sosp_idx").unwrap_or(248_077),
            eosp_idx: u32_field(v, "eosp_idx").unwrap_or(248_078),
            audio_special_token: "<|AUDIO|>",
            audio_special_token_id: u32_field(v, "audio_special_token_id").unwrap_or(248_091),
            audio_special_token_no_latent: "<|AUDIO_NO_LATENT|>",
            audio_special_no_latent_id: u32_field(v, "audio_special_no_latent_id")
                .unwrap_or(248_092),
        };

        let cfg = Self {
            backbone,
            audio_encoder,
            red_vae,
            patch_encoder,
            dit,
            tokens,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Qwen3.5 runner config for the nested `backbone_config` (MTP stripped).
    pub fn qwen35_config(&self) -> Qwen35Config {
        backbone_to_qwen35(&self.backbone)
    }
}

fn u_field(obj: &Value, key: &str) -> Option<usize> {
    obj.get(key).and_then(|v| v.as_u64()).map(|n| n as usize)
}

fn f_field(obj: &Value, key: &str) -> Option<f64> {
    obj.get(key).and_then(|v| v.as_f64())
}

fn u32_field(obj: &Value, key: &str) -> Option<u32> {
    obj.get(key).and_then(|v| v.as_u64()).map(|n| n as u32)
}

fn parse_audio_encoder(v: Option<&Value>) -> Result<AudioEncoderConfig> {
    let mut c = AudioEncoderConfig::default();
    let Some(o) = v else {
        return Ok(c);
    };
    if let Some(n) = u_field(o, "d_model") {
        c.d_model = n;
    }
    if let Some(n) = u_field(o, "encoder_layers").or_else(|| u_field(o, "num_hidden_layers")) {
        c.encoder_layers = n;
    }
    if let Some(n) = u_field(o, "encoder_attention_heads") {
        c.encoder_attention_heads = n;
    }
    if let Some(n) = u_field(o, "encoder_ffn_dim") {
        c.encoder_ffn_dim = n;
    }
    if let Some(n) = u_field(o, "num_mel_bins") {
        c.num_mel_bins = n;
    }
    if let Some(n) = u_field(o, "output_dim") {
        c.output_dim = n;
    }
    if let Some(n) = u_field(o, "max_source_positions") {
        c.max_source_positions = n;
    }
    if let Some(n) = u_field(o, "n_window") {
        c.n_window = n;
    }
    Ok(c)
}

fn parse_backbone(v: Option<&Value>) -> Result<BackboneConfig> {
    let mut c = BackboneConfig::default();
    let Some(o) = v else {
        return Ok(c);
    };
    if let Some(n) = u_field(o, "hidden_size") {
        c.hidden_size = n;
    }
    if let Some(n) = u_field(o, "intermediate_size") {
        c.intermediate_size = n;
    }
    let layers = u_field(o, "num_hidden_layers").unwrap_or(c.num_hidden_layers);
    let mtp = u_field(o, "mtp_num_hidden_layers").unwrap_or(0);
    c.num_hidden_layers = layers.saturating_sub(mtp);
    if let Some(n) = u_field(o, "num_attention_heads") {
        c.num_attention_heads = n;
    }
    if let Some(n) = u_field(o, "num_key_value_heads") {
        c.num_key_value_heads = n;
    }
    if let Some(n) = u_field(o, "head_dim") {
        c.head_dim = n;
    }
    if let Some(n) = u_field(o, "vocab_size") {
        c.vocab_size = n;
    }
    if let Some(n) = u_field(o, "max_position_embeddings") {
        c.max_position_embeddings = n;
    }
    if let Some(n) = f_field(o, "rms_norm_eps") {
        c.rms_norm_eps = n as f32;
    }
    if let Some(n) = u_field(o, "full_attention_interval") {
        c.full_attention_interval = n;
    }
    if let Some(n) = u_field(o, "linear_conv_kernel_dim") {
        c.linear_conv_kernel_dim = n;
    }
    if let Some(n) = f_field(o, "partial_rotary_factor") {
        c.partial_rotary_factor = n as f32;
    } else if let Some(rp) = o.get("rope_parameters")
        && let Some(n) = f_field(rp, "partial_rotary_factor")
    {
        c.partial_rotary_factor = n as f32;
    }
    Ok(c)
}

fn parse_red_vae(v: Option<&Value>) -> Result<RedVaeConfig> {
    let mut c = RedVaeConfig::default();
    let Some(o) = v else {
        return Ok(c);
    };
    if let Some(n) = u_field(o, "hidden_size") {
        c.hidden_size = n;
    }
    if let Some(n) = u_field(o, "intermediate_size") {
        c.intermediate_size = n;
    }
    if let Some(n) = u_field(o, "num_hidden_layers") {
        c.num_hidden_layers = n;
    }
    if let Some(n) = u_field(o, "num_attention_heads") {
        c.num_attention_heads = n;
    }
    if let Some(n) = u_field(o, "num_key_value_heads") {
        c.num_key_value_heads = n;
    }
    if let Some(n) = u_field(o, "out_dim") {
        c.out_dim = n;
    }
    if let Some(n) = u_field(o, "audio_sample_rate") {
        c.audio_sample_rate = n;
    }
    if let Some(n) = u_field(o, "audio_patch_size") {
        c.audio_patch_size = n;
    }
    if let Some(n) = u_field(o, "extra_downsample_rate") {
        c.extra_downsample_rate = n;
    }
    if let Some(n) = u_field(o, "sliding_window") {
        c.sliding_window = n;
    }
    Ok(c)
}

fn parse_patch(v: Option<&Value>) -> Result<PatchEncoderConfig> {
    let mut c = PatchEncoderConfig::default();
    let Some(o) = v else {
        return Ok(c);
    };
    if let Some(n) = u_field(o, "depth") {
        c.depth = n;
    }
    if let Some(n) = u_field(o, "hidden_size") {
        c.hidden_size = n;
    }
    if let Some(n) = u_field(o, "num_heads") {
        c.num_heads = n;
    }
    if let Some(n) = u_field(o, "mlp_ratio") {
        c.mlp_ratio = n;
    }
    if let Some(n) = u_field(o, "out_dim") {
        c.out_dim = n;
    }
    if let Some(n) = u_field(o, "patch_size") {
        c.patch_size = n;
    }
    if let Some(n) = u_field(o, "vae_dim") {
        c.vae_dim = n;
    }
    Ok(c)
}

fn parse_dit(v: Option<&Value>) -> Result<DitConfig> {
    let mut c = DitConfig::default();
    let Some(o) = v else {
        return Ok(c);
    };
    if let Some(n) = u_field(o, "backbone_hidden_size") {
        c.backbone_hidden_size = n;
    }
    if let Some(n) = u_field(o, "depth") {
        c.depth = n;
    }
    if let Some(n) = u_field(o, "hidden_size") {
        c.hidden_size = n;
    }
    if let Some(n) = u_field(o, "num_heads") {
        c.num_heads = n;
    }
    if let Some(n) = f_field(o, "mlp_ratio") {
        c.mlp_ratio = n as f32;
    }
    if let Some(n) = u_field(o, "patch_size") {
        c.patch_size = n;
    }
    if let Some(n) = u_field(o, "history_patches") {
        c.history_patches = n;
    }
    if let Some(n) = u_field(o, "vae_channels") {
        c.vae_channels = n;
    }
    if let Some(n) = f_field(o, "train_cfg_rate") {
        c.train_cfg_rate = n as f32;
    }
    Ok(c)
}

fn backbone_to_qwen35(b: &BackboneConfig) -> Qwen35Config {
    let rope_dim = ((b.head_dim as f32) * b.partial_rotary_factor).round() as usize;
    // Released FireRedAudio backbone_config linear_* fields (see HF config.json).
    let linear_num_key_heads = 16usize;
    let linear_key_head_dim = 128usize;
    let linear_num_value_heads = 32usize;
    let linear_value_head_dim = 128usize;
    Qwen35Config {
        vocab_size: b.vocab_size,
        hidden_size: b.hidden_size,
        intermediate_size: b.intermediate_size,
        num_hidden_layers: b.num_hidden_layers,
        nextn_predict_layers: 0,
        num_attention_heads: b.num_attention_heads,
        num_key_value_heads: b.num_key_value_heads,
        key_length: b.head_dim,
        value_length: b.head_dim,
        max_position_embeddings: b.max_position_embeddings,
        rms_norm_eps: b.rms_norm_eps as f64,
        rope_theta: 10_000_000.0,
        rope_dim_count: rope_dim.max(1),
        rope_dim_sections: vec![11, 11, 10],
        mrope_interleaved: true,
        rms_norm_offset: true,
        full_attention_interval: b.full_attention_interval,
        ssm_conv_kernel: b.linear_conv_kernel_dim,
        ssm_group_count: linear_num_key_heads,
        ssm_inner_size: linear_num_value_heads.saturating_mul(linear_value_head_dim),
        ssm_state_size: linear_key_head_dim,
        ssm_time_step_rank: linear_num_value_heads,
        tie_word_embeddings: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// Open HF dir safetensors + strip `backbone_llm.` for [`rlx_qwen35::Qwen35Weights::from_loader`].
pub fn load_qwen35_backbone(
    model_dir: &Path,
    cfg: &FireRedAudioConfig,
) -> Result<(Qwen35Config, rlx_qwen35::Qwen35Weights)> {
    use crate::prefix::PrefixStripLoader;
    use crate::weights::PREFIX_BACKBONE;
    use anyhow::Context as _;
    use rlx_core::{HfTranslatingLoader, SafetensorsMmapLoader};
    use rlx_qwen35::Qwen35Weights;

    let qcfg = cfg.qwen35_config();
    let mmap = SafetensorsMmapLoader::open(model_dir)
        .with_context(|| format!("open FireRedAudio safetensors in {model_dir:?}"))?;
    let stripped = PrefixStripLoader::new(mmap, PREFIX_BACKBONE);
    let mut loader = HfTranslatingLoader::new(stripped);
    let weights = Qwen35Weights::from_loader(&mut loader, &qcfg)
        .context("load FireRedAudio backbone_llm → Qwen35Weights")?;
    Ok((qcfg, weights))
}
