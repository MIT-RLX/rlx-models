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

//! DFlash drafter hyper-parameters, read from a `general.architecture =
//! "dflash"` GGUF.

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgufFile, MetaValue};

/// DFlash draft head.
///
/// Eagle-style: it has **no token embedding and no LM head of its own** — it
/// consumes the TARGET model's intermediate residual streams and reuses the
/// target's `lm_head` to score its proposals. `fc` fuses the taps:
/// `[n_taps * hidden, hidden]`.
#[derive(Debug, Clone)]
pub struct DflashConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    /// Tokens proposed per draft step (`dflash.block_size`).
    pub block_size: usize,
    /// Which TARGET layers feed `fc` (`dflash.target_layers`), e.g.
    /// `[2, 14, 26, 38, 50]` for Muse-Glimmer-30B's 52 layers.
    pub target_layers: Vec<usize>,
    /// Sliding-window width; DFlash marks every layer local.
    pub sliding_window: Option<usize>,
    /// Token the noise block is filled with; every slot but the anchor
    /// starts as MASK and is denoised in one pass.
    pub mask_token_id: Option<u32>,
    /// Slot 0 of a block is a prediction slot (`true`) or a bonus anchor
    /// carrying the last committed token (`false`, SpecForge exports).
    /// DFlash2 always conditions its lattice on slot 0 as the anchor.
    pub sample_from_anchor: bool,
    /// The DFlash2 modules. `None` on a v1 checkpoint.
    pub dflash2: Option<Dflash2Config>,
    /// `output_multiplier` on the draft logits.
    pub logit_scale: Option<f32>,
    /// `tanh` softcap applied to the draft logits.
    pub final_logit_softcapping: Option<f32>,
    /// Scale applied to the noise-block token embeddings.
    pub embedding_scale: Option<f32>,
    /// RoPE pairing flavor, which depends on the checkpoint FORMAT, not the
    /// architecture: GGUF converters permute Q/K so the halves interleave
    /// (GPT-J), while a safetensors export keeps the HF split-half layout
    /// (NeoX). Applying the wrong one is not a crash — the drafter still emits
    /// confident, plausible logits, and acceptance quietly collapses.
    pub rope_style: rlx_ir::RopeStyle,
}

/// The two DFlash2 modules, present iff `dflash.selector_top_k > 0`.
///
/// Detection matches upstream: a DFlash2 checkpoint is recognised from its
/// own metadata, never from a user-supplied flag.
#[derive(Debug, Clone, Copy)]
pub struct Dflash2Config {
    /// Taps in the depthwise kernel (`t` in `Σ_t k_t ⊙ x_{i-t}`); 2 in the
    /// released checkpoints.
    pub conv_kernel_size: usize,
    /// Channels sharing one dynamic correction `δ`; 16 in the released
    /// checkpoints.
    pub conv_group_size: usize,
    /// Width of the `A`/`B` selector codebooks; 256 in the released
    /// checkpoints.
    pub selector_rank: usize,
    /// Candidates kept per block position; 16 in the released checkpoints.
    pub selector_top_k: usize,
}

impl Dflash2Config {
    /// Number of channel groups sharing one dynamic correction.
    pub fn n_groups(&self, hidden_size: usize) -> usize {
        hidden_size / self.conv_group_size
    }

    /// Output width of `kernel_projection`: one `δ` per (group, tap, side),
    /// where side 0 is the pre-sublayer conv and side 1 the post-sublayer
    /// one. Both sides come from a single projection of the pre-norm
    /// hidden, so the conv costs one GEMM per sublayer, not two.
    pub fn dynamic_dim(&self, hidden_size: usize) -> usize {
        2 * self.conv_kernel_size * self.n_groups(hidden_size)
    }
}

fn u32_at(raw: &GgufFile, key: &str) -> Option<u32> {
    raw.metadata.get(key).and_then(MetaValue::as_u32)
}

fn f32_at(raw: &GgufFile, key: &str) -> Option<f32> {
    raw.metadata.get(key).and_then(|v| match v {
        MetaValue::F32(x) => Some(*x),
        _ => None,
    })
}

fn bool_at(raw: &GgufFile, key: &str) -> Option<bool> {
    raw.metadata.get(key).and_then(|v| match v {
        MetaValue::Bool(x) => Some(*x),
        _ => None,
    })
}

/// A scale is "absent" when the key is missing *or* zero — upstream writes
/// `0.0` to mean "no scaling", and applying it literally would zero the
/// logits.
fn scale_at(raw: &GgufFile, key: &str) -> Option<f32> {
    f32_at(raw, key).filter(|v| *v != 0.0)
}

impl DflashConfig {
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .unwrap_or_default();
        if arch != "dflash" {
            bail!("DflashConfig::from_gguf expected general.architecture=\"dflash\", got {arch:?}");
        }
        let hidden_size =
            u32_at(raw, "dflash.embedding_length").context("dflash.embedding_length")? as usize;
        let num_attention_heads = u32_at(raw, "dflash.attention.head_count")
            .context("dflash.attention.head_count")? as usize;
        let head_dim = u32_at(raw, "dflash.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads.max(1));

        // `target_layers` is the whole point of the arch: without it there is
        // nothing to fuse, so treat a missing/empty array as a hard error rather
        // than silently drafting from noise.
        let target_layers: Vec<usize> = match raw.metadata.get("dflash.target_layers") {
            Some(MetaValue::Array(a)) => a
                .iter()
                .filter_map(MetaValue::as_u32)
                .map(|v| v as usize)
                .collect(),
            _ => Vec::new(),
        };
        if target_layers.is_empty() {
            bail!("dflash GGUF is missing `dflash.target_layers` — nothing to fuse");
        }

        // DFlash2 is announced by `selector_top_k > 0`; the other three keys
        // are then mandatory, because a half-configured selector would build
        // a graph that runs and drafts noise.
        let dflash2 = match u32_at(raw, "dflash.selector_top_k").unwrap_or(0) {
            0 => None,
            top_k => {
                let need = |k: &str| -> Result<usize> {
                    u32_at(raw, k)
                        .map(|v| v as usize)
                        .filter(|v| *v > 0)
                        .with_context(|| format!("DFlash2 checkpoint is missing `{k}`"))
                };
                let cfg = Dflash2Config {
                    conv_kernel_size: need("dflash.conv_kernel_size")?,
                    conv_group_size: need("dflash.conv_group_size")?,
                    selector_rank: need("dflash.selector_rank")?,
                    selector_top_k: top_k as usize,
                };
                if !hidden_size.is_multiple_of(cfg.conv_group_size) {
                    bail!(
                        "DFlash2 hidden_size {hidden_size} is not divisible by conv_group_size {}",
                        cfg.conv_group_size
                    );
                }
                Some(cfg)
            }
        };

        // Same source order as rlx-llama-base: the tokenizer array when the
        // converter embedded one, else the arch key.
        let vocab_size = raw
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                MetaValue::Array(a) => Some(a.len()),
                _ => None,
            })
            .or_else(|| u32_at(raw, "dflash.vocab_size").map(|v| v as usize))
            .context(
                "dflash GGUF has no vocab size — neither tokenizer.ggml.tokens \
                 nor dflash.vocab_size",
            )?;

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size: u32_at(raw, "dflash.feed_forward_length")
                .context("dflash.feed_forward_length")? as usize,
            num_hidden_layers: u32_at(raw, "dflash.block_count").context("dflash.block_count")?
                as usize,
            num_attention_heads,
            num_key_value_heads: u32_at(raw, "dflash.attention.head_count_kv")
                .context("dflash.attention.head_count_kv")?
                as usize,
            head_dim,
            rms_norm_eps: f32_at(raw, "dflash.attention.layer_norm_rms_epsilon").unwrap_or(1e-5)
                as f64,
            rope_theta: f32_at(raw, "dflash.rope.freq_base").unwrap_or(500_000.0) as f64,
            max_position_embeddings: u32_at(raw, "dflash.context_length").unwrap_or(8192) as usize,
            block_size: u32_at(raw, "dflash.block_size").unwrap_or(16) as usize,
            target_layers,
            sliding_window: u32_at(raw, "dflash.attention.sliding_window").map(|v| v as usize),
            mask_token_id: u32_at(raw, "tokenizer.ggml.mask_token_id"),
            sample_from_anchor: bool_at(raw, "dflash.sample_from_anchor").unwrap_or(true),
            dflash2,
            logit_scale: scale_at(raw, "dflash.logit_scale"),
            final_logit_softcapping: scale_at(raw, "dflash.final_logit_softcapping"),
            embedding_scale: scale_at(raw, "dflash.embedding_scale"),
            rope_style: rlx_ir::RopeStyle::GptJ,
        })
    }

    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_proj_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads.max(1)
    }

    /// Width of the concatenated tap vector `fc` consumes.
    pub fn fused_input_dim(&self) -> usize {
        self.target_layers.len() * self.hidden_size
    }
}

/// Map an HF-checkpoint tensor name to the GGUF-style name the builders use.
///
/// The released DFlash drafters ship as safetensors with transformers naming;
/// the graph builders speak GGUF (`blk.N.attn_q.weight`). Returns `None` for a
/// name with no counterpart, so a caller can report it rather than silently
/// build a graph with a missing weight.
pub fn hf_to_dflash_name(hf: &str) -> Option<String> {
    // Encoder / head-level tensors.
    match hf {
        "fc.weight" => return Some("fc.weight".into()),
        // The drafter's post-`fc` norm. HF calls it `hidden_norm`; GGUF exports
        // it as the encoder output norm, which is what it actually is.
        "hidden_norm.weight" => return Some("enc.output_norm.weight".into()),
        "norm.weight" => return Some("output_norm.weight".into()),
        "d2t" => return Some("d2t".into()),
        _ => {}
    }

    // `layers.N.…` (bare) and `model.layers.N.…` both occur in the wild.
    let rest = hf
        .strip_prefix("model.layers.")
        .or_else(|| hf.strip_prefix("layers."))?;
    let (idx, tail) = rest.split_once('.')?;
    idx.parse::<usize>().ok()?;

    let suffix = match tail {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        // DFlash2 conv / selector, for whenever a safetensors DFlash2 lands.
        "attention_conv.base_kernel" => "attn_conv_base",
        "attention_conv.kernel_projection" => "attn_conv_proj.weight",
        "mlp_conv.base_kernel" => "ffn_conv_base",
        "mlp_conv.kernel_projection" => "ffn_conv_proj.weight",
        _ => return None,
    };
    Some(format!("blk.{idx}.{suffix}"))
}

impl DflashConfig {
    /// Parse a transformers `config.json` from an HF DFlash drafter.
    ///
    /// The released small drafters (`*/qwen3-8b-dflash-*`) ship safetensors, not
    /// GGUF, and put the DFlash-specific fields under `dflash_config`.
    pub fn from_hf_json(v: &serde_json::Value) -> Result<Self> {
        let arch_ok = v["architectures"]
            .as_array()
            .map(|a| {
                a.iter()
                    .any(|s| s.as_str().unwrap_or("").contains("DFlash"))
            })
            .unwrap_or(false);
        if !arch_ok {
            bail!(
                "config.json is not a DFlash drafter (architectures={})",
                v["architectures"]
            );
        }
        let d = &v["dflash_config"];
        let usize_at = |k: &str| v[k].as_u64().map(|x| x as usize);

        let target_layers: Vec<usize> = d["target_layer_ids"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_default();
        if target_layers.is_empty() {
            bail!("DFlash config.json has no `dflash_config.target_layer_ids` — nothing to fuse");
        }

        let hidden_size = usize_at("hidden_size").context("hidden_size")?;
        let num_attention_heads = usize_at("num_attention_heads").context("num_attention_heads")?;

        // DFlash2's extras live under `dflash_config` too; absent on a v1 export.
        let d2_at = |k: &str| d[k].as_u64().map(|x| x as usize).filter(|x| *x > 0);
        let dflash2 = match d2_at("selector_top_k") {
            None => None,
            Some(top_k) => {
                let need = |k: &str| -> Result<usize> {
                    d2_at(k).with_context(|| format!("DFlash2 config.json is missing `{k}`"))
                };
                let cfg = Dflash2Config {
                    conv_kernel_size: need("conv_kernel_size")?,
                    conv_group_size: need("conv_group_size")?,
                    selector_rank: need("selector_rank")?,
                    selector_top_k: top_k,
                };
                if !hidden_size.is_multiple_of(cfg.conv_group_size) {
                    bail!(
                        "DFlash2 hidden_size {hidden_size} is not divisible by conv_group_size {}",
                        cfg.conv_group_size
                    );
                }
                Some(cfg)
            }
        };

        Ok(Self {
            vocab_size: usize_at("vocab_size").context("vocab_size")?,
            hidden_size,
            intermediate_size: usize_at("intermediate_size").context("intermediate_size")?,
            num_hidden_layers: usize_at("num_hidden_layers").context("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: usize_at("num_key_value_heads").context("num_key_value_heads")?,
            head_dim: usize_at("head_dim").unwrap_or(hidden_size / num_attention_heads.max(1)),
            rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            rope_theta: v["rope_theta"].as_f64().unwrap_or(1_000_000.0),
            max_position_embeddings: usize_at("max_position_embeddings").unwrap_or(8192),
            block_size: usize_at("block_size")
                .or_else(|| d["block_size"].as_u64().map(|x| x as usize))
                .unwrap_or(16),
            target_layers,
            // `"sliding_window": null` is the common case and means "no window".
            sliding_window: usize_at("sliding_window").filter(|w| *w > 0),
            mask_token_id: d["mask_token_id"].as_u64().map(|x| x as u32),
            sample_from_anchor: d["sample_from_anchor"].as_bool().unwrap_or(true),
            dflash2,
            logit_scale: v["output_multiplier"]
                .as_f64()
                .map(|x| x as f32)
                .filter(|x| *x != 0.0),
            final_logit_softcapping: v["final_logit_softcapping"]
                .as_f64()
                .map(|x| x as f32)
                .filter(|x| *x != 0.0),
            embedding_scale: v["input_embedding_scale"]
                .as_f64()
                .map(|x| x as f32)
                .filter(|x| *x != 0.0),
            rope_style: rlx_ir::RopeStyle::NeoX,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim `dflash_config` shape from
    /// `jacksonkek/qwen3-8b-dflash-perfectblend-step49000`.
    fn qwen3_8b_drafter_json() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DFlashDraftModel"],
            "block_size": 16,
            "dflash_config": { "mask_token_id": 151669, "target_layer_ids": [1, 9, 17, 25, 33] },
            "head_dim": 128, "hidden_size": 4096, "intermediate_size": 12288,
            "max_position_embeddings": 40960, "num_attention_heads": 32,
            "num_hidden_layers": 5, "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000, "sliding_window": null,
            "vocab_size": 151936
        })
    }

    #[test]
    fn parses_a_real_hf_drafter_config() {
        let cfg = DflashConfig::from_hf_json(&qwen3_8b_drafter_json()).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 5);
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.target_layers, vec![1, 9, 17, 25, 33]);
        assert_eq!(cfg.mask_token_id, Some(151669));
        assert_eq!(cfg.fused_input_dim(), 5 * 4096);
        // `"sliding_window": null` must not become Some(0), which would make
        // every mask empty.
        assert_eq!(cfg.sliding_window, None);
        assert!(cfg.dflash2.is_none(), "this export is DFlash v1");
        // Safetensors keeps HF's split-half layout; only GGUF is permuted.
        assert_eq!(cfg.rope_style, rlx_ir::RopeStyle::NeoX);
    }

    /// Without `target_layer_ids` there is nothing to fuse; a drafter built
    /// anyway would read uninitialised features and propose noise.
    #[test]
    fn hf_config_without_target_layers_is_refused() {
        let mut v = qwen3_8b_drafter_json();
        v["dflash_config"]["target_layer_ids"] = serde_json::json!([]);
        assert!(DflashConfig::from_hf_json(&v).is_err());
    }

    #[test]
    fn hf_names_map_onto_the_builder_keys() {
        assert_eq!(hf_to_dflash_name("fc.weight").as_deref(), Some("fc.weight"));
        assert_eq!(
            hf_to_dflash_name("hidden_norm.weight").as_deref(),
            Some("enc.output_norm.weight")
        );
        assert_eq!(
            hf_to_dflash_name("norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            hf_to_dflash_name("layers.3.self_attn.q_proj.weight").as_deref(),
            Some("blk.3.attn_q.weight")
        );
        // Both prefixes occur in released checkpoints.
        assert_eq!(
            hf_to_dflash_name("model.layers.0.mlp.down_proj.weight").as_deref(),
            Some("blk.0.ffn_down.weight")
        );
        assert_eq!(
            hf_to_dflash_name("layers.2.post_attention_layernorm.weight").as_deref(),
            Some("blk.2.ffn_norm.weight")
        );
        // Unknown names are reported, not silently dropped.
        assert_eq!(hf_to_dflash_name("layers.0.mystery.weight"), None);
        assert_eq!(hf_to_dflash_name("optimizer_state"), None);
    }

    /// Every key the builders ask for must be produced by the mapper, or the
    /// graph build fails late with a missing-weight error instead of here.
    #[test]
    fn mapper_covers_every_key_the_builder_loads() {
        let cfg = DflashConfig::from_hf_json(&qwen3_8b_drafter_json()).unwrap();
        let mut produced: Vec<String> = vec![
            "fc.weight".into(),
            "hidden_norm.weight".into(),
            "norm.weight".into(),
        ]
        .into_iter()
        .filter_map(|k: String| hf_to_dflash_name(&k))
        .collect();
        for i in 0..cfg.num_hidden_layers {
            for tail in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                produced.push(hf_to_dflash_name(&format!("layers.{i}.{tail}")).unwrap());
            }
        }
        for want in [
            "fc.weight",
            "enc.output_norm.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.4.attn_q.weight",
        ] {
            assert!(
                produced.iter().any(|p| p == want),
                "mapper never emits {want}"
            );
        }
    }
}
