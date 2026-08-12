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

//! Pluggable weight loader trait (plan #56).
//!
//! Borrowed from MAX's `max/python/max/graph/weights/` layout:
//! `load_safetensors.py`, `load_gguf.py`, plus a `load.py` dispatcher
//! that detects format from the file extension.
//!
//! Today the only impl is safetensors (via the existing
//! [`WeightMap::from_file`]). Adding a new format = one new struct
//! that implements [`WeightLoader`] + an extension match in
//! [`load_from_path`]. The model graph builders take any
//! `&mut dyn WeightLoader` so they don't care which format the
//! weights came from.

use anyhow::{Context, Result, anyhow};
use rlx_gguf::MetaValue;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// Walk the GGUF metadata for `{arch}.block_count -
/// {arch}.nextn_predict_layers`. Returns `Some(main_layer_count)`
/// when the file has explicit MTP metadata, else `None`.
fn compute_mtp_layer_threshold(file: &rlx_gguf::GgufFile) -> Option<u32> {
    let arch = file
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)?;
    let block_count = file
        .metadata
        .get(&format!("{arch}.block_count"))
        .and_then(MetaValue::as_u32)?;
    let nextn = file
        .metadata
        .get(&format!("{arch}.nextn_predict_layers"))
        .and_then(MetaValue::as_u32)?;
    if nextn == 0 {
        return None;
    }
    Some(block_count.saturating_sub(nextn))
}

use crate::gguf_resolve::resolve_gguf_tensor_name;
use crate::gguf_support::gguf_architecture_str;
use crate::weight_map::PackedWeightTensor;
use crate::weight_map::WeightMap;
use rlx_ir::quant::QuantScheme;

/// Translate a Hugging Face / safetensors-convention tensor name to
/// the GGUF / llama.cpp convention. Returns `None` when no mapping
/// exists (caller should treat the name as already-GGUF or as an
/// architecture-specific weight that this mapper doesn't know about,
/// e.g. MTP heads — see [`is_mtp_weight`]).
///
/// The mapping mirrors the table baked into `llama.cpp`'s
/// `gguf-py/gguf/tensor_mapping.py` for the LLaMA-family architectures
/// (Qwen3 reuses it). When adding new architectures, prefer extending
/// this function over forking it.
pub fn hf_to_gguf_name(hf: &str) -> Option<String> {
    // Top-level (non-layer) tensors.
    match hf {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }
    // Layer tensors: `model.layers.{i}.<tail>.weight` → `blk.{i}.<gguf-tail>.weight`.
    let rest = hf.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let (idx_str, tail_with_dot) = rest.split_at(dot);
    let tail = &tail_with_dot[1..]; // skip the '.'
    let idx: usize = idx_str.parse().ok()?;
    let gguf_tail = match tail {
        "input_layernorm.weight" => "attn_norm.weight",
        // Llama-style: `post_attention_layernorm` ≡ pre-FFN norm.
        // Gemma 2 disagrees (it has 4 distinct norms) and uses a
        // dedicated `Gemma2GgufResolver` to override this entry. Don't
        // add Gemma 2 names here — they'd collide with Llama callers.
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{idx}.{gguf_tail}"))
}

/// Inverse of [`hf_to_gguf_name`] — translate a GGUF / llama.cpp
/// tensor name back to the safetensors / HuggingFace convention. Used
/// by drain-style loaders (e.g. `Qwen3Generator::from_loader`) that
/// cache weights by name and need the cache key to match what the
/// builder will ask for.
pub fn gguf_to_hf_name(gguf: &str) -> Option<String> {
    match gguf {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = gguf.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let (idx_str, tail_with_dot) = rest.split_at(dot);
    let tail = &tail_with_dot[1..];
    let idx: usize = idx_str.parse().ok()?;
    let hf_tail = match tail {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{idx}.{hf_tail}"))
}

/// GGUF → HuggingFace name for Qwen3.5 / Qwen3.5-VL / Fara multimodal
/// safetensors (`model.language_model.*` + gated DeltaNet `linear_attn.*`).
///
/// Tried by [`HfTranslatingLoader`] after the Llama-family
/// [`gguf_to_hf_name`] mapping misses.
pub fn gguf_to_hf_qwen35_name(gguf: &str) -> Option<String> {
    match gguf {
        "token_embd.weight" => {
            return Some("model.language_model.embed_tokens.weight".into());
        }
        "output_norm.weight" => return Some("model.language_model.norm.weight".into()),
        // Tied checkpoints omit lm_head; untied 9B-class may ship it here.
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = gguf.strip_prefix("blk.")?;
    let (idx_str, tail) = rest.split_once('.')?;
    let idx: usize = idx_str.parse().ok()?;
    let hf_tail = match tail {
        "attn_norm.weight" => "input_layernorm.weight",
        // Qwen3.5 GGUF uses `post_attention_norm` for the pre-FFN RMSNorm
        // on both linear and full-attn trunk layers.
        "post_attention_norm.weight" | "ffn_norm.weight" => "post_attention_layernorm.weight",
        // Full attention.
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        // Gated DeltaNet / linear attention.
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_a" => "linear_attn.A_log",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        // Dense MLP.
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.language_model.layers.{idx}.{hf_tail}"))
}

/// Candidate HF names for a GGUF-style key (Llama flat + Qwen3.5 nested).
pub fn gguf_to_hf_name_candidates(gguf: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(3);
    if let Some(h) = gguf_to_hf_name(gguf) {
        out.push(h);
    }
    if let Some(h) = gguf_to_hf_qwen35_name(gguf) {
        if !out.iter().any(|x| x == &h) {
            out.push(h);
        }
    }
    // Untied multimodal checkpoints sometimes nest lm_head under language_model.
    if gguf == "output.weight" {
        let nested = "model.language_model.embed_tokens.weight".to_string();
        let alt = "model.language_model.lm_head.weight".to_string();
        if !out.iter().any(|x| x == &alt) {
            out.push(alt);
        }
        let _ = nested;
    }
    out
}

/// Arch-aware variant of [`gguf_to_hf_name`]. Falls back to the
/// generic mapping when `arch` doesn't carry overrides; for arches
/// whose 1↔1 alias disagrees with the Llama convention (Gemma 2/3/4:
/// `ffn_norm` is the pre-FFN norm, not the post-attention norm), it
/// returns the arch-correct HF name instead. Used by drain-style
/// `from_loader` paths to compute cache keys that match what the
/// builder will ask for.
pub fn gguf_to_hf_name_for_arch(gguf: &str, arch: &str) -> Option<String> {
    if matches!(
        arch,
        "gemma2" | "gemma3" | "gemma3n" | "gemma4" | "gemma4moe"
    ) {
        match gguf {
            "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
            "output_norm.weight" => return Some("model.norm.weight".into()),
            "output.weight" => return Some("lm_head.weight".into()),
            _ => {}
        }
        let rest = gguf.strip_prefix("blk.")?;
        let dot = rest.find('.')?;
        let (idx_str, tail_with_dot) = rest.split_at(dot);
        let tail = &tail_with_dot[1..];
        let idx: usize = idx_str.parse().ok()?;
        let hf_tail = match tail {
            "attn_norm.weight" => "input_layernorm.weight",
            "post_attention_norm.weight" => "post_attention_layernorm.weight",
            "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
            "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            // Gemma 4 per-head Q/K RMSNorms — applied between
            // q_proj/k_proj and RoPE. Without this mapping the drain
            // stored the weights under their GGUF names and the
            // packed builder's `model.layers.N.self_attn.{q,k}_norm`
            // lookup missed, panicking with "packed cache miss …
            // self_attn.q_norm.weight" once the builder started
            // requesting them (task #50).
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            // Gemma 4 per-layer output scalar (shape [1]) — multiplied
            // with the layer output before the residual add.
            "layer_output_scale.weight" => "self_attn.output_scale.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            _ => return None,
        };
        return Some(format!("model.layers.{idx}.{hf_tail}"));
    }
    if arch == "muse-glimmer" {
        match gguf {
            "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
            "output_norm.weight" => return Some("model.norm.weight".into()),
            "output.weight" => return Some("lm_head.weight".into()),
            _ => {}
        }
        let rest = gguf.strip_prefix("blk.")?;
        let dot = rest.find('.')?;
        let (idx_str, tail_with_dot) = rest.split_at(dot);
        let tail = &tail_with_dot[1..];
        let idx: usize = idx_str.parse().ok()?;
        // Gemma-style four-norm naming, but the gains are consumed verbatim —
        // Muse Glimmer's converter already folded `weight + 1` in, so this arch
        // must NOT go through `is_gemma_norm_weight`'s `-1` delta.
        let hf_tail = match tail {
            "attn_norm.weight" => "input_layernorm.weight",
            "post_attention_norm.weight" => "post_attention_layernorm.weight",
            "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
            "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            // Attention OUTPUT gate — distinct from the MLP's `ffn_gate`.
            "attn_gate.weight" => "self_attn.gate_proj.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            _ => return None,
        };
        return Some(format!("model.layers.{idx}.{hf_tail}"));
    }
    if matches!(arch, "phi3" | "phi4") {
        match gguf {
            "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
            "output_norm.weight" => return Some("model.norm.weight".into()),
            "output.weight" => return Some("lm_head.weight".into()),
            _ => {}
        }
        let rest = gguf.strip_prefix("blk.")?;
        let dot = rest.find('.')?;
        let (idx_str, tail_with_dot) = rest.split_at(dot);
        let tail = &tail_with_dot[1..];
        let idx: usize = idx_str.parse().ok()?;
        let hf_tail = match tail {
            "attn_norm.weight" => "input_layernorm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "attn_qkv.weight" => "self_attn.qkv.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "ffn_up.weight" => "mlp.gate_up.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            _ => return None,
        };
        return Some(format!("model.layers.{idx}.{hf_tail}"));
    }
    gguf_to_hf_name(gguf)
}

/// Arch-aware inverse of [`gguf_to_hf_name_for_arch`]. Used by
/// [`crate::gguf_resolve::Gemma2GgufResolver`] when builders request
/// HF tensor names against a Gemma 2/3/4 GGUF file.
pub fn hf_to_gguf_name_for_arch(hf: &str, arch: &str) -> Option<String> {
    if matches!(
        arch,
        "gemma2" | "gemma3" | "gemma3n" | "gemma4" | "gemma4moe"
    ) {
        match hf {
            "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
            "model.norm.weight" => return Some("output_norm.weight".into()),
            "lm_head.weight" => return Some("output.weight".into()),
            _ => {}
        }
        let rest = hf.strip_prefix("model.layers.")?;
        let dot = rest.find('.')?;
        let (idx_str, tail_with_dot) = rest.split_at(dot);
        let tail = &tail_with_dot[1..];
        let idx: usize = idx_str.parse().ok()?;
        let gguf_tail = match tail {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "post_attention_norm.weight",
            "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
            "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "self_attn.output_scale.weight" => "layer_output_scale.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            _ => return hf_to_gguf_name(hf),
        };
        return Some(format!("blk.{idx}.{gguf_tail}"));
    }
    if arch == "muse-glimmer" {
        match hf {
            "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
            "model.norm.weight" => return Some("output_norm.weight".into()),
            "lm_head.weight" => return Some("output.weight".into()),
            _ => {}
        }
        let rest = hf.strip_prefix("model.layers.")?;
        let dot = rest.find('.')?;
        let (idx_str, tail_with_dot) = rest.split_at(dot);
        let tail = &tail_with_dot[1..];
        let idx: usize = idx_str.parse().ok()?;
        let gguf_tail = match tail {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "post_attention_norm.weight",
            "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
            "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "self_attn.gate_proj.weight" => "attn_gate.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            _ => return hf_to_gguf_name(hf),
        };
        return Some(format!("blk.{idx}.{gguf_tail}"));
    }
    if matches!(arch, "phi3" | "phi4") {
        match hf {
            "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
            "model.norm.weight" => return Some("output_norm.weight".into()),
            "lm_head.weight" => return Some("output.weight".into()),
            _ => {}
        }
        let rest = hf.strip_prefix("model.layers.")?;
        let dot = rest.find('.')?;
        let (idx_str, tail_with_dot) = rest.split_at(dot);
        let tail = &tail_with_dot[1..];
        let idx: usize = idx_str.parse().ok()?;
        let gguf_tail = match tail {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "self_attn.qkv.weight" => "attn_qkv.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "mlp.gate_up.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            _ => return hf_to_gguf_name(hf),
        };
        return Some(format!("blk.{idx}.{gguf_tail}"));
    }
    hf_to_gguf_name(hf)
}

/// Match GGUF tensor names that hold a Gemma RMSNorm gain. Covers all
/// four per-layer norms in the V2/V3/V4 sandwich (attn / post_attention
/// / ffn / post_ffw), per-head Q/K norms, plus the final `output_norm`,
/// in both GGUF-native and HF spellings — drain order is undefined, so
/// we may see either convention at the call site.
///
/// `convert_hf_to_gguf` / `gemma.py` bakes `(1 + gamma)` into every
/// `*norm.weight` (including `attn_q_norm` / `attn_k_norm`). Subtract
/// 1 on `take` so `gemma_rms`'s `1 + w` matches llama.cpp for all norms.
fn is_gemma_norm_weight(name: &str) -> bool {
    if name == "output_norm.weight" || name == "model.norm.weight" {
        return true;
    }
    if let Some(rest) = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.').map(|x| x.1))
    {
        return matches!(
            rest,
            "attn_norm.weight"
                | "post_attention_norm.weight"
                | "ffn_norm.weight"
                | "post_ffw_norm.weight"
                | "attn_q_norm.weight"
                | "attn_k_norm.weight"
        );
    }
    if let Some(rest) = name
        .strip_prefix("model.layers.")
        .and_then(|r| r.split_once('.').map(|x| x.1))
    {
        return matches!(
            rest,
            "input_layernorm.weight"
                | "post_attention_layernorm.weight"
                | "pre_feedforward_layernorm.weight"
                | "post_feedforward_layernorm.weight"
                | "self_attn.q_norm.weight"
                | "self_attn.k_norm.weight"
        );
    }
    false
}

/// True if the GGUF tensor name **looks like** a Multi-Token
/// Prediction head by its name alone — substring match on
/// `mtp_*` / `*.mtp` / `output_mtp_*` style. Covers MTP variants
/// that name their heads explicitly.
///
/// **NOT enough on its own** for the most common unsloth /
/// DeepSeek-V3 convention, which encodes MTP heads as *extra
/// `blk.N` layers* with N beyond the main `block_count`. For that
/// case use [`GgufLoader::mtp_layer_threshold`] / the loader's
/// `is_mtp_tensor` instance method — they read
/// `{arch}.nextn_predict_layers` from the GGUF metadata and treat
/// trailing `blk.*` indices accordingly.
pub fn is_mtp_weight(name: &str) -> bool {
    name.contains("mtp_") || name.contains(".mtp") || name.starts_with("mtp")
}

/// True when the tensor is already host float storage (not a block quant).
///
/// Copying these to `Vec<f32>` is not a Q4→F32 *expand* — used by Laguna's
/// packed path for norms / biases while forbidding quant dequant.
pub fn is_native_float_ggml(dtype: rlx_gguf::GgmlType) -> bool {
    use rlx_gguf::GgmlType;
    matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// Map GGML storage type to RLX packed matmul scheme (K-quants only).
pub fn ggml_type_to_quant_scheme(dtype: rlx_gguf::GgmlType) -> Option<QuantScheme> {
    use rlx_gguf::GgmlType;
    match dtype {
        GgmlType::Q2K => Some(QuantScheme::GgufQ2K),
        GgmlType::Q3K => Some(QuantScheme::GgufQ3K),
        GgmlType::Q4K => Some(QuantScheme::GgufQ4K),
        GgmlType::Q5K => Some(QuantScheme::GgufQ5K),
        GgmlType::Q6K => Some(QuantScheme::GgufQ6K),
        GgmlType::Q8K => Some(QuantScheme::GgufQ8K),
        GgmlType::Q4_0 => Some(QuantScheme::GgufQ4_0),
        GgmlType::Q5_0 => Some(QuantScheme::GgufQ5_0),
        GgmlType::Q8_0 => Some(QuantScheme::GgufQ8_0),
        // PrismML Bonsai 1-bit / Ternary 2-bit — keep packed so the
        // 27B never expands to ~108 GB of F32 weights.
        GgmlType::Q1_0 => Some(QuantScheme::GgufQ1_0),
        GgmlType::Q2_0 => Some(QuantScheme::GgufQ2_0),
        // Fermion five-value ternary (Neutrino-8B) — keep packed so the 8B
        // stays ~4 GB instead of expanding to ~32 GB of F32 weights.
        GgmlType::FV5 => Some(QuantScheme::GgufFV5),
        GgmlType::FV5B => Some(QuantScheme::GgufFV5B),
        // IQ family. Everything downstream already supported these — the IQ
        // variants exist in `QuantScheme`, `rlx_gguf::iq_dequant` has every
        // kernel, rlx-cpu's `dequant_block` has all 9 arms, Metal ships
        // `iq{1_m,1_s,2_s,2_xs,2_xxs,3_s,3_xxs,4_nl}_mv_f32`, and the GPU
        // dequant scheme-id table reserves ids 6/7/12-18 for them. Only THIS
        // mapping was missing, so `packed_meta` returned None and every IQ GGUF
        // silently fell back to F32 expansion — 111 GB for a 30B, i.e. an
        // instant OOM. One arm each is all that was needed.
        GgmlType::IQ4NL => Some(QuantScheme::GgufIQ4NL),
        GgmlType::IQ4XS => Some(QuantScheme::GgufIQ4XS),
        GgmlType::IQ2XXS => Some(QuantScheme::GgufIQ2XXS),
        GgmlType::IQ2XS => Some(QuantScheme::GgufIQ2XS),
        GgmlType::IQ2S => Some(QuantScheme::GgufIQ2S),
        GgmlType::IQ3XXS => Some(QuantScheme::GgufIQ3XXS),
        GgmlType::IQ3S => Some(QuantScheme::GgufIQ3S),
        GgmlType::IQ1S => Some(QuantScheme::GgufIQ1S),
        GgmlType::IQ1M => Some(QuantScheme::GgufIQ1M),
        _ => None,
    }
}

/// Whether [`QuantScheme`] may stay packed in `Op::DequantMatMul` graphs.
///
/// `GgufQ6K` requires `rlx-gguf` ≥ 0.2.1 with signed scale bytes in
/// [`rlx_gguf::dequant_q6_k_block`] (crates.io 0.2.1 used `as f32` and
/// skewed v_proj / down_proj). When the block path disagrees with
/// [`rlx_gguf::dequant_q6_k`], callers fall back to F32
/// [`WeightLoader::take_transposed`].
pub fn dequant_matmul_supported(scheme: QuantScheme) -> bool {
    match scheme {
        QuantScheme::GgufQ6K => q6k_dequant_matmul_supported(),
        _ => true,
    }
}

fn q6k_dequant_matmul_supported() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(probe_q6k_block_dequant)
}

/// Synthetic Q6_K block: d=1, scale byte 0xFF (i8 −1), q=1 at slot 0.
fn probe_q6k_block_dequant() -> bool {
    use rlx_gguf::{QK_K, dequant_q6_k, dequant_q6_k_block};
    const BLK: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
    let mut block = [0u8; BLK];
    let sc_off = QK_K / 2 + QK_K / 4;
    block[sc_off] = 0xFF;
    block[0] = 0x21;
    block[QK_K / 2] = 0x08;
    block[BLK - 2..].copy_from_slice(&half::f16::ONE.to_le_bytes());

    let mut out_block = [0f32; QK_K];
    dequant_q6_k_block(&block, &mut out_block);
    let full = match dequant_q6_k(&block, QK_K) {
        Ok(v) => v,
        Err(_) => return false,
    };
    (out_block[0] - full[0]).abs() < 1e-4
}

/// Structure-only metadata for an MLX packed Linear (see
/// [`WeightLoader::packed_mlx_meta`]): everything needed to size the graph's
/// params EXCEPT the large `w_q` codes, which are deferred to per-shard load.
#[derive(Clone, Debug)]
pub struct PackedMlxMeta {
    /// Byte length of the deferred `w_q` codes (sizes the U8 param).
    pub w_q_len: usize,
    /// Small per-group scales (affine: f32 LE; mxfp: u8) — kept, not deferred.
    pub scales: Vec<u8>,
    /// Small per-group biases (affine: f32 LE; mxfp: dummy) — kept.
    pub biases: Vec<u8>,
    pub scheme: rlx_ir::quant::QuantScheme,
    /// Logical dense output shape `[out_features, in_features]`.
    pub out_shape: Vec<usize>,
    pub n_groups: usize,
}

/// Common interface every weight format must satisfy. Mirrors the
/// existing `WeightMap` API so the safetensors impl is a one-line
/// adapter.
///
/// Register additional on-disk formats with [`crate::weight_registry::register_weight_format`].
pub trait WeightLoader: Send {
    /// Format id (`safetensors`, `gguf`, or a custom registration).
    fn format_id(&self) -> &'static str {
        "unknown"
    }
    /// Read a tensor's raw bytes into `buf` instead of borrowing them.
    ///
    /// Borrowed reads come out of an mmap, which makes every touched page
    /// resident and leaves a full second copy of the checkpoint behind the
    /// backend arena. On macOS that memory is unreclaimable short of `munmap`.
    /// Streaming through one reused buffer caps the resident cost at the
    /// largest single tensor. Returns `false` when this loader has no streaming
    /// backing, in which case the caller should fall back to
    /// [`Self::tensor_bytes_borrowed`].
    fn read_tensor_bytes_into(&self, _key: &str, _buf: &mut Vec<u8>) -> Result<bool> {
        Ok(false)
    }
    /// Release any page cache backing this loader's file mapping.
    ///
    /// A borrowed packed upload touches every byte of the checkpoint, so once
    /// those bytes are copied into a backend arena the page cache holds a
    /// second full-size copy of the model. Callers invoke this right after
    /// upload; the loader stays usable and later reads simply re-fault from
    /// disk. Default is a no-op for loaders that own their bytes.
    fn release_mapped_pages(&self) {}
    /// Number of distinct weights in the file.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Take the named tensor as `(f32_data, shape)`. Removes from the
    /// loader so callers can detect "weights I never used."
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)>;
    /// Same as `take` but transposed (last two dims swapped). Most
    /// safetensors weights are stored row-major-of-PyTorch
    /// convention, which RLX's IR consumes column-major; this helper
    /// is the convention-bridge.
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)>;
    /// Take packed K-quant bytes when supported; default returns `None`.
    fn take_packed(&mut self, key: &str) -> Result<Option<crate::weight_map::PackedWeightTensor>> {
        let _ = key;
        Ok(None)
    }
    /// Take a quantized Linear as separate packed MLX arrays (weight codes +
    /// f32 scales + f32 biases) for a 4-input `Op::DequantMatMul`. Default
    /// `None` — only MLX-native loaders implement it. Callers try this
    /// *before* [`take_packed`](Self::take_packed) (single-blob GGUF).
    fn take_packed_mlx(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
        let _ = key;
        Ok(None)
    }
    /// **Zero-copy** borrow of a packed mxfp expert's mmap bytes (codes + e8m0
    /// scales), no owned `MlxPackedLinear` copy. Default: unsupported (`None`) → the
    /// caller falls back to [`Self::take_packed_mlx`]. Only mmap loaders that back
    /// experts by direct shard slices override this. Used by the paged-MoE hot loop
    /// to write codes straight from mmap → device.
    fn borrow_packed_mlx(&self, key: &str) -> Option<rlx_mlx_io::PackedMlxBorrow<'_>> {
        let _ = key;
        None
    }
    /// Drop a [`Self::borrow_packed_mlx`] tensor's mmap pages once copied elsewhere
    /// (bounds the page cache; pairs with the zero-copy borrow). Default no-op.
    fn dontneed_packed_mlx(&self, key: &str) {
        let _ = key;
    }
    /// `MADV_WILLNEED`-prefetch a batch of tensors so a subsequent serial gather
    /// hits warm page cache instead of latency-bound per-tensor mmap faults — the
    /// key to saturating a fast (Thunderbolt/NVMe) link on the scattered per-expert
    /// MoE reads. Default no-op; mmap loaders override it.
    fn prewarm(&self, keys: &[&str]) {
        let _ = keys;
    }
    /// Borrow packed bytes without marking taken (GGUF mmap path).
    fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        let _ = key;
        None
    }
    /// Non-destructive K-quant metadata: `(scheme, shape)` for a packed tensor,
    /// or `None` for non-K-quant / absent tensors. Unlike [`take_packed`](Self::take_packed)
    /// this copies **no bytes** and does **not** mark the tensor taken, so a
    /// bucketed builder can re-read it on every graph rebuild. Bytes are
    /// uploaded separately, zero-copy, via
    /// [`tensor_bytes_borrowed`](Self::tensor_bytes_borrowed) after compile —
    /// which keeps the quantized model from being duplicated in RSS. Default
    /// `None` (loaders without a zero-copy packed path fall back to `take_packed`).
    fn packed_meta(&self, key: &str) -> Option<(rlx_ir::quant::QuantScheme, Vec<usize>)> {
        let _ = key;
        None
    }
    /// **Structure-only** metadata for an MLX packed Linear: the byte length of
    /// the (large) `w_q` codes plus the small scales/biases + scheme/shape, so a
    /// builder can size the U8 codes param WITHOUT retaining the codes
    /// themselves. Default `None` — normal loaders load packed weights whole via
    /// [`take_packed_mlx`](Self::take_packed_mlx); a structure-only wrapper
    /// returns `Some` (dropping the codes) so a coordinator can build a graph
    /// larger than its RAM. The worker later re-fetches its shard's codes.
    fn packed_mlx_meta(&mut self, key: &str) -> Result<Option<PackedMlxMeta>> {
        let _ = key;
        Ok(None)
    }
    /// Names that haven't been taken yet — useful for "did the
    /// model use every weight?" hygiene checks.
    fn remaining_keys(&self) -> Vec<String>;
    /// Architecture name from the underlying file (`general.architecture`
    /// for GGUF, `None` for safetensors). Drain-style consumers use this
    /// to pick an arch-specific reverse name mapping when the canonical
    /// HF name depends on the model family (e.g. Gemma 2's 4 norms per
    /// layer don't share the Llama 2-norm reverse alias).
    fn arch_hint(&self) -> Option<&str> {
        None
    }
}

impl WeightLoader for WeightMap {
    fn format_id(&self) -> &'static str {
        "safetensors"
    }
    fn len(&self) -> usize {
        Self::len(self)
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        Self::take(self, key)
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        Self::take_transposed(self, key)
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.keys().map(|s| s.to_string()).collect()
    }
}

/// Forward every method to the boxed inner so a `Box<dyn WeightLoader>` is
/// itself a [`WeightLoader`]. Lets a callsite pick a loader at runtime (e.g.
/// mlx-affine vs plain safetensors) and still feed it to generic-over-`L`
/// consumers like [`HfTranslatingLoader`]. All defaulted methods are forwarded
/// too — otherwise the box would hide the inner's `take_packed`/`take_packed_mlx`.
impl WeightLoader for Box<dyn WeightLoader> {
    fn format_id(&self) -> &'static str {
        (**self).format_id()
    }
    fn len(&self) -> usize {
        (**self).len()
    }
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        (**self).take(key)
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        (**self).take_transposed(key)
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<crate::weight_map::PackedWeightTensor>> {
        (**self).take_packed(key)
    }
    fn take_packed_mlx(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
        (**self).take_packed_mlx(key)
    }
    fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        (**self).tensor_bytes_borrowed(key)
    }
    fn read_tensor_bytes_into(&self, key: &str, buf: &mut Vec<u8>) -> Result<bool> {
        (**self).read_tensor_bytes_into(key, buf)
    }
    fn packed_meta(&self, key: &str) -> Option<(rlx_ir::quant::QuantScheme, Vec<usize>)> {
        (**self).packed_meta(key)
    }
    fn remaining_keys(&self) -> Vec<String> {
        (**self).remaining_keys()
    }
    fn arch_hint(&self) -> Option<&str> {
        (**self).arch_hint()
    }
}

/// F32 tensor stored in a shared host cache (`Arc` avoids cloning multi-GB
/// weights on every graph build).
pub type ArcF32Tensor = (Arc<Vec<f32>>, Vec<usize>);

/// Cache-blocked row-major 2-D transpose. `src` is `[rows, cols]` row-major
/// (element `(i, j)` at `i*cols + j`); returns `[cols, rows]` row-major
/// (`(i, j)` moves to `(j, i)`).
///
/// Loaders bridge PyTorch's row-major linear weights to RLX's column-major IR,
/// so *every* 2-D weight is transposed on *every* graph build. The naive
/// `dst[j*rows + i] = src[i*cols + j]` loop writes down a fresh column each
/// step, evicting a cache line per element — pathological for multi-GB
/// matrices. Iterating in tiles keeps a small block of the strided destination
/// axis resident, cutting cache/TLB misses by roughly the tile size. Output is
/// byte-identical to the naive loop (see `transpose_matches_naive`).
pub(crate) fn transpose_2d_rowmajor(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols, "transpose shape mismatch");
    let mut dst = vec![0f32; src.len()];
    // ~64 f32 ≈ one cache line per axis; a 64×64 tile is 16 KiB, comfortably
    // inside L1/L2 while amortizing the strided write stride over the block.
    const BLK: usize = 64;
    let mut i0 = 0;
    while i0 < rows {
        let i1 = (i0 + BLK).min(rows);
        let mut j0 = 0;
        while j0 < cols {
            let j1 = (j0 + BLK).min(cols);
            for i in i0..i1 {
                let row = &src[i * cols..i * cols + cols];
                for j in j0..j1 {
                    dst[j * rows + i] = row[j];
                }
            }
            j0 = j1;
        }
        i0 = i1;
    }
    dst
}

/// [`WeightLoader`] view over a frozen `HashMap` of [`ArcF32Tensor`].
/// Graph builders [`take`](WeightLoader::take) tensors out of this view
/// (one `Vec` copy per weight into compile params) without cloning the
/// entire cache first.
pub struct ArcCacheLoader<'a> {
    cache: &'a HashMap<String, ArcF32Tensor>,
}

impl<'a> ArcCacheLoader<'a> {
    pub fn new(cache: &'a HashMap<String, ArcF32Tensor>) -> Self {
        Self { cache }
    }
}

impl WeightLoader for ArcCacheLoader<'_> {
    fn format_id(&self) -> &'static str {
        "arc-cache"
    }
    fn len(&self) -> usize {
        self.cache.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .cache
            .get(key)
            .ok_or_else(|| anyhow!("weight not found in arc cache: {key}"))?;
        Ok((data.as_ref().clone(), shape.clone()))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        // Borrow the cached `Arc` and transpose straight out of it — the old
        // path went through `take`, materializing a full throwaway clone of the
        // (multi-GB) weight just to feed the transpose. Here the only
        // allocation is the transposed result itself.
        let (data, shape) = self
            .cache
            .get(key)
            .ok_or_else(|| anyhow!("weight not found in arc cache: {key}"))?;
        if shape.len() != 2 {
            anyhow::bail!("transpose requires 2D, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        Ok((
            transpose_2d_rowmajor(data.as_ref(), rows, cols),
            vec![cols, rows],
        ))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.cache.keys().cloned().collect()
    }
}

// ─── MLX (mlx-community) loader ───────────────────────────────────
//
// mlx-lm checkpoints keep HF-standard tensor names (`model.layers.N.
// self_attn.q_proj.weight`, …) but quantized packs store the Linear as
// three tensors: `*.weight` (packed uint32 codes), `*.scales`, `*.biases`.
// A `config.json` `quantization` block selects affine / mxfp. This loader
// dequantizes to dense f32 for gather tables + norms, and hands the packed
// affine triples straight to a 4-input `Op::DequantMatMul` on backends that
// support `QuantScheme::MlxAffine`.

/// A quantized Linear as separate packed MLX arrays for a 4-input
/// `Op::DequantMatMul` (`x`, `w_q`, `scales`, `biases`).
/// A quantized Linear as separate packed MLX arrays (weight codes + scales +
/// biases as raw bytes, with `scale_dtype()`/`bias_dtype()` helpers). Re-exports
/// the rlx-mlx-io carrier so downstream crates need no direct rlx-mlx-io dep.
pub use rlx_mlx_io::MlxPackedLinear;
/// Host dequantizer for one MLX-affine packed matrix (`codes`,`scales`,`biases`
/// → row-major `[rows, n_groups*group_size]` F32). Re-exported so dedicated
/// crates (e.g. rlx-laguna) can add an affine matmul path without a direct
/// rlx-mlx-io dependency.
pub use rlx_mlx_io::dequant_affine_f32;

/// [`WeightLoader`] over an mlx-community weight directory / safetensors. The
/// backing is either eager (whole checkpoint in RAM) or lazy (mmap, tensors
/// materialized on demand) — see [`MlxLoader::open`] vs [`MlxLoader::open_lazy`].
pub struct MlxLoader {
    w: Box<dyn rlx_mlx_io::MlxRead>,
    taken: HashSet<String>,
}

impl MlxLoader {
    /// Open an mlx-community model directory, `.safetensors`, `.npz`, or `.npy`
    /// **eagerly** (whole checkpoint loaded into RAM).
    pub fn open(path: &str) -> Result<Self> {
        let w =
            rlx_mlx_io::load_path(path).with_context(|| format!("load mlx weights from {path}"))?;
        Ok(Self {
            w: Box::new(w),
            taken: HashSet::new(),
        })
    }

    /// Open an mlx-community directory / `.safetensors` **lazily** (mmap; each
    /// tensor materialized only when taken). Peak RAM is one tensor + dequant
    /// scratch, so a pipeline worker can load its shard of a model far larger
    /// than its own RAM. `.npz`/`.npy` aren't supported — use [`Self::open`].
    pub fn open_lazy(path: &str) -> Result<Self> {
        let w = rlx_mlx_io::load_path_lazy(path)
            .with_context(|| format!("lazily load mlx weights from {path}"))?;
        Ok(Self {
            w: Box::new(w),
            taken: HashSet::new(),
        })
    }

    /// `MADV_WILLNEED`-prefetch the given tensors (best effort) so their shard
    /// pages read ahead while the caller builds+compiles its stage — overlapping
    /// disk IO with compile instead of paying it as page-faults during the first
    /// forward. No-op for the eager loader (already resident). Pass the stage's
    /// expected weight keys, or [`Self::prewarm_all`] for the whole shard.
    pub fn prewarm(&self, keys: &[&str]) {
        self.w.prewarm(keys);
    }

    /// Prefetch every tensor this (lazy) loader indexes — call right after
    /// `open_lazy`, before building the stage graph, to warm the page cache.
    pub fn prewarm_all(&self) {
        let keys = self.w.logical_keys();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        self.w.prewarm(&refs);
    }

    /// True when the mlx checkpoint declares a `quantization` block.
    pub fn is_quantized(&self) -> bool {
        self.w.mlx_config().quantization.is_some()
    }
}

impl WeightLoader for MlxLoader {
    fn format_id(&self) -> &'static str {
        "mlx"
    }
    fn len(&self) -> usize {
        self.w
            .logical_keys()
            .iter()
            .filter(|k| !self.taken.contains(*k))
            .count()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let out = self.w.take_dense_f32(key)?;
        self.taken.insert(key.to_string());
        Ok(out)
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            anyhow::bail!("mlx transpose requires 2D, got {shape:?} for {key}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        Ok((transpose_2d_rowmajor(&data, rows, cols), vec![cols, rows]))
    }
    fn take_packed_mlx(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
        let Some(p) = self.w.take_packed_linear(key)? else {
            return Ok(None);
        };
        self.taken.insert(key.to_string());
        Ok(Some(p))
    }
    fn prewarm(&self, keys: &[&str]) {
        MlxLoader::prewarm(self, keys);
    }
    fn borrow_packed_mlx(&self, key: &str) -> Option<rlx_mlx_io::PackedMlxBorrow<'_>> {
        self.w.borrow_packed(key)
    }
    fn dontneed_packed_mlx(&self, key: &str) {
        self.w.dontneed_packed(key);
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.w
            .logical_keys()
            .into_iter()
            .filter(|k| !self.taken.contains(k))
            .collect()
    }
}

/// Adapter that lets a HF-safetensors-backed [`WeightLoader`] satisfy
/// requests phrased in GGUF-style names (`blk.N.attn_q.weight` etc.).
///
/// Builders like `Qwen35Weights::from_loader` address tensors using
/// the GGUF / llama.cpp convention; the underlying safetensors file
/// stores them under HF / PyTorch names (`model.layers.N.self_attn.q_proj.weight`).
/// This wrapper:
///
/// 1. Tries the requested key verbatim (in case it's already-HF or
///    the file was named GGUF-style).
/// 2. Tries [`gguf_to_hf_name`] to translate the GGUF key → HF key.
/// 3. Returns the underlying loader's error otherwise.
pub struct HfTranslatingLoader<L: WeightLoader> {
    inner: L,
}

impl<L: WeightLoader> HfTranslatingLoader<L> {
    pub fn new(inner: L) -> Self {
        Self { inner }
    }
    pub fn into_inner(self) -> L {
        self.inner
    }
    pub fn inner(&self) -> &L {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut L {
        &mut self.inner
    }
}

impl<L: WeightLoader> WeightLoader for HfTranslatingLoader<L> {
    fn format_id(&self) -> &'static str {
        self.inner.format_id()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self.inner.take(key) {
            Ok(v) => Ok(v),
            Err(first) => {
                for hf in gguf_to_hf_name_candidates(key) {
                    if let Ok(v) = self.inner.take(&hf) {
                        return Ok(v);
                    }
                }
                Err(first)
            }
        }
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self.inner.take_transposed(key) {
            Ok(v) => Ok(v),
            Err(first) => {
                for hf in gguf_to_hf_name_candidates(key) {
                    if let Ok(v) = self.inner.take_transposed(&hf) {
                        return Ok(v);
                    }
                }
                Err(first)
            }
        }
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<crate::weight_map::PackedWeightTensor>> {
        self.inner.take_packed(key)
    }
    fn release_mapped_pages(&self) {
        self.inner.release_mapped_pages();
    }
    fn read_tensor_bytes_into(&self, key: &str, buf: &mut Vec<u8>) -> Result<bool> {
        self.inner.read_tensor_bytes_into(key, buf)
    }
    fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        self.inner.tensor_bytes_borrowed(key)
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
}

/// Dispatch on the file extension via [`crate::weight_registry`].
pub fn load_from_path(path: &str) -> Result<Box<dyn WeightLoader>> {
    crate::weight_registry::open_weight_loader(Path::new(path))
}

// ─── GGUF adapter ─────────────────────────────────────────────────
//
// Wraps `rlx_gguf::GgufFile` so it satisfies `WeightLoader`. Tracks
// taken keys in a side-set since `dequant_f32` borrows the file
// immutably; the alternative — pre-decoding every tensor at load
// time — defeats the point of GGUF's lazy block layout.

pub struct GgufLoader {
    file: rlx_gguf::GgufFile,
    arch: String,
    taken: HashSet<String>,
    /// When true, `remaining_keys` / `len` / `take` treat MTP-head
    /// weights as ordinary tensors instead of hiding them. The base
    /// qwen3 builder ignores MTP tensors regardless — this flag
    /// only changes the *visibility* in the `WeightLoader` surface
    /// so downstream MTP-aware builders can iterate them through
    /// the standard drain pattern.
    include_mtp: bool,
    /// First `blk.N` index that belongs to an MTP head, computed
    /// from `{arch}.block_count - {arch}.nextn_predict_layers` at
    /// construction. `None` for files without the metadata key
    /// (= no MTP heads encoded as trailing blocks).
    mtp_layer_threshold: Option<u32>,
}

impl GgufLoader {
    pub fn from_file(path: &str) -> Result<Self> {
        let file = crate::gguf_support::load_gguf_file(std::path::Path::new(path))?;
        let arch = gguf_architecture_str(&file)
            .unwrap_or("unknown")
            .to_string();
        let mtp_layer_threshold = compute_mtp_layer_threshold(&file);
        Ok(Self {
            file,
            arch,
            taken: HashSet::new(),
            include_mtp: false,
            mtp_layer_threshold,
        })
    }

    /// An empty loader (no tensors) tagged with `arch`. Placeholder for runtimes
    /// that hold their weights out-of-band (e.g. the laguna mlx-affine host path,
    /// whose `PackedMlx` weights carry bytes inline and never touch a GGUF mmap).
    pub fn empty(arch: &str) -> Self {
        Self {
            file: rlx_gguf::GgufFile::empty(),
            arch: arch.to_string(),
            taken: HashSet::new(),
            include_mtp: false,
            mtp_layer_threshold: None,
        }
    }

    pub fn architecture(&self) -> &str {
        &self.arch
    }

    /// First `blk.N` index that the GGUF metadata reports as an MTP
    /// head, derived from `{arch}.block_count -
    /// {arch}.nextn_predict_layers`. `None` for files where the
    /// `nextn_predict_layers` key is absent (= no MTP, or MTP is
    /// encoded under a different naming scheme — fall back to
    /// [`is_mtp_weight`] in that case).
    pub fn mtp_layer_threshold(&self) -> Option<u32> {
        self.mtp_layer_threshold
    }

    /// Borrow the underlying parsed `GgufFile` so callers (e.g. arch
    /// builders that read `general.architecture`-specific keys)
    /// don't have to re-parse 800+ tensor headers a second time.
    pub fn file(&self) -> &rlx_gguf::GgufFile {
        &self.file
    }

    /// Borrow the raw on-disk byte slice for a tensor without
    /// marking it taken. Returns `None` if the key doesn't resolve
    /// or the byte range is invalid. Used by the qwen35 packed-
    /// upload path to stream K-quant bytes from mmap straight into
    /// the compiled arena, skipping a per-tensor `Vec<u8>`
    /// allocation (≈ 16 GB on Qwen3.6-27B Q4_K_M).
    pub fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        let real = self.resolve(key).ok()?;
        let t = self.file.get(&real)?;
        self.file.tensor_bytes(t).ok()
    }

    /// `pread` a tensor into `buf` rather than borrowing it from the mmap —
    /// see [`rlx_gguf::GgufFile::read_tensor_bytes_into`]. Same name resolution
    /// as [`Self::tensor_bytes_borrowed`].
    pub fn read_tensor_bytes_into(&self, key: &str, buf: &mut Vec<u8>) -> Result<bool> {
        let Ok(real) = self.resolve(key) else {
            return Ok(false);
        };
        let Some(t) = self.file.get(&real) else {
            return Ok(false);
        };
        self.file.read_tensor_bytes_into(t, buf)
    }

    /// Variant of [`Self::take_packed`] that returns only the
    /// `(scheme, shape)` metadata without copying bytes. The caller
    /// uploads bytes separately via [`Self::tensor_bytes_borrowed`]
    /// after the graph is compiled — eliminates the per-tensor
    /// `Vec<u8>` allocation. Marks the tensor taken on success;
    /// returns `Ok(None)` for non-K-quant dtypes so the caller can
    /// fall back to the dequant path.
    pub fn take_packed_metadata(
        &mut self,
        key: &str,
    ) -> Result<Option<(rlx_ir::quant::QuantScheme, Vec<usize>)>> {
        let real = self.resolve(key)?;
        if self.taken.contains(&real) {
            return Err(anyhow!("weight already taken: {key} (→ {real})"));
        }
        if !self.include_mtp && self.is_mtp_tensor(&real) {
            return Err(anyhow!(
                "refusing to take MTP weight `{real}` without include_mtp(true)"
            ));
        }
        let t = self
            .file
            .get(&real)
            .ok_or_else(|| anyhow!("tensor missing: {real}"))?;
        let Some(scheme) = ggml_type_to_quant_scheme(t.dtype) else {
            return Ok(None);
        };
        if !dequant_matmul_supported(scheme) {
            return Ok(None);
        }
        let mut shape = t.shape.clone();
        shape.reverse();
        self.taken.insert(real);
        Ok(Some((scheme, shape)))
    }

    /// True if `name` is an MTP weight under this file's naming
    /// scheme. Combines the substring heuristic ([`is_mtp_weight`])
    /// with the model-aware `blk.N where N >= threshold` check.
    pub fn is_mtp_tensor(&self, name: &str) -> bool {
        if is_mtp_weight(name) {
            return true;
        }
        if let Some(thresh) = self.mtp_layer_threshold {
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some(dot) = rest.find('.') {
                    if let Ok(idx) = rest[..dot].parse::<u32>() {
                        if idx >= thresh {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Toggle MTP-weight visibility. With `include = true`, MTP
    /// heads show up in `remaining_keys()` (and count toward `len()`)
    /// — drain-style consumers like
    /// `Qwen3Generator::from_loader` will then pull them into the
    /// weights cache. Default off so non-MTP models behave exactly
    /// as before. Call this before any `take()` / drain so the
    /// inclusion choice is consistent across the load.
    pub fn include_mtp(&mut self, include: bool) -> &mut Self {
        self.include_mtp = include;
        self
    }

    /// Take a tensor's **packed bytes** (no dequant), plus its
    /// [`rlx_ir::quant::QuantScheme`] and safetensors-style shape.
    /// Returns `None` when the tensor is stored uncompressed
    /// (F32/F16/BF16) — caller should fall back to `take()` for
    /// those.
    ///
    /// Used by the qwen3 builder's *packed-weights mode*: the LM
    /// head + per-layer matmul weights stay in the arena as raw
    /// K-quant bytes, and the graph emits
    /// `Op::DequantMatMul { scheme }` instead of `Op::MatMul` for
    /// them. Cuts the load-time memory footprint by ~7-9× on
    /// Q4_K_M / Q6_K models — the unblocker for ≥14 B Qwen3 / Llama
    /// GGUFs on commodity Macs.
    pub fn take_packed(&mut self, key: &str) -> Result<Option<PackedWeightTensor>> {
        let real = self.resolve(key)?;
        if self.taken.contains(&real) {
            return Err(anyhow!("weight already taken: {key} (→ {real})"));
        }
        if !self.include_mtp && self.is_mtp_tensor(&real) {
            return Err(anyhow!(
                "refusing to take MTP weight `{real}` without include_mtp(true)"
            ));
        }
        let t = self
            .file
            .get(&real)
            .ok_or_else(|| anyhow!("tensor missing: {real}"))?;
        // Map ggml dtype → our QuantScheme. K-quants and Q4_0/Q5_0/Q8_0 can stay
        // packed on CPU; Q4_1/Q5_1 + uncompressed F32/F16/BF16 fall through
        // F32/F16/BF16 fall through to the dequant path (return
        // None — caller switches to `take`).
        let Some(scheme) = ggml_type_to_quant_scheme(t.dtype) else {
            return Ok(None);
        };
        if !dequant_matmul_supported(scheme) {
            return Ok(None);
        };
        let bytes = self
            .file
            .tensor_bytes(t)
            .with_context(|| format!("read packed bytes for {real}"))?
            .to_vec();
        let mut shape = t.shape.clone();
        // Match the safetensors-style shape ordering applied by
        // `take` — GGUF stores innermost-first, safetensors stores
        // outermost-first; byte layout is identical.
        shape.reverse();
        self.taken.insert(real);
        Ok(Some((bytes, scheme, shape)))
    }

    /// Copy an already-uncompressed float tensor to host `Vec<f32>`.
    ///
    /// Accepts only `GgmlType` F32 / F16 / BF16. Quantized tensors error
    /// (Laguna: "F32 expand FORBIDDEN") — use [`Self::take_packed`] /
    /// [`Self::take_packed_metadata`] instead.
    pub fn take_native_f32(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let real = self.resolve(key)?;
        if self.taken.contains(&real) {
            return Err(anyhow!("weight already taken: {key} (→ {real})"));
        }
        if !self.include_mtp && self.is_mtp_tensor(&real) {
            return Err(anyhow!(
                "refusing to take MTP weight `{real}` without include_mtp(true)"
            ));
        }
        let t = self
            .file
            .get(&real)
            .ok_or_else(|| anyhow!("tensor missing: {real}"))?;
        if !is_native_float_ggml(t.dtype) {
            return Err(anyhow!(
                "F32 expand FORBIDDEN (`{key}` → {real}, arch={}, {:?}): \
                 quantized tensors must stay packed (`take_packed` / \
                 `take_packed_metadata`). Only native F32/F16/BF16 may copy \
                 to host float via take_native_f32.",
                self.arch,
                t.dtype
            ));
        }
        let (data, raw_shape) = self
            .file
            .dequant_f32(&real)
            .with_context(|| format!("native float copy for {real}"))?;
        self.taken.insert(real);
        let mut shape = raw_shape;
        shape.reverse();
        Ok((data, shape))
    }

    /// Take a single MTP weight by name. Bypasses the `include_mtp`
    /// filter so callers can grab specific heads without flipping
    /// the global visibility. Returns an error if the name isn't a
    /// recognized MTP weight (use `take` for non-MTP keys).
    pub fn take_mtp(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if !self.is_mtp_tensor(key) {
            return Err(anyhow!("not an MTP weight under this file's scheme: {key}"));
        }
        if !self.file.tensors.contains_key(key) {
            return Err(anyhow!("MTP weight not found in GGUF: {key}"));
        }
        if self.taken.contains(key) {
            return Err(anyhow!("MTP weight already taken: {key}"));
        }
        if self.arch == "laguna" {
            let t = self
                .file
                .get(key)
                .ok_or_else(|| anyhow!("MTP weight not found in GGUF: {key}"))?;
            if !is_native_float_ggml(t.dtype) && !crate::gguf_support::laguna_allow_f32_expand() {
                return Err(anyhow!(
                    "F32 expand disabled for Laguna GGUF (`{key}` / take_mtp); \
                     set RLX_LAGUNA_ALLOW_F32_EXPAND=1 to opt in"
                ));
            }
        }
        let (data, raw_shape) = self.file.dequant_f32(key)?;
        self.taken.insert(key.to_string());
        let mut shape = raw_shape;
        shape.reverse();
        Ok((data, shape))
    }
}

impl GgufLoader {
    /// Resolve a caller-supplied key (HF or GGUF naming) to the
    /// actual GGUF tensor name via registered architecture resolvers.
    fn resolve(&self, key: &str) -> Result<String> {
        resolve_gguf_tensor_name(&self.file, &self.arch, key)
            .ok_or_else(|| anyhow!("weight not found in GGUF (arch={}): {key}", self.arch))
    }
}

impl WeightLoader for GgufLoader {
    fn format_id(&self) -> &'static str {
        "gguf"
    }
    fn arch_hint(&self) -> Option<&str> {
        Some(&self.arch)
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<crate::weight_map::PackedWeightTensor>> {
        self.take_packed(key)
    }
    fn read_tensor_bytes_into(&self, key: &str, buf: &mut Vec<u8>) -> Result<bool> {
        GgufLoader::read_tensor_bytes_into(self, key, buf)
    }
    fn release_mapped_pages(&self) {
        self.file.release_mapped_pages();
    }
    fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        GgufLoader::tensor_bytes_borrowed(self, key)
    }
    fn packed_meta(&self, key: &str) -> Option<(rlx_ir::quant::QuantScheme, Vec<usize>)> {
        // Non-destructive twin of `take_packed_metadata`: resolves the tensor
        // and reports (scheme, shape) without touching `self.taken`, so the
        // bucketed decode/prefill builders can re-read it on every rebuild while
        // the bytes are served zero-copy by `tensor_bytes_borrowed`.
        let real = self.resolve(key).ok()?;
        if !self.include_mtp && self.is_mtp_tensor(&real) {
            return None;
        }
        let t = self.file.get(&real)?;
        let scheme = ggml_type_to_quant_scheme(t.dtype)?;
        if !dequant_matmul_supported(scheme) {
            return None;
        }
        let mut shape = t.shape.clone();
        shape.reverse();
        Some((scheme, shape))
    }
    fn len(&self) -> usize {
        self.file
            .tensors
            .keys()
            .filter(|k| !self.taken.contains(*k) && (self.include_mtp || !self.is_mtp_tensor(k)))
            .count()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let real = self.resolve(key)?;
        if self.taken.contains(&real) {
            return Err(anyhow!("weight already taken: {key} (→ {real})"));
        }
        if !self.include_mtp && self.is_mtp_tensor(&real) {
            return Err(anyhow!(
                "refusing to take MTP weight `{real}` without include_mtp(true); \
                 use loader.take_mtp(...) for explicit MTP grabs or \
                 loader.include_mtp(true) to include them in drains"
            ));
        }
        if self.arch == "laguna" {
            let t = self
                .file
                .get(&real)
                .ok_or_else(|| anyhow!("tensor missing: {real}"))?;
            if !is_native_float_ggml(t.dtype) && !crate::gguf_support::laguna_allow_f32_expand() {
                return Err(anyhow!(
                    "F32 expand disabled for Laguna GGUF (`{key}` → {real}, {:?}): \
                     use packed/mmap (`take_packed` / `take_packed_metadata` / \
                     `rlx-laguna --packed-load`), or opt in with \
                     `RLX_LAGUNA_ALLOW_F32_EXPAND=1` / `--allow-f32-expand` \
                     (see header sniff `F32-expand≈`). \
                     Native F32/F16/BF16 side tensors may use `take` / `take_native_f32`.",
                    t.dtype
                ));
            }
        }
        let (mut data, raw_shape) = self.file.dequant_f32(&real)?;
        self.taken.insert(real.clone());
        // Gemma's GGUF conversion script bakes the `(1 + gamma)` offset
        // into every norm weight (see llama.cpp `convert_hf_to_gguf.py`
        // → `GemmaModel.modify_tensors`). HF/safetensors stores raw
        // gamma. Subtract 1 here so the loader publishes the
        // safetensors convention and `GemmaRmsNormStage`'s explicit
        // `+1` stays correct for both sources. Without this the RMS
        // gain is systematically inflated and logits skew (cosine
        // ~0.7 vs llama.cpp; structural, not numerical noise).
        if matches!(
            self.arch.as_str(),
            "gemma" | "gemma2" | "gemma3" | "gemma3n" | "gemma4"
        ) && is_gemma_norm_weight(&real)
        {
            for v in data.iter_mut() {
                *v -= 1.0;
            }
        }
        // GGUF/ggml report tensor shapes innermost-first (`ne[0]` is
        // the fastest-varying dim) while safetensors reports outermost-
        // first. The actual byte layout is identical row-major — only
        // the shape label is reversed. Reverse to match safetensors so
        // existing builders work unchanged; no data movement.
        let mut shape = raw_shape;
        shape.reverse();
        Ok((data, shape))
    }
    /// **BREAKING CHANGE in 0.2.0:** prior to 0.2.0 this method was
    /// a no-op for GGUF (returned the bytes unchanged with the GGUF
    /// shape label) which silently produced garbage logits when the
    /// builder expected `[in, out]` row-major. From 0.2.0 onwards
    /// `take` normalizes GGUF's reverse-shape convention so this
    /// method matches the safetensors variant byte-for-byte.
    /// Downstream code that explicitly worked around the old buggy
    /// behavior (manually re-transposing the result) must drop that
    /// workaround.
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        // After the safetensors normalization in `take`, this matches
        // the WeightMap implementation byte-for-byte.
        let (data, shape) = self.take(key)?;
        // A 1D tensor (e.g. Gemma 4's AltUp/Laurel/per-layer scale vectors that
        // live under .self_attn./.mlp. namespaces) transposes to itself — return
        // as-is instead of erroring so the packed drain doesn't abort.
        if shape.len() == 1 {
            return Ok((data, shape));
        }
        if shape.len() != 2 {
            return Err(anyhow!("transpose requires 2D, got {shape:?}"));
        }
        let (rows, cols) = (shape[0], shape[1]);
        Ok((transpose_2d_rowmajor(&data, rows, cols), vec![cols, rows]))
    }
    fn remaining_keys(&self) -> Vec<String> {
        // MTP weights default to invisible — they belong to optional
        // speculative heads and the base qwen3 builder ignores them.
        // Callers wanting MTP-aware loading flip `include_mtp(true)`
        // first, which surfaces them here.
        self.file
            .tensors
            .keys()
            .filter(|k| {
                !self.taken.contains(k.as_str()) && (self.include_mtp || !self.is_mtp_tensor(k))
            })
            .cloned()
            .collect()
    }
}

impl GgufLoader {
    /// Tensor names that look like MTP heads under this file's
    /// scheme (combines the substring heuristic with the
    /// model-aware `blk.N where N >= threshold` check — see
    /// [`is_mtp_tensor`](Self::is_mtp_tensor)).
    /// Returned unfiltered by `remaining_keys` so consumers wanting
    /// to wire MTP can find them explicitly.
    pub fn mtp_keys(&self) -> Vec<String> {
        self.file
            .tensors
            .keys()
            .filter(|k| self.is_mtp_tensor(k))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Naive reference transpose the blocked helper must match byte-for-byte.
    fn transpose_naive(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut dst = vec![0f32; src.len()];
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
        dst
    }

    #[test]
    fn transpose_matches_naive() {
        // Cover shapes that straddle the 64-wide tile: exact tile, sub-tile,
        // and non-square with both dims crossing a block boundary.
        for &(rows, cols) in &[(3usize, 5usize), (64, 64), (65, 1), (1, 65), (100, 70)] {
            let src: Vec<f32> = (0..rows * cols).map(|n| n as f32).collect();
            assert_eq!(
                transpose_2d_rowmajor(&src, rows, cols),
                transpose_naive(&src, rows, cols),
                "blocked != naive at {rows}x{cols}"
            );
        }
    }

    #[test]
    fn arc_cache_take_transposed_is_correct_and_nondestructive() {
        // [2, 3] row-major → [3, 2].
        let mut cache: HashMap<String, ArcF32Tensor> = HashMap::new();
        cache.insert(
            "w".into(),
            (Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), vec![2, 3]),
        );
        let mut ldr = ArcCacheLoader::new(&cache);
        let (t, shape) = ldr.take_transposed("w").unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // View loader: the backing cache is untouched, so a second take works.
        assert!(ldr.take_transposed("w").is_ok());
    }

    #[test]
    fn unknown_extension_errors() {
        let r = load_from_path("/tmp/no-such-thing.bin");
        match r {
            Err(e) => assert!(e.to_string().contains("unsupported")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn gemma_qk_norm_names_get_gguf_gamma_unbake() {
        assert!(is_gemma_norm_weight("blk.0.attn_q_norm.weight"));
        assert!(is_gemma_norm_weight("blk.7.attn_k_norm.weight"));
        assert!(is_gemma_norm_weight(
            "model.layers.0.self_attn.q_norm.weight"
        ));
        assert!(is_gemma_norm_weight(
            "model.layers.3.self_attn.k_norm.weight"
        ));
        assert!(!is_gemma_norm_weight("blk.0.attn_q.weight"));
    }

    #[test]
    fn hf_to_gguf_top_level() {
        assert_eq!(
            hf_to_gguf_name("model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            hf_to_gguf_name("lm_head.weight").as_deref(),
            Some("output.weight")
        );
    }

    #[test]
    fn hf_to_gguf_per_layer() {
        let cases = [
            (
                "model.layers.0.self_attn.q_proj.weight",
                "blk.0.attn_q.weight",
            ),
            (
                "model.layers.7.self_attn.o_proj.weight",
                "blk.7.attn_output.weight",
            ),
            (
                "model.layers.3.mlp.gate_proj.weight",
                "blk.3.ffn_gate.weight",
            ),
            (
                "model.layers.12.mlp.down_proj.weight",
                "blk.12.ffn_down.weight",
            ),
            (
                "model.layers.4.input_layernorm.weight",
                "blk.4.attn_norm.weight",
            ),
            (
                "model.layers.4.post_attention_layernorm.weight",
                "blk.4.ffn_norm.weight",
            ),
            (
                "model.layers.0.self_attn.q_norm.weight",
                "blk.0.attn_q_norm.weight",
            ),
        ];
        for (hf, gguf) in cases {
            assert_eq!(
                hf_to_gguf_name(hf).as_deref(),
                Some(gguf),
                "mismatch for {hf}"
            );
        }
    }

    #[test]
    fn hf_to_gguf_gemma4_output_scale() {
        assert_eq!(
            hf_to_gguf_name_for_arch("model.layers.5.self_attn.output_scale.weight", "gemma4")
                .as_deref(),
            Some("blk.5.layer_output_scale.weight")
        );
        assert_eq!(
            gguf_to_hf_name_for_arch("blk.5.layer_output_scale.weight", "gemma4").as_deref(),
            Some("model.layers.5.self_attn.output_scale.weight")
        );
    }

    /// Muse Glimmer's four per-layer norms round-trip without colliding.
    ///
    /// The Llama convention aliases `post_attention_layernorm` → `ffn_norm`,
    /// which is WRONG here: Muse Glimmer ships `post_attention_norm` (post-attn,
    /// pre-residual) and `ffn_norm` (pre-FFN) as two distinct tensors. Getting
    /// this wrong silently loads one norm's gain in the other's slot.
    #[test]
    fn muse_glimmer_four_norms_round_trip() {
        for (hf, gguf) in [
            (
                "model.layers.7.input_layernorm.weight",
                "blk.7.attn_norm.weight",
            ),
            (
                "model.layers.7.post_attention_layernorm.weight",
                "blk.7.post_attention_norm.weight",
            ),
            (
                "model.layers.7.pre_feedforward_layernorm.weight",
                "blk.7.ffn_norm.weight",
            ),
            (
                "model.layers.7.post_feedforward_layernorm.weight",
                "blk.7.post_ffw_norm.weight",
            ),
            (
                "model.layers.7.self_attn.q_norm.weight",
                "blk.7.attn_q_norm.weight",
            ),
            (
                "model.layers.7.self_attn.k_norm.weight",
                "blk.7.attn_k_norm.weight",
            ),
            // Attention OUTPUT gate — must NOT alias the MLP's `ffn_gate`.
            (
                "model.layers.7.self_attn.gate_proj.weight",
                "blk.7.attn_gate.weight",
            ),
            (
                "model.layers.7.mlp.gate_proj.weight",
                "blk.7.ffn_gate.weight",
            ),
        ] {
            assert_eq!(
                hf_to_gguf_name_for_arch(hf, "muse-glimmer").as_deref(),
                Some(gguf),
                "hf→gguf mismatch for {hf}"
            );
            assert_eq!(
                gguf_to_hf_name_for_arch(gguf, "muse-glimmer").as_deref(),
                Some(hf),
                "gguf→hf mismatch for {gguf}"
            );
        }
    }

    /// Every per-layer tensor `unsloth/Muse-Glimmer-30B-GGUF` actually ships
    /// maps to a DISTINCT HF name.
    ///
    /// The real file carries 731 tensors = 52 layers × the 14 tails below, plus
    /// `token_embd` / `output_norm` / `output`. A missing tail would panic the
    /// builder on load; a colliding one would silently load the wrong tensor,
    /// so assert the mapping is both total and injective.
    #[test]
    fn muse_glimmer_real_tensor_set_maps_injectively() {
        const PER_LAYER_TAILS: [&str; 14] = [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_gate.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];
        let mut seen = std::collections::HashSet::new();
        for tail in PER_LAYER_TAILS {
            let gguf = format!("blk.0.{tail}");
            let hf = gguf_to_hf_name_for_arch(&gguf, "muse-glimmer")
                .unwrap_or_else(|| panic!("muse-glimmer has no HF mapping for {gguf}"));
            assert!(seen.insert(hf.clone()), "collision: {gguf} → {hf}");
        }
        assert_eq!(seen.len(), PER_LAYER_TAILS.len());
        for (gguf, hf) in [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
        ] {
            assert_eq!(
                gguf_to_hf_name_for_arch(gguf, "muse-glimmer").as_deref(),
                Some(hf)
            );
        }
        // 52 layers × 14 + 3 top-level == the real file's tensor count.
        assert_eq!(52 * PER_LAYER_TAILS.len() + 3, 731);
    }

    /// Muse Glimmer must NOT be treated as a Gemma norm: its converter already
    /// folded `weight + 1` in and llama.cpp consumes the gain verbatim, so
    /// routing it through the Gemma `-1` unbake would subtract 1 twice.
    #[test]
    fn muse_glimmer_norms_are_not_gemma_unbaked() {
        // `is_gemma_norm_weight` is name-based, so it necessarily matches the
        // shared spellings; the guard is that the Gemma *arch list* excludes
        // muse-glimmer, keeping the delta off this arch's load path.
        for arch in ["gemma2", "gemma3", "gemma3n", "gemma4", "gemma4moe"] {
            assert_ne!(arch, "muse-glimmer");
        }
        // The dedicated branch wins before any Gemma handling.
        assert_eq!(
            gguf_to_hf_name_for_arch("blk.0.ffn_norm.weight", "muse-glimmer").as_deref(),
            Some("model.layers.0.pre_feedforward_layernorm.weight")
        );
    }

    #[test]
    fn hf_to_gguf_unknown_returns_none() {
        assert!(hf_to_gguf_name("model.layers.0.some_new_thing.weight").is_none());
        assert!(hf_to_gguf_name("model.layers.foo.input_layernorm.weight").is_none());
    }

    #[test]
    fn qwen35_hf_aliases_cover_linear_attn_and_nested_prefix() {
        assert_eq!(
            gguf_to_hf_qwen35_name("token_embd.weight").as_deref(),
            Some("model.language_model.embed_tokens.weight")
        );
        assert_eq!(
            gguf_to_hf_qwen35_name("output_norm.weight").as_deref(),
            Some("model.language_model.norm.weight")
        );
        let cases = [
            (
                "blk.0.attn_qkv.weight",
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            ),
            (
                "blk.0.attn_gate.weight",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            ),
            (
                "blk.0.ssm_conv1d.weight",
                "model.language_model.layers.0.linear_attn.conv1d.weight",
            ),
            (
                "blk.0.ssm_dt.bias",
                "model.language_model.layers.0.linear_attn.dt_bias",
            ),
            (
                "blk.0.ssm_a",
                "model.language_model.layers.0.linear_attn.A_log",
            ),
            (
                "blk.0.ssm_beta.weight",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            ),
            (
                "blk.0.ssm_alpha.weight",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            ),
            (
                "blk.0.ssm_norm.weight",
                "model.language_model.layers.0.linear_attn.norm.weight",
            ),
            (
                "blk.0.ssm_out.weight",
                "model.language_model.layers.0.linear_attn.out_proj.weight",
            ),
            (
                "blk.0.post_attention_norm.weight",
                "model.language_model.layers.0.post_attention_layernorm.weight",
            ),
            (
                "blk.11.attn_q.weight",
                "model.language_model.layers.11.self_attn.q_proj.weight",
            ),
        ];
        for (gguf, hf) in cases {
            assert_eq!(
                gguf_to_hf_qwen35_name(gguf).as_deref(),
                Some(hf),
                "mismatch for {gguf}"
            );
            assert!(
                gguf_to_hf_name_candidates(gguf).iter().any(|c| c == hf),
                "candidates missing {hf}"
            );
        }
    }

    #[test]
    fn mtp_detection() {
        assert!(is_mtp_weight("mtp_blk.0.attn_q.weight"));
        assert!(is_mtp_weight("output_mtp_0.weight"));
        assert!(is_mtp_weight("model.layers.0.mtp_head.weight"));
        assert!(!is_mtp_weight("blk.0.attn_q.weight"));
        assert!(!is_mtp_weight("output.weight"));
    }

    /// Build a tiny GGUF with `qwen35.block_count = 25` and
    /// `qwen35.nextn_predict_layers = 1`, then verify the loader's
    /// model-aware detector classifies `blk.24.*` as MTP while
    /// `blk.0.*` stays in the base model. This is the unsloth /
    /// DeepSeek-V3 convention — substring-based `is_mtp_weight`
    /// alone wouldn't catch it.
    #[test]
    fn ggml_block_quants_map_to_packed_scheme() {
        use rlx_gguf::GgmlType;
        assert_eq!(
            ggml_type_to_quant_scheme(GgmlType::Q4_0),
            Some(rlx_ir::quant::QuantScheme::GgufQ4_0)
        );
        assert_eq!(
            ggml_type_to_quant_scheme(GgmlType::Q5_0),
            Some(rlx_ir::quant::QuantScheme::GgufQ5_0)
        );
        assert_eq!(
            ggml_type_to_quant_scheme(GgmlType::Q8_0),
            Some(rlx_ir::quant::QuantScheme::GgufQ8_0)
        );
        // PrismML Bonsai Q1_0 / Ternary Q2_0 stay packed for
        // DequantMatMul instead of expanding to ~108 GB of F32.
        assert_eq!(
            ggml_type_to_quant_scheme(GgmlType::Q1_0),
            Some(rlx_ir::quant::QuantScheme::GgufQ1_0)
        );
        assert_eq!(
            ggml_type_to_quant_scheme(GgmlType::Q2_0),
            Some(rlx_ir::quant::QuantScheme::GgufQ2_0)
        );
        assert!(dequant_matmul_supported(
            rlx_ir::quant::QuantScheme::GgufQ1_0
        ));
        assert!(dequant_matmul_supported(
            rlx_ir::quant::QuantScheme::GgufQ2_0
        ));
    }

    #[test]
    fn q6k_dequant_matmul_follows_block_probe() {
        use rlx_ir::quant::QuantScheme;
        assert!(dequant_matmul_supported(QuantScheme::GgufQ4K));
        assert_eq!(
            dequant_matmul_supported(QuantScheme::GgufQ6K),
            super::probe_q6k_block_dequant()
        );
    }

    #[test]
    fn gguf_loader_threshold_based_mtp_detection() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes()); // 3 tensors
        buf.extend_from_slice(&3u64.to_le_bytes()); // 3 KV
        // KV: general.architecture = "qwen35" (type 8 = string)
        let write_string = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes()); // type 4 = u32
            buf.extend_from_slice(&v.to_le_bytes());
        };
        write_string(&mut buf, "general.architecture", "qwen35");
        write_u32(&mut buf, "qwen35.block_count", 25);
        write_u32(&mut buf, "qwen35.nextn_predict_layers", 1);
        // Three tensors: blk.0.attn_q.weight (main), blk.24.attn_q.weight (MTP),
        // and token_embd.weight (always main).
        let write_tensor = |buf: &mut Vec<u8>, name: &str, shape: &[usize], off: u64| {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &d in shape {
                buf.extend_from_slice(&(d as u64).to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // F32
            buf.extend_from_slice(&off.to_le_bytes());
        };
        write_tensor(&mut buf, "blk.0.attn_q.weight", &[4, 4], 0);
        write_tensor(&mut buf, "blk.24.attn_q.weight", &[4, 4], 64);
        write_tensor(&mut buf, "token_embd.weight", &[4, 4], 128);
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        // 3 × 4×4 f32 = 192 bytes of data.
        for _ in 0..(3 * 16) {
            buf.extend_from_slice(&0.5f32.to_le_bytes());
        }
        let path = std::env::temp_dir().join("rlx_mtp_threshold_test.gguf");
        std::fs::write(&path, &buf).unwrap();
        let loader = GgufLoader::from_file(path.to_str().unwrap()).unwrap();

        assert_eq!(loader.mtp_layer_threshold(), Some(24));
        assert!(!loader.is_mtp_tensor("blk.0.attn_q.weight"));
        assert!(loader.is_mtp_tensor("blk.24.attn_q.weight"));
        assert!(!loader.is_mtp_tensor("token_embd.weight"));
        let mtp = loader.mtp_keys();
        assert_eq!(mtp, vec!["blk.24.attn_q.weight".to_string()]);

        std::fs::remove_file(&path).ok();
    }

    /// Synthesize a tiny GGUF file in memory with two GGUF-named
    /// tensors (`token_embd.weight` and `blk.0.attn_q.weight`) plus
    /// one MTP weight (`output_mtp_0.weight`). Then verify:
    ///   1. `take()` resolves the HF names via the mapper.
    ///   2. `remaining_keys()` hides the MTP weight.
    ///   3. `mtp_keys()` surfaces it for callers that opt in.
    #[test]
    fn gguf_loader_resolves_hf_names_and_skips_mtp() {
        let mut tensors = Vec::new();
        let mut data: Vec<f32> = Vec::new();

        // tensor #1: token_embd.weight, shape [3, 4], values 0..12
        let t1: Vec<f32> = (0..12).map(|x| x as f32).collect();
        tensors.push(("token_embd.weight", vec![3usize, 4], data.len()));
        data.extend_from_slice(&t1);

        // tensor #2: blk.0.attn_q.weight, shape [4, 4], values 100..116
        let t2: Vec<f32> = (100..116).map(|x| x as f32).collect();
        tensors.push(("blk.0.attn_q.weight", vec![4usize, 4], data.len()));
        data.extend_from_slice(&t2);

        // tensor #3: output_mtp_0.weight (MTP head) — present but skipped
        let t3: Vec<f32> = vec![0.5f32; 8];
        tensors.push(("output_mtp_0.weight", vec![2usize, 4], data.len()));
        data.extend_from_slice(&t3);

        // Build the GGUF byte stream by hand.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count

        // Tensor info section.
        for (name, shape, _) in &tensors {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &d in shape {
                buf.extend_from_slice(&(d as u64).to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // dtype = F32
            // Offset within the data segment — patched after alignment.
            buf.extend_from_slice(&0u64.to_le_bytes());
        }
        // Align to DEFAULT_ALIGNMENT before data section.
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        let data_start = buf.len();
        for v in &data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // Patch the offsets we wrote as 0 above.
        let header = (4 + 4 + 8 + 8) as usize; // magic + version + tensor_count + kv_count
        let mut cursor = header;
        for (name, shape, byte_off) in &tensors {
            let name_len_bytes = 8;
            let name_bytes = name.len();
            let n_dims_bytes = 4;
            let dims_bytes = shape.len() * 8;
            let dtype_bytes = 4;
            let off_bytes = 8;
            let info_size =
                name_len_bytes + name_bytes + n_dims_bytes + dims_bytes + dtype_bytes + off_bytes;
            let off_field_at = cursor + info_size - off_bytes;
            let final_off = (*byte_off * 4) as u64; // f32 byte offset within data segment
            for i in 0..8 {
                buf[off_field_at + i] = (final_off >> (i * 8)) as u8;
            }
            cursor += info_size;
        }
        let _ = data_start;

        // Write to a temp file (GgufFile reads from a path).
        let path = std::env::temp_dir().join("rlx_test_qwen3_mini.gguf");
        std::fs::write(&path, &buf).unwrap();

        let mut loader = GgufLoader::from_file(path.to_str().unwrap()).unwrap();
        // Pre-MTP: 3 tensors total, MTP is hidden so visible count = 2.
        assert_eq!(loader.len(), 2);

        // HF-name lookup resolves via the mapper. GGUF reports shapes
        // innermost-first while safetensors reports outermost-first;
        // byte layout is identical, only the shape label flips. The
        // synthetic GGUF here was built with shape `[3, 4]`, so the
        // loader hands back `[4, 3]` with the same bytes.
        let (out, shape) = loader
            .take("model.embed_tokens.weight")
            .expect("hf-named token_embd should resolve");
        assert_eq!(shape, vec![4, 3]);
        assert_eq!(&out, &t1);

        let (out, shape) = loader
            .take("model.layers.0.self_attn.q_proj.weight")
            .expect("hf-named attn_q should resolve");
        assert_eq!(shape, vec![4, 4]);
        assert_eq!(&out, &t2);

        // MTP weight stays out of remaining_keys, in mtp_keys.
        assert_eq!(loader.remaining_keys(), Vec::<String>::new());
        assert_eq!(loader.mtp_keys(), vec!["output_mtp_0.weight".to_string()]);

        // include_mtp(true): MTP weights become visible in
        // remaining_keys + drainable via take(), and `take_mtp`
        // works for explicit grabs without the flag.
        let mut loader2 = GgufLoader::from_file(path.to_str().unwrap()).unwrap();
        loader2.include_mtp(true);
        let visible: std::collections::HashSet<String> =
            loader2.remaining_keys().into_iter().collect();
        assert!(visible.contains("token_embd.weight"));
        assert!(visible.contains("blk.0.attn_q.weight"));
        assert!(
            visible.contains("output_mtp_0.weight"),
            "MTP weight should be visible with include_mtp(true)"
        );
        let (mtp_data, mtp_shape) = loader2.take_mtp("output_mtp_0.weight").unwrap();
        assert_eq!(mtp_shape, vec![4, 2]);
        assert_eq!(mtp_data, t3);

        // include_mtp(false) — default — refuses MTP via take().
        let mut loader3 = GgufLoader::from_file(path.to_str().unwrap()).unwrap();
        let err = loader3.take("output_mtp_0.weight").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("include_mtp(true)"),
            "expected MTP guard error, got: {msg}"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_gguf_file_errors() {
        // .gguf is now a known extension → error comes from `open`,
        // not from the dispatcher.
        let r = load_from_path("/tmp/no-such-thing-rlx-gguf-test.gguf");
        assert!(r.is_err());
    }

    #[test]
    fn laguna_gguf_f32_take_is_forbidden() {
        use rlx_gguf::{GgmlType, GgufWriter, MetaValue, bytes_for_public};
        // SAFETY: isolate from opt-in left by another test.
        unsafe { std::env::remove_var("RLX_LAGUNA_ALLOW_F32_EXPAND") };
        let path = std::env::temp_dir().join("rlx_laguna_forbid_f32_take.gguf");
        let mut w = GgufWriter::new();
        w.set_arch("laguna");
        w.set_meta("laguna.block_count", MetaValue::U32(1));
        w.set_meta("laguna.embedding_length", MetaValue::U32(16));
        let n = 256usize;
        let nbytes = bytes_for_public(GgmlType::Q4K, n).unwrap();
        w.add_tensor_bytes(
            "blk.0.ffn_down.weight",
            vec![16, 16],
            GgmlType::Q4K,
            vec![0u8; nbytes],
        )
        .unwrap();
        let bias = vec![1.0f32, 2.0, 3.0, 4.0];
        let bias_bytes: Vec<u8> = bytemuck::cast_slice(&bias).to_vec();
        w.add_tensor_bytes("blk.0.exp_probs_b.bias", vec![4], GgmlType::F32, bias_bytes)
            .unwrap();
        w.write_to_path(&path).unwrap();

        let mut loader = GgufLoader::from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(loader.architecture(), "laguna");
        let err = loader.take("blk.0.ffn_down.weight").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("disabled") || msg.contains("FORBIDDEN"),
            "expected Laguna F32 forbid, got: {msg}"
        );
        let err2 = loader.take_native_f32("blk.0.ffn_down.weight").unwrap_err();
        assert!(
            format!("{err2:#}").contains("FORBIDDEN"),
            "take_native_f32 must refuse Q4K"
        );
        let (data, shape) = loader.take_native_f32("blk.0.exp_probs_b.bias").unwrap();
        assert_eq!(shape, vec![4]);
        assert_eq!(data, bias);
        // take() on native F32 is allowed for Laguna.
        let mut loader2 = GgufLoader::from_file(path.to_str().unwrap()).unwrap();
        let (data2, _) = loader2.take("blk.0.exp_probs_b.bias").unwrap();
        assert_eq!(data2, bias);

        // Opt-in: quant take succeeds.
        unsafe { std::env::set_var("RLX_LAGUNA_ALLOW_F32_EXPAND", "1") };
        let mut loader3 = GgufLoader::from_file(path.to_str().unwrap()).unwrap();
        let (qdata, _) = loader3.take("blk.0.ffn_down.weight").unwrap();
        assert_eq!(qdata.len(), 256);
        unsafe { std::env::remove_var("RLX_LAGUNA_ALLOW_F32_EXPAND") };

        std::fs::remove_file(&path).ok();
    }
}
