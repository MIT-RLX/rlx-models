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

//! Pluggable GGUF tensor-name resolution per `general.architecture`.

use rlx_gguf::GgufFile;
use std::sync::{Mutex, OnceLock};

use crate::weight_loader::{
    gguf_to_hf_name, gguf_to_hf_name_for_arch, hf_to_gguf_name, hf_to_gguf_name_for_arch,
};

/// Resolve a builder-requested tensor name to the name stored in a GGUF file.
pub trait GgufTensorNameResolver: Send + Sync {
    fn matches_arch(&self, arch: &str) -> bool;
    fn resolve(&self, file: &GgufFile, requested_key: &str) -> Option<String>;
}

/// HF `model.layers.N.*` ↔ GGUF `blk.N.*` (Llama, Qwen3, Qwen35, …).
pub struct LlamaFamilyGgufResolver;

impl GgufTensorNameResolver for LlamaFamilyGgufResolver {
    fn matches_arch(&self, arch: &str) -> bool {
        matches!(
            arch,
            "llama"
                | "llama4"
                | "phi3"
                | "phi4"
                | "qwen3"
                | "qwen2"
                | "qwen35"
                | "qwen35moe"
                | "qwen36"
                | "gemma"
                | "gemma2"
                | "mistral"
        )
    }

    fn resolve(&self, file: &GgufFile, key: &str) -> Option<String> {
        if file.tensors.contains_key(key) {
            return Some(key.to_string());
        }
        if let Some(g) = hf_to_gguf_name(key) {
            if file.tensors.contains_key(&g) {
                return Some(g);
            }
        }
        if let Some(h) = gguf_to_hf_name(key) {
            if file.tensors.contains_key(&h) {
                return Some(h);
            }
        }
        None
    }
}

/// Strip common HF prefixes and match verbatim tensor names (architecture-agnostic fallback).
pub struct PrefixStripGgufResolver;

/// Alias for [`PrefixStripGgufResolver`] (older name).
pub type PassThroughGgufResolver = PrefixStripGgufResolver;

impl GgufTensorNameResolver for PrefixStripGgufResolver {
    fn matches_arch(&self, _arch: &str) -> bool {
        true
    }

    fn resolve(&self, file: &GgufFile, key: &str) -> Option<String> {
        let mut k = key.to_string();
        for prefix in [
            "model.diffusion_model.",
            "diffusion_model.",
            "transformer.",
            "model.",
        ] {
            if let Some(rest) = k.strip_prefix(prefix) {
                k = rest.to_string();
                break;
            }
        }
        if file.tensors.contains_key(&k) {
            return Some(k);
        }
        if file.tensors.contains_key(key) {
            return Some(key.to_string());
        }
        None
    }
}

/// Gemma 2/3/4: 4 RMSNorms per layer disagree with the Llama 2-norm convention.
///
/// Llama treats `post_attention_layernorm` as the pre-FFN norm and aliases it
/// to `ffn_norm`. Gemma 2/3/4 (V2/V3/V4 layer styles) have a dedicated
/// `post_attention_norm` between the attention output and the residual add,
/// *and* a separate `ffn_norm` / `post_ffw_norm` pair around the MLP. Without
/// this resolver, the Llama mapper would alias `post_attention_layernorm` to
/// `ffn_norm`, collide with `pre_feedforward_layernorm`, and silently load
/// the wrong tensor. The tail map is identical across these arches — only
/// the GGUF arch tag (and runtime details like sliding-window stride) differ.
pub struct Gemma2GgufResolver;

impl GgufTensorNameResolver for Gemma2GgufResolver {
    fn matches_arch(&self, arch: &str) -> bool {
        matches!(
            arch,
            "gemma2" | "gemma3" | "gemma3n" | "gemma4" | "gemma4moe"
        )
    }

    fn resolve(&self, file: &GgufFile, key: &str) -> Option<String> {
        // Identity hit first — accept native GGUF names verbatim.
        if file.tensors.contains_key(key) {
            return Some(key.to_string());
        }
        // Handle the four-norm-per-layer scheme explicitly. The Llama mapper
        // is wrong for `post_attention_layernorm` (it aliases to `ffn_norm`,
        // which Gemma 2 reserves for the pre-FFN norm) and has no entry at
        // all for the `pre_feedforward_layernorm`/`post_feedforward_layernorm`
        // pair.
        if let Some(rest) = key.strip_prefix("model.layers.") {
            if let Some((idx, tail)) = rest.split_once('.') {
                let gguf_tail = match tail {
                    "post_attention_layernorm.weight" => Some("post_attention_norm.weight"),
                    "pre_feedforward_layernorm.weight" => Some("ffn_norm.weight"),
                    "post_feedforward_layernorm.weight" => Some("post_ffw_norm.weight"),
                    // Inverse of `gguf_to_hf_name_for_arch` — Gemma 4 GGUF stores
                    // per-layer output scalars as `layer_output_scale`, not HF
                    // `self_attn.output_scale`.
                    "self_attn.output_scale.weight" => Some("layer_output_scale.weight"),
                    _ => None,
                };
                if let Some(t) = gguf_tail {
                    let g = format!("blk.{idx}.{t}");
                    if file.tensors.contains_key(&g) {
                        return Some(g);
                    }
                }
            }
        }
        // Fall through to Llama-family mapping for everything else
        // (input_layernorm, attn/mlp weights, lm_head, embeddings, …).
        LlamaFamilyGgufResolver.resolve(file, key)
    }
}

/// Muse Glimmer: Gemma-style four norms per layer **plus** an attention output
/// gate.
///
/// Shares the Gemma norm naming (`attn_norm` / `post_attention_norm` /
/// `ffn_norm` / `post_ffw_norm`), so it needs the same override of the Llama
/// `post_attention_layernorm → ffn_norm` alias. It is deliberately NOT folded
/// into [`Gemma2GgufResolver`]: Gemma's converter bakes `(1 + gamma)` into every
/// norm gain and the loader subtracts 1 on take, whereas Muse Glimmer's
/// converter folds `weight + 1` in already and llama.cpp consumes the gain
/// verbatim. Routing it through the Gemma path would subtract 1 twice.
///
/// The extra tensor is `attn_gate` (`[n_embd, n_head * head_dim]`), exposed to
/// builders as `self_attn.gate_proj` so it can't collide with the MLP's
/// `mlp.gate_proj` (GGUF `ffn_gate`).
pub struct MuseGlimmerGgufResolver;

impl GgufTensorNameResolver for MuseGlimmerGgufResolver {
    fn matches_arch(&self, arch: &str) -> bool {
        arch == "muse-glimmer"
    }

    fn resolve(&self, file: &GgufFile, key: &str) -> Option<String> {
        // Identity hit first — accept native GGUF names verbatim.
        if file.tensors.contains_key(key) {
            return Some(key.to_string());
        }
        if let Some(rest) = key.strip_prefix("model.layers.") {
            if let Some((idx, tail)) = rest.split_once('.') {
                let gguf_tail = match tail {
                    "post_attention_layernorm.weight" => Some("post_attention_norm.weight"),
                    "pre_feedforward_layernorm.weight" => Some("ffn_norm.weight"),
                    "post_feedforward_layernorm.weight" => Some("post_ffw_norm.weight"),
                    "self_attn.q_norm.weight" => Some("attn_q_norm.weight"),
                    "self_attn.k_norm.weight" => Some("attn_k_norm.weight"),
                    "self_attn.gate_proj.weight" => Some("attn_gate.weight"),
                    _ => None,
                };
                if let Some(t) = gguf_tail {
                    let g = format!("blk.{idx}.{t}");
                    if file.tensors.contains_key(&g) {
                        return Some(g);
                    }
                }
            }
        }
        // Everything else (input_layernorm, q/k/v/o, MLP, embeddings, lm_head)
        // follows the Llama convention unchanged.
        LlamaFamilyGgufResolver.resolve(file, key)
    }
}

/// Phi 3 / 4: fused `attn_qkv` and `ffn_up` (gate∥up) tensors.
pub struct Phi3GgufResolver;

impl GgufTensorNameResolver for Phi3GgufResolver {
    fn matches_arch(&self, arch: &str) -> bool {
        matches!(arch, "phi3" | "phi4")
    }

    fn resolve(&self, file: &GgufFile, key: &str) -> Option<String> {
        if file.tensors.contains_key(key) {
            return Some(key.to_string());
        }
        for arch in ["phi3", "phi4"] {
            if let Some(g) = hf_to_gguf_name_for_arch(key, arch) {
                if file.tensors.contains_key(&g) {
                    return Some(g);
                }
            }
            if let Some(h) = gguf_to_hf_name_for_arch(key, arch) {
                if file.tensors.contains_key(&h) {
                    return Some(h);
                }
            }
        }
        None
    }
}

/// Qwen3.5 native `blk.N.*` names; also accept HF aliases via the Llama mapper.
pub struct Qwen35NativeGgufResolver;

impl GgufTensorNameResolver for Qwen35NativeGgufResolver {
    fn matches_arch(&self, arch: &str) -> bool {
        matches!(arch, "qwen35" | "qwen35moe" | "qwen36")
    }

    fn resolve(&self, file: &GgufFile, key: &str) -> Option<String> {
        if file.tensors.contains_key(key) {
            return Some(key.to_string());
        }
        LlamaFamilyGgufResolver.resolve(file, key)
    }
}

static CUSTOM_RESOLVERS: Mutex<Vec<Box<dyn GgufTensorNameResolver>>> = Mutex::new(Vec::new());
static BUILTIN_RESOLVERS: OnceLock<Vec<Box<dyn GgufTensorNameResolver>>> = OnceLock::new();

fn builtin_resolvers() -> &'static Vec<Box<dyn GgufTensorNameResolver>> {
    BUILTIN_RESOLVERS.get_or_init(|| {
        vec![
            Box::new(Qwen35NativeGgufResolver),
            Box::new(MuseGlimmerGgufResolver),
            Box::new(Gemma2GgufResolver),
            Box::new(Phi3GgufResolver),
            Box::new(LlamaFamilyGgufResolver),
            Box::new(PrefixStripGgufResolver),
        ]
    })
}

/// Register built-in GGUF resolvers (idempotent). Called automatically on first resolve; \
/// call from `main` if you register custom resolvers and need ordering guarantees.
pub fn ensure_builtin_resolvers() {
    let _ = builtin_resolvers();
}

/// Register a custom resolver (call before first GGUF load). Later registrations win
/// among resolvers that match the same architecture.
pub fn register_gguf_tensor_resolver(resolver: Box<dyn GgufTensorNameResolver>) {
    CUSTOM_RESOLVERS
        .lock()
        .expect("gguf resolver registry lock")
        .push(resolver);
}

/// Resolve `requested_key` against tensors in `file` using registered resolvers.
pub fn resolve_gguf_tensor_name(
    file: &GgufFile,
    arch: &str,
    requested_key: &str,
) -> Option<String> {
    for r in builtin_resolvers().iter() {
        if r.matches_arch(arch) {
            if let Some(name) = r.resolve(file, requested_key) {
                return Some(name);
            }
        }
    }
    let custom = CUSTOM_RESOLVERS
        .lock()
        .expect("gguf resolver registry lock");
    for r in custom.iter() {
        if r.matches_arch(arch) {
            if let Some(name) = r.resolve(file, requested_key) {
                return Some(name);
            }
        }
    }
    if file.tensors.contains_key(requested_key) {
        return Some(requested_key.to_string());
    }
    if let Some(g) = hf_to_gguf_name(requested_key) {
        if file.tensors.contains_key(&g) {
            return Some(g);
        }
    }
    if let Some(h) = gguf_to_hf_name(requested_key) {
        if file.tensors.contains_key(&h) {
            return Some(h);
        }
    }
    None
}
