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

//! Unsloth / llama.cpp GGUF layout for Laguna
//! ([unsloth/Laguna-S-2.1-GGUF](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF)).
//!
//! Reference: llama.cpp `#25165` / `src/models/laguna.cpp`.
//! - `general.architecture = "laguna"`
//! - Metadata prefix: `laguna.*`
//! - Softplus attn gate: `blk.N.attn_gate.weight`
//! - MoE: `ffn_*_exps` + `ffn_*_shexp` + `ffn_gate_inp` + `exp_probs_b.bias`

/// Accepted `general.architecture` tags.
pub const GGUF_ARCHES: &[&str] = &["laguna"];

/// Map a Unsloth / llama.cpp GGUF tensor name → [`crate::eager::TextWeights`] key.
///
/// Returns `None` for MoE expert packs that need dequant + reshape.
pub fn gguf_to_eager_key(name: &str) -> Option<String> {
    if name == "token_embd.weight" {
        return Some("embed".into());
    }
    if name == "output_norm.weight" {
        return Some("norm".into());
    }
    if name == "output.weight" {
        return Some("unembed".into());
    }
    let rest = name.strip_prefix("blk.")?;
    let (idx_s, rest) = rest.split_once('.')?;
    let layer: usize = idx_s.parse().ok()?;
    let key = match rest {
        "attn_norm.weight" => format!("layers.{layer}.attn_norm"),
        "ffn_norm.weight" => format!("layers.{layer}.ffn_norm"),
        "attn_q_norm.weight" => format!("layers.{layer}.q_norm"),
        "attn_k_norm.weight" => format!("layers.{layer}.k_norm"),
        "attn_q.weight" => format!("layers.{layer}.wq"),
        "attn_k.weight" => format!("layers.{layer}.wk"),
        "attn_v.weight" => format!("layers.{layer}.wv"),
        "attn_output.weight" => format!("layers.{layer}.wo"),
        "attn_gate.weight" => format!("layers.{layer}.wg"),
        "ffn_gate.weight" => format!("layers.{layer}.gate"),
        "ffn_up.weight" => format!("layers.{layer}.up"),
        "ffn_down.weight" => format!("layers.{layer}.down"),
        "ffn_gate_inp.weight" => format!("layers.{layer}.gate_weight"),
        "exp_probs_b.bias" | "ffn_exp_probs_b.bias" => format!("layers.{layer}.gate_bias"),
        "ffn_gate_shexp.weight" => format!("layers.{layer}.shared_gate"),
        "ffn_up_shexp.weight" => format!("layers.{layer}.shared_up"),
        "ffn_down_shexp.weight" => format!("layers.{layer}.shared_down"),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            return None;
        }
        _ => return None,
    };
    Some(key)
}

/// Layout notes for loaders (ggml packing vs HF).
pub const LAYOUT_NOTES: &str = "\
Laguna GGUF quirks (vs HF safetensors):
- Linear weights follow ggml [in, out] conventions (transpose vs HF [out, in])
- Softplus attn gate: blk.N.attn_gate.weight — width n_head (per-head) or n_head*head_dim
- Dense lead layers: ffn_{gate,up,down}; MoE: ffn_*_exps + ffn_*_shexp + ffn_gate_inp
- Router score bias: blk.N.exp_probs_b.bias [n_expert]
- gate/up_exps [H, I, n_expert]; down_exps [I, H, n_expert]
- Split Unsloth UD quants: pass first shard (…-00001-of-0000N.gguf); siblings load by name
- Needs llama.cpp with arch laguna (PR #25165 / poolside laguna branch)
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renames_stem_and_gate() {
        assert_eq!(
            gguf_to_eager_key("token_embd.weight").as_deref(),
            Some("embed")
        );
        assert_eq!(
            gguf_to_eager_key("blk.0.attn_gate.weight").as_deref(),
            Some("layers.0.wg")
        );
        assert_eq!(
            gguf_to_eager_key("blk.1.ffn_gate_inp.weight").as_deref(),
            Some("layers.1.gate_weight")
        );
        assert!(gguf_to_eager_key("blk.1.ffn_gate_exps.weight").is_none());
    }
}
