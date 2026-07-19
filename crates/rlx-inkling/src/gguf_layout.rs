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

//! Unsloth / llama.cpp GGUF layout for Inkling
//! ([unsloth/inkling-GGUF](https://huggingface.co/unsloth/inkling-GGUF)).
//!
//! Sniffed from `UD-IQ1_S` (Jul 2026):
//! - Shard `…-00001-of-00007.gguf` — metadata + tokenizer only (`n_tensors = 0`)
//! - Later shards hold `token_embd.*`, `blk.N.*`, `output.*`
//! - `general.architecture = "inkling"`
//! - Metadata prefix: `inkling.*`

/// Accepted `general.architecture` tags.
pub const GGUF_ARCHES: &[&str] = &["inkling", "inkling_mm_model"];

/// Map a Unsloth GGUF tensor name → [`crate::eager::TextWeights`] key
/// (dense split form used by the reference forward).
///
/// Returns `None` for MoE expert packs that need dequant + reshape
/// (handled by the GGUF weight loader, not a 1:1 rename).
pub fn gguf_to_eager_key(name: &str) -> Option<String> {
    if name == "token_embd.weight" {
        return Some("embed".into());
    }
    if name == "token_embd_norm.weight" {
        return Some("embed_norm".into());
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
        "ffn_norm.weight" => format!("layers.{layer}.mlp_norm"),
        "attn_q_norm.weight" => format!("layers.{layer}.q_norm"),
        "attn_k_norm.weight" => format!("layers.{layer}.k_norm"),
        "attn_q.weight" => format!("layers.{layer}.wq"),
        "attn_k.weight" => format!("layers.{layer}.wk"),
        "attn_v.weight" => format!("layers.{layer}.wv"),
        "attn_r.weight" => format!("layers.{layer}.wr"),
        "attn_output.weight" => format!("layers.{layer}.wo"),
        "attn_rel_proj.weight" => format!("layers.{layer}.rel_proj"),
        "shortconv_k.weight" => format!("layers.{layer}.k_sconv"),
        "shortconv_v.weight" => format!("layers.{layer}.v_sconv"),
        "shortconv_attn.weight" => format!("layers.{layer}.attn_sconv"),
        "shortconv_mlp.weight" => format!("layers.{layer}.mlp_sconv"),
        "ffn_gate.weight" => format!("layers.{layer}.gate"),
        "ffn_up.weight" => format!("layers.{layer}.up"),
        "ffn_down.weight" => format!("layers.{layer}.down"),
        "ffn_gscale.weight" => format!("layers.{layer}.mlp_global_scale"),
        "ffn_gate_inp.weight" => format!("layers.{layer}.gate_weight"),
        "exp_probs_b.bias" => format!("layers.{layer}.gate_bias"),
        "ffn_gate_shexp.weight" => format!("layers.{layer}.shared_gate"),
        "ffn_up_shexp.weight" => format!("layers.{layer}.shared_up"),
        "ffn_down_shexp.weight" => format!("layers.{layer}.shared_down"),
        // Expert packs — loader must fuse/dequant; names documented for sniffs.
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            return None;
        }
        _ => return None,
    };
    Some(key)
}

/// Layout notes for loaders (layout quirks vs HF transformers packing).
///
/// Confirmed against Unsloth `UD-IQ1_S` shard-00002 header (Jul 2026).
pub const LAYOUT_NOTES: &str = "\
Unsloth inkling GGUF quirks (vs transformers safetensors):
- shortconv_*.weight is [kernel, channels]: SWA [4,2048], global [4,1024]
- attn_rel_proj.weight is [rel_extent, d_rel]: SWA [512,16], global [1024,16]
- Linear weights follow ggml [in, out] conventions (e.g. attn_q [6144,8192])
- Dense blk.0..1: ffn_{gate,up,down}; MoE from blk.2: ffn_*_exps + ffn_*_shexp
- Router ffn_gate_inp [H, n_routed+n_shared] = [6144,258]; exp_probs_b.bias [256]
- gate/up_exps [H, I, n_routed]=[6144,3072,256]; down_exps [I,H,n_routed]=[3072,6144,256]
- shexp packs use last-dim 2 (shared experts); UD-IQ1_S mixes IQ1S/IQ3XXS/Q5K/Q6K/Q8_0/F32
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renames_stem_and_dense() {
        assert_eq!(
            gguf_to_eager_key("token_embd.weight").as_deref(),
            Some("embed")
        );
        assert_eq!(
            gguf_to_eager_key("blk.0.attn_q.weight").as_deref(),
            Some("layers.0.wq")
        );
        assert_eq!(
            gguf_to_eager_key("blk.2.ffn_gate_inp.weight").as_deref(),
            Some("layers.2.gate_weight")
        );
        assert!(gguf_to_eager_key("blk.2.ffn_gate_exps.weight").is_none());
    }
}
