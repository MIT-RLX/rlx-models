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

//! MiniMax-M3 weights loader — normalizes HF safetensors names to the keys the
//! flow expects and stacks per-expert MoE weights.
//!
//! HF ships the text model under a `language_model.` prefix and stores each
//! routed expert separately as `…experts.{j}.{w1,w2,w3}.weight` (`w1`=gate,
//! `w3`=up, `w2`=down). This module strips the prefix and stacks the experts
//! into the `experts.gate_up_proj [E, 2·inter, hidden]` / `experts.down_proj
//! [E, hidden, inter]` tensors that [`super::moe::emit_m3_moe`] consumes.
//!
//! NOTE: real M3 is 428B — loading it as f32 exceeds any local machine. This
//! path is built so it *can* run on adequate hardware; it is exercised in tests
//! only against tiny synthetic snapshots.

use anyhow::{Result, anyhow};
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;

use super::config::MiniMaxM3Config;

pub type Snapshot = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Map a GGUF (ggml `blk.*`) tensor name to the flow parameter name — best-effort
/// per llama.cpp ggml-org/llama.cpp#24908, for the 1:1-renamed tensors.
///
/// Returns `None` for the stacked MoE expert tensors (`ffn_{gate,up,down}_exps`),
/// whose gate/up halves must be combined into `experts.gate_up_proj` — that
/// (byte-layout-dependent) step is a follow-up pending a real M3 GGUF.
pub fn gguf_to_flow_name(name: &str) -> Option<String> {
    // Global tensors.
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    // Per-layer `blk.{i}.<rest>`.
    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let i: usize = idx.parse().ok()?;
    let lp = format!("model.layers.{i}");
    let sa = format!("{lp}.self_attn");
    let mp = format!("{lp}.block_sparse_moe");
    let mapped = match tail {
        "attn_norm.weight" => format!("{lp}.input_layernorm.weight"),
        "ffn_norm.weight" => format!("{lp}.post_attention_layernorm.weight"),
        "attn_q.weight" => format!("{sa}.q_proj.weight"),
        "attn_k.weight" => format!("{sa}.k_proj.weight"),
        "attn_v.weight" => format!("{sa}.v_proj.weight"),
        "attn_output.weight" => format!("{sa}.o_proj.weight"),
        "attn_q_norm.weight" => format!("{sa}.q_norm.weight"),
        "attn_k_norm.weight" => format!("{sa}.k_norm.weight"),
        "attn_index_q.weight" => format!("{sa}.index_q_proj.weight"),
        "attn_index_k.weight" => format!("{sa}.index_k_proj.weight"),
        "attn_index_q_norm.weight" => format!("{sa}.index_q_norm.weight"),
        "attn_index_k_norm.weight" => format!("{sa}.index_k_norm.weight"),
        "ffn_gate_inp.weight" => format!("{mp}.gate.weight"),
        "ffn_gate_inp.bias" | "exp_probs_b.bias" => format!("{mp}.e_score_correction_bias"),
        "ffn_gate_shexp.weight" => format!("{mp}.shared_experts.gate_proj.weight"),
        "ffn_up_shexp.weight" => format!("{mp}.shared_experts.up_proj.weight"),
        "ffn_down_shexp.weight" => format!("{mp}.shared_experts.down_proj.weight"),
        "ffn_gate.weight" => format!("{lp}.mlp.gate_proj.weight"),
        "ffn_up.weight" => format!("{lp}.mlp.up_proj.weight"),
        "ffn_down.weight" => format!("{lp}.mlp.down_proj.weight"),
        // Stacked experts need gate/up combining → handled separately.
        _ => return None,
    };
    Some(mapped)
}

/// Load and normalize the M3 text weights from a safetensors/gguf path into an
/// f32 snapshot keyed by the flow's parameter names.
pub fn load_m3_text_snapshot(cfg: &MiniMaxM3Config, path: &str) -> Result<Snapshot> {
    let mut wm =
        WeightMap::from_file(path).map_err(|e| anyhow!("minimax-m3: load weights {path}: {e}"))?;
    let keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
    let mut raw: Snapshot = HashMap::with_capacity(keys.len());
    for k in keys {
        let v = wm.take(&k)?;
        let nk = k
            .strip_prefix("language_model.")
            .map(str::to_string)
            .unwrap_or(k);
        raw.insert(nk, v);
    }
    normalize_snapshot(cfg, raw)
}

/// Stack per-expert MoE tensors in an already-name-normalized snapshot. Pure
/// (no I/O) so it is unit-testable against synthetic tensors.
pub fn normalize_snapshot(cfg: &MiniMaxM3Config, mut raw: Snapshot) -> Result<Snapshot> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let e = cfg.num_local_experts;

    for i in 0..cfg.num_hidden_layers {
        if !cfg.is_moe_layer(i) {
            continue;
        }
        let mp = format!("model.layers.{i}.block_sparse_moe");
        // Already stacked? (e.g. a snapshot produced by this fn) — skip.
        if raw.contains_key(&format!("{mp}.experts.gate_up_proj")) {
            continue;
        }
        // No per-expert tensors present (tiny test synths the stacked form) — skip.
        if !raw.contains_key(&format!("{mp}.experts.0.w1.weight")) {
            continue;
        }
        let mut gate_up = Vec::with_capacity(e * 2 * inter * hidden);
        let mut down = Vec::with_capacity(e * hidden * inter);
        for j in 0..e {
            let (w1, _) = raw
                .remove(&format!("{mp}.experts.{j}.w1.weight"))
                .ok_or_else(|| anyhow!("missing {mp}.experts.{j}.w1.weight"))?;
            let (w3, _) = raw
                .remove(&format!("{mp}.experts.{j}.w3.weight"))
                .ok_or_else(|| anyhow!("missing {mp}.experts.{j}.w3.weight"))?;
            let (w2, _) = raw
                .remove(&format!("{mp}.experts.{j}.w2.weight"))
                .ok_or_else(|| anyhow!("missing {mp}.experts.{j}.w2.weight"))?;
            // gate_up per expert = [w1 (gate) ; w3 (up)] → [2·inter, hidden].
            gate_up.extend_from_slice(&w1);
            gate_up.extend_from_slice(&w3);
            down.extend_from_slice(&w2); // [hidden, inter]
        }
        raw.insert(
            format!("{mp}.experts.gate_up_proj"),
            (gate_up, vec![e, 2 * inter, hidden]),
        );
        raw.insert(
            format!("{mp}.experts.down_proj"),
            (down, vec![e, hidden, inter]),
        );
    }
    Ok(raw)
}
