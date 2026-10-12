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

//! A tiny synthetic `qwen4exp` checkpoint, and the shape contract it comes
//! from.
//!
//! Qwen3.8-Flash-Next is 512 × 56 B and its smallest published quantization is
//! tens of gigabytes, so nothing here can run the real model. What this
//! preserves is the *architecture*: the shipped layer schedule
//! (`[linear, linear, linear, indexed]`), hyper-connections at all three
//! sites, a PLE module on layer 1, the QSA indexer and the shared-expert MoE —
//! only the widths shrink.
//!
//! [`tensor_shapes`] is the single statement of each tensor's name and
//! orientation, in the form the GGUF loader hands over (torch `[out, in]`).
//! `tests/tensor_manifest.rs` checks it against the published checkpoint's
//! real tensor list, so a shape typo here fails there rather than silently
//! short-writing an arena row.

use std::collections::HashMap;

use crate::config::{LayerKind, Qwen4ExpConfig};

/// A deterministic, small-magnitude filler.
pub fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.5
        })
        .collect()
}

/// A 4-layer `qwen4exp` with the published schedule and tiny widths.
///
/// `indexer_budget` is a parameter because QSA selection is the identity
/// exactly when the budget covers every block (see
/// [`Qwen4ExpConfig::qsa_is_dense`]) — a test that wants the selection to
/// actually engage has to shrink it.
pub fn tiny_config(indexer_budget: usize) -> Qwen4ExpConfig {
    const LAYERS: usize = 4;
    let layer_types: Vec<LayerKind> = (0..LAYERS)
        .map(|i| {
            if (i + 1).is_multiple_of(4) {
                LayerKind::Indexed
            } else {
                LayerKind::Linear
            }
        })
        .collect();
    let compress_ratios = layer_types
        .iter()
        .map(|k| match k {
            LayerKind::Indexed => 4,
            LayerKind::Linear => 0,
        })
        .collect();
    Qwen4ExpConfig {
        vocab_size: 32,
        hidden_size: 16,
        num_hidden_layers: LAYERS,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 1024,
        tie_word_embeddings: false,
        // Synthetic gammas are written already-offset, like a GGUF's.
        rms_norm_offset: false,
        rope_theta: 10_000.0,
        rope_dim_count: 4,
        rope_dim_sections: vec![1, 1, 0, 0],
        mrope_interleaved: true,
        layer_types,
        compress_ratios,
        ssm_conv_kernel: 4,
        ssm_state_size: 4,
        ssm_group_count: 2,
        ssm_time_step_rank: 6,
        ssm_inner_size: 24,
        output_gate_sigmoid: true,
        hc_count: 4,
        hc_lowrank: 8,
        indexer_n_heads: 2,
        indexer_head_dim: 8,
        indexer_budget,
        indexer_compress_ratio: 4,
        // One PLE module, on a linear-attention layer, as the real model has.
        ple_layers: vec![1],
        ple_ngram_size: 3,
        ple_heads_per_ngram: 2,
        ple_conv_kernel: 4,
        ple_head_dim: 4,
        ple_eos_token_id: 31,
        ple_image_token_id: 0,
        // Small enough that the prime search is instant and the row indices
        // stay inside a table a test can actually allocate.
        ngram_vocab_size_base: 1_000,
        make_ngram_vocab_size_divisible_by: 128,
        seed: 1234,
        ple_constants: None,
        num_experts: 6,
        num_experts_per_tok: 2,
        expert_ffn_size: 12,
        shared_expert_ffn_size: 12,
        norm_topk_prob: true,
        expert_weights_scale: 1.0,
        num_nextn_predict_layers: 0,
        with_mtp: false,
    }
}

/// Every tensor the checkpoint carries, with the shape the GGUF loader reports.
///
/// GGUF stores dims innermost-first, so a projection written `[in, out]` in the
/// file arrives here as `[out, in]`; a `[in, mid, experts]` expert bank arrives
/// as `[experts, mid, in]`. Both orientations are what the emitters assume.
pub fn tensor_shapes(cfg: &Qwen4ExpConfig) -> Vec<(String, Vec<usize>)> {
    let h = cfg.hidden_size;
    let hc_dim = cfg.hc_dim();
    let lr = cfg.hc_lowrank;
    let mut v: Vec<(String, Vec<usize>)> = vec![
        ("token_embd.weight".into(), vec![cfg.vocab_size, h]),
        ("output_hc_norm.weight".into(), vec![hc_dim]),
        ("output_hc_down.weight".into(), vec![lr, hc_dim]),
        ("output_hc_up.weight".into(), vec![hc_dim, lr]),
    ];
    if !cfg.tie_word_embeddings {
        v.push(("output.weight".into(), vec![cfg.vocab_size, h]));
    }
    if !cfg.ple_layers.is_empty() {
        // The real one is [320001536, 160] — 51 B weights. A synthetic
        // checkpoint sizes it from the hash so a test can hold it.
        let rows = crate::config::PleHash::derive(cfg, 0).padded_rows as usize;
        v.push((
            "per_layer_token_embd.weight".into(),
            vec![rows, cfg.ple_head_dim],
        ));
    }

    // `block_count()` includes the MTP block only when `with_mtp` is set, which
    // mirrors the checkpoint layout: the draft block ships as its own
    // `mtp-*.gguf`.
    for i in 0..cfg.block_count() {
        let b = format!("blk.{i}");
        let is_mtp = i >= cfg.num_hidden_layers;
        for site in ["hc_attn", "hc_ffn"] {
            v.push((format!("{b}.{site}_norm.weight"), vec![hc_dim]));
            v.push((format!("{b}.{site}_down.weight"), vec![lr, hc_dim]));
            v.push((format!("{b}.{site}_up.weight"), vec![hc_dim, lr]));
            v.push((
                format!("{b}.{site}_inject.weight"),
                vec![cfg.hc_count, hc_dim],
            ));
        }

        // The MTP block is a QSA attention layer whatever the trunk's schedule
        // says at that index.
        let kind = if is_mtp {
            LayerKind::Indexed
        } else {
            cfg.layer_kind(i)
        };
        match kind {
            LayerKind::Linear => {
                let nv = cfg.n_v_heads();
                v.extend([
                    (format!("{b}.attn_qkv.weight"), vec![cfg.gdn_conv_dim(), h]),
                    (
                        format!("{b}.attn_gate.weight"),
                        vec![cfg.gdn_value_dim(), h],
                    ),
                    (
                        format!("{b}.ssm_conv1d.weight"),
                        vec![cfg.gdn_conv_dim(), cfg.ssm_conv_kernel],
                    ),
                    (format!("{b}.ssm_dt.bias"), vec![nv]),
                    (format!("{b}.ssm_a"), vec![nv]),
                    (format!("{b}.ssm_beta.weight"), vec![nv, h]),
                    (format!("{b}.ssm_alpha.weight"), vec![nv, h]),
                    (format!("{b}.ssm_norm.weight"), vec![cfg.ssm_state_size]),
                    (format!("{b}.ssm_out.weight"), vec![h, cfg.gdn_value_dim()]),
                ]);
            }
            LayerKind::Indexed => {
                let hd = cfg.head_dim;
                let q_dim = cfg.num_attention_heads * hd;
                let kv_dim = cfg.num_key_value_heads * hd;
                let idx = cfg.indexer_head_dim;
                v.extend([
                    // Query AND gate, interleaved per head.
                    (format!("{b}.attn_q.weight"), vec![2 * q_dim, h]),
                    (format!("{b}.attn_q_norm.weight"), vec![hd]),
                    (format!("{b}.attn_k.weight"), vec![kv_dim, h]),
                    (format!("{b}.attn_k_norm.weight"), vec![hd]),
                    (format!("{b}.attn_v.weight"), vec![kv_dim, h]),
                    (format!("{b}.attn_output.weight"), vec![h, q_dim]),
                    (
                        format!("{b}.indexer.q_proj.weight"),
                        vec![cfg.indexer_n_heads * idx, h],
                    ),
                    (format!("{b}.indexer.k_proj.weight"), vec![idx, h]),
                    (format!("{b}.indexer.q_norm.weight"), vec![idx]),
                    (format!("{b}.indexer.k_norm.weight"), vec![idx]),
                ]);
            }
        }

        if !is_mtp && cfg.ple_layer_index(i).is_some() {
            let pe = cfg.ple_embed_dim();
            v.extend([
                (format!("{b}.ple_key.weight"), vec![hc_dim, pe]),
                (format!("{b}.ple_value.weight"), vec![h, pe]),
                (format!("{b}.ple_norm_key.weight"), vec![hc_dim]),
                (format!("{b}.ple_norm_query.weight"), vec![hc_dim]),
                (format!("{b}.ple_norm_conv.weight"), vec![hc_dim]),
                (
                    format!("{b}.ple_conv1d.weight"),
                    vec![hc_dim, cfg.ple_conv_kernel],
                ),
            ]);
        }

        let e = cfg.num_experts;
        let ef = cfg.expert_ffn_size;
        let sf = cfg.shared_expert_ffn_size;
        v.extend([
            (format!("{b}.ffn_gate_inp.weight"), vec![e, h]),
            (format!("{b}.ffn_gate_exps.weight"), vec![e, ef, h]),
            (format!("{b}.ffn_up_exps.weight"), vec![e, ef, h]),
            (format!("{b}.ffn_down_exps.weight"), vec![e, h, ef]),
            (format!("{b}.ffn_gate_inp_shexp.weight"), vec![h]),
            (format!("{b}.ffn_gate_shexp.weight"), vec![sf, h]),
            (format!("{b}.ffn_up_shexp.weight"), vec![sf, h]),
            (format!("{b}.ffn_down_shexp.weight"), vec![h, sf]),
        ]);

        if is_mtp {
            v.extend([
                // `cat([fc_embedding, fc_hidden], dim=1)` — one projection of
                // width `2 · hidden`, applied per hyper-connection stream.
                (format!("{b}.nextn.eh_proj.weight"), vec![h, 2 * h]),
                (format!("{b}.nextn.enorm.weight"), vec![h]),
                // Grouped over a stream, like every other hyper-connection norm.
                (format!("{b}.nextn.hnorm.weight"), vec![hc_dim]),
                (format!("{b}.nextn.hc_head_norm.weight"), vec![hc_dim]),
                (format!("{b}.nextn.hc_head_down.weight"), vec![lr, hc_dim]),
                (format!("{b}.nextn.hc_head_up.weight"), vec![hc_dim, lr]),
            ]);
        }
    }
    v
}

/// Synthesize every tensor in [`tensor_shapes`] as a name → (data, shape) map.
///
/// Norm gains are centred on 1.0 (a GGUF's are, because the converter folded
/// the reference's `1 +` in) rather than on 0, so a synthetic forward pass has
/// the same activation scale as a real one. `ssm_a` is made negative for the
/// same reason: it is a decay coefficient the converter stores as
/// `-exp(A_log)`, and a positive one makes the recurrence diverge.
pub fn synth_tensors(cfg: &Qwen4ExpConfig, seed: u64) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut out = HashMap::new();
    let mut s = seed;
    for (name, shape) in tensor_shapes(cfg) {
        let n: usize = shape.iter().product();
        s = s.wrapping_add(7);
        let data = if name.ends_with("norm.weight") {
            fill(n, s).iter().map(|x| 1.0 + x).collect()
        } else if name.ends_with(".ssm_a") {
            fill(n, s).iter().map(|x| -(x.abs() + 0.01)).collect()
        } else {
            fill(n, s)
        };
        out.insert(name, (data, shape));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tiny_config_is_valid() {
        tiny_config(2048).validate().unwrap();
    }

    /// The shape table and the config's own tensor manifest are two
    /// independent statements of the checkpoint contract. If they disagree, one
    /// of them is wrong.
    #[test]
    fn shapes_cover_exactly_the_manifest() {
        for with_mtp in [false, true] {
            let mut cfg = tiny_config(2048);
            cfg.num_nextn_predict_layers = 1;
            cfg.with_mtp = with_mtp;
            let mut from_shapes: Vec<String> =
                tensor_shapes(&cfg).into_iter().map(|(n, _)| n).collect();
            from_shapes.sort();
            assert_eq!(from_shapes, cfg.tensor_manifest(), "with_mtp = {with_mtp}");
        }
    }

    /// Every synthetic tensor's data length must match its declared shape, or
    /// the arena silently reads past a row.
    #[test]
    fn synthetic_data_matches_its_shapes() {
        let cfg = tiny_config(2048);
        for (name, (data, shape)) in synth_tensors(&cfg, 1) {
            let n: usize = shape.iter().product();
            assert_eq!(
                data.len(),
                n,
                "{name} has {} values for {shape:?}",
                data.len()
            );
        }
    }
}
