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

//! Motif-3 text decoder flow.
//!
//! With MHC enabled (the shipped configuration) the residual stream is
//! `[1, seq, E, hidden]` for the whole stack:
//!
//! ```text
//!   embed → expand ×E
//!     N × [ mhc_attn gates → reduce → RMSNorm → GDLA  → mhc combine
//!           mhc_ffn  gates → reduce → RMSNorm → FFN   → mhc combine ]
//!   mean over E → RMSNorm → lm_head
//! ```
//!
//! With `mhc_enabled = false` it degrades to the ordinary pre-norm residual
//! stack, which is what `MotifDecoderLayer.forward` does.
//!
//! The FFN is a dense PolyNorm MLP on the first `n_dense_first_layers` layers
//! and the 384-expert PolyNorm MoE everywhere after.

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::{LayerAttn, MotifConfig, POLYNORM_EPS};
use crate::gdla::{GdlaDims, ROPE_COS, ROPE_SIN, SWA_ROPE_COS, SWA_ROPE_SIN, emit_gdla_attention};
use crate::mhc::{MhcDims, apply_h_pre, combine, emit_mhc_gates};
use crate::moe::{MotifMoeDims, emit_motif_mlp_2d, emit_motif_moe};
use crate::polynorm::PolyNormSpec;

/// Motif's token-embedding tensor name.
pub const EMBED_KEY: &str = "model.embed_tokens.weight";

fn rmsnorm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{key}.weight"), false)?;
    let zb = emit.synth_param(
        &format!("{key}.zb"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, g, zb, eps))
}

/// Build the Motif-3 prefill graph for a fixed `seq`.
///
/// `weights` must already have been through
/// [`crate::weights::prepare_checkpoint`].
///
/// Inputs: `input_ids [1, seq]`, `rope_cos`/`rope_sin` `[seq, qk_rope_head_dim/2]`
/// (see [`MotifConfig::rope_tables`]) and — when the config interleaves
/// sliding-window layers — `swa_rope_cos`/`swa_rope_sin`
/// ([`MotifConfig::swa_rope_tables`]).
pub fn build_motif_text_flow(
    cfg: &MotifConfig,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    cfg.validate()?;
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let half = cfg.qk_rope_head_dim() / 2;
    let expansion = if cfg.mhc_enabled {
        cfg.mhc_expansion_rate
    } else {
        1
    };

    let mhc = MhcDims {
        hidden,
        expansion,
        sinkhorn_iters: cfg.mhc_sinkhorn_iters,
        h_post_coeff: 1.0 + cfg.mhc_h_post_alpha_end,
        seq,
    };
    let poly = PolyNormSpec {
        eps: POLYNORM_EPS,
        hidden_clamp: cfg.hidden_clamp,
        output_scale: cfg.polynorm_output_scale,
        clamp_result: true,
    };
    let moe = MotifMoeDims {
        hidden,
        moe_inter: cfg.moe_intermediate_size(),
        num_experts: cfg.num_experts,
        top_k: cfg.experts_top_k,
        route_scale: cfg.route_scale,
        has_expert_bias: cfg.load_balance_coeff.is_some(),
        has_shared_expert: cfg.num_shared_experts > 0,
        poly,
        seq,
    };
    // The dense MLP is a MotifMLP: PolyNormTorch, so no clamp on the product.
    let dense_poly = PolyNormSpec {
        clamp_result: false,
        ..poly
    };

    let mut flow = ModelFlow::new("motif3")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, seq], f))
        .input(ROPE_COS, Shape::new(&[seq, half], f))
        .input(ROPE_SIN, Shape::new(&[seq, half], f))
        .zero_beta_named("motif.zero_beta.hidden", hidden)
        .embed(EMBED_KEY);
    if cfg.has_sliding_layers() {
        flow = flow
            .input(SWA_ROPE_COS, Shape::new(&[seq, half], f))
            .input(SWA_ROPE_SIN, Shape::new(&[seq, half], f));
    }

    let stream = Shape::new(&[1, seq, expansion, hidden], f);
    let flat = Shape::new(&[1, seq, hidden], f);

    if cfg.mhc_enabled {
        let (si, ei, hi) = (seq as i64, expansion as i64, hidden as i64);
        let out_shape = stream.clone();
        flow = flow.plugin_named("mhc_expand", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("mhc_expand needs the embedding output"))?
                .hir_id();
            let mut gb = HirMut::new(emit.hir());
            let x = gb.reshape_(x, vec![1, si, 1, hi]);
            let x = gb.expand_(x, vec![1, si, ei, hi]);
            Ok(Some(emit.wrap(x, out_shape.clone())))
        });
    }

    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let is_moe = cfg.is_moe_layer(i);
        let dense_inter = cfg.intermediate_size;
        let gdla = GdlaDims {
            hidden,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            grouped_ratio: cfg.grouped_ratio(),
            head_dim: cfg.head_dim(),
            qk_rope_head_dim: cfg.qk_rope_head_dim(),
            v_head_dim: cfg.v_head_dim(),
            q_lora_rank: cfg.q_lora_rank,
            kv_lora_rank: cfg.kv_lora_rank,
            window: match cfg.layer_attn(i) {
                LayerAttn::Sliding(w) => Some(w),
                LayerAttn::Global => None,
            },
            score_scale: cfg.attn_score_scale(i),
            eps,
            seq,
        };
        let mhc_on = cfg.mhc_enabled;
        let out_shape = if mhc_on { stream.clone() } else { flat.clone() };
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("layer{i} needs a hidden input"))?
                .hir_id();

            // ── attention sublayer ──
            let (attn_in, gates) = if mhc_on {
                let g = emit_mhc_gates(emit, &format!("{prefix}.mhc_attn"), x, mhc)?;
                let reduced = {
                    let mut gb = HirMut::new(emit.hir());
                    apply_h_pre(&mut gb, x, g.h_pre, mhc)
                };
                (reduced, Some(g))
            } else {
                (x, None)
            };
            let normed = rmsnorm(
                emit,
                &format!("{prefix}.input_layernorm"),
                attn_in,
                hidden,
                eps,
            )?;
            let attn = emit_gdla_attention(emit, &format!("{prefix}.self_attn"), normed, gdla)?;
            let h = match gates {
                Some(g) => {
                    let mut gb = HirMut::new(emit.hir());
                    combine(&mut gb, x, attn, g, mhc)
                }
                None => {
                    let mut gb = HirMut::new(emit.hir());
                    gb.add(x, attn)
                }
            };

            // ── FFN sublayer ──
            let (ffn_in, gates) = if mhc_on {
                let g = emit_mhc_gates(emit, &format!("{prefix}.mhc_ffn"), h, mhc)?;
                let reduced = {
                    let mut gb = HirMut::new(emit.hir());
                    apply_h_pre(&mut gb, h, g.h_pre, mhc)
                };
                (reduced, Some(g))
            } else {
                (h, None)
            };
            let normed = rmsnorm(
                emit,
                &format!("{prefix}.post_attention_layernorm"),
                ffn_in,
                hidden,
                eps,
            )?;
            let ffn = if is_moe {
                emit_motif_moe(emit, &format!("{prefix}.moe"), normed, moe)?
            } else {
                let x2d = {
                    let mut gb = HirMut::new(emit.hir());
                    gb.reshape_(normed, vec![seq as i64, hidden as i64])
                };
                let y = emit_motif_mlp_2d(
                    emit,
                    &format!("{prefix}.mlp"),
                    x2d,
                    dense_inter,
                    dense_poly,
                )?;
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(y, vec![1, seq as i64, hidden as i64])
            };
            let out = match gates {
                Some(g) => {
                    let mut gb = HirMut::new(emit.hir());
                    combine(&mut gb, h, ffn, g, mhc)
                }
                None => {
                    let mut gb = HirMut::new(emit.hir());
                    gb.add(h, ffn)
                }
            };
            Ok(Some(emit.wrap(out, out_shape.clone())))
        });
    }

    if cfg.mhc_enabled {
        let out_shape = flat.clone();
        flow = flow.plugin_named("mhc_reduce", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("mhc_reduce needs the last layer output"))?
                .hir_id();
            let mut gb = HirMut::new(emit.hir());
            let x = gb.mean(x, vec![2], false);
            Ok(Some(emit.wrap(x, out_shape.clone())))
        });
    }

    flow = flow.final_norm(eps);
    let flow = if with_lm_head {
        flow.lm_head(cfg.vocab_size, hidden, cfg.tie_word_embeddings)
    } else {
        flow.output("hidden")
    };
    flow.build_with(&mut WeightMapSource(weights), None)
}
