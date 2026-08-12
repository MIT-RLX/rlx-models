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

//! Ling 3.0 text decoder flow:
//! `word_embeddings → N × (RMSNorm → (KDA | MLA) → +res → RMSNorm → (MLP | MoE)
//! → +res) → RMSNorm → lm_head`.
//!
//! Two interleavings run at once: attention alternates KDA / MLA on a
//! `layer_group_size` cycle, and the FFN is dense for the first
//! `first_k_dense_replace` layers and MoE after. For Ling-3.0-tiny that is 18 KDA
//! + 6 MLA layers (MLA at 3/7/11/15/19/23) and one dense MLP (layer 0).
//!
//! Attention weights live under `model.layers.{i}.attention` (Bailing naming),
//! not `.self_attn`.

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_deepseek::moe::{DeepseekMoeDims, emit_deepseek_moe};
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::{AttnKind, LingConfig};
use crate::kda::{KdaDims, emit_kda_attention};
use crate::mla::{MlaDims, ROPE_COS, ROPE_SIN, emit_mla_attention};

/// Bailing's token-embedding tensor name.
pub const EMBED_KEY: &str = "model.word_embeddings.weight";

use crate::quant::{Quant, QuantPlan, linear};

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

fn dense_mlp(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId, q: Quant) -> Result<HirNodeId> {
    let gate = linear(emit, &format!("{prefix}.gate_proj"), x, q)?;
    let up = linear(emit, &format!("{prefix}.up_proj"), x, q)?;
    let swiglu = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.silu(gate);
        gb.mul(a, up)
    };
    linear(emit, &format!("{prefix}.down_proj"), swiglu, q)
}

/// Build the Ling 3.0 text prefill graph for a fixed `seq`.
///
/// `weights` must already have been through [`crate::weights::prepare_checkpoint`]
/// when it came from a stock HF checkpoint.
///
/// Inputs: `input_ids [1, seq]`, `rope_cos`/`rope_sin` `[seq, qk_rope_head_dim/2]`
/// (see [`LingConfig::rope_tables`]). The RoPE tables are only consumed by the MLA
/// layers; KDA carries no positional encoding.
pub fn build_ling_text_flow(
    cfg: &LingConfig,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    build_ling_text_flow_quant(cfg, weights, seq, with_lm_head, Quant::F32)
}

/// [`build_ling_text_flow`] at a chosen weight precision.
///
/// With [`Quant::MXFP4`] the projections are quantized host-side at build time
/// and the routed experts become `Op::DequantGroupedMatMulMlx` params declared
/// by name — their bytes are uploaded after compile by
/// [`crate::streaming::stream_expert_banks`], so `weights` must NOT carry the
/// stacked f32 banks (skip [`crate::weights::prepare_checkpoint`]; use
/// [`crate::streaming::load_and_compile`]).
pub fn build_ling_text_flow_quant(
    cfg: &LingConfig,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
    quant: Quant,
) -> Result<BuiltModel> {
    build_ling_text_flow_plan(cfg, weights, seq, with_lm_head, quant.into())
}

/// [`build_ling_text_flow_quant`] with the LM head's precision chosen
/// separately — see [`QuantPlan`].
pub fn build_ling_text_flow_plan(
    cfg: &LingConfig,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
    plan: QuantPlan,
) -> Result<BuiltModel> {
    cfg.validate()?;
    let quant = plan.proj;
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let half = cfg.qk_rope_head_dim / 2;

    let mla = MlaDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        q_lora_rank: cfg.q_lora_rank,
        kv_lora_rank: cfg.kv_lora_rank,
        qk_nope_head_dim: cfg.qk_nope_head_dim,
        qk_rope_head_dim: cfg.qk_rope_head_dim,
        v_head_dim: cfg.v_head_dim,
        gate: cfg.attn_gate(),
        eps,
        seq,
        quant,
    };
    let kda = KdaDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        head_dim: cfg.head_dim,
        conv_kernel: cfg.short_conv_kernel_size,
        no_lora: cfg.no_kda_lora,
        lower_bound: cfg.kda_lower_bound,
        eps,
        seq,
        quant,
    };
    let moe = DeepseekMoeDims {
        hidden,
        moe_inter: cfg.moe_intermediate_size,
        n_routed: cfg.num_experts,
        top_k: cfg.num_experts_per_tok,
        n_group: cfg.n_group,
        topk_group: cfg.topk_group,
        routed_scaling: cfg.routed_scaling_factor,
        shared_inter: cfg.shared_intermediate_size(),
        seq,
        // MXFP4 wants the stock `[E,N,K]` order and overrides this.
        experts_pretransposed: true,
        mxfp4_group: quant.group_size().map(|g| g as u32),
    };

    let mut flow = ModelFlow::new("ling3")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, seq], f))
        .input(ROPE_COS, Shape::new(&[seq, half], f))
        .input(ROPE_SIN, Shape::new(&[seq, half], f))
        .zero_beta_named("ling.zero_beta.hidden", hidden)
        // Bailing calls the embedding table `word_embeddings`, not `embed_tokens`.
        .embed(EMBED_KEY);

    let hs = Shape::new(&[1, seq, hidden], f);
    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let kind = cfg.attn_kind(i);
        let is_moe = cfg.is_moe_layer(i);
        let hs = hs.clone();
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("layer{i} needs a hidden input"))?
                .hir_id();
            let normed = rmsnorm(emit, &format!("{prefix}.input_layernorm"), x, hidden, eps)?;
            let attn_prefix = format!("{prefix}.attention");
            let attn = match kind {
                AttnKind::Mla => emit_mla_attention(emit, &attn_prefix, normed, mla)?,
                AttnKind::Kda => emit_kda_attention(emit, &attn_prefix, normed, kda)?,
            };
            let h = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, attn)
            };
            let normed2 = rmsnorm(
                emit,
                &format!("{prefix}.post_attention_layernorm"),
                h,
                hidden,
                eps,
            )?;
            let ffn = if is_moe {
                emit_deepseek_moe(emit, &format!("{prefix}.mlp"), normed2, moe)?
            } else {
                dense_mlp(emit, &format!("{prefix}.mlp"), normed2, quant)?
            };
            let out = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(h, ffn)
            };
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    flow = flow.final_norm(eps);
    let vocab = cfg.vocab_size;
    let tied = cfg.tie_word_embeddings;
    let flow = if !with_lm_head {
        flow.output("hidden")
    } else if tied || plan.lm_head == Quant::F32 {
        // Tied heads reuse the f32 embedding table, which has no MXFP4 form
        // (it is gathered, not multiplied) — quantizing here would need a
        // second copy of it, which is the opposite of the point.
        flow.lm_head(vocab, hidden, tied).output("logits")
    } else {
        // 157184 x 1536 = 0.97 GB f32, the single largest matmul weight in the
        // model and ~a third of decode's per-token traffic → 0.12 GB packed.
        flow.plugin_named("lm_head", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("lm_head needs a hidden input"))?
                .hir_id();
            let y = linear(emit, "lm_head", x, plan.lm_head)?;
            Ok(Some(emit.wrap(y, Shape::new(&[1, seq, vocab], f))))
        })
        .output("logits")
    };
    flow.build_with(&mut WeightMapSource(weights), None)
}
