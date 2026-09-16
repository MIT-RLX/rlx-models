// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1 DSpark** — the integrated speculative-decoding draft head.
//!
//! `n_mtp_layers` extra blocks (`mtp.N.*` in the checkpoint) run over a short
//! block of draft positions, conditioned on the main model's hidden state. They
//! reuse the backbone's embedding and LM head, route over their own smaller
//! expert bank (`dspark_n_routed_experts`, 128/top-3 in the GA checkpoint), and
//! read the main model through `main_hidden` — the mean over the
//! Hyper-Connection copies of the stream entering each `dspark_target_layer_ids`
//! layer.
//!
//! Three things make DSpark attention unlike the backbone's:
//!
//! * Its sliding-window cache is filled from the **main** model's stream
//!   (`wkv(main_x)`), not from its own draft tokens. Prefill therefore only
//!   *seeds* that cache and produces no draft.
//! * Within a step every draft position attends to **every** draft key — the
//!   block is not causal internally (`get_dspark_topk_idxs` hands the same index
//!   list to all `block_size` queries).
//! * The draft positions sit at `start_pos + 1 ..`, one past the main token.
//!
//! What this module builds is the *forward pass* of the draft head, which is
//! exactly what the released `inference/model.py` implements — the reference
//! notes that the accept/reject loop around it is out of scope there too, and
//! drafting is not lossless, so the verification policy belongs to the caller.
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/model.py`
//! (`DSparkAttention`, `DSparkBlock`, `Transformer.forward_spec`).

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_graph::{
    V41Inputs, build_v41_moe, hc_mixes, hc_post, hc_reduce, identity_pre_mix, rope_table,
};
use crate::standard_decoder::{
    build_dspark_confidence_head, build_dspark_markov_head, build_v4_o_lora,
    build_v4_sink_attention, emit_proj, load_dense_dequant, load_norm, load_p, load_proj,
    load_transposed_param, load_v4_wo_a, rope_tail, synth_const, synth_zero,
};
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Graph input / output names for the DSpark caches.
pub mod names {
    /// Main-model window KV held by DSpark stage `stage`: `[cache_len, head_dim]`.
    pub fn window_kv(stage: usize) -> String {
        format!("dspark.winkv.{stage}")
    }
    /// The main-model latent this step appends: `[1, head_dim]`.
    pub fn window_kv_new(stage: usize) -> String {
        format!("dspark.kvnew.{stage}")
    }
}

/// Seed DSpark's window caches from a prefill.
///
/// At `start_pos == 0` the reference's `DSparkAttention.forward` computes
/// `kv_norm(wkv(main_x))` over the whole prompt, writes it into the ring and
/// returns the draft stream untouched — no draft is produced. This builds only
/// that computation: `main_hidden [seq, dim · n_targets]` in, one
/// `[keep, head_dim]` cache per stage out, where `keep = min(seq, window_size)`
/// are the most recent positions the ring retains.
///
/// Returns the graph, its parameters, and the per-stage output names.
pub fn build_v41_dspark_seed(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    if spec.n_mtp_layers == 0 {
        return Err(anyhow!("deepseek_v41: this checkpoint has no DSpark stages"));
    }
    let mut g = Graph::new("deepseek_v41_dspark_seed");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let (d, hd) = (spec.dim, spec.head_dim);
    let rd = spec.rope_head_dim & !1;
    let eps = spec.rms_norm_eps;
    let n_targets = spec.dspark_target_layer_ids.len();
    let keep = seq.min(spec.window_size);
    let zb_d = synth_zero(&mut g, &mut params, "v41s.zb.d", d);
    let zb_hd = synth_zero(&mut g, &mut params, "v41s.zb.hd", hd);

    let main_hidden = g.input("main_hidden", Shape::new(&[seq, d * n_targets], f));
    let main_x = build_main_x(&mut g, &mut params, packed, weights, spec, main_hidden, seq)?;

    // DSpark stages have `compress_ratio == 0`, so the plain `rope_theta` table
    // with no YaRN is the right one — same choice the backbone makes for a
    // sliding-window-only layer.
    let positions: Vec<usize> = (0..seq).collect();
    let (cos, sin, _) = rope_table(
        &mut g,
        &mut params,
        &positions,
        rd,
        spec.rope_theta,
        None,
        "s.win",
    );

    let mut outs = Vec::new();
    let mut names_out = Vec::new();
    for stage in 0..spec.n_mtp_layers {
        let lp = spec.layer_prefix(spec.n_layers + stage);
        let wkv = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wkv.weight"),
        )?;
        let kv = emit_proj(&mut g, main_x, &wkv, Shape::new(&[seq, hd], f));
        let kv_norm = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.kv_norm.weight"),
            0.0,
        )?;
        let kv = g.rms_norm(kv, kv_norm, zb_hd, eps);
        let kv = rope_tail(&mut g, kv, cos, sin, seq, 1, hd, rd);
        // the ring only retains the last `window_size` positions
        let kept = if keep < seq {
            g.narrow_(kv, 0, seq - keep, keep)
        } else {
            kv
        };
        outs.push(kept);
        names_out.push(names::window_kv(stage));
    }
    let _ = zb_d;
    g.set_outputs(outs);
    Ok((g, params, names_out))
}

/// `main_norm(main_proj(main_hidden))` — stage 0 owns the projection that turns
/// the concatenated target-layer states into the DSpark stream's conditioning.
fn build_main_x(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    spec: &DeepseekV41Spec,
    main_hidden: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let f = DType::F32;
    let lp = spec.layer_prefix(spec.n_layers); // stage 0
    let proj = load_proj(g, params, packed, weights, &format!("{lp}.main_proj.weight"))?;
    let x = emit_proj(g, main_hidden, &proj, Shape::new(&[rows, spec.dim], f));
    let norm = load_norm(g, params, weights, &format!("{lp}.main_norm.weight"), 0.0)?;
    let zb = synth_zero(g, params, "v41s.mainx.zb", spec.dim);
    Ok(g.rms_norm(x, norm, zb, spec.rms_norm_eps))
}

/// One DSpark draft step at main-model position `pos`.
///
/// Inputs: `main_hidden [1, dim · n_targets]` and `draft_ids [1, block_size]`
/// (slot 0 is the accepted token, the rest the noise id), plus each stage's
/// window cache under [`names::window_kv`]. Outputs, in order:
///
/// 1. `logits [block_size, vocab]` — before the Markov bias,
/// 2. `hidden [block_size, dim]` — the collapsed stream the confidence head reads,
/// 3. one `[1, head_dim]` main-model latent per stage to append to its ring.
///
/// The Markov bias and the sampling loop are sequential by construction (each
/// draft token's bias depends on the previous one's sample), so they live in
/// [`build_v41_dspark_markov_step`] and the caller's loop rather than in this
/// graph.
pub fn build_v41_dspark_step(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    pos: usize,
    cache_len: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    if spec.n_mtp_layers == 0 || spec.dspark_block_size == 0 {
        return Err(anyhow!("deepseek_v41: this checkpoint has no DSpark stages"));
    }
    let mut g = Graph::new("deepseek_v41_dspark_step");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let (d, hc, nh, hd) = (spec.dim, spec.hc_mult, spec.n_heads, spec.head_dim);
    let ql = spec.q_lora_rank;
    let rd = spec.rope_head_dim & !1;
    let eps = spec.rms_norm_eps;
    let block = spec.dspark_block_size;
    let n_targets = spec.dspark_target_layer_ids.len();
    let zb_d = synth_zero(&mut g, &mut params, "v41k.zb.d", d);
    let zb_ql = synth_zero(&mut g, &mut params, "v41k.zb.ql", ql);
    let zb_hd = synth_zero(&mut g, &mut params, "v41k.zb.hd", hd);

    let main_hidden = g.input("main_hidden", Shape::new(&[1, d * n_targets], f));
    let main_x = build_main_x(&mut g, &mut params, packed, weights, spec, main_hidden, 1)?;

    // The main token sits at `pos`; the draft block starts one past it.
    let (cos_m, sin_m, _) = rope_table(
        &mut g,
        &mut params,
        &[pos],
        rd,
        spec.rope_theta,
        None,
        "k.main",
    );
    let draft_positions: Vec<usize> = (0..block).map(|i| pos + 1 + i).collect();
    let (cos_b, sin_b, sininv_b) = rope_table(
        &mut g,
        &mut params,
        &draft_positions,
        rd,
        spec.rope_theta,
        None,
        "k.blk",
    );

    let draft_ids = g.input("draft_ids", Shape::new(&[1, block], DType::I32));
    let (embed_w, _, _) = load_dense_dequant(&mut g, &mut params, weights, "embed.weight")?;
    let e0 = g.gather_(embed_w, draft_ids, 0);
    let e0 = g.reshape_(e0, vec![block as i64, 1, d as i64]);
    let ones_hc = synth_const(&mut g, &mut params, "v41k.hc.ones", vec![1f32; hc], &[1, hc, 1]);
    let mut h = g.mul(e0, ones_hc);
    let mut pre_mix = identity_pre_mix(&mut g, &mut params, block, hc);

    let mut kv_outs: Vec<(NodeId, String)> = Vec::new();

    for stage in 0..spec.n_mtp_layers {
        let il = spec.n_layers + stage;
        let lp = spec.layer_prefix(il);
        if spec.ratio(il) != 0 {
            return Err(anyhow!(
                "deepseek_v41: DSpark stage {stage} has compress_ratio {} (must be 0)",
                spec.ratio(il)
            ));
        }

        // ── attention sub-block ──
        let residual = h;
        let fn_a = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.hc_attn_fn"))?;
        let sc_a = load_p(&mut g, &mut params, weights, &format!("{lp}.hc_attn_scale"), false)?;
        let bs_a = load_p(&mut g, &mut params, weights, &format!("{lp}.hc_attn_base"), false)?;
        let (attn_pre, attn_post, attn_comb) = hc_mixes(
            &mut g,
            &mut params,
            h,
            fn_a,
            sc_a,
            bs_a,
            block,
            hc,
            d,
            eps,
            spec.hc_eps,
            spec.hc_mult_sinkhorn_iters,
            &format!("{lp}.ka"),
        );
        let xa = hc_reduce(&mut g, h, pre_mix, block, hc);
        let an = load_norm(&mut g, &mut params, weights, &format!("{lp}.attn_norm.weight"), 0.0)?;
        let xa = g.rms_norm(xa, an, zb_d, eps);

        let wq_a = load_proj(&mut g, &mut params, packed, weights, &format!("{lp}.attn.wq_a.weight"))?;
        let qr = emit_proj(&mut g, xa, &wq_a, Shape::new(&[block, ql], f));
        let q_norm = load_norm(&mut g, &mut params, weights, &format!("{lp}.attn.q_norm.weight"), 0.0)?;
        let qr = g.rms_norm(qr, q_norm, zb_ql, eps);
        let wq_b = load_proj(&mut g, &mut params, packed, weights, &format!("{lp}.attn.wq_b.weight"))?;
        let q = emit_proj(&mut g, qr, &wq_b, Shape::new(&[block, nh * hd], f));
        let q = rope_tail(&mut g, q, cos_b, sin_b, block, nh, hd, rd);

        // one `wkv` serves both the main-model latent and the draft keys
        let wkv = load_proj(&mut g, &mut params, packed, weights, &format!("{lp}.attn.wkv.weight"))?;
        let kv_norm = load_norm(&mut g, &mut params, weights, &format!("{lp}.attn.kv_norm.weight"), 0.0)?;

        let main_kv = emit_proj(&mut g, main_x, &wkv, Shape::new(&[1, hd], f));
        let main_kv = g.rms_norm(main_kv, kv_norm, zb_hd, eps);
        let main_kv = rope_tail(&mut g, main_kv, cos_m, sin_m, 1, 1, hd, rd);
        kv_outs.push((main_kv, names::window_kv_new(stage)));

        let draft_kv = emit_proj(&mut g, xa, &wkv, Shape::new(&[block, hd], f));
        let draft_kv = g.rms_norm(draft_kv, kv_norm, zb_hd, eps);
        let draft_kv = rope_tail(&mut g, draft_kv, cos_b, sin_b, block, 1, hd, rd);

        let window = if cache_len > 0 {
            let cached = g.input(names::window_kv(stage), Shape::new(&[cache_len, hd], f));
            g.concat_(vec![cached, main_kv], 0)
        } else {
            main_kv
        };
        let n_window = cache_len + 1;
        let kv_all = g.concat_(vec![window, draft_kv], 0);
        let n_keys = n_window + block;
        // Every key is visible to every draft query: the window slots are all
        // real, and the block is deliberately non-causal within itself.
        let mask = synth_const(
            &mut g,
            &mut params,
            &format!("{lp}.k.mask"),
            vec![0f32; block * n_keys],
            &[block, n_keys],
        );

        let sink = load_p(&mut g, &mut params, weights, &format!("{lp}.attn.attn_sink"), false)?;
        let q3 = g.reshape_(q, vec![block as i64, nh as i64, hd as i64]);
        let o = build_v4_sink_attention(
            &mut g,
            &mut params,
            q3,
            kv_all,
            mask,
            sink,
            (hd as f32).powf(-0.5),
            block,
            nh,
            hd,
            n_keys,
            &format!("{lp}.k"),
        );
        let o_flat = g.reshape_(o, vec![block as i64, (nh * hd) as i64]);
        let o_inv = rope_tail(&mut g, o_flat, cos_b, sininv_b, block, nh, hd, rd);
        let dpg = spec.dim_per_group();
        let wo_a = load_v4_wo_a(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_a.weight"),
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
        )?;
        let wo_b = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.attn.wo_b.weight"))?;
        let attn_out = build_v4_o_lora(
            &mut g, o_inv, wo_a, wo_b, block, spec.n_groups, spec.o_lora_rank, dpg, d,
        );
        h = hc_post(&mut g, attn_out, residual, attn_post, attn_comb, block, hc, d);

        // ── FFN sub-block (the stage's own, smaller expert bank) ──
        let residual = h;
        let fn_f = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.hc_ffn_fn"))?;
        let sc_f = load_p(&mut g, &mut params, weights, &format!("{lp}.hc_ffn_scale"), false)?;
        let bs_f = load_p(&mut g, &mut params, weights, &format!("{lp}.hc_ffn_base"), false)?;
        let (ffn_pre, ffn_post, ffn_comb) = hc_mixes(
            &mut g,
            &mut params,
            h,
            fn_f,
            sc_f,
            bs_f,
            block,
            hc,
            d,
            eps,
            spec.hc_eps,
            spec.hc_mult_sinkhorn_iters,
            &format!("{lp}.kf"),
        );
        let xf = hc_reduce(&mut g, h, attn_pre, block, hc);
        let fnorm = load_norm(&mut g, &mut params, weights, &format!("{lp}.ffn_norm.weight"), 0.0)?;
        let xf = g.rms_norm(xf, fnorm, zb_d, eps);
        let ffn_out = build_v41_moe(&mut g, &mut params, packed, weights, spec, il, xf, block, None)?;
        h = hc_post(&mut g, ffn_out, residual, ffn_post, ffn_comb, block, hc, d);
        pre_mix = ffn_pre;
    }

    // ── head: collapse, norm, project through the backbone's LM head ──
    let last = spec.layer_prefix(spec.n_layers + spec.n_mtp_layers - 1);
    let x = hc_reduce(&mut g, h, pre_mix, block, hc);
    let norm = load_norm(&mut g, &mut params, weights, &format!("{last}.norm.weight"), 0.0)?;
    let xn = g.rms_norm(x, norm, zb_d, eps);
    let head = load_proj(&mut g, &mut params, packed, weights, "head.weight")?;
    let logits = emit_proj(&mut g, xn, &head, Shape::new(&[block, spec.vocab_size], f));
    let logits = g.reshape_(logits, vec![block as i64, spec.vocab_size as i64]);

    let mut outs = vec![logits, x];
    let mut names_out = vec!["logits".to_string(), "hidden".to_string()];
    for (n, name) in kv_outs {
        outs.push(n);
        names_out.push(name);
    }
    g.set_outputs(outs);
    Ok((g, params, names_out))
}

/// The Markov bias and confidence heads, as one graph over the whole draft block.
///
/// Inputs: `token_ids [block_size]` (the token *preceding* each draft slot, i.e.
/// `[accepted, draft_0, …]`) and `hidden [block_size, dim]`. Outputs
/// `bias [block_size, vocab]` and `confidence [block_size]`.
///
/// The reference interleaves this with sampling — draft slot `i`'s bias depends
/// on slot `i-1`'s sampled token — so a caller doing true sequential sampling
/// runs this one row at a time (`block_size = 1`). Running it over the whole
/// block at once is exact when the caller already knows the token sequence, which
/// is the case when re-scoring or verifying.
pub fn build_v41_dspark_markov_step(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    rows: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("deepseek_v41_dspark_markov");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let last = spec.layer_prefix(spec.n_layers + spec.n_mtp_layers.max(1) - 1);
    let rank = spec.dspark_markov_rank;

    let token_ids = g.input("token_ids", Shape::new(&[rows], DType::I32));
    let hidden = g.input("hidden", Shape::new(&[rows, spec.dim], f));
    let m_embed = load_p(
        &mut g,
        &mut params,
        weights,
        &format!("{last}.markov_head.embed.weight"),
        false,
    )?;
    let m_head = load_p(
        &mut g,
        &mut params,
        weights,
        &format!("{last}.markov_head.head.weight"),
        true,
    )?;
    let (bias, embed) = build_dspark_markov_head(
        &mut g,
        m_embed,
        m_head,
        token_ids,
        rows,
        rank,
        spec.vocab_size,
    );
    let proj = load_p(
        &mut g,
        &mut params,
        weights,
        &format!("{last}.confidence_head.proj.weight"),
        true,
    )?;
    let conf = build_dspark_confidence_head(&mut g, hidden, embed, proj, rows);
    g.set_outputs(vec![bias, conf]);
    Ok((g, params))
}

/// The draft token ids for one step: slot 0 is the accepted token, the rest the
/// noise id (`DSparkBlock.forward_embed`).
pub fn dspark_draft_ids(spec: &DeepseekV41Spec, accepted: i32) -> Vec<i32> {
    let mut v = vec![spec.dspark_noise_token_id as i32; spec.dspark_block_size];
    if let Some(first) = v.first_mut() {
        *first = accepted;
    }
    v
}

/// A prefill's `V41Inputs` with `main_hidden` collection switched on — DSpark
/// cannot be seeded without it.
pub fn dspark_prefill_inputs(base: &V41Inputs) -> V41Inputs {
    V41Inputs {
        emit_main_hidden: true,
        ..base.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DSpark guard must fire before any tensor is requested.
    struct NoWeights;
    impl WeightLoader for NoWeights {
        fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            Err(anyhow!("unexpected tensor read: {k}"))
        }
        fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            self.take(k)
        }
        fn len(&self) -> usize {
            0
        }
        fn remaining_keys(&self) -> Vec<String> {
            Vec::new()
        }
    }

    fn spec() -> DeepseekV41Spec {
        DeepseekV41Spec::from_config(&serde_json::json!({
            "vocab_size": 64, "dim": 32, "num_hidden_layers": 6, "head_dim": 32,
            "num_attention_heads": 2, "o_lora_rank": 8, "n_routed_experts": 4,
            "moe_intermediate_size": 16, "o_groups": 2, "q_lora_rank": 16, "num_experts_per_tok": 2,
            "rope_head_dim": 16, "sliding_window": 4,
            "compress_ratios": [0, 0, 2, 2, 1, 1, 0, 0],
            "kv_source_layers": [2, 4], "index_source_layers": [2, 4, 5],
            "index_n_heads": 2, "index_head_dim": 32, "index_topk": 3,
            "n_mtp_layers": 2, "dspark_block_size": 3, "dspark_noise_token_id": 5,
            "dspark_target_layer_ids": [4, 5], "dspark_markov_rank": 16,
            "dspark_n_routed_experts": 2, "dspark_n_activated_experts": 2,
        }))
        .unwrap()
    }

    #[test]
    fn draft_ids_put_the_accepted_token_first() {
        let s = spec();
        assert_eq!(dspark_draft_ids(&s, 41), vec![41, 5, 5]);
    }

    #[test]
    fn stages_live_under_the_mtp_prefix_with_their_own_bank() {
        let s = spec();
        assert_eq!(s.layer_prefix(6), "mtp.0");
        assert_eq!(s.layer_prefix(7), "mtp.1");
        assert_eq!(s.moe_dims(5), (4, 2), "backbone bank");
        assert_eq!(s.moe_dims(6), (2, 2), "DSpark bank");
        // stages must be sliding-window only
        assert_eq!(s.ratio(6), 0);
        assert_eq!(s.ratio(7), 0);
    }

    #[test]
    fn seed_rejects_a_checkpoint_without_dspark() {
        let mut s = spec();
        s.n_mtp_layers = 0;
        let mut w = NoWeights;
        let mut packed = HashMap::new();
        let err = build_v41_dspark_seed(&s, &mut w, 4, &mut packed)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no DSpark stages"), "{err}");
    }
}
