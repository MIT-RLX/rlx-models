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

//! MiniMax-M3 single-token **decode** graph with a KV cache.
//!
//! One query token at absolute position `p = past_len` attends to the cached
//! (post-RoPE) keys/values plus its own fresh K/V. For sparse layers the MSA
//! indexer runs over the cached (post-RoPE) index keys — the fast-path spirit of
//! llama.cpp #24908, expressed with the same ops as prefill but `seq_q = 1` and
//! no causal masking (the lone query is the newest position, so every cached key
//! is already causally valid). Fresh per-layer `k`/`v` (and `idx_k` on sparse
//! layers) are exported as side outputs for the runner to append to the cache.
//!
//! Output order: `[logits, (k_new_i, v_new_i, [idx_k_new_i])* per layer]`.

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow, SideOutputs};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{CmpOp, MaskKind, Op, PadMode, ReduceOp};
use rlx_ir::{DType, HirGraphExt, HirNodeId, RopeStyle, Shape};
use std::sync::{Arc, Mutex};

use super::config::MiniMaxM3Config;
use super::mlp::emit_dense_mlp;
use super::moe::{M3MoeDims, emit_m3_moe};
use super::ops::{gemma_rmsnorm, per_head_gemma_rmsnorm};
use super::{ROPE_COS, ROPE_SIN};

const BIG: f32 = 1e30;

type Sink = Arc<Mutex<Vec<HirNodeId>>>;

fn repeat_kv(
    g: &mut HirMut,
    x: HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> HirNodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

/// MSA additive bias `[1, num_heads, 1, klen]` for the decode query over the
/// cached index keys. Pushes the fresh `idx_k` row to `sink`.
fn emit_decode_msa_bias(
    emit: &mut Emit<'_>,
    prefix: &str,
    normed: HirNodeId,
    cfg: &MiniMaxM3Config,
    layer_idx: usize,
    past_len: usize,
    sink: &Sink,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let ih = cfg.sparse.index_n_heads;
    let id = cfg.sparse.index_head_dim;
    let block = cfg.sparse.block_size;
    let num_heads = cfg.num_attention_heads;
    let n_rot = cfg.n_rot();
    let eps = cfg.rms_norm_eps;
    let group = num_heads / ih;

    let klen = past_len + 1;
    let n_blk = klen.div_ceil(block);
    let kpad = n_blk * block;
    let eff_k = cfg.sparse.topk_blocks.min(n_blk).max(1);
    let qb = past_len / block;

    // Local-block force-include (+BIG) for the query's own block band.
    let mut lb = vec![0f32; n_blk];
    for l in 0..cfg.sparse.local_blocks {
        lb[qb.saturating_sub(l)] = BIG;
    }
    let local_boost = emit.synth_param(
        &format!("{prefix}.dec_local_boost"),
        lb,
        Shape::new(&[1, n_blk], f),
    );
    let zeros = emit.synth_param(
        &format!("{prefix}.dec_zeros"),
        vec![0.0; n_blk],
        Shape::new(&[1, n_blk], f),
    );
    let negbig = emit.synth_param(
        &format!("{prefix}.dec_negbig"),
        vec![-BIG; n_blk],
        Shape::new(&[1, n_blk], f),
    );

    let iq = {
        let w = emit.load_param(&format!("{prefix}.index_q_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };
    let ik_new = {
        let w = emit.load_param(&format!("{prefix}.index_k_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };
    let iq = per_head_gemma_rmsnorm(
        emit,
        &format!("{prefix}.index_q_norm"),
        iq,
        1,
        1,
        ih,
        id,
        eps,
    )?;
    let ik_new = per_head_gemma_rmsnorm(
        emit,
        &format!("{prefix}.index_k_norm"),
        ik_new,
        1,
        1,
        1,
        id,
        eps,
    )?;
    let cos = emit.flow_input(ROPE_COS)?.hir_id();
    let sin = emit.flow_input(ROPE_SIN)?.hir_id();
    let past_idxk = emit.flow_input(&format!("past_idxk_{layer_idx}"))?.hir_id();

    let mut gb = HirMut::new(emit.hir());
    let iq = gb.rope_n(iq, cos, sin, id, n_rot);
    let ik_new = gb.rope_n(ik_new, cos, sin, id, n_rot);
    sink.lock().expect("sink").push(ik_new);
    let idx_k_cache = gb.concat_(vec![past_idxk, ik_new], 1); // [1, klen, id]
    let ik2d = gb.reshape_(idx_k_cache, vec![klen as i64, id as i64]);
    let ik_t = gb.transpose_(ik2d, vec![1, 0]); // [id, klen]

    let mut head_biases: Vec<HirNodeId> = Vec::with_capacity(num_heads);
    for h in 0..ih {
        let iq_h = gb.narrow_(iq, 2, h * id, id); // [1,1,id]
        let iq_h2d = gb.reshape_(iq_h, vec![1, id as i64]);
        let score = gb.mm(iq_h2d, ik_t); // [1, klen]
        let score_p = if kpad > klen {
            gb.pad_(
                score,
                vec![[0, 0], [0, kpad - klen]],
                PadMode::Constant(-BIG),
            )
        } else {
            score
        };
        let blk = gb.reshape_(score_p, vec![1, n_blk as i64, block as i64]);
        let blkmax = gb.add_node(
            Op::Reduce {
                op: ReduceOp::Max,
                axes: vec![2],
                keep_dim: false,
            },
            vec![blk],
            Shape::new(&[1, n_blk], f),
        );
        let blkmax = gb.add(blkmax, local_boost);
        let top_idx = gb.add_node(
            Op::TopK { k: eff_k },
            vec![blkmax],
            Shape::new(&[1, eff_k], f),
        );
        let top_val = gb.add_node(
            Op::GatherElements { axis: 1 },
            vec![blkmax, top_idx],
            Shape::new(&[1, eff_k], f),
        );
        let thresh = gb.narrow_(top_val, 1, eff_k - 1, 1); // [1,1]
        let thresh = gb.add_node(
            Op::Expand {
                target_shape: vec![1, n_blk as i64],
            },
            vec![thresh],
            Shape::new(&[1, n_blk], f),
        );
        let keep = gb.add_node(
            Op::Compare(CmpOp::Ge),
            vec![blkmax, thresh],
            Shape::new(&[1, n_blk], DType::Bool),
        );
        let keep_add = gb.add_node(
            Op::Where,
            vec![keep, zeros, negbig],
            Shape::new(&[1, n_blk], f),
        );
        let ka3 = gb.reshape_(keep_add, vec![1, n_blk as i64, 1]);
        let ka3 = gb.add_node(
            Op::Expand {
                target_shape: vec![1, n_blk as i64, block as i64],
            },
            vec![ka3],
            Shape::new(&[1, n_blk, block], f),
        );
        let kt = gb.reshape_(ka3, vec![1, kpad as i64]);
        let kt = if kpad > klen {
            gb.narrow_(kt, 1, 0, klen)
        } else {
            kt
        };
        let head_bias4 = gb.reshape_(kt, vec![1, 1, 1, klen as i64]);
        for _ in 0..group {
            head_biases.push(head_bias4);
        }
    }
    Ok(gb.concat_(head_biases, 1)) // [1, num_heads, 1, klen]
}

/// Decode self-attention for one layer; pushes `k_new`, `v_new` (and `idx_k_new`
/// on sparse layers) to `sink`. Returns `[1, 1, hidden]`.
fn emit_m3_decode_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    normed: HirNodeId,
    cfg: &MiniMaxM3Config,
    layer_idx: usize,
    past_len: usize,
    sparse: bool,
    sink: &Sink,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let h = cfg.num_attention_heads;
    let kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim();
    let n_rot = cfg.n_rot();
    let eps = cfg.rms_norm_eps;
    let group = h / kv;
    let klen = past_len + 1;

    let q = {
        let w = emit.load_param(&format!("{prefix}.q_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };
    let k_new = {
        let w = emit.load_param(&format!("{prefix}.k_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };
    let v_new = {
        let w = emit.load_param(&format!("{prefix}.v_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };
    let q = per_head_gemma_rmsnorm(emit, &format!("{prefix}.q_norm"), q, 1, 1, h, hd, eps)?;
    let k_new =
        per_head_gemma_rmsnorm(emit, &format!("{prefix}.k_norm"), k_new, 1, 1, kv, hd, eps)?;
    let cos = emit.flow_input(ROPE_COS)?.hir_id();
    let sin = emit.flow_input(ROPE_SIN)?.hir_id();
    let past_k = emit.flow_input(&format!("past_k_{layer_idx}"))?.hir_id();
    let past_v = emit.flow_input(&format!("past_v_{layer_idx}"))?.hir_id();

    let (q, k_cache, v_cache) = {
        let mut gb = HirMut::new(emit.hir());
        let q = gb.rope_n_styled(q, cos, sin, hd, n_rot, RopeStyle::NeoX);
        let k_new = gb.rope_n_styled(k_new, cos, sin, hd, n_rot, RopeStyle::NeoX);
        // Export the fresh (post-RoPE) K and (raw) V rows.
        {
            let mut s = sink.lock().expect("sink");
            s.push(k_new);
            s.push(v_new);
        }
        let k_cache = gb.concat_(vec![past_k, k_new], 1); // [1, klen, kv*hd]
        let v_cache = gb.concat_(vec![past_v, v_new], 1);
        (q, k_cache, v_cache)
    };

    let out_shape = Shape::new(&[1, 1, h * hd], f);
    let attn = if sparse {
        let bias = emit_decode_msa_bias(emit, prefix, normed, cfg, layer_idx, past_len, sink)?;
        let mut gb = HirMut::new(emit.hir());
        let k_rep = repeat_kv(&mut gb, k_cache, kv, hd, group);
        let v_rep = repeat_kv(&mut gb, v_cache, kv, hd, group);
        gb.attention_bias(q, k_rep, v_rep, bias, h, hd, out_shape)
    } else {
        let mut gb = HirMut::new(emit.hir());
        let k_rep = repeat_kv(&mut gb, k_cache, kv, hd, group);
        let v_rep = repeat_kv(&mut gb, v_cache, kv, hd, group);
        // Single newest query → every cached key (positions 0..=past_len) is valid.
        let _ = klen;
        gb.attention_kind(q, k_rep, v_rep, h, hd, MaskKind::None, out_shape)
    };

    let o_w = emit.load_param(&format!("{prefix}.o_proj.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(attn, o_w))
}

/// Per-layer side-output descriptor: `(k_idx, v_idx, Option<idxk_idx>)` into the
/// decode graph's output vector (index 0 is logits).
pub type DecodeOutputLayout = Vec<(usize, usize, Option<usize>)>;

/// Build the decode graph for a fixed `past_len`, plus the side-output layout.
pub fn build_m3_decode_flow(
    cfg: &MiniMaxM3Config,
    weights: &mut WeightMap,
    past_len: usize,
) -> Result<(BuiltModel, DecodeOutputLayout)> {
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let half = cfg.n_rot() / 2;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
    let id = cfg.sparse.index_head_dim;

    let mut flow = ModelFlow::new("minimax_m3_decode")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, 1], f))
        .input(ROPE_COS, Shape::new(&[1, half], f))
        .input(ROPE_SIN, Shape::new(&[1, half], f));
    for i in 0..cfg.num_hidden_layers {
        flow = flow.input(format!("past_k_{i}"), Shape::new(&[1, past_len, kv_dim], f));
        flow = flow.input(format!("past_v_{i}"), Shape::new(&[1, past_len, kv_dim], f));
        if cfg.is_sparse_layer(i) {
            flow = flow.input(format!("past_idxk_{i}"), Shape::new(&[1, past_len, id], f));
        }
    }
    flow = flow.token_embed();

    // Side-output order across layers (k, v, [idxk]); compute now.
    let mut layout: DecodeOutputLayout = Vec::with_capacity(cfg.num_hidden_layers);
    let mut next = 1usize; // out[0] = logits
    for i in 0..cfg.num_hidden_layers {
        let k = next;
        let v = next + 1;
        next += 2;
        let idxk = if cfg.is_sparse_layer(i) {
            let x = next;
            next += 1;
            Some(x)
        } else {
            None
        };
        layout.push((k, v, idxk));
    }

    let sink = SideOutputs::new();
    let sink_inner = sink.inner();
    let hs = Shape::new(&[1, 1, hidden], f);
    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let is_moe = cfg.is_moe_layer(i);
        let sparse = cfg.is_sparse_layer(i);
        let cfg = cfg.clone();
        let sink = sink_inner.clone();
        let hs = hs.clone();
        let moe = M3MoeDims {
            hidden,
            moe_inter: cfg.moe_intermediate_size,
            shared_inter: cfg.shared_inter(),
            n_routed: cfg.num_local_experts,
            top_k: cfg.num_experts_per_tok,
            routed_scaling: cfg.routed_scaling_factor,
            alpha: cfg.swiglu_alpha,
            limit: cfg.swiglu_limit,
            seq: 1,
        };
        let dense_inter = cfg.dense_intermediate_size;
        let (alpha, limit) = (cfg.swiglu_alpha, cfg.swiglu_limit);
        flow = flow.plugin_named(format!("dlayer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("dlayer{i} needs input"))?
                .hir_id();
            let normed = gemma_rmsnorm(emit, &format!("{prefix}.input_layernorm"), x, hidden, eps)?;
            let a = emit_m3_decode_attention(
                emit,
                &format!("{prefix}.self_attn"),
                normed,
                &cfg,
                i,
                past_len,
                sparse,
                &sink,
            )?;
            let h = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, a)
            };
            let normed2 = gemma_rmsnorm(
                emit,
                &format!("{prefix}.post_attention_layernorm"),
                h,
                hidden,
                eps,
            )?;
            let ffn = if is_moe {
                emit_m3_moe(emit, &format!("{prefix}.block_sparse_moe"), normed2, moe)?
            } else {
                emit_dense_mlp(
                    emit,
                    &format!("{prefix}.mlp"),
                    normed2,
                    1,
                    hidden,
                    dense_inter,
                    alpha,
                    limit,
                )?
            };
            let out = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(h, ffn)
            };
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    let vocab = cfg.vocab_size;
    let tie = cfg.tie_word_embeddings;
    flow = flow.plugin_named("lm_head", move |emit, prev| {
        let h = prev.ok_or_else(|| anyhow!("lm_head needs input"))?.hir_id();
        let normed = gemma_rmsnorm(emit, "model.norm", h, hidden, eps)?;
        let key = if tie {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        let lm_w = emit.load_param(key, true)?;
        let mut gb = HirMut::new(emit.hir());
        let logits = gb.mm(normed, lm_w);
        Ok(Some(emit.wrap(logits, Shape::new(&[1, 1, vocab], f))))
    });
    flow = flow.output("logits");

    let built = flow.build_with(&mut WeightMapSource(weights), None)?;
    let built = built.with_extra_hir_outputs(sink.drain());
    Ok((built, layout))
}
