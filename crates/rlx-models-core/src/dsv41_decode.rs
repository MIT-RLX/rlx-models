// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1** single-token decode step with a KV cache.
//!
//! Re-running the prefill for every token is O(n²); this is the O(1)-per-token
//! path. The graph takes the new token id plus the cached state and returns
//! `logits [1, vocab]` together with the pieces the host appends to the cache.
//!
//! Three kinds of state cross a step, and they are not symmetric:
//!
//! * **Sliding-window KV** — one roped latent per position, per layer. Every
//!   layer owns its own.
//! * **Compressed KV and index keys** — owned only by `kv_source_layers` and read
//!   by everything up to the next source, so the cache is keyed by *source*
//!   layer, not by consumer.
//! * **The compressor's partial group** — a `ratio > 1` layer emits one latent
//!   per `ratio` tokens, so on the other `ratio - 1` steps it has nothing to
//!   append and instead hands back the running `wkv`/`wgate` pair to accumulate.
//!
//! The Hyper-Connection pre-mix is *not* cross-step state: it is threaded within
//! one forward and restarts from the one-hot mix for each token, exactly as in
//! prefill. Engram look-back *is* cross-step, and the host supplies it by passing
//! the earlier tokens as `history` to
//! [`EngramHashPlan::hash_ids`](crate::dsv41_engram::EngramHashPlan::hash_ids).
//!
//! Correctness is the standard KV-cache induction: fed the same tokens, the
//! accumulated cache equals what prefill computes internally, so decode logits
//! equal prefill logits — which is what `decode_matches_prefill` asserts.

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_engram::load_and_build_v41_engram;
use crate::dsv41_graph::{
    NEG, V41Inputs, build_v41_index_score, build_v41_moe, compressed_causal_mask, exact_topk_mask,
    hc_mean, hc_mixes, hc_post, hc_reduce, identity_pre_mix, main_hidden_node, rope_table,
};
use crate::standard_decoder::{
    build_v4_o_lora, build_v4_sink_attention, emit_proj, load_dense_dequant, load_norm, load_p,
    load_proj, load_transposed_param, load_v4_wo_a, rope_tail, synth_const, synth_zero,
};
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Names of the per-step cache inputs and outputs, so a host can wire buffers
/// without re-deriving the layout.
pub mod names {
    /// Sliding-window KV cache for layer `il`: `[cache_len, head_dim]`.
    pub fn window_kv(il: usize) -> String {
        format!("winkv.{il}")
    }
    /// The new roped window latent to append: `[1, head_dim]`.
    pub fn window_kv_new(il: usize) -> String {
        format!("kvnew.{il}")
    }
    /// Compressed KV owned by source layer `src`: `[compress_len, head_dim]`.
    pub fn compress_kv(src: usize) -> String {
        format!("compkv.{src}")
    }
    /// A newly completed compressed latent to append: `[1, head_dim]`.
    pub fn compress_kv_new(src: usize) -> String {
        format!("compnew.{src}")
    }
    /// Index keys owned by source layer `src`: `[compress_len, index_head_dim]`.
    pub fn index_k(src: usize) -> String {
        format!("indexk.{src}")
    }
    /// A newly completed index key to append: `[1, index_head_dim]`.
    pub fn index_k_new(src: usize) -> String {
        format!("indexknew.{src}")
    }
    /// The compressor's accumulated group so far: `[filled, head_dim]` each.
    pub fn group_kv(src: usize) -> String {
        format!("groupkv.{src}")
    }
    pub fn group_score(src: usize) -> String {
        format!("groupscore.{src}")
    }
    /// This step's contribution to a still-incomplete group: `[1, head_dim]`.
    pub fn group_kv_new(src: usize) -> String {
        format!("groupkvnew.{src}")
    }
    pub fn group_score_new(src: usize) -> String {
        format!("groupscorenew.{src}")
    }
}

/// Per-source compressed-cache geometry for one decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressStep {
    pub source_layer: usize,
    pub ratio: usize,
    /// Latents already in the cache when the step begins (`pos / ratio`).
    pub len_before: usize,
    /// Latents visible to this step's query (`(pos + 1) / ratio`).
    pub len_after: usize,
    /// Whether this step completes a group and emits a new latent.
    pub fires: bool,
    /// Entries of the partial group the host carries in (`pos % ratio`).
    pub group_filled: usize,
}

impl CompressStep {
    fn new(source_layer: usize, ratio: usize, pos: usize) -> Self {
        CompressStep {
            source_layer,
            ratio,
            len_before: pos / ratio,
            len_after: (pos + 1) / ratio,
            fires: (pos + 1).is_multiple_of(ratio),
            group_filled: pos % ratio,
        }
    }
}

/// The cache geometry a step needs, derived from `pos` alone — the host uses it
/// to size buffers before building the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V41DecodePlan {
    pub pos: usize,
    /// Window entries the host must supply. One fewer than the window, because
    /// this step's own token takes the last slot: the ring holds `window_size`
    /// positions *including* the new one, so feeding `min(pos, window_size)`
    /// would let the query see one position further back than prefill does.
    pub cache_len: usize,
    /// One entry per distinct `kv_source_layer` reachable at this position.
    pub sources: Vec<CompressStep>,
}

impl V41DecodePlan {
    pub fn new(spec: &DeepseekV41Spec, pos: usize) -> Self {
        let mut sources: Vec<CompressStep> = Vec::new();
        for il in 0..spec.n_layers {
            if !spec.is_kv_source(il) {
                continue;
            }
            let ratio = spec.ratio(il);
            if ratio == 0 {
                continue;
            }
            sources.push(CompressStep::new(il, ratio, pos));
        }
        V41DecodePlan {
            pos,
            cache_len: pos.min(spec.window_size.saturating_sub(1)),
            sources,
        }
    }

    fn for_source(&self, src: usize) -> Option<&CompressStep> {
        self.sources.iter().find(|s| s.source_layer == src)
    }
}

/// Host-side KV cache for a decode loop.
///
/// Holds exactly the three kinds of state [`build_deepseek_v41_decode`] expects,
/// and knows how to feed a step and fold its outputs back in. The window is a
/// bounded rolling buffer; the compressed caches grow.
pub struct V41DecodeCache {
    window_size: usize,
    head_dim: usize,
    index_head_dim: usize,
    /// `[il] -> rows of head_dim`, most recent last, at most `window_size - 1`.
    window: Vec<Vec<f32>>,
    compress: HashMap<usize, Vec<f32>>,
    index_k: HashMap<usize, Vec<f32>>,
    group_kv: HashMap<usize, Vec<f32>>,
    group_score: HashMap<usize, Vec<f32>>,
}

impl V41DecodeCache {
    pub fn new(spec: &DeepseekV41Spec) -> Self {
        V41DecodeCache {
            window_size: spec.window_size,
            head_dim: spec.head_dim,
            index_head_dim: spec.index_head_dim,
            window: vec![Vec::new(); spec.n_layers],
            compress: HashMap::new(),
            index_k: HashMap::new(),
            group_kv: HashMap::new(),
            group_score: HashMap::new(),
        }
    }

    /// Named input buffers for the step described by `plan`, ready to hand to
    /// `Session::run` alongside `input_ids`.
    pub fn step_inputs(&self, plan: &V41DecodePlan) -> Vec<(String, &[f32])> {
        let mut out: Vec<(String, &[f32])> = Vec::new();
        if plan.cache_len > 0 {
            for (il, rows) in self.window.iter().enumerate() {
                debug_assert_eq!(rows.len(), plan.cache_len * self.head_dim);
                out.push((names::window_kv(il), rows.as_slice()));
            }
        }
        for s in &plan.sources {
            let src = s.source_layer;
            if s.len_before > 0 {
                if let Some(c) = self.compress.get(&src) {
                    out.push((names::compress_kv(src), c.as_slice()));
                }
                if self.index_head_dim > 0
                    && let Some(k) = self.index_k.get(&src)
                {
                    out.push((names::index_k(src), k.as_slice()));
                }
            }
            if s.ratio > 1 && s.fires && s.group_filled > 0 {
                if let Some(v) = self.group_kv.get(&src) {
                    out.push((names::group_kv(src), v.as_slice()));
                }
                if let Some(v) = self.group_score.get(&src) {
                    out.push((names::group_score(src), v.as_slice()));
                }
            }
        }
        out
    }

    /// Fold one step's outputs back in. `names`/`values` are the name list
    /// [`build_deepseek_v41_decode`] returned and the session's outputs, in the
    /// same order.
    pub fn apply(&mut self, plan: &V41DecodePlan, names: &[String], values: &[Vec<f32>]) -> Result<()> {
        if names.len() != values.len() {
            return Err(anyhow!(
                "deepseek_v41 decode: {} output names for {} outputs",
                names.len(),
                values.len()
            ));
        }
        for (name, v) in names.iter().zip(values) {
            let Some((tag, idx)) = name.rsplit_once('.') else {
                continue; // `logits`
            };
            let i: usize = match idx.parse() {
                Ok(i) => i,
                Err(_) => continue,
            };
            match tag {
                "kvnew" => {
                    let rows = &mut self.window[i];
                    rows.extend_from_slice(v);
                    // the new token occupies the last ring slot, so the buffer
                    // handed to the NEXT step holds at most window_size - 1
                    let cap = self.window_size.saturating_sub(1) * self.head_dim;
                    if rows.len() > cap {
                        rows.drain(..rows.len() - cap);
                    }
                }
                "compnew" => self.compress.entry(i).or_default().extend_from_slice(v),
                "indexknew" => self.index_k.entry(i).or_default().extend_from_slice(v),
                "groupkvnew" => self.group_kv.entry(i).or_default().extend_from_slice(v),
                "groupscorenew" => self.group_score.entry(i).or_default().extend_from_slice(v),
                _ => {}
            }
        }
        // a completed group clears what it consumed
        for s in &plan.sources {
            if s.ratio > 1 && s.fires {
                self.group_kv.remove(&s.source_layer);
                self.group_score.remove(&s.source_layer);
            }
        }
        Ok(())
    }
}

/// State threaded through one decode step's layer loop.
struct StepShared {
    compress_kv: Option<NodeId>,
    index_k: Option<NodeId>,
    topk_mask: Option<NodeId>,
    candidates: Option<NodeId>,
    len_after: usize,
}

/// Build one decode step at absolute position `pos`.
///
/// Returns the graph, its parameters, and the output-name list in graph-output
/// order (`logits` first). Inputs are named by [`names`]; the host sizes them
/// from [`V41DecodePlan`].
pub fn build_deepseek_v41_decode(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    pos: usize,
    inputs: &V41Inputs,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    spec.validate()?;
    let plan = V41DecodePlan::new(spec, pos);
    let mut g = Graph::new("deepseek_v41_decode");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let (d, hc, nh, hd) = (spec.dim, spec.hc_mult, spec.n_heads, spec.head_dim);
    let ql = spec.q_lora_rank;
    let rd = spec.rope_head_dim & !1;
    let eps = spec.rms_norm_eps;
    let rows = 1usize;
    let cache_len = plan.cache_len;
    let zb_d = synth_zero(&mut g, &mut params, "v41d.zb.d", d);
    let zb_ql = synth_zero(&mut g, &mut params, "v41d.zb.ql", ql);
    let zb_hd = synth_zero(&mut g, &mut params, "v41d.zb.hd", hd);

    let yarn = (spec.original_seq_len > 0 && spec.rope_factor > 1.0).then_some((
        spec.original_seq_len,
        spec.rope_factor,
        spec.beta_fast,
        spec.beta_slow,
    ));
    // q and the window KV rotate at the current position; the two bases are the
    // same split as prefill (compressed layers use `compress_rope_theta` + YaRN).
    let (cos_w, sin_w, sininv_w) =
        rope_table(&mut g, &mut params, &[pos], rd, spec.rope_theta, None, "d.win");
    let (cos_k, sin_k, sininv_k) = rope_table(
        &mut g,
        &mut params,
        &[pos],
        rd,
        spec.compress_rope_theta,
        yarn,
        "d.comp",
    );
    // A latent completed at this step stands for the first token of its group,
    // which is `pos + 1 - ratio`.
    let mut latent_rope: HashMap<usize, (NodeId, NodeId)> = HashMap::new();
    for s in &plan.sources {
        if !s.fires || latent_rope.contains_key(&s.ratio) {
            continue;
        }
        let p = pos + 1 - s.ratio;
        let (c, sn, _) = rope_table(
            &mut g,
            &mut params,
            &[p],
            rd,
            spec.compress_rope_theta,
            yarn,
            &format!("d.lat{}", s.ratio),
        );
        latent_rope.insert(s.ratio, (c, sn));
    }

    let mut outputs: Vec<NodeId> = Vec::new();
    let mut output_names: Vec<String> = Vec::new();

    let input_ids = g.input("input_ids", Shape::new(&[1, 1], DType::I32));
    let (embed_w, _, _) = load_dense_dequant(&mut g, &mut params, weights, "embed.weight")?;
    let h0 = g.gather_(embed_w, input_ids, 0);
    let h0 = g.reshape_(h0, vec![1, 1, d as i64]);
    let ones_hc = synth_const(&mut g, &mut params, "v41d.hc.ones", vec![1f32; hc], &[1, hc, 1]);
    let mut h = g.mul(h0, ones_hc);
    let mut pre_mix = identity_pre_mix(&mut g, &mut params, rows, hc);

    let mut shared = StepShared {
        compress_kv: None,
        index_k: None,
        topk_mask: None,
        candidates: None,
        len_after: 0,
    };

    let mut main_hiddens: Vec<NodeId> = Vec::new();
    for il in 0..spec.n_layers {
        let lp = spec.layer_prefix(il);

        if let Some(e) = &spec.engram
            && let Some(hash_idx) = e.layer_hash_index(il)
        {
            {
                let cols = e.n_hash_cols();
                let n_eng = e.layer_ids.len();
                if inputs.engram_rows.len() != n_eng * cols {
                    return Err(anyhow!(
                        "deepseek_v41 decode: engram_rows has {} entries, expected {} (1 token × {n_eng} layers × {cols} cols)",
                        inputs.engram_rows.len(),
                        n_eng * cols
                    ));
                }
                let slice: Vec<f32> = inputs.engram_rows[hash_idx * cols..(hash_idx + 1) * cols]
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                let row_ids = synth_const(
                    &mut g,
                    &mut params,
                    &format!("{lp}.engram.rows"),
                    slice,
                    &[1, cols],
                );
                h = load_and_build_v41_engram(
                    &mut g, &mut params, weights, &lp, h, row_ids, None, rows, hc, d, e, eps,
                )?;
            }
        }

        if inputs.emit_main_hidden && spec.dspark_target_layer_ids.contains(&il) {
            main_hiddens.push(hc_mean(&mut g, h, rows, d));
        }

        let ratio = spec.ratio(il);
        let (cos, sin, sin_inv) = if ratio > 0 {
            (cos_k, sin_k, sininv_k)
        } else {
            (cos_w, sin_w, sininv_w)
        };

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
            rows,
            hc,
            d,
            eps,
            spec.hc_eps,
            spec.hc_mult_sinkhorn_iters,
            &format!("{lp}.da"),
        );
        let xa = hc_reduce(&mut g, h, pre_mix, rows, hc);
        let an = load_norm(&mut g, &mut params, weights, &format!("{lp}.attn_norm.weight"), 0.0)?;
        let xa = g.rms_norm(xa, an, zb_d, eps);

        let wq_a = load_proj(&mut g, &mut params, packed, weights, &format!("{lp}.attn.wq_a.weight"))?;
        let qr = emit_proj(&mut g, xa, &wq_a, Shape::new(&[rows, ql], f));
        let q_norm = load_norm(&mut g, &mut params, weights, &format!("{lp}.attn.q_norm.weight"), 0.0)?;
        let qr = g.rms_norm(qr, q_norm, zb_ql, eps);
        let wq_b = load_proj(&mut g, &mut params, packed, weights, &format!("{lp}.attn.wq_b.weight"))?;
        let q = emit_proj(&mut g, qr, &wq_b, Shape::new(&[rows, nh * hd], f));
        let q = rope_tail(&mut g, q, cos, sin, rows, nh, hd, rd);

        let wkv = load_proj(&mut g, &mut params, packed, weights, &format!("{lp}.attn.wkv.weight"))?;
        let kv = emit_proj(&mut g, xa, &wkv, Shape::new(&[rows, hd], f));
        let kv_norm = load_norm(&mut g, &mut params, weights, &format!("{lp}.attn.kv_norm.weight"), 0.0)?;
        let kv = g.rms_norm(kv, kv_norm, zb_hd, eps);
        let kv_new = rope_tail(&mut g, kv, cos, sin, rows, 1, hd, rd);
        outputs.push(kv_new);
        output_names.push(names::window_kv_new(il));

        // the window this query attends over: everything cached plus itself
        let window = if cache_len > 0 {
            let cached = g.input(names::window_kv(il), Shape::new(&[cache_len, hd], f));
            g.concat_(vec![cached, kv_new], 0)
        } else {
            kv_new
        };
        let n_window = cache_len + 1;

        // ── compressed KV: produced at a source, read by everyone after it ──
        if let Some(step) = plan.for_source(il) {
            let (latent, new_nodes) =
                build_step_compressor(&mut g, &mut params, weights, &lp, xa, step, hd, eps)?;
            for (node, name) in new_nodes {
                outputs.push(node);
                output_names.push(name);
            }
            // extend the caches, whether or not this step added to them
            let cached_comp = (step.len_before > 0)
                .then(|| g.input(names::compress_kv(il), Shape::new(&[step.len_before, hd], f)));
            let cached_ik = (step.len_before > 0 && spec.index_head_dim > 0).then(|| {
                g.input(
                    names::index_k(il),
                    Shape::new(&[step.len_before, spec.index_head_dim], f),
                )
            });

            if let Some(lat) = latent {
                let (cos_l, sin_l) = latent_rope[&step.ratio];
                if spec.index_head_dim > 0 {
                    let ihd = spec.index_head_dim;
                    let wk = load_p(&mut g, &mut params, weights, &format!("{lp}.attn.indexer.wk.weight"), true)?;
                    let k_norm = load_norm(
                        &mut g,
                        &mut params,
                        weights,
                        &format!("{lp}.attn.indexer.k_norm.weight"),
                        0.0,
                    )?;
                    let zb_i = synth_zero(&mut g, &mut params, &format!("{lp}.dzb.ihd"), ihd);
                    let k = g.mm(lat, wk);
                    let k = g.rms_norm(k, k_norm, zb_i, eps);
                    let k = rope_tail(&mut g, k, cos_l, sin_l, 1, 1, ihd, rd);
                    outputs.push(k);
                    output_names.push(names::index_k_new(il));
                    shared.index_k = Some(match cached_ik {
                        Some(c) => g.concat_(vec![c, k], 0),
                        None => k,
                    });
                }
                let comp = rope_tail(&mut g, lat, cos_l, sin_l, 1, 1, hd, rd);
                outputs.push(comp);
                output_names.push(names::compress_kv_new(il));
                shared.compress_kv = Some(match cached_comp {
                    Some(c) => g.concat_(vec![c, comp], 0),
                    None => comp,
                });
            } else {
                shared.compress_kv = cached_comp;
                shared.index_k = cached_ik;
            }
            shared.len_after = step.len_after;
        }

        let ncomp = if ratio > 0 {
            plan.for_source(spec.kv_source_for(il).unwrap_or(il))
                .map(|s| s.len_after)
                .unwrap_or(0)
        } else {
            0
        };

        let (kv_all, mask, n_keys) = if ncomp == 0 {
            // every provided window slot is real, so no mask is needed
            let m = synth_const(
                &mut g,
                &mut params,
                &format!("{lp}.d.maskw"),
                vec![0f32; n_window],
                &[1, n_window],
            );
            (window, m, n_window)
        } else {
            if shared.len_after != ncomp {
                return Err(anyhow!(
                    "deepseek_v41 decode: layer {il} expects {ncomp} compressed positions, source has {}",
                    shared.len_after
                ));
            }
            let comp = shared.compress_kv.ok_or_else(|| {
                anyhow!("deepseek_v41 decode: layer {il} reads compressed KV with no source")
            })?;
            // Every cached latent is causally visible to this query by
            // construction, so the causal mask is all-zero and only the Indexer's
            // budget can drop anything.
            let causal_c = compressed_causal_mask(
                &mut g,
                &mut params,
                &[ncomp],
                ncomp,
                &format!("{lp}.d.maskc"),
            );
            if spec.is_index_source(il) && spec.index_head_dim > 0 {
                let index_k = shared.index_k.ok_or_else(|| {
                    anyhow!("deepseek_v41 decode: layer {il} indexes with no index keys")
                })?;
                let wq_b_i = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.indexer.wq_b.weight"),
                    true,
                )?;
                let wpj = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.indexer.weights_proj.weight"),
                    true,
                )?;
                let mut score = build_v41_index_score(
                    &mut g,
                    &mut params,
                    qr,
                    xa,
                    index_k,
                    wq_b_i,
                    wpj,
                    cos,
                    sin,
                    rows,
                    spec.index_n_heads,
                    spec.index_head_dim,
                    rd,
                    ncomp,
                    &format!("{lp}.d"),
                );
                score = g.add(score, causal_c);
                if spec.is_candidate_source(il) && spec.candidate_block_size > 0 {
                    shared.candidates = Some(crate::dsv41_graph::build_v41_candidate_mask(
                        &mut g,
                        &mut params,
                        score,
                        causal_c,
                        rows,
                        ncomp,
                        &[ncomp],
                        spec.candidate_block_size,
                        spec.candidate_topk_blocks,
                        &format!("{lp}.d"),
                    ));
                } else if spec.uses_candidates(il)
                    && let Some(c) = shared.candidates
                {
                    score = g.add(score, c);
                }
                let base = match shared.candidates {
                    Some(c) if spec.uses_candidates(il) => c,
                    _ => causal_c,
                };
                shared.topk_mask = Some(if ncomp > spec.index_topk && spec.index_topk > 0 {
                    exact_topk_mask(
                        &mut g,
                        &mut params,
                        score,
                        base,
                        rows,
                        ncomp,
                        spec.index_topk,
                        &format!("{lp}.d"),
                    )
                } else {
                    base
                });
            }
            let comp_mask = shared.topk_mask.unwrap_or(causal_c);
            let win_mask = synth_const(
                &mut g,
                &mut params,
                &format!("{lp}.d.maskw"),
                vec![0f32; n_window],
                &[1, n_window],
            );
            let kv_all = g.concat_(vec![window, comp], 0);
            let full = g.concat_(vec![win_mask, comp_mask], 1);
            (kv_all, full, n_window + ncomp)
        };

        let sink = load_p(&mut g, &mut params, weights, &format!("{lp}.attn.attn_sink"), false)?;
        let q3 = g.reshape_(q, vec![rows as i64, nh as i64, hd as i64]);
        let o = build_v4_sink_attention(
            &mut g,
            &mut params,
            q3,
            kv_all,
            mask,
            sink,
            (hd as f32).powf(-0.5),
            rows,
            nh,
            hd,
            n_keys,
            &format!("{lp}.d"),
        );
        let o_flat = g.reshape_(o, vec![rows as i64, (nh * hd) as i64]);
        let o_inv = rope_tail(&mut g, o_flat, cos, sin_inv, rows, nh, hd, rd);
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
            &mut g,
            o_inv,
            wo_a,
            wo_b,
            rows,
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
            d,
        );
        h = hc_post(&mut g, attn_out, residual, attn_post, attn_comb, rows, hc, d);

        // ── FFN sub-block ──
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
            rows,
            hc,
            d,
            eps,
            spec.hc_eps,
            spec.hc_mult_sinkhorn_iters,
            &format!("{lp}.df"),
        );
        let xf = hc_reduce(&mut g, h, attn_pre, rows, hc);
        let fnorm = load_norm(&mut g, &mut params, weights, &format!("{lp}.ffn_norm.weight"), 0.0)?;
        let xf = g.rms_norm(xf, fnorm, zb_d, eps);
        let ffn_out = build_v41_moe(&mut g, &mut params, packed, weights, spec, il, xf, rows, None)?;
        h = hc_post(&mut g, ffn_out, residual, ffn_post, ffn_comb, rows, hc, d);
        pre_mix = ffn_pre;
    }

    let x = hc_reduce(&mut g, h, pre_mix, rows, hc);
    let fnorm = load_norm(&mut g, &mut params, weights, "norm.weight", 0.0)?;
    let x = g.rms_norm(x, fnorm, zb_d, eps);
    let head = load_proj(&mut g, &mut params, packed, weights, "head.weight")?;
    let logits = emit_proj(&mut g, x, &head, Shape::new(&[rows, spec.vocab_size], f));
    let logits = g.reshape_(logits, vec![rows as i64, spec.vocab_size as i64]);

    let mut all = vec![logits];
    let mut all_names = vec!["logits".to_string()];
    if inputs.emit_main_hidden {
        all.push(main_hidden_node(&mut g, &main_hiddens, spec, rows)?);
        all_names.push("main_hidden".to_string());
    }
    all.extend(outputs);
    all_names.extend(output_names);
    g.set_outputs(all);
    let _ = NEG;
    Ok((g, params, all_names))
}

/// One decode step of the KV Compressor.
///
/// `ratio == 1` completes a group every step, so it is a plain projection with no
/// carried state. Above that, the step either completes the group — pooling the
/// carried `ratio - 1` entries together with this token's — or contributes to it,
/// in which case there is no latent and the raw `wkv`/`wgate` pair is handed back.
#[allow(clippy::too_many_arguments)]
fn build_step_compressor(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x: NodeId,
    step: &CompressStep,
    hd: usize,
    eps: f32,
) -> Result<(Option<NodeId>, Vec<(NodeId, String)>)> {
    let f = DType::F32;
    let wkv = load_p(g, params, weights, &format!("{lp}.attn.compressor.wkv.weight"), true)?;
    let norm_w = load_norm(g, params, weights, &format!("{lp}.attn.compressor.norm.weight"), 0.0)?;
    let zb = synth_zero(g, params, &format!("{lp}.d.comp.zb"), hd);
    let kv = g.mm(x, wkv); // [1, hd]

    if step.ratio == 1 {
        let latent = g.rms_norm(kv, norm_w, zb, eps);
        return Ok((Some(latent), Vec::new()));
    }

    let wgate = load_p(g, params, weights, &format!("{lp}.attn.compressor.wgate.weight"), true)?;
    let score = g.mm(x, wgate); // [1, hd]
    if !step.fires {
        // still filling: hand both halves back for the host to accumulate
        return Ok((
            None,
            vec![
                (kv, names::group_kv_new(step.source_layer)),
                (score, names::group_score_new(step.source_layer)),
            ],
        ));
    }

    let filled = step.group_filled;
    debug_assert_eq!(filled, step.ratio - 1, "a firing step completes the group");
    let (kv_group, sc_group) = if filled > 0 {
        let gk = g.input(names::group_kv(step.source_layer), Shape::new(&[filled, hd], f));
        let gs = g.input(names::group_score(step.source_layer), Shape::new(&[filled, hd], f));
        (g.concat_(vec![gk, kv], 0), g.concat_(vec![gs, score], 0))
    } else {
        (kv, score)
    };
    let (r, dd) = (step.ratio as i64, hd as i64);
    let kv3 = g.reshape_(kv_group, vec![1, r, dd]);
    let sc3 = g.reshape_(sc_group, vec![1, r, dd]);
    let sct = g.transpose_(sc3, vec![0, 2, 1]); // [1, hd, ratio]
    let w = g.sm(sct, -1);
    let w = g.transpose_(w, vec![0, 2, 1]);
    let prod = g.mul(kv3, w);
    let pooled = g.sum(prod, vec![1], false); // [1, hd]
    let pooled = g.reshape_(pooled, vec![1, dd]);
    Ok((Some(g.rms_norm(pooled, norm_w, zb, eps)), Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> DeepseekV41Spec {
        DeepseekV41Spec::from_config(&serde_json::json!({
            "vocab_size": 64, "dim": 32, "num_hidden_layers": 6, "head_dim": 32,
            "num_attention_heads": 2, "o_lora_rank": 8, "n_routed_experts": 4,
            "moe_intermediate_size": 16, "o_groups": 2, "q_lora_rank": 16,
            "rope_head_dim": 16, "sliding_window": 4,
            "compress_ratios": [0, 0, 2, 2, 1, 1],
            "kv_source_layers": [2, 4], "index_source_layers": [2, 4, 5],
            "index_n_heads": 2, "index_head_dim": 32, "index_topk": 3,
        }))
        .unwrap()
    }

    #[test]
    fn plan_tracks_group_filling_and_firing() {
        let s = spec();
        // ratio-2 source (layer 2) fires on odd positions; ratio-1 (layer 4) always
        for pos in 0..8 {
            let p = V41DecodePlan::new(&s, pos);
            let r2 = p.for_source(2).unwrap();
            let r1 = p.for_source(4).unwrap();
            assert_eq!(r2.ratio, 2);
            assert_eq!(r2.fires, pos % 2 == 1, "pos {pos}");
            assert_eq!(r2.group_filled, pos % 2, "pos {pos}");
            assert_eq!(r2.len_before, pos / 2, "pos {pos}");
            assert_eq!(r2.len_after, pos.div_ceil(2), "pos {pos}");
            assert!(r1.fires, "pos {pos}: a ratio-1 source fires every step");
            assert_eq!(r1.len_after, pos + 1, "pos {pos}");
            assert_eq!(r1.group_filled, 0);
        }
    }

    #[test]
    fn plan_leaves_the_last_window_slot_for_this_token() {
        let s = spec(); // sliding_window 4
        // key count is cache_len + 1 and must equal prefill's min(pos+1, window)
        for pos in 0..10 {
            let got = V41DecodePlan::new(&s, pos).cache_len + 1;
            assert_eq!(got, (pos + 1).min(4), "pos {pos}");
        }
    }

    #[test]
    fn plan_lists_only_kv_sources() {
        let s = spec();
        let p = V41DecodePlan::new(&s, 5);
        let srcs: Vec<usize> = p.sources.iter().map(|x| x.source_layer).collect();
        assert_eq!(srcs, vec![2, 4], "index-only sources own no cache");
    }
}
