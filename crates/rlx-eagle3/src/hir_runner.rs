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

//! Stateful runner around a compiled-graph-per-`past_seq` set.
//! Holds the pre-transposed weights once and feeds the draft step
//! across `n` speculation rounds. Used by
//! [`crate::speculator::Eagle3Speculator::with_hir_runner`] to back
//! the real `propose()` path with the HIR draft graph instead of
//! the scalar reference.
//!
//! Numerical parity vs the scalar reference is pinned by
//! `tests/hir_parity.rs`; throughput numbers are in
//! `crates/rlx-eagle3/E2E_RESULTS.md`.

use anyhow::{Context, Result};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::draft::DraftGeom;
use crate::hir_draft::{build_draft_step_graph, input_names as I, tensor_names as T};
use crate::weights::Eagle3DraftWeights;

/// Pre-transposed weight buffers. Each rlx-mm convention wants
/// `[K, N]`; the on-disk safetensors store `[N, K]`. We do the
/// transpose once at construction so each `compiled.set_param` call
/// is just a zero-cost slice handoff (the param-view fast path in
/// rlx-mlx / rlx-metal).
pub struct TransposedWeights {
    pub q_t: Vec<f32>,
    pub k_t: Vec<f32>,
    pub v_t: Vec<f32>,
    pub o_t: Vec<f32>,
    pub gate_t: Vec<f32>,
    pub up_t: Vec<f32>,
    pub down_t: Vec<f32>,
    pub lm_t: Vec<f32>,
    pub zero_beta: Vec<f32>,
}

fn transpose_rc(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

impl TransposedWeights {
    pub fn from_weights(weights: &Eagle3DraftWeights, geom: DraftGeom) -> Result<Self> {
        let get = |name: &str| -> Result<&[f32]> {
            weights
                .get(name)
                .map(|t| t.data.as_slice())
                .with_context(|| format!("missing tensor `{name}`"))
        };
        let q_dim = geom.n_heads * geom.head_dim;
        let kv_dim = geom.n_kv_heads * geom.head_dim;
        let two_h = 2 * geom.h_draft;
        Ok(Self {
            q_t: transpose_rc(get("decoder.self_attn.q_proj.weight")?, q_dim, two_h),
            k_t: transpose_rc(get("decoder.self_attn.k_proj.weight")?, kv_dim, two_h),
            v_t: transpose_rc(get("decoder.self_attn.v_proj.weight")?, kv_dim, two_h),
            o_t: transpose_rc(get("decoder.self_attn.o_proj.weight")?, geom.h_draft, q_dim),
            gate_t: transpose_rc(
                get("decoder.mlp.gate_proj.weight")?,
                geom.intermediate,
                geom.h_draft,
            ),
            up_t: transpose_rc(
                get("decoder.mlp.up_proj.weight")?,
                geom.intermediate,
                geom.h_draft,
            ),
            down_t: transpose_rc(
                get("decoder.mlp.down_proj.weight")?,
                geom.h_draft,
                geom.intermediate,
            ),
            lm_t: transpose_rc(get("lm_head.weight")?, geom.draft_vocab, geom.h_draft),
            zero_beta: vec![0.0f32; geom.h_draft],
        })
    }
}

fn set_all_params(
    compiled: &mut CompiledGraph,
    weights: &Eagle3DraftWeights,
    tr: &TransposedWeights,
) -> Result<()> {
    let get = |name: &str| -> Result<&[f32]> {
        weights
            .get(name)
            .map(|t| t.data.as_slice())
            .with_context(|| format!("missing `{name}`"))
    };
    compiled.set_param(T::INPUT_LAYERNORM, get("decoder.input_layernorm.weight")?);
    compiled.set_param(T::HIDDEN_NORM, get("decoder.hidden_norm.weight")?);
    compiled.set_param(
        T::POST_ATTN_LN,
        get("decoder.post_attention_layernorm.weight")?,
    );
    compiled.set_param(T::NORM, get("norm.weight")?);
    compiled.set_param(T::Q_PROJ, &tr.q_t);
    compiled.set_param(T::K_PROJ, &tr.k_t);
    compiled.set_param(T::V_PROJ, &tr.v_t);
    compiled.set_param(T::O_PROJ, &tr.o_t);
    compiled.set_param(T::GATE_PROJ, &tr.gate_t);
    compiled.set_param(T::UP_PROJ, &tr.up_t);
    compiled.set_param(T::DOWN_PROJ, &tr.down_t);
    compiled.set_param(T::LM_HEAD, &tr.lm_t);
    compiled.set_param(T::ZERO_BETA, &tr.zero_beta);
    Ok(())
}

fn rope_row(position: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for k in 0..half {
        let exp = -(2.0 * k as f64) / (head_dim as f64);
        let freq = (theta as f64).powf(exp);
        let angle = (position as f64) * freq;
        cos[k] = angle.cos() as f32;
        sin[k] = angle.sin() as f32;
    }
    (cos, sin)
}

/// Per-step KV cache held between `propose()` rounds. Cleared each
/// time the speculator starts a new propose call.
#[derive(Default)]
pub struct DraftKvCache {
    pub past_k: Vec<f32>,
    pub past_v: Vec<f32>,
    /// How many rows currently live in `past_k`/`past_v`. Each row
    /// is `kv_dim = n_kv_heads * head_dim` f32 values.
    pub seq: usize,
}

/// Owns one compiled draft graph per `past_seq` value (0..n_max).
/// Reused across propose rounds.
pub struct HirDraftRunner {
    geom: DraftGeom,
    graphs: Vec<CompiledGraph>,
    /// Embedding table — borrowed view onto
    /// [`Eagle3DraftWeights::get("embed_tokens.weight")`]. We keep it
    /// here so `step()` can do the host-side embed lookup without
    /// re-walking the weights map.
    embed_tokens: Vec<f32>,
    /// d2t offsets snapshot.
    d2t_offsets: Vec<u32>,
}

impl HirDraftRunner {
    /// Compile `n_max` graphs (one per past_seq value) on `device`.
    /// `n_max` should be ≥ the largest `n` you'll ever pass to
    /// `propose()` (typically `cfg.speculative_tokens`).
    pub fn new(
        weights: &Eagle3DraftWeights,
        geom: DraftGeom,
        n_max: usize,
        device: Device,
    ) -> Result<Self> {
        let tr = TransposedWeights::from_weights(weights, geom)?;
        let session = Session::new(device);
        let mut graphs: Vec<CompiledGraph> = Vec::with_capacity(n_max);
        for past_seq in 0..n_max {
            let g = build_draft_step_graph(geom, past_seq);
            let mut compiled = session.compile(g);
            set_all_params(&mut compiled, weights, &tr)?;
            graphs.push(compiled);
        }
        let embed = weights
            .get("embed_tokens.weight")
            .context("missing embed_tokens.weight")?;
        let embed_tokens = embed.data.clone();
        let d2t_offsets = weights.d2t().to_vec();
        Ok(Self {
            geom,
            graphs,
            embed_tokens,
            d2t_offsets,
        })
    }

    pub fn n_max(&self) -> usize {
        self.graphs.len()
    }

    pub fn geom(&self) -> DraftGeom {
        self.geom
    }

    /// Run one step. `step_idx` selects which compiled graph to use
    /// (must be `< n_max`). Updates `cache` in place.
    ///
    /// Returns `(draft_logits[V_draft], new_hidden[H])`.
    pub fn step(
        &mut self,
        step_idx: usize,
        prev_target_token: u32,
        prev_hidden: &[f32],
        cache: &mut DraftKvCache,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        anyhow::ensure!(
            step_idx < self.graphs.len(),
            "step_idx {step_idx} >= n_max {}",
            self.graphs.len()
        );
        anyhow::ensure!(
            cache.seq == step_idx,
            "cache.seq ({}) must equal step_idx ({step_idx}) — caller drives KV growth",
            cache.seq,
        );
        let g = self.geom;
        let row_off = (prev_target_token as usize) * g.h_draft;
        let prev_embed = &self.embed_tokens[row_off..row_off + g.h_draft];
        let (cos, sin) = rope_row(step_idx, g.head_dim, g.rope_theta);

        let outs = self.graphs[step_idx].run(&[
            (I::PREV_EMBED, prev_embed),
            (I::PREV_HIDDEN, prev_hidden),
            (I::PAST_K, cache.past_k.as_slice()),
            (I::PAST_V, cache.past_v.as_slice()),
            (I::ROPE_COS, cos.as_slice()),
            (I::ROPE_SIN, sin.as_slice()),
        ]);
        let logits = outs[0].clone();
        let new_hidden = outs[1].clone();
        let new_k = &outs[2];
        let new_v = &outs[3];

        // Replace the KV cache with the graph's `new_k`/`new_v`
        // (which already include past + current along seq axis).
        cache.past_k = new_k.to_vec();
        cache.past_v = new_v.to_vec();
        cache.seq = step_idx + 1;
        Ok((logits, new_hidden))
    }

    /// Apply the d2t offset to a draft-vocab argmax to get a
    /// target-vocab token id. Mirrors vLLM's
    /// `input_ids = input_ids + self.d2t[input_ids]`.
    pub fn draft_id_to_target(&self, draft_id: u32) -> u32 {
        draft_id + self.d2t_offsets[draft_id as usize]
    }
}
