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

//! CPU-eager talker backbone (MRoPE + KV) — parity reference path.

use crate::config::TalkerConfig;
use crate::load::{Qwen3TtsWeightStore, remap_talker_weights};
use crate::mrope::{talker_decode_rope_into, talker_rope_half, talker_rope_index_prefill};
use crate::talker::math::{gqa_attention1_into, matvec_into};
use crate::talker::rope::build_inv_freq;
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rlx_core::KvCacheState;
use std::collections::HashMap;

pub struct TalkerEagerModel {
    layers: Vec<TalkerLayer>,
    norm_weight: Vec<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    head_half: usize,
    kv: Vec<LayerKv>,
    pub(crate) past_len: usize,
    pub(crate) rope_delta: i64,
    norm_eps: f32,
    talker_cfg: TalkerConfig,
    inv_freq: Vec<f64>,
    /// Flat `[past_len * head_half]` decode cos/sin (indexed by `past_len` before each step).
    decode_rope_cos_bank: Vec<f32>,
    decode_rope_sin_bank: Vec<f32>,
    decode_rope_cos: Vec<f32>,
    decode_rope_sin: Vec<f32>,
    work_hidden: Vec<f32>,
    work_q: Vec<f32>,
    work_k: Vec<f32>,
    work_v: Vec<f32>,
    work_attn: Vec<f32>,
    work_scratch: Vec<f32>,
    work_gate: Vec<f32>,
    work_up: Vec<f32>,
    work_qkv: Vec<f32>,
    work_gate_up: Vec<f32>,
    attn_weights: Vec<f32>,
    work_kv_head: Vec<f32>,
    max_attn_tokens: usize,
}

pub(crate) struct TalkerLayer {
    wq: Array2<f32>,
    wk: Array2<f32>,
    wv: Array2<f32>,
    /// Stacked `[Q; K; V]` for one matvec in [`TalkerLayer::forward_one`].
    wqkv: Array2<f32>,
    wo: Array2<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: Array2<f32>,
    up: Array2<f32>,
    /// Stacked `[gate; up]` for one matvec in [`TalkerLayer::forward_one`].
    gate_up: Array2<f32>,
    down: Array2<f32>,
    q_dim: usize,
    kv_dim: usize,
    inter_dim: usize,
}

pub(crate) struct LayerKv {
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
}

pub(crate) fn kv_rows(kv: &[f32], dim: usize) -> usize {
    kv.len().checked_div(dim).unwrap_or(0)
}

impl TalkerEagerModel {
    pub fn open(store: &Qwen3TtsWeightStore, talker: &TalkerConfig) -> Result<Self> {
        let mut wm = store.load_talker_backbone()?;
        let map = remap_talker_weights(&mut wm)?;
        Self::open_from_map(&map, talker)
    }

    /// Build from an already-remapped weight map (avoids a second mmap parse +
    /// allocation when the caller has already loaded the backbone).
    pub fn open_from_map(
        map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        talker: &TalkerConfig,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(talker.num_hidden_layers);
        for i in 0..talker.num_hidden_layers {
            layers.push(load_layer(map, i)?);
        }
        let norm_weight = take1d(map, "model.norm.weight")?;
        let head_half = talker.head_dim / 2;
        let hidden = talker.hidden_size;
        let q_dim = talker.num_attention_heads * talker.head_dim;
        let kv_dim = talker.num_key_value_heads * talker.head_dim;
        let inter_dim = talker.intermediate_size;
        let max_attn_tokens = 256usize;
        Ok(Self {
            layers,
            norm_weight,
            n_heads: talker.num_attention_heads,
            n_kv_heads: talker.num_key_value_heads,
            head_dim: talker.head_dim,
            hidden,
            head_half,
            kv: (0..talker.num_hidden_layers)
                .map(|_| LayerKv {
                    k: Vec::new(),
                    v: Vec::new(),
                })
                .collect(),
            past_len: 0,
            rope_delta: 0,
            norm_eps: talker.rms_norm_eps as f32,
            talker_cfg: talker.clone(),
            inv_freq: build_inv_freq(talker.head_dim, talker.rope_theta),
            decode_rope_cos_bank: Vec::new(),
            decode_rope_sin_bank: Vec::new(),
            decode_rope_cos: vec![0f32; head_half],
            decode_rope_sin: vec![0f32; head_half],
            work_hidden: vec![0f32; hidden],
            work_q: vec![0f32; q_dim],
            work_k: vec![0f32; kv_dim],
            work_v: vec![0f32; kv_dim],
            work_attn: vec![0f32; q_dim],
            work_scratch: vec![0f32; hidden],
            work_gate: vec![0f32; inter_dim],
            work_up: vec![0f32; inter_dim],
            work_qkv: vec![0f32; q_dim + 2 * kv_dim],
            work_gate_up: vec![0f32; 2 * inter_dim],
            attn_weights: vec![0f32; talker.num_attention_heads * max_attn_tokens],
            // Layout used by gqa_attention1_into: k_gather[n_keys*head_dim] +
            // v_gather[n_keys*head_dim] + batch_scores[>= repeats*n_keys]. Size
            // for the worst case where n_keys == max_attn_tokens.
            work_kv_head: vec![
                0f32;
                max_attn_tokens
                    * (2 * talker.head_dim
                        + talker.num_attention_heads
                            / talker.num_key_value_heads.max(1))
            ],
            max_attn_tokens,
        })
    }

    /// Grow attention scratch + RoPE bank to support `horizon` cached keys.
    /// No-op if the buffers are already large enough.
    pub fn ensure_attn_horizon(&mut self, horizon: usize) {
        if horizon <= self.max_attn_tokens {
            return;
        }
        self.max_attn_tokens = horizon;
        self.attn_weights.resize(self.n_heads * horizon, 0.0);
        let repeats = self.n_heads / self.n_kv_heads.max(1);
        self.work_kv_head
            .resize(horizon * (2 * self.head_dim + repeats), 0.0);
        let half = self.head_half;
        self.decode_rope_cos_bank.clear();
        self.decode_rope_sin_bank.clear();
        self.decode_rope_cos_bank.resize(horizon * half, 0.0);
        self.decode_rope_sin_bank.resize(horizon * half, 0.0);
        self.warm_decode_rope_bank();
    }

    /// Precompute decode RoPE rows `bank[i] = rope(i + rope_delta)` for `i < max_attn_tokens`.
    pub fn warm_decode_rope_bank(&mut self) {
        let half = self.head_half;
        let n = self.max_attn_tokens;
        let len = n * half;
        if self.decode_rope_cos_bank.len() < len {
            self.decode_rope_cos_bank.resize(len, 0.0);
            self.decode_rope_sin_bank.resize(len, 0.0);
        }
        let delta = self.rope_delta;
        let cfg = &self.talker_cfg;
        for i in 0..n {
            let pos = (i as i64 + delta) as usize;
            let off = i * half;
            if cfg.rope_scaling.is_some() {
                crate::talker::rope::rope_slice_into(
                    &self.inv_freq,
                    pos,
                    self.head_dim,
                    &mut self.decode_rope_cos_bank[off..off + half],
                    &mut self.decode_rope_sin_bank[off..off + half],
                );
            } else {
                let (c, s) = talker_rope_half(cfg, pos, half);
                self.decode_rope_cos_bank[off..off + half].copy_from_slice(&c);
                self.decode_rope_sin_bank[off..off + half].copy_from_slice(&s);
            }
        }
    }

    pub fn reset_kv(&mut self) {
        self.past_len = 0;
        self.rope_delta = 0;
        for kv in &mut self.kv {
            kv.k.clear();
            kv.v.clear();
        }
    }

    pub fn rope_delta(&self) -> i64 {
        self.rope_delta
    }

    /// Host K/V after prefill/decode (parity / isolation tests).
    pub fn kv_cache_state(&self) -> KvCacheState {
        KvCacheState {
            past_len: self.past_len,
            layers_k: self.kv.iter().map(|l| l.k.clone()).collect(),
            layers_v: self.kv.iter().map(|l| l.v.clone()).collect(),
        }
    }

    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    fn reserve_kv_horizon(&mut self, horizon: usize) {
        let cap = horizon.saturating_mul(self.kv_dim());
        for kv in &mut self.kv {
            kv.k.reserve(cap);
            kv.v.reserve(cap);
        }
    }

    /// Run a single decoder layer on `x` (fresh KV for that layer).
    pub fn forward_layer(
        &mut self,
        layer_idx: usize,
        x: ArrayView2<f32>,
        positions: &[usize],
        start_pos: usize,
    ) -> Result<Array2<f32>> {
        ensure!(layer_idx < self.layers.len());
        let mut kv = LayerKv {
            k: Vec::new(),
            v: Vec::new(),
        };
        let kv_dim = self.kv_dim();
        self.layers[layer_idx].forward(
            x,
            &mut kv,
            kv_dim,
            &self.talker_cfg,
            positions,
            start_pos,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.head_half,
            self.norm_eps,
            None,
        )
    }

    /// Run prefill through decoder layer `through` inclusive (`0` = layer 0 only).
    pub fn prefill_through_layer(
        &mut self,
        embeds: ArrayView2<f32>,
        through: usize,
    ) -> Result<Array2<f32>> {
        let (seq, h) = embeds.dim();
        ensure!(h == self.hidden);
        ensure!(through < self.layers.len());
        let mask = vec![1u8; seq];
        let (positions, rope_delta) = talker_rope_index_prefill(&mask);
        self.rope_delta = rope_delta;
        self.reset_kv();
        let kv_dim = self.kv_dim();
        let mut x = embeds.to_owned();
        for (li, layer) in self.layers.iter().enumerate().take(through + 1) {
            x = layer.forward(
                x.view(),
                &mut self.kv[li],
                kv_dim,
                &self.talker_cfg,
                &positions,
                0,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.head_half,
                self.norm_eps,
                None,
            )?;
        }
        Ok(x)
    }

    /// Per-layer last-token hidden (after each block, then final norm) for parity diff.
    pub fn prefill_layer_last_rows(&mut self, embeds: ArrayView2<f32>) -> Result<Vec<Vec<f32>>> {
        let (seq, h) = embeds.dim();
        ensure!(h == self.hidden);
        let mask = vec![1u8; seq];
        let (positions, rope_delta) = talker_rope_index_prefill(&mask);
        self.rope_delta = rope_delta;
        self.reset_kv();
        let kv_dim = self.kv_dim();
        let mut x = embeds.to_owned();
        let mut rows = Vec::with_capacity(self.layers.len() + 1);
        for (li, layer) in self.layers.iter().enumerate() {
            x = layer.forward(
                x.view(),
                &mut self.kv[li],
                kv_dim,
                &self.talker_cfg,
                &positions,
                0,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.head_half,
                self.norm_eps,
                None,
            )?;
            rows.push(x.row(x.nrows() - 1).to_vec());
        }
        x = rms_norm2(x.view(), &self.norm_weight, self.norm_eps);
        rows.push(x.row(x.nrows() - 1).to_vec());
        Ok(rows)
    }

    pub fn prefill(&mut self, embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (seq, h) = embeds.dim();
        ensure!(h == self.hidden);
        let mask = vec![1u8; seq];
        let (positions, rope_delta) = talker_rope_index_prefill(&mask);
        self.rope_delta = rope_delta;
        self.reset_kv();
        self.reserve_kv_horizon(seq.saturating_add(64));
        let kv_dim = self.kv_dim();
        let mut x = embeds.to_owned();
        for (li, layer) in self.layers.iter().enumerate() {
            x = layer.forward(
                x.view(),
                &mut self.kv[li],
                kv_dim,
                &self.talker_cfg,
                &positions,
                0,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.head_half,
                self.norm_eps,
                None,
            )?;
        }
        x = rms_norm2(x.view(), &self.norm_weight, self.norm_eps);
        self.past_len = seq;
        self.warm_decode_rope_bank();
        Ok(x)
    }

    /// Layer-0 GQA attention output (pre-`o_proj`) at last token, first `n` dims of head 0.
    pub fn layer0_attn_pre_o_last16(&mut self, embeds: ArrayView2<f32>) -> Result<Vec<f32>> {
        let (seq, h) = embeds.dim();
        ensure!(h == self.hidden);
        let mask = vec![1u8; seq];
        let (positions, rope_delta) = talker_rope_index_prefill(&mask);
        self.rope_delta = rope_delta;
        self.reset_kv();
        let layer = &self.layers[0];
        let x_norm = rms_norm2(embeds, &layer.attn_norm, self.norm_eps);
        let mut q = linear2(x_norm.view(), layer.wq.view());
        let mut k = linear2(x_norm.view(), layer.wk.view());
        let v = linear2(x_norm.view(), layer.wv.view());
        qk_norm_heads(
            &mut q,
            &layer.q_norm,
            self.n_heads,
            self.head_dim,
            self.norm_eps,
        );
        qk_norm_heads(
            &mut k,
            &layer.k_norm,
            self.n_kv_heads,
            self.head_dim,
            self.norm_eps,
        );
        apply_rope_qk(
            &mut q,
            &mut k,
            &self.talker_cfg,
            &positions,
            0,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.head_half,
        );
        let mut kv = LayerKv {
            k: Vec::new(),
            v: Vec::new(),
        };
        append_kv(&mut kv, self.kv_dim(), &k, &v);
        let t_k = kv_rows(&kv.k, self.kv_dim());
        let k_view = ArrayView2::from_shape((t_k, self.kv_dim()), &kv.k).expect("k shape");
        let v_view = ArrayView2::from_shape((t_k, self.kv_dim()), &kv.v).expect("v shape");
        let attn = gqa_attention(
            q.view(),
            k_view,
            v_view,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );
        let ti = seq - 1;
        let n = 16.min(self.head_dim);
        Ok((0..n).map(|i| attn[[ti, i]]).collect())
    }

    /// Layer-0 Q/K for head 0 at last token after RoPE (`2 * n` floats each).
    pub fn layer0_qk_head0_last(
        &mut self,
        embeds: ArrayView2<f32>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (seq, h) = embeds.dim();
        ensure!(h == self.hidden);
        let mask = vec![1u8; seq];
        let (positions, rope_delta) = talker_rope_index_prefill(&mask);
        self.rope_delta = rope_delta;
        self.reset_kv();
        let layer = &self.layers[0];
        let x_norm = rms_norm2(embeds, &layer.attn_norm, self.norm_eps);
        let mut q = linear2(x_norm.view(), layer.wq.view());
        let mut k = linear2(x_norm.view(), layer.wk.view());
        qk_norm_heads(
            &mut q,
            &layer.q_norm,
            self.n_heads,
            self.head_dim,
            self.norm_eps,
        );
        qk_norm_heads(
            &mut k,
            &layer.k_norm,
            self.n_kv_heads,
            self.head_dim,
            self.norm_eps,
        );
        apply_rope_qk(
            &mut q,
            &mut k,
            &self.talker_cfg,
            &positions,
            0,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.head_half,
        );
        let ti = seq - 1;
        let n = 16.min(self.head_dim);
        let mut qv = Vec::with_capacity(n);
        let mut kv = Vec::with_capacity(n);
        for i in 0..n {
            qv.push(q[[ti, i]]);
            kv.push(k[[ti, i]]);
        }
        Ok((qv, kv))
    }

    /// Last-token hidden after layer-0 attention residual (parity vs HF).
    pub fn layer0_after_attn_last(&mut self, embeds: ArrayView2<f32>) -> Result<Vec<f32>> {
        let (seq, h) = embeds.dim();
        ensure!(h == self.hidden);
        let mask = vec![1u8; seq];
        let (positions, rope_delta) = talker_rope_index_prefill(&mask);
        self.rope_delta = rope_delta;
        self.reset_kv();
        let kv_dim = self.kv_dim();
        let out = self.layers[0].forward_attn_residual(
            embeds,
            &mut self.kv[0],
            kv_dim,
            &self.talker_cfg,
            &positions,
            0,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.head_half,
            self.norm_eps,
            None,
        )?;
        Ok(out.row(out.nrows() - 1).to_vec())
    }

    /// KV decode; writes the last hidden row into `out` (fused matvec path, no `Array2` per layer).
    pub fn decode_step_into(&mut self, embed: ArrayView1<f32>, out: &mut [f32]) -> Result<()> {
        ensure!(embed.len() == self.hidden);
        ensure!(out.len() == self.hidden);
        self.work_hidden.copy_from_slice(embed.as_slice().unwrap());
        let p = self.past_len;
        let need = (p + 1) * self.head_half;
        if self.decode_rope_cos_bank.len() < need {
            self.warm_decode_rope_bank();
        }
        if p < self.max_attn_tokens && self.decode_rope_cos_bank.len() >= need {
            let roff = p * self.head_half;
            self.decode_rope_cos
                .copy_from_slice(&self.decode_rope_cos_bank[roff..roff + self.head_half]);
            self.decode_rope_sin
                .copy_from_slice(&self.decode_rope_sin_bank[roff..roff + self.head_half]);
        } else {
            talker_decode_rope_into(
                &self.talker_cfg,
                &self.inv_freq,
                p,
                self.rope_delta,
                &mut self.decode_rope_cos,
                &mut self.decode_rope_sin,
            );
        }
        let kv_dim = self.kv_dim();
        let max_attn = self.max_attn_tokens;
        for (li, layer) in self.layers.iter().enumerate() {
            layer.forward_one(
                &mut self.work_hidden,
                &mut self.work_q,
                &mut self.work_k,
                &mut self.work_v,
                &mut self.work_attn,
                &mut self.work_scratch,
                &mut self.work_gate,
                &mut self.work_up,
                &mut self.work_qkv,
                &mut self.work_gate_up,
                &mut self.attn_weights,
                &mut self.work_kv_head,
                &mut self.kv[li],
                kv_dim,
                &self.decode_rope_cos,
                &self.decode_rope_sin,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.norm_eps,
                max_attn,
            )?;
        }
        rms_norm1_into(&self.work_hidden, &self.norm_weight, self.norm_eps, out)?;
        self.past_len += 1;
        Ok(())
    }

    pub fn decode_step(&mut self, embed: ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut out = vec![0f32; self.hidden];
        self.decode_step_into(embed, &mut out)?;
        Ok(Array1::from_vec(out))
    }
}

/// External KV cache for the speculative early-exit draft path. Mirrors
/// `LayerKv` but holds only the first `n_layers` of the talker's KV — sized
/// independently of the talker's own KV so the draft can roll back without
/// touching the verifier state.
#[cfg(feature = "speculative-decode")]
pub struct DraftKvCache {
    layers: Vec<LayerKv>,
}

#[cfg(feature = "speculative-decode")]
impl DraftKvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers)
                .map(|_| LayerKv {
                    k: Vec::new(),
                    v: Vec::new(),
                })
                .collect(),
        }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn reset(&mut self) {
        for kv in &mut self.layers {
            kv.k.clear();
            kv.v.clear();
        }
    }

    pub fn past_len(&self, kv_dim: usize) -> usize {
        if let Some(l) = self.layers.first() {
            kv_rows(&l.k, kv_dim)
        } else {
            0
        }
    }

    pub fn rollback(&mut self, n: usize, kv_dim: usize) {
        let p = self.past_len(kv_dim);
        let target = p.saturating_sub(n);
        for kv in &mut self.layers {
            kv.k.truncate(target * kv_dim);
            kv.v.truncate(target * kv_dim);
        }
    }

    /// Internal-only mutable access to the per-layer KV vec. Required by
    /// `LearnedDraft::decode_step` to hand `&mut LayerKv` to
    /// `TalkerLayer::forward_one`.
    pub(crate) fn layer_kv_mut(&mut self, i: usize) -> &mut LayerKv {
        &mut self.layers[i]
    }
}

// Speculative-decode support: batched K+1 decode and KV rollback. Kept in a
// separate impl block so the feature surface area is isolated and reviewable.
#[cfg(feature = "speculative-decode")]
impl TalkerEagerModel {
    /// Batched decode of `m = embeds.nrows()` new tokens. Returns the final
    /// hidden rows `[m, hidden]` *after* `model.norm` — the same projection
    /// surface as [`Self::decode_step`] — and advances `past_len` by `m`,
    /// appending `m` rows to every per-layer K/V cache.
    ///
    /// Positions used for RoPE are `[past_len + rope_delta + i]` for
    /// `i in 0..m`, matching the per-step decode path. The causal mask is
    /// implicit in [`gqa_attention`] via `kv_off = t_k - t_q = past_len`.
    ///
    /// This is the "verifier" half of speculative decoding: callers pass
    /// `K + 1` embeddings (last-accepted-input + K drafted-inputs), get back
    /// `K + 1` hidden rows, and turn those into `K + 1` next-g0 distributions
    /// via the talker's `lm_head`. If only the first `n_accept ≤ K` drafts
    /// agree, the caller must call [`Self::rollback_kv`] with
    /// `K - n_accept` to undo the unused tail.
    ///
    /// Uses the stacked `wqkv` + `gate_up` weights so each layer issues 4
    /// sgemms (`wqkv`, `wo`, `gate_up`, `down`) instead of the prefill path's
    /// 7 (`wq`/`wk`/`wv` + `wo` + `gate`/`up`/`down`). Important at small
    /// batch sizes (`K + 1 = 2..8`) where sgemm overhead dominates.
    pub fn decode_batched(&mut self, embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (m, h) = embeds.dim();
        ensure!(h == self.hidden, "embed hidden {} != {}", h, self.hidden);
        ensure!(m >= 1, "decode_batched needs >= 1 input row");
        let p0 = self.past_len;
        let positions: Vec<usize> = (0..m)
            .map(|i| (p0 as i64 + i as i64 + self.rope_delta).max(0) as usize)
            .collect();
        let kv_dim = self.kv_dim();
        let mut x = embeds.to_owned();
        for (li, layer) in self.layers.iter().enumerate() {
            x = layer.forward_decode_batched(
                x.view(),
                &mut self.kv[li],
                kv_dim,
                &self.talker_cfg,
                &positions,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.head_half,
                self.norm_eps,
            )?;
        }
        x = rms_norm2(x.view(), &self.norm_weight, self.norm_eps);
        self.past_len += m;
        Ok(x)
    }

    /// Discard the last `n` rows of every per-layer K/V cache and roll
    /// `past_len` back by `n`. Clamped to `[0, past_len]`.
    ///
    /// Used by the speculative loop to undo K/V rows that came from drafted
    /// tokens the verifier rejected. Safe to call with `n = 0` (no-op).
    ///
    /// Note: this does NOT alter `rope_delta` — the decode RoPE bank uses
    /// `past_len + rope_delta`, so the next decode-step picks up the right
    /// position automatically once `past_len` is rewound.
    pub fn rollback_kv(&mut self, n: usize) {
        let n = n.min(self.past_len);
        if n == 0 {
            return;
        }
        let kv_dim = self.kv_dim();
        for kv in &mut self.kv {
            let keep = (self.past_len - n) * kv_dim;
            kv.k.truncate(keep);
            kv.v.truncate(keep);
        }
        self.past_len -= n;
    }

    pub fn past_len(&self) -> usize {
        self.past_len
    }

    /// Number of transformer layers — used by EarlyExitDraft to cap the
    /// requested draft depth.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn kv_dim_for_draft(&self) -> usize {
        self.kv_dim()
    }

    /// Run input embed through the first `kv.n_layers()` transformer layers
    /// using an *external* draft KV cache. Position is the absolute talker
    /// position (= `kv.past_len()` before this call). Returns the hidden
    /// state after `model.norm`, suitable for application of the codec head.
    ///
    /// **Does not** touch the model's own KV cache or `past_len`. The
    /// caller manages `kv` independently.
    pub fn early_exit_decode_step(
        &mut self,
        embed: &[f32],
        kv: &mut DraftKvCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        ensure!(embed.len() == self.hidden, "embed len mismatch");
        let n_layers = kv.layers.len();
        ensure!(
            n_layers <= self.layers.len(),
            "draft n_layers {} > talker layers {}",
            n_layers,
            self.layers.len()
        );

        // Decode RoPE for `position`. Reuses the talker's precomputed bank
        // when the position is in range.
        let half = self.head_half;
        let need = (position + 1) * half;
        if self.decode_rope_cos_bank.len() < need {
            self.warm_decode_rope_bank();
        }
        if position < self.max_attn_tokens && self.decode_rope_cos_bank.len() >= need {
            let roff = position * half;
            self.decode_rope_cos
                .copy_from_slice(&self.decode_rope_cos_bank[roff..roff + half]);
            self.decode_rope_sin
                .copy_from_slice(&self.decode_rope_sin_bank[roff..roff + half]);
        } else {
            talker_decode_rope_into(
                &self.talker_cfg,
                &self.inv_freq,
                position,
                self.rope_delta,
                &mut self.decode_rope_cos,
                &mut self.decode_rope_sin,
            );
        }

        self.work_hidden.copy_from_slice(embed);
        let kv_dim = self.kv_dim();
        let max_attn = self.max_attn_tokens;
        for li in 0..n_layers {
            self.layers[li].forward_one(
                &mut self.work_hidden,
                &mut self.work_q,
                &mut self.work_k,
                &mut self.work_v,
                &mut self.work_attn,
                &mut self.work_scratch,
                &mut self.work_gate,
                &mut self.work_up,
                &mut self.work_qkv,
                &mut self.work_gate_up,
                &mut self.attn_weights,
                &mut self.work_kv_head,
                &mut kv.layers[li],
                kv_dim,
                &self.decode_rope_cos,
                &self.decode_rope_sin,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.norm_eps,
                max_attn,
            )?;
        }
        let mut out = vec![0f32; self.hidden];
        rms_norm1_into(
            &self.work_hidden,
            &self.norm_weight,
            self.norm_eps,
            &mut out,
        )?;
        Ok(out)
    }
}

impl TalkerLayer {
    /// Single-token decode forward (no alloc; fused QKV + gate/up matvecs).
    pub(crate) fn forward_one(
        &self,
        x: &mut [f32],
        _q: &mut [f32],
        _k: &mut [f32],
        _v: &mut [f32],
        attn_out: &mut [f32],
        scratch: &mut [f32],
        _gate: &mut [f32],
        _up: &mut [f32],
        work_qkv: &mut [f32],
        work_gate_up: &mut [f32],
        attn_weights: &mut [f32],
        kv_head_scratch: &mut [f32],
        kv: &mut LayerKv,
        kv_dim: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        max_attn: usize,
    ) -> Result<()> {
        let hidden = x.len();
        rms_norm1_into(x, &self.attn_norm, eps, scratch)?;
        matvec_into(self.wqkv.view(), scratch, work_qkv)?;
        let (q_part, kv_part) = work_qkv.split_at_mut(self.q_dim);
        let (k_part, v_part) = kv_part.split_at_mut(self.kv_dim);
        qk_norm_heads1(q_part, &self.q_norm, n_heads, head_dim, eps);
        qk_norm_heads1(k_part, &self.k_norm, n_kv_heads, head_dim, eps);
        apply_rope1_flat(q_part, n_heads, head_dim, rope_cos, rope_sin);
        apply_rope1_flat(k_part, n_kv_heads, head_dim, rope_cos, rope_sin);
        append_kv1(kv, kv_dim, k_part, v_part);
        let t_k = kv_rows(&kv.k, kv_dim);
        gqa_attention1_into(
            q_part,
            &kv.k,
            &kv.v,
            t_k,
            kv_dim,
            attn_out,
            attn_weights,
            kv_head_scratch,
            max_attn,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        linear1_into(attn_out, self.wo.view(), scratch)?;
        for i in 0..hidden {
            scratch[i] += x[i];
        }
        rms_norm1_into(scratch, &self.ffn_norm, eps, x)?;
        matvec_into(self.gate_up.view(), x, work_gate_up)?;
        let (gate_part, up_part) = work_gate_up.split_at_mut(self.inter_dim);
        for i in 0..self.inter_dim {
            gate_part[i] = silu1(gate_part[i]) * up_part[i];
        }
        linear1_into(gate_part, self.down.view(), x)?;
        for i in 0..hidden {
            x[i] += scratch[i];
        }
        Ok(())
    }

    /// Last-token hidden after attention residual (before FFN).
    pub fn forward_attn_residual(
        &self,
        x: ArrayView2<f32>,
        kv: &mut LayerKv,
        kv_dim: usize,
        talker: &TalkerConfig,
        positions: &[usize],
        start_pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        head_half: usize,
        eps: f32,
        decode_rope: Option<(&[f32], &[f32])>,
    ) -> Result<Array2<f32>> {
        let h = rms_norm2(x, &self.attn_norm, eps);
        let mut q = linear2(h.view(), self.wq.view());
        let mut k = linear2(h.view(), self.wk.view());
        let v = linear2(h.view(), self.wv.view());
        qk_norm_heads(&mut q, &self.q_norm, n_heads, head_dim, eps);
        qk_norm_heads(&mut k, &self.k_norm, n_kv_heads, head_dim, eps);
        if let Some((cos, sin)) = decode_rope {
            apply_rope_qk_precomputed(&mut q, &mut k, cos, sin, n_heads, n_kv_heads, head_dim);
        } else {
            apply_rope_qk(
                &mut q, &mut k, talker, positions, start_pos, n_heads, n_kv_heads, head_dim,
                head_half,
            );
        }
        append_kv(kv, kv_dim, &k, &v);
        let t_k = kv_rows(&kv.k, kv_dim);
        let k_view = ArrayView2::from_shape((t_k, kv_dim), &kv.k).expect("k shape");
        let v_view = ArrayView2::from_shape((t_k, kv_dim), &kv.v).expect("v shape");
        let attn = gqa_attention(q.view(), k_view, v_view, n_heads, n_kv_heads, head_dim);
        let attn_out = linear2(attn.view(), self.wo.view());
        Ok(x.to_owned() + attn_out)
    }

    fn forward(
        &self,
        x: ArrayView2<f32>,
        kv: &mut LayerKv,
        kv_dim: usize,
        talker: &TalkerConfig,
        positions: &[usize],
        start_pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        head_half: usize,
        eps: f32,
        decode_rope: Option<(&[f32], &[f32])>,
    ) -> Result<Array2<f32>> {
        let mut out = self.forward_attn_residual(
            x,
            kv,
            kv_dim,
            talker,
            positions,
            start_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            head_half,
            eps,
            decode_rope,
        )?;
        let h2 = rms_norm2(out.view(), &self.ffn_norm, eps);
        let gate = linear2(h2.view(), self.gate.view());
        let up = linear2(h2.view(), self.up.view());
        let ff = linear2((silu2(gate.view()) * up).view(), self.down.view());
        out = out + ff;
        Ok(out)
    }

    /// Multi-row decode forward that reuses the *stacked* `wqkv` and `gate_up`
    /// weights from the single-step path, so each layer does 4 sgemms
    /// (`wqkv`, `wo`, `gate_up`, `down`) instead of 7. Position semantics
    /// match the standard decode path: positions come from `positions[i]`,
    /// the KV cache is **appended to** (not reset), and the implicit causal
    /// mask in [`gqa_attention`] is `kv_off = t_k - t_q = past_len`.
    #[cfg(feature = "speculative-decode")]
    fn forward_decode_batched(
        &self,
        x: ArrayView2<f32>,
        kv: &mut LayerKv,
        kv_dim: usize,
        talker: &TalkerConfig,
        positions: &[usize],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        head_half: usize,
        eps: f32,
    ) -> Result<Array2<f32>> {
        let h = rms_norm2(x, &self.attn_norm, eps);
        // Fused QKV: one sgemm produces [M, q_dim + 2*kv_dim].
        let qkv = linear2(h.view(), self.wqkv.view());
        let q_dim = self.q_dim;
        let kv_d = self.kv_dim;
        let (m, _) = qkv.dim();
        let mut q = Array2::<f32>::zeros((m, q_dim));
        let mut k = Array2::<f32>::zeros((m, kv_d));
        let mut v = Array2::<f32>::zeros((m, kv_d));
        for ti in 0..m {
            let row = qkv.row(ti);
            let row_slice = row.as_slice().expect("qkv row contiguous");
            let mut q_row = q.row_mut(ti);
            q_row
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&row_slice[..q_dim]);
            let mut k_row = k.row_mut(ti);
            k_row
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&row_slice[q_dim..q_dim + kv_d]);
            let mut v_row = v.row_mut(ti);
            v_row
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&row_slice[q_dim + kv_d..]);
        }
        qk_norm_heads(&mut q, &self.q_norm, n_heads, head_dim, eps);
        qk_norm_heads(&mut k, &self.k_norm, n_kv_heads, head_dim, eps);
        apply_rope_qk(
            &mut q, &mut k, talker, positions, 0, n_heads, n_kv_heads, head_dim, head_half,
        );
        append_kv(kv, kv_dim, &k, &v);
        let t_k = kv_rows(&kv.k, kv_dim);
        let k_view = ArrayView2::from_shape((t_k, kv_dim), &kv.k).expect("k shape");
        let v_view = ArrayView2::from_shape((t_k, kv_dim), &kv.v).expect("v shape");
        let attn = gqa_attention(q.view(), k_view, v_view, n_heads, n_kv_heads, head_dim);
        let attn_out = linear2(attn.view(), self.wo.view());
        let mut out = x.to_owned() + attn_out;
        // Fused gate_up: one sgemm produces [M, 2*inter_dim].
        let h2 = rms_norm2(out.view(), &self.ffn_norm, eps);
        let gu = linear2(h2.view(), self.gate_up.view());
        let inter = self.inter_dim;
        let mut act = Array2::<f32>::zeros((m, inter));
        for ti in 0..m {
            let row = gu.row(ti);
            let row_slice = row.as_slice().expect("gate_up row contiguous");
            let mut out_row = act.row_mut(ti);
            let out_slice = out_row.as_slice_mut().unwrap();
            for j in 0..inter {
                let g = row_slice[j];
                let u = row_slice[inter + j];
                out_slice[j] = (g / (1.0 + (-g).exp())) * u;
            }
        }
        let ff = linear2(act.view(), self.down.view());
        out = out + ff;
        Ok(out)
    }
}

fn apply_rope_qk_precomputed(
    q: &mut Array2<f32>,
    k: &mut Array2<f32>,
    cos: &[f32],
    sin: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    for hi in 0..n_heads {
        rotate_head(q, 0, hi * head_dim, cos, sin, head_dim);
    }
    for hi in 0..n_kv_heads {
        rotate_head(k, 0, hi * head_dim, cos, sin, head_dim);
    }
}

fn apply_rope_qk(
    q: &mut Array2<f32>,
    k: &mut Array2<f32>,
    talker: &TalkerConfig,
    positions: &[usize],
    start_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    head_half: usize,
) {
    let seq = q.nrows();
    for ti in 0..seq {
        let pos = if ti < positions.len() {
            positions[ti]
        } else {
            start_pos + ti
        };
        let (cos, sin) = talker_rope_half(talker, pos, head_half);
        for hi in 0..n_heads {
            rotate_head(q, ti, hi * head_dim, &cos, &sin, head_dim);
        }
        for hi in 0..n_kv_heads {
            rotate_head(k, ti, hi * head_dim, &cos, &sin, head_dim);
        }
    }
}

fn stack_proj_weights(parts: &[Array2<f32>]) -> Array2<f32> {
    let out_dim: usize = parts.iter().map(|w| w.nrows()).sum();
    let in_dim = parts[0].ncols();
    let mut out = Array2::<f32>::zeros((out_dim, in_dim));
    // Row-major + contiguous: each row is a memcpy, not a scalar loop.
    let out_slice = out.as_slice_mut().expect("contiguous");
    let mut row = 0usize;
    for w in parts {
        debug_assert_eq!(w.ncols(), in_dim);
        let src = w.as_slice().expect("contiguous part");
        let n = w.nrows() * in_dim;
        out_slice[row * in_dim..row * in_dim + n].copy_from_slice(src);
        row += w.nrows();
    }
    out
}

pub(crate) fn load_layer(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    i: usize,
) -> Result<TalkerLayer> {
    let p = format!("model.layers.{i}");
    let wq = take2d(map, &format!("{p}.self_attn.q_proj.weight"))?;
    let wk = take2d(map, &format!("{p}.self_attn.k_proj.weight"))?;
    let wv = take2d(map, &format!("{p}.self_attn.v_proj.weight"))?;
    let gate = take2d(map, &format!("{p}.mlp.gate_proj.weight"))?;
    let up = take2d(map, &format!("{p}.mlp.up_proj.weight"))?;
    let q_dim = wq.nrows();
    let kv_dim = wk.nrows();
    let inter_dim = gate.nrows();
    Ok(TalkerLayer {
        wqkv: stack_proj_weights(&[wq.clone(), wk.clone(), wv.clone()]),
        gate_up: stack_proj_weights(&[gate.clone(), up.clone()]),
        wq,
        wk,
        wv,
        wo: take2d(map, &format!("{p}.self_attn.o_proj.weight"))?,
        q_norm: take1d(map, &format!("{p}.self_attn.q_norm.weight"))?,
        k_norm: take1d(map, &format!("{p}.self_attn.k_norm.weight"))?,
        attn_norm: take1d(map, &format!("{p}.input_layernorm.weight"))?,
        ffn_norm: take1d(map, &format!("{p}.post_attention_layernorm.weight"))?,
        gate,
        up,
        down: take2d(map, &format!("{p}.mlp.down_proj.weight"))?,
        q_dim,
        kv_dim,
        inter_dim,
    })
}

fn take2d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array2<f32>> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 2);
    Array2::from_shape_vec((shape[0], shape[1]), data.clone()).with_context(|| key.to_string())
}

pub(crate) fn take1d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Vec<f32>> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 1);
    Ok(data.clone())
}

fn linear2(x: ArrayView2<f32>, w: ArrayView2<f32>) -> Array2<f32> {
    // Safetensors / HF Linear weight is [out_features, in_features]; y = x @ W^T.
    x.dot(&w.t())
}

fn rms_norm2(x: ArrayView2<f32>, weight: &[f32], eps: f32) -> Array2<f32> {
    let (t, d) = x.dim();
    let mut out = Array2::<f32>::zeros((t, d));
    for i in 0..t {
        let row = x.row(i);
        let mut sum = 0f32;
        for v in row.iter() {
            sum += v * v;
        }
        let inv = 1.0 / (sum / d as f32 + eps).sqrt();
        for j in 0..d {
            out[[i, j]] = row[j] * inv * weight[j];
        }
    }
    out
}

fn qk_norm_heads(q: &mut Array2<f32>, weight: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    let t = q.nrows();
    for ti in 0..t {
        for h in 0..n_heads {
            let off = h * head_dim;
            let mut sum = 0f32;
            for di in 0..head_dim {
                sum += q[[ti, off + di]] * q[[ti, off + di]];
            }
            let inv = 1.0 / (sum / head_dim as f32 + eps).sqrt();
            for di in 0..head_dim {
                q[[ti, off + di]] *= inv * weight[di];
            }
        }
    }
}

fn silu2(x: ArrayView2<f32>) -> Array2<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

/// HF `apply_rotary_pos_emb` on one attention head (`cos`/`sin` length `head_dim/2`).
fn rotate_head(
    x: &mut Array2<f32>,
    row: usize,
    col_off: usize,
    cos: &[f32],
    sin: &[f32],
    head_dim: usize,
) {
    let half = head_dim / 2;
    for i in 0..half {
        let c = cos[i];
        let s = sin[i];
        let x0 = x[[row, col_off + i]];
        let x1 = x[[row, col_off + half + i]];
        x[[row, col_off + i]] = x0 * c - x1 * s;
        x[[row, col_off + half + i]] = x1 * c + x0 * s;
    }
}

pub(crate) fn rms_norm1_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
    ensure!(x.len() == out.len() && x.len() == weight.len());
    let d = x.len();
    let mut sum = 0f32;
    for v in x {
        sum += v * v;
    }
    let inv = 1.0 / (sum / d as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = x[i] * inv * weight[i];
    }
    Ok(())
}

fn linear1_into(x: &[f32], w: ArrayView2<f32>, out: &mut [f32]) -> Result<()> {
    matvec_into(w, x, out)
}

fn silu1(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn qk_norm_heads1(q: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let off = h * head_dim;
        let mut sum = 0f32;
        for di in 0..head_dim {
            let v = q[off + di];
            sum += v * v;
        }
        let inv = 1.0 / (sum / head_dim as f32 + eps).sqrt();
        for di in 0..head_dim {
            q[off + di] *= inv * weight[di];
        }
    }
}

fn apply_rope1_flat(x: &mut [f32], n_heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let c = cos[i];
            let s = sin[i];
            let x0 = x[off + i];
            let x1 = x[off + half + i];
            x[off + i] = x0 * c - x1 * s;
            x[off + half + i] = x0 * s + x1 * c;
        }
    }
}

fn append_kv1(kv: &mut LayerKv, dim: usize, k: &[f32], v: &[f32]) {
    debug_assert_eq!(k.len(), dim);
    debug_assert_eq!(v.len(), dim);
    kv.k.extend_from_slice(k);
    kv.v.extend_from_slice(v);
}

fn append_kv(kv: &mut LayerKv, dim: usize, k: &Array2<f32>, v: &Array2<f32>) {
    let (t_new, d) = k.dim();
    debug_assert_eq!(d, dim);
    debug_assert_eq!(v.dim(), (t_new, dim));
    for row in 0..t_new {
        kv.k.extend_from_slice(k.row(row).as_slice().unwrap());
        kv.v.extend_from_slice(v.row(row).as_slice().unwrap());
    }
}

fn gqa_attention(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Array2<f32> {
    let t_q = q.nrows();
    let t_k = k.nrows();
    let kv_off = t_k.saturating_sub(t_q);
    let repeats = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut out = Array2::<f32>::zeros((t_q, n_heads * head_dim));
    for qi in 0..t_q {
        let kq = qi + kv_off;
        for hi in 0..n_heads {
            let kv_h = hi / repeats;
            let mut scores = vec![0f64; kq + 1];
            for ki in 0..=kq {
                let mut dot = 0f64;
                for di in 0..head_dim {
                    dot += f64::from(q[[qi, hi * head_dim + di]])
                        * f64::from(k[[ki, kv_h * head_dim + di]]);
                }
                scores[ki] = dot * scale;
            }
            let max_w = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0f64;
            let mut weights = vec![0f32; kq + 1];
            for (ki, s) in scores.iter().enumerate() {
                let ew = (s - max_w).exp();
                weights[ki] = ew as f32;
                sum += ew;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for w in &mut weights {
                *w = (*w as f64 * inv) as f32;
            }
            for di in 0..head_dim {
                let mut acc = 0f64;
                for ki in 0..=kq {
                    acc += f64::from(weights[ki]) * f64::from(v[[ki, kv_h * head_dim + di]]);
                }
                out[[qi, hi * head_dim + di]] = acc as f32;
            }
        }
    }
    out
}
