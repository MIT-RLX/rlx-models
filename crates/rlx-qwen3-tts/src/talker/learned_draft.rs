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

//! Learned draft model for talker speculative decoding —
//! `cfg(feature = "speculative-decode")`.
//!
//! Holds an *independent* stack of Qwen3-shaped transformer layers loaded
//! from a separate safetensors file. Same per-layer architecture as the
//! verifier talker (`TalkerLayer`) and the same head/embedding tables (which
//! are reused from the verifier — no need to ship duplicates) — only the
//! transformer-layer weights differ.
//!
//! # Why a separate file
//!
//! The draft's predictive value depends on it being trained to *imitate the
//! verifier's argmax* under matched inputs. The early-exit experiment
//! (`SpecConfig::early_exit_layers`) showed that using the verifier's own
//! first N untrained-as-draft layers gives ~0% acceptance on this model
//! (Qwen3-TTS Base, sharp codec g0 distributions). A trained draft that
//! distills the verifier's next-token distribution onto a smaller stack is
//! the mechanism that's expected to push acceptance into the
//! breakeven-and-above zone.
//!
//! # Weight layout
//!
//! The safetensors file is expected to follow the HF convention used
//! elsewhere in this crate:
//!
//! - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` for `i` in `0..N`
//! - `model.layers.{i}.self_attn.{q,k}_norm.weight`
//! - `model.layers.{i}.input_layernorm.weight`
//! - `model.layers.{i}.post_attention_layernorm.weight`
//! - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
//! - `model.norm.weight`
//!
//! No `lm_head.weight` is required — the verifier's `codec_head` is reused
//! to map draft hidden states to g0 logits. Likewise, no `embed_tokens` /
//! `codec_embedding` table is required — the verifier's `codec_embedding`
//! handles input embedding. This keeps the draft file small (~`N` × per-
//! layer cost) and avoids any drift between the draft's input/output
//! representations and the verifier's.
//!
//! # Hidden-size constraint (v1)
//!
//! The v1 loader requires the draft to share `hidden_size`, `head_dim`,
//! `num_attention_heads`, and `num_key_value_heads` with the verifier
//! talker. That lets the draft consume `codec_emb_t` directly and emit
//! hiddens that go straight into the verifier's `codec_head` for sampling.
//! A smaller-hidden draft requires up/down projection layers at the I/O
//! boundary, which is a v2 extension.

use anyhow::{Context, Result, ensure};
use ndarray::ArrayView2;
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use std::collections::HashSet;
use std::path::Path;

use crate::config::TalkerConfig;
use crate::mrope::talker_decode_rope_into;
use crate::talker::eager::{DraftKvCache, TalkerLayer, load_layer, rms_norm1_into, take1d};
use crate::talker::rope::build_inv_freq;

/// A small (N-layer) transformer that drafts future g0 tokens for the
/// talker's speculative-decoding loop.
pub struct LearnedDraft {
    layers: Vec<TalkerLayer>,
    norm_weight: Vec<f32>,
    kv: DraftKvCache,
    talker_cfg: TalkerConfig,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    norm_eps: f32,
    inv_freq: Vec<f64>,
    rope_delta: i64,
    // Work buffers — sized once at open() so the per-step path is alloc-free.
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
    decode_rope_cos: Vec<f32>,
    decode_rope_sin: Vec<f32>,
    max_attn_tokens: usize,
}

impl LearnedDraft {
    /// Open a learned draft from a safetensors directory. The directory must
    /// contain `model.safetensors` (or the canonical name used elsewhere in
    /// this crate) with `n_layers` worth of HF-style `model.layers.{i}.*`
    /// keys plus a `model.norm.weight`.
    ///
    /// `talker_cfg` is the *verifier* talker's config — it's reused to size
    /// attention heads, head_dim, hidden_size, RoPE base, and norm_eps so
    /// the draft can drop into the verifier's KV/embedding/head plumbing
    /// without conversion.
    pub fn open(model_dir: &Path, talker_cfg: &TalkerConfig, n_layers: usize) -> Result<Self> {
        ensure!(n_layers >= 1, "learned draft needs at least 1 layer");
        let checkpoint = SafetensorsCheckpoint::open(model_dir)
            .with_context(|| format!("open learned-draft safetensors at {model_dir:?}"))?;
        let mut want: HashSet<String> = HashSet::new();
        for i in 0..n_layers {
            let p = format!("model.layers.{i}");
            want.insert(format!("{p}.self_attn.q_proj.weight"));
            want.insert(format!("{p}.self_attn.k_proj.weight"));
            want.insert(format!("{p}.self_attn.v_proj.weight"));
            want.insert(format!("{p}.self_attn.o_proj.weight"));
            want.insert(format!("{p}.self_attn.q_norm.weight"));
            want.insert(format!("{p}.self_attn.k_norm.weight"));
            want.insert(format!("{p}.input_layernorm.weight"));
            want.insert(format!("{p}.post_attention_layernorm.weight"));
            want.insert(format!("{p}.mlp.gate_proj.weight"));
            want.insert(format!("{p}.mlp.up_proj.weight"));
            want.insert(format!("{p}.mlp.down_proj.weight"));
        }
        want.insert("model.norm.weight".into());
        let mut wm = checkpoint
            .load_selected(&want)
            .with_context(|| "load_selected for learned draft")?;
        let mut map = std::collections::HashMap::with_capacity(want.len());
        for k in want.iter() {
            let v = wm
                .take(k)
                .with_context(|| format!("missing draft weight {k}"))?;
            map.insert(k.clone(), v);
        }

        Self::from_map(map, talker_cfg, n_layers)
    }

    /// Build from an already-parsed weight map.
    pub fn from_map(
        map: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
        talker_cfg: &TalkerConfig,
        n_layers: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(load_layer(&map, i)?);
        }
        let norm_weight = take1d(&map, "model.norm.weight")?;
        let hidden = talker_cfg.hidden_size;
        let head_dim = talker_cfg.head_dim;
        let head_half = head_dim / 2;
        let q_dim = talker_cfg.num_attention_heads * head_dim;
        let kv_dim = talker_cfg.num_key_value_heads * head_dim;
        let inter_dim = talker_cfg.intermediate_size;
        let max_attn_tokens = 256usize;

        Ok(Self {
            layers,
            norm_weight,
            kv: DraftKvCache::new(n_layers),
            talker_cfg: talker_cfg.clone(),
            n_heads: talker_cfg.num_attention_heads,
            n_kv_heads: talker_cfg.num_key_value_heads,
            head_dim,
            hidden,
            norm_eps: talker_cfg.rms_norm_eps as f32,
            inv_freq: build_inv_freq(head_dim, talker_cfg.rope_theta),
            rope_delta: 0,
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
            attn_weights: vec![0f32; talker_cfg.num_attention_heads * max_attn_tokens],
            work_kv_head: vec![
                0f32;
                max_attn_tokens
                    * (2 * head_dim
                        + talker_cfg.num_attention_heads
                            / talker_cfg.num_key_value_heads.max(1))
            ],
            decode_rope_cos: vec![0f32; head_half],
            decode_rope_sin: vec![0f32; head_half],
            max_attn_tokens,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub fn reset_kv(&mut self) {
        self.kv.reset();
        self.rope_delta = 0;
    }

    pub fn past_len(&self) -> usize {
        let kv_dim = self.n_kv_heads * self.head_dim;
        self.kv.past_len(kv_dim)
    }

    pub fn rollback_kv(&mut self, n: usize) {
        let kv_dim = self.n_kv_heads * self.head_dim;
        self.kv.rollback(n, kv_dim);
    }

    pub fn set_rope_delta(&mut self, delta: i64) {
        self.rope_delta = delta;
    }

    /// One-step forward through all draft layers at `position`. Appends one
    /// row to the draft KV cache and returns the post-`model.norm` hidden
    /// state (`Vec<f32>` of length `hidden_size`).
    pub fn decode_step(&mut self, embed: &[f32], position: usize) -> Result<Vec<f32>> {
        ensure!(embed.len() == self.hidden, "embed len mismatch");

        talker_decode_rope_into(
            &self.talker_cfg,
            &self.inv_freq,
            position,
            self.rope_delta,
            &mut self.decode_rope_cos,
            &mut self.decode_rope_sin,
        );

        self.work_hidden.copy_from_slice(embed);
        let kv_dim = self.n_kv_heads * self.head_dim;
        let max_attn = self.max_attn_tokens;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;
        let eps = self.norm_eps;
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
                self.kv.layer_kv_mut(li),
                kv_dim,
                &self.decode_rope_cos,
                &self.decode_rope_sin,
                n_heads,
                n_kv_heads,
                head_dim,
                eps,
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

    /// Sync the draft KV to the verifier's prefill state by replaying each
    /// prefill row through `decode_step`. One-time cost at the start of an
    /// utterance; subsequent commits stay in sync via the spec loop.
    pub fn prefill_sync(&mut self, prefill_embeds: ArrayView2<f32>) -> Result<()> {
        let n = prefill_embeds.nrows();
        for r in 0..n {
            let row = prefill_embeds.row(r);
            let _ = self.decode_step(row.as_slice().expect("contiguous"), r)?;
        }
        Ok(())
    }
}
