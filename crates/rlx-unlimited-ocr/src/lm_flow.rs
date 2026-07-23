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

//! MoE decoder graph — dense early layers ([`crate::config::UnlimitedOcrConfig::first_k_dense_replace`]),
//! routed + shared experts afterwards, rolling sliding-window attention.
//!
//! Eager host-f32 port of `modeling_deepseekv2.py`'s `use_mla=False` path
//! (`SlidingWindowLlamaAttention` / `mha_eager`): plain multi-head attention
//! (`q_proj`/`k_proj`/`v_proj`/`o_proj`, no bias, rotate-half RoPE, θ=10000,
//! `head_dim=128`), `DeepseekV2MoE` routing (softmax gate, greedy top-k
//! `sorted=false`, `norm_topk_prob=false` so `weight = topk_prob *
//! routed_scaling_factor` unchanged, `routed_scaling_factor=1.0` — this
//! checkpoint's `config.json` doesn't override either), and a
//! `shared_experts` MLP (`intermediate_size = moe_intermediate_size *
//! n_shared_experts`) whose output is *added* (not gated) to the routed sum.
//!
//! KV cache is a ring buffer matching `SlidingWindowLlamaAttention.forward`:
//! the full prefill is kept forever; the following `sliding_window` (128)
//! decode steps grow the cache (warmup); once `prefill_len + window` tokens
//! are cached, further decode steps overwrite ring slots in place. Attention
//! always runs over the *entire* current cache (prefill + ring), which is
//! naturally causal since evicted (overwritten) slots always hold tokens
//! older than the current query.

use crate::config::UnlimitedOcrConfig;
use crate::nn;
use crate::weights::{UnlimitedOcrWeightPrefix, UnlimitedOcrWeightStore};
use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;

/// `MoEGate.routed_scaling_factor` — not present in this checkpoint's
/// `config.json` (nor parsed by [`UnlimitedOcrConfig`]), so we use
/// `DeepseekV2Config`'s library default (1.0).
const ROUTED_SCALING_FACTOR: f32 = 1.0;
/// `MoEGate.norm_topk_prob` — same story, library default `False`.
const NORM_TOPK_PROB: bool = false;

/// Per-layer routing shape, precomputed from [`UnlimitedOcrConfig`].
#[derive(Debug, Clone, Copy)]
pub struct MoeLayerShape {
    pub is_dense: bool,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
}

/// One expert's (or the dense/shared MLP's) SwiGLU weights, all `[out, in]`
/// (PyTorch `nn.Linear` layout, no bias — `hidden_act = "silu"`).
struct SwiGluWeights {
    gate_w: Vec<f32>,
    up_w: Vec<f32>,
    down_w: Vec<f32>,
}

impl SwiGluWeights {
    /// `x` is `[rows, hidden]`; returns `[rows, hidden]`.
    fn forward(&self, x: &[f32], rows: usize, hidden: usize) -> Result<Vec<f32>> {
        let inter = self.gate_w.len() / hidden;
        let mut gate = nn::linear_wt(x, rows, hidden, &self.gate_w, inter, None)?;
        let up = nn::linear_wt(x, rows, hidden, &self.up_w, inter, None)?;
        nn::silu(&mut gate);
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g *= *u;
        }
        nn::linear_wt(&gate, rows, inter, &self.down_w, hidden, None)
    }
}

struct LmLayerWeights {
    input_ln: Vec<f32>,
    post_ln: Vec<f32>,
    q_w: Vec<f32>,
    k_w: Vec<f32>,
    v_w: Vec<f32>,
    o_w: Vec<f32>,
    is_dense: bool,
    dense_mlp: Option<SwiGluWeights>,
    moe_gate_w: Option<Vec<f32>>, // [n_routed_experts, hidden]
    shared_experts: Option<SwiGluWeights>,
    experts: Vec<SwiGluWeights>, // len == n_routed_experts for MoE layers
}

/// One decoder layer's running KV state.
struct LayerKv {
    /// `[cur_len, num_heads, head_dim]` row-major (`num_heads*head_dim == hidden`).
    k: Vec<f32>,
    v: Vec<f32>,
    /// Number of tokens cached from prefill (never evicted).
    prefill_len: usize,
    /// `None` until the ring region (`prefill_len..prefill_len+window`) has
    /// been fully warmed up by cat-appending; `Some(next_slot_offset)` once
    /// steady-state in-place ring overwrites begin.
    ring_pos: Option<usize>,
}

/// Per-layer KV cache handle, populated by [`LmFlow::prefill`] and mutated
/// by [`LmFlow::decode_step`].
#[derive(Default)]
pub struct KvCache {
    layers: Vec<LayerKv>,
}

/// Uncompiled DeepSeek-V2-MoE decoder graph + weights.
pub struct LmFlow {
    pub config: UnlimitedOcrConfig,
    embed_tokens: Option<Vec<f32>>, // [vocab, hidden]
    norm: Option<Vec<f32>>,
    lm_head: Option<Vec<f32>>, // [vocab, hidden]
    layers: Vec<LmLayerWeights>,
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
}

impl LmFlow {
    pub fn from_config(config: UnlimitedOcrConfig) -> Self {
        let head_dim = config.head_dim();
        let (cos_table, sin_table) =
            nn::rope_tables(config.max_position_embeddings, head_dim, config.rope_theta);
        Self {
            config,
            embed_tokens: None,
            norm: None,
            lm_head: None,
            layers: Vec::new(),
            cos_table,
            sin_table,
        }
    }

    /// Token embedding lookup: `ids` → `[ids.len() * hidden_size]` (row-major).
    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let table = self.embed_tokens.as_ref().context("lm not loaded")?;
        let vocab = table.len() / hidden;
        let mut out = vec![0f32; ids.len() * hidden];
        for (i, &id) in ids.iter().enumerate() {
            ensure!(
                (id as usize) < vocab,
                "embed_tokens: id {id} out of range (vocab={vocab})"
            );
            let src = &table[id as usize * hidden..(id as usize + 1) * hidden];
            out[i * hidden..(i + 1) * hidden].copy_from_slice(src);
        }
        Ok(out)
    }

    /// Prefill the decoder over `inputs_embeds` (`[n_tokens, hidden_size]`),
    /// returning the last-position logits (`[vocab_size]`) and a fresh KV cache.
    pub fn prefill(&self, inputs_embeds: &[f32], n_tokens: usize) -> Result<(Vec<f32>, KvCache)> {
        let hidden = self.config.hidden_size;
        ensure!(
            inputs_embeds.len() == n_tokens * hidden,
            "prefill: inputs_embeds len {} != {n_tokens}*{hidden}",
            inputs_embeds.len()
        );
        let heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim();
        let positions: Vec<usize> = (0..n_tokens).collect();

        let mut x = inputs_embeds.to_vec();
        let mut kv = KvCache {
            layers: Vec::with_capacity(self.layers.len()),
        };
        for layer in &self.layers {
            let normed = nn::rms_norm(
                &x,
                n_tokens,
                hidden,
                &layer.input_ln,
                self.config.rms_norm_eps as f32,
            );
            let mut q = nn::linear_wt(&normed, n_tokens, hidden, &layer.q_w, hidden, None)?;
            let mut k = nn::linear_wt(&normed, n_tokens, hidden, &layer.k_w, hidden, None)?;
            let v = nn::linear_wt(&normed, n_tokens, hidden, &layer.v_w, hidden, None)?;
            nn::apply_rope(
                &mut q,
                n_tokens,
                heads,
                head_dim,
                &positions,
                &self.cos_table,
                &self.sin_table,
            );
            nn::apply_rope(
                &mut k,
                n_tokens,
                heads,
                head_dim,
                &positions,
                &self.cos_table,
                &self.sin_table,
            );

            let merged = attention(&q, n_tokens, &k, &v, n_tokens, heads, head_dim, 0);
            let attn_out = nn::linear_wt(&merged, n_tokens, hidden, &layer.o_w, hidden, None)?;
            nn::add_inplace(&mut x, &attn_out);

            let normed2 = nn::rms_norm(
                &x,
                n_tokens,
                hidden,
                &layer.post_ln,
                self.config.rms_norm_eps as f32,
            );
            let ffn = self.layer_mlp(layer, &normed2, n_tokens)?;
            nn::add_inplace(&mut x, &ffn);

            kv.layers.push(LayerKv {
                k,
                v,
                prefill_len: n_tokens,
                ring_pos: None,
            });
        }

        let normed_final = nn::rms_norm(
            &x,
            n_tokens,
            hidden,
            self.norm.as_ref().context("lm not loaded")?,
            self.config.rms_norm_eps as f32,
        );
        let last = &normed_final[(n_tokens - 1) * hidden..n_tokens * hidden];
        let logits = self.compute_logits(last)?;
        Ok((logits, kv))
    }

    /// Decode one step at absolute position `pos`, given the embedded next
    /// token (`[hidden_size]`) and the running KV cache from [`Self::prefill`].
    pub fn decode_step(
        &self,
        step_embed: &[f32],
        pos: usize,
        kv: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        ensure!(
            step_embed.len() == hidden,
            "decode_step: embed len {} != {hidden}",
            step_embed.len()
        );
        ensure!(
            kv.layers.len() == self.layers.len(),
            "decode_step: KV cache layer count mismatch"
        );
        let heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim();
        let window = if self.config.sliding_window == 0 {
            usize::MAX
        } else {
            self.config.sliding_window
        };

        let mut x = step_embed.to_vec();
        for (layer, layer_kv) in self.layers.iter().zip(kv.layers.iter_mut()) {
            let normed = nn::rms_norm(
                &x,
                1,
                hidden,
                &layer.input_ln,
                self.config.rms_norm_eps as f32,
            );
            let mut q = nn::linear_wt(&normed, 1, hidden, &layer.q_w, hidden, None)?;
            let mut k_new = nn::linear_wt(&normed, 1, hidden, &layer.k_w, hidden, None)?;
            let v_new = nn::linear_wt(&normed, 1, hidden, &layer.v_w, hidden, None)?;
            let positions = [pos];
            nn::apply_rope(
                &mut q,
                1,
                heads,
                head_dim,
                &positions,
                &self.cos_table,
                &self.sin_table,
            );
            nn::apply_rope(
                &mut k_new,
                1,
                heads,
                head_dim,
                &positions,
                &self.cos_table,
                &self.sin_table,
            );

            let cur_len = layer_kv.k.len() / hidden;
            if cur_len < layer_kv.prefill_len + window {
                layer_kv.k.extend_from_slice(&k_new);
                layer_kv.v.extend_from_slice(&v_new);
                let new_len = cur_len + 1;
                if new_len >= layer_kv.prefill_len + window {
                    layer_kv.ring_pos = Some(0);
                }
            } else {
                let ring_pos = layer_kv.ring_pos.unwrap_or(0);
                let slot = layer_kv.prefill_len + ring_pos;
                layer_kv.k[slot * hidden..(slot + 1) * hidden].copy_from_slice(&k_new);
                layer_kv.v[slot * hidden..(slot + 1) * hidden].copy_from_slice(&v_new);
                layer_kv.ring_pos = Some((ring_pos + 1) % window);
            }

            let n_k = layer_kv.k.len() / hidden;
            let merged = attention(
                &q,
                1,
                &layer_kv.k,
                &layer_kv.v,
                n_k,
                heads,
                head_dim,
                n_k - 1,
            );
            let attn_out = nn::linear_wt(&merged, 1, hidden, &layer.o_w, hidden, None)?;
            nn::add_inplace(&mut x, &attn_out);

            let normed2 = nn::rms_norm(
                &x,
                1,
                hidden,
                &layer.post_ln,
                self.config.rms_norm_eps as f32,
            );
            let ffn = self.layer_mlp(layer, &normed2, 1)?;
            nn::add_inplace(&mut x, &ffn);
        }

        let normed_final = nn::rms_norm(
            &x,
            1,
            hidden,
            self.norm.as_ref().context("lm not loaded")?,
            self.config.rms_norm_eps as f32,
        );
        self.compute_logits(&normed_final)
    }

    fn layer_mlp(&self, layer: &LmLayerWeights, normed: &[f32], rows: usize) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        if layer.is_dense {
            return layer
                .dense_mlp
                .as_ref()
                .context("lm not loaded")?
                .forward(normed, rows, hidden);
        }
        let gate_w = layer.moe_gate_w.as_ref().context("lm not loaded")?;
        let n_routed = layer.experts.len();
        let logits = nn::linear_wt(normed, rows, hidden, gate_w, n_routed, None)?;

        let mut out = vec![0f32; rows * hidden];
        for r in 0..rows {
            let row_logits = &logits[r * n_routed..(r + 1) * n_routed];
            let top = nn::topk_softmax(row_logits, self.config.num_experts_per_tok);
            let denom: f32 = if NORM_TOPK_PROB && top.len() > 1 {
                top.iter().map(|(_, w)| *w).sum::<f32>() + 1e-20
            } else {
                1.0
            };
            let x_row = &normed[r * hidden..(r + 1) * hidden];
            let dst = &mut out[r * hidden..(r + 1) * hidden];
            for (idx, w) in &top {
                let weight = if NORM_TOPK_PROB && top.len() > 1 {
                    (*w / denom) * ROUTED_SCALING_FACTOR
                } else {
                    *w * ROUTED_SCALING_FACTOR
                };
                let expert_out = layer.experts[*idx].forward(x_row, 1, hidden)?;
                for (d, v) in dst.iter_mut().zip(expert_out.iter()) {
                    *d += weight * *v;
                }
            }
            let shared_out = layer
                .shared_experts
                .as_ref()
                .context("lm not loaded")?
                .forward(x_row, 1, hidden)?;
            for (d, v) in dst.iter_mut().zip(shared_out.iter()) {
                *d += *v;
            }
        }
        Ok(out)
    }

    fn compute_logits(&self, hidden_row: &[f32]) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let lm_head = self.lm_head.as_ref().context("lm not loaded")?;
        let vocab = lm_head.len() / hidden;
        nn::linear_wt(hidden_row, 1, hidden, lm_head, vocab, None)
    }

    pub fn layer_shape(&self, layer_idx: usize) -> MoeLayerShape {
        MoeLayerShape {
            is_dense: self.config.is_dense_layer(layer_idx),
            n_routed_experts: self.config.n_routed_experts,
            n_shared_experts: self.config.n_shared_experts,
            num_experts_per_tok: self.config.num_experts_per_tok,
            moe_intermediate_size: self.config.moe_intermediate_size,
        }
    }

    pub fn load(&mut self, store: &UnlimitedOcrWeightStore) -> Result<()> {
        let mut globals = store.load_keys(&[
            UnlimitedOcrWeightPrefix::embed_tokens(),
            UnlimitedOcrWeightPrefix::lm_norm(),
            UnlimitedOcrWeightPrefix::lm_head(),
        ])?;
        self.embed_tokens = Some(
            globals
                .take(UnlimitedOcrWeightPrefix::embed_tokens())
                .context("embed_tokens")?
                .0,
        );
        self.norm = Some(
            globals
                .take(UnlimitedOcrWeightPrefix::lm_norm())
                .context("model.norm")?
                .0,
        );
        self.lm_head = Some(
            globals
                .take(UnlimitedOcrWeightPrefix::lm_head())
                .context("lm_head")?
                .0,
        );

        self.layers = Vec::with_capacity(self.config.num_hidden_layers);
        for i in 0..self.config.num_hidden_layers {
            let mut map = store.load_lm_layer(i)?;
            let take = |m: &mut WeightMap, key: String| -> Result<Vec<f32>> {
                Ok(m.take(&key)
                    .with_context(|| format!("lm layer {i}: {key}"))?
                    .0)
            };

            let input_ln = take(&mut map, UnlimitedOcrWeightPrefix::lm_input_layernorm(i))?;
            let post_ln = take(
                &mut map,
                UnlimitedOcrWeightPrefix::lm_post_attention_layernorm(i),
            )?;
            let q_w = take(&mut map, UnlimitedOcrWeightPrefix::lm_attn(i, "q_proj"))?;
            let k_w = take(&mut map, UnlimitedOcrWeightPrefix::lm_attn(i, "k_proj"))?;
            let v_w = take(&mut map, UnlimitedOcrWeightPrefix::lm_attn(i, "v_proj"))?;
            let o_w = take(&mut map, UnlimitedOcrWeightPrefix::lm_attn(i, "o_proj"))?;

            let is_dense = self.config.is_dense_layer(i);
            let (dense_mlp, moe_gate_w, shared_experts, experts) = if is_dense {
                let dense = SwiGluWeights {
                    gate_w: take(
                        &mut map,
                        UnlimitedOcrWeightPrefix::lm_dense_mlp(i, "gate_proj"),
                    )?,
                    up_w: take(
                        &mut map,
                        UnlimitedOcrWeightPrefix::lm_dense_mlp(i, "up_proj"),
                    )?,
                    down_w: take(
                        &mut map,
                        UnlimitedOcrWeightPrefix::lm_dense_mlp(i, "down_proj"),
                    )?,
                };
                (Some(dense), None, None, Vec::new())
            } else {
                let gate_w = take(&mut map, UnlimitedOcrWeightPrefix::lm_moe_gate(i))?;
                let shared = SwiGluWeights {
                    gate_w: take(
                        &mut map,
                        UnlimitedOcrWeightPrefix::lm_moe_shared_expert(i, "gate_proj"),
                    )?,
                    up_w: take(
                        &mut map,
                        UnlimitedOcrWeightPrefix::lm_moe_shared_expert(i, "up_proj"),
                    )?,
                    down_w: take(
                        &mut map,
                        UnlimitedOcrWeightPrefix::lm_moe_shared_expert(i, "down_proj"),
                    )?,
                };
                let n_experts = store.count_experts(i);
                ensure!(
                    n_experts == self.config.n_routed_experts,
                    "lm layer {i}: expert count {n_experts} != n_routed_experts {}",
                    self.config.n_routed_experts
                );
                let mut experts = Vec::with_capacity(n_experts);
                for e in 0..n_experts {
                    experts.push(SwiGluWeights {
                        gate_w: take(
                            &mut map,
                            UnlimitedOcrWeightPrefix::lm_moe_expert(i, e, "gate_proj"),
                        )?,
                        up_w: take(
                            &mut map,
                            UnlimitedOcrWeightPrefix::lm_moe_expert(i, e, "up_proj"),
                        )?,
                        down_w: take(
                            &mut map,
                            UnlimitedOcrWeightPrefix::lm_moe_expert(i, e, "down_proj"),
                        )?,
                    });
                }
                (None, Some(gate_w), Some(shared), experts)
            };

            self.layers.push(LmLayerWeights {
                input_ln,
                post_ln,
                q_w,
                k_w,
                v_w,
                o_w,
                is_dense,
                dense_mlp,
                moe_gate_w,
                shared_experts,
                experts,
            });
        }
        Ok(())
    }

    /// Convenience one-shot forward: prefill and return only the
    /// last-position logits, discarding the KV cache. Callers that need to
    /// keep decoding should call [`Self::prefill`] directly instead.
    pub fn forward(&self, inputs_embeds: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
        self.prefill(inputs_embeds, n_tokens)
            .map(|(logits, _kv)| logits)
    }
}

/// Causal multi-head attention: `n_q` queries (`[n_q, heads*head_dim]`)
/// against `n_k` keys/values (`[n_k, heads*head_dim]`). Query `qi` may
/// attend key `ki` iff `ki <= qi + q_offset` (prefill: `q_offset=0`,
/// `n_q==n_k`, plain causal; decode: `n_q==1`, `q_offset=n_k-1` so the
/// single new query sees the entire — already causally valid — cache).
fn attention(
    q: &[f32],
    n_q: usize,
    k: &[f32],
    v: &[f32],
    n_k: usize,
    heads: usize,
    head_dim: usize,
    q_offset: usize,
) -> Vec<f32> {
    let hidden = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut merged = vec![0f32; n_q * hidden];
    let mut row = vec![0f32; n_k];
    for head in 0..heads {
        let off = head * head_dim;
        for qi in 0..n_q {
            let qvec = &q[qi * hidden + off..qi * hidden + off + head_dim];
            let limit = (qi + q_offset + 1).min(n_k);
            for slot in row.iter_mut().take(limit) {
                *slot = 0.0;
            }
            for ki in 0..limit {
                let kvec = &k[ki * hidden + off..ki * hidden + off + head_dim];
                let dot: f32 = qvec.iter().zip(kvec.iter()).map(|(a, b)| a * b).sum();
                row[ki] = dot * scale;
            }
            for slot in row.iter_mut().skip(limit) {
                *slot = f32::NEG_INFINITY;
            }
            nn::softmax_rows(&mut row, 1, n_k);
            let dst = &mut merged[qi * hidden + off..qi * hidden + off + head_dim];
            for ki in 0..limit {
                let w = row[ki];
                if w == 0.0 {
                    continue;
                }
                let vvec = &v[ki * hidden + off..ki * hidden + off + head_dim];
                for d in 0..head_dim {
                    dst[d] += w * vvec[d];
                }
            }
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> UnlimitedOcrConfig {
        UnlimitedOcrConfig::from_json_str(
            r#"{
                "model_type": "unlimited-ocr",
                "first_k_dense_replace": 1,
                "n_routed_experts": 64,
                "n_shared_experts": 2,
                "num_experts_per_tok": 6,
                "moe_intermediate_size": 896
            }"#,
        )
        .expect("parse")
    }

    #[test]
    fn layer_zero_is_dense_rest_are_moe() {
        let flow = LmFlow::from_config(cfg());
        assert!(flow.layer_shape(0).is_dense);
        assert!(!flow.layer_shape(1).is_dense);
        assert_eq!(flow.layer_shape(1).n_routed_experts, 64);
    }

    #[test]
    fn attention_single_head_matches_manual_softmax() {
        // 1 head, head_dim=1, 2 keys: q=[1.0], k=[[1.0],[0.0]], v=[[10.0],[20.0]].
        let q = [1.0f32];
        let k = [1.0f32, 0.0];
        let v = [10.0f32, 20.0];
        let out = attention(&q, 1, &k, &v, 2, 1, 1, 1);
        // scale = 1.0; scores = [1*1, 1*0] = [1, 0]; softmax favors index 0.
        let s0 = 1.0f32.exp();
        let s1 = 0.0f32.exp();
        let w0 = s0 / (s0 + s1);
        let w1 = s1 / (s0 + s1);
        let expected = w0 * 10.0 + w1 * 20.0;
        assert!((out[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn attention_causal_offset_hides_future_keys() {
        // 1 head, head_dim=1, query 0 with offset 0 (prefill row 0) can only see key 0.
        let q = [1.0f32];
        let k = [1.0f32, 5.0];
        let v = [10.0f32, 999.0];
        let out = attention(&q, 1, &k, &v, 2, 1, 1, 0);
        assert!((out[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn forward_matches_prefill_last_position_logits() {
        // `forward()` is a bail-free convenience wrapper — before weights are
        // loaded, `compute_logits` errors identically through both paths.
        let flow = LmFlow::from_config(cfg());
        let inputs = vec![0f32; flow.config.hidden_size];
        let forward_err = flow.forward(&inputs, 1).map_err(|e| e.to_string());
        let prefill_err = flow
            .prefill(&inputs, 1)
            .map_err(|e| e.to_string())
            .map(|_| ());
        assert_eq!(forward_err.unwrap_err(), prefill_err.unwrap_err());
    }
}
