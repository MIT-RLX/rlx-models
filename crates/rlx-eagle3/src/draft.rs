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

//! EAGLE3 draft model — pure-Rust reference implementation.
//!
//! Mirrors the architecture pinned from vllm-project/speculators
//! `src/speculators/models/eagle3/{core,model_definitions}.py`.
//!
//! ## Forward pass per speculation step
//!
//! ```text
//! Init (once per propose call):
//!   h_aux_concat = cat(h_low, h_mid, h_high)            # [3*H_target]
//!   if input_norm: h_aux_concat = input_norm(h_aux_concat)
//!   h0 = fc(h_aux_concat)                               # [H_draft]
//!
//! Each step i ∈ 0..n:
//!   embed = embed_tokens[prev_target_token]             # [H_draft]
//!   # Modified first decoder layer:
//!   #   - split: not on the input, but in the attention input
//!   #   - input_layernorm acts on embed_half ONLY
//!   #   - hidden_norm acts on hidden_half ONLY (separate RMSNorm)
//!   #   - residual is hidden_half (or hidden_normed if norm_before_residual)
//!   embed_normed = rms_norm(embed, input_layernorm)
//!   hidden_normed = rms_norm(h_i, hidden_norm)
//!   residual = if norm_before_residual { hidden_normed } else { h_i }
//!   x = cat(embed_normed, hidden_normed)                # [2*H_draft]
//!   q = q_proj(x)                                       # [N_heads * head_dim] = [H_draft]
//!   k = k_proj(x)                                       # [N_kv_heads * head_dim]
//!   v = v_proj(x)
//!   q, k = rope(q, k, pos=i)
//!   append (k, v) into draft KV cache
//!   attn_out = GQA(q, kv_cache)                         # [H_draft]
//!   attn_out = o_proj(attn_out)
//!   h_attn = residual + attn_out
//!   h_mlp_in = rms_norm(h_attn, post_attention_layernorm)
//!   mlp_out = down_proj( silu(gate_proj(h_mlp_in)) * up_proj(h_mlp_in) )
//!   h_{i+1} = h_attn + mlp_out                          # [H_draft]
//!
//!   logits = lm_head( rms_norm(h_{i+1}, norm) )         # [V_draft]
//!   draft_token = argmax(logits)
//!   prev_target_token = draft_token + d2t[draft_token]
//! ```
//!
//! ## Tensor name convention
//!
//! All weights are loaded by [`crate::weights::Eagle3DraftWeights`]
//! which canonicalizes the on-disk names to the form below (stripping
//! `model.` / `midlayer.` / `layers.0.` prefixes):
//!
//! | Tensor | Shape |
//! |---|---|
//! | `fc.weight` | `[H_draft, 3*H_target]` |
//! | `embed_tokens.weight` | `[V_target, H_draft]` |
//! | `input_norm.weight` (if `norm_before_fc`) | `[3*H_target]` |
//! | `decoder.input_layernorm.weight` | `[H_draft]` |
//! | `decoder.hidden_norm.weight` | `[H_draft]` |
//! | `decoder.self_attn.q_proj.weight` | `[H_draft, 2*H_draft]` |
//! | `decoder.self_attn.k_proj.weight` | `[N_kv*head_dim, 2*H_draft]` |
//! | `decoder.self_attn.v_proj.weight` | `[N_kv*head_dim, 2*H_draft]` |
//! | `decoder.self_attn.o_proj.weight` | `[H_draft, H_draft]` |
//! | `decoder.post_attention_layernorm.weight` | `[H_draft]` |
//! | `decoder.mlp.gate_proj.weight` | `[I, H_draft]` |
//! | `decoder.mlp.up_proj.weight` | `[I, H_draft]` |
//! | `decoder.mlp.down_proj.weight` | `[H_draft, I]` |
//! | `norm.weight` | `[H_draft]` |
//! | `lm_head.weight` | `[V_draft, H_draft]` |

use anyhow::{Context, Result};

use crate::config::Eagle3Config;
use crate::reference::{
    add_in_place, argmax, gqa_attention, matvec, mul_in_place, rms_norm, rope_in_place,
    silu_in_place,
};
use crate::weights::Eagle3DraftWeights;

/// All draft model weights as borrowed `&[f32]` slices into
/// [`Eagle3DraftWeights`], plus the cached config geometry that
/// every step needs. Built once; reused across propose calls.
pub struct DraftWeightRefs<'a> {
    pub fc: &'a [f32],
    pub embed_tokens: &'a [f32],
    pub input_norm: Option<&'a [f32]>,
    pub input_layernorm: &'a [f32],
    pub hidden_norm: &'a [f32],
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub o_proj: &'a [f32],
    pub post_attention_layernorm: &'a [f32],
    pub gate_proj: &'a [f32],
    pub up_proj: &'a [f32],
    pub down_proj: &'a [f32],
    pub norm: &'a [f32],
    pub lm_head: &'a [f32],
}

impl<'a> DraftWeightRefs<'a> {
    /// Resolve all required tensors from the loaded checkpoint.
    /// Returns a clear error listing what's missing.
    pub fn from_weights(weights: &'a Eagle3DraftWeights, cfg: &Eagle3Config) -> Result<Self> {
        let get = |name: &str| -> Result<&'a [f32]> {
            weights
                .get(name)
                .map(|t| t.data.as_slice())
                .with_context(|| format!("eagle3 draft missing tensor `{name}`"))
        };

        let input_norm = if cfg.norm_before_fc {
            Some(get("input_norm.weight")?)
        } else {
            None
        };

        Ok(Self {
            fc: get("fc.weight")?,
            embed_tokens: get("embed_tokens.weight")?,
            input_norm,
            input_layernorm: get("decoder.input_layernorm.weight")?,
            hidden_norm: get("decoder.hidden_norm.weight")?,
            q_proj: get("decoder.self_attn.q_proj.weight")?,
            k_proj: get("decoder.self_attn.k_proj.weight")?,
            v_proj: get("decoder.self_attn.v_proj.weight")?,
            o_proj: get("decoder.self_attn.o_proj.weight")?,
            post_attention_layernorm: get("decoder.post_attention_layernorm.weight")?,
            gate_proj: get("decoder.mlp.gate_proj.weight")?,
            up_proj: get("decoder.mlp.up_proj.weight")?,
            down_proj: get("decoder.mlp.down_proj.weight")?,
            norm: get("norm.weight")?,
            lm_head: get("lm_head.weight")?,
        })
    }
}

/// Per-position RoPE cos/sin row generator. Standard Llama
/// `inv_freq_k = theta^(-2k / head_dim)` formula.
///
/// Returns `(cos[head_dim/2], sin[head_dim/2])` for the given `position`.
pub fn rope_row(position: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    let p = position as f64;
    for k in 0..half {
        let exp = -(2.0 * k as f64) / (head_dim as f64);
        let freq = (theta as f64).powf(exp);
        let angle = p * freq;
        cos[k] = angle.cos() as f32;
        sin[k] = angle.sin() as f32;
    }
    (cos, sin)
}

/// Stateful draft model. Holds the KV cache across speculation
/// steps inside one `propose()` call. The caller (Eagle3Speculator)
/// is responsible for resetting between propose calls.
pub struct Eagle3DraftReference<'a> {
    cfg: &'a Eagle3Config,
    weights: DraftWeightRefs<'a>,
    /// `H_target`, `H_draft`, draft `intermediate`, head geometry.
    pub geom: DraftGeom,
    /// Per-layer KV cache. Single layer ⇒ flat `Vec<f32>` of
    /// `[step, n_kv_heads, head_dim]` row-major. Grows by one row
    /// per step.
    past_k: Vec<f32>,
    past_v: Vec<f32>,
    /// Current sequence length in the KV cache.
    cache_seq: usize,
}

/// Cached geometry pulled out of [`Eagle3Config::transformer_layer_config`].
#[derive(Debug, Clone, Copy)]
pub struct DraftGeom {
    pub h_draft: usize,
    pub h_target: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub draft_vocab: usize,
    pub target_vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub norm_before_residual: bool,
    pub norm_before_fc: bool,
}

impl DraftGeom {
    pub fn from_cfg(cfg: &Eagle3Config) -> Self {
        let tl = &cfg.transformer_layer_config;
        Self {
            h_draft: tl.hidden_size,
            h_target: cfg.target_hidden_size(),
            intermediate: tl.intermediate_size,
            n_heads: tl.num_attention_heads,
            n_kv_heads: tl.num_key_value_heads,
            head_dim: tl.head_dim,
            draft_vocab: cfg.draft_vocab_size(),
            target_vocab: tl.vocab_size,
            rope_theta: tl
                .rope_parameters
                .as_ref()
                .map(|r| r.rope_theta)
                .unwrap_or(10_000.0),
            rms_eps: tl.rms_norm_eps,
            norm_before_residual: cfg.norm_before_residual,
            norm_before_fc: cfg.norm_before_fc,
        }
    }
}

impl<'a> Eagle3DraftReference<'a> {
    pub fn new(cfg: &'a Eagle3Config, weights: DraftWeightRefs<'a>) -> Self {
        let geom = DraftGeom::from_cfg(cfg);
        Self {
            cfg,
            weights,
            geom,
            past_k: Vec::new(),
            past_v: Vec::new(),
            cache_seq: 0,
        }
    }

    pub fn cfg(&self) -> &Eagle3Config {
        self.cfg
    }
    pub fn geom(&self) -> DraftGeom {
        self.geom
    }
    /// KV cache state — exposed for numerical-parity tests against
    /// the HIR draft graph, which takes past_k/past_v as inputs.
    pub fn past_k(&self) -> &[f32] {
        &self.past_k
    }
    pub fn past_v(&self) -> &[f32] {
        &self.past_v
    }
    pub fn cache_seq(&self) -> usize {
        self.cache_seq
    }

    /// Reset the per-propose KV cache. Call before each new
    /// `propose()` round.
    pub fn reset(&mut self) {
        self.past_k.clear();
        self.past_v.clear();
        self.cache_seq = 0;
    }

    /// Compute the initial hidden state from `n_aux` per-layer aux
    /// hidden states (each of length `h_target`). Applies optional
    /// `input_norm` then `fc`.
    pub fn init_hidden(&self, aux: &[Vec<f32>]) -> Vec<f32> {
        let g = self.geom;
        // Concatenate aux hidden states along the feature axis.
        let total = aux.len() * g.h_target;
        let mut h_aux = Vec::with_capacity(total);
        for layer in aux {
            assert_eq!(
                layer.len(),
                g.h_target,
                "init_hidden: aux layer len {} != h_target {}",
                layer.len(),
                g.h_target
            );
            h_aux.extend_from_slice(layer);
        }
        // Optional pre-fc RMSNorm.
        let h_aux = if let Some(gamma) = self.weights.input_norm {
            assert!(g.norm_before_fc);
            rms_norm(&h_aux, gamma, g.rms_eps)
        } else {
            h_aux
        };
        // fc: [H_draft, total] @ h_aux[total] → [H_draft]
        matvec(self.weights.fc, g.h_draft, total, &h_aux)
    }

    /// Run one speculation step. Updates the internal KV cache.
    ///
    /// - `prev_hidden`: hidden state from the previous step (or
    ///   [`init_hidden`] on step 0).
    /// - `prev_target_token`: target-vocab token id used to look up
    ///   the embedding (on step 0, this is the last token of the
    ///   verifier's context).
    ///
    /// Returns `(draft_logits[V_draft], new_hidden[H_draft])`.
    pub fn step(
        &mut self,
        prev_hidden: &[f32],
        prev_target_token: u32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let g = self.geom;
        assert_eq!(prev_hidden.len(), g.h_draft, "prev_hidden shape");
        let kv_dim = g.n_kv_heads * g.head_dim;
        let q_dim = g.n_heads * g.head_dim;
        // Note: q_dim (n_heads * head_dim) is NOT required to equal
        // h_draft. RedHatAI/gemma-4-31B-it-speculator.eagle3 has
        // q_dim = 32*256 = 8192 vs h_draft = 5376 — `o_proj` then
        // projects 8192 → 5376 back into the residual stream.
        anyhow::ensure!(
            (prev_target_token as usize) < g.target_vocab,
            "prev_target_token {} out of target_vocab {}",
            prev_target_token,
            g.target_vocab,
        );

        // ── Embed ────────────────────────────────────────────────────
        // embed_tokens: [V_target, H_draft], row prev_target_token.
        let row_off = (prev_target_token as usize) * g.h_draft;
        let embed = &self.weights.embed_tokens[row_off..row_off + g.h_draft];

        // ── Modified first-layer norms (split-and-norm) ──────────────
        let embed_normed = rms_norm(embed, self.weights.input_layernorm, g.rms_eps);
        let hidden_normed = rms_norm(prev_hidden, self.weights.hidden_norm, g.rms_eps);
        let residual: Vec<f32> = if g.norm_before_residual {
            hidden_normed.clone()
        } else {
            prev_hidden.to_vec()
        };

        // ── Concat then q/k/v projection ─────────────────────────────
        let mut x = Vec::with_capacity(2 * g.h_draft);
        x.extend_from_slice(&embed_normed);
        x.extend_from_slice(&hidden_normed);
        let mut q = matvec(self.weights.q_proj, q_dim, 2 * g.h_draft, &x);
        let mut k = matvec(self.weights.k_proj, kv_dim, 2 * g.h_draft, &x);
        let v = matvec(self.weights.v_proj, kv_dim, 2 * g.h_draft, &x);

        // ── RoPE on q and k at current position ──────────────────────
        let (cos, sin) = rope_row(self.cache_seq, g.head_dim, g.rope_theta);
        rope_in_place(&mut q, g.n_heads, g.head_dim, &cos, &sin);
        rope_in_place(&mut k, g.n_kv_heads, g.head_dim, &cos, &sin);

        // ── Append k,v into per-layer KV cache ───────────────────────
        self.past_k.extend_from_slice(&k);
        self.past_v.extend_from_slice(&v);
        self.cache_seq += 1;

        // ── GQA attention over all past entries (including current) ──
        let attn = gqa_attention(
            &q,
            &self.past_k,
            &self.past_v,
            g.n_heads,
            g.n_kv_heads,
            g.head_dim,
            self.cache_seq,
        );
        let attn_out = matvec(self.weights.o_proj, g.h_draft, q_dim, &attn);

        // ── Residual + MLP ───────────────────────────────────────────
        let mut h_attn = residual;
        add_in_place(&mut h_attn, &attn_out);

        let mlp_in = rms_norm(&h_attn, self.weights.post_attention_layernorm, g.rms_eps);
        let mut gate = matvec(self.weights.gate_proj, g.intermediate, g.h_draft, &mlp_in);
        let up = matvec(self.weights.up_proj, g.intermediate, g.h_draft, &mlp_in);
        silu_in_place(&mut gate);
        mul_in_place(&mut gate, &up);
        let mlp_out = matvec(self.weights.down_proj, g.h_draft, g.intermediate, &gate);

        let mut new_hidden = h_attn;
        add_in_place(&mut new_hidden, &mlp_out);

        // ── Final norm + LM head over draft vocab ────────────────────
        let final_normed = rms_norm(&new_hidden, self.weights.norm, g.rms_eps);
        let logits = matvec(
            self.weights.lm_head,
            g.draft_vocab,
            g.h_draft,
            &final_normed,
        );

        Ok((logits, new_hidden))
    }

    /// Convenience: take an argmax of the draft logits and return
    /// the target token id (via the d2t offset map).
    pub fn argmax_to_target_id(&self, logits: &[f32], d2t: &crate::d2t::D2tMap) -> u32 {
        let draft_id = argmax(logits);
        d2t.map_token(draft_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesizes a config for a tiny but architecturally complete
    /// draft model — small enough to run in a test, large enough
    /// that every code path executes.
    fn tiny_cfg() -> Eagle3Config {
        let json = r#"{
            "draft_vocab_size": 8,
            "norm_before_residual": false,
            "eagle_aux_hidden_state_layer_ids": [0, 1, 2],
            "transformer_layer_config": {
                "model_type": "llama",
                "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 2,
                "vocab_size": 16,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }"#;
        Eagle3Config::from_bytes(json.as_bytes()).unwrap()
    }

    /// Deterministic ramp; salt distinguishes weight tensors.
    fn ramp(n: usize, salt: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(0x9E3779B1).wrapping_add(salt)) >> 16;
                ((x as f32) / (1u32 << 16) as f32 - 0.5) * 0.05
            })
            .collect()
    }

    /// Build a synthetic Eagle3DraftWeights using bytes-roundtrip so
    /// canonicalization runs too.
    fn synth_weights(cfg: &Eagle3Config) -> Eagle3DraftWeights {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype as StDtype, TensorView};
        use std::collections::HashMap;

        let g = DraftGeom::from_cfg(cfg);
        let kv_dim = g.n_kv_heads * g.head_dim;

        let mut buffers: Vec<(String, Vec<f32>, Vec<usize>)> = vec![
            (
                "fc.weight".into(),
                ramp(g.h_draft * 3 * g.h_target, 1),
                vec![g.h_draft, 3 * g.h_target],
            ),
            (
                "embed_tokens.weight".into(),
                ramp(g.target_vocab * g.h_draft, 2),
                vec![g.target_vocab, g.h_draft],
            ),
            (
                "midlayer.input_layernorm.weight".into(),
                vec![1.0; g.h_draft],
                vec![g.h_draft],
            ),
            (
                "midlayer.hidden_norm.weight".into(),
                vec![1.0; g.h_draft],
                vec![g.h_draft],
            ),
            (
                "midlayer.self_attn.q_proj.weight".into(),
                ramp(g.h_draft * 2 * g.h_draft, 3),
                vec![g.h_draft, 2 * g.h_draft],
            ),
            (
                "midlayer.self_attn.k_proj.weight".into(),
                ramp(kv_dim * 2 * g.h_draft, 4),
                vec![kv_dim, 2 * g.h_draft],
            ),
            (
                "midlayer.self_attn.v_proj.weight".into(),
                ramp(kv_dim * 2 * g.h_draft, 5),
                vec![kv_dim, 2 * g.h_draft],
            ),
            (
                "midlayer.self_attn.o_proj.weight".into(),
                ramp(g.h_draft * g.h_draft, 6),
                vec![g.h_draft, g.h_draft],
            ),
            (
                "midlayer.post_attention_layernorm.weight".into(),
                vec![1.0; g.h_draft],
                vec![g.h_draft],
            ),
            (
                "midlayer.mlp.gate_proj.weight".into(),
                ramp(g.intermediate * g.h_draft, 7),
                vec![g.intermediate, g.h_draft],
            ),
            (
                "midlayer.mlp.up_proj.weight".into(),
                ramp(g.intermediate * g.h_draft, 8),
                vec![g.intermediate, g.h_draft],
            ),
            (
                "midlayer.mlp.down_proj.weight".into(),
                ramp(g.h_draft * g.intermediate, 9),
                vec![g.h_draft, g.intermediate],
            ),
            ("norm.weight".into(), vec![1.0; g.h_draft], vec![g.h_draft]),
            (
                "lm_head.weight".into(),
                ramp(g.draft_vocab * g.h_draft, 10),
                vec![g.draft_vocab, g.h_draft],
            ),
        ];

        // Build the on-disk safetensors blob.
        let mut tensor_bytes: Vec<Vec<u8>> = Vec::new();
        for (_, data, _) in &buffers {
            tensor_bytes.push(bytemuck::cast_slice::<f32, u8>(data.as_slice()).to_vec());
        }
        let mut views: HashMap<&str, TensorView<'_>> = HashMap::new();
        for ((name, _, shape), bytes) in buffers.iter().zip(&tensor_bytes) {
            views.insert(
                name.as_str(),
                TensorView::new(StDtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
            );
        }
        // d2t: identity-ish offsets so target_id == draft_id (offset 0).
        let d2t_data: Vec<u32> = vec![0; g.draft_vocab];
        let d2t_bytes: Vec<u8> = bytemuck::cast_slice(&d2t_data).to_vec();
        views.insert(
            "d2t",
            TensorView::new(StDtype::U32, vec![g.draft_vocab], d2t_bytes.as_slice()).unwrap(),
        );

        let blob = serialize(&views, None).unwrap();
        // Pin lifetime of tensor_bytes until after blob is built.
        let _ = buffers.drain(..);
        Eagle3DraftWeights::from_bytes(&blob).unwrap()
    }

    #[test]
    fn rope_row_position_zero_is_identity() {
        let (cos, sin) = rope_row(0, 8, 10_000.0);
        for c in cos {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for s in sin {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn draft_geom_pulls_cfg_fields() {
        let cfg = tiny_cfg();
        let g = DraftGeom::from_cfg(&cfg);
        assert_eq!(g.h_draft, 8);
        assert_eq!(g.h_target, 8); // target_hidden_size == draft when null
        assert_eq!(g.intermediate, 16);
        assert_eq!(g.n_heads, 4);
        assert_eq!(g.n_kv_heads, 2);
        assert_eq!(g.head_dim, 2);
        assert_eq!(g.draft_vocab, 8);
        assert_eq!(g.target_vocab, 16);
        assert!(!g.norm_before_residual);
    }

    #[test]
    fn weight_refs_reports_missing_tensors() {
        // Empty weights → from_weights must fail with a clear msg.
        use safetensors::serialize;
        use safetensors::tensor::{Dtype as StDtype, TensorView};
        use std::collections::HashMap;
        let d2t: Vec<u32> = vec![0; 8];
        let d2t_bytes: Vec<u8> = bytemuck::cast_slice(&d2t).to_vec();
        let d2t_view = TensorView::new(StDtype::U32, vec![8], &d2t_bytes).unwrap();
        let mut map: HashMap<&str, TensorView<'_>> = HashMap::new();
        map.insert("d2t", d2t_view);
        let blob = serialize(&map, None).unwrap();
        let w = Eagle3DraftWeights::from_bytes(&blob).unwrap();

        let cfg = tiny_cfg();
        let res = DraftWeightRefs::from_weights(&w, &cfg);
        let err = res.err().expect("expected missing-tensor error");
        let msg = format!("{err:?}");
        assert!(msg.contains("fc.weight"), "got: {msg}");
    }

    #[test]
    fn end_to_end_step_produces_finite_logits_and_grows_kv() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let refs = DraftWeightRefs::from_weights(&weights, &cfg).unwrap();
        let mut draft = Eagle3DraftReference::new(&cfg, refs);

        // 3 aux layers of hidden_size = h_target = 8.
        let aux: Vec<Vec<f32>> = vec![
            (0..8).map(|i| (i as f32) * 0.01).collect(),
            (0..8).map(|i| (i as f32) * -0.01).collect(),
            (0..8).map(|i| (i as f32) * 0.005).collect(),
        ];
        let h0 = draft.init_hidden(&aux);
        assert_eq!(h0.len(), 8);
        for v in &h0 {
            assert!(v.is_finite(), "init_hidden produced non-finite {v}");
        }

        // 3 speculation steps; cache should grow by 1 each step;
        // logits should have draft_vocab_size and be finite.
        let mut hidden = h0;
        let mut prev_token: u32 = 0;
        for step in 0..3 {
            let (logits, new_hidden) = draft.step(&hidden, prev_token).unwrap();
            assert_eq!(logits.len(), 8);
            assert_eq!(new_hidden.len(), 8);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "step {step} produced non-finite logits"
            );
            assert_eq!(draft.cache_seq, step + 1);
            assert_eq!(draft.past_k.len(), (step + 1) * 4); // n_kv_heads * head_dim = 4
            assert_eq!(draft.past_v.len(), (step + 1) * 4);
            hidden = new_hidden;
            // Bounce through d2t (identity here ⇒ target == draft).
            prev_token = argmax(&logits);
            assert!(prev_token < 8, "argmax in draft vocab");
        }
        draft.reset();
        assert_eq!(draft.cache_seq, 0);
        assert!(draft.past_k.is_empty() && draft.past_v.is_empty());
    }

    #[test]
    fn reset_clears_kv_cache_between_propose_rounds() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let refs = DraftWeightRefs::from_weights(&weights, &cfg).unwrap();
        let mut draft = Eagle3DraftReference::new(&cfg, refs);
        let aux: Vec<Vec<f32>> = vec![vec![0.0; 8]; 3];
        let h0 = draft.init_hidden(&aux);
        let _ = draft.step(&h0, 0).unwrap();
        let _ = draft.step(&h0, 0).unwrap();
        assert_eq!(draft.cache_seq, 2);
        draft.reset();
        assert_eq!(draft.cache_seq, 0);
        // After reset, the next step starts at position 0 again.
        let _ = draft.step(&h0, 0).unwrap();
        assert_eq!(draft.cache_seq, 1);
    }
}
