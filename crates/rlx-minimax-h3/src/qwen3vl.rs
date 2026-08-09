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

//! The Qwen3-VL conditioner, run only as far as MiniMax-H3 reads it.
//!
//! H3 conditions on `hidden_states[50]` — the state after **50 decoder layers**,
//! before the final norm. That has a useful consequence: the last 14 layers, the
//! output norm and the 778M-parameter `lm_head` are never evaluated, so this
//! builds a graph over layers `0..50` only and never loads the rest.
//! [`layers_to_run`] is the single place that mapping lives.
//!
//! ## mRoPE degenerates for text
//!
//! Qwen3-VL positions tokens on three axes with `mrope_section = [24, 20, 20]`.
//! A **text-only** prompt gives every token the same coordinate on all three
//! axes, so every section sees the same angle and the whole thing collapses to
//! ordinary NeoX RoPE — regardless of how the sections are interleaved. That is
//! what [`text_rope_tables`] builds, and it is exact for text.
//!
//! Prompts carrying images are a different matter: the vision tower has to run
//! and the three axes genuinely diverge. This module handles the text path and
//! [`H3Qwen3VlEncoder::encode_tokens`] rejects anything else rather than
//! silently conditioning on wrong angles.
//!
//! ## Status
//!
//! Structurally tested on CPU with synthetic weights — shapes, finiteness,
//! causality, GQA head mapping and the tap position. It has **not** been checked
//! against the reference: the released encoder is ~60 GB across 14 shards and
//! was not fetched for this port.

use crate::config::H3TextEncoderConfig;
use crate::text_encoder::H3TextConditioning;
use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, plugin_named};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

/// Prefix every language-model tensor carries in the released checkpoint.
pub const LM_PREFIX: &str = "model.language_model";

/// How many decoder layers to run for the conditioning tap.
///
/// HuggingFace's `hidden_states` tuple puts the embedding output at index 0, so
/// `hidden_states[50]` is the output of layer index 49 — i.e. **50 layers have
/// run**. Off-by-one here is a silent quality regression, not a crash.
///
/// This deliberately does **not** clamp to the stack depth: a checkpoint too
/// shallow to reach the tap is a configuration error, and quietly conditioning
/// on whatever layer happened to be last would be far worse than failing.
/// [`H3TextEncoderConfig::validate`] is what enforces the depth.
#[must_use]
pub fn layers_to_run(_cfg: &H3TextEncoderConfig) -> usize {
    H3TextEncoderConfig::TAP_LAYER
}

/// NeoX RoPE tables for a text-only prompt.
///
/// Returns `(cos, sin)`, each `[seq_len * head_dim / 2]`.
#[must_use]
pub fn text_rope_tables(seq_len: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq_len * half];
    let mut sin = vec![0.0f32; seq_len * half];
    for (i, (c, s)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
        let (pos, k) = (i / half, i % half);
        let inv = 1.0 / (theta as f64).powf(2.0 * k as f64 / head_dim as f64);
        let angle = pos as f64 * inv;
        *c = angle.cos() as f32;
        *s = angle.sin() as f32;
    }
    (cos, sin)
}

/// Parameter keys the tapped stack reads — everything beyond layer
/// [`layers_to_run`] is deliberately absent.
#[must_use]
pub fn parameter_keys(cfg: &H3TextEncoderConfig) -> Vec<String> {
    let mut keys = vec![format!("{LM_PREFIX}.embed_tokens.weight")];
    for l in 0..layers_to_run(cfg) {
        let p = format!("{LM_PREFIX}.layers.{l}");
        for s in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            keys.push(format!("{p}.{s}"));
        }
    }
    keys
}

/// The tapped Qwen3-VL text stack, compiled for one prompt length.
pub struct H3Qwen3VlEncoder {
    compiled: CompiledGraph,
    cfg: H3TextEncoderConfig,
    seq_len: usize,
    device: Device,
}

impl H3Qwen3VlEncoder {
    #[must_use]
    pub fn config(&self) -> &H3TextEncoderConfig {
        &self.cfg
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Run the tap over a text-only token sequence.
    pub fn encode_tokens(&mut self, token_ids: &[u32]) -> Result<H3TextConditioning> {
        ensure!(
            token_ids.len() == self.seq_len,
            "this encoder is compiled for {} tokens, got {}",
            self.seq_len,
            token_ids.len()
        );
        for (i, &t) in token_ids.iter().enumerate() {
            ensure!(
                (t as usize) < self.cfg.vocab_size,
                "token {i} is id {t}, outside the {}-entry vocabulary",
                self.cfg.vocab_size
            );
        }
        let ids: Vec<f32> = token_ids.iter().map(|&t| t as f32).collect();
        let (cos, sin) = text_rope_tables(self.seq_len, self.cfg.head_dim, self.cfg.rope_theta);
        let outs = self
            .compiled
            .run(&[("input_ids", &ids), ("cos", &cos), ("sin", &sin)]);
        let hidden = outs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("the text encoder returned no output"))?;
        H3TextConditioning::text_only(hidden, self.cfg.hidden_size)
    }
}

/// Compile the tapped stack for a fixed prompt length.
pub fn compile_text_encoder(
    cfg: &H3TextEncoderConfig,
    weights: &mut WeightMap,
    device: Device,
    seq_len: usize,
) -> Result<H3Qwen3VlEncoder> {
    cfg.validate()?;
    ensure!(seq_len > 0, "the prompt must hold at least one token");

    let built = build_flow(cfg, weights, seq_len).context("MiniMax-H3: build text encoder flow")?;
    let typed = built.typed_params.clone();
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    let opts =
        rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);

    Ok(H3Qwen3VlEncoder {
        compiled,
        cfg: cfg.clone(),
        seq_len,
        device,
    })
}

fn build_flow(
    cfg: &H3TextEncoderConfig,
    weights: &mut WeightMap,
    seq_len: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let hd = cfg.head_dim;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    ensure!(
        nkv > 0 && nh.is_multiple_of(nkv),
        "{nh} query heads do not group into {nkv} key/value heads"
    );
    let group = nh / nkv;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let ffn = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps;
    let layers = layers_to_run(cfg);

    let flow = ModelFlow::new("minimax_h3_text_encoder")
        .with_profile(CompileProfile::encoder())
        .input("input_ids", Shape::new(&[1, seq_len], DType::I32))
        .input("cos", Shape::new(&[seq_len, hd / 2], f))
        .input("sin", Shape::new(&[seq_len, hd / 2], f))
        .stage(plugin_named("embed", move |emit, _h| {
            let ids = emit.flow_input("input_ids")?;
            let cos = emit.flow_input("cos")?;
            let sin = emit.flow_input("sin")?;
            emit.set_named("te_cos", cos.hir_id());
            emit.set_named("te_sin", sin.hir_id());
            let z_hidden = emit.synth_zeros("te_zeros_hidden", hidden);
            let z_head = emit.synth_zeros("te_zeros_head", hd);
            emit.set_named("te_zeros_hidden", z_hidden);
            emit.set_named("te_zeros_head", z_head);

            let embed = emit.load_param(&format!("{LM_PREFIX}.embed_tokens.weight"), false)?;
            let mut gb = HirMut::new(emit.hir());
            let x = gb.gather_(embed, ids.hir_id(), 0);
            let x = gb.reshape_(x, vec![1, seq_len as i64, hidden as i64]);
            Ok(Some(emit.wrap(x, Shape::new(&[1, seq_len, hidden], f))))
        }))
        .repeat_layers(layers, move |l| {
            decoder_layer(
                l, seq_len, hidden, nh, nkv, group, hd, q_dim, kv_dim, ffn, eps,
            )
        })
        // No final norm: H3 reads the *unnormalized* state.
        .output("conditioning");

    flow.build_with(&mut WeightMapSource(weights), None)
}

/// GQA widening: emit each key/value head `group` times.
fn repeat_kv(gb: &mut HirMut<'_>, x: HirNodeId, nkv: usize, hd: usize, group: usize) -> HirNodeId {
    if group == 1 {
        return x;
    }
    let mut pieces = Vec::with_capacity(nkv * group);
    for h in 0..nkv {
        let slice = gb.narrow_(x, 2, h * hd, hd);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    gb.concat_(pieces, 2)
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer(
    l: usize,
    seq: usize,
    hidden: usize,
    nh: usize,
    nkv: usize,
    group: usize,
    hd: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn: usize,
    eps: f32,
) -> FlowStage {
    let p = format!("{LM_PREFIX}.layers.{l}");
    plugin_named(format!("te_layer{l}"), move |emit, h| {
        let x = h.ok_or_else(|| anyhow!("text encoder layer {l} needs a hidden state"))?;
        let f = DType::F32;
        let shape = Shape::new(&[1, seq, hidden], f);

        let n1 = emit.load_param(&format!("{p}.input_layernorm.weight"), false)?;
        let qw = emit.load_param(&format!("{p}.self_attn.q_proj.weight"), true)?;
        let kw = emit.load_param(&format!("{p}.self_attn.k_proj.weight"), true)?;
        let vw = emit.load_param(&format!("{p}.self_attn.v_proj.weight"), true)?;
        let ow = emit.load_param(&format!("{p}.self_attn.o_proj.weight"), true)?;
        let qn = emit.load_param(&format!("{p}.self_attn.q_norm.weight"), false)?;
        let kn = emit.load_param(&format!("{p}.self_attn.k_norm.weight"), false)?;
        let n2 = emit.load_param(&format!("{p}.post_attention_layernorm.weight"), false)?;
        let gw = emit.load_param(&format!("{p}.mlp.gate_proj.weight"), true)?;
        let uw = emit.load_param(&format!("{p}.mlp.up_proj.weight"), true)?;
        let dw = emit.load_param(&format!("{p}.mlp.down_proj.weight"), true)?;

        let z_hidden = emit.named("te_zeros_hidden")?;
        let z_head = emit.named("te_zeros_head")?;
        let cos = emit.named("te_cos")?;
        let sin = emit.named("te_sin")?;

        let mut gb = HirMut::new(emit.hir());
        let residual = x.hir_id();
        let hx = gb.rms_norm(residual, n1, z_hidden, eps);

        let q = gb.mm(hx, qw);
        let k = gb.mm(hx, kw);
        let v = gb.mm(hx, vw);

        // Qwen3 normalizes every head of Q and K before the rotation.
        let mut per_head = |t: HirNodeId, gamma: HirNodeId, heads: usize, width: usize| {
            let flat = gb.reshape_(t, vec![1, (seq * heads) as i64, hd as i64]);
            let nrm = gb.rms_norm(flat, gamma, z_head, eps);
            let back = gb.reshape_(nrm, vec![1, seq as i64, width as i64]);
            gb.rope(back, cos, sin, hd)
        };
        let q = per_head(q, qn, nh, q_dim);
        let k = per_head(k, kn, nkv, kv_dim);

        let k = repeat_kv(&mut gb, k, nkv, hd, group);
        let v = repeat_kv(&mut gb, v, nkv, hd, group);

        let attn = gb.attention_kind(
            q,
            k,
            v,
            nh,
            hd,
            MaskKind::Causal,
            Shape::new(&[1, seq, q_dim], f),
        );
        let attn = gb.mm(attn, ow);
        let x1 = gb.add(residual, attn);

        let hx = gb.rms_norm(x1, n2, z_hidden, eps);
        let gate = gb.mm(hx, gw);
        let up = gb.mm(hx, uw);
        let act = gb.silu(gate);
        let prod = gb.mul(act, up);
        let down = gb.mm(prod, dw);
        let out = gb.add(x1, down);

        let _ = ffn;
        Ok(Some(emit.wrap(out, shape)))
    })
}

/// Deterministic synthetic weights for the tapped stack, for CPU tests.
#[must_use]
pub fn synthetic_weights(cfg: &H3TextEncoderConfig, seed: u64) -> WeightMap {
    use std::collections::HashMap;
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (i, key) in parameter_keys(cfg).into_iter().enumerate() {
        let shape = parameter_shape(cfg, &key).unwrap_or_else(|| vec![cfg.hidden_size]);
        let n: usize = shape.iter().product();
        let fan_in = *shape.last().unwrap_or(&1);
        let is_norm = key.ends_with("layernorm.weight")
            || key.ends_with("q_norm.weight")
            || key.ends_with("k_norm.weight");
        let scale = if is_norm {
            0.0
        } else {
            1.0 / (fan_in as f32).sqrt()
        };
        let data = (0..n)
            .map(|j| {
                let base = if is_norm { 1.0 } else { 0.0 };
                base + scale * hash_unit(seed, i as u64, j as u64)
            })
            .collect();
        tensors.insert(key, (data, shape));
    }
    WeightMap::from_tensors(tensors)
}

/// The shape every tapped parameter must have.
#[must_use]
pub fn parameter_shape(cfg: &H3TextEncoderConfig, key: &str) -> Option<Vec<usize>> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    Some(match key {
        k if k.ends_with("embed_tokens.weight") => vec![cfg.vocab_size, hidden],
        k if k.ends_with("input_layernorm.weight")
            || k.ends_with("post_attention_layernorm.weight") =>
        {
            vec![hidden]
        }
        k if k.ends_with("q_proj.weight") => vec![q_dim, hidden],
        k if k.ends_with("k_proj.weight") || k.ends_with("v_proj.weight") => vec![kv_dim, hidden],
        k if k.ends_with("o_proj.weight") => vec![hidden, q_dim],
        k if k.ends_with("q_norm.weight") || k.ends_with("k_norm.weight") => vec![cfg.head_dim],
        k if k.ends_with("gate_proj.weight") || k.ends_with("up_proj.weight") => {
            vec![cfg.intermediate_size, hidden]
        }
        k if k.ends_with("down_proj.weight") => vec![hidden, cfg.intermediate_size],
        _ => return None,
    })
}

fn hash_unit(seed: u64, a: u64, b: u64) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(a.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(b.wrapping_mul(0x94D0_49BB_1331_11EB));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 52) as f32 / 2048.0) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released() -> H3TextEncoderConfig {
        H3TextEncoderConfig {
            hidden_size: 5120,
            num_hidden_layers: 64,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 25_600,
            rms_norm_eps: 1e-6,
            rope_theta: 5e6,
            vocab_size: 151_936,
            mrope_section: [24, 20, 20],
            mrope_interleaved: true,
        }
    }

    #[test]
    fn the_tap_runs_fifty_layers_not_sixty_four() {
        let c = released();
        assert_eq!(layers_to_run(&c), 50);
        // A stack too shallow to reach the tap is rejected outright rather
        // than silently conditioning on its last layer.
        let mut shallow = released();
        shallow.num_hidden_layers = 6;
        assert!(shallow.validate().is_err());
        let keys = parameter_keys(&c);
        assert!(keys.iter().any(|k| k.contains(".layers.49.")));
        assert!(
            !keys.iter().any(|k| k.contains(".layers.50.")),
            "layer 50 and beyond must not be loaded"
        );
        // The final norm and lm_head are never read.
        assert!(
            !keys
                .iter()
                .any(|k| k.ends_with("model.language_model.norm.weight"))
        );
        assert!(!keys.iter().any(|k| k.contains("lm_head")));
        // 1 embedding + 50 layers x 11 tensors.
        assert_eq!(keys.len(), 1 + 50 * 11);
    }

    #[test]
    fn every_tapped_key_has_a_declared_shape() {
        let c = released();
        for k in parameter_keys(&c) {
            assert!(parameter_shape(&c, &k).is_some(), "no shape for {k}");
        }
    }

    #[test]
    fn gqa_grouping_matches_the_released_config() {
        let c = released();
        assert_eq!(c.num_attention_heads % c.num_key_value_heads, 0);
        assert_eq!(c.num_attention_heads / c.num_key_value_heads, 8);
        assert_eq!(
            parameter_shape(&c, "x.self_attn.q_proj.weight").unwrap(),
            vec![8192, 5120]
        );
        assert_eq!(
            parameter_shape(&c, "x.self_attn.k_proj.weight").unwrap(),
            vec![1024, 5120]
        );
    }

    #[test]
    fn rope_tables_are_bounded_and_start_at_identity() {
        let (cos, sin) = text_rope_tables(4, 128, 5e6);
        assert_eq!(cos.len(), 4 * 64);
        // Position 0 has angle 0 on every frequency.
        assert!(cos[..64].iter().all(|&c| (c - 1.0).abs() < 1e-6));
        assert!(sin[..64].iter().all(|&s| s.abs() < 1e-6));
        assert!(
            cos.iter()
                .chain(&sin)
                .all(|v| v.is_finite() && v.abs() <= 1.0)
        );
    }

    #[test]
    fn rope_frequencies_decay_with_channel_index() {
        let (cos, _) = text_rope_tables(2, 128, 5e6);
        // At position 1 the angle falls with k, so cos rises toward 1.
        let row = &cos[64..128];
        assert!(
            row.windows(2).all(|w| w[1] >= w[0] - 1e-6),
            "cos should rise as the frequency falls"
        );
    }

    #[test]
    fn mrope_sections_cover_half_the_head() {
        // Text-only prompts collapse mRoPE to plain RoPE, but the section split
        // still has to describe the head or the config is inconsistent.
        let c = released();
        assert_eq!(c.mrope_section.iter().sum::<usize>(), c.head_dim / 2);
        c.validate().unwrap();
    }

    #[test]
    fn synthetic_weights_are_deterministic_and_bounded() {
        let mut c = released();
        c.num_hidden_layers = 2;
        c.vocab_size = 32;
        c.hidden_size = 16;
        c.num_attention_heads = 4;
        c.num_key_value_heads = 2;
        c.head_dim = 8;
        c.intermediate_size = 32;
        let a = synthetic_weights(&c, 5);
        let b = synthetic_weights(&c, 5);
        let k = format!("{LM_PREFIX}.layers.0.self_attn.q_proj.weight");
        assert_eq!(a.get(&k).unwrap().0, b.get(&k).unwrap().0);
        assert!(
            a.get(&k)
                .unwrap()
                .0
                .iter()
                .all(|v| v.is_finite() && v.abs() < 1.0)
        );
    }
}
