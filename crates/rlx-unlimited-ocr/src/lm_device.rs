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

//! Compiled MoE LM session — `compile_built` on the runner device with
//! host-managed ring KV between decode steps.

use crate::compile_support::lm_runtime_guard_for_pack;
use crate::config::UnlimitedOcrConfig;
use crate::expert_pack::PackedLmWeights;
use crate::lm_graph::{
    build_unlimited_ocr_decode_built_from_pack, build_unlimited_ocr_prefill_built_from_pack,
    compute_rope_slice,
};
use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::compile_built;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::sync::Arc;

/// Host ring KV matching [`crate::lm_flow::LmFlow`] / HF SlidingWindowLlamaAttention.
struct LayerKv {
    /// `[cur_len, hidden]` row-major (`hidden = num_kv_heads * head_dim`).
    k: Vec<f32>,
    v: Vec<f32>,
    prefill_len: usize,
    ring_pos: Option<usize>,
}

pub struct DeviceKvCache {
    layers: Vec<LayerKv>,
}

/// Compiled Unlimited-OCR LM on a concrete RLX [`Device`].
pub struct CompiledLm {
    device: Device,
    config: UnlimitedOcrConfig,
    weights: Arc<PackedLmWeights>,
    prefill: HashMap<usize, CompiledGraph>,
    decode: HashMap<usize, CompiledGraph>,
}

impl CompiledLm {
    pub fn new(device: Device, weights: Arc<PackedLmWeights>) -> Self {
        let config = weights.config.clone();
        Self {
            device,
            config,
            weights,
            prefill: HashMap::new(),
            decode: HashMap::new(),
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn config(&self) -> &UnlimitedOcrConfig {
        &self.config
    }

    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.weights.embed_tokens_lookup(ids)
    }

    fn ensure_prefill(&mut self, seq: usize) -> Result<&CompiledGraph> {
        if !self.prefill.contains_key(&seq) {
            let built =
                build_unlimited_ocr_prefill_built_from_pack(&self.config, &self.weights, 1, seq)?;
            let device = self.device;
            let compiled =
                lm_runtime_guard_for_pack(device, &self.weights, || compile_built(built, device))?;
            self.prefill.insert(seq, compiled);
        }
        Ok(self.prefill.get(&seq).expect("prefill just inserted"))
    }

    fn ensure_decode(&mut self, past_seq: usize) -> Result<&CompiledGraph> {
        if !self.decode.contains_key(&past_seq) {
            let built = build_unlimited_ocr_decode_built_from_pack(
                &self.config,
                &self.weights,
                1,
                past_seq,
            )?;
            let device = self.device;
            let compiled =
                lm_runtime_guard_for_pack(device, &self.weights, || compile_built(built, device))?;
            self.decode.insert(past_seq, compiled);
        }
        Ok(self.decode.get(&past_seq).expect("decode just inserted"))
    }

    /// Prefill over `[n_tokens, hidden]` embeds → last-token logits + KV cache.
    pub fn prefill(
        &mut self,
        inputs_embeds: &[f32],
        n_tokens: usize,
    ) -> Result<(Vec<f32>, DeviceKvCache)> {
        let hidden = self.config.hidden_size;
        ensure!(
            inputs_embeds.len() == n_tokens * hidden,
            "prefill: embeds len {} != {n_tokens}*{hidden}",
            inputs_embeds.len()
        );
        let _ = self.ensure_prefill(n_tokens)?;
        let device = self.device;
        let outs = lm_runtime_guard_for_pack(device, &self.weights, || {
            let compiled = self.prefill.get_mut(&n_tokens).expect("prefill");
            compiled.run(&[("inputs_embeds", inputs_embeds)])
        });
        let logits = outs.first().context("prefill missing logits")?.clone();
        let n_layers = self.config.num_hidden_layers;
        ensure!(
            outs.len() == 1 + 2 * n_layers,
            "prefill: expected {} outputs, got {}",
            1 + 2 * n_layers,
            outs.len()
        );
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let k = outs[1 + 2 * i].clone();
            let v = outs[1 + 2 * i + 1].clone();
            ensure!(
                k.len() == n_tokens * hidden && v.len() == n_tokens * hidden,
                "prefill KV layer {i}: unexpected len"
            );
            layers.push(LayerKv {
                k,
                v,
                prefill_len: n_tokens,
                ring_pos: None,
            });
        }
        Ok((logits, DeviceKvCache { layers }))
    }

    /// One decode step at absolute position `pos`.
    pub fn decode_step(
        &mut self,
        step_embed: &[f32],
        pos: usize,
        kv: &mut DeviceKvCache,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let n_layers = self.config.num_hidden_layers;
        ensure!(step_embed.len() == hidden, "decode: embed len");
        ensure!(kv.layers.len() == n_layers, "decode: KV layer count");

        let window = if self.config.sliding_window == 0 {
            usize::MAX
        } else {
            self.config.sliding_window
        };

        // Build past feeds (exclude ring overwrite slot in steady state).
        let mut past_feeds: Vec<(String, Vec<f32>)> = Vec::with_capacity(2 * n_layers);
        let mut past_seq = 0usize;
        for (i, layer) in kv.layers.iter().enumerate() {
            let (pk, pv) = past_for_decode(layer, window, hidden)?;
            if i == 0 {
                past_seq = pk.len() / hidden;
            } else {
                ensure!(pk.len() / hidden == past_seq, "decode past len mismatch");
            }
            past_feeds.push((format!("past_k_{i}"), pk));
            past_feeds.push((format!("past_v_{i}"), pv));
        }

        let (cos_2d, sin_2d) = compute_rope_slice(&self.config, pos);
        let feed_owned = past_feeds;

        let _ = self.ensure_decode(past_seq)?;
        let device = self.device;
        let outs = lm_runtime_guard_for_pack(device, &self.weights, || {
            let compiled = self.decode.get_mut(&past_seq).expect("decode");
            let mut run_pairs: Vec<(&str, &[f32])> = vec![
                ("inputs_embeds", step_embed),
                ("rope_cos", &cos_2d),
                ("rope_sin", &sin_2d),
            ];
            for (name, data) in &feed_owned {
                run_pairs.push((name.as_str(), data.as_slice()));
            }
            compiled.run(&run_pairs)
        });
        let logits = outs.first().context("decode missing logits")?.clone();
        ensure!(
            outs.len() == 1 + 2 * n_layers,
            "decode: expected {} outputs, got {}",
            1 + 2 * n_layers,
            outs.len()
        );

        // Side outputs are concat(past, new) — take last token as k_new/v_new,
        // then apply host ring update.
        for i in 0..n_layers {
            let full_k = &outs[1 + 2 * i];
            let full_v = &outs[1 + 2 * i + 1];
            let n_full = full_k.len() / hidden;
            ensure!(n_full >= 1, "decode KV empty");
            let k_new = &full_k[(n_full - 1) * hidden..n_full * hidden];
            let v_new = &full_v[(n_full - 1) * hidden..n_full * hidden];
            apply_ring_update(&mut kv.layers[i], k_new, v_new, hidden, window)?;
        }

        Ok(logits)
    }
}

/// Past tensors fed into the decode graph.
///
/// In ring steady-state, the slot about to be overwritten is dropped so
/// `concat(past, k_new)` restores length `prefill_len + window`.
fn past_for_decode(layer: &LayerKv, _window: usize, hidden: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let cur_len = layer.k.len() / hidden;
    ensure!(cur_len > 0, "empty KV");
    if let Some(ring_pos) = layer.ring_pos {
        // Steady state: buffer length == prefill + window; drop overwrite slot.
        let slot = layer.prefill_len + ring_pos;
        ensure!(slot < cur_len, "ring slot out of range");
        Ok((
            concat_without_row(&layer.k, cur_len, hidden, slot),
            concat_without_row(&layer.v, cur_len, hidden, slot),
        ))
    } else {
        Ok((layer.k.clone(), layer.v.clone()))
    }
}

fn concat_without_row(data: &[f32], n_rows: usize, hidden: usize, drop: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity((n_rows - 1) * hidden);
    for r in 0..n_rows {
        if r == drop {
            continue;
        }
        out.extend_from_slice(&data[r * hidden..(r + 1) * hidden]);
    }
    out
}

fn apply_ring_update(
    layer: &mut LayerKv,
    k_new: &[f32],
    v_new: &[f32],
    hidden: usize,
    window: usize,
) -> Result<()> {
    ensure!(k_new.len() == hidden && v_new.len() == hidden);
    let cur_len = layer.k.len() / hidden;
    if cur_len < layer.prefill_len + window {
        layer.k.extend_from_slice(k_new);
        layer.v.extend_from_slice(v_new);
        let new_len = cur_len + 1;
        if new_len >= layer.prefill_len + window {
            layer.ring_pos = Some(0);
        }
    } else {
        let ring_pos = layer.ring_pos.unwrap_or(0);
        let slot = layer.prefill_len + ring_pos;
        layer.k[slot * hidden..(slot + 1) * hidden].copy_from_slice(k_new);
        layer.v[slot * hidden..(slot + 1) * hidden].copy_from_slice(v_new);
        layer.ring_pos = Some((ring_pos + 1) % window);
    }
    Ok(())
}
