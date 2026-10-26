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
    build_unlimited_ocr_decode_built_from_pack_ext, build_unlimited_ocr_decode_chunk_built,
    build_unlimited_ocr_prefill_built_from_pack_ext, compute_rope_slice,
};
use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::compile_built;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::sync::Arc;

/// Host ring KV matching [`crate::lm_flow::LmFlow`] / HF SlidingWindowLlamaAttention.
struct LayerKv {
    /// `[cap, kv_w]` row-major, where `kv_w = num_kv_heads * head_dim` — the
    /// **pre-repeat** width the graph caches, which is narrower than
    /// `hidden_size` under GQA.
    ///
    /// In the full-causal path this is kept **pre-padded to the current KV
    /// bucket** with `valid` real rows at the front, so a decode step can feed
    /// `&k[..cap * kv_w]` directly instead of cloning the cache. The clone
    /// was per layer per token — 12 x 2 x up-to-126 MB on a 4k transcript.
    k: Vec<f32>,
    v: Vec<f32>,
    /// Rows of real history. Equals `k.len() / kv_w` on the windowed path,
    /// which does not pre-pad.
    valid: usize,
    prefill_len: usize,
    ring_pos: Option<usize>,
}

impl LayerKv {
    /// Grow the buffer to the bucket that covers `valid + 1` rows (the row this
    /// step is about to write), zero-filling the padding. Idempotent within a
    /// bucket, so most steps do no work at all.
    fn pad_to_bucket(&mut self, kv_w: usize) {
        let cap = bucket_for(self.valid + 1);
        if self.k.len() != cap * kv_w {
            self.k.resize(cap * kv_w, 0.0);
            self.v.resize(cap * kv_w, 0.0);
        }
    }

    /// Write one row at `valid` and advance. The buffer is already large enough
    /// because [`Self::pad_to_bucket`] ran before the step.
    fn append_in_place(&mut self, k_new: &[f32], v_new: &[f32], kv_w: usize) {
        let at = self.valid * kv_w;
        self.k[at..at + kv_w].copy_from_slice(k_new);
        self.v[at..at + kv_w].copy_from_slice(v_new);
        self.valid += 1;
    }
}

pub struct DeviceKvCache {
    layers: Vec<LayerKv>,
    /// Identifies THIS cache, so resident device handles bound from it are
    /// never reused for a different one.
    ///
    /// Load-bearing: resident K/V that is re-seeded only when the bucket
    /// changes answers the PREVIOUS prompt on a second generate() with the
    /// same runner — a real bug shipped once in rlx-qwen35's Metal path. Every
    /// prefill mints a new cache, so comparing ids catches it; comparing bucket
    /// widths does not.
    id: u64,
}

/// Mints [`DeviceKvCache::id`].
static KV_CACHE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// What a decode graph's resident K/V handles currently hold.
#[derive(Debug, Clone, Copy)]
struct ResidentKv {
    /// The cache the handles were bound from.
    cache_id: u64,
    /// Rows of real history in the handles — must equal the incoming cache's
    /// `valid`, or a step was taken outside this path (speculative rollback,
    /// a chunk step) and the handles are stale.
    valid: usize,
}

impl DeviceKvCache {
    /// Rows of real history.
    pub fn valid(&self) -> usize {
        self.layers.first().map_or(0, |l| l.valid)
    }

    /// Drop history back to `valid` rows.
    ///
    /// Speculative decode writes the whole draft into the cache and then keeps
    /// only the accepted prefix. The dropped rows stay in the buffer as
    /// padding and are overwritten by the next step; the mask never shows them
    /// to attention, because it is built from `valid`.
    pub fn rollback_to(&mut self, valid: usize) {
        for l in self.layers.iter_mut() {
            debug_assert!(valid <= l.valid, "rollback must not invent history");
            l.valid = valid;
        }
    }
}

/// Compiled Unlimited-OCR LM on a concrete RLX [`Device`].
pub struct CompiledLm {
    device: Device,
    config: UnlimitedOcrConfig,
    weights: Arc<PackedLmWeights>,
    prefill: HashMap<(usize, bool), CompiledGraph>,
    decode: HashMap<usize, CompiledGraph>,
    /// Verify graphs, keyed by `(bucket, chunk width)`.
    chunk: HashMap<(usize, usize, bool), CompiledGraph>,
    /// What each decode graph's resident K/V handles hold, keyed by `past_seq`.
    /// Absent = the next step must (re)bind.
    resident: HashMap<usize, ResidentKv>,
    /// Decode graphs whose backend refused the resident contract — those keep
    /// the host feed forever rather than retrying every step.
    resident_unsupported: std::collections::HashSet<usize>,
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
            chunk: HashMap::new(),
            resident: HashMap::new(),
            resident_unsupported: std::collections::HashSet::new(),
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn config(&self) -> &UnlimitedOcrConfig {
        &self.config
    }

    /// Distinct decode graphs compiled so far, keyed by `past_seq`.
    ///
    /// Exposed because the whole point of bucketing is that this stays small:
    /// a full-causal run that compiles one graph per generated token is
    /// functionally correct and practically unusable, and nothing else in the
    /// output would show it.
    pub fn compiled_decode_graphs(&self) -> usize {
        self.decode.len()
    }

    /// Compiled prefill graphs, keyed by sequence length.
    pub fn compiled_prefill_graphs(&self) -> usize {
        self.prefill.len()
    }

    /// Decode graphs currently holding a RESIDENT K/V cache.
    ///
    /// Exposed so a test can tell "the resident path agrees with the host feed"
    /// apart from "the resident path never ran" — the two look identical in
    /// output, and the fallback is silent by design (an unsupported backend,
    /// a windowed checkpoint, or `RLX_DISABLE_GPU_KV`).
    pub fn resident_decode_graphs(&self) -> usize {
        self.resident.len()
    }

    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.weights.embed_tokens_lookup(ids)
    }

    fn ensure_prefill(&mut self, seq: usize, with_hidden: bool) -> Result<&CompiledGraph> {
        let key = (seq, with_hidden);
        if !self.prefill.contains_key(&key) {
            let built = build_unlimited_ocr_prefill_built_from_pack_ext(
                &self.config,
                &self.weights,
                1,
                seq,
                with_hidden,
            )?;
            let device = self.device;
            let compiled =
                lm_runtime_guard_for_pack(device, &self.weights, || compile_built(built, device))?;
            self.prefill.insert(key, compiled);
        }
        Ok(self.prefill.get(&key).expect("prefill just inserted"))
    }

    fn ensure_decode(&mut self, past_seq: usize) -> Result<&CompiledGraph> {
        if !self.decode.contains_key(&past_seq) {
            let built = build_unlimited_ocr_decode_built_from_pack_ext(
                &self.config,
                &self.weights,
                1,
                past_seq,
                self.window().is_none(),
            )?;
            let device = self.device;
            let compiled =
                lm_runtime_guard_for_pack(device, &self.weights, || compile_built(built, device))?;
            self.decode.insert(past_seq, compiled);
        }
        Ok(self.decode.get(&past_seq).expect("decode just inserted"))
    }

    /// [`Self::prefill`] that also returns the last token's post-final-norm
    /// hidden state — the seed FastMTP's draft head needs.
    pub fn prefill_with_hidden(
        &mut self,
        inputs_embeds: &[f32],
        n_tokens: usize,
    ) -> Result<(Vec<f32>, DeviceKvCache, Vec<f32>)> {
        let (logits, kv, all) = self.prefill_with_all_hidden(inputs_embeds, n_tokens)?;
        let hidden = self.config.hidden_size;
        let last = all[(n_tokens - 1) * hidden..].to_vec();
        Ok((logits, kv, last))
    }

    /// [`Self::prefill_with_hidden`] keeping **every** position's hidden state,
    /// laid out `[n_tokens, hidden]`.
    ///
    /// A draft head needs the whole run, not just the last row: its own
    /// attention layer has to be primed over the prompt or it proposes tokens
    /// with no context behind them.
    pub fn prefill_with_all_hidden(
        &mut self,
        inputs_embeds: &[f32],
        n_tokens: usize,
    ) -> Result<(Vec<f32>, DeviceKvCache, Vec<f32>)> {
        let (logits, kv, hidden) = self.prefill_inner(inputs_embeds, n_tokens, true)?;
        let hidden = hidden.expect("hidden requested");
        let want = n_tokens * self.config.hidden_size;
        ensure!(
            hidden.len() == want,
            "prefill hidden tap is {} elements, want {n_tokens}*{} = {want}",
            hidden.len(),
            self.config.hidden_size
        );
        Ok((logits, kv, hidden))
    }

    /// Prefill over `[n_tokens, hidden]` embeds → last-token logits + KV cache.
    pub fn prefill(
        &mut self,
        inputs_embeds: &[f32],
        n_tokens: usize,
    ) -> Result<(Vec<f32>, DeviceKvCache)> {
        self.prefill_inner(inputs_embeds, n_tokens, false)
            .map(|(l, kv, _)| (l, kv))
    }

    fn prefill_inner(
        &mut self,
        inputs_embeds: &[f32],
        n_tokens: usize,
        with_hidden: bool,
    ) -> Result<(Vec<f32>, DeviceKvCache, Option<Vec<f32>>)> {
        let hidden = self.config.hidden_size;
        let kv_w = self.config.kv_hidden();
        ensure!(
            inputs_embeds.len() == n_tokens * hidden,
            "prefill: embeds len {} != {n_tokens}*{hidden}",
            inputs_embeds.len()
        );
        let _ = self.ensure_prefill(n_tokens, with_hidden)?;
        let device = self.device;
        let key = (n_tokens, with_hidden);
        let outs = lm_runtime_guard_for_pack(device, &self.weights, || {
            let compiled = self.prefill.get_mut(&key).expect("prefill");
            compiled.run(&[("inputs_embeds", inputs_embeds)])
        });
        let logits = outs.first().context("prefill missing logits")?.clone();
        let n_layers = self.config.num_hidden_layers;
        let want = if with_hidden { 2 } else { 1 } + 2 * n_layers;
        ensure!(
            outs.len() == want,
            "prefill: expected {want} outputs, got {}",
            outs.len()
        );
        let tapped = with_hidden.then(|| outs.last().expect("hidden tap").clone());
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let k = outs[1 + 2 * i].clone();
            let v = outs[1 + 2 * i + 1].clone();
            ensure!(
                k.len() == n_tokens * kv_w && v.len() == n_tokens * kv_w,
                "prefill KV layer {i}: unexpected len"
            );
            layers.push(LayerKv {
                valid: n_tokens,
                k,
                v,
                prefill_len: n_tokens,
                ring_pos: None,
            });
        }
        Ok((
            logits,
            DeviceKvCache {
                layers,
                id: KV_CACHE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            },
            tapped,
        ))
    }

    /// Rolling-window span, or `None` when the checkpoint has no sliding window
    /// (`sliding_window == 0`) and attention is plain causal over the whole
    /// history — DeepSeek-OCR derivatives such as `jinaai/jina-ocr-v1`.
    fn window(&self) -> Option<usize> {
        (self.config.sliding_window > 0).then_some(self.config.sliding_window)
    }

    /// One decode step at absolute position `pos`.
    pub fn decode_step(
        &mut self,
        step_embed: &[f32],
        pos: usize,
        kv: &mut DeviceKvCache,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let kv_w = self.config.kv_hidden();
        let n_layers = self.config.num_hidden_layers;
        ensure!(step_embed.len() == hidden, "decode: embed len");
        ensure!(kv.layers.len() == n_layers, "decode: KV layer count");

        let window = self.window();

        // Build past feeds.
        //
        // Full-causal: each layer's buffer is already padded to its bucket, so
        // the feed is a borrowed slice — no per-step clone of the cache.
        // Windowed: the ring path still materializes, because it drops the slot
        // about to be overwritten and so cannot hand out a contiguous slice.
        let (past_seq, past_valid) = match window {
            None => {
                for layer in kv.layers.iter_mut() {
                    layer.pad_to_bucket(kv_w);
                }
                let first = &kv.layers[0];
                let (cap, valid) = (first.k.len() / kv_w, first.valid);
                for layer in kv.layers.iter() {
                    ensure!(layer.k.len() / kv_w == cap, "decode past len mismatch");
                    ensure!(layer.valid == valid, "decode past valid-len mismatch");
                }
                (cap, valid)
            }
            Some(_) => {
                let (k, _) = past_for_decode(&kv.layers[0], kv_w)?;
                let n = k.len() / kv_w;
                (n, n)
            }
        };
        // Only the windowed path needs owned buffers.
        let windowed_feeds: Vec<(String, Vec<f32>)> = if window.is_some() {
            let mut out = Vec::with_capacity(2 * n_layers);
            for (i, layer) in kv.layers.iter().enumerate() {
                let (pk, pv) = past_for_decode(layer, kv_w)?;
                ensure!(pk.len() / kv_w == past_seq, "decode past len mismatch");
                out.push((format!("past_k_{i}"), pk));
                out.push((format!("past_v_{i}"), pv));
            }
            out
        } else {
            Vec::new()
        };
        let past_names: Vec<(String, String)> = (0..n_layers)
            .map(|i| (format!("past_k_{i}"), format!("past_v_{i}")))
            .collect();

        let (cos_2d, sin_2d) = compute_rope_slice(&self.config, pos);
        // `1.0` = attend, `0.0` = ignore, over (padded past ++ new token).
        let mask = if window.is_none() {
            let mut m = vec![0f32; past_seq + 1];
            m[..past_valid].fill(1.0);
            m[past_seq] = 1.0;
            Some(m)
        } else {
            None
        };

        let _ = self.ensure_decode(past_seq)?;
        let device = self.device;

        // RESIDENT K/V, full-causal only.
        //
        // The in-place append (`Op::KvAppend`) made the cache GROWTH O(1), but
        // handing the whole padded cache back in as a `past_k_*` input every
        // step is O(context) on its own, and measurement said it is the bigger
        // half: `tests/inplace_kv_bench.rs` showed per-step time still growing
        // ~6x from 256 to 4096 rows with the append on. So keep the cache
        // resident in the executable, feed nothing, and fold the one new row in
        // on device — which is also where the append's output already is, since
        // it aliases the bound buffer.
        //
        // The WINDOWED path cannot do this: `past_for_decode` drops the slot
        // about to be recycled, so its feed is not a monotonically growing
        // prefix of one buffer.
        let want_resident = window.is_none()
            && !self.resident_unsupported.contains(&past_seq)
            && resident_kv_device(device);
        let rebind = want_resident
            && self
                .resident
                .get(&past_seq)
                .copied()
                .is_none_or(|r| r.cache_id != kv.id || r.valid != past_valid);
        if rebind && !self.bind_resident_kv(past_seq, kv, kv_w, n_layers) {
            self.resident_unsupported.insert(past_seq);
            self.resident.remove(&past_seq);
        }
        let resident = want_resident && !self.resident_unsupported.contains(&past_seq);

        let read_only_logits = resident.then_some([0usize]);
        let outs = lm_runtime_guard_for_pack(device, &self.weights, || {
            let compiled = self.decode.get_mut(&past_seq).expect("decode");
            let mut run_pairs: Vec<(&str, &[f32])> = vec![
                ("inputs_embeds", step_embed),
                ("rope_cos", &cos_2d),
                ("rope_sin", &sin_2d),
            ];
            if let Some(mask) = &mask {
                run_pairs.push(("mask", mask.as_slice()));
            }
            if resident {
                // Nothing to feed: the handles hold the history.
            } else if windowed_feeds.is_empty() {
                for (i, (kn, vn)) in past_names.iter().enumerate() {
                    let layer = &kv.layers[i];
                    run_pairs.push((kn.as_str(), layer.k.as_slice()));
                    run_pairs.push((vn.as_str(), layer.v.as_slice()));
                }
            } else {
                for (name, data) in &windowed_feeds {
                    run_pairs.push((name.as_str(), data.as_slice()));
                }
            }
            match &read_only_logits {
                Some(idx) => compiled.run_read_outputs(&run_pairs, Some(idx)),
                None => compiled.run(&run_pairs),
            }
        });
        let logits = outs.first().context("decode missing logits")?.clone();

        if resident {
            // The append wrote the new row at `past_seq` (the spare row), not at
            // `valid` — the graph is compiled for a fixed bucket and cannot know
            // how much of it is real. Read that one row (O(1)) to keep the host
            // mirror able to re-seed a wider bucket, then relocate it on device.
            let compiled = self.decode.get_mut(&past_seq).expect("decode");
            for i in 0..n_layers {
                let k_new = compiled
                    .read_output_row(1 + 2 * i, past_seq, kv_w)
                    .with_context(|| format!("resident K row read, layer {i}"))?;
                let v_new = compiled
                    .read_output_row(2 + 2 * i, past_seq, kv_w)
                    .with_context(|| format!("resident V row read, layer {i}"))?;
                apply_ring_update(&mut kv.layers[i], &k_new, &v_new, kv_w, window)?;
            }
            ensure!(
                compiled.feed_kv_row(past_seq, past_valid, kv_w),
                "resident K/V row feed refused after binding succeeded"
            );
            self.resident.insert(
                past_seq,
                ResidentKv {
                    cache_id: kv.id,
                    valid: past_valid + 1,
                },
            );
            return Ok(logits);
        }

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
            let n_full = full_k.len() / kv_w;
            ensure!(n_full >= 1, "decode KV empty");
            let k_new = &full_k[(n_full - 1) * kv_w..n_full * kv_w];
            let v_new = &full_v[(n_full - 1) * kv_w..n_full * kv_w];
            apply_ring_update(&mut kv.layers[i], k_new, v_new, kv_w, window)?;
        }

        Ok(logits)
    }

    /// Bind this graph's `past_k_*` / `past_v_*` inputs as resident handles
    /// seeded from the host cache, and register which output carries each
    /// grown cache. `false` = the backend does not support it.
    ///
    /// Padded to the DECLARED input width, which is `past_seq + 1` whenever the
    /// flow took the spare row for the in-place append — not `past_seq`.
    fn bind_resident_kv(
        &mut self,
        past_seq: usize,
        kv: &DeviceKvCache,
        kv_w: usize,
        n_layers: usize,
    ) -> bool {
        let Some(compiled) = self.decode.get_mut(&past_seq) else {
            return false;
        };
        if !compiled.capabilities().kv_resident {
            return false;
        }
        let spare = rlx_flow::blocks::kv_spare_rows(&[1]);
        let width = (past_seq + spare) * kv_w;
        let mut pad = vec![0f32; width];
        for i in 0..n_layers {
            let (k_name, v_name) = (format!("past_k_{i}"), format!("past_v_{i}"));
            for (name, src) in [(&k_name, &kv.layers[i].k), (&v_name, &kv.layers[i].v)] {
                pad.clear();
                pad.resize(width, 0.0);
                let n = src.len().min(width);
                pad[..n].copy_from_slice(&src[..n]);
                if !compiled.bind_gpu_handle(name.as_str(), &pad) {
                    return false;
                }
            }
            if !compiled.register_kv_row_feed(&k_name, 1 + 2 * i)
                || !compiled.register_kv_row_feed(&v_name, 2 + 2 * i)
            {
                return false;
            }
        }
        self.resident.insert(
            past_seq,
            ResidentKv {
                cache_id: kv.id,
                valid: kv.valid(),
            },
        );
        true
    }
}

impl CompiledLm {
    /// Drop cache history back to `valid` rows (see
    /// [`DeviceKvCache::rollback_to`]).
    pub fn rollback(&self, kv: &mut DeviceKvCache, valid: usize) {
        kv.rollback_to(valid);
    }

    /// The live `[valid, kv_w]` key and value rows of one layer.
    ///
    /// Only the real history — the bucket padding past `valid` is stale by
    /// design, so handing it out would make a correct cache look wrong. Exists
    /// so a test can check what speculation left behind: whether a rejected
    /// draft was rolled back is invisible in the emitted tokens when the model
    /// is small enough that argmax survives the extra keys.
    pub fn layer_kv<'a>(
        &self,
        kv: &'a DeviceKvCache,
        layer: usize,
    ) -> Option<(&'a [f32], &'a [f32])> {
        let kv_w = self.config.kv_hidden();
        let l = kv.layers.get(layer)?;
        let n = l.valid * kv_w;
        Some((&l.k[..n], &l.v[..n]))
    }

    /// Decode `n` tokens in one forward against the cache — the speculative
    /// verify pass. Returns per-position logits.
    ///
    /// All `n` rows are appended to the cache; a caller that accepts fewer
    /// calls [`DeviceKvCache::rollback_to`]. Full-causal only: the windowed
    /// ring cache cannot hand out the contiguous padded slice this needs.
    pub fn decode_chunk(
        &mut self,
        embeds: &[f32],
        first_pos: usize,
        n: usize,
        kv: &mut DeviceKvCache,
    ) -> Result<Vec<Vec<f32>>> {
        self.decode_chunk_inner(embeds, first_pos, n, kv, false)
            .map(|(l, _)| l)
    }

    /// [`Self::decode_chunk`] that also returns each position's post-final-norm
    /// hidden state, so a speculative loop can reseed its draft head from the
    /// last accepted position without a second forward.
    pub fn decode_chunk_with_hidden(
        &mut self,
        embeds: &[f32],
        first_pos: usize,
        n: usize,
        kv: &mut DeviceKvCache,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let (logits, hidden) = self.decode_chunk_inner(embeds, first_pos, n, kv, true)?;
        Ok((logits, hidden.expect("hidden requested")))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_chunk_inner(
        &mut self,
        embeds: &[f32],
        first_pos: usize,
        n: usize,
        kv: &mut DeviceKvCache,
        with_hidden: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>)> {
        let hidden = self.config.hidden_size;
        let kv_w = self.config.kv_hidden();
        let n_layers = self.config.num_hidden_layers;
        let heads = self.config.num_attention_heads;
        let vocab = self.config.vocab_size;
        ensure!(n >= 1, "decode_chunk: n must be >= 1");
        ensure!(
            self.window().is_none(),
            "decode_chunk is only defined for the full-causal path"
        );
        ensure!(embeds.len() == n * hidden, "decode_chunk: embeds length");
        ensure!(kv.layers.len() == n_layers, "decode_chunk: KV layer count");

        // Pad so the whole chunk fits, then feed the buffers by reference.
        for layer in kv.layers.iter_mut() {
            let cap = bucket_for(layer.valid + n);
            if layer.k.len() != cap * kv_w {
                layer.k.resize(cap * kv_w, 0.0);
                layer.v.resize(cap * kv_w, 0.0);
            }
        }
        let valid = kv.layers[0].valid;
        let cap = kv.layers[0].k.len() / kv_w;
        for layer in kv.layers.iter() {
            ensure!(
                layer.valid == valid && layer.k.len() / kv_w == cap,
                "ragged KV"
            );
        }

        let mut cos = Vec::with_capacity(n * hidden);
        let mut sin = Vec::with_capacity(n * hidden);
        for i in 0..n {
            let (c, s) = compute_rope_slice(&self.config, first_pos + i);
            cos.extend_from_slice(&c);
            sin.extend_from_slice(&s);
        }
        // The graph picks its mask kind from `seq`: a single query gets the
        // cheaper binary keep-mask, a chunk gets the additive per-query bias.
        // Build whichever it declared, or the leaf shape will not match.
        let mask = if n == 1 {
            let mut m = vec![0f32; cap + 1];
            m[..valid].fill(1.0);
            m[cap] = 1.0;
            m
        } else {
            chunk_bias_mask(heads, n, valid, cap)
        };

        let key = (cap, n, with_hidden);
        if !self.chunk.contains_key(&key) {
            let built = build_unlimited_ocr_decode_chunk_built(
                &self.config,
                &self.weights,
                1,
                cap,
                n,
                true,
                with_hidden,
            )?;
            let device = self.device;
            let compiled =
                lm_runtime_guard_for_pack(device, &self.weights, || compile_built(built, device))?;
            self.chunk.insert(key, compiled);
        }

        let names: Vec<(String, String)> = (0..n_layers)
            .map(|i| (format!("past_k_{i}"), format!("past_v_{i}")))
            .collect();
        let device = self.device;
        let outs = lm_runtime_guard_for_pack(device, &self.weights, || {
            let compiled = self.chunk.get_mut(&key).expect("chunk graph");
            let mut pairs: Vec<(&str, &[f32])> = vec![
                ("inputs_embeds", embeds),
                ("rope_cos", &cos),
                ("rope_sin", &sin),
                ("mask", &mask),
            ];
            for (i, (kn, vn)) in names.iter().enumerate() {
                pairs.push((kn.as_str(), kv.layers[i].k.as_slice()));
                pairs.push((vn.as_str(), kv.layers[i].v.as_slice()));
            }
            compiled.run(&pairs)
        });
        let want = if with_hidden { 2 } else { 1 } + 2 * n_layers;
        ensure!(
            outs.len() == want,
            "decode_chunk: expected {want} outputs, got {}",
            outs.len()
        );

        // Side outputs are concat(padded past, chunk); the chunk's own rows are
        // the last `n`.
        for l in 0..n_layers {
            for (tensor, into_k) in [(&outs[1 + 2 * l], true), (&outs[1 + 2 * l + 1], false)] {
                let rows = tensor.len() / kv_w;
                ensure!(rows >= n, "decode_chunk: KV too short");
                let src = &tensor[(rows - n) * kv_w..];
                let layer = &mut kv.layers[l];
                let at = valid * kv_w;
                let dst = if into_k { &mut layer.k } else { &mut layer.v };
                dst[at..at + n * kv_w].copy_from_slice(src);
            }
        }
        for layer in kv.layers.iter_mut() {
            layer.valid = valid + n;
        }

        let logits = &outs[0];
        ensure!(logits.len() == n * vocab, "decode_chunk: logits length");
        let per_pos: Vec<Vec<f32>> = (0..n)
            .map(|i| logits[i * vocab..(i + 1) * vocab].to_vec())
            .collect();
        let tapped = with_hidden.then(|| {
            let h = outs.last().expect("hidden tap");
            (0..n)
                .map(|i| h[i * hidden..(i + 1) * hidden].to_vec())
                .collect::<Vec<_>>()
        });
        Ok((per_pos, tapped))
    }
}

/// Additive attention bias `[1, heads, n, cap + n]` for a chunk starting after
/// `valid` real rows: `0` where visible, `-inf` elsewhere.
///
/// Encodes two things a binary key-padding mask cannot: the bucket padding
/// (`valid..cap`) is invisible, and query `i` may not see chunk position
/// `j > i`. Leaking the latter would let a draft token attend to its own
/// successors, which inflates acceptance and silently changes the output.
fn chunk_bias_mask(heads: usize, n: usize, valid: usize, cap: usize) -> Vec<f32> {
    let keys = cap + n;
    let mut m = vec![f32::NEG_INFINITY; heads * n * keys];
    for h in 0..heads {
        for i in 0..n {
            let row = (h * n + i) * keys;
            m[row..row + valid].fill(0.0);
            for j in 0..=i {
                m[row + cap + j] = 0.0;
            }
        }
    }
    m
}

/// KV bucket granularity for full-causal (no sliding window) decode.
///
/// The decode graph's `past_seq` is the bucket, not the exact history length,
/// so it is compiled once per bucket instead of once per generated token.
/// Smaller buckets waste less per-step padding bandwidth at the cost of more
/// compiles; 256 keeps both bounded for 4k-token OCR transcripts.
pub const KV_BUCKET: usize = 256;

/// Smallest multiple of [`KV_BUCKET`] that is `>= n`, and at least `KV_BUCKET`.
fn bucket_for(n: usize) -> usize {
    n.max(1).div_ceil(KV_BUCKET) * KV_BUCKET
}

/// Opt into resident K/V only on backends where THIS crate's parity tests have
/// actually run.
///
/// The row-feed contract is backend-wide and `capabilities().kv_resident` says
/// whether a backend implements it — but implementing it is not the same as
/// getting it right, and two backends have shipped real bugs in exactly this
/// path: rlx-qwen35's Metal resident cache answered the previous prompt on a
/// second generate(), and wgpu's native GGUF decode got stuck repeating a
/// token. Both were silent. So the default is the set that
/// `tests/resident_kv_parity.rs` has been run against, and anything else needs
/// `RLX_UNLIMITED_OCR_RESIDENT_KV_ANY=1` — a request to measure, not a claim
/// that it works. `RLX_DISABLE_GPU_KV=1` turns the whole path off.
///
/// The verified set, and what each entry rests on:
///
/// * **CPU, Metal** — parity green on an M-series mac; 1.46-1.61x per decode
///   step over the host feed (`tests/inplace_kv_bench.rs`).
/// * **CUDA** — parity green on an RTX 3080 Ti (both debug and release,
///   including the stale-handle and rollback cases), and by far the biggest win
///   measured anywhere: **3.0x @256, 3.96x @1024, 4.6x @2048**, four runs on an
///   idle box agreeing to within 0.5% (0.50/0.86/1.33ms resident vs
///   1.50/3.40/6.10ms host-feed). The gain grows with context because the feed
///   it removes is O(context) — a per-step PCIe upload of the whole padded
///   cache, 16.8 MB at the bench config, out of pageable host memory — while
///   the resident arm only grows with the attention it genuinely does.
///
///   Take the box's load average before quoting any of this. The first pass
///   here was measured while that machine ran someone else's benchmark and read
///   1.91x; contention is roughly additive, so it hits the faster arm hardest
///   and squeezes ratios toward 1.0. Same code, same config, idle: 4.6x.
///
/// Not in the set: mlx, wgpu, vulkan, rocm. Not "broken" — untested, and wgpu
/// is where the known bug was, so promote one only by running the parity tests
/// on it with the override and then adding it here.
fn resident_kv_device(device: Device) -> bool {
    if !rlx_core::gpu_kv::device_supports_resident_kv_row_feed(device) {
        return false;
    }
    matches!(device, Device::Cpu | Device::Metal | Device::Cuda)
        || rlx_ir::env::flag("RLX_UNLIMITED_OCR_RESIDENT_KV_ANY")
}

/// Whether [`CompiledLm`] will keep its K/V cache resident on `device`.
///
/// Public so the parity tests can tell "this backend is outside the verified
/// set" (skip, and say so) from "this backend is in it and the resident path
/// still did not engage" (a bug). See [`resident_kv_device`].
pub fn resident_kv_available(device: Device) -> bool {
    resident_kv_device(device)
}

/// Past tensors fed into the decode graph.
///
/// In ring steady-state, the slot about to be overwritten is dropped so
/// `concat(past, k_new)` restores length `prefill_len + window`.
fn past_for_decode(layer: &LayerKv, kv_w: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let cur_len = layer.k.len() / kv_w;
    ensure!(cur_len > 0, "empty KV");
    if let Some(ring_pos) = layer.ring_pos {
        // Steady state: buffer length == prefill + window; drop overwrite slot.
        let slot = layer.prefill_len + ring_pos;
        ensure!(slot < cur_len, "ring slot out of range");
        Ok((
            concat_without_row(&layer.k, cur_len, kv_w, slot),
            concat_without_row(&layer.v, cur_len, kv_w, slot),
        ))
    } else {
        Ok((layer.k.clone(), layer.v.clone()))
    }
}

fn concat_without_row(data: &[f32], n_rows: usize, kv_w: usize, drop: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity((n_rows - 1) * kv_w);
    for r in 0..n_rows {
        if r == drop {
            continue;
        }
        out.extend_from_slice(&data[r * kv_w..(r + 1) * kv_w]);
    }
    out
}

/// Append `k_new`/`v_new` to the cache.
///
/// `window = None` (no sliding window) grows the cache forever — plain causal
/// attention re-reads the whole history. `Some(w)` keeps `prefill_len + w` rows
/// and recycles the post-prefill tail in a ring.
fn apply_ring_update(
    layer: &mut LayerKv,
    k_new: &[f32],
    v_new: &[f32],
    kv_w: usize,
    window: Option<usize>,
) -> Result<()> {
    ensure!(k_new.len() == kv_w && v_new.len() == kv_w);
    let Some(window) = window else {
        // Buffer is pre-padded to the bucket; write the row in place rather
        // than growing (which would invalidate the borrowed feed slices).
        layer.append_in_place(k_new, v_new, kv_w);
        return Ok(());
    };
    let cur_len = layer.k.len() / kv_w;
    if cur_len < layer.prefill_len + window {
        layer.k.extend_from_slice(k_new);
        layer.v.extend_from_slice(v_new);
        layer.valid = cur_len + 1;
        let new_len = cur_len + 1;
        if new_len >= layer.prefill_len + window {
            layer.ring_pos = Some(0);
        }
    } else {
        let ring_pos = layer.ring_pos.unwrap_or(0);
        let slot = layer.prefill_len + ring_pos;
        layer.k[slot * kv_w..(slot + 1) * kv_w].copy_from_slice(k_new);
        layer.v[slot * kv_w..(slot + 1) * kv_w].copy_from_slice(v_new);
        layer.ring_pos = Some((ring_pos + 1) % window);
    }
    Ok(())
}
