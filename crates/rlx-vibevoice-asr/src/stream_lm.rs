// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Dense BF16 Qwen2 LM session for Streaming ASR: embeds prefill, multi-token
// embeds continue (past KV), and single-token decode — all via `inputs_embeds`.
//
// Performance notes:
// - LM weights are dequantized once into an in-RAM snapshot; bucket rebuilds
//   clone from that map (no repeated safetensors I/O).
// - Intermediate speech-frame steps use a KV-only decode graph (no lm_head);
//   logits are computed only on the last frame of a continue and during text
//   generation (~25× fewer vocab projections per audio chunk).

use anyhow::{Result, anyhow, ensure};
use rlx_core::autoregressive::{KvCacheState, kv_from_prefill_outputs, run_bucketed_kv_decode};
use rlx_core::flow_bridge::{WeightLoaderSource, compile_options_from_profile};
use rlx_core::flow_util::{compile_built, graph_from_built};
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::blocks::{
    LmHeadStage, Qwen3DecodeLayerSpec, Qwen3DecoderSpec, RopeTablesStage, qwen3_decode_layer_fused,
    qwen3_prefill_layer_fused_kv,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_ir::{DType, Shape};
use rlx_qwen3::Qwen3Config;
use rlx_runtime::Device;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::{LmConfig, TOK_ENDOFTEXT, TOK_IM_END, TOK_TEXT_CHUNK_END};
use crate::embed::argmax;
use crate::lm::qwen3_config;
use crate::load_streaming::{StreamingWeightStore, map_streaming_lm_key};

/// Stop tokens for streaming chunk generation.
const STREAM_STOP: [i64; 3] = [TOK_TEXT_CHUNK_END, TOK_IM_END, TOK_ENDOFTEXT];

type WeightSnap = Arc<HashMap<String, (Vec<f32>, Vec<usize>)>>;

#[derive(Default)]
struct StreamCaches {
    prefill: HashMap<usize, rlx_runtime::CompiledGraph>,
    /// Decode with lm_head (text gen + last speech frame).
    decode: Option<(u64, BucketedCompileCache)>,
    /// Decode without lm_head (intermediate speech frames → KV only).
    decode_kv: Option<(u64, BucketedCompileCache)>,
}

/// Dense Streaming LM over a safetensors checkpoint.
pub struct StreamLm {
    pub cfg: Qwen3Config,
    device: Device,
    token_embed: Vec<f32>,
    vocab: usize,
    hidden: usize,
    /// HF-keyed f32 LM weights (loaded once).
    lm_snap: WeightSnap,
    caches: RefCell<StreamCaches>,
}

/// WeightLoader over an in-RAM snapshot (clones tensors on `take`).
struct SnapshotLmLoader {
    snap: WeightSnap,
}

impl WeightLoader for SnapshotLmLoader {
    fn format_id(&self) -> &'static str {
        "safetensors-snap"
    }
    fn len(&self) -> usize {
        self.snap.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let mapped = map_streaming_lm_key(key);
        let (data, shape) = self
            .snap
            .get(&mapped)
            .ok_or_else(|| anyhow!("LM snapshot missing `{mapped}` (for `{key}`)"))?;
        Ok((data.clone(), shape.clone()))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            return Ok((data, shape));
        }
        let (r, c) = (shape[0], shape[1]);
        let mut out = vec![0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = data[i * c + j];
            }
        }
        Ok((out, vec![c, r]))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.snap.keys().cloned().collect()
    }
}

impl StreamLm {
    pub fn load(store: StreamingWeightStore, lm: &LmConfig, device: Device) -> Result<Self> {
        let (token_embed, vocab, hidden) = store.load_token_embed()?;
        ensure!(
            hidden == lm.hidden_size,
            "embed hidden {hidden} != config {}",
            lm.hidden_size
        );
        ensure!(
            vocab == lm.vocab_size,
            "embed vocab {vocab} != config {}",
            lm.vocab_size
        );
        // One-shot BF16→f32 load into RAM; later graph builds clone from this.
        let mut wm = store.load_language_model_weights()?;
        let keys: Vec<String> = wm.keys().map(str::to_string).collect();
        let mut snap = HashMap::with_capacity(keys.len());
        for k in keys {
            snap.insert(k.clone(), wm.take(&k)?);
        }
        Ok(Self {
            cfg: qwen3_config(lm),
            device,
            token_embed,
            vocab,
            hidden,
            lm_snap: Arc::new(snap),
            caches: RefCell::new(StreamCaches::default()),
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn token_embed(&self) -> &[f32] {
        &self.token_embed
    }

    /// Borrow embedding row for `id` (no allocation).
    pub fn embed_row(&self, id: i64) -> Result<&[f32]> {
        ensure!(id >= 0 && (id as usize) < self.vocab, "token id {id} OOB");
        let t = id as usize;
        Ok(&self.token_embed[t * self.hidden..(t + 1) * self.hidden])
    }

    pub fn embed_token(&self, id: i64) -> Result<Vec<f32>> {
        Ok(self.embed_row(id)?.to_vec())
    }

    fn skip_fusion(&self) -> bool {
        matches!(self.device, Device::Metal)
    }

    fn snap_loader(&self) -> SnapshotLmLoader {
        SnapshotLmLoader {
            snap: self.lm_snap.clone(),
        }
    }

    /// Prefill from `inputs_embeds` `[seq * hidden]` → last-token logits + KV.
    pub fn prefill_embeds(
        &self,
        inputs_embeds: &[f32],
        seq: usize,
    ) -> Result<(Vec<f32>, KvCacheState)> {
        let batch = 1;
        let skip_fusion = self.skip_fusion();
        let cfg = &self.cfg;
        ensure!(
            inputs_embeds.len() == seq * self.hidden,
            "inputs_embeds len {} != seq*hidden {}",
            inputs_embeds.len(),
            seq * self.hidden
        );

        let outs = {
            let mut caches = self.caches.borrow_mut();
            if let std::collections::hash_map::Entry::Vacant(e) = caches.prefill.entry(seq) {
                let mut loader = self.snap_loader();
                let built = build_stream_prefill_built(cfg, &mut loader, batch, seq, skip_fusion)?;
                let params = built.params().clone();
                let mut prefill = compile_built(built, self.device)?;
                for (n, d) in &params {
                    prefill.set_param(n, d);
                }
                e.insert(prefill);
            }
            let prefill = caches.prefill.get_mut(&seq).expect("prefill cached");
            prefill.run(&[("inputs_embeds", inputs_embeds)])
        };
        ensure!(
            outs[0].len() == batch * self.vocab,
            "prefill logits len {} != {}",
            outs[0].len(),
            batch * self.vocab
        );
        let kv_dim = cfg.kv_proj_dim();
        kv_from_prefill_outputs(outs, batch, seq, kv_dim, cfg.num_hidden_layers)
    }

    /// Ensure decode bucket ladders exist up to `max_total`.
    pub fn prepare_decode_ladder(&self, max_total: u64) {
        let mut caches = self.caches.borrow_mut();
        if !matches!(&caches.decode, Some((mt, _)) if *mt == max_total) {
            caches.decode = Some((
                max_total,
                BucketedCompileCache::power_of_two_ladder(self.device, 1, max_total),
            ));
        }
        if !matches!(&caches.decode_kv, Some((mt, _)) if *mt == max_total) {
            caches.decode_kv = Some((
                max_total,
                BucketedCompileCache::power_of_two_ladder(self.device, 1, max_total),
            ));
        }
    }

    /// Feed `n` embed rows `[n * hidden]` into the KV cache.
    /// Intermediate rows update KV only; logits come from the last row.
    pub fn continue_embeds(
        &self,
        embeds: &[f32],
        n: usize,
        kv: &mut KvCacheState,
        max_total: u64,
    ) -> Result<Vec<f32>> {
        ensure!(
            embeds.len() == n * self.hidden,
            "embeds len {} != n*hidden {}",
            embeds.len(),
            n * self.hidden
        );
        ensure!(n > 0, "continue_embeds requires n > 0");
        for i in 0..n.saturating_sub(1) {
            let row = &embeds[i * self.hidden..(i + 1) * self.hidden];
            self.decode_step_embed_inner(row, kv, max_total, /*with_logits*/ false)?;
        }
        let last = &embeds[(n - 1) * self.hidden..n * self.hidden];
        self.decode_step_embed_inner(last, kv, max_total, /*with_logits*/ true)
    }

    /// One embeds decode step with logits (text generation).
    pub fn decode_step_embed(
        &self,
        embed: &[f32],
        kv: &mut KvCacheState,
        max_total: u64,
    ) -> Result<Vec<f32>> {
        self.decode_step_embed_inner(embed, kv, max_total, true)
    }

    fn decode_step_embed_inner(
        &self,
        embed: &[f32],
        kv: &mut KvCacheState,
        max_total: u64,
        with_logits: bool,
    ) -> Result<Vec<f32>> {
        ensure!(
            embed.len() == self.hidden,
            "embed len {} != hidden {}",
            embed.len(),
            self.hidden
        );
        let batch = 1;
        let skip_fusion = self.skip_fusion();
        let cfg = &self.cfg;
        let layers = cfg.num_hidden_layers;
        let kv_dim = cfg.kv_proj_dim();
        let past_seq = kv.past_len;

        let mut caches = self.caches.borrow_mut();
        let slot = if with_logits {
            &mut caches.decode
        } else {
            &mut caches.decode_kv
        };
        if !matches!(slot, Some((mt, _)) if *mt == max_total) {
            *slot = Some((
                max_total,
                BucketedCompileCache::power_of_two_ladder(self.device, 1, max_total),
            ));
        }
        let decode_cache = &mut slot.as_mut().expect("decode cache").1;
        let mut decode_profile = CompileProfile::llama32_decode();
        if skip_fusion {
            decode_profile.fusion.skip = true;
        }
        let options = compile_options_from_profile(
            &decode_profile,
            self.device,
            KernelDispatchConfig::default(),
        );

        let upper = decode_cache
            .bucket_for(past_seq as u64)
            .and_then(|idx| {
                decode_cache
                    .buckets()
                    .nth(idx)
                    .map(|r| (r.end - 1) as usize)
            })
            .unwrap_or(past_seq);
        let (cos, sin) = rope_slice(cfg, past_seq);
        let mask = bucket_decode_mask(past_seq, upper);
        let fixed = [
            CacheRunInput {
                name: "inputs_embeds",
                data: embed,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_cos",
                data: &cos,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_sin",
                data: &sin,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: &mask,
                row_inner: None,
            },
        ];

        let snap = self.lm_snap.clone();
        let cfg_c = cfg.clone();
        let (logits, new_k, new_v) = run_bucketed_kv_decode(
            decode_cache,
            past_seq,
            kv,
            kv_dim,
            layers,
            &fixed,
            |upper_u64| {
                let mut loader = SnapshotLmLoader { snap: snap.clone() };
                let built = build_stream_decode_embeds_built(
                    &cfg_c,
                    &mut loader,
                    batch,
                    upper_u64 as usize,
                    skip_fusion,
                    with_logits,
                )
                .expect("build decode embeds");
                graph_from_built(built).expect("lower decode embeds")
            },
            &options,
        )?;
        *kv = KvCacheState {
            past_len: past_seq + 1,
            layers_kv_base: vec![0; new_k.len()],
            layers_k: new_k,
            layers_v: new_v,
        };
        Ok(logits)
    }

    /// Greedy decode until a streaming stop token (or `max_new`).
    pub fn generate_until_chunk_end(
        &self,
        kv: &mut KvCacheState,
        first_logits: &[f32],
        max_new: usize,
        max_total: u64,
    ) -> Result<Vec<i64>> {
        let mut next = argmax(first_logits);
        let mut generated = Vec::new();
        for _ in 0..max_new {
            if STREAM_STOP.contains(&next) {
                break;
            }
            generated.push(next);
            let emb = self.embed_row(next)?;
            let logits = self.decode_step_embed(emb, kv, max_total)?;
            next = argmax(&logits);
        }
        let tce = self.embed_row(TOK_TEXT_CHUNK_END)?;
        let _ = self.decode_step_embed(tce, kv, max_total)?;
        Ok(generated)
    }
}

fn build_stream_prefill_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    skip_fusion: bool,
) -> Result<BuiltModel> {
    let mut profile = CompileProfile::llama32_prefill();
    if skip_fusion {
        profile.fusion.skip = true;
    }
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let (cos_data, sin_data) = rope_tables(cfg);
    let spec = Qwen3DecoderSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        eps,
        hidden_shape: hidden_shape.clone(),
        batch,
        seq,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
        mask: rlx_ir::op::MaskKind::Causal,
    };
    let kv_sink = SideOutputs::new();
    let mut flow = ModelFlow::new("vibeasr_stream_prefill")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape)
        .rope_tables(RopeTablesStage::param(
            cfg.max_position_embeddings,
            dh / 2,
            cos_data,
            sin_data,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);
    flow = flow.repeat_layers(cfg.num_hidden_layers, {
        let spec = spec.clone();
        let sink = kv_sink.clone();
        move |i| qwen3_prefill_layer_fused_kv(i, spec.clone(), sink.inner())
    });
    flow = flow.gather_last_token_at(batch, seq).final_norm(eps);
    let built = flow
        .raw_stage(FlowStage::LmHead(LmHeadStage::separate(
            "lm_head.weight",
            cfg.vocab_size,
            h,
        )))
        .output("logits")
        .build(&mut WeightLoaderSource(weights))?
        .with_extra_hir_outputs(kv_sink.drain());
    Ok(built)
}

fn build_stream_decode_embeds_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    skip_fusion: bool,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    let mut profile = CompileProfile::llama32_decode();
    if skip_fusion {
        profile.fusion.skip = true;
    }
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let half = dh / 2;
    let kv_dim = cfg.kv_proj_dim();
    let hidden_shape = Shape::new(&[batch, 1, h], f);
    let past_kv_shape = Shape::new(&[batch, past_seq, kv_dim], f);
    let decode_spec = Qwen3DecodeLayerSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        kv_group_size: cfg.kv_group_size(),
        eps,
        use_custom_mask: true,
        hidden_shape: hidden_shape.clone(),
        batch,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
    };
    let kv_out = SideOutputs::new();
    let name = if with_lm_head {
        "vibeasr_stream_decode_embeds"
    } else {
        "vibeasr_stream_decode_embeds_kv"
    };
    let mut flow = ModelFlow::new(name)
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape)
        .input("rope_cos", Shape::new(&[1, half], f))
        .input("rope_sin", Shape::new(&[1, half], f))
        .input("mask", Shape::new(&[batch, past_seq + 1], f));
    for layer_idx in 0..cfg.num_hidden_layers {
        flow = flow
            .input(format!("past_k_{layer_idx}"), past_kv_shape.clone())
            .input(format!("past_v_{layer_idx}"), past_kv_shape.clone());
    }
    let flow = flow
        .bind_decode_inputs(cfg.num_hidden_layers, true, true)
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh)
        .repeat_layers(cfg.num_hidden_layers, {
            let spec = decode_spec.clone();
            let sink = kv_out.clone();
            move |i| qwen3_decode_layer_fused(i, spec.clone(), sink.inner())
        })
        .final_norm(eps);
    let built = if with_lm_head {
        flow.raw_stage(FlowStage::LmHead(LmHeadStage::separate(
            "lm_head.weight",
            cfg.vocab_size,
            h,
        )))
        .output("logits")
        .build(&mut WeightLoaderSource(weights))?
        .with_extra_hir_outputs(kv_out.drain())
    } else {
        flow.output("hidden_states")
            .build(&mut WeightLoaderSource(weights))?
            .with_extra_hir_outputs(kv_out.drain())
    };
    Ok(built)
}

fn rope_tables(cfg: &Qwen3Config) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos = vec![0f32; cfg.max_position_embeddings * half];
    let mut sin = vec![0f32; cfg.max_position_embeddings * half];
    for pos in 0..cfg.max_position_embeddings {
        for i in 0..half {
            let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos[pos * half + i] = c as f32;
            sin[pos * half + i] = s as f32;
        }
    }
    (cos, sin)
}

fn rope_slice(cfg: &Qwen3Config, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for i in 0..half {
        let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
        let (s, c) = (pos as f64 * freq).sin_cos();
        cos[i] = c as f32;
        sin[i] = s as f32;
    }
    (cos, sin)
}
