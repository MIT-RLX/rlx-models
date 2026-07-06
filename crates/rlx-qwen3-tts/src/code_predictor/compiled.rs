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

//! Compiled code-predictor (5-layer Qwen3, `inputs_embeds` + KV decode).

use crate::codec_frame::{Qwen3TtsGraphProfiles, Qwen3TtsGraphRole, cp_decode_graph_parts};
use crate::compile_opts::{cp_compile_device, metal_compile_guard, talker_compile_options};
use crate::config::CodePredictorConfig;
use crate::cp_frame::build_qwen3_tts_cp_prefill_two_built;
use crate::kv_util::commit_kv_layers;
use crate::load::{Qwen3TtsWeightStore, remap_code_predictor_weights};
use crate::talker::math::{
    bucket_decode_hidden_into, last_decode_hidden_into, linear_logits_into, sample_greedy,
};
use crate::talker::rope::{rope_prefill_feeds, rope_slice, rope_tables_full};
use crate::weights::weight_map_from_cache;
use anyhow::{Result, ensure};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rlx_core::autoregressive::{KvCacheState, kv_from_prefill_outputs, run_bucketed_kv_decode};
use rlx_core::flow_util::compile_cache_ensure_built_with_options;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, CompileCache};
use std::path::Path;
use std::sync::Arc;

const CP_PREFILL_SEQ: usize = 2;
const CP_DECODE_BUCKET_MAX: u64 = 32;
/// Match eager CP rope tables (`CpEagerModel` caps at 4096); HF lists 65536 but AR depth ≤ 32.
const CP_ROPE_TABLE_LEN: usize = 4096;

pub struct CpCompiledEngine {
    qwen3: rlx_qwen3::Qwen3Config,
    /// Session device (Metal/CPU/CUDA); compile caches may run on CPU when Metal is session.
    session_device: Device,
    compile_device: Device,
    hidden: usize,
    kv_dim: usize,
    n_layers: usize,
    head_half: usize,
    inv_freq: Vec<f64>,
    weights: Arc<crate::load::TensorSnapshot>,
    prefill_profile: CompileProfile,
    decode_profile: CompileProfile,
    past_len: usize,
    kv: KvCacheState,
    prefill_cache: CompileCache,
    decode_cache: BucketedCompileCache,
    prefill_scratch: Vec<f32>,
    decode_embed: Vec<f32>,
    hidden_row: Vec<f32>,
    last_raw_hidden: Vec<f32>,
    logits: Vec<f32>,
    mask_buf: Vec<f32>,
}

/// Full `[max_pos, head_half]` rope tables with active prefill positions at the front (talker pattern).
fn cp_prefill_rope_feeds(
    inv_freq: &[f64],
    positions: &[usize],
    head_dim: usize,
    rope_table_len: usize,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (mut cos, mut sin) = rope_tables_full(inv_freq, rope_table_len, head_dim);
    let (seq_cos, seq_sin) = rope_prefill_feeds(inv_freq, positions, head_dim);
    for t in 0..positions.len() {
        let off = t * head_half;
        cos[off..off + head_half].copy_from_slice(&seq_cos[off..off + head_half]);
        sin[off..off + head_half].copy_from_slice(&seq_sin[off..off + head_half]);
    }
    (cos, sin)
}

fn cp_compile_guard<R, F>(session_device: Device, compile_device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if compile_device == Device::Cpu {
        f()
    } else {
        metal_compile_guard(session_device, f)
    }
}

fn bucket_decode_mask_into(past_seq: usize, upper: usize, out: &mut Vec<f32>) {
    out.resize(upper + 1, 0.0);
    for (i, slot) in out.iter_mut().enumerate().take(upper + 1) {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }
}

impl CpCompiledEngine {
    pub fn open(
        model_dir: &Path,
        store: &Qwen3TtsWeightStore,
        cp: &CodePredictorConfig,
        device: Device,
    ) -> Result<Self> {
        let mut wm = store.load_code_predictor_backbone()?;
        let weights = remap_code_predictor_weights(&mut wm)?;
        let compile_device = cp_compile_device(device);
        let profiles = Qwen3TtsGraphProfiles::for_role(
            model_dir,
            Qwen3TtsGraphRole::CodePredictor,
            compile_device,
        );
        let prefill = profiles.prefill;
        let decode = profiles.decode;
        let mut qwen3 = cp.to_qwen3_config();
        qwen3.max_position_embeddings = qwen3.max_position_embeddings.min(CP_ROPE_TABLE_LEN);
        let hidden = cp.hidden_size;
        let kv_dim = qwen3.kv_proj_dim();
        let n_layers = cp.num_hidden_layers;
        let head_half = cp.head_dim / 2;
        let inv_freq = crate::talker::rope::build_inv_freq(cp.head_dim, cp.rope_theta);
        Ok(Self {
            qwen3,
            session_device: device,
            compile_device,
            hidden,
            kv_dim,
            n_layers,
            head_half,
            inv_freq,
            weights: Arc::new(weights),
            prefill_profile: prefill,
            decode_profile: decode,
            past_len: 0,
            kv: KvCacheState {
                past_len: 0,
                layers_k: vec![Vec::new(); n_layers],
                layers_v: vec![Vec::new(); n_layers],
                layers_kv_base: vec![0; n_layers],
            },
            prefill_cache: CompileCache::new(compile_device, 4),
            decode_cache: BucketedCompileCache::power_of_two_ladder(
                compile_device,
                1,
                CP_DECODE_BUCKET_MAX,
            ),
            prefill_scratch: vec![0f32; hidden * CP_PREFILL_SEQ],
            decode_embed: vec![0f32; hidden],
            hidden_row: vec![0f32; hidden],
            last_raw_hidden: Vec::new(),
            logits: vec![0f32; cp.vocab_size],
            mask_buf: Vec::new(),
        })
    }

    #[doc(hidden)]
    pub fn last_raw_hidden(&self) -> &[f32] {
        &self.last_raw_hidden
    }

    #[doc(hidden)]
    pub fn export_kv_state(&self) -> (KvCacheState, usize) {
        (self.kv.clone(), self.past_len)
    }

    #[doc(hidden)]
    pub fn import_kv_state(&mut self, kv: KvCacheState, past_len: usize) {
        self.kv = kv;
        self.past_len = past_len;
    }

    pub fn warmup(&mut self, max_frames: usize) -> Result<()> {
        let mut embeds = Array2::<f32>::zeros((CP_PREFILL_SEQ, self.hidden));
        embeds[[0, 0]] = 1e-4;
        self.reset_kv();
        self.prefill(embeds.view())?;
        if crate::synth_opts::lazy_talk_buckets()
            && !crate::synth_opts::auto_precompile_horizon(max_frames)
        {
            let emb = vec![0f32; self.hidden];
            let _ = self.decode_step(ArrayView1::from(&emb))?;
        } else {
            self.precompile_decode_buckets()?;
        }
        Ok(())
    }

    /// Warm CP decode buckets (past ≤ 16 per frame; ladder tops at 32).
    fn precompile_decode_buckets(&mut self) -> Result<()> {
        let keys: Vec<u64> = self
            .decode_cache
            .buckets()
            .map(|r| r.end.saturating_sub(1))
            .filter(|&k| k <= CP_DECODE_BUCKET_MAX)
            .collect();
        let opts = talker_compile_options(&self.decode_profile, self.compile_device);
        for &key in &keys {
            let weights = Arc::clone(&self.weights);
            let qwen3 = self.qwen3.clone();
            let decode_profile = self.decode_profile.clone();
            cp_compile_guard(self.session_device, self.compile_device, || {
                let _ = self.decode_cache.ensure_graph_with_params(
                    key,
                    move |upper| {
                        cp_decode_graph_parts(&qwen3, weights.as_ref(), &decode_profile, upper)
                            .expect("cp decode graph")
                    },
                    &opts,
                );
            });
        }
        Ok(())
    }

    pub fn reset_kv(&mut self) {
        self.past_len = 0;
        self.kv = KvCacheState {
            past_len: 0,
            layers_k: vec![Vec::new(); self.n_layers],
            layers_v: vec![Vec::new(); self.n_layers],
            layers_kv_base: vec![0; self.n_layers],
        };
    }

    pub fn prefill(&mut self, embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (seq, h) = embeds.dim();
        ensure!(h == self.hidden, "cp embed hidden mismatch");
        ensure!(
            seq <= CP_PREFILL_SEQ,
            "cp prefill seq {seq} > {CP_PREFILL_SEQ}"
        );
        let flat: Vec<f32> = embeds.iter().copied().collect();
        let positions: Vec<usize> = (0..seq).collect();
        let rope_table_len = self.qwen3.max_position_embeddings;
        let (rope_cos, rope_sin) = cp_prefill_rope_feeds(
            &self.inv_freq,
            &positions,
            self.qwen3.head_dim,
            rope_table_len,
            self.head_half,
        );
        let opts = talker_compile_options(&self.prefill_profile, self.compile_device);
        let key = ((1u64) << 32) | (seq as u64);
        let qwen3 = self.qwen3.clone();
        let weights = Arc::clone(&self.weights);
        let profile = self.prefill_profile.clone();
        let built = {
            let mut wm = weight_map_from_cache(weights.as_ref())?;
            if seq == crate::cp_frame::CP_PREFILL_TWO {
                build_qwen3_tts_cp_prefill_two_built(
                    &qwen3,
                    &mut wm,
                    &profile,
                    Some(rope_cos),
                    Some(rope_sin),
                )?
            } else {
                crate::codec_frame::build_qwen3_tts_prefill_built(
                    &qwen3,
                    &mut wm,
                    seq,
                    &profile,
                    Some(rope_cos),
                    Some(rope_sin),
                )?
            }
        };
        let compiled = cp_compile_guard(self.session_device, self.compile_device, || {
            compile_cache_ensure_built_with_options(&mut self.prefill_cache, key, built, &opts)
        })?;
        let outputs = compiled.run(&[("inputs_embeds", flat.as_slice())]);
        let (hidden_out, kv) =
            kv_from_prefill_outputs(outputs, 1, seq, self.kv_dim, self.n_layers)?;
        self.kv = kv;
        self.past_len = seq;
        let rows = hidden_out.len() / self.hidden;
        Ok(Array2::from_shape_vec((rows, self.hidden), hidden_out)?)
    }

    pub fn decode_step(&mut self, embed: ArrayView1<f32>) -> Result<Array1<f32>> {
        ensure!(embed.len() == self.hidden);
        self.decode_embed.copy_from_slice(embed.as_slice().unwrap());
        cp_compile_guard(self.session_device, self.compile_device, || {
            self.run_decode_step_inner()
        })?;
        Ok(Array1::from_vec(self.hidden_row.clone()))
    }

    fn run_decode_step_inner(&mut self) -> Result<()> {
        let past_seq = self.past_len;
        let pos = past_seq;
        let (cos, sin) = rope_slice(&self.inv_freq, pos, self.qwen3.head_dim);
        let upper = self
            .decode_cache
            .bucket_for(past_seq as u64)
            .map(|idx| {
                self.decode_cache
                    .buckets()
                    .nth(idx)
                    .map(|r| (r.end - 1) as usize)
                    .unwrap_or(past_seq)
            })
            .unwrap_or(past_seq);
        bucket_decode_mask_into(past_seq, upper, &mut self.mask_buf);
        let fixed = [
            CacheRunInput {
                name: "inputs_embeds",
                data: self.decode_embed.as_slice(),
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
                data: self.mask_buf.as_slice(),
                row_inner: None,
            },
        ];
        let opts = talker_compile_options(&self.decode_profile, self.compile_device);
        let weights = Arc::clone(&self.weights);
        let qwen3 = self.qwen3.clone();
        let decode_profile = self.decode_profile.clone();
        let (hidden_vec, new_k, new_v) = run_bucketed_kv_decode(
            &mut self.decode_cache,
            past_seq,
            &self.kv,
            self.kv_dim,
            self.n_layers,
            &fixed,
            move |upper| {
                cp_decode_graph_parts(&qwen3, weights.as_ref(), &decode_profile, upper)
                    .expect("cp decode graph")
            },
            &opts,
        )?;
        commit_kv_layers(&mut self.kv.layers_k, &mut self.kv.layers_v, &new_k, &new_v);
        self.kv.past_len = past_seq + 1;
        self.past_len += 1;
        self.last_raw_hidden = hidden_vec.clone();
        bucket_decode_hidden_into(&hidden_vec, self.hidden, &mut self.hidden_row)?;
        Ok(())
    }

    fn prefill_stacked(&mut self, seq: usize) -> Result<()> {
        ensure!(seq <= CP_PREFILL_SEQ);
        let flat_len = seq * self.hidden;
        let flat = self.prefill_scratch[..flat_len].to_vec();
        self.run_prefill_flat(&flat, seq)
    }

    fn run_prefill_flat(&mut self, flat: &[f32], seq: usize) -> Result<()> {
        ensure!(flat.len() == seq * self.hidden);
        let positions: Vec<usize> = (0..seq).collect();
        let rope_table_len = self.qwen3.max_position_embeddings;
        let (rope_cos, rope_sin) = cp_prefill_rope_feeds(
            &self.inv_freq,
            &positions,
            self.qwen3.head_dim,
            rope_table_len,
            self.head_half,
        );
        let opts = talker_compile_options(&self.prefill_profile, self.compile_device);
        let key = ((1u64) << 32) | (seq as u64);
        let qwen3 = self.qwen3.clone();
        let weights = Arc::clone(&self.weights);
        let profile = self.prefill_profile.clone();
        let built = {
            let mut wm = weight_map_from_cache(weights.as_ref())?;
            if seq == crate::cp_frame::CP_PREFILL_TWO {
                build_qwen3_tts_cp_prefill_two_built(
                    &qwen3,
                    &mut wm,
                    &profile,
                    Some(rope_cos),
                    Some(rope_sin),
                )?
            } else {
                crate::codec_frame::build_qwen3_tts_prefill_built(
                    &qwen3,
                    &mut wm,
                    seq,
                    &profile,
                    Some(rope_cos),
                    Some(rope_sin),
                )?
            }
        };
        let compiled = cp_compile_guard(self.session_device, self.compile_device, || {
            compile_cache_ensure_built_with_options(&mut self.prefill_cache, key, built, &opts)
        })?;
        let outputs = compiled.run(&[("inputs_embeds", flat)]);
        let (hidden_out, kv) =
            kv_from_prefill_outputs(outputs, 1, seq, self.kv_dim, self.n_layers)?;
        self.kv = kv;
        self.past_len = seq;
        last_decode_hidden_into(&hidden_out, self.hidden, &mut self.hidden_row)?;
        Ok(())
    }

    pub fn predict_groups(
        &mut self,
        talker_codec: &Array2<f32>,
        group_embeds: &[Array2<f32>],
        lm_heads: &[Array2<f32>],
        talker_hidden: ArrayView1<f32>,
        group0: u32,
    ) -> Result<Vec<u32>> {
        cp_compile_guard(self.session_device, self.compile_device, || {
            self.predict_groups_inner(talker_codec, group_embeds, lm_heads, talker_hidden, group0)
        })
    }

    fn predict_groups_inner(
        &mut self,
        talker_codec: &Array2<f32>,
        group_embeds: &[Array2<f32>],
        lm_heads: &[Array2<f32>],
        talker_hidden: ArrayView1<f32>,
        group0: u32,
    ) -> Result<Vec<u32>> {
        ensure!(talker_hidden.len() == self.hidden);
        self.reset_kv();
        let h = self.hidden;
        self.prefill_scratch[..h].copy_from_slice(talker_hidden.as_slice().unwrap());
        let e0 = talker_codec.row(group0 as usize);
        self.prefill_scratch[h..h * 2].copy_from_slice(e0.as_slice().unwrap());
        self.prefill_stacked(CP_PREFILL_SEQ)?;
        let mut codes = vec![group0];
        for step in 0..lm_heads.len() {
            linear_logits_into(
                ArrayView1::from(&self.hidden_row),
                lm_heads[step].view(),
                &mut self.logits,
            )?;
            let tok = sample_greedy(&self.logits);
            codes.push(tok);
            if step + 1 < lm_heads.len() {
                let row = group_embeds[step].row(tok as usize);
                self.decode_embed.copy_from_slice(row.as_slice().unwrap());
                self.run_decode_step_inner()?;
            }
        }
        Ok(codes)
    }
}
