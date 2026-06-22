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

#[cfg(all(feature = "tokenizer", feature = "timestamps", feature = "silero-vad"))]
use crate::audio::pcm_to_mel_sized;
use crate::audio::{MelSpectrogram, N_FRAMES, pcm_slice_to_mel, pcm_to_mel};
#[cfg(feature = "word-dtw")]
use crate::backend::align_cache_key;
use crate::backend::{
    WhisperCompileOpts, WhisperGraphCtx, decode_bucket_ladder, geometry_cache_key,
    metal_compile_guard, prefill_cache_key, whisper_decoder_device, whisper_encoder_device,
    whisper_use_gpu_kv,
};
use crate::batch::{batched_prompt_f32, replicate_encoder_for_beams};
use crate::builder::WhisperGraphOpts;
use crate::cache::{
    WhisperCrossCache, WhisperKvCache, apply_bucketed_decode_step, cross_from_outputs,
    kv_from_prefill_outputs,
};
use crate::config::WhisperConfig;
use crate::decode::{
    EOT_TOKEN, SuppressionMask, batched_logits_row_owned, beam_search_decode_kv,
    beam_search_decode_kv_batched, initial_prompt_opts, last_logits_row,
};
use crate::fused::{FusedDecoderWeights, FusedEncoderWeights};
use crate::mel::stack_mels;
#[cfg(all(feature = "tokenizer", feature = "timestamps"))]
use crate::timestamp_parse::build_transcript;
#[cfg(all(feature = "tokenizer", feature = "timestamps"))]
use crate::transcript::WhisperTranscript;
#[cfg(all(feature = "tokenizer", any(feature = "word-dtw", feature = "word-w2v")))]
use crate::transcript::{WordAlignMode, WordTiming};
use crate::vad::{VadConfig, segments_by_vad};
use crate::weights::WhisperWeightPrefix;
use anyhow::{Context, Result, bail, ensure};
use rlx_core::flow_util::{
    bucket_cache_ensure_built, compile_cache_ensure_built_with_options, graph_from_built,
};
use rlx_core::validate_standard_device;
use rlx_core::weight_map::WeightMap;
use rlx_core::{
    GpuKvBinding, cross_attn_gpu_handles_ready, install_cross_attn_gpu_handles,
    run_bucketed_kv_decode_gpu, run_bucketed_kv_decode_keyed_batched, sync_gpu_kv_to_host,
};
use rlx_ir::DType;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, CompileCache};
use rlx_runtime::{CompiledGraph, Device};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct WhisperRunnerBuilder {
    weights: Option<PathBuf>,
    config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    config: Option<WhisperConfig>,
    device: Option<Device>,
    mel_frames: usize,
    max_decode_steps: usize,
    beam_size: usize,
    language: Option<String>,
    translate: bool,
    timestamps: bool,
    activation_dtype: DType,
    use_f16_compute: bool,
    vad_config: Option<VadConfig>,
    max_region_batch: usize,
    encoder_attn_chunk: usize,
    parallel_align: bool,
    no_pad: bool,
}

impl Default for WhisperRunnerBuilder {
    fn default() -> Self {
        Self {
            weights: None,
            config_path: None,
            tokenizer_path: None,
            config: None,
            device: None,
            mel_frames: 0,
            max_decode_steps: 0,
            beam_size: 0,
            language: None,
            translate: false,
            timestamps: false,
            activation_dtype: DType::F32,
            use_f16_compute: false,
            vad_config: None,
            max_region_batch: 10,
            encoder_attn_chunk: crate::builder::DEFAULT_ENCODER_ATTN_CHUNK,
            parallel_align: true,
            no_pad: false,
        }
    }
}

impl WhisperRunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.weights = Some(path.into());
        self
    }
    pub fn config_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config_path = Some(path.into());
        self
    }
    pub fn tokenizer_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }
    pub fn config(mut self, cfg: WhisperConfig) -> Self {
        self.config = Some(cfg);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }
    pub fn translate(mut self, on: bool) -> Self {
        self.translate = on;
        self
    }
    pub fn timestamps(mut self, on: bool) -> Self {
        self.timestamps = on;
        self
    }
    pub fn activation_dtype(mut self, dt: DType) -> Self {
        self.activation_dtype = dt;
        self
    }
    pub fn use_f16_compute(mut self, on: bool) -> Self {
        self.use_f16_compute = on;
        self
    }
    pub fn vad_config(mut self, cfg: VadConfig) -> Self {
        self.vad_config = Some(cfg);
        self
    }
    pub fn max_region_batch(mut self, n: usize) -> Self {
        self.max_region_batch = n.max(1);
        self
    }
    /// Parallel CPU DTW alignment across transcript segments when GPU align is off.
    /// Skip the OpenAI 30 s pad/trim for the default (non-VAD) transcribe path, using
    /// variable-length mel for short clips. Faster/lighter, but diverges numerically from
    /// the reference (which always processes a full 30 s window) — opt-in, default off.
    pub fn no_pad(mut self, on: bool) -> Self {
        self.no_pad = on;
        self
    }
    pub fn parallel_align(mut self, on: bool) -> Self {
        self.parallel_align = on;
        self
    }
    pub fn encoder_attn_chunk(mut self, n: usize) -> Self {
        self.encoder_attn_chunk = n;
        self
    }
    pub fn max_decode_steps(mut self, n: usize) -> Self {
        self.max_decode_steps = n;
        self
    }
    pub fn beam_size(mut self, n: usize) -> Self {
        self.beam_size = n;
        self
    }
    pub fn mel_frames(mut self, n: usize) -> Self {
        self.mel_frames = n;
        self
    }
    /// Size encoder graphs for `pcm` (bucketed mel frames, avoids 30 s pad waste).
    pub fn mel_frames_for_pcm(mut self, pcm: &[f32]) -> Self {
        self.mel_frames = crate::mel::mel_geometry_frames_for_pcm(pcm);
        self
    }

    pub fn build(self) -> Result<WhisperRunner> {
        let weights_path = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?;
        if !weights_path.exists() {
            bail!("weights file not found: {weights_path:?}");
        }
        let weights_dir = weights_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("weights path has no parent"))?;
        let cfg_path = self
            .config_path
            .clone()
            .unwrap_or_else(|| weights_dir.join("config.json"));
        let cfg = match self.config {
            Some(c) => c,
            None => WhisperConfig::from_file(&cfg_path)
                .with_context(|| format!("reading config {cfg_path:?}"))?,
        };
        let tok_path = self
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| weights_dir.join("tokenizer.json"));
        let device = self.device.unwrap_or(Device::Cpu);
        // ANE/CoreML is outside the standard device set; permit it only when
        // this crate is built with the `coreml` feature (which links the
        // backend). Other exotic devices still fail the standard check.
        let allow_ane = cfg!(feature = "coreml") && matches!(device, Device::Ane);
        if !allow_ane {
            validate_standard_device("whisper", device)?;
        }
        let mel_frames = if self.mel_frames == 0 {
            N_FRAMES
        } else {
            self.mel_frames
        };
        let max_decode_steps = if self.max_decode_steps == 0 {
            cfg.max_target_positions.saturating_sub(8)
        } else {
            self.max_decode_steps
        };
        let wt = weights_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 weights path"))?;
        let mut weights_cache = WeightMap::snapshot_from_path(wt)?;
        // Detect prefixes from key names only — avoids cloning the full f32 snapshot.
        let pfx = WhisperWeightPrefix::detect_with(|k| weights_cache.contains_key(k));
        let fused = FusedDecoderWeights::from_checkpoint(&weights_cache, &cfg, &pfx)?;
        let fused_enc = FusedEncoderWeights::from_checkpoint(&weights_cache, &cfg, &pfx)?;
        fused.merge_into_tensors(&mut weights_cache);
        fused_enc.merge_into_tensors(&mut weights_cache);
        let mut graph_opts = if self.use_f16_compute || self.activation_dtype == DType::F16 {
            WhisperGraphOpts::f16_mixed()
        } else {
            WhisperGraphOpts::default()
        };
        if self.encoder_attn_chunk != crate::builder::DEFAULT_ENCODER_ATTN_CHUNK {
            graph_opts.encoder_attn_chunk = self.encoder_attn_chunk;
            graph_opts.cross_attn_chunk = self.encoder_attn_chunk;
        }
        let suppression = SuppressionMask::from_config(&cfg);

        let f16 = self.use_f16_compute || self.activation_dtype == DType::F16;
        let mut compile_opts = WhisperCompileOpts::new(device, f16, &weights_path);
        let encoder_device = whisper_encoder_device(device);
        let decode_device = whisper_decoder_device(device);
        let prefill_device = decode_device;
        if decode_device != device {
            let cpu_opts = WhisperCompileOpts::new(decode_device, f16, &weights_path);
            compile_opts.cross = cpu_opts.cross.clone();
            compile_opts.decode = cpu_opts.decode.clone();
            compile_opts.prefill = cpu_opts.prefill;
        }
        if encoder_device != device {
            compile_opts.encoder =
                WhisperCompileOpts::new(encoder_device, f16, &weights_path).encoder;
        }
        let align_device = match encoder_device {
            Device::Metal | Device::Cuda | Device::Gpu => encoder_device,
            _ => Device::Cpu,
        };
        if align_device != device {
            compile_opts.align = WhisperCompileOpts::new(align_device, f16, &weights_path).encoder;
        }
        let use_gpu_kv = whisper_use_gpu_kv(device, decode_device);

        let enc_seq = cfg.encoder_seq_len(mel_frames);
        let weights_cache = Arc::new(weights_cache);
        let graph_ctx = WhisperGraphCtx {
            cfg: cfg.clone(),
            pfx: pfx.clone(),
            weights: Arc::clone(&weights_cache),
            enc_seq,
            mel_frames,
            graph_opts,
            fused: Some(fused.clone()),
            fused_enc: Some(fused_enc.clone()),
        };

        let mut enc_compile_cache = CompileCache::new(encoder_device, 8);
        let mut cross_compile_cache = CompileCache::new(decode_device, 8);
        metal_compile_guard(encoder_device, || -> Result<()> {
            compile_cache_ensure_built_with_options(
                &mut enc_compile_cache,
                geometry_cache_key(0, 1),
                graph_ctx.build_encoder(1)?,
                &compile_opts.encoder,
            )?;
            Ok(())
        })?;
        metal_compile_guard(decode_device, || -> Result<()> {
            compile_cache_ensure_built_with_options(
                &mut cross_compile_cache,
                geometry_cache_key(0, 1),
                graph_ctx.build_cross(1)?,
                &compile_opts.cross,
            )?;
            Ok(())
        })?;

        let max_past = cfg.max_target_positions.max(1);
        let decode_compile_cache = decode_bucket_ladder(decode_device, max_past as u64);

        #[cfg(feature = "tokenizer")]
        let tokenizer = {
            ensure!(tok_path.exists(), "tokenizer not found: {tok_path:?}");
            Some(
                tokenizers::Tokenizer::from_file(&tok_path)
                    .map_err(|e| anyhow::anyhow!("load tokenizer {tok_path:?}: {e}"))?,
            )
        };

        let cross_input_names: Vec<String> = (0..cfg.decoder_layers)
            .flat_map(|i| [format!("cross_k_{i}"), format!("cross_v_{i}")])
            .collect();

        let model_name = crate::alignment_heads::model_nickname(&cfg, wt);

        Ok(WhisperRunner {
            graph_ctx,
            device,
            encoder_device,
            decode_device,
            prefill_device,
            align_device,
            activation_dtype: self.activation_dtype,
            suppression,
            max_decode_steps,
            beam_size: self.beam_size,
            max_region_batch: self.max_region_batch,
            parallel_align: self.parallel_align,
            no_pad: self.no_pad,
            vad_config: self.vad_config,
            compile_opts,
            use_gpu_kv,
            gpu_kv_binding: GpuKvBinding::default(),
            cross_gpu_epoch: 0,
            cross_gpu_bound_epoch: u64::MAX,
            decode_batch_tag: u64::MAX,
            enc_compile_cache,
            cross_compile_cache,
            prefill_compile_cache: CompileCache::new(prefill_device, 8),
            align_compile_cache: CompileCache::new(align_device, 8),
            decode_compile_cache,
            decode_token_f32: Vec::new(),
            decode_pos_ix: Vec::new(),
            decode_mask: Vec::new(),
            cross_input_names,
            language: self.language,
            translate: self.translate,
            timestamps: self.timestamps,
            weights_path: weights_path.clone(),
            model_name,
            #[cfg(feature = "tokenizer")]
            tokenizer,
            last_full_enc: None,
            last_full_enc_seq: 0,
            geometry_epoch: 0,
            active_mel_frames: mel_frames,
            active_enc_seq: enc_seq,
            decode_geometry_epoch: u64::MAX,
        })
    }
}

/// Stage timings from [`WhisperRunner::bench_greedy_pipeline`].
#[derive(Debug, Clone)]
pub struct WhisperBenchReport {
    pub encode_ms: f64,
    pub cross_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_steps: usize,
    pub greedy_ms: f64,
    /// Logits after prompt prefill (`[1, prompt_len, vocab]` layout).
    pub last_prefill_logits: Vec<f32>,
}

pub struct WhisperRunner {
    graph_ctx: WhisperGraphCtx,
    pub device: Device,
    /// Device used for the mel encoder (may match [`Self::device`] on GPU backends).
    encoder_device: Device,
    /// Device used for bucketed decode graphs (CPU when `device` needs host decoder).
    decode_device: Device,
    /// Device used for prompt prefill (same as [`Self::decode_device`]).
    prefill_device: Device,
    /// Device for align-hidden graphs (Metal/CUDA when encoder runs on GPU).
    // Read only by word-alignment (DTW); always constructed.
    #[allow(dead_code)]
    align_device: Device,
    pub activation_dtype: DType,
    suppression: SuppressionMask,
    max_decode_steps: usize,
    beam_size: usize,
    max_region_batch: usize,
    parallel_align: bool,
    no_pad: bool,
    vad_config: Option<VadConfig>,
    compile_opts: WhisperCompileOpts,
    use_gpu_kv: bool,
    gpu_kv_binding: GpuKvBinding,
    /// Bumped on each new cross cache; GPU cross handles rebind when epochs differ.
    cross_gpu_epoch: u64,
    cross_gpu_bound_epoch: u64,
    decode_batch_tag: u64,
    enc_compile_cache: CompileCache,
    cross_compile_cache: CompileCache,
    prefill_compile_cache: CompileCache,
    // Read only by word-alignment (DTW); always constructed.
    #[allow(dead_code)]
    align_compile_cache: CompileCache,
    decode_compile_cache: BucketedCompileCache,
    decode_token_f32: Vec<f32>,
    decode_pos_ix: Vec<f32>,
    decode_mask: Vec<f32>,
    cross_input_names: Vec<String>,
    language: Option<String>,
    translate: bool,
    timestamps: bool,
    #[allow(dead_code)]
    weights_path: PathBuf,
    // Read only by word-alignment (DTW); always constructed.
    #[allow(dead_code)]
    model_name: String,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<tokenizers::Tokenizer>,
    /// Cached full-file encoder output for word alignment (avoids re-encode).
    last_full_enc: Option<Vec<f32>>,
    last_full_enc_seq: usize,
    /// Bumped when mel / encoder-seq geometry changes (invalidates cross/prefill/decode caches).
    geometry_epoch: u64,
    active_mel_frames: usize,
    active_enc_seq: usize,
    decode_geometry_epoch: u64,
}

impl WhisperRunner {
    pub fn builder() -> WhisperRunnerBuilder {
        WhisperRunnerBuilder::default()
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.graph_ctx.cfg
    }

    /// Number of bucketed decode graphs compiled so far (bench / tuning).
    pub fn decode_buckets_compiled(&self) -> usize {
        self.decode_compile_cache.compiled_count()
    }

    fn prepare_decode_step_inputs(&mut self, tokens: &[u32], past_seq: usize, upper: usize) {
        self.decode_token_f32.clear();
        self.decode_token_f32
            .extend(tokens.iter().map(|&t| t as f32));
        self.decode_pos_ix.clear();
        self.decode_pos_ix.resize(tokens.len(), past_seq as f32);
        let mask = bucket_decode_mask(past_seq, upper);
        if self.decode_mask.len() != mask.len() {
            self.decode_mask = mask;
        } else {
            self.decode_mask.copy_from_slice(&mask);
        }
    }

    pub fn mel_frames(&self) -> usize {
        self.active_mel_frames
    }

    pub fn enc_seq(&self) -> usize {
        self.active_enc_seq
    }

    /// Device that runs bucketed decode graphs (may differ from [`Self::device`] on Metal/MLX).
    pub fn decode_device(&self) -> Device {
        self.decode_device
    }

    /// Device that runs the mel encoder graph.
    pub fn encoder_device(&self) -> Device {
        self.encoder_device
    }

    /// Device that runs cross, prefill, and decode graphs.
    pub fn stage_device(&self) -> Device {
        self.decode_device
    }

    pub fn uses_gpu_kv(&self) -> bool {
        self.use_gpu_kv
    }

    pub fn parallel_align(&self) -> bool {
        self.parallel_align
    }

    pub fn set_parallel_align(&mut self, on: bool) {
        self.parallel_align = on;
    }

    pub fn set_max_region_batch(&mut self, n: usize) {
        self.max_region_batch = n.max(1);
    }

    fn prepare_geometry(&mut self, mel_frames: usize) -> Result<()> {
        let enc_seq = self.graph_ctx.cfg.encoder_seq_len(mel_frames);
        if self.active_mel_frames == mel_frames && self.active_enc_seq == enc_seq {
            return Ok(());
        }
        self.active_mel_frames = mel_frames;
        self.active_enc_seq = enc_seq;
        self.geometry_epoch = self.geometry_epoch.saturating_add(1);
        self.gpu_kv_binding = GpuKvBinding::default();
        self.cross_gpu_bound_epoch = u64::MAX;
        Ok(())
    }

    /// Bucket mel frames for `pcm` and invalidate compile caches when geometry changes.
    pub fn prepare_pcm_geometry(&mut self, pcm: &[f32]) -> Result<()> {
        let mel_frames = crate::mel::mel_geometry_frames_for_pcm(pcm);
        self.prepare_geometry(mel_frames)
    }

    fn ensure_encoder(&mut self, batch: usize, mel_frames: usize) -> Result<()> {
        self.prepare_geometry(mel_frames)?;
        let key = geometry_cache_key(self.geometry_epoch, batch);
        if self.enc_compile_cache.contains(key) {
            return Ok(());
        }
        let built = self.graph_ctx.build_encoder_sized(batch, mel_frames)?;
        let opts = self.compile_opts.encoder.clone();
        metal_compile_guard(self.encoder_device, || -> Result<()> {
            compile_cache_ensure_built_with_options(
                &mut self.enc_compile_cache,
                key,
                built,
                &opts,
            )?;
            Ok(())
        })
    }

    fn bind_cross_gpu_if_needed(
        compiled: &mut CompiledGraph,
        cross: &WhisperCrossCache,
        enc_seq: usize,
        d_model: usize,
        n_layers: usize,
        epoch: u64,
        bound_epoch: u64,
        use_gpu: bool,
    ) -> Result<bool> {
        if !use_gpu {
            return Ok(false);
        }
        if epoch == bound_epoch && cross_attn_gpu_handles_ready(compiled) {
            return Ok(true);
        }
        install_cross_attn_gpu_handles(compiled, cross, enc_seq, d_model, n_layers)?;
        Ok(true)
    }

    fn ensure_cross(&mut self, batch: usize) -> Result<()> {
        let key = geometry_cache_key(self.geometry_epoch, batch);
        if self.cross_compile_cache.contains(key) {
            return Ok(());
        }
        let built = self
            .graph_ctx
            .build_cross_sized(batch, self.active_enc_seq)?;
        let opts = self.compile_opts.cross.clone();
        metal_compile_guard(self.decode_device, || -> Result<()> {
            compile_cache_ensure_built_with_options(
                &mut self.cross_compile_cache,
                key,
                built,
                &opts,
            )?;
            Ok(())
        })
    }

    #[cfg(feature = "word-dtw")]
    fn ensure_align_hidden(
        &mut self,
        dec_seq: usize,
        enc_seq: usize,
        max_layer: usize,
    ) -> Result<()> {
        let mel_frames = crate::mel::mel_frames_for_enc_seq(enc_seq);
        self.prepare_geometry(mel_frames)?;
        let key = align_cache_key(self.geometry_epoch, dec_seq, max_layer);
        if self.align_compile_cache.contains(key) {
            return Ok(());
        }
        let built = self
            .graph_ctx
            .build_align_hidden_sized(1, dec_seq, enc_seq, max_layer)?;
        let opts = self.compile_opts.align.clone();
        metal_compile_guard(self.align_device, || -> Result<()> {
            compile_cache_ensure_built_with_options(
                &mut self.align_compile_cache,
                key,
                built,
                &opts,
            )?;
            Ok(())
        })
    }

    #[cfg(feature = "word-dtw")]
    fn run_align_hidden(
        &mut self,
        cross: &WhisperCrossCache,
        token_ids: &[u32],
        enc_seq: usize,
        max_layer: usize,
    ) -> Result<Vec<f32>> {
        let dec_seq = token_ids.len();
        self.ensure_align_hidden(dec_seq, enc_seq, max_layer)?;
        self.ensure_cross(1)?;
        let key = align_cache_key(self.geometry_epoch, dec_seq, max_layer);
        let token_f32: Vec<f32> = token_ids.iter().map(|&t| t as f32).collect();
        let mut inputs: Vec<(&str, &[f32])> = vec![("token_ids", &token_f32)];
        for i in 0..self.graph_ctx.cfg.decoder_layers {
            inputs.push((
                self.cross_input_names[2 * i].as_str(),
                cross.layers_k[i].as_slice(),
            ));
            inputs.push((
                self.cross_input_names[2 * i + 1].as_str(),
                cross.layers_v[i].as_slice(),
            ));
        }
        metal_compile_guard(self.align_device, || {
            self.align_compile_cache
                .get_or_compile(key, || panic!("align cache missing"))
                .run(&inputs)
        })
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("align hidden produced no output"))
    }

    pub fn encode_mel(&mut self, mel: &MelSpectrogram) -> Result<Vec<f32>> {
        self.encode_mel_inner(mel, true)
    }

    fn encode_mel_inner(&mut self, mel: &MelSpectrogram, cache_full: bool) -> Result<Vec<f32>> {
        self.prepare_geometry(mel.n_frames)?;
        self.ensure_encoder(1, mel.n_frames)?;
        let key = geometry_cache_key(self.geometry_epoch, 1);
        let enc = metal_compile_guard(self.encoder_device, || {
            self.enc_compile_cache
                .get_or_compile(key, || panic!("encoder cache missing"))
                .run(&[("mel", &mel.data)])
        })
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("encoder produced no output"))?;
        if cache_full {
            self.last_full_enc = Some(enc.clone());
            self.last_full_enc_seq = self.active_enc_seq;
        }
        Ok(enc)
    }

    pub fn encode_pcm(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        // Default: OpenAI 30 s pad/trim (reference-identical). `no_pad` uses variable-length
        // mel for short clips — faster/lighter but numerically divergent (opt-in).
        let mel = if self.no_pad {
            pcm_slice_to_mel(&self.graph_ctx.cfg, samples)
        } else {
            pcm_to_mel(&self.graph_ctx.cfg, samples)
        };
        self.encode_mel(&mel)
    }

    pub fn encode_wav(&mut self, path: &Path) -> Result<Vec<f32>> {
        let samples = crate::audio::load_wav_mono_f32(path)?;
        self.encode_pcm(&samples)
    }

    fn cross_cache(&mut self, enc: &[f32]) -> Result<WhisperCrossCache> {
        self.ensure_cross(1)?;
        let key = geometry_cache_key(self.geometry_epoch, 1);
        let outs = metal_compile_guard(self.decode_device, || {
            self.cross_compile_cache
                .get_or_compile(key, || panic!("cross cache missing"))
                .run(&[("encoder_hidden", enc)])
        });
        let cross = cross_from_outputs(
            self.graph_ctx.cfg.decoder_layers,
            1,
            self.active_enc_seq,
            self.graph_ctx.cfg.d_model,
            &outs,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        self.cross_gpu_epoch = self.cross_gpu_epoch.saturating_add(1);
        Ok(cross)
    }

    pub fn prefill_prompt(
        &mut self,
        cross: &WhisperCrossCache,
        prompt_tokens: &[u32],
        batch: usize,
    ) -> Result<(Vec<f32>, WhisperKvCache)> {
        let dec_seq = prompt_tokens.len();
        let key = prefill_cache_key(self.geometry_epoch, batch, dec_seq);

        metal_compile_guard(self.prefill_device, || {
            compile_cache_ensure_built_with_options(
                &mut self.prefill_compile_cache,
                key,
                self.graph_ctx
                    .build_prefill_sized(batch, dec_seq, self.active_enc_seq)?,
                &self.compile_opts.prefill,
            )
        })?;
        let token_f32 = if batch == 1 {
            prompt_tokens.iter().map(|&t| t as f32).collect()
        } else {
            batched_prompt_f32(prompt_tokens, batch)
        };
        let enc_seq = self.active_enc_seq;
        let d_model = self.graph_ctx.cfg.d_model;
        let n_layers = self.graph_ctx.cfg.decoder_layers;
        let epoch = self.cross_gpu_epoch;
        let bound_epoch = self.cross_gpu_bound_epoch;
        let use_gpu = self.use_gpu_kv;
        let mut cross_on_gpu = use_gpu && bound_epoch == epoch;
        let cross_bound = {
            let prefill = self
                .prefill_compile_cache
                .get_or_compile(key, || panic!("prefill cache missing"));
            Self::bind_cross_gpu_if_needed(
                prefill,
                cross,
                enc_seq,
                d_model,
                n_layers,
                epoch,
                bound_epoch,
                use_gpu,
            )?
        };
        if cross_bound {
            self.cross_gpu_bound_epoch = epoch;
            cross_on_gpu = true;
        }
        let prefill = self
            .prefill_compile_cache
            .get_or_compile(key, || panic!("prefill cache missing"));
        let mut inputs: Vec<(&str, &[f32])> = vec![("token_ids", &token_f32)];
        if !cross_on_gpu {
            for i in 0..self.graph_ctx.cfg.decoder_layers {
                inputs.push((
                    self.cross_input_names[2 * i].as_str(),
                    cross.layers_k[i].as_slice(),
                ));
                inputs.push((
                    self.cross_input_names[2 * i + 1].as_str(),
                    cross.layers_v[i].as_slice(),
                ));
            }
        }
        let outputs = metal_compile_guard(self.prefill_device, || prefill.run(&inputs));
        ensure!(!outputs.is_empty(), "prefill returned no outputs");
        let logits = outputs[0].clone();
        let kv = kv_from_prefill_outputs(
            self.graph_ctx.cfg.decoder_layers,
            batch,
            dec_seq,
            self.graph_ctx.cfg.d_model,
            &outputs[1..],
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        Ok((logits, kv))
    }

    fn decode_step_bucketed(
        &mut self,
        cross: &WhisperCrossCache,
        token: u32,
        cache: &mut WhisperKvCache,
        batch: usize,
    ) -> Result<Vec<f32>> {
        self.decode_step_batch(cross, std::slice::from_ref(&token), cache, batch, false)
    }

    fn decode_step_batch(
        &mut self,
        cross: &WhisperCrossCache,
        tokens: &[u32],
        cache: &mut WhisperKvCache,
        batch: usize,
        sync_kv_to_host: bool,
    ) -> Result<Vec<f32>> {
        ensure!(
            tokens.len() == batch,
            "decode_step_batch: expected {batch} tokens, got {}",
            tokens.len()
        );
        self.ensure_decode_batch(batch)?;
        let past_seq = cache.past_len;
        let bucket_key = past_seq as u64;
        if self.use_gpu_kv {
            return self.decode_step_batch_gpu(
                cross,
                tokens,
                cache,
                batch,
                bucket_key,
                past_seq,
                sync_kv_to_host,
            );
        }
        self.decode_step_batch_host(cross, tokens, cache, batch, bucket_key, past_seq)
    }

    fn decode_step_batch_gpu(
        &mut self,
        cross: &WhisperCrossCache,
        tokens: &[u32],
        cache: &mut WhisperKvCache,
        batch: usize,
        key: u64,
        past_seq: usize,
        sync_kv_to_host: bool,
    ) -> Result<Vec<f32>> {
        let graph_ctx = self.graph_ctx.clone();
        let decode_opts = self.compile_opts.decode.clone();
        let d_model = self.graph_ctx.cfg.d_model;
        let n_layers = self.graph_ctx.cfg.decoder_layers;
        let enc_seq = self.active_enc_seq;

        metal_compile_guard(self.decode_device, || {
            bucket_cache_ensure_built(
                &mut self.decode_compile_cache,
                key,
                |upper| graph_ctx.build_decode_step_sized(batch, upper as usize, enc_seq),
                &decode_opts,
            )
        })
        .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;

        let upper = self
            .decode_upper_for_key(key)
            .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;
        self.prepare_decode_step_inputs(tokens, past_seq, upper);
        let token_f32 = &self.decode_token_f32;
        let pos_ix = &self.decode_pos_ix;
        let mask = &self.decode_mask;
        let mut specs: Vec<CacheRunInput<'_>> = vec![
            CacheRunInput {
                name: "token_id",
                data: token_f32,
                row_inner: None,
            },
            CacheRunInput {
                name: "pos_ix",
                data: pos_ix,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: mask,
                row_inner: None,
            },
        ];
        let epoch = self.cross_gpu_epoch;
        let bound_epoch = self.cross_gpu_bound_epoch;
        let use_gpu = self.use_gpu_kv;
        let enc_seq = self.active_enc_seq;
        let mut cross_on_gpu = use_gpu && bound_epoch == epoch;
        if let Some(compiled) = self.decode_compile_cache.compiled_for_key_mut(key) {
            if Self::bind_cross_gpu_if_needed(
                compiled,
                cross,
                enc_seq,
                d_model,
                n_layers,
                epoch,
                bound_epoch,
                use_gpu,
            )? {
                self.cross_gpu_bound_epoch = epoch;
                cross_on_gpu = true;
            }
        }
        if !cross_on_gpu {
            for i in 0..n_layers {
                specs.push(CacheRunInput {
                    name: self.cross_input_names[2 * i].as_str(),
                    data: cross.layers_k[i].as_slice(),
                    row_inner: None,
                });
                specs.push(CacheRunInput {
                    name: self.cross_input_names[2 * i + 1].as_str(),
                    data: cross.layers_v[i].as_slice(),
                    row_inner: None,
                });
            }
        }

        let upper_u = upper as u64;
        let prev_upper = self.gpu_kv_binding.upper;
        let bucket_changed = prev_upper != 0 && prev_upper != upper_u;
        let handles_live = self
            .decode_compile_cache
            .compiled_for_key_mut(key)
            .map(|c| c.has_gpu_handle("past_k_0"))
            .unwrap_or(false);
        let refresh_kv = if self.decode_device == Device::Gpu {
            // wgpu handle feeds drift within a bucket; re-upload prefix each step.
            true
        } else {
            bucket_changed || !handles_live
        };

        let logits = metal_compile_guard(self.decode_device, || {
            run_bucketed_kv_decode_gpu(
                &mut self.decode_compile_cache,
                key,
                past_seq,
                cache,
                &mut self.gpu_kv_binding,
                d_model,
                n_layers,
                &specs,
                |upper| {
                    let built = graph_ctx
                        .build_decode_step_sized(batch, upper as usize, enc_seq)
                        .expect("whisper decode step built");
                    graph_from_built(built).expect("whisper decode step graph")
                },
                &decode_opts,
                refresh_kv,
            )
        })?;

        let force_host_kv = self.decode_device == Device::Gpu;
        let next_upper = self
            .decode_upper_for_key((past_seq + 1) as u64)
            .unwrap_or(upper);
        let leaves_bucket = next_upper != upper;

        if sync_kv_to_host || leaves_bucket || force_host_kv {
            if let Some(compiled) = self.decode_compile_cache.compiled_for_key_mut(key) {
                sync_gpu_kv_to_host(compiled, cache, d_model, n_layers)?;
            }
        }
        Ok(logits)
    }

    fn ensure_decode_batch(&mut self, batch: usize) -> Result<()> {
        let batch_tag = batch as u64;
        if self.decode_batch_tag == batch_tag && self.decode_geometry_epoch == self.geometry_epoch {
            return Ok(());
        }
        self.gpu_kv_binding = GpuKvBinding::default();
        self.decode_batch_tag = batch_tag;
        self.decode_geometry_epoch = self.geometry_epoch;
        let max_past = self.graph_ctx.cfg.max_target_positions.max(1) as u64;
        self.decode_compile_cache = decode_bucket_ladder(self.decode_device, max_past);
        Ok(())
    }

    fn decode_upper_for_key(&self, key: u64) -> Option<usize> {
        self.decode_compile_cache.bucket_for(key).and_then(|idx| {
            self.decode_compile_cache
                .buckets()
                .nth(idx)
                .map(|r| (r.end - 1) as usize)
        })
    }

    fn decode_step_batch_host(
        &mut self,
        cross: &WhisperCrossCache,
        tokens: &[u32],
        cache: &mut WhisperKvCache,
        batch: usize,
        key: u64,
        past_seq: usize,
    ) -> Result<Vec<f32>> {
        let graph_ctx = self.graph_ctx.clone();
        let d_model = self.graph_ctx.cfg.d_model;
        let n_layers = self.graph_ctx.cfg.decoder_layers;
        let upper = self
            .decode_upper_for_key(key)
            .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;
        self.prepare_decode_step_inputs(tokens, past_seq, upper);
        let token_f32 = &self.decode_token_f32;
        let pos_ix = &self.decode_pos_ix;
        let mask = &self.decode_mask;
        let mut specs: Vec<CacheRunInput<'_>> = vec![
            CacheRunInput {
                name: "token_id",
                data: token_f32,
                row_inner: None,
            },
            CacheRunInput {
                name: "pos_ix",
                data: pos_ix,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: mask,
                row_inner: None,
            },
        ];
        let epoch = self.cross_gpu_epoch;
        let bound_epoch = self.cross_gpu_bound_epoch;
        let use_gpu = self.use_gpu_kv;
        let enc_seq = self.active_enc_seq;
        let mut cross_on_gpu = use_gpu && bound_epoch == epoch;
        if let Some(compiled) = self.decode_compile_cache.compiled_for_key_mut(key) {
            if Self::bind_cross_gpu_if_needed(
                compiled,
                cross,
                enc_seq,
                d_model,
                n_layers,
                epoch,
                bound_epoch,
                use_gpu,
            )? {
                self.cross_gpu_bound_epoch = epoch;
                cross_on_gpu = true;
            }
        }
        if !cross_on_gpu {
            for i in 0..n_layers {
                specs.push(CacheRunInput {
                    name: self.cross_input_names[2 * i].as_str(),
                    data: cross.layers_k[i].as_slice(),
                    row_inner: None,
                });
                specs.push(CacheRunInput {
                    name: self.cross_input_names[2 * i + 1].as_str(),
                    data: cross.layers_v[i].as_slice(),
                    row_inner: None,
                });
            }
        }

        let (logits, new_k, new_v) = metal_compile_guard(self.decode_device, || {
            run_bucketed_kv_decode_keyed_batched(
                &mut self.decode_compile_cache,
                key,
                past_seq,
                batch,
                cache,
                d_model,
                n_layers,
                &specs,
                |upper| {
                    let built = graph_ctx
                        .build_decode_step_sized(batch, upper as usize, enc_seq)
                        .expect("whisper decode step built");
                    graph_from_built(built).expect("whisper decode step graph")
                },
                &self.compile_opts.decode,
            )
        })?;

        apply_bucketed_decode_step(cache, new_k, new_v, batch, d_model)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(logits)
    }

    /// Exchange bucketed decode compile caches (precision: share CPU-compiled graphs).
    pub fn swap_decode_cache(&mut self, other: &mut Self) {
        std::mem::swap(
            &mut self.decode_compile_cache,
            &mut other.decode_compile_cache,
        );
        std::mem::swap(&mut self.decode_batch_tag, &mut other.decode_batch_tag);
        std::mem::swap(
            &mut self.decode_geometry_epoch,
            &mut other.decode_geometry_epoch,
        );
        self.gpu_kv_binding = GpuKvBinding::default();
        other.gpu_kv_binding = GpuKvBinding::default();
    }

    /// Single greedy decode step (for cross-backend parity checks).
    pub fn decode_one_step(
        &mut self,
        cross: &WhisperCrossCache,
        token: u32,
        cache: &mut WhisperKvCache,
    ) -> Result<Vec<f32>> {
        self.decode_step_bucketed(cross, token, cache, 1)
    }

    fn decode_step(
        &mut self,
        cross: &WhisperCrossCache,
        token: u32,
        cache: &mut WhisperKvCache,
        batch: usize,
    ) -> Result<Vec<f32>> {
        self.decode_step_bucketed(cross, token, cache, batch)
    }

    pub fn encode_mel_batch(&mut self, mels: &[MelSpectrogram]) -> Result<Vec<f32>> {
        if mels.is_empty() {
            return Ok(Vec::new());
        }
        let mel_frames = mels.iter().map(|m| m.n_frames).max().unwrap();
        self.prepare_geometry(mel_frames)?;
        let batch = mels.len();
        let mel_input: Vec<f32> = if batch == 1 {
            mels[0].data.clone()
        } else {
            let padded: Vec<MelSpectrogram> = mels
                .iter()
                .map(|m| crate::mel::pad_mel_to_frames(m, mel_frames))
                .collect();
            stack_mels(&padded)
        };
        self.ensure_encoder(batch, mel_frames)?;
        let key = geometry_cache_key(self.geometry_epoch, batch);
        metal_compile_guard(self.encoder_device, || {
            self.enc_compile_cache
                .get_or_compile(key, || panic!("encoder cache missing"))
                .run(&[("mel", &mel_input)])
        })
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("encoder produced no output"))
    }

    /// Per-stage greedy-decode timings (after optional warmup). Includes last prefill logits for parity checks.
    #[cfg(feature = "tokenizer")]
    pub fn bench_greedy_pipeline(
        &mut self,
        pcm: &[f32],
        decode_steps: usize,
        warmup: usize,
    ) -> Result<(WhisperBenchReport, String)> {
        use std::time::Instant;
        let mel = pcm_to_mel(&self.graph_ctx.cfg, pcm);
        for _ in 0..warmup {
            let enc = self.encode_mel(&mel)?;
            self.bench_greedy_from_encoder(&enc, decode_steps.min(2))?;
        }
        let t_enc = Instant::now();
        let enc = self.encode_mel(&mel)?;
        let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
        let (mut report, transcript) = self.bench_greedy_from_encoder(&enc, decode_steps)?;
        report.encode_ms = encode_ms;
        report.greedy_ms =
            report.encode_ms + report.cross_ms + report.prefill_ms + report.decode_ms;
        Ok((report, transcript))
    }

    /// Greedy decode benchmark from a fixed encoder output (cross-backend precision: share CPU `enc`).
    #[cfg(feature = "tokenizer")]
    pub fn bench_greedy_from_encoder(
        &mut self,
        enc: &[f32],
        decode_steps: usize,
    ) -> Result<(WhisperBenchReport, String)> {
        use std::time::Instant;
        let t_cross = Instant::now();
        let cross = self.cross_cache_batch(enc, 1)?;
        let cross_ms = t_cross.elapsed().as_secs_f64() * 1000.0;
        let (mut report, transcript) = self.bench_greedy_from_cross(&cross, decode_steps)?;
        report.cross_ms = cross_ms;
        report.greedy_ms =
            report.encode_ms + report.cross_ms + report.prefill_ms + report.decode_ms;
        Ok((report, transcript))
    }

    /// Greedy decode from a fixed cross-attention cache (share CPU cross for precision).
    #[cfg(feature = "tokenizer")]
    pub fn bench_greedy_from_cross(
        &mut self,
        cross: &WhisperCrossCache,
        decode_steps: usize,
    ) -> Result<(WhisperBenchReport, String)> {
        use std::time::Instant;

        let prompt = self.build_prompt()?;
        let t_pre = Instant::now();
        let (prefill_logits, cache) = self.prefill_prompt(cross, &prompt, 1)?;
        let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
        let (mut report, transcript) = self.bench_greedy_decode_from_state(
            cross,
            &prompt,
            prefill_logits,
            cache,
            decode_steps,
        )?;
        report.prefill_ms = prefill_ms;
        report.greedy_ms =
            report.encode_ms + report.cross_ms + report.prefill_ms + report.decode_ms;
        Ok((report, transcript))
    }

    /// Greedy decode from CPU prefill logits + KV (cross-backend decode parity).
    #[cfg(feature = "tokenizer")]
    pub fn bench_greedy_decode_from_state(
        &mut self,
        cross: &WhisperCrossCache,
        prompt: &[u32],
        prefill_logits: Vec<f32>,
        mut cache: WhisperKvCache,
        decode_steps: usize,
    ) -> Result<(WhisperBenchReport, String)> {
        use std::time::Instant;

        let steps = decode_steps.min(self.max_decode_steps);
        let vocab = self.graph_ctx.cfg.vocab_size;
        let eot = self.eot_id()?;
        let last_prefill_logits = prefill_logits.clone();

        let t_dec = Instant::now();
        let mut tokens = prompt.to_vec();
        let mut next_logits = last_logits_row(&prefill_logits, prompt.len(), vocab);
        let mut done_steps = 0usize;
        for (n_gen, _) in (0..steps).enumerate() {
            let mut row = next_logits;
            let next = self.suppression.argmax_next(&mut row, n_gen == 0);
            tokens.push(next);
            done_steps += 1;
            if next == eot {
                break;
            }
            let step_logits = self.decode_step(cross, next, &mut cache, 1)?;
            next_logits = if step_logits.len() == vocab {
                step_logits
            } else {
                // Bucketed decode graphs emit a single new-token row; not `past_len` rows.
                last_logits_row(&step_logits, 1, vocab)
            };
        }
        let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        let transcript = self.decode_tokens(&tokens)?;

        let report = WhisperBenchReport {
            encode_ms: 0.0,
            cross_ms: 0.0,
            prefill_ms: 0.0,
            decode_ms,
            decode_steps: done_steps,
            greedy_ms: 0.0,
            last_prefill_logits,
        };
        Ok((report, transcript))
    }

    pub fn cross_cache_batch(&mut self, enc: &[f32], batch: usize) -> Result<WhisperCrossCache> {
        self.ensure_cross(batch)?;
        let key = geometry_cache_key(self.geometry_epoch, batch);
        let outs = metal_compile_guard(self.decode_device, || {
            self.cross_compile_cache
                .get_or_compile(key, || panic!("cross cache missing"))
                .run(&[("encoder_hidden", enc)])
        });
        let cross = cross_from_outputs(
            self.graph_ctx.cfg.decoder_layers,
            batch,
            self.active_enc_seq,
            self.graph_ctx.cfg.d_model,
            &outs,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        self.cross_gpu_epoch = self.cross_gpu_epoch.saturating_add(1);
        Ok(cross)
    }

    #[cfg(feature = "tokenizer")]
    pub fn transcribe_greedy(&mut self, pcm: &[f32]) -> Result<String> {
        self.transcribe_cached(pcm, 1)
    }

    #[cfg(feature = "tokenizer")]
    pub fn transcribe_beam(&mut self, pcm: &[f32]) -> Result<String> {
        let beam = if self.beam_size == 0 {
            5
        } else {
            self.beam_size
        };
        self.transcribe_cached(pcm, beam)
    }

    #[cfg(feature = "tokenizer")]
    pub fn transcribe_with_vad(&mut self, pcm: &[f32]) -> Result<String> {
        let vad = self.vad_config.clone().unwrap_or_default();
        let regions = segments_by_vad(&vad, pcm);
        if regions.len() <= 1 {
            return self.transcribe_cached(pcm, 1);
        }
        let beam = if self.beam_size == 0 {
            1
        } else {
            self.beam_size
        };
        let texts = self.transcribe_regions_batched(pcm, &regions, beam)?;
        Ok(texts.join(" "))
    }

    #[cfg(feature = "tokenizer")]
    pub fn transcribe_regions_batched(
        &mut self,
        pcm: &[f32],
        regions: &[crate::audio::SpeechSegment],
        beam_size: usize,
    ) -> Result<Vec<String>> {
        if regions.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(regions.len());
        let prompt = self.build_prompt()?;
        for chunk in regions.chunks(self.max_region_batch) {
            let n = chunk.len();
            let mels: Vec<MelSpectrogram> = chunk
                .iter()
                .map(|seg| pcm_slice_to_mel(&self.graph_ctx.cfg, &pcm[seg.start..seg.end]))
                .collect();
            let enc_n = self.encode_mel_batch(&mels)?;
            let texts = if beam_size <= 1 {
                self.greedy_decode_batch(&enc_n, n, &prompt)?
            } else {
                self.beam_decode_batch(&enc_n, n, beam_size, &prompt)?
            };
            out.extend(texts);
        }
        Ok(out)
    }

    #[cfg(feature = "tokenizer")]
    fn greedy_decode_batch(
        &mut self,
        enc: &[f32],
        n_regions: usize,
        prompt: &[u32],
    ) -> Result<Vec<String>> {
        let cross = self.cross_cache_batch(enc, n_regions)?;
        let (prefill_logits, mut cache) = self.prefill_prompt(&cross, prompt, n_regions)?;
        let mut tokens: Vec<Vec<u32>> = (0..n_regions).map(|_| prompt.to_vec()).collect();
        let mut done = vec![false; n_regions];
        let vocab = self.graph_ctx.cfg.vocab_size;
        let eot = self.eot_id()?;
        let mut last_logits = prefill_logits;
        let mut logits_dec_seq = prompt.len();

        for _ in 0..self.max_decode_steps {
            if done.iter().all(|&d| d) {
                break;
            }
            let mut step_tokens = vec![eot; n_regions];
            for b in 0..n_regions {
                if done[b] {
                    continue;
                }
                let mut row =
                    batched_logits_row_owned(&last_logits, b, n_regions, logits_dec_seq, vocab);
                let at_begin = tokens[b].len() == prompt.len();
                step_tokens[b] = self.suppression.argmax_next(&mut row, at_begin);
            }
            let new_logits =
                self.decode_step_batch(&cross, &step_tokens, &mut cache, n_regions, false)?;
            last_logits = new_logits;
            logits_dec_seq = 1;
            for b in 0..n_regions {
                if done[b] {
                    continue;
                }
                tokens[b].push(step_tokens[b]);
                if step_tokens[b] == eot {
                    done[b] = true;
                }
            }
        }
        tokens.into_iter().map(|t| self.decode_tokens(&t)).collect()
    }

    #[cfg(feature = "tokenizer")]
    fn beam_decode_batch(
        &mut self,
        enc: &[f32],
        n_regions: usize,
        beam_size: usize,
        prompt: &[u32],
    ) -> Result<Vec<String>> {
        let plane = self.active_enc_seq * self.graph_ctx.cfg.d_model;
        let enc_rep = replicate_encoder_for_beams(enc, n_regions, beam_size, plane);
        let batch = n_regions * beam_size;
        let cross = self.cross_cache_batch(&enc_rep, batch)?;
        let (prefill_logits, cache) = self.prefill_prompt(&cross, prompt, batch)?;
        let eot = self.eot_id()?;
        let cross_ref = &cross;
        let suffixes = beam_search_decode_kv_batched(
            &prefill_logits,
            prompt.len(),
            cache,
            n_regions,
            beam_size,
            self.max_decode_steps,
            self.graph_ctx.cfg.vocab_size,
            eot,
            |tokens, cache| self.decode_step_batch(cross_ref, tokens, cache, batch, true),
        )?;
        suffixes
            .into_iter()
            .map(|suffix| {
                let mut t = prompt.to_vec();
                t.extend(suffix);
                self.decode_tokens(&t)
            })
            .collect()
    }

    #[cfg(feature = "tokenizer")]
    fn greedy_extend_after_prefill(
        &mut self,
        cross: &WhisperCrossCache,
        prompt: &[u32],
        mut cache: WhisperKvCache,
        prefill_logits: &[f32],
        max_steps: usize,
    ) -> Result<Vec<u32>> {
        let vocab = self.graph_ctx.cfg.vocab_size;
        let eot = self.eot_id()?;
        let prompt_len = prompt.len();
        let mut tokens = prompt.to_vec();
        let mut next_logits = last_logits_row(prefill_logits, prompt_len, vocab);
        for (n_gen, _) in (0..max_steps).enumerate() {
            let mut row = next_logits;
            let next = self.suppression.argmax_next(&mut row, n_gen == 0);
            tokens.push(next);
            if next == eot {
                break;
            }
            let step_logits = self.decode_step(cross, next, &mut cache, 1)?;
            next_logits = if step_logits.len() == vocab {
                step_logits
            } else {
                last_logits_row(&step_logits, 1, vocab)
            };
        }
        Ok(tokens)
    }

    fn transcribe_cross(&mut self, cross: WhisperCrossCache, beam_size: usize) -> Result<String> {
        let prompt = self.build_prompt()?;
        let cross_ref = &cross;
        if beam_size <= 1 {
            let (prefill_logits, cache) = self.prefill_prompt(cross_ref, &prompt, 1)?;
            let tokens = self.greedy_extend_after_prefill(
                cross_ref,
                &prompt,
                cache,
                &prefill_logits,
                self.max_decode_steps,
            )?;
            return self.decode_tokens(&tokens);
        }
        let (prefill_logits, base_cache) = self.prefill_prompt(cross_ref, &prompt, 1)?;
        let extra = beam_search_decode_kv(
            &prefill_logits,
            prompt.len(),
            base_cache,
            self.eot_id()?,
            beam_size,
            self.max_decode_steps,
            self.graph_ctx.cfg.vocab_size,
            |token, cache| {
                let mut branch = cache.clone();
                let logits = self.decode_step(cross_ref, token, &mut branch, 1)?;
                let mut row = last_logits_row(&logits, 1, self.graph_ctx.cfg.vocab_size);
                self.suppression.apply(&mut row);
                Ok((row, branch))
            },
        )?;
        let mut tokens = prompt;
        tokens.extend(extra);
        self.decode_tokens(&tokens)
    }

    #[cfg(feature = "tokenizer")]
    pub fn build_prompt(&self) -> Result<Vec<u32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tokenizer not loaded"))?;
        initial_prompt_opts(
            tok,
            self.language.as_deref(),
            self.translate,
            self.timestamps,
        )
    }

    #[cfg(feature = "tokenizer")]
    fn eot_id(&self) -> Result<u32> {
        self.tokenizer
            .as_ref()
            .and_then(|t| t.token_to_id(EOT_TOKEN))
            .ok_or_else(|| anyhow::anyhow!("tokenizer missing {EOT_TOKEN}"))
    }

    #[cfg(feature = "tokenizer")]
    fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tokenizer not loaded"))?;
        tok.decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("decode tokens: {e}"))
    }

    fn transcribe_cached(&mut self, pcm: &[f32], beam_size: usize) -> Result<String> {
        if self.vad_config.is_some() {
            return self.transcribe_with_vad(pcm);
        }
        let enc = self.encode_pcm(pcm)?;
        let cross = self.cross_cache(&enc)?;
        self.transcribe_cross(cross, beam_size)
    }

    pub fn vad_enabled(&self) -> bool {
        self.vad_config.is_some()
    }

    #[cfg(all(feature = "tokenizer", feature = "timestamps"))]
    pub fn transcribe_structured(
        &mut self,
        pcm: &[f32],
        beam_size: usize,
        time_offset_sec: f32,
    ) -> Result<WhisperTranscript> {
        let duration = pcm.len() as f32 / crate::audio::SAMPLE_RATE as f32;
        let prompt = self.build_prompt_timestamps()?;
        let prompt_len = prompt.len();
        let enc = self.encode_pcm(pcm)?;
        let cross = self.cross_cache(&enc)?;
        let tokens = self.transcribe_cross_tokens(cross, beam_size.max(1))?;
        let eot = self.eot_id()?;
        let tok = self.tokenizer.as_ref().unwrap();
        let transcript = build_transcript(
            tok,
            &tokens,
            prompt_len,
            time_offset_sec,
            duration,
            self.language.as_deref(),
            eot,
        )?;
        Ok(transcript)
    }

    #[cfg(all(feature = "tokenizer", feature = "timestamps"))]
    pub fn transcribe_structured_vad(
        &mut self,
        pcm: &[f32],
        beam_size: usize,
    ) -> Result<WhisperTranscript> {
        let vad = self.vad_config.clone().unwrap_or_default();
        let regions = segments_by_vad(&vad, pcm);
        let duration = pcm.len() as f32 / crate::audio::SAMPLE_RATE as f32;
        let mut all_segments: Vec<crate::transcript::TranscriptSegment> = Vec::new();
        let prompt = self.build_prompt_timestamps()?;
        let prompt_len = prompt.len();
        let eot = self.eot_id()?;
        let mut seg_id = 0u32;
        for chunk in regions.chunks(self.max_region_batch) {
            let mels: Vec<MelSpectrogram> = chunk
                .iter()
                .map(|seg| pcm_slice_to_mel(&self.graph_ctx.cfg, &pcm[seg.start..seg.end]))
                .collect();
            let enc_n = self.encode_mel_batch(&mels)?;
            let token_batches = if beam_size <= 1 {
                self.greedy_decode_batch_tokens(&enc_n, chunk.len(), &prompt)?
            } else {
                self.beam_decode_batch_tokens(&enc_n, chunk.len(), beam_size, &prompt)?
            };
            let tok = self.tokenizer.as_ref().unwrap();
            for (seg, tokens) in chunk.iter().zip(token_batches) {
                let offset = seg.start as f32 / crate::audio::SAMPLE_RATE as f32;
                let t = build_transcript(
                    tok,
                    &tokens,
                    prompt_len,
                    offset,
                    duration,
                    self.language.as_deref(),
                    eot,
                )?;
                for s in t.segments {
                    let mut s = s;
                    s.id = seg_id;
                    seg_id += 1;
                    all_segments.push(s);
                }
            }
        }
        let transcript = WhisperTranscript {
            language: self.language.clone(),
            duration,
            segments: all_segments,
        };
        Ok(transcript)
    }

    #[cfg(all(feature = "tokenizer", feature = "timestamps", feature = "silero-vad"))]
    /// Structured transcript with Silero VAD regions: batched encode, serial decode per chunk
    /// when a chunk has multiple regions (batched decode for multi-region is still experimental).
    pub fn transcribe_structured_silero(&mut self, pcm: &[f32]) -> Result<WhisperTranscript> {
        let regions = crate::silero_vad::silero_segments(pcm)?;
        if regions.is_empty() {
            return self.transcribe_structured(pcm, 1, 0.0);
        }
        // Full-file encoder cache for subsequent word alignment (one Metal encode).
        let full_enc = self.encode_pcm(pcm)?;
        let full_seq = self.last_full_enc_seq;
        let region_mel_frames = self.active_mel_frames;
        let duration = pcm.len() as f32 / crate::audio::SAMPLE_RATE as f32;
        let mut all_segments: Vec<crate::transcript::TranscriptSegment> = Vec::new();
        let prompt = self.build_prompt_timestamps()?;
        let prompt_len = prompt.len();
        let eot = self.eot_id()?;
        let mut seg_id = 0u32;
        let mut token_batches: Vec<Vec<u32>> = Vec::new();
        for chunk in regions.chunks(self.max_region_batch) {
            let mels: Vec<MelSpectrogram> = chunk
                .iter()
                .map(|seg| {
                    pcm_to_mel_sized(
                        &self.graph_ctx.cfg,
                        &pcm[seg.start..seg.end],
                        region_mel_frames,
                    )
                })
                .collect();
            let enc_n = self.encode_mel_batch(&mels)?;
            if chunk.len() == 1 {
                let batch = self.greedy_decode_batch_tokens(&enc_n, 1, &prompt)?;
                token_batches.extend(batch);
            } else {
                let plane = self.active_enc_seq * self.graph_ctx.cfg.d_model;
                for i in 0..chunk.len() {
                    let enc = &enc_n[i * plane..(i + 1) * plane];
                    let cross = self.cross_cache(enc)?;
                    token_batches.push(self.transcribe_cross_tokens(cross, 1)?);
                }
            }
        }
        let tok = self.tokenizer.as_ref().unwrap();
        for (seg, tokens) in regions.iter().zip(token_batches) {
            let offset = seg.start as f32 / crate::audio::SAMPLE_RATE as f32;
            let t = build_transcript(
                tok,
                &tokens,
                prompt_len,
                offset,
                duration,
                self.language.as_deref(),
                eot,
            )?;
            for s in t.segments {
                let mut s = s;
                s.id = seg_id;
                seg_id += 1;
                all_segments.push(s);
            }
        }
        self.last_full_enc = Some(full_enc);
        self.last_full_enc_seq = full_seq;
        Ok(WhisperTranscript {
            language: self.language.clone(),
            duration,
            segments: all_segments,
        })
    }

    #[cfg(all(
        feature = "tokenizer",
        feature = "timestamps",
        not(feature = "silero-vad")
    ))]
    pub fn transcribe_structured_silero(&mut self, pcm: &[f32]) -> Result<WhisperTranscript> {
        self.transcribe_structured(pcm, 1, 0.0)
    }

    #[cfg(all(feature = "tokenizer", any(feature = "word-dtw", feature = "word-w2v")))]
    /// Fill [`TranscriptSegment::words`] using DTW or Wav2Vec2 forced alignment.
    ///
    /// DTW on Metal/CUDA runs a compiled align-hidden graph on the encoder device, then
    /// batched cross-attention QK on CPU. Falls back to parallel CPU forward on `Cpu`.
    pub fn apply_word_alignment(
        &mut self,
        pcm: &[f32],
        transcript: &mut WhisperTranscript,
        mode: WordAlignMode,
    ) -> Result<()> {
        match mode {
            WordAlignMode::Off => {}
            #[cfg(feature = "word-dtw")]
            WordAlignMode::Dtw => {
                let heads = crate::alignment_heads::load_alignment_heads(
                    &self.graph_ctx.cfg,
                    &self.model_name,
                )?;
                let wm = self.graph_ctx.weight_map();
                let pfx = self.graph_ctx.pfx.clone();
                let cfg = self.graph_ctx.cfg.clone();
                let d = cfg.d_model;
                let sot = self.build_prompt_timestamps()?;
                let eot = self.eot_id()?;
                crate::timestamp_parse::normalize_segment_times(
                    &mut transcript.segments,
                    pcm.len() as f32 / crate::audio::SAMPLE_RATE as f32,
                );
                let (full_enc, full_seq) = match self.last_full_enc.as_ref() {
                    Some(enc) if !enc.is_empty() => (enc.clone(), self.last_full_enc_seq),
                    _ => {
                        let enc = self.encode_pcm(pcm)?;
                        (enc, self.last_full_enc_seq)
                    }
                };
                let tok = self
                    .tokenizer
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("tokenizer required"))?;
                let use_full = transcript.segments.len() == 1
                    && transcript.segments[0].start <= 0.0
                    && transcript.segments[0].end
                        >= pcm.len() as f32 / crate::audio::SAMPLE_RATE as f32 - 0.05;

                struct DtwJob {
                    text: String,
                    start: f32,
                    end: f32,
                    text_tokens: Vec<u32>,
                    enc_slice: Vec<f32>,
                    local_seq: usize,
                }

                let jobs: Vec<DtwJob> = transcript
                    .segments
                    .iter()
                    .map(|seg| {
                        let (enc_slice, local_seq) = if use_full {
                            (full_enc.clone(), full_seq)
                        } else {
                            crate::dtw::slice_encoder_hidden(
                                &full_enc, full_seq, d, seg.start, seg.end,
                            )
                        };
                        let text_tokens = tok
                            .encode(seg.text.as_str(), false)
                            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                            .get_ids()
                            .to_vec();
                        Ok(DtwJob {
                            text: seg.text.clone(),
                            start: seg.start,
                            end: seg.end,
                            text_tokens,
                            enc_slice,
                            local_seq,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                let max_layer = heads
                    .pairs
                    .iter()
                    .map(|(l, _)| *l)
                    .max()
                    .unwrap_or(cfg.decoder_layers.saturating_sub(1));
                let d = cfg.d_model;
                let head_dim = cfg.decoder_head_dim();
                let pfx_ref = &pfx;
                let align_one_cpu = |job: &DtwJob| -> Result<Vec<WordTiming>> {
                    if job.local_seq == 0 {
                        return Ok(crate::cross_attn_align::interpolate_words_in_segment(
                            &job.text, job.start, job.end,
                        ));
                    }
                    let words = crate::cross_attn_align::find_word_alignment(
                        &tok,
                        &heads,
                        &cfg,
                        &wm,
                        pfx_ref,
                        &sot,
                        &job.text_tokens,
                        eot,
                        &job.enc_slice,
                        job.local_seq,
                        job.start,
                        7,
                    )?;
                    if words.is_empty() {
                        Ok(crate::cross_attn_align::interpolate_words_in_segment(
                            &job.text, job.start, job.end,
                        ))
                    } else {
                        Ok(words)
                    }
                };

                let aligned: Vec<Vec<WordTiming>> = if self.align_device != Device::Cpu {
                    let mut out = Vec::with_capacity(jobs.len());
                    for job in &jobs {
                        if job.local_seq == 0 {
                            out.push(crate::cross_attn_align::interpolate_words_in_segment(
                                &job.text, job.start, job.end,
                            ));
                            continue;
                        }
                        let mut all_ids = sot.clone();
                        all_ids.extend_from_slice(&job.text_tokens);
                        all_ids.push(eot);
                        let cross = self.cross_cache(&job.enc_slice)?;
                        let hidden =
                            self.run_align_hidden(&cross, &all_ids, job.local_seq, max_layer)?;
                        let words = crate::cross_attn_align::find_word_alignment_from_hidden(
                            &tok,
                            &heads,
                            &wm,
                            pfx_ref,
                            &sot,
                            &job.text_tokens,
                            &hidden,
                            &job.enc_slice,
                            job.local_seq,
                            d,
                            head_dim,
                            job.start,
                            7,
                        )?;
                        if words.is_empty() {
                            out.push(crate::cross_attn_align::interpolate_words_in_segment(
                                &job.text, job.start, job.end,
                            ));
                        } else {
                            out.push(words);
                        }
                    }
                    out
                } else if self.parallel_align && jobs.len() > 1 {
                    use rayon::prelude::*;
                    jobs.par_iter()
                        .map(align_one_cpu)
                        .collect::<Result<Vec<_>>>()?
                } else {
                    jobs.iter().map(align_one_cpu).collect::<Result<Vec<_>>>()?
                };

                for (seg, words) in transcript.segments.iter_mut().zip(aligned) {
                    seg.words = words;
                }
            }
            #[cfg(not(feature = "word-dtw"))]
            WordAlignMode::Dtw => {
                anyhow::bail!("rebuild with --features word-dtw for DTW alignment");
            }
            WordAlignMode::Wav2Vec2 => {
                #[cfg(feature = "word-w2v")]
                {
                    let lang = self.language.as_deref().unwrap_or("en");
                    let mut session = rlx_wav2vec2_asr::AlignSession::new();
                    crate::forced_align::apply_forced_align(
                        mode,
                        Some(&mut session),
                        pcm,
                        &mut transcript.segments,
                        lang,
                    )?;
                }
                #[cfg(not(feature = "word-w2v"))]
                {
                    anyhow::bail!("rebuild with --features word-w2v for Wav2Vec2 alignment");
                }
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "tokenizer", feature = "timestamps"))]
    fn build_prompt_timestamps(&self) -> Result<Vec<u32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tokenizer not loaded"))?;
        initial_prompt_opts(tok, self.language.as_deref(), self.translate, true)
    }

    #[cfg(feature = "tokenizer")]
    fn transcribe_cross_tokens(
        &mut self,
        cross: WhisperCrossCache,
        beam_size: usize,
    ) -> Result<Vec<u32>> {
        #[cfg(feature = "timestamps")]
        let prompt = self.build_prompt_timestamps()?;
        #[cfg(not(feature = "timestamps"))]
        let prompt = self.build_prompt()?;
        let cross_ref = &cross;
        if beam_size <= 1 {
            let (prefill_logits, cache) = self.prefill_prompt(cross_ref, &prompt, 1)?;
            return self.greedy_extend_after_prefill(
                cross_ref,
                &prompt,
                cache,
                &prefill_logits,
                self.max_decode_steps,
            );
        }
        let (prefill_logits, base_cache) = self.prefill_prompt(cross_ref, &prompt, 1)?;
        let extra = beam_search_decode_kv(
            &prefill_logits,
            prompt.len(),
            base_cache,
            self.eot_id()?,
            beam_size,
            self.max_decode_steps,
            self.graph_ctx.cfg.vocab_size,
            |token, cache| {
                let mut branch = cache.clone();
                let logits = self.decode_step(cross_ref, token, &mut branch, 1)?;
                let mut row = last_logits_row(&logits, 1, self.graph_ctx.cfg.vocab_size);
                self.suppression.apply(&mut row);
                Ok((row, branch))
            },
        )?;
        let mut tokens = prompt;
        tokens.extend(extra);
        Ok(tokens)
    }

    #[cfg(feature = "tokenizer")]
    fn greedy_decode_batch_tokens(
        &mut self,
        enc: &[f32],
        n_regions: usize,
        prompt: &[u32],
    ) -> Result<Vec<Vec<u32>>> {
        let cross = self.cross_cache_batch(enc, n_regions)?;
        let (prefill_logits, mut cache) = self.prefill_prompt(&cross, prompt, n_regions)?;
        let mut tokens: Vec<Vec<u32>> = (0..n_regions).map(|_| prompt.to_vec()).collect();
        let mut done = vec![false; n_regions];
        let vocab = self.graph_ctx.cfg.vocab_size;
        let eot = self.eot_id()?;
        let mut last_logits = prefill_logits;
        let mut logits_dec_seq = prompt.len();
        for _ in 0..self.max_decode_steps {
            if done.iter().all(|&d| d) {
                break;
            }
            let mut step_tokens = vec![eot; n_regions];
            for b in 0..n_regions {
                if done[b] {
                    continue;
                }
                let mut row =
                    batched_logits_row_owned(&last_logits, b, n_regions, logits_dec_seq, vocab);
                let at_begin = tokens[b].len() == prompt.len();
                step_tokens[b] = self.suppression.argmax_next(&mut row, at_begin);
            }
            let new_logits =
                self.decode_step_batch(&cross, &step_tokens, &mut cache, n_regions, false)?;
            last_logits = new_logits;
            logits_dec_seq = 1;
            for b in 0..n_regions {
                if done[b] {
                    continue;
                }
                tokens[b].push(step_tokens[b]);
                if step_tokens[b] == eot {
                    done[b] = true;
                }
            }
        }
        Ok(tokens)
    }

    #[cfg(feature = "tokenizer")]
    fn beam_decode_batch_tokens(
        &mut self,
        enc: &[f32],
        n_regions: usize,
        beam_size: usize,
        prompt: &[u32],
    ) -> Result<Vec<Vec<u32>>> {
        let plane = self.active_enc_seq * self.graph_ctx.cfg.d_model;
        let enc_rep = replicate_encoder_for_beams(enc, n_regions, beam_size, plane);
        let batch = n_regions * beam_size;
        let cross = self.cross_cache_batch(&enc_rep, batch)?;
        let (prefill_logits, cache) = self.prefill_prompt(&cross, prompt, batch)?;
        let eot = self.eot_id()?;
        let cross_ref = &cross;
        let suffixes = beam_search_decode_kv_batched(
            &prefill_logits,
            prompt.len(),
            cache,
            n_regions,
            beam_size,
            self.max_decode_steps,
            self.graph_ctx.cfg.vocab_size,
            eot,
            |tokens, cache| self.decode_step_batch(cross_ref, tokens, cache, batch, true),
        )?;
        Ok(suffixes
            .into_iter()
            .map(|suffix| {
                let mut t = prompt.to_vec();
                t.extend(suffix);
                t
            })
            .collect())
    }
}
