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

//! Native 12Hz speech tokenizer **decode** (HF `Qwen3TTSTokenizerV2Decoder`).

use super::gpu_conv::{conv_transpose1d_maybe_gpu, run_convnext_maybe_gpu};
use super::gpu_matmul::GpuMatmulCache;
use super::ops::{
    ChT, ConvWorkspace, FlatConv1d, FlatTransConv1d, causal_conv1d,
    causal_conv1d_flat_cht_maybe_gpu, causal_conv1d_flat_maybe_gpu, conv_transpose1d_flat_cht,
    linear2, rms_norm, snake_beta_cht, swiglu,
};
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, Array3, ArrayView2};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const PREFIX: &str = "decoder.";
const CODEBOOK_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub sliding_window: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
    pub layer_scale_initial_scale: f32,
}

#[derive(Clone)]
struct Codebook {
    embedding: Array2<f32>,
}

#[derive(Clone)]
struct VqLayer {
    codebook: Codebook,
}

#[derive(Clone)]
struct RvqBranch {
    #[allow(dead_code)]
    input_proj: Array3<f32>,
    output_proj: Array3<f32>,
    layers: Vec<VqLayer>,
}

struct TransformerLayer {
    wq: Array2<f32>,
    wk: Array2<f32>,
    wv: Array2<f32>,
    wo: Array2<f32>,
    attn_norm: Array1<f32>,
    ffn_norm: Array1<f32>,
    gate: Array2<f32>,
    up: Array2<f32>,
    down: Array2<f32>,
    attn_scale: Array1<f32>,
    ffn_scale: Array1<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

pub(crate) struct PreTransformer {
    pub(crate) input_proj_w: Array2<f32>,
    pub(crate) input_proj_b: Array1<f32>,
    pub(crate) output_proj_w: Array2<f32>,
    pub(crate) output_proj_b: Array1<f32>,
    pub(crate) norm_w: Array1<f32>,
    layers: Vec<TransformerLayer>,
    inv_freq: Vec<f64>,
    sliding_window: usize,
    eps: f32,
}

struct ConvNeXtBlock {
    dw_flat: FlatConv1d,
    dw_weight: Array3<f32>,
    dw_bias: Array1<f32>,
    norm_w: Array1<f32>,
    norm_b: Array1<f32>,
    pw1_w: Array2<f32>,
    pw1_b: Array1<f32>,
    pw2_w: Array2<f32>,
    pw2_b: Array1<f32>,
    gamma: Array1<f32>,
}

#[allow(dead_code)]
struct DecoderResidualUnit {
    act1_alpha: Array1<f32>,
    act1_beta: Array1<f32>,
    conv1_flat: FlatConv1d,
    conv1_weight: Array3<f32>,
    conv1_bias: Array1<f32>,
    conv1_dilation: usize,
    act2_alpha: Array1<f32>,
    act2_beta: Array1<f32>,
    conv2_flat: FlatConv1d,
    conv2_weight: Array3<f32>,
    conv2_bias: Array1<f32>,
}

struct UpsampleStage {
    up_flat: FlatTransConv1d,
    up_weight: Array3<f32>,
    up_bias: Array1<f32>,
    up_stride: usize,
    convnext: ConvNeXtBlock,
}

#[allow(dead_code)]
struct DecoderBlock {
    act_alpha: Array1<f32>,
    act_beta: Array1<f32>,
    up_flat: FlatTransConv1d,
    up_weight: Array3<f32>,
    up_bias: Array1<f32>,
    up_stride: usize,
    residuals: Vec<DecoderResidualUnit>,
}

#[allow(dead_code)]
pub struct St12HzDecoder {
    model_dir: PathBuf,
    /// Subset of decoder tensors for GPU `pre_transformer` compile (not full weight map).
    pt_weight_snapshot: HashMap<String, (Vec<f32>, Vec<usize>)>,
    cfg: DecoderConfig,
    rvq_first: RvqBranch,
    rvq_rest: RvqBranch,
    pre_conv_flat: FlatConv1d,
    pre_conv_weight: Array3<f32>,
    pre_conv_bias: Array1<f32>,
    pre_transformer: PreTransformer,
    upsample: Vec<UpsampleStage>,
    decoder_entry_flat: FlatConv1d,
    decoder_entry_weight: Array3<f32>,
    decoder_entry_bias: Array1<f32>,
    decoder_blocks: Vec<DecoderBlock>,
    final_act_alpha: Array1<f32>,
    final_act_beta: Array1<f32>,
    final_conv_flat: FlatConv1d,
    final_conv_weight: Array3<f32>,
    final_conv_bias: Array1<f32>,
    _total_upsample: usize,
    pt_gpu: Option<super::compiled_pt::PreTransformerGpu>,
    gpu_mm: Option<GpuMatmulCache>,
    conv_ws: ConvWorkspace,
    dec_a: ChT,
    dec_b: ChT,
    dec_tmp: ChT,
    warmed_codec_frames: usize,
}

impl St12HzDecoder {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let tok_dir = model_dir.join("speech_tokenizer");
        let cfg = load_decoder_config(&tok_dir)?;
        let ckpt = SafetensorsCheckpoint::open(&tok_dir)?;
        let want: std::collections::HashSet<String> = ckpt
            .keys()
            .filter(|k| k.starts_with(PREFIX))
            .map(str::to_string)
            .collect();
        let mut wm = ckpt.load_selected(&want)?;
        // Move tensors out of the WeightMap into a HashMap directly (no copy).
        // from_tensors borrows immutably, so we need owned storage anyway, but
        // we can `take` each tensor instead of cloning its Vec<f32>.
        let keys: Vec<String> = wm.keys().map(str::to_string).collect();
        let mut map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::with_capacity(keys.len());
        for key in keys {
            let (data, shape) = wm
                .take(&key)
                .with_context(|| format!("tensor {key} missing after load"))?;
            map.insert(key, (data, shape));
        }
        Self::from_tensors(model_dir, &cfg, &map)
    }

    fn from_tensors(
        model_dir: &Path,
        cfg: &DecoderConfig,
        map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<Self> {
        let rvq_first = load_rvq_branch(map, &format!("{PREFIX}quantizer.rvq_first"), 1)?;
        let rvq_rest = load_rvq_branch(
            map,
            &format!("{PREFIX}quantizer.rvq_rest"),
            cfg.num_quantizers - cfg.num_semantic_quantizers,
        )?;

        let (pre_conv_w, pre_conv_b) = take_conv(map, &format!("{PREFIX}pre_conv"))?;
        let pre_conv_flat = FlatConv1d::from_view(pre_conv_w.view(), Some(pre_conv_b.view()), 1, 1);
        let pre_transformer = load_pre_transformer(map, cfg)?;

        let mut upsample = Vec::new();
        for (i, &ratio) in cfg.upsampling_ratios.iter().enumerate() {
            let (tw, tb) = take_conv(map, &format!("{PREFIX}upsample.{i}.0"))?;
            let cn = load_convnext(map, &format!("{PREFIX}upsample.{i}.1"))?;
            let up_flat = FlatTransConv1d::from_view(tw.view(), Some(tb.view()), ratio);
            upsample.push(UpsampleStage {
                up_flat,
                up_weight: tw,
                up_bias: tb,
                up_stride: ratio,
                convnext: cn,
            });
        }

        let (decoder_entry_w, decoder_entry_b) = take_conv(map, &format!("{PREFIX}decoder.0"))?;
        let decoder_entry_flat =
            FlatConv1d::from_view(decoder_entry_w.view(), Some(decoder_entry_b.view()), 1, 1);
        let mut decoder_blocks = Vec::new();
        for i in 0..cfg.upsample_rates.len() {
            decoder_blocks.push(load_decoder_block(map, cfg, i)?);
        }
        let out_idx = cfg.upsample_rates.len() + 1;
        let final_act_alpha = take1d(map, &format!("{PREFIX}decoder.{out_idx}.alpha"))?;
        let final_act_beta = take1d(map, &format!("{PREFIX}decoder.{out_idx}.beta"))?;
        let (final_conv_w, final_conv_b) =
            take_conv(map, &format!("{PREFIX}decoder.{}", out_idx + 1))?;
        let final_conv_flat =
            FlatConv1d::from_view(final_conv_w.view(), Some(final_conv_b.view()), 1, 1);

        let total_upsample: usize = cfg
            .upsample_rates
            .iter()
            .chain(cfg.upsampling_ratios.iter())
            .product();

        let pt_weight_snapshot: HashMap<_, _> = map
            .iter()
            .filter(|(k, _)| k.starts_with("decoder.pre_transformer"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            pt_weight_snapshot,
            cfg: cfg.clone(),
            rvq_first,
            rvq_rest,
            pre_conv_flat,
            pre_conv_weight: pre_conv_w,
            pre_conv_bias: pre_conv_b,
            pre_transformer,
            upsample,
            decoder_entry_flat,
            decoder_entry_weight: decoder_entry_w,
            decoder_entry_bias: decoder_entry_b,
            decoder_blocks,
            final_act_alpha,
            final_act_beta,
            final_conv_flat,
            final_conv_weight: final_conv_w,
            final_conv_bias: final_conv_b,
            _total_upsample: total_upsample,
            pt_gpu: None,
            gpu_mm: None,
            conv_ws: ConvWorkspace::new(),
            dec_a: ChT::default(),
            dec_b: ChT::default(),
            dec_tmp: ChT::default(),
            warmed_codec_frames: 0,
        })
    }

    /// Extend GPU compile warmup when a longer utterance needs more conv shapes.
    pub fn ensure_warmup(
        &mut self,
        device: rlx_runtime::Device,
        n_codec_frames: usize,
    ) -> Result<()> {
        if n_codec_frames <= self.warmed_codec_frames {
            return Ok(());
        }
        self.warmup(device, Some(n_codec_frames))?;
        self.warmed_codec_frames = n_codec_frames;
        Ok(())
    }

    /// Warm GPU pre_transformer compile cache (no-op on CPU). Chainable after `open`.
    pub fn tap_warmup(mut self, device: rlx_runtime::Device) -> Result<Self> {
        self.warmup(device, None)?;
        Ok(self)
    }

    /// Warm GPU pre_transformer + conv matmul compile caches (no-op on CPU).
    pub fn warmup(
        &mut self,
        device: rlx_runtime::Device,
        n_codec_frames: Option<usize>,
    ) -> Result<()> {
        let n = n_codec_frames.unwrap_or(32);
        let pre_conv = self.pre_conv_shape();
        let upsample = self.upsample_shapes();
        let convnext = self.convnext_shapes();
        let decoder_entry = self.decoder_entry_shape();
        let residual_units = self.residual_unit_shapes();
        let final_conv = self.final_conv_shape();
        if GpuMatmulCache::available(device) {
            let cache = self
                .gpu_mm
                .get_or_insert_with(|| GpuMatmulCache::new(device));
            cache.warmup_for_codec_frames(
                n,
                pre_conv,
                &upsample,
                &convnext,
                decoder_entry,
                &residual_units,
                final_conv,
            )?;
        }
        if super::compiled_pt::PreTransformerGpu::available(device) {
            if self.pt_gpu.is_none() {
                self.pt_gpu = Some(super::compiled_pt::PreTransformerGpu::open(
                    &self.model_dir,
                    &self.cfg,
                    &self.pt_weight_snapshot,
                    device,
                )?);
            }
            let gpu = self.pt_gpu.as_mut().expect("pt_gpu");
            gpu.warmup(n)?;
        }
        self.warmed_codec_frames = self.warmed_codec_frames.max(n);
        Ok(())
    }

    fn pre_conv_shape(&self) -> (usize, usize, usize) {
        let (out, in_ch, k) = self.pre_conv_weight.dim();
        (in_ch, k, out)
    }

    fn upsample_shapes(&self) -> Vec<(usize, usize, usize, usize)> {
        self.upsample
            .iter()
            .map(|s| {
                let (in_ch, out_ch, k) = s.up_weight.dim();
                (in_ch, out_ch, k, s.up_stride)
            })
            .collect()
    }

    fn convnext_shapes(&self) -> Vec<(usize, usize, usize)> {
        self.upsample
            .iter()
            .map(|s| {
                let (_, _, k) = s.convnext.dw_weight.dim();
                let pw1_out = s.convnext.pw1_w.nrows();
                let pw2_out = s.convnext.pw2_w.nrows();
                (k, pw1_out, pw2_out)
            })
            .collect()
    }

    fn decoder_entry_shape(&self) -> (usize, usize, usize) {
        let (out, in_ch, k) = self.decoder_entry_weight.dim();
        (in_ch, k, out)
    }

    fn residual_unit_shapes(&self) -> Vec<(usize, usize, usize, usize, usize)> {
        let mut shapes = Vec::new();
        for block in &self.decoder_blocks {
            for ru in &block.residuals {
                let (c1_out, c1_in, c1_k) = ru.conv1_weight.dim();
                let (_, _, c2_k) = ru.conv2_weight.dim();
                shapes.push((c1_in, c1_out, c1_k, ru.conv1_dilation, c2_k));
            }
        }
        shapes
    }

    fn final_conv_shape(&self) -> (usize, usize, usize) {
        let (out, in_ch, k) = self.final_conv_weight.dim();
        (in_ch, k, out)
    }

    pub fn decode(&mut self, frames: &[Vec<u32>], device: rlx_runtime::Device) -> Result<Vec<f32>> {
        use std::time::Instant;
        let timing = crate::synth_opts::synth_timing_enabled();
        let t0 = Instant::now();

        ensure!(!frames.is_empty(), "decode: empty codec frames");
        for (i, f) in frames.iter().enumerate() {
            ensure!(
                f.len() == 16,
                "decode: frame {i} has {} codes, expected 16",
                f.len()
            );
        }
        let t = frames.len();
        let mut codes = Array2::<u32>::zeros((16, t));
        for (ti, frame) in frames.iter().enumerate() {
            for (ki, &c) in frame.iter().enumerate() {
                codes[[ki, ti]] = c;
            }
        }
        let t_q = Instant::now();
        let mut hidden = Self::quantizer_decode(
            &self.rvq_first,
            &self.rvq_rest,
            self.cfg.codebook_dim,
            codes.view(),
        )
        .context("quantizer_decode")?;
        {
            let gpu = self.gpu_mm.as_mut();
            hidden = causal_conv1d_flat_maybe_gpu(
                hidden.view(),
                &self.pre_conv_flat,
                &mut self.conv_ws,
                gpu,
            );
        }
        if timing {
            eprintln!(
                "[qwen3-tts timing] speech quantizer+pre_conv: {:.2}s",
                t_q.elapsed().as_secs_f64()
            );
        }

        let t_pt = Instant::now();
        let seq = hidden.t().to_owned();
        let seq = self
            .run_pre_transformer(seq, device)
            .context("pre_transformer")?;
        if timing {
            eprintln!(
                "[qwen3-tts timing] speech pre_transformer: {:.2}s",
                t_pt.elapsed().as_secs_f64()
            );
        }

        let t_up = Instant::now();
        let mut x = seq.t().to_owned();
        for stage in &self.upsample {
            if let Some(cache) = self.gpu_mm.as_mut() {
                x = super::gpu_conv::conv_transpose1d_gpu_flat(cache, x.view(), &stage.up_flat);
                x = super::gpu_conv::run_convnext_gpu(
                    cache,
                    &mut self.conv_ws,
                    &x,
                    &stage.convnext.dw_flat,
                    stage.convnext.norm_w.view(),
                    stage.convnext.norm_b.view(),
                    stage.convnext.pw1_w.view(),
                    stage.convnext.pw1_b.view(),
                    stage.convnext.pw2_w.view(),
                    stage.convnext.pw2_b.view(),
                    stage.convnext.gamma.view(),
                )?;
            } else {
                x = conv_transpose1d_maybe_gpu(
                    x.view(),
                    stage.up_weight.view(),
                    Some(stage.up_bias.view()),
                    stage.up_stride,
                    None,
                );
                x = run_convnext_maybe_gpu(
                    &x,
                    stage.convnext.dw_weight.view(),
                    stage.convnext.dw_bias.view(),
                    stage.convnext.norm_w.view(),
                    stage.convnext.norm_b.view(),
                    stage.convnext.pw1_w.view(),
                    stage.convnext.pw1_b.view(),
                    stage.convnext.pw2_w.view(),
                    stage.convnext.pw2_b.view(),
                    stage.convnext.gamma.view(),
                    None,
                )?;
            }
        }
        if timing {
            eprintln!(
                "[qwen3-tts timing] speech upsample+convnext: {:.2}s",
                t_up.elapsed().as_secs_f64()
            );
        }

        let t_dec = Instant::now();
        self.dec_a.adopt_from_array2(&x);
        {
            let gpu = self.gpu_mm.as_mut();
            causal_conv1d_flat_cht_maybe_gpu(
                self.dec_a.view(),
                &self.decoder_entry_flat,
                &mut self.conv_ws,
                gpu,
                &mut self.dec_b,
            );
        }
        std::mem::swap(&mut self.dec_a, &mut self.dec_b);
        let blocks = &self.decoder_blocks;
        let gpu = &mut self.gpu_mm;
        for block in blocks {
            run_decoder_block_cht(
                &mut self.dec_a,
                &mut self.dec_b,
                &mut self.dec_tmp,
                &mut self.conv_ws,
                block,
                gpu,
            );
        }
        snake_beta_cht(
            &mut self.dec_a,
            self.final_act_alpha.view(),
            self.final_act_beta.view(),
        );
        {
            let gpu = self.gpu_mm.as_mut();
            causal_conv1d_flat_cht_maybe_gpu(
                self.dec_a.view(),
                &self.final_conv_flat,
                &mut self.conv_ws,
                gpu,
                &mut self.dec_b,
            );
        }
        std::mem::swap(&mut self.dec_a, &mut self.dec_b);
        if timing {
            eprintln!(
                "[qwen3-tts timing] speech decoder conv: {:.2}s",
                t_dec.elapsed().as_secs_f64()
            );
        }

        let samples = self.dec_a.t;
        let mut pcm = Vec::with_capacity(samples);
        for ti in 0..samples {
            let v = self.dec_a.data[ti].clamp(-1.0, 1.0);
            pcm.push(v);
        }
        if timing {
            eprintln!(
                "[qwen3-tts timing] speech decode total: {:.2}s",
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(pcm)
    }

    fn quantizer_decode(
        rvq_first: &RvqBranch,
        rvq_rest: &RvqBranch,
        codebook_dim: usize,
        codes: ArrayView2<u32>,
    ) -> Result<Array2<f32>> {
        let t = codes.dim().1;
        let mut out = Array2::<f32>::zeros((codebook_dim, t));
        let vq_dim = codebook_dim / 2;
        let first = Self::rvq_branch_decode(rvq_first, codes.slice(ndarray::s![0..1, ..]), vq_dim)?;
        out += &first;
        if codes.dim().0 > 1 {
            let rest =
                Self::rvq_branch_decode(rvq_rest, codes.slice(ndarray::s![1.., ..]), vq_dim)?;
            out += &rest;
        }
        Ok(out)
    }

    fn rvq_branch_decode(
        branch: &RvqBranch,
        codes: ArrayView2<u32>,
        vq_dim: usize,
    ) -> Result<Array2<f32>> {
        let n_q = codes.dim().0;
        ensure!(n_q == branch.layers.len(), "rvq layer count mismatch");
        let t = codes.dim().1;
        let mut acc = Array2::<f32>::zeros((vq_dim, t));
        for (li, layer) in branch.layers.iter().enumerate() {
            let row = codes.row(li);
            for ti in 0..t {
                let id = row[ti] as usize;
                ensure!(
                    id < layer.codebook.embedding.nrows(),
                    "codebook id {id} oob"
                );
                for di in 0..vq_dim {
                    acc[[di, ti]] += layer.codebook.embedding[[id, di]];
                }
            }
        }
        Ok(causal_conv1d(
            acc.view(),
            branch.output_proj.view(),
            None,
            1,
            1,
        ))
    }

    fn run_pre_transformer(
        &mut self,
        h: Array2<f32>,
        device: rlx_runtime::Device,
    ) -> Result<Array2<f32>> {
        if super::compiled_pt::PreTransformerGpu::available(device) {
            if self.pt_gpu.is_none() {
                self.pt_gpu = Some(super::compiled_pt::PreTransformerGpu::open(
                    &self.model_dir,
                    &self.cfg,
                    &self.pt_weight_snapshot,
                    device,
                )?);
            }
            let gpu = self.pt_gpu.as_mut().expect("pt_gpu");
            return super::compiled_pt::run_pre_transformer_hybrid(
                &self.cfg,
                &self.pre_transformer,
                h,
                gpu,
            );
        }
        self.run_pre_transformer_cpu(h)
    }

    fn run_pre_transformer_cpu(&self, mut h: Array2<f32>) -> Result<Array2<f32>> {
        let pt = &self.pre_transformer;
        h = linear2(
            h.view(),
            pt.input_proj_w.view(),
            Some(pt.input_proj_b.view()),
        );
        let t = h.nrows();
        let (cos, sin) = rope_tables(&pt.inv_freq, t, pt.layers[0].head_dim);
        for layer in &pt.layers {
            h = transformer_layer(h.view(), layer, &cos, &sin, pt.sliding_window, pt.eps)?;
        }
        h = rms_norm(h.view(), pt.norm_w.view(), pt.eps);
        h = linear2(
            h.view(),
            pt.output_proj_w.view(),
            Some(pt.output_proj_b.view()),
        );
        Ok(h)
    }

    /// Streaming decode: a single chunk using the KV-cached pre-transformer.
    ///
    /// `frames` is the FULL cumulative codec-frame buffer (not just the new
    /// frames). The KV cache in `kv_state` is for the pre-transformer; pass
    /// the same state across all calls within an utterance. The downstream
    /// conv chain (upsample, decoder blocks, final_conv) still runs on the
    /// cumulative buffer — only the pre-transformer attention work is
    /// saved.
    ///
    /// Available with `cfg(feature = "incremental-decode")`.
    #[cfg(feature = "incremental-decode")]
    pub fn decode_with_pt_cache(
        &mut self,
        frames: &[Vec<u32>],
        _device: rlx_runtime::Device,
        kv_state: &mut PreTransformerKvState,
        pt_cache: &mut Array2<f32>,
    ) -> Result<Vec<f32>> {
        ensure!(!frames.is_empty(), "decode: empty codec frames");
        for (i, f) in frames.iter().enumerate() {
            ensure!(
                f.len() == 16,
                "decode: frame {i} has {} codes, expected 16",
                f.len()
            );
        }
        let t_total = frames.len();
        let t_past = kv_state.tokens_processed;
        let t_new = t_total.saturating_sub(t_past);

        // 1. Quantize+pre_conv on the FULL prefix (cumulative cost; small).
        let mut codes = Array2::<u32>::zeros((16, t_total));
        for (ti, frame) in frames.iter().enumerate() {
            for (ki, &c) in frame.iter().enumerate() {
                codes[[ki, ti]] = c;
            }
        }
        let mut hidden = Self::quantizer_decode(
            &self.rvq_first,
            &self.rvq_rest,
            self.cfg.codebook_dim,
            codes.view(),
        )?;
        {
            let gpu = self.gpu_mm.as_mut();
            hidden = causal_conv1d_flat_maybe_gpu(
                hidden.view(),
                &self.pre_conv_flat,
                &mut self.conv_ws,
                gpu,
            );
        }

        // 2. Pre-transformer: incremental for the new tail.
        let seq_full = hidden.t().to_owned();
        if t_new > 0 {
            let new_slice = seq_full.slice(ndarray::s![t_past..t_total, ..]).to_owned();
            let pt_out_new = self
                .pre_transformer_incremental(new_slice.view(), kv_state)
                .context("pre_transformer_incremental")?;
            // Append to the cumulative PT output cache.
            let d = pt_out_new.ncols();
            let prev_rows = pt_cache.nrows();
            let mut grown = Array2::<f32>::zeros((prev_rows + t_new, d));
            if prev_rows > 0 {
                let assign_slice = pt_cache.slice(ndarray::s![..prev_rows, ..]).to_owned();
                grown
                    .slice_mut(ndarray::s![..prev_rows, ..])
                    .assign(&assign_slice);
            }
            grown
                .slice_mut(ndarray::s![prev_rows.., ..])
                .assign(&pt_out_new);
            *pt_cache = grown;
        }
        let seq = pt_cache.clone();

        // 3. Run the downstream conv chain on the cumulative PT output —
        //    upsample/decoder/final convs all operate on full buffer.
        let mut x = seq.t().to_owned();
        for stage in &self.upsample {
            if let Some(cache) = self.gpu_mm.as_mut() {
                x = super::gpu_conv::conv_transpose1d_gpu_flat(cache, x.view(), &stage.up_flat);
                x = super::gpu_conv::run_convnext_gpu(
                    cache,
                    &mut self.conv_ws,
                    &x,
                    &stage.convnext.dw_flat,
                    stage.convnext.norm_w.view(),
                    stage.convnext.norm_b.view(),
                    stage.convnext.pw1_w.view(),
                    stage.convnext.pw1_b.view(),
                    stage.convnext.pw2_w.view(),
                    stage.convnext.pw2_b.view(),
                    stage.convnext.gamma.view(),
                )?;
            } else {
                x = conv_transpose1d_maybe_gpu(
                    x.view(),
                    stage.up_weight.view(),
                    Some(stage.up_bias.view()),
                    stage.up_stride,
                    None,
                );
                x = run_convnext_maybe_gpu(
                    &x,
                    stage.convnext.dw_weight.view(),
                    stage.convnext.dw_bias.view(),
                    stage.convnext.norm_w.view(),
                    stage.convnext.norm_b.view(),
                    stage.convnext.pw1_w.view(),
                    stage.convnext.pw1_b.view(),
                    stage.convnext.pw2_w.view(),
                    stage.convnext.pw2_b.view(),
                    stage.convnext.gamma.view(),
                    None,
                )?;
            }
        }
        self.dec_a.adopt_from_array2(&x);
        {
            let gpu = self.gpu_mm.as_mut();
            causal_conv1d_flat_cht_maybe_gpu(
                self.dec_a.view(),
                &self.decoder_entry_flat,
                &mut self.conv_ws,
                gpu,
                &mut self.dec_b,
            );
        }
        std::mem::swap(&mut self.dec_a, &mut self.dec_b);
        let blocks = &self.decoder_blocks;
        let gpu = &mut self.gpu_mm;
        for block in blocks {
            run_decoder_block_cht(
                &mut self.dec_a,
                &mut self.dec_b,
                &mut self.dec_tmp,
                &mut self.conv_ws,
                block,
                gpu,
            );
        }
        snake_beta_cht(
            &mut self.dec_a,
            self.final_act_alpha.view(),
            self.final_act_beta.view(),
        );
        {
            let gpu = self.gpu_mm.as_mut();
            causal_conv1d_flat_cht_maybe_gpu(
                self.dec_a.view(),
                &self.final_conv_flat,
                &mut self.conv_ws,
                gpu,
                &mut self.dec_b,
            );
        }
        std::mem::swap(&mut self.dec_a, &mut self.dec_b);

        let samples = self.dec_a.t;
        let mut pcm = Vec::with_capacity(samples);
        for t in 0..samples {
            pcm.push(self.dec_a.data[t]);
        }
        Ok(pcm)
    }

    /// Incremental pre-transformer for streaming decode.
    ///
    /// Processes only the NEW codec frames (`h_new` rows). Past K, V from
    /// previous calls are held in `state` per layer; the sliding-window mask
    /// covers (cached + new) tokens. KV caches trim to `sliding_window`.
    ///
    /// Available with `cfg(feature = "incremental-decode")`. Public so the
    /// streaming decoder can drive it.
    #[cfg(feature = "incremental-decode")]
    pub fn pre_transformer_incremental(
        &self,
        h_new: ArrayView2<f32>,
        state: &mut PreTransformerKvState,
    ) -> Result<Array2<f32>> {
        let pt = &self.pre_transformer;
        let n_new = h_new.nrows();
        if n_new == 0 {
            return Ok(Array2::<f32>::zeros((0, pt.output_proj_w.nrows())));
        }
        let mut h = linear2(h_new, pt.input_proj_w.view(), Some(pt.input_proj_b.view()));

        // Build RoPE tables for the new tokens' absolute positions
        // [past_tokens .. past_tokens + n_new].
        let head_dim = pt.layers[0].head_dim;
        let past_tokens = state.tokens_processed;
        let total_after = past_tokens + n_new;
        let (cos_full, sin_full) =
            rope_tables_offset(&pt.inv_freq, past_tokens, total_after, head_dim);

        if state.layers.is_empty() {
            state.layers = (0..pt.layers.len()).map(|_| PtLayerKv::default()).collect();
        }
        for (li, layer) in pt.layers.iter().enumerate() {
            h = transformer_layer_incremental(
                h.view(),
                layer,
                &cos_full,
                &sin_full,
                pt.sliding_window,
                pt.eps,
                past_tokens,
                &mut state.layers[li],
            )?;
        }
        h = rms_norm(h.view(), pt.norm_w.view(), pt.eps);
        h = linear2(
            h.view(),
            pt.output_proj_w.view(),
            Some(pt.output_proj_b.view()),
        );
        state.tokens_processed = total_after;
        Ok(h)
    }
}

/// State buffer for one causal Conv1d in the streaming decoder.
///
/// Holds the trailing `(kernel-1) * dilation` input samples from the previous
/// call so the next call can prepend them to its new input and produce
/// sample-identical outputs to a full-prefix decode.
///
/// Memory: `in_ch * taps * 4` bytes per stage. For typical sizes (taps≈7,
/// in_ch up to 1024) that's ~28 KB per stage — negligible.
#[cfg(feature = "incremental-decode")]
#[derive(Default, Clone)]
pub struct Conv1dStateBuf {
    /// Per-channel trailing samples, layout [in_ch, taps] flat.
    pub trailing: Vec<f32>,
    pub in_ch: usize,
    pub taps: usize,
}

#[cfg(feature = "incremental-decode")]
impl Conv1dStateBuf {
    pub fn taps_held(&self) -> usize {
        self.taps
    }
}

/// Apply a causal Conv1d to ONLY the new input samples, using cached trailing
/// inputs as state. Returns the output for `x_new` (same time-axis length as
/// `x_new.t`).
///
/// SAFETY/CORRECTNESS: equivalent to running the conv over the full
/// cumulative input and taking the last `x_new.t` outputs — except much
/// cheaper, because the conv runs on `(trailing.taps + x_new.t)` samples
/// instead of `(total_history)`.
#[cfg(feature = "incremental-decode")]
pub fn causal_conv1d_incremental(
    state: &mut Conv1dStateBuf,
    x_new: super::ops::ChT,
    flat: &super::ops::FlatConv1d,
    ws: &mut super::ops::ConvWorkspace,
    gpu: Option<&mut super::gpu_matmul::GpuMatmulCache>,
) -> super::ops::ChT {
    use super::ops::ChT;
    let in_ch = x_new.ch;
    let n_new = x_new.t;
    let want_taps = (flat.k - 1) * flat.dilation;

    // Initialize state on first call.
    if state.in_ch == 0 {
        state.in_ch = in_ch;
        state.taps = 0;
    }
    debug_assert_eq!(state.in_ch, in_ch);

    // Concat state.trailing + x_new along time axis into a single ChT.
    let total_t = state.taps + n_new;
    let mut concat = ChT::default();
    concat.ensure(in_ch, total_t);
    for c in 0..in_ch {
        let dst = c * total_t;
        for ti in 0..state.taps {
            concat.data[dst + ti] = state.trailing[c * state.taps + ti];
        }
        for ti in 0..n_new {
            concat.data[dst + state.taps + ti] = x_new.data[c * n_new + ti];
        }
    }

    // Run the conv on the concatenated buffer.
    let mut full_out = ChT::default();
    super::ops::causal_conv1d_flat_cht_maybe_gpu(concat.view(), flat, ws, gpu, &mut full_out);

    // Take the last `n_new` outputs. The first `state.taps` outputs match the
    // tail of the previous call and have already been emitted (the conv
    // produces `t_in` outputs for `t_in` inputs at stride=1).
    let out_ch = full_out.ch;
    let total_out_t = full_out.t;
    let take = n_new.min(total_out_t);
    let skip = total_out_t - take;
    let mut new_out = ChT::default();
    new_out.ensure(out_ch, take);
    for c in 0..out_ch {
        for ti in 0..take {
            new_out.data[c * take + ti] = full_out.data[c * total_out_t + skip + ti];
        }
    }

    // Update state: keep last `want_taps` input columns from concat.
    let keep = want_taps.min(total_t);
    let mut new_trail = vec![0f32; in_ch * keep];
    if keep > 0 {
        for c in 0..in_ch {
            for ti in 0..keep {
                let src_ti = total_t - keep + ti;
                new_trail[c * keep + ti] = concat.data[c * total_t + src_ti];
            }
        }
    }
    state.trailing = new_trail;
    state.taps = keep;

    new_out
}

/// KV cache for one transformer layer in the streaming pre-transformer.
/// `k` and `v` hold up to `sliding_window` rows each (kv_dim columns each).
#[cfg(feature = "incremental-decode")]
#[derive(Default, Clone)]
pub struct PtLayerKv {
    /// Row-major [t_cached, n_kv_heads * head_dim].
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub kv_dim: usize,
    pub cached_rows: usize,
}

/// State carried across `pre_transformer_incremental` calls.
#[cfg(feature = "incremental-decode")]
#[derive(Default)]
pub struct PreTransformerKvState {
    pub layers: Vec<PtLayerKv>,
    pub tokens_processed: usize,
}

/// Like `rope_tables` but spans positions `[start, end)` only.
#[cfg(feature = "incremental-decode")]
fn rope_tables_offset(
    inv_freq: &[f64],
    start: usize,
    end: usize,
    head_dim: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let half = head_dim / 2;
    let n = end - start;
    let mut cos = vec![vec![0f32; half]; n];
    let mut sin = vec![vec![0f32; half]; n];
    for (i, pos) in (start..end).enumerate() {
        for k in 0..half {
            let angle = pos as f64 * inv_freq[k];
            cos[i][k] = angle.cos() as f32;
            sin[i][k] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

/// Incremental transformer layer forward.
///
/// `x` is the NEW tokens (n_new × hidden). `past_tokens` is how many tokens
/// were processed in previous calls. `kv` holds the K/V cache for past
/// tokens. Returns output for the new tokens only.
#[cfg(feature = "incremental-decode")]
fn transformer_layer_incremental(
    x: ArrayView2<f32>,
    layer: &TransformerLayer,
    cos_new: &[Vec<f32>],
    sin_new: &[Vec<f32>],
    sliding_window: usize,
    eps: f32,
    past_tokens: usize,
    kv: &mut PtLayerKv,
) -> Result<Array2<f32>> {
    let (n_new, d) = x.dim();
    let residual = x.to_owned();
    let h = rms_norm(x, layer.attn_norm.view(), eps);
    let mut q = linear2(h.view(), layer.wq.view(), None);
    let mut k_new = linear2(h.view(), layer.wk.view(), None);
    let v_new = linear2(h.view(), layer.wv.view(), None);
    let q_dim = q.ncols();
    let kv_dim = k_new.ncols();
    apply_rope_matrix_offset(&mut q, layer.head_dim, cos_new, sin_new, 0);
    apply_rope_matrix_offset(&mut k_new, layer.head_dim, cos_new, sin_new, 0);

    // Append K_new / V_new to cache, then trim to sliding_window rows.
    if kv.kv_dim == 0 {
        kv.kv_dim = kv_dim;
    }
    for row in 0..n_new {
        let row_slice_k: Vec<f32> = (0..kv_dim).map(|c| k_new[[row, c]]).collect();
        let row_slice_v: Vec<f32> = (0..kv_dim).map(|c| v_new[[row, c]]).collect();
        kv.k.extend_from_slice(&row_slice_k);
        kv.v.extend_from_slice(&row_slice_v);
        kv.cached_rows += 1;
    }
    if kv.cached_rows > sliding_window {
        let drop = kv.cached_rows - sliding_window;
        kv.k.drain(..drop * kv_dim);
        kv.v.drain(..drop * kv_dim);
        kv.cached_rows = sliding_window;
    }
    let t_total = kv.cached_rows;
    let kv_window_start = past_tokens + n_new - t_total;

    let scale = 1.0 / (layer.head_dim as f32).sqrt();
    let n_heads = layer.n_heads;
    let n_kv = layer.n_kv_heads;
    let hd = layer.head_dim;
    let repeats = n_heads / n_kv;
    let mut attn_out = Array2::<f32>::zeros((n_new, q_dim));
    let mut weights = vec![f32::NEG_INFINITY; t_total];
    for hi in 0..n_heads {
        let kv_h = hi / repeats;
        for qi in 0..n_new {
            let q_abs = past_tokens + qi;
            for w in weights.iter_mut() {
                *w = f32::NEG_INFINITY;
            }
            let mut max_w = f32::NEG_INFINITY;
            for ki in 0..t_total {
                let k_abs = kv_window_start + ki;
                if k_abs > q_abs || q_abs - k_abs >= sliding_window {
                    continue;
                }
                let mut dot = 0f32;
                for di in 0..hd {
                    dot += q[[qi, hi * hd + di]] * kv.k[ki * kv.kv_dim + kv_h * hd + di];
                }
                dot *= scale;
                weights[ki] = dot;
                max_w = max_w.max(dot);
            }
            let mut sum = 0f32;
            for w in weights.iter_mut() {
                if w.is_finite() {
                    *w = (*w - max_w).exp();
                    sum += *w;
                } else {
                    *w = 0.0;
                }
            }
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
            for di in 0..hd {
                let mut acc = 0f32;
                for ki in 0..t_total {
                    acc += weights[ki] * kv.v[ki * kv.kv_dim + kv_h * hd + di];
                }
                attn_out[[qi, hi * hd + di]] = acc;
            }
        }
    }
    let mut out = linear2(attn_out.view(), layer.wo.view(), None);
    for i in 0..n_new {
        for j in 0..d {
            out[[i, j]] = residual[[i, j]] + layer.attn_scale[j] * out[[i, j]];
        }
    }
    let residual2 = out.clone();
    let h2 = rms_norm(out.view(), layer.ffn_norm.view(), eps);
    let ff = swiglu(
        linear2(h2.view(), layer.gate.view(), None).view(),
        linear2(h2.view(), layer.up.view(), None).view(),
        &layer.down,
    );
    for i in 0..n_new {
        for j in 0..d {
            out[[i, j]] = residual2[[i, j]] + layer.ffn_scale[j] * ff[[i, j]];
        }
    }
    Ok(out)
}

#[cfg(feature = "incremental-decode")]
fn apply_rope_matrix_offset(
    x: &mut Array2<f32>,
    head_dim: usize,
    cos_tbl: &[Vec<f32>],
    sin_tbl: &[Vec<f32>],
    _start_row: usize,
) {
    let (t, d) = x.dim();
    let half = head_dim / 2;
    debug_assert!(d % head_dim == 0);
    let n_heads_total = d / head_dim;
    for row in 0..t {
        let cos = &cos_tbl[row];
        let sin = &sin_tbl[row];
        for h in 0..n_heads_total {
            let base = h * head_dim;
            for k in 0..half {
                let x0 = x[[row, base + k]];
                let x1 = x[[row, base + half + k]];
                x[[row, base + k]] = x0 * cos[k] - x1 * sin[k];
                x[[row, base + half + k]] = x0 * sin[k] + x1 * cos[k];
            }
        }
    }
}

fn load_decoder_config(tok_dir: &Path) -> Result<DecoderConfig> {
    let raw =
        std::fs::read(tok_dir.join("config.json")).context("read speech_tokenizer/config.json")?;
    let v: serde_json::Value = serde_json::from_slice(&raw)?;
    let dc = v
        .get("decoder_config")
        .context("decoder_config in speech_tokenizer/config.json")?;
    Ok(serde_json::from_value(dc.clone())?)
}

fn load_rvq_branch(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    n_layers: usize,
) -> Result<RvqBranch> {
    let input_proj = take3d(map, &format!("{prefix}.input_proj.weight"))?;
    let output_proj = take3d(map, &format!("{prefix}.output_proj.weight"))?;
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let sum = take2d(
            map,
            &format!("{prefix}.vq.layers.{i}._codebook.embedding_sum"),
        )?;
        let usage = take1d(
            map,
            &format!("{prefix}.vq.layers.{i}._codebook.cluster_usage"),
        )?;
        layers.push(VqLayer {
            codebook: Codebook {
                embedding: codebook_embed(&sum, &usage),
            },
        });
    }
    Ok(RvqBranch {
        input_proj,
        output_proj,
        layers,
    })
}

fn codebook_embed(sum: &Array2<f32>, usage: &Array1<f32>) -> Array2<f32> {
    let (v, d) = sum.dim();
    let mut out = Array2::<f32>::zeros((v, d));
    for i in 0..v {
        let denom = usage[i].max(CODEBOOK_EPS);
        for j in 0..d {
            out[[i, j]] = sum[[i, j]] / denom;
        }
    }
    out
}

fn load_pre_transformer(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    cfg: &DecoderConfig,
) -> Result<PreTransformer> {
    let p = format!("{PREFIX}pre_transformer");
    let input_proj_w = take2d(map, &format!("{p}.input_proj.weight"))?;
    let input_proj_b = take1d(map, &format!("{p}.input_proj.bias"))?;
    let output_proj_w = take2d(map, &format!("{p}.output_proj.weight"))?;
    let output_proj_b = take1d(map, &format!("{p}.output_proj.bias"))?;
    let norm_w = take1d(map, &format!("{p}.norm.weight"))?;
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("{p}.layers.{i}");
        layers.push(TransformerLayer {
            wq: take2d(map, &format!("{lp}.self_attn.q_proj.weight"))?,
            wk: take2d(map, &format!("{lp}.self_attn.k_proj.weight"))?,
            wv: take2d(map, &format!("{lp}.self_attn.v_proj.weight"))?,
            wo: take2d(map, &format!("{lp}.self_attn.o_proj.weight"))?,
            attn_norm: take1d(map, &format!("{lp}.input_layernorm.weight"))?,
            ffn_norm: take1d(map, &format!("{lp}.post_attention_layernorm.weight"))?,
            gate: take2d(map, &format!("{lp}.mlp.gate_proj.weight"))?,
            up: take2d(map, &format!("{lp}.mlp.up_proj.weight"))?,
            down: take2d(map, &format!("{lp}.mlp.down_proj.weight"))?,
            attn_scale: take1d(map, &format!("{lp}.self_attn_layer_scale.scale"))?,
            ffn_scale: take1d(map, &format!("{lp}.mlp_layer_scale.scale"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        });
    }
    let half = cfg.head_dim / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / cfg.head_dim as f64))
        .collect();
    Ok(PreTransformer {
        input_proj_w,
        input_proj_b,
        output_proj_w,
        output_proj_b,
        norm_w,
        layers,
        inv_freq,
        sliding_window: cfg.sliding_window,
        eps: cfg.rms_norm_eps,
    })
}

fn load_convnext(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
) -> Result<ConvNeXtBlock> {
    let (dw_weight, dw_bias) = take_conv(map, &format!("{prefix}.dwconv"))?;
    let dw_flat = FlatConv1d::from_view(dw_weight.view(), Some(dw_bias.view()), 1, 1);
    Ok(ConvNeXtBlock {
        dw_flat,
        dw_weight,
        dw_bias,
        norm_w: take1d(map, &format!("{prefix}.norm.weight"))?,
        norm_b: take1d(map, &format!("{prefix}.norm.bias"))?,
        pw1_w: take2d(map, &format!("{prefix}.pwconv1.weight"))?,
        pw1_b: take1d(map, &format!("{prefix}.pwconv1.bias"))?,
        pw2_w: take2d(map, &format!("{prefix}.pwconv2.weight"))?,
        pw2_b: take1d(map, &format!("{prefix}.pwconv2.bias"))?,
        gamma: take1d(map, &format!("{prefix}.gamma"))?,
    })
}

fn load_decoder_block(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    cfg: &DecoderConfig,
    idx: usize,
) -> Result<DecoderBlock> {
    let p = format!("{PREFIX}decoder.{}", idx + 1);
    let rate = cfg.upsample_rates[idx];
    let (up_weight, up_bias) = take_conv(map, &format!("{p}.block.1"))?;
    let mut residuals = Vec::new();
    for (bi, dilation) in (2..=4).zip([1usize, 3, 9]) {
        let rp = format!("{p}.block.{bi}");
        let (conv1_w, conv1_b) = take_conv(map, &format!("{rp}.conv1"))?;
        let (conv2_w, conv2_b) = take_conv(map, &format!("{rp}.conv2"))?;
        residuals.push(DecoderResidualUnit {
            act1_alpha: take1d(map, &format!("{rp}.act1.alpha"))?,
            act1_beta: take1d(map, &format!("{rp}.act1.beta"))?,
            conv1_flat: FlatConv1d::from_view(conv1_w.view(), Some(conv1_b.view()), 1, dilation),
            conv1_weight: conv1_w,
            conv1_bias: conv1_b,
            conv1_dilation: dilation,
            act2_alpha: take1d(map, &format!("{rp}.act2.alpha"))?,
            act2_beta: take1d(map, &format!("{rp}.act2.beta"))?,
            conv2_flat: FlatConv1d::from_view(conv2_w.view(), Some(conv2_b.view()), 1, 1),
            conv2_weight: conv2_w,
            conv2_bias: conv2_b,
        });
    }
    Ok(DecoderBlock {
        act_alpha: take1d(map, &format!("{p}.block.0.alpha"))?,
        act_beta: take1d(map, &format!("{p}.block.0.beta"))?,
        up_flat: FlatTransConv1d::from_view(up_weight.view(), Some(up_bias.view()), rate),
        up_weight,
        up_bias,
        up_stride: rate,
        residuals,
    })
}

fn run_decoder_block_cht(
    cur: &mut ChT,
    next: &mut ChT,
    tmp: &mut ChT,
    ws: &mut ConvWorkspace,
    b: &DecoderBlock,
    gpu: &mut Option<GpuMatmulCache>,
) {
    snake_beta_cht(cur, b.act_alpha.view(), b.act_beta.view());
    match gpu.as_mut() {
        Some(cache) => {
            next.adopt_from_array2(&super::gpu_conv::conv_transpose1d_gpu_flat(
                cache,
                cur.view(),
                &b.up_flat,
            ));
        }
        None => conv_transpose1d_flat_cht(cur.view(), &b.up_flat, ws, next),
    }
    std::mem::swap(cur, next);
    for ru in &b.residuals {
        run_residual_unit_cht(cur, next, tmp, ws, ru, gpu);
    }
}

fn run_residual_unit_cht(
    cur: &mut ChT,
    next: &mut ChT,
    tmp: &mut ChT,
    ws: &mut ConvWorkspace,
    u: &DecoderResidualUnit,
    gpu: &mut Option<GpuMatmulCache>,
) {
    let n = cur.ch * cur.t;
    tmp.ensure(cur.ch, cur.t);
    tmp.data[..n].copy_from_slice(&cur.data[..n]);
    snake_beta_cht(cur, u.act1_alpha.view(), u.act1_beta.view());
    causal_conv1d_flat_cht_maybe_gpu(cur.view(), &u.conv1_flat, ws, gpu.as_mut(), next);
    std::mem::swap(cur, next);
    snake_beta_cht(cur, u.act2_alpha.view(), u.act2_beta.view());
    causal_conv1d_flat_cht_maybe_gpu(cur.view(), &u.conv2_flat, ws, gpu.as_mut(), next);
    std::mem::swap(cur, next);
    for i in 0..n {
        cur.data[i] += tmp.data[i];
    }
}

fn rope_tables(
    inv_freq: &[f64],
    seq_len: usize,
    head_dim: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let half = head_dim / 2;
    let mut cos = vec![vec![0f32; half]; seq_len];
    let mut sin = vec![vec![0f32; half]; seq_len];
    for pos in 0..seq_len {
        for i in 0..half {
            let angle = pos as f64 * inv_freq[i];
            cos[pos][i] = angle.cos() as f32;
            sin[pos][i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

fn apply_rope_matrix(
    x: &mut Array2<f32>,
    head_dim: usize,
    cos_tbl: &[Vec<f32>],
    sin_tbl: &[Vec<f32>],
) {
    let (t, d) = x.dim();
    let half = head_dim / 2;
    let n_heads = d / head_dim;
    for qi in 0..t {
        for h in 0..n_heads {
            let base = h * head_dim;
            for i in 0..half {
                let x0 = x[[qi, base + i]];
                let x1 = x[[qi, base + half + i]];
                let c = cos_tbl[qi][i];
                let s = sin_tbl[qi][i];
                x[[qi, base + i]] = x0 * c - x1 * s;
                x[[qi, base + half + i]] = x0 * s + x1 * c;
            }
        }
    }
}

fn transformer_layer(
    x: ArrayView2<f32>,
    layer: &TransformerLayer,
    cos_tbl: &[Vec<f32>],
    sin_tbl: &[Vec<f32>],
    sliding_window: usize,
    eps: f32,
) -> Result<Array2<f32>> {
    let (t, d) = x.dim();
    let residual = x.to_owned();
    let h = rms_norm(x, layer.attn_norm.view(), eps);
    let mut q = linear2(h.view(), layer.wq.view(), None);
    let mut k = linear2(h.view(), layer.wk.view(), None);
    let v = linear2(h.view(), layer.wv.view(), None);
    let q_dim = q.ncols();
    apply_rope_matrix(&mut q, layer.head_dim, cos_tbl, sin_tbl);
    apply_rope_matrix(&mut k, layer.head_dim, cos_tbl, sin_tbl);
    let scale = 1.0 / (layer.head_dim as f32).sqrt();
    let n_heads = layer.n_heads;
    let n_kv = layer.n_kv_heads;
    let hd = layer.head_dim;
    let repeats = n_heads / n_kv;
    let mut attn_out = Array2::<f32>::zeros((t, q_dim));
    for hi in 0..n_heads {
        let kv_h = hi / repeats;
        for qi in 0..t {
            let mut weights = vec![f32::NEG_INFINITY; t];
            let mut max_w = f32::NEG_INFINITY;
            for ki in 0..t {
                if ki > qi || qi - ki >= sliding_window {
                    continue;
                }
                let mut dot = 0f32;
                for di in 0..hd {
                    dot += q[[qi, hi * hd + di]] * k[[ki, kv_h * hd + di]];
                }
                dot *= scale;
                weights[ki] = dot;
                max_w = max_w.max(dot);
            }
            let mut sum = 0f32;
            for ki in 0..t {
                if weights[ki].is_finite() {
                    weights[ki] = (weights[ki] - max_w).exp();
                    sum += weights[ki];
                } else {
                    weights[ki] = 0.0;
                }
            }
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
            for di in 0..hd {
                let mut acc = 0f32;
                for ki in 0..t {
                    acc += weights[ki] * v[[ki, kv_h * hd + di]];
                }
                attn_out[[qi, hi * hd + di]] = acc;
            }
        }
    }
    let mut out = linear2(attn_out.view(), layer.wo.view(), None);
    for i in 0..t {
        for j in 0..d {
            out[[i, j]] = residual[[i, j]] + layer.attn_scale[j] * out[[i, j]];
        }
    }
    let residual2 = out.clone();
    let h2 = rms_norm(out.view(), layer.ffn_norm.view(), eps);
    let ff = swiglu(
        linear2(h2.view(), layer.gate.view(), None).view(),
        linear2(h2.view(), layer.up.view(), None).view(),
        &layer.down,
    );
    for i in 0..t {
        for j in 0..d {
            out[[i, j]] = residual2[[i, j]] + layer.ffn_scale[j] * ff[[i, j]];
        }
    }
    Ok(out)
}

fn take_conv(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
) -> Result<(Array3<f32>, Array1<f32>)> {
    let w = take3d(map, &format!("{prefix}.conv.weight"))?;
    let b = take1d(map, &format!("{prefix}.conv.bias"))?;
    Ok((w, b))
}

pub(in crate::speech_tokenizer) fn take2d(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
) -> Result<Array2<f32>> {
    let (data, shape) = map
        .get(key)
        .with_context(|| format!("missing tensor {key}"))?;
    ensure!(shape.len() == 2, "{key}: expected rank 2");
    Array2::from_shape_vec((shape[0], shape[1]), data.clone()).with_context(|| key.to_string())
}

pub(in crate::speech_tokenizer) fn take1d(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
) -> Result<Array1<f32>> {
    let (data, shape) = map
        .get(key)
        .with_context(|| format!("missing tensor {key}"))?;
    ensure!(shape.len() == 1, "{key}: expected rank 1");
    Array1::from_shape_vec(shape[0], data.clone()).with_context(|| key.to_string())
}

pub(crate) fn take3d(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
) -> Result<Array3<f32>> {
    let (data, shape) = map
        .get(key)
        .with_context(|| format!("missing tensor {key}"))?;
    ensure!(shape.len() == 3, "{key}: expected rank 3");
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data.clone())
        .with_context(|| key.to_string())
}

#[cfg(all(test, feature = "incremental-decode"))]
mod kv_cache_tests {
    use super::*;

    /// KV-cached pre-transformer must produce sample-identical output to the
    /// full pre-transformer for the same input. Splits the input into chunks
    /// of varying sizes, runs incremental, concatenates, and compares.
    #[test]
    fn kv_cache_matches_full_pre_transformer() {
        let dir = match std::env::var("RLX_QWEN3_TTS_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => return,
        };
        let model_dir = dir;
        let decoder_dir = model_dir.join("speech_tokenizer");
        if !decoder_dir.join("model.safetensors").is_file() {
            return;
        }
        let decoder = match St12HzDecoder::open(&model_dir) {
            Ok(d) => d,
            Err(_) => return,
        };
        let hidden = decoder.pre_transformer.input_proj_w.ncols();
        // Synthesize a deterministic input.
        let t_total = 24;
        let mut h = Array2::<f32>::zeros((t_total, hidden));
        for i in 0..t_total {
            for j in 0..hidden {
                h[[i, j]] = ((i * 31 + j * 7) as f32 % 17.0 - 8.0) / 8.0;
            }
        }
        let full = decoder.run_pre_transformer_cpu(h.clone()).expect("full pt");

        // Incremental: chunks of 4, 8, 4, 8.
        let chunks = [4usize, 8, 4, 8];
        let mut state = PreTransformerKvState::default();
        let mut row = 0usize;
        let mut concat = Array2::<f32>::zeros(full.dim());
        for &k in &chunks {
            let slice = h.slice(ndarray::s![row..row + k, ..]).to_owned();
            let out = decoder
                .pre_transformer_incremental(slice.view(), &mut state)
                .expect("inc pt");
            for i in 0..k {
                for j in 0..out.ncols() {
                    concat[[row + i, j]] = out[[i, j]];
                }
            }
            row += k;
        }

        let mut max_diff = 0f32;
        for i in 0..t_total {
            for j in 0..full.ncols() {
                let d = (concat[[i, j]] - full[[i, j]]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "KV-cached PT differs from full PT by {max_diff}, expected <1e-3"
        );
    }

    /// `causal_conv1d_incremental` chunked → identical output to the full
    /// conv on the cumulative buffer. Exercises the conv state-passing helper
    /// that's the foundation of the rest of the conv chain rewrite.
    #[test]
    fn conv1d_incremental_matches_full() {
        let dir = match std::env::var("RLX_QWEN3_TTS_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => return,
        };
        let decoder_dir = dir.join("speech_tokenizer");
        if !decoder_dir.join("model.safetensors").is_file() {
            return;
        }
        let mut decoder = match St12HzDecoder::open(&dir) {
            Ok(d) => d,
            Err(_) => return,
        };
        // Use the pre_conv weights as our test conv.
        let in_ch = decoder.pre_conv_flat.in_ch;
        let t_total = 40;
        let mut x = super::super::ops::ChT::default();
        x.ensure(in_ch, t_total);
        for c in 0..in_ch {
            for ti in 0..t_total {
                x.data[c * t_total + ti] = ((c * 13 + ti * 7) as f32 % 19.0 - 9.0) / 9.0;
            }
        }

        // Full reference: single call on the entire buffer.
        let mut ref_out = super::super::ops::ChT::default();
        super::super::ops::causal_conv1d_flat_cht_maybe_gpu(
            x.view(),
            &decoder.pre_conv_flat,
            &mut decoder.conv_ws,
            None,
            &mut ref_out,
        );

        // Chunked: split [10, 14, 16] frames.
        let chunks = [10usize, 14, 16];
        let mut state = Conv1dStateBuf::default();
        let out_ch = decoder.pre_conv_flat.out_ch;
        let mut concat_out = vec![0f32; out_ch * t_total];
        let mut row = 0usize;
        for &k in &chunks {
            let mut chunk = super::super::ops::ChT::default();
            chunk.ensure(in_ch, k);
            for c in 0..in_ch {
                for ti in 0..k {
                    chunk.data[c * k + ti] = x.data[c * t_total + row + ti];
                }
            }
            let out = super::causal_conv1d_incremental(
                &mut state,
                chunk,
                &decoder.pre_conv_flat,
                &mut decoder.conv_ws,
                None,
            );
            for c in 0..out_ch {
                for ti in 0..out.t {
                    concat_out[c * t_total + row + ti] = out.data[c * out.t + ti];
                }
            }
            row += k;
        }

        // Compare. Causal Conv1d with reflect-pad on the LEFT means the
        // very first few outputs depend on the (zero-padded) past, which
        // differs between the chunked and full runs. So we compare past
        // the first `(k-1)*dilation` output positions.
        let skip = (decoder.pre_conv_flat.k - 1) * decoder.pre_conv_flat.dilation;
        let mut max_diff = 0f32;
        for c in 0..out_ch {
            for ti in skip..t_total {
                let r = ref_out.data[c * ref_out.t + ti];
                let g = concat_out[c * t_total + ti];
                let d = (r - g).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "Incremental conv1d differs from full by {max_diff} after first {skip} samples"
        );
    }
}
