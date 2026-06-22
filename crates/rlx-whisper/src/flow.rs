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

use crate::audio::N_FRAMES;
use crate::config::WhisperConfig;
use crate::weights::WhisperWeightPrefix;
use anyhow::Result;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};

#[derive(Debug, Clone)]
pub struct WhisperEncoderFlow<'a> {
    cfg: &'a WhisperConfig,
    batch: usize,
    mel_frames: usize,
    pfx: WhisperWeightPrefix,
    graph_opts: crate::builder::WhisperGraphOpts,
    fused_enc: Option<&'a crate::fused::FusedEncoderWeights>,
}

impl<'a> WhisperEncoderFlow<'a> {
    pub fn new(
        cfg: &'a WhisperConfig,
        weights: &WeightMap,
        batch: usize,
        mel_frames: usize,
    ) -> Self {
        Self::new_opts(
            cfg,
            weights,
            batch,
            mel_frames,
            crate::builder::WhisperGraphOpts::default(),
            None,
        )
    }

    pub fn new_opts(
        cfg: &'a WhisperConfig,
        weights: &WeightMap,
        batch: usize,
        mel_frames: usize,
        graph_opts: crate::builder::WhisperGraphOpts,
        fused_enc: Option<&'a crate::fused::FusedEncoderWeights>,
    ) -> Self {
        Self {
            cfg,
            batch,
            mel_frames,
            pfx: WhisperWeightPrefix::detect(weights),
            graph_opts,
            fused_enc,
        }
    }

    pub fn build(self, weights: &mut WeightMap) -> Result<BuiltModel> {
        build_whisper_encoder_built_opts(
            self.cfg,
            weights,
            &self.pfx,
            self.batch,
            self.mel_frames,
            self.graph_opts,
            self.fused_enc,
        )
    }
}

pub fn build_whisper_encoder_built(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    mel_frames: usize,
) -> Result<BuiltModel> {
    build_whisper_encoder_built_opts(
        cfg,
        weights,
        pfx,
        batch,
        mel_frames,
        crate::builder::WhisperGraphOpts::default(),
        None,
    )
}

pub fn build_whisper_encoder_built_opts(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    mel_frames: usize,
    graph_opts: crate::builder::WhisperGraphOpts,
    fused_enc: Option<&crate::fused::FusedEncoderWeights>,
) -> Result<BuiltModel> {
    let enc_seq = cfg.encoder_seq_len(mel_frames);
    let f = DType::F32;
    let hidden_shape = Shape::new(&[batch, enc_seq, cfg.d_model], f);
    let cfg = cfg.clone();
    let pfx = pfx.clone();
    let fused_enc = fused_enc.cloned();

    ModelFlow::new("whisper_encoder")
        .with_profile(CompileProfile::encoder())
        .input("mel", Shape::new(&[batch, cfg.num_mel_bins, mel_frames], f))
        .plugin_named("whisper.encoder", move |emit, _| {
            let mel = emit.flow_input("mel")?.hir_id();
            let hir = emit
                .module
                .as_hir_mut()
                .expect("whisper encoder flow requires HIR stage");
            let mut b = crate::builder::WhisperBuilder::new(
                hir,
                emit.params,
                emit.weights,
                &pfx,
                batch,
                graph_opts,
            );
            if let Some(ref fused_enc) = fused_enc {
                b = b.with_fused_enc(fused_enc);
            }
            let hidden = b.emit_encoder(&cfg, mel, mel_frames, enc_seq)?;
            Ok(Some(emit.wrap(hidden, hidden_shape.clone())))
        })
        .output("encoder_hidden")
        .build(&mut WeightMapSource(weights))
}

#[derive(Debug, Clone)]
pub struct WhisperDecoderFlow<'a> {
    cfg: &'a WhisperConfig,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
    pfx: WhisperWeightPrefix,
}

impl<'a> WhisperDecoderFlow<'a> {
    pub fn new(
        cfg: &'a WhisperConfig,
        weights: &WeightMap,
        batch: usize,
        dec_seq: usize,
        enc_seq: usize,
    ) -> Self {
        Self {
            cfg,
            batch,
            dec_seq,
            enc_seq,
            pfx: WhisperWeightPrefix::detect(weights),
        }
    }

    pub fn build(self, weights: &mut WeightMap) -> Result<BuiltModel> {
        build_whisper_decoder_built(
            self.cfg,
            weights,
            &self.pfx,
            self.batch,
            self.dec_seq,
            self.enc_seq,
        )
    }
}

pub fn build_whisper_decoder_built(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let h = cfg.d_model;
    let logits_shape = Shape::new(&[batch, dec_seq, cfg.vocab_size], f);
    let cfg = cfg.clone();
    let pfx = pfx.clone();

    ModelFlow::new("whisper_decoder")
        .with_profile(CompileProfile::encoder())
        .input("token_ids", Shape::new(&[batch, dec_seq], f))
        .input("encoder_hidden", Shape::new(&[batch, enc_seq, h], f))
        .plugin_named("whisper.decoder", move |emit, _| {
            let tokens = emit.flow_input("token_ids")?.hir_id();
            let enc = emit.flow_input("encoder_hidden")?.hir_id();
            let hir = emit
                .module
                .as_hir_mut()
                .expect("whisper decoder flow requires HIR stage");
            let mut b = crate::builder::WhisperBuilder::new(
                hir,
                emit.params,
                emit.weights,
                &pfx,
                batch,
                crate::builder::WhisperGraphOpts::default(),
            );
            let logits = b.emit_decoder(&cfg, tokens, enc, dec_seq, enc_seq)?;
            Ok(Some(emit.wrap(logits, logits_shape.clone())))
        })
        .output("logits")
        .build(&mut WeightMapSource(weights))
}

/// Default 30 s chunk mel width.
pub fn default_mel_frames() -> usize {
    N_FRAMES
}

pub fn build_whisper_encoder_graph_sized(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    batch: usize,
    mel_frames: usize,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let pfx = WhisperWeightPrefix::detect(weights);
    rlx_core::flow_util::graph_from_built(build_whisper_encoder_built(
        cfg, weights, &pfx, batch, mel_frames,
    )?)
}

pub fn build_whisper_cross_kv_built(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    enc_seq: usize,
) -> Result<BuiltModel> {
    let (hir, params) = crate::builder::build_whisper_cross_kv_hir(
        cfg,
        &mut WeightMapSource(weights),
        pfx,
        batch,
        enc_seq,
    )?;
    rlx_core::flow_util::built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}

pub fn build_whisper_decoder_prefill_built(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
) -> Result<BuiltModel> {
    build_whisper_decoder_prefill_built_ext(cfg, weights, pfx, batch, dec_seq, enc_seq, true)
}

pub fn build_whisper_decoder_prefill_built_ext(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
    use_cross_cache: bool,
) -> Result<BuiltModel> {
    build_whisper_decoder_prefill_built_ext_opts(
        cfg,
        weights,
        pfx,
        batch,
        dec_seq,
        enc_seq,
        use_cross_cache,
        crate::builder::WhisperGraphOpts::default(),
        None,
    )
}

pub fn build_whisper_decoder_prefill_built_ext_opts(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
    use_cross_cache: bool,
    graph_opts: crate::builder::WhisperGraphOpts,
    fused: Option<&crate::fused::FusedDecoderWeights>,
) -> Result<BuiltModel> {
    let (hir, params) = crate::builder::build_whisper_decoder_prefill_hir_ext_opts(
        cfg,
        &mut WeightMapSource(weights),
        pfx,
        batch,
        dec_seq,
        enc_seq,
        use_cross_cache,
        graph_opts,
        fused,
    )?;
    rlx_core::flow_util::built_from_hir_with_profile(hir, params, CompileProfile::llama32_prefill())
}

pub fn build_whisper_align_hidden_built_ext_opts(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
    max_layer: usize,
    graph_opts: crate::builder::WhisperGraphOpts,
    fused: Option<&crate::fused::FusedDecoderWeights>,
) -> Result<BuiltModel> {
    let (hir, params) = crate::builder::build_whisper_align_hidden_hir_ext_opts(
        cfg,
        &mut WeightMapSource(weights),
        pfx,
        batch,
        dec_seq,
        enc_seq,
        max_layer,
        graph_opts,
        fused,
    )?;
    rlx_core::flow_util::built_from_hir_with_profile(hir, params, CompileProfile::llama32_prefill())
}

pub fn build_whisper_decode_step_built(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    past_seq: usize,
    enc_seq: usize,
) -> Result<BuiltModel> {
    build_whisper_decode_step_built_ext(cfg, weights, pfx, batch, past_seq, enc_seq, false)
}

pub fn build_whisper_decode_step_built_ext(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    past_seq: usize,
    enc_seq: usize,
    use_custom_mask: bool,
) -> Result<BuiltModel> {
    build_whisper_decode_step_built_ext_opts(
        cfg,
        weights,
        pfx,
        batch,
        past_seq,
        enc_seq,
        use_custom_mask,
        crate::builder::WhisperGraphOpts::default(),
        None,
    )
}

pub fn build_whisper_decode_step_built_ext_opts(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    past_seq: usize,
    enc_seq: usize,
    use_custom_mask: bool,
    graph_opts: crate::builder::WhisperGraphOpts,
    fused: Option<&crate::fused::FusedDecoderWeights>,
) -> Result<BuiltModel> {
    let (hir, params) = crate::builder::build_whisper_decode_step_hir_ext_opts(
        cfg,
        &mut WeightMapSource(weights),
        pfx,
        batch,
        past_seq,
        enc_seq,
        use_custom_mask,
        graph_opts,
        fused,
    )?;
    rlx_core::flow_util::built_from_hir_with_profile(hir, params, CompileProfile::gemma_decode())
}

pub fn build_whisper_decoder_graph_sized(
    cfg: &WhisperConfig,
    weights: &mut WeightMap,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let pfx = WhisperWeightPrefix::detect(weights);
    rlx_core::flow_util::graph_from_built(build_whisper_decoder_built(
        cfg, weights, &pfx, batch, dec_seq, enc_seq,
    )?)
}
