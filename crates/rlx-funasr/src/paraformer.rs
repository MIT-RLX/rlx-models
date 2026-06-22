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

//! **Paraformer** — non-autoregressive ASR.
//!
//! Pipeline: LFR/CMVN features → SAN-M encoder (graph) → CIF predictor
//! ([`crate::cif`], host) producing one acoustic embedding per output token →
//! SAN-M decoder (graph) → per-token `argmax`. The decoder consumes the CIF
//! acoustic embeddings directly as its input sequence (the decoder token
//! embedding table is unused at inference), exactly as in `paraformer/model.py`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::built_from_hir;
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{FusionPolicy, HirModule, HirNodeId};
use rlx_runtime::Device;

use crate::cache::GraphCache;
use crate::cif::{self, PredictorWeights};
use crate::config::ParaformerConfig;
use crate::frontend::WavFrontend;
use crate::sanm::{Graph, build_sanm_encoder};
use crate::tokenizer::Tokenizer;
use crate::weights::RefSource;

/// A loaded Paraformer model.
pub struct Paraformer {
    cfg: ParaformerConfig,
    weights: WeightMap,
    frontend: WavFrontend,
    tokenizer: Option<Tokenizer>,
    device: Device,
    enc_cache: GraphCache,
    dec_cache: GraphCache,
}

impl Paraformer {
    /// Open a Paraformer model directory.
    pub fn open(dir: &Path, device: Device) -> Result<Self> {
        let mut cfg = ParaformerConfig::from_dir(dir)?;
        let weights = crate::weights::load_dir(dir)?;
        // Trust the checkpoint for the vocabulary size.
        if let Some((_, s)) = weights.get("decoder.output_layer.weight") {
            if !s.is_empty() {
                cfg.vocab_size = s[0];
            }
        }
        let cmvn = crate::frontend::load_configured_cmvn(dir);
        let frontend = WavFrontend::new(cfg.frontend.clone(), cmvn);
        let tokenizer = Tokenizer::from_dir(dir).ok();
        Ok(Self {
            cfg,
            weights,
            frontend,
            tokenizer,
            device,
            enc_cache: GraphCache::new(4),
            dec_cache: GraphCache::new(4),
        })
    }

    /// Construct from an in-memory config + weights (used by tests).
    pub fn from_parts(cfg: ParaformerConfig, weights: WeightMap, device: Device) -> Self {
        let frontend = WavFrontend::new(cfg.frontend.clone(), None);
        Self {
            cfg,
            weights,
            frontend,
            tokenizer: None,
            device,
            enc_cache: GraphCache::new(4),
            dec_cache: GraphCache::new(4),
        }
    }

    /// The model configuration.
    pub fn config(&self) -> &ParaformerConfig {
        &self.cfg
    }

    /// Run the SAN-M encoder over features `[t, input_size]`; returns the
    /// encoder hidden states `[t, d]`.
    pub fn encode(&self, feats: &[f32], t: usize) -> Result<Vec<f32>> {
        let in_dim = self.cfg.encoder.input_size;
        ensure!(feats.len() == t * in_dim, "feature length mismatch");
        let cfg = &self.cfg;
        let weights = &self.weights;
        let build = || -> anyhow::Result<rlx_flow::BuiltModel> {
            let mut params = HashMap::new();
            let mut hir =
                HirModule::new("paraformer_encoder").with_fusion_policy(FusionPolicy::Direct);
            {
                let mut src = RefSource(weights);
                let mut g = Graph::new(&mut hir, &mut params, &mut src);
                let x = g.input("feats", &[1, t, in_dim]);
                let enc = build_sanm_encoder(&mut g, x, &cfg.encoder, "encoder", t, false)?;
                g.set_output(enc);
            }
            built_from_hir(hir, params)
        };
        self.enc_cache
            .run(t as u64, self.device, build, &[("feats", feats)])?
            .into_iter()
            .next()
            .context("paraformer encoder produced no output")
    }

    /// Run the SAN-M decoder over the encoder states and CIF acoustic
    /// embeddings; returns the per-token logits `[l, vocab]`.
    pub fn decode_logits(
        &self,
        encoder_out: &[f32],
        t: usize,
        acoustic: &[f32],
        l: usize,
    ) -> Result<Vec<f32>> {
        let d = self.cfg.decoder.dim;
        ensure!(encoder_out.len() == t * d, "encoder length mismatch");
        ensure!(acoustic.len() == l * d, "acoustic length mismatch");
        let cfg = &self.cfg;
        let weights = &self.weights;
        let build = || -> anyhow::Result<rlx_flow::BuiltModel> {
            let mut params = HashMap::new();
            let mut hir =
                HirModule::new("paraformer_decoder").with_fusion_policy(FusionPolicy::Direct);
            {
                let mut src = RefSource(weights);
                let mut g = Graph::new(&mut hir, &mut params, &mut src);
                let memory = g.input("memory", &[1, t, d]);
                let tgt = g.input("tgt", &[1, l, d]);
                let logits = build_decoder(&mut g, tgt, memory, cfg, l, t)?;
                g.set_output(logits);
            }
            built_from_hir(hir, params)
        };
        let key = ((t as u64) << 32) | l as u64;
        self.dec_cache
            .run(
                key,
                self.device,
                build,
                &[("memory", encoder_out), ("tgt", acoustic)],
            )?
            .into_iter()
            .next()
            .context("paraformer decoder produced no output")
    }

    /// Load the CIF predictor head weights from the checkpoint.
    fn predictor_weights(&self) -> Result<PredictorWeights> {
        let g = |k: &str| -> Result<(Vec<f32>, Vec<usize>)> {
            let (d, s) = self
                .weights
                .get(k)
                .with_context(|| format!("missing predictor weight {k}"))?;
            Ok((d.to_vec(), s.to_vec()))
        };
        let (conv_w, _) = g("predictor.cif_conv1d.weight")?;
        let (conv_b, _) = g("predictor.cif_conv1d.bias")?;
        let (out_w, _) = g("predictor.cif_output.weight")?;
        let (out_b, _) = g("predictor.cif_output.bias")?;
        Ok(PredictorWeights {
            conv_w,
            conv_b,
            out_w,
            out_b: out_b.first().copied().unwrap_or(0.0),
        })
    }

    /// Transcribe mono 16 kHz PCM → token ids (blank/sos/eos removed).
    pub fn transcribe_ids(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        let feats = self.frontend.extract(pcm);
        ensure!(feats.n_frames > 0, "audio too short for Paraformer");
        let in_dim = self.cfg.encoder.input_size;
        ensure!(
            feats.feat_dim == in_dim,
            "frontend dim {} != encoder input {}",
            feats.feat_dim,
            in_dim
        );
        let t = feats.n_frames;
        let d = self.cfg.encoder.output_size;

        let encoder_out = self.encode(&feats.data, t)?;
        let pw = self.predictor_weights()?;
        let alphas = cif::compute_alphas(&encoder_out, t, d, &pw, &self.cfg.predictor);
        let (acoustic, l) =
            cif::integrate_and_fire(&encoder_out, t, d, &alphas, &self.cfg.predictor);
        if l == 0 {
            return Ok(Vec::new());
        }
        let logits = self.decode_logits(&encoder_out, t, &acoustic, l)?;
        Ok(argmax_decode(
            &logits,
            l,
            self.cfg.vocab_size,
            self.cfg.blank_id,
        ))
    }

    /// Transcribe mono 16 kHz PCM → text.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        let ids = self.transcribe_ids(pcm)?;
        Ok(match &self.tokenizer {
            Some(tok) => tok.decode(&ids, true),
            None => ids
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        })
    }
}

/// Build the SAN-M decoder graph → logits `[1, l, vocab]`.
fn build_decoder(
    g: &mut Graph,
    tgt: HirNodeId,
    memory: HirNodeId,
    cfg: &ParaformerConfig,
    l: usize,
    t: usize,
) -> Result<HirNodeId> {
    let dc = &cfg.decoder;
    let d = dc.dim;
    let eps = dc.ln_eps;
    let mut x = tgt;
    // cross-attention layers
    for i in 0..dc.att_layer_num {
        let p = format!("decoder.decoders.{i}");
        x = g.decoder_layer(
            x,
            Some(memory),
            &p,
            d,
            dc.n_heads,
            dc.linear_units,
            dc.self_kernel,
            dc.self_sanm_shfit,
            true,
            l,
            t,
            eps,
        )?;
    }
    // FSMN-only layers (no cross-attention)
    for i in 0..dc.num_blocks.saturating_sub(dc.att_layer_num) {
        let p = format!("decoder.decoders2.{i}");
        x = g.decoder_layer(
            x,
            None,
            &p,
            d,
            dc.n_heads,
            dc.linear_units,
            dc.self_kernel,
            0,
            true,
            l,
            t,
            eps,
        )?;
    }
    // feed-forward-only layer
    {
        let p = "decoder.decoders3.0";
        x = g.decoder_layer(
            x,
            None,
            p,
            d,
            dc.n_heads,
            dc.linear_units,
            dc.self_kernel,
            0,
            false,
            l,
            t,
            eps,
        )?;
    }
    x = g.layer_norm(
        x,
        "decoder.after_norm.weight",
        "decoder.after_norm.bias",
        eps,
    )?;
    g.linear(
        x,
        "decoder.output_layer.weight",
        Some("decoder.output_layer.bias"),
        cfg.vocab_size,
    )
}

/// Per-token argmax, dropping the blank id (Paraformer is non-autoregressive:
/// one token per acoustic embedding, no repeat-collapsing).
pub fn argmax_decode(logits: &[f32], l: usize, vocab: usize, blank: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(l);
    for ti in 0..l {
        let row = &logits[ti * vocab..(ti + 1) * vocab];
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        if best != blank {
            ids.push(best as u32);
        }
    }
    ids
}
