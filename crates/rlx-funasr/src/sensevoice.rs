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

//! **SenseVoiceSmall** — encoder-only multilingual CTC ASR with rich
//! language / event / emotion tags.
//!
//! `[LID, EVENT, EMO, TEXTNORM]` prompt embeddings are gathered on the host and
//! prepended to the LFR features; the SAN-M encoder (`num_blocks` main +
//! `tp_blocks` temporal) runs on the selected RLX device through a CTC head;
//! greedy CTC (`argmax → unique_consecutive → drop blank`) runs on the host.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::built_from_hir;
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{FusionPolicy, HirModule};
use rlx_runtime::Device;

use crate::cache::GraphCache;
use crate::config::SenseVoiceConfig;
use crate::frontend::WavFrontend;
use crate::sanm::{Graph, build_sanm_encoder};
use crate::tokenizer::Tokenizer;
use crate::weights::RefSource;

/// A loaded SenseVoiceSmall model.
pub struct SenseVoice {
    cfg: SenseVoiceConfig,
    weights: WeightMap,
    frontend: WavFrontend,
    tokenizer: Option<Tokenizer>,
    device: Device,
    cache: GraphCache,
}

/// A SenseVoice transcription with its rich tags.
#[derive(Debug, Clone)]
pub struct SenseVoiceResult {
    /// Decoded text (rich tags stripped).
    pub text: String,
    /// Rich `<|...|>` tags (language / event / emotion / text-norm).
    pub tags: Vec<String>,
    /// Raw CTC token ids.
    pub token_ids: Vec<u32>,
}

impl SenseVoice {
    /// Open a SenseVoiceSmall model directory.
    pub fn open(dir: &Path, device: Device) -> Result<Self> {
        let mut cfg = SenseVoiceConfig::from_dir(dir)?;
        let weights = crate::weights::load_dir(dir)?;
        if let Some((_, s)) = weights.get("ctc.ctc_lo.weight") {
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
            cache: GraphCache::new(8),
        })
    }

    /// Construct directly (used by tests with synthetic weights).
    pub fn from_parts(cfg: SenseVoiceConfig, weights: WeightMap, device: Device) -> Self {
        let frontend = WavFrontend::new(cfg.frontend.clone(), None);
        Self {
            cfg,
            weights,
            frontend,
            tokenizer: None,
            device,
            cache: GraphCache::new(8),
        }
    }

    /// The model configuration.
    pub fn config(&self) -> &SenseVoiceConfig {
        &self.cfg
    }

    /// Run the encoder + CTC head over a feature matrix `[t_total, 560]` and
    /// return the per-frame logits `[t_total, vocab]`.
    pub fn run_logits(&self, feats: &[f32], t_total: usize) -> Result<Vec<f32>> {
        let in_dim = self.cfg.encoder.input_size;
        ensure!(feats.len() == t_total * in_dim, "feature length mismatch");
        let cfg = &self.cfg;
        let weights = &self.weights;
        let build = || -> anyhow::Result<rlx_flow::BuiltModel> {
            let mut params = HashMap::new();
            let mut hir = HirModule::new("sensevoice").with_fusion_policy(FusionPolicy::Direct);
            {
                let mut src = RefSource(weights);
                let mut g = Graph::new(&mut hir, &mut params, &mut src);
                let x = g.input("feats", &[1, t_total, in_dim]);
                let enc = build_sanm_encoder(&mut g, x, &cfg.encoder, "encoder", t_total, true)?;
                let logits = g.linear(
                    enc,
                    "ctc.ctc_lo.weight",
                    Some("ctc.ctc_lo.bias"),
                    cfg.vocab_size,
                )?;
                g.set_output(logits);
            }
            built_from_hir(hir, params)
        };
        self.cache
            .run(t_total as u64, self.device, build, &[("feats", feats)])?
            .into_iter()
            .next()
            .context("sensevoice encoder produced no output")
    }

    /// Run the frontend + encoder + CTC head over PCM and return the per-frame
    /// logits `[t_total, vocab]` plus `t_total` (audio frames + 4 prompt frames).
    pub fn logits(&self, pcm: &[f32], lang: &str, use_itn: bool) -> Result<(Vec<f32>, usize)> {
        let feats = self.frontend.extract(pcm);
        ensure!(feats.n_frames > 0, "audio too short for SenseVoice");
        let in_dim = self.cfg.encoder.input_size;
        ensure!(
            feats.feat_dim == in_dim,
            "frontend dim {} != encoder input {}",
            feats.feat_dim,
            in_dim
        );

        let prefix = self.prompt_prefix(lang, use_itn)?; // [4, in_dim]
        let t_total = feats.n_frames + 4;
        let mut input = Vec::with_capacity(t_total * in_dim);
        input.extend_from_slice(&prefix);
        input.extend_from_slice(&feats.data);

        let logits = self.run_logits(&input, t_total)?;
        Ok((logits, t_total))
    }

    /// Transcribe mono 16 kHz PCM.
    pub fn transcribe(&self, pcm: &[f32], lang: &str, use_itn: bool) -> Result<SenseVoiceResult> {
        let (logits, t_total) = self.logits(pcm, lang, use_itn)?;
        let ids = ctc_greedy(&logits, t_total, self.cfg.vocab_size, self.cfg.blank_id);
        let (text, tags) = match &self.tokenizer {
            Some(tok) => (tok.decode(&ids, true), tok.tags(&ids)),
            None => (
                ids.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                Vec::new(),
            ),
        };
        Ok(SenseVoiceResult {
            text,
            tags,
            token_ids: ids,
        })
    }

    /// Gather the `[LID, EVENT, EMO, TEXTNORM]` embedding rows from `embed.weight`.
    fn prompt_prefix(&self, lang: &str, use_itn: bool) -> Result<Vec<f32>> {
        let (emb, shape) = self
            .weights
            .get("embed.weight")
            .context("SenseVoice checkpoint missing embed.weight")?;
        ensure!(shape.len() == 2, "embed.weight must be rank-2");
        let dim = shape[1];
        let rows = [
            SenseVoiceConfig::lid(lang),
            1, // EVENT
            2, // EMO
            SenseVoiceConfig::textnorm(use_itn),
        ];
        let mut out = Vec::with_capacity(4 * dim);
        for r in rows {
            ensure!(r < shape[0], "embed row {r} out of range");
            out.extend_from_slice(&emb[r * dim..(r + 1) * dim]);
        }
        Ok(out)
    }
}

/// Greedy CTC: per-frame argmax → collapse consecutive repeats → drop blank.
pub fn ctc_greedy(logits: &[f32], t: usize, vocab: usize, blank: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut prev = usize::MAX;
    for ti in 0..t {
        let row = &logits[ti * vocab..(ti + 1) * vocab];
        let am = argmax(row);
        if am != prev {
            ids.push(am as u32);
            prev = am;
        }
    }
    ids.into_iter().filter(|&x| x as usize != blank).collect()
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_collapse_and_blank() {
        // vocab 3, blank=0. frames argmax: 0,1,1,0,2 → unique:0,1,0,2 → drop blank:1,2
        let v = 3;
        let logits = vec![
            9., 0., 0., // 0
            0., 9., 0., // 1
            0., 9., 0., // 1
            9., 0., 0., // 0
            0., 0., 9., // 2
        ];
        let ids = ctc_greedy(&logits, 5, v, 0);
        assert_eq!(ids, vec![1, 2]);
    }
}
