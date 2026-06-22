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

//! **CT-Transformer** punctuation restoration.
//!
//! Token ids → embedding (host gather) → SAN-M encoder (graph) → a per-token
//! linear classifier over the punctuation label set. The host inserts each
//! predicted punctuation symbol after its token (`CTTransformer.punc_forward`
//! + the argmax prediction in `inference`).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::built_from_hir;
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{FusionPolicy, HirModule};
use rlx_runtime::Device;

use crate::cache::GraphCache;
use crate::config::CtTransformerConfig;
use crate::sanm::{Graph, build_sanm_encoder};
use crate::tokenizer::Tokenizer;
use crate::weights::RefSource;

/// A loaded CT-Transformer punctuation model.
pub struct CtTransformer {
    cfg: CtTransformerConfig,
    weights: WeightMap,
    tokenizer: Option<Tokenizer>,
    device: Device,
    cache: GraphCache,
}

impl CtTransformer {
    /// Open a CT-Transformer model directory.
    pub fn open(dir: &Path, device: Device) -> Result<Self> {
        let mut cfg = CtTransformerConfig::from_dir(dir)?;
        let weights = crate::weights::load_dir(dir)?;
        if let Some((_, s)) = weights.get("embed.weight") {
            if s.len() == 2 {
                cfg.vocab_size = s[0];
                cfg.embed_unit = s[1];
                cfg.encoder.input_size = s[1];
            }
        }
        let tokenizer = Tokenizer::from_dir(dir).ok();
        Ok(Self {
            cfg,
            weights,
            tokenizer,
            device,
            cache: GraphCache::new(4),
        })
    }

    /// Construct from an in-memory config + weights (used by tests).
    pub fn from_parts(cfg: CtTransformerConfig, weights: WeightMap, device: Device) -> Self {
        Self {
            cfg,
            weights,
            tokenizer: None,
            device,
            cache: GraphCache::new(4),
        }
    }

    /// The model configuration.
    pub fn config(&self) -> &CtTransformerConfig {
        &self.cfg
    }

    /// Predict a punctuation label id per input token. `embed` is the
    /// host-gathered token embedding `[t, embed_unit]`.
    pub fn run_punc(&self, embed: &[f32], t: usize) -> Result<Vec<u32>> {
        let e = self.cfg.embed_unit;
        ensure!(embed.len() == t * e, "embedding length mismatch");
        let punc = self.cfg.punc_list.len();
        let cfg = &self.cfg;
        let weights = &self.weights;
        let build = || -> anyhow::Result<rlx_flow::BuiltModel> {
            let mut params = HashMap::new();
            let mut hir = HirModule::new("ct_transformer").with_fusion_policy(FusionPolicy::Direct);
            {
                let mut src = RefSource(weights);
                let mut g = Graph::new(&mut hir, &mut params, &mut src);
                let x = g.input("embed", &[1, t, e]);
                let enc = build_sanm_encoder(&mut g, x, &cfg.encoder, "encoder", t, false)?;
                let logits = g.linear(enc, "decoder.weight", Some("decoder.bias"), punc)?;
                g.set_output(logits);
            }
            built_from_hir(hir, params)
        };
        let logits = self
            .cache
            .run(t as u64, self.device, build, &[("embed", embed)])?
            .into_iter()
            .next()
            .context("ct-transformer produced no output")?;
        let mut out = Vec::with_capacity(t);
        for ti in 0..t {
            let row = &logits[ti * punc..(ti + 1) * punc];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            out.push(best as u32);
        }
        Ok(out)
    }

    /// Restore punctuation in an ASR transcript. Tokenization splits CJK into
    /// characters and keeps ASCII alphanumeric runs as words; punctuation is
    /// appended after each token according to the model's prediction.
    pub fn restore(&self, text: &str) -> Result<String> {
        let Some(tok) = &self.tokenizer else {
            return Ok(text.to_string());
        };
        let units = split_units(text);
        if units.is_empty() {
            return Ok(String::new());
        }
        // gather embeddings for known tokens (unknown → row 0 / <unk>)
        let (emb, shape) = self
            .weights
            .get("embed.weight")
            .context("CT-Transformer missing embed.weight")?;
        ensure!(shape.len() == 2, "embed.weight must be rank-2");
        let e = shape[1];
        let mut embed = Vec::with_capacity(units.len() * e);
        for u in &units {
            let id = tok.id_of(u).unwrap_or(0) as usize;
            let id = id.min(shape[0] - 1);
            embed.extend_from_slice(&emb[id * e..(id + 1) * e]);
        }
        let punc = self.run_punc(&embed, units.len())?;
        let mut out = String::new();
        for (u, &pid) in units.iter().zip(&punc) {
            out.push_str(u);
            if let Some(sym) = self.cfg.punc_list.get(pid as usize) {
                if sym != "_" && sym != "<unk>" {
                    out.push_str(sym);
                }
            }
        }
        Ok(out)
    }
}

/// Split text into punctuation units: each CJK char is its own unit; ASCII
/// alphanumeric runs are kept whole; whitespace is dropped.
fn split_units(text: &str) -> Vec<String> {
    let mut units = Vec::new();
    let mut word = String::new();
    let flush = |w: &mut String, units: &mut Vec<String>| {
        if !w.is_empty() {
            units.push(std::mem::take(w));
        }
    };
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            word.push(c);
        } else if c.is_whitespace() {
            flush(&mut word, &mut units);
        } else {
            flush(&mut word, &mut units);
            units.push(c.to_string());
        }
    }
    flush(&mut word, &mut units);
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_split() {
        let u = split_units("你好 world 啊");
        assert_eq!(u, vec!["你", "好", "world", "啊"]);
    }
}
