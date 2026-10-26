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

//! Any dense decoder `config.json` describes — Llama, Qwen2/3, Mistral,
//! Gemma 2/3, Phi-3, OLMoE, Qwen MoE, … ([`crate::standard_decoder`]) — as
//! a package, with no family code: the config-driven builder's full-sequence
//! graph through [`export_forward`], run over the whole context for each
//! token. The exporter [`detect`](super::detect) falls back to for a
//! checkpoint no family claims.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail};

use super::decoder::{Built, Forward, export_forward};
use super::{Checkpoint, ExportOptions, ExportReport};
use crate::standard_decoder::{DecoderSpec, RopeScaling, build_standard_decoder_packed};
use crate::weight_loader::WeightLoader;

struct StandardForward {
    spec: DecoderSpec,
    max_positions: usize,
}

impl StandardForward {
    /// The builder's RoPE table, `rows` long: as `build_standard_decoder_packed`
    /// computes it (partial rotary, scaling, attention factor).
    fn table(&self, rows: usize, theta: f64, sc: &RopeScaling) -> (Vec<f32>, Vec<f32>) {
        let dh = self.spec.head_dim;
        let half = dh / 2;
        let n_rot = if self.spec.partial_rotary_factor < 1.0 {
            ((self.spec.partial_rotary_factor * dh as f64) as usize) & !1
        } else {
            dh
        };
        let mscale = sc.mscale();
        let mut cos = vec![0f32; rows * half];
        let mut sin = vec![0f32; rows * half];
        for pos in 0..rows {
            for i in 0..n_rot / 2 {
                let (s, c) = (pos as f64 * sc.inv_freq(i, n_rot, theta)).sin_cos();
                cos[pos * half + i] = (c * mscale) as f32;
                sin[pos * half + i] = (s * mscale) as f32;
            }
        }
        (cos, sin)
    }
}

impl Forward for StandardForward {
    fn family(&self) -> String {
        self.spec.arch.clone()
    }
    fn exported_by(&self) -> &str {
        "rlx-models-core (standard decoder)"
    }
    fn max_positions(&self) -> usize {
        self.max_positions
    }
    fn vocab_size(&self) -> usize {
        self.spec.vocab_size
    }
    fn forward(&self, weights: &mut dyn WeightLoader, seq: usize) -> Result<Built> {
        let mut packed = HashMap::new();
        let built = build_standard_decoder_packed(
            &self.spec,
            weights,
            1,
            seq,
            true,
            false,
            false,
            &mut packed,
        )?;
        if !packed.is_empty() {
            bail!(
                "{}: its weights are quantized ({} packed); export from the f32/bf16 checkpoint",
                self.spec.arch,
                packed.len()
            );
        }
        Ok(built)
    }
    fn tables(&self, rows: usize) -> Vec<(String, Vec<f32>, usize)> {
        let half = self.spec.head_dim / 2;
        let mut out = Vec::new();
        let (c, s) = self.table(rows, self.spec.rope_theta, &self.spec.rope_scaling);
        out.push(("rope.cos.global".to_string(), c, half));
        out.push(("rope.sin.global".to_string(), s, half));
        if let Some((local_theta, _)) = self.spec.rope_dual {
            let (c, s) = self.table(rows, local_theta, &RopeScaling::None);
            out.push(("rope.cos.local".to_string(), c, half));
            out.push(("rope.sin.local".to_string(), s, half));
        }
        out
    }
}

/// Whether the config-driven builder takes the checkpoint at `ck`.
pub fn supports(ck: &Checkpoint) -> bool {
    ck.format == super::Format::Safetensors
        && crate::standard_decoder::classify_config(&ck.config).supported
}

/// Export the dense decoder in `dir` (`config.json`, safetensors,
/// `tokenizer.json`) as the package `out`, by its config alone.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let loader = ck.loader()?;
    let spec = DecoderSpec::from_config_json(&ck.path, loader.as_ref())?;
    drop(loader);
    let text = ck.config.get("text_config").unwrap_or(&ck.config);
    let max_positions = text
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .map_or(usize::MAX, |n| n as usize);
    export_forward(
        &StandardForward {
            spec,
            max_positions,
        },
        dir,
        out,
        opts,
    )
}
