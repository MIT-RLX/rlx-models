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

//! Inkling runner — eager text generate on the verified reference path.
//!
//! Full Unsloth GGUF dequant → generate is the next step; today this runner:
//! - sniffs GGUF metadata (`InklingTextConfig::from_gguf_path`)
//! - runs greedy generate from a loaded [`TextWeights`] (HF tiny fixture / synth)

use crate::config::InklingTextConfig;
use crate::eager::{TextWeights, forward_logits, greedy_next};
use anyhow::{Result, bail};

pub struct InklingRunner {
    pub cfg: InklingTextConfig,
    pub weights: TextWeights,
}

impl InklingRunner {
    pub fn new(cfg: InklingTextConfig, weights: TextWeights) -> Self {
        Self { cfg, weights }
    }

    pub fn config(&self) -> &InklingTextConfig {
        &self.cfg
    }

    pub fn predict_logits(&self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        forward_logits(&self.cfg, &self.weights, prompt_ids)
    }

    /// Greedy decode (full recompute each step — fine for tiny / debug).
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            bail!("empty prompt");
        }
        let mut ids = prompt_ids.to_vec();
        for _ in 0..n_new {
            let next = greedy_next(&self.cfg, &self.weights, &ids)?;
            ids.push(next);
            on_token(next);
            if next == self.cfg.eos_token_id {
                break;
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{synthetic_text_weights, tiny_cfg};

    #[test]
    fn synth_generate_extends_prompt() {
        let cfg = tiny_cfg();
        let w = synthetic_text_weights(&cfg);
        let runner = InklingRunner::new(cfg.clone(), w);
        let prompt = vec![1u32, 2, 3];
        let mut new_toks = Vec::new();
        let out = runner
            .generate(&prompt, 3, |t| new_toks.push(t))
            .expect("generate");
        assert_eq!(out.len(), prompt.len() + new_toks.len());
        assert_eq!(&out[..prompt.len()], prompt.as_slice());
        assert!(!new_toks.is_empty());
        assert!(new_toks.iter().all(|&t| (t as usize) < cfg.vocab_size));
    }
}
