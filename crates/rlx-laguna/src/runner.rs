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

//! Laguna runners — synth eager ([`LagunaRunner`]) and packed GGUF
//! ([`LagunaPackedRunner`]) with KV-cached generate (no quant F32 expand).

use crate::config::LagunaConfig;
use crate::eager::{TextWeights, forward_logits, greedy_next};
use crate::packed::LagunaPackedWeights;
use anyhow::{Result, bail};
use rlx_core::GgufLoader;
use std::path::Path;

pub struct LagunaRunner {
    pub cfg: LagunaConfig,
    pub weights: TextWeights,
}

impl LagunaRunner {
    pub fn new(cfg: LagunaConfig, weights: TextWeights) -> Self {
        Self { cfg, weights }
    }

    pub fn builder() -> LagunaRunnerBuilder {
        LagunaRunnerBuilder::default()
    }

    /// Opt-in F32 expand of a Laguna GGUF into eager [`TextWeights`].
    ///
    /// Requires [`crate::memory::allow_f32_expand`] (`RLX_LAGUNA_ALLOW_F32_EXPAND=1`
    /// or `--allow-f32-expand`). Check header sniff `F32-expand≈` first — packed
    /// generate is preferred.
    pub fn try_from_gguf_f32(path: impl AsRef<Path>) -> Result<Self> {
        crate::memory::refuse_f32_expand("LagunaRunner::try_from_gguf_f32")?;
        let path = path.as_ref();
        let mut loader = GgufLoader::from_file(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF8 GGUF path: {}", path.display()))?,
        )?;
        let cfg = LagunaConfig::from_gguf(loader.file())?;
        let weights = crate::weights::load_text_weights_from_gguf_f32(&mut loader)?;
        Ok(Self { cfg, weights })
    }

    pub fn config(&self) -> &LagunaConfig {
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

#[derive(Default)]
pub struct LagunaRunnerBuilder {
    cfg: Option<LagunaConfig>,
    weights: Option<TextWeights>,
}

impl LagunaRunnerBuilder {
    pub fn config(mut self, cfg: LagunaConfig) -> Self {
        self.cfg = Some(cfg);
        self
    }

    pub fn weights(mut self, weights: TextWeights) -> Self {
        self.weights = Some(weights);
        self
    }

    pub fn build(self) -> Result<LagunaRunner> {
        let cfg = self
            .cfg
            .ok_or_else(|| anyhow::anyhow!("LagunaRunner: missing config"))?;
        let weights = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("LagunaRunner: missing weights"))?;
        Ok(LagunaRunner::new(cfg, weights))
    }
}

/// Production path: mmap GGUF + packed mat metadata + native-F32 side tensors.
///
/// Retains [`GgufLoader`] so packed generate / [`crate::device_matmul::DeviceMatmul`]
/// can borrow quantized bytes without F32 expand.
pub struct LagunaPackedRunner {
    pub cfg: LagunaConfig,
    pub weights: LagunaPackedWeights,
    /// Kept alive for mmap-backed packed byte borrows.
    pub loader: GgufLoader,
}

impl LagunaPackedRunner {
    pub fn from_gguf_packed(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut loader = GgufLoader::from_file(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("rlx-laguna: non-UTF8 path {}", path.display()))?,
        )
        .map_err(|e| anyhow::anyhow!("rlx-laguna: packed mmap open {}: {e:#}", path.display()))?;
        if loader.architecture() != "laguna" {
            bail!(
                "rlx-laguna: expected general.architecture=laguna, got {}",
                loader.architecture()
            );
        }
        let cfg = LagunaConfig::from_gguf(loader.file())?;
        let weights = LagunaPackedWeights::from_loader(&mut loader, &cfg)?;
        Ok(Self {
            cfg,
            weights,
            loader,
        })
    }

    /// Load an **mlx-community Laguna directory** (HF `config.json` + affine
    /// safetensors) — the packed weights carry their bytes inline
    /// (`MatWeight::PackedMlx`), so the retained `loader` is an empty GGUF
    /// placeholder (the host forward only reads it for the GGUF-`Packed` branch,
    /// which never fires on an all-affine model).
    pub fn from_mlx_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let dir_s = dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("rlx-laguna: non-UTF8 mlx dir {}", dir.display()))?;
        let cfg = LagunaConfig::from_json_path(dir.join("config.json"))?;
        if cfg.model_type != "laguna" {
            bail!(
                "rlx-laguna: expected model_type=laguna, got {}",
                cfg.model_type
            );
        }
        let weights = crate::mlx_load::load_mlx_weights(dir_s, &cfg)?;
        Ok(Self {
            cfg,
            weights,
            loader: GgufLoader::empty("laguna"),
        })
    }

    pub fn config(&self) -> &LagunaConfig {
        &self.cfg
    }

    pub fn weights(&self) -> &LagunaPackedWeights {
        &self.weights
    }

    pub fn loader(&self) -> &GgufLoader {
        &self.loader
    }

    pub fn predict_next(&self, prompt_ids: &[u32]) -> Result<u32> {
        crate::packed_forward::greedy_next(&self.cfg, &self.weights, &self.loader, prompt_ids, None)
    }

    /// KV-cached greedy decode on packed mmap weights (no full F32 expand).
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.generate_with_device(prompt_ids, n_new, None, &mut on_token)
    }

    /// Like [`Self::generate`], optionally accelerating packed matmuls via
    /// [`crate::device_matmul::DeviceMatmul`] (Metal / MLX / …).
    pub fn generate_with_device(
        &self,
        prompt_ids: &[u32],
        n_new: usize,
        accel: Option<&mut crate::device_matmul::DeviceMatmul>,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        crate::packed_forward::generate(
            &self.cfg,
            &self.weights,
            &self.loader,
            prompt_ids,
            n_new,
            &mut on_token,
            accel,
        )
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
        let runner = LagunaRunner::new(cfg.clone(), w);
        let prompt = vec![1u32, 2, 3];
        let mut new_toks = Vec::new();
        let out = runner.generate(&prompt, 3, |t| new_toks.push(t)).unwrap();
        assert_eq!(out.len(), prompt.len() + new_toks.len());
        assert_eq!(new_toks.len(), 3);
    }

    #[test]
    fn gguf_f32_path_is_refused_by_default() {
        if crate::memory::allow_f32_expand() {
            return;
        }
        let err = match LagunaRunner::try_from_gguf_f32("/tmp/Laguna-XS-2.1-Q4_K_M.gguf") {
            Ok(_) => panic!("expected refuse_f32_expand"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("disabled") || msg.contains("FORBIDDEN") || msg.contains("allow-f32"),
            "unexpected: {msg}"
        );
    }
}
