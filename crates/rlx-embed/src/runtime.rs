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

//! Unified high-level embedding loader (auto-detect arch, lazy recompile).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_core::gguf_config::{EmbedGgufKind, embed_gguf_kind};
use rlx_core::validate_standard_device;
use rlx_core::weights::pick_default;
use rlx_gguf::GgufFile;
use rlx_runtime::{CompiledGraph, Device};

use rlx_core::weight_map::WeightMap;

#[cfg(feature = "hf-download")]
use super::arch::default_pooling;
use super::arch::{Arch, detect_arch, detect_arch_from_gguf};
use super::pooling::Pooling;

/// High-level embedding model — auto-detects BERT / NomicBERT / NomicVision.
pub struct RlxEmbed {
    compiled: CompiledGraph,
    arch: Arch,
    hidden_size: usize,
    device: Device,
    #[allow(dead_code)]
    pooling: Pooling,
    compiled_bs: (usize, usize),
    config_path: Option<PathBuf>,
    weights_path: PathBuf,
}

impl RlxEmbed {
    /// Load from a local directory (`config.json` + `model.safetensors`) on CPU.
    pub fn from_dir(dir: &Path, pooling: Pooling) -> Result<Self> {
        Self::from_dir_on(dir, pooling, Device::Cpu)
    }

    /// Load from a local directory on the given device.
    pub fn from_dir_on(dir: &Path, pooling: Pooling, device: Device) -> Result<Self> {
        validate_standard_device("embed", device)?;
        let weights_path = pick_default(dir)?;
        let config_path = resolve_embed_config_path(dir, &weights_path)?;
        let arch = resolve_embed_arch(config_path.as_deref(), &weights_path)?;
        let (hidden_size, compiled, _) =
            compile_model(arch, config_path.as_deref(), &weights_path, 1, 1, device)?;

        Ok(Self {
            compiled,
            arch,
            hidden_size,
            device,
            pooling,
            compiled_bs: (1, 1),
            config_path,
            weights_path,
        })
    }

    /// Load from a `.gguf` file or a directory containing one (optional sidecar `config.json`).
    pub fn from_weights(path: &Path, pooling: Pooling) -> Result<Self> {
        Self::from_weights_on(path, pooling, Device::Cpu)
    }

    /// Load weights path on the given device.
    pub fn from_weights_on(path: &Path, pooling: Pooling, device: Device) -> Result<Self> {
        validate_standard_device("embed", device)?;
        let weights_path = pick_default(path)?;
        let config_path = path
            .parent()
            .map(|p| p.join("config.json"))
            .filter(|p| p.is_file());
        let arch = resolve_embed_arch(config_path.as_deref(), &weights_path)?;
        let (hidden_size, compiled, _) =
            compile_model(arch, config_path.as_deref(), &weights_path, 1, 1, device)?;

        Ok(Self {
            compiled,
            arch,
            hidden_size,
            device,
            pooling,
            compiled_bs: (1, 1),
            config_path,
            weights_path,
        })
    }

    /// Execution device for this instance.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Load by HuggingFace repo id (downloads when `hf-download` feature enabled).
    #[cfg(feature = "hf-download")]
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on(repo_id, Device::Cpu)
    }

    /// Load by HuggingFace repo id on the given device.
    #[cfg(feature = "hf-download")]
    pub fn from_pretrained_on(repo_id: &str, device: Device) -> Result<Self> {
        validate_standard_device("embed", device)?;
        let repo = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .build()?
            .model(repo_id.to_string());
        let config_file = repo.get("config.json")?;
        let weights_file = repo.get("model.safetensors")?;

        let arch = detect_arch(&config_file)?;
        let pooling = default_pooling(repo_id);
        let (hidden_size, compiled, _) =
            compile_model(arch, Some(&config_file), &weights_file, 1, 1, device)?;

        Ok(Self {
            compiled,
            arch,
            hidden_size,
            device,
            pooling,
            compiled_bs: (1, 1),
            config_path: Some(config_file),
            weights_path: weights_file,
        })
    }

    pub fn dim(&self) -> usize {
        self.hidden_size
    }

    pub fn arch(&self) -> Arch {
        self.arch
    }

    /// Forward on pre-tokenized inputs; returns flattened hidden states.
    pub fn forward(
        &mut self,
        inputs: &[(&str, &[f32])],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<f32>> {
        self.ensure_compiled(batch, seq)?;
        let outputs = self.compiled.run(inputs);
        Ok(outputs.into_iter().next().unwrap_or_default())
    }

    fn ensure_compiled(&mut self, batch: usize, seq: usize) -> Result<()> {
        if self.compiled_bs == (batch, seq) {
            return Ok(());
        }
        let (_, compiled, _) = compile_model(
            self.arch,
            self.config_path.as_deref(),
            &self.weights_path,
            batch,
            seq,
            self.device,
        )?;
        self.compiled = compiled;
        self.compiled_bs = (batch, seq);
        Ok(())
    }
}

fn resolve_embed_config_path(dir: &Path, weights: &Path) -> Result<Option<PathBuf>> {
    let sidecar = dir.join("config.json");
    if sidecar.is_file() {
        return Ok(Some(sidecar));
    }
    if weights.extension().and_then(|s| s.to_str()) == Some("gguf") {
        return Ok(None);
    }
    bail!("{dir:?}: missing config.json (required for safetensors checkpoints)");
}

fn resolve_embed_arch(config_path: Option<&Path>, weights_path: &Path) -> Result<Arch> {
    if let Some(cfg) = config_path {
        return detect_arch(cfg);
    }
    let file = pick_default(weights_path)?;
    if file.extension().and_then(|s| s.to_str()) == Some("gguf") {
        return detect_arch_from_gguf(&file);
    }
    bail!("cannot detect embedding arch without config.json or a .gguf file");
}

/// Compile an embedding graph for the given batch/seq on `device`.
pub fn compile_model(
    arch: Arch,
    config_path: Option<&Path>,
    weights_path: &Path,
    batch: usize,
    seq: usize,
    device: Device,
) -> Result<(usize, CompiledGraph, HashMap<String, Vec<f32>>)> {
    validate_standard_device("embed", device)?;
    let file = pick_default(weights_path)?;
    if file.extension().and_then(|s| s.to_str()) == Some("gguf") {
        rlx_core::gguf_validate_arch(&file, rlx_core::EMBED_GGUF_ARCHES)?;
    }
    let mut wm = WeightMap::from_resolved_path(weights_path)?;

    let (built, hidden_size) = match arch {
        Arch::Bert => {
            let cfg = load_bert_config(config_path, weights_path)?;
            let hs = cfg.hidden_size;
            let built = rlx_bert::flow::build_bert_built(&cfg, &mut wm, batch, seq)?;
            (built, hs)
        }
        Arch::NomicBert => {
            let cfg = load_nomic_config(config_path, weights_path)?;
            let hs = cfg.hidden_size;
            let built = rlx_nomic::flow::build_nomic_built(&cfg, &mut wm, batch, seq)?;
            (built, hs)
        }
        Arch::NomicVision => {
            let cfg_path = config_path.context("NomicVision requires config.json")?;
            let cfg = rlx_core::config::NomicVisionConfig::from_file(cfg_path)?;
            let hs = cfg.hidden_size;
            let built = rlx_vision::flow::build_nomic_vision_built(&cfg, &mut wm, batch)?;
            (built.model, hs)
        }
    };

    let params = built.params().clone();
    let compiled = rlx_core::flow_util::compile_built(built, device)?;
    Ok((hidden_size, compiled, params))
}

fn load_bert_config(
    config_path: Option<&Path>,
    weights_path: &Path,
) -> Result<rlx_core::config::BertConfig> {
    if let Some(p) = config_path {
        return rlx_core::config::BertConfig::from_file(p);
    }
    let raw = GgufFile::from_path(weights_path)?;
    if !matches!(embed_gguf_kind(&raw)?, EmbedGgufKind::Bert) {
        bail!("weights are not a BERT-family GGUF; use NomicBERT config or checkpoint");
    }
    rlx_core::config::BertConfig::from_gguf(&raw)
}

fn load_nomic_config(
    config_path: Option<&Path>,
    weights_path: &Path,
) -> Result<rlx_core::config::NomicBertConfig> {
    if let Some(p) = config_path {
        return rlx_core::config::NomicBertConfig::from_file(p);
    }
    let raw = GgufFile::from_path(weights_path)?;
    if !matches!(embed_gguf_kind(&raw)?, EmbedGgufKind::NomicBert) {
        bail!("weights are not a nomic-bert GGUF; use BERT config or checkpoint");
    }
    rlx_core::config::NomicBertConfig::from_gguf(&raw)
}

/// Compile on CPU (convenience for tests and default [`RlxEmbed::from_dir`]).
pub fn compile_model_cpu(
    arch: Arch,
    config_path: Option<&Path>,
    weights_path: &Path,
    batch: usize,
    seq: usize,
) -> Result<(usize, CompiledGraph, HashMap<String, Vec<f32>>)> {
    compile_model(arch, config_path, weights_path, batch, seq, Device::Cpu)
}
