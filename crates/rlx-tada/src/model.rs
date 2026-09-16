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

//! Checkpoint layout and top-level loading.
//!
//! ```text
//! <root>/
//!   tada-1b/config.json            HumeAI/tada-1b   (or tada-3b-ml/)
//!   tada-1b/model.safetensors      backbone + diffusion head
//!   tada-codec/decoder/model.safetensors      the vocoder — REQUIRED
//!   tada-codec/encoder/model.safetensors
//!   tada-codec/aligner/model.safetensors      (or aligner-<lang>/ )
//!   tokenizer/tokenizer.json       any ungated Llama-3.2 tokenizer mirror
//! ```
//!
//! The codec encoder and the aligner are needed only to *build* a prompt from
//! reference audio, and are loaded separately by
//! [`PromptBuilder`](crate::prompt_builder::PromptBuilder). The decoder is
//! needed on every synthesis.
//!
//! `tada-1b` also carries a decoder under a `_decoder.` prefix. It is a
//! different, unused set of weights — see [`TadaModel::open`].

use crate::backbone::{Backbone, InputEmbedder};
use crate::codec::CodecDecoder;
use crate::config::{DecoderConfig, TadaConfig};
use crate::head::{DiffusionHead, SolveOptions};
use crate::prof;
use crate::synth::Synthesizer;
use crate::tokenizer::TadaTokenizer;
use crate::weights::TensorStore;
use anyhow::{Context, Result, bail};
use rlx_llama32::Llama32Config;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Identify a checkpoint for the compile cache: path, size and mtime.
///
/// Compiled LIR is only valid for the graph it was built from, and the graph
/// shape follows the weights. Keying on the file's identity means swapping
/// `tada-1b` for `tada-3b-ml` cannot silently reuse the wrong blob.
fn checkpoint_tag(path: &Path) -> Result<String> {
    let meta = std::fs::metadata(path)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let name = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("tada");
    Ok(format!("{name}_{}_{mtime}", meta.len()))
}

/// Directory names tried for the backbone, most specific first.
const BACKBONE_DIRS: &[&str] = &["tada-1b", "tada-3b-ml", "."];

/// Resolve the backbone directory inside `root`.
pub fn find_backbone(root: &Path) -> Result<PathBuf> {
    for name in BACKBONE_DIRS {
        let dir = root.join(name);
        if dir.join("config.json").is_file() && has_safetensors(&dir) {
            return Ok(dir);
        }
    }
    bail!(
        "no TADA backbone under {} — expected one of {BACKBONE_DIRS:?} to hold \
         config.json plus model.safetensors",
        root.display()
    )
}

fn has_safetensors(dir: &Path) -> bool {
    dir.join("model.safetensors").is_file()
}

/// Everything needed to turn a cached voice prompt plus text into audio.
pub struct TadaModel;

impl TadaModel {
    /// Load the synthesis half of the stack.
    pub fn open(root: &Path, device: Device, solve: &SolveOptions) -> Result<Synthesizer> {
        let backbone_dir = find_backbone(root)?;
        let config_path = backbone_dir.join("config.json");
        let cfg = TadaConfig::from_file(&config_path)?;
        // Deserialized from the same file rather than mapped field by field:
        // TADA's config *is* a Llama config with extra keys, and re-deriving it
        // would be a silent-divergence hazard the first time a field moves.
        let llama_cfg = Llama32Config::from_file(&config_path)
            .with_context(|| format!("read Llama config from {}", config_path.display()))?;

        let weights = backbone_dir.join("model.safetensors");
        let store = Arc::new(
            TensorStore::open(&weights).with_context(|| format!("open {}", weights.display()))?,
        );
        // Cache key for compiled graphs: changes whenever the checkpoint does.
        let tag = checkpoint_tag(&weights)?;

        let embed = prof::stage("embedder", || InputEmbedder::load(store.clone(), &cfg))?;
        // Two branches when guidance is on: the conditional forward and the
        // text-suppressed one it is measured against.
        let batch = if solve.uses_guidance() { 2 } else { 1 };
        let backbone = prof::stage("backbone weights", || {
            Backbone::new(store.clone(), llama_cfg, device, batch, &tag)
        })?;
        let head = prof::stage("head weights", || {
            DiffusionHead::load(store.clone(), &cfg, "prediction_head.")
        })?;
        // The vocoder MUST come from `tada-codec/decoder`. `tada-1b` also
        // carries a full decoder under a `_decoder.` prefix, but it is **not**
        // the same weights — 195 of its 201 tensors differ from the published
        // decoder, by up to 1.19. Upstream never uses it: `from_pretrained`
        // fetches `HumeAI/tada-codec/decoder` separately and lets the bundled
        // copy fall out as unexpected keys. Loading it instead yields latents
        // that are correct to 1e-5 and audio that transcribes as a single
        // syllable, which is why this is an error rather than a fallback.
        let dec_path = root.join("tada-codec/decoder/model.safetensors");
        let dec_store = Arc::new(TensorStore::open(&dec_path).with_context(|| {
            format!(
                "codec decoder not found at {} — download `HumeAI/tada-codec`. \
                 The `_decoder.*` copy inside {} is a different, unused set of \
                 weights and will not produce speech.",
                dec_path.display(),
                weights.display()
            )
        })?);
        let decoder = Some(prof::stage("codec decoder weights", || {
            CodecDecoder::load(dec_store, "", DecoderConfig::default())
        })?);
        let tokenizer = prof::stage("tokenizer", || TadaTokenizer::load(&root.join("tokenizer")))?;

        Ok(Synthesizer::with_tag(
            cfg, device, embed, backbone, head, decoder, tokenizer, &tag,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_backbone_names_what_it_looked_for() {
        let dir = std::env::temp_dir().join("rlx_tada_empty_root");
        std::fs::create_dir_all(&dir).unwrap();
        let err = find_backbone(&dir).unwrap_err().to_string();
        assert!(err.contains("tada-1b"), "{err}");
        assert!(err.contains("config.json"), "{err}");
    }
}
