// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Safetensors loader for microsoft/VibeVoice-ASR-Streaming-* checkpoints.

use anyhow::{Context, Result, ensure};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::weights::{VaeEncoderWeights, load_vae_from_map};

pub const PREFIX_ACOUSTIC_ENC: &str = "model.acoustic_tokenizer.encoder.";
pub const PREFIX_SEMANTIC_ENC: &str = "model.semantic_tokenizer.encoder.";
pub const PREFIX_ACOUSTIC_CONN: &str = "model.acoustic_connector.";
pub const PREFIX_SEMANTIC_CONN: &str = "model.semantic_connector.";
pub const PREFIX_LANGUAGE_MODEL: &str = "model.language_model.";
pub const KEY_LM_HEAD: &str = "lm_head.weight";
pub const KEY_EMBED_TOKENS: &str = "model.language_model.embed_tokens.weight";

/// Mmap-backed Streaming ASR checkpoint.
#[derive(Clone)]
pub struct StreamingWeightStore {
    dir: PathBuf,
    checkpoint: Arc<SafetensorsCheckpoint>,
    all_keys: Arc<HashSet<String>>,
}

impl StreamingWeightStore {
    pub fn open(weights_path: &Path) -> Result<Self> {
        let dir = resolve_model_dir(weights_path)?;
        let checkpoint = Arc::new(SafetensorsCheckpoint::open(&dir)?);
        let all_keys = Arc::new(checkpoint.keys().map(str::to_string).collect());
        Ok(Self {
            dir,
            checkpoint,
            all_keys,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.dir
    }

    pub fn load_prefixes(&self, prefixes: &[&str]) -> Result<WeightMap> {
        let want: HashSet<String> = self
            .all_keys
            .iter()
            .filter(|k| prefixes.iter().any(|p| k.starts_with(p)))
            .cloned()
            .collect();
        ensure!(
            !want.is_empty(),
            "no checkpoint keys match {prefixes:?} under {:?}",
            self.dir
        );
        self.checkpoint.load_selected(&want)
    }

    pub fn load_keys(&self, keys: &[&str]) -> Result<WeightMap> {
        let want: HashSet<String> = keys.iter().map(|k| (*k).to_string()).collect();
        self.checkpoint.load_selected(&want)
    }

    /// Acoustic + semantic ConvNeXt encoders and SpeechConnectors.
    pub fn load_vae_pair(&self) -> Result<(VaeEncoderWeights, VaeEncoderWeights)> {
        let mut wm = self.load_prefixes(&[
            PREFIX_ACOUSTIC_ENC,
            PREFIX_SEMANTIC_ENC,
            PREFIX_ACOUSTIC_CONN,
            PREFIX_SEMANTIC_CONN,
        ])?;
        let acoustic = load_vae_from_map(
            &mut wm,
            "model.acoustic_tokenizer.encoder",
            "model.acoustic_connector",
        )?;
        let semantic = load_vae_from_map(
            &mut wm,
            "model.semantic_tokenizer.encoder",
            "model.semantic_connector",
        )?;
        Ok((acoustic, semantic))
    }

    /// Qwen2 trunk + untied lm_head (`model.language_model.*` + `lm_head.weight`).
    pub fn load_language_model_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_LANGUAGE_MODEL, KEY_LM_HEAD])
    }

    pub fn load_token_embed(&self) -> Result<(Vec<f32>, usize, usize)> {
        let mut wm = self.load_keys(&[KEY_EMBED_TOKENS])?;
        let (data, shape) = wm.take(KEY_EMBED_TOKENS)?;
        ensure!(
            shape.len() == 2,
            "embed_tokens shape {shape:?}, expected [vocab, hidden]"
        );
        let (vocab, hidden) = (shape[0], shape[1]);
        ensure!(
            data.len() == vocab * hidden,
            "embed_tokens len {} != vocab*hidden {}",
            data.len(),
            vocab * hidden
        );
        Ok((data, vocab, hidden))
    }
}

pub fn resolve_model_dir(weights_path: &Path) -> Result<PathBuf> {
    if weights_path.is_dir() {
        return Ok(weights_path.to_path_buf());
    }
    weights_path
        .parent()
        .map(Path::to_path_buf)
        .filter(|p| p.is_dir())
        .with_context(|| format!("not a model dir: {weights_path:?}"))
}

/// Maps Qwen2 decoder keys (`model.*`, `lm_head.*`) onto the Streaming checkpoint
/// names (`model.language_model.*`, `lm_head.weight`).
pub struct StreamingLmLoader<'a> {
    inner: &'a mut WeightMap,
}

impl<'a> StreamingLmLoader<'a> {
    pub fn new(inner: &'a mut WeightMap) -> Self {
        Self { inner }
    }
}

/// Map Qwen2 flow keys (`model.*`, `lm_head.*`) onto Streaming checkpoint names
/// (`model.language_model.*`, `lm_head.weight`).
pub fn map_streaming_lm_key(key: &str) -> String {
    if key == "lm_head.weight" {
        return KEY_LM_HEAD.to_string();
    }
    if let Some(rest) = key.strip_prefix("model.") {
        return format!("{PREFIX_LANGUAGE_MODEL}{rest}");
    }
    key.to_string()
}

impl WeightLoader for StreamingLmLoader<'_> {
    fn format_id(&self) -> &'static str {
        "safetensors"
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take(&map_streaming_lm_key(key))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take_transposed(&map_streaming_lm_key(key))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_lm_keys() {
        assert_eq!(map_streaming_lm_key("lm_head.weight"), "lm_head.weight");
        assert_eq!(
            map_streaming_lm_key("model.embed_tokens.weight"),
            "model.language_model.embed_tokens.weight"
        );
        assert_eq!(
            map_streaming_lm_key("model.layers.0.self_attn.q_proj.weight"),
            "model.language_model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            map_streaming_lm_key("model.norm.weight"),
            "model.language_model.norm.weight"
        );
    }
}
