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

//! Base weights with the LoRA already folded in.
//!
//! The merge happens **inside a [`WeightLoader`]** rather than as a pass over
//! an assembled bundle. `Qwen35Weights::from_loader` drains its loader tensor
//! by tensor, so intercepting `take` folds each delta at the moment the base
//! weight is materialized: one f32 copy, no second pass, and no separate
//! "merged checkpoint" to keep in sync.
//!
//! It also puts the merge on the *canonical* orientation. `take_transposed`
//! here is deliberately implemented as "take, merge, then transpose" instead
//! of delegating, because a delta folded into an already-transposed weight is
//! silently wrong — the model still runs and still returns probabilities.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::safetensors_checkpoint::SafetensorsMmapLoader;
use rlx_core::weight_loader::{HfTranslatingLoader, WeightLoader};
pub use rlx_qwen35::hf::HfQwen35Fixups;
use rlx_qwen35::hf::hf_decoder_layer_count;
use rlx_qwen35::{Qwen35Config, Qwen35Weights};

use crate::lora::{LoraAdapter, module_path_for_gguf};

/// Wraps a loader and folds a peft LoRA into every targeted tensor as it is
/// taken.
pub struct LoraMergingLoader<L: WeightLoader> {
    inner: L,
    adapter: LoraAdapter,
    merged: HashSet<String>,
}

impl<L: WeightLoader> LoraMergingLoader<L> {
    pub fn new(inner: L, adapter: LoraAdapter) -> Self {
        Self {
            inner,
            adapter,
            merged: HashSet::new(),
        }
    }

    /// Adapter modules that were never folded into anything.
    ///
    /// A non-empty result means the name map and the adapter disagree, which
    /// would leave those layers running as the bare base model — accuracy
    /// drops but nothing errors, so callers should treat it as fatal.
    pub fn unmerged(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .adapter
            .modules()
            .filter(|m| !self.merged.contains(*m))
            .map(str::to_string)
            .collect();
        v.sort();
        v
    }

    pub fn merged_count(&self) -> usize {
        self.merged.len()
    }

    fn apply(&mut self, key: &str, data: &mut [f32], shape: &[usize]) -> Result<()> {
        let Some(module) = module_path_for_gguf(key) else {
            return Ok(());
        };
        if self.adapter.get(&module).is_none() {
            return Ok(());
        }
        if shape.len() != 2 {
            bail!("{key}: adapter targets {module} but the base tensor is rank {shape:?}");
        }
        if self.adapter.merge_into(&module, data, shape[0], shape[1])? {
            self.merged.insert(module);
        }
        Ok(())
    }
}

impl<L: WeightLoader> WeightLoader for LoraMergingLoader<L> {
    fn format_id(&self) -> &'static str {
        self.inner.format_id()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.inner.take(key)?;
        self.apply(key, &mut data, &shape)?;
        Ok((data, shape))
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            bail!("transpose requires 2D, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        let mut t = vec![0f32; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = data[i * cols + j];
            }
        }
        Ok((t, vec![cols, rows]))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }

    fn arch_hint(&self) -> Option<&str> {
        self.inner.arch_hint()
    }
}

/// A base checkpoint plus the adapter, resolved into a runnable bundle.
pub struct KevWeights {
    pub cfg: Qwen35Config,
    pub weights: Qwen35Weights,
    /// How many adapter modules were folded in.
    pub merged: usize,
}

/// Load an HF Qwen3.5 base with **no** adapter — the cold start for training.
pub fn load_merged_base_only(base_dir: &Path) -> Result<KevWeights> {
    let empty = LoraAdapter::empty();
    load_merged(base_dir, empty)
}

/// Load an HF Qwen3.5 base directory and fold `adapter` into it.
///
/// `base_dir` is a plain `Qwen/Qwen3.5-*-Base` snapshot (`config.json` +
/// safetensors shards). Fails if any adapter module finds no home rather
/// than running a partially-adapted model.
pub fn load_merged(base_dir: &Path, adapter: LoraAdapter) -> Result<KevWeights> {
    let cfg_path = base_dir.join("config.json");
    let mut cfg = Qwen35Config::from_hf_config_json(&cfg_path)
        .with_context(|| format!("reading {}", cfg_path.display()))?;

    let base = SafetensorsMmapLoader::open(base_dir)
        .with_context(|| format!("opening safetensors in {}", base_dir.display()))?;

    // Reconcile the trunk depth against the tensors on disk — see
    // `hf_decoder_layer_count`. We never build the MTP head, so every real
    // decoder layer is a trunk layer and `nextn_predict_layers` is 0.
    if let Some(found) = hf_decoder_layer_count(base.checkpoint().keys()) {
        let from_cfg = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        if found != from_cfg {
            cfg.num_hidden_layers = found;
            cfg.nextn_predict_layers = 0;
        }
    }
    let mut loader =
        LoraMergingLoader::new(HfQwen35Fixups::new(HfTranslatingLoader::new(base)), adapter);

    let weights = Qwen35Weights::from_loader(&mut loader, &cfg)
        .with_context(|| format!("assembling Qwen3.5 weights from {}", base_dir.display()))?;

    let unmerged = loader.unmerged();
    if !unmerged.is_empty() {
        bail!(
            "{} adapter module(s) were never merged, so those layers would run as \
             the bare base model: {}{}",
            unmerged.len(),
            unmerged
                .iter()
                .take(6)
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            if unmerged.len() > 6 { ", …" } else { "" }
        );
    }

    Ok(KevWeights {
        cfg,
        weights,
        merged: loader.merged_count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A loader serving one named tensor, to drive the wrappers directly.
    struct One {
        key: String,
        data: Vec<f32>,
        shape: Vec<usize>,
    }

    impl WeightLoader for One {
        fn format_id(&self) -> &'static str {
            "test"
        }
        fn len(&self) -> usize {
            1
        }
        fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            if key != self.key {
                anyhow::bail!("no such tensor: {key}");
            }
            Ok((self.data.clone(), self.shape.clone()))
        }
        fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            self.take(key)
        }
        fn remaining_keys(&self) -> Vec<String> {
            vec![self.key.clone()]
        }
    }

    /// `ssm_a` must arrive as `-exp(A_log)`.
    ///
    /// Getting this wrong is invisible at position 0 — the recurrent state is
    /// zero there, so the decay does nothing — and compounds afterwards. On
    /// Qwen3.5-0.8B one `A_log` entry is positive, so the raw value makes
    /// that head's state *grow*.
    #[test]
    fn a_log_becomes_a_negative_decay_coefficient() {
        let a_log = vec![0.2559f32, -2.0312, -6.5938];
        let mut l = HfQwen35Fixups::new(One {
            key: "blk.0.ssm_a".into(),
            data: a_log.clone(),
            shape: vec![3],
        });
        let (got, shape) = l.take("blk.0.ssm_a").expect("take");
        assert_eq!(shape, vec![3]);
        assert_eq!(l.converted(), 1);
        for (g, raw) in got.iter().zip(&a_log) {
            assert!((g - -raw.exp()).abs() < 1e-6, "got {g} for A_log {raw}");
            assert!(*g < 0.0, "a decay coefficient must be negative, got {g}");
        }
        // The positive A_log is exactly the entry that would make the state grow.
        assert!(a_log[0] > 0.0 && got[0] < 0.0);
    }

    #[test]
    fn other_tensors_pass_through_untouched() {
        let data = vec![1.0f32, -2.0, 3.0];
        let mut l = HfQwen35Fixups::new(One {
            key: "blk.0.attn_q.weight".into(),
            data: data.clone(),
            shape: vec![3],
        });
        let (got, _) = l.take("blk.0.attn_q.weight").expect("take");
        assert_eq!(got, data);
        assert_eq!(l.converted(), 0);
    }

    /// The trunk depth must come from the tensors, not from
    /// `num_hidden_layers - mtp_num_hidden_layers`: HF keeps MTP in a
    /// separate `mtp.*` module, so that subtraction drops a real layer.
    #[test]
    fn decoder_layers_are_counted_from_the_tensor_names() {
        let keys = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.22.mlp.up_proj.weight",
            "model.language_model.layers.23.mlp.up_proj.weight",
            // Separate module — must NOT extend the trunk.
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.fc.weight",
        ];
        assert_eq!(hf_decoder_layer_count(keys.into_iter()), Some(24));
    }

    #[test]
    fn layer_count_is_none_without_any_layers() {
        assert_eq!(
            hf_decoder_layer_count(["model.embed_tokens.weight"].into_iter()),
            None
        );
    }
}
