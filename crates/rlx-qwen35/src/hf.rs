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

//! Reading HF-native (safetensors) Qwen3.5 checkpoints with a builder written
//! against llama.cpp GGUFs: the one tensor whose values differ, and the trunk
//! depth the config misreports.

use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;

/// Number of real decoder layers in an HF checkpoint, from the tensor names.
///
/// `Qwen35Config::from_hf_config_value` computes the trunk depth as
/// `num_hidden_layers - mtp_num_hidden_layers`. That is right for a
/// GGUF-derived layout, where llama.cpp appends the MTP block as the last
/// `blk.N` — but **wrong for HF safetensors**, where `Qwen3.5-0.8B-Base`
/// carries `model.language_model.layers.0..23` *and* a separate `mtp.*`
/// module. Trusting the config there drops the last decoder layer and the
/// model still runs, one block short.
///
/// So count the layers that actually exist. `mtp.layers.N` is excluded: it is
/// a different module, and we never build the MTP head. So are the layer
/// stacks of a vision or audio tower sharing the file: MiniCPM-V 4.6/4.7 ship
/// `model.vision_tower.encoder.layers.0..26` next to a 24-layer trunk, and
/// counting them would ask for three decoder layers that do not exist.
pub fn hf_decoder_layer_count<'a>(keys: impl Iterator<Item = &'a str>) -> Option<usize> {
    const TOWERS: [&str; 4] = ["vision", "visual", "audio", "merger"];
    let mut max_idx: Option<usize> = None;
    for k in keys {
        if k.starts_with("mtp.") || k.contains(".mtp.") {
            continue;
        }
        let mut parts = k.splitn(2, ".layers.");
        let (Some(owner), Some(rest)) = (parts.next(), parts.next()) else {
            continue;
        };
        if TOWERS.iter().any(|t| owner.contains(t)) {
            continue;
        }
        let Some(idx) = rest.split('.').next().and_then(|d| d.parse::<usize>().ok()) else {
            continue;
        };
        max_idx = Some(max_idx.map_or(idx, |m: usize| m.max(idx)));
    }
    max_idx.map(|m| m + 1)
}

/// Converts HF-native tensors into the form the `rlx-qwen35` builder expects.
///
/// The builder was written against llama.cpp-converted GGUFs, and the GGUF
/// converter does not just rename tensors — it *transforms* one. The Gated
/// DeltaNet decay coefficient is stored in HF as `linear_attn.A_log`, and
/// llama.cpp writes `A = -exp(A_log)` into `blk.N.ssm_a`. The builder then
/// uses `ssm_a` directly (`decay = exp(A · softplus(dt))`), with a comment
/// recording that applying `-exp` a second time was a real bug once.
///
/// Feeding raw `A_log` through that path is silently catastrophic in a way
/// that is easy to miss: at position 0 the recurrent state is zero, so the
/// decay is unobservable and the first token is *exact*. Every later position
/// compounds the error. On Qwen3.5-0.8B the raw `A_log` even contains a
/// positive entry, so that head's state **grows** instead of decaying.
pub struct HfQwen35Fixups<L: WeightLoader> {
    inner: L,
    converted: usize,
}

impl<L: WeightLoader> HfQwen35Fixups<L> {
    pub fn new(inner: L) -> Self {
        Self {
            inner,
            converted: 0,
        }
    }

    /// How many `ssm_a` tensors were converted from `A_log`.
    pub fn converted(&self) -> usize {
        self.converted
    }

    /// `ssm_a` is the only tensor whose *values* differ between the HF and
    /// GGUF layouts.
    fn is_a_log(key: &str) -> bool {
        key.ends_with(".ssm_a") || key.ends_with("linear_attn.A_log")
    }

    fn fix(&mut self, key: &str, data: &mut [f32]) {
        if !Self::is_a_log(key) {
            return;
        }
        for v in data.iter_mut() {
            *v = -v.exp();
        }
        self.converted += 1;
    }
}

impl<L: WeightLoader> WeightLoader for HfQwen35Fixups<L> {
    fn format_id(&self) -> &'static str {
        self.inner.format_id()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.inner.take(key)?;
        self.fix(key, &mut data);
        Ok((data, shape))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.inner.take_transposed(key)?;
        self.fix(key, &mut data);
        Ok((data, shape))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
    fn arch_hint(&self) -> Option<&str> {
        self.inner.arch_hint()
    }
}

/// The loader every HF safetensors read goes through: HF tensor names served
/// under the GGUF names the builder asks for, with `A_log` converted.
///
/// The runner reads an HF checkpoint twice — once to build, and again after a
/// Metal/MLX upload released the host copy of the projections. The second read
/// once skipped the conversion, so every graph compiled after the reload saw a
/// raw `A_log` (Qwen3.5-0.8B `predict_logits` off by 10.9 on Metal and MLX).
pub fn hf_weight_loader(
    inner: Box<dyn WeightLoader>,
) -> HfQwen35Fixups<rlx_core::HfTranslatingLoader<Box<dyn WeightLoader>>> {
    HfQwen35Fixups::new(rlx_core::HfTranslatingLoader::new(inner))
}

/// The text model of an HF Qwen3.5 directory (a VLM's language model
/// included), as the runner reads it: the config with the trunk depth taken
/// from the tensors, and the weights through [`hf_weight_loader`].
pub fn hf_text_model(
    dir: &std::path::Path,
) -> anyhow::Result<(crate::Qwen35Config, crate::Qwen35Weights)> {
    let mut cfg = crate::Qwen35Config::from_hf_config_json(&dir.join("config.json"))?;
    let inner: Box<dyn WeightLoader> = Box::new(rlx_core::SafetensorsMmapLoader::open(dir)?);
    let keys = inner.remaining_keys();
    if let Some(found) = hf_decoder_layer_count(keys.iter().map(String::as_str)) {
        cfg.num_hidden_layers = found;
        cfg.nextn_predict_layers = 0;
    }
    let mut loader = hf_weight_loader(inner);
    let weights = crate::Qwen35Weights::from_loader(&mut loader, &cfg)?;
    Ok((cfg, weights))
}

#[cfg(test)]
mod tests {
    use super::hf_decoder_layer_count;

    #[test]
    fn layer_count_ignores_vision_tower_layers() {
        // MiniCPM-V 4.7-1B: a 27-layer SigLIP tower beside a 24-layer trunk.
        let keys = [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.23.mlp.up_proj.weight",
            "model.vision_tower.encoder.layers.26.mlp.fc1.weight",
            "model.merger.mlp.0.linear_1.weight",
            "mtp.layers.0.input_layernorm.weight",
        ];
        assert_eq!(hf_decoder_layer_count(keys.into_iter()), Some(24));
    }
}
