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

//! Map Voxtral-4B-TTS checkpoint keys → Llama builder keys.

use rlx_core::weight_loader::WeightLoader;
use std::collections::HashMap;

/// Maps HF/Llama flow keys (`model.layers.*.input_layernorm`, …) to Mistral
/// consolidated names (`layers.*.attention_norm`, `feed_forward`, …).
pub struct BackbonePrefixLoader<'a> {
    inner: &'a mut dyn WeightLoader,
}

impl<'a> BackbonePrefixLoader<'a> {
    pub fn new(inner: &'a mut dyn WeightLoader) -> Self {
        Self { inner }
    }

    pub fn map_key(key: &str) -> String {
        match key {
            "model.norm.weight" => "norm.weight".into(),
            "model.embed_tokens.weight" => "tok_embeddings.weight".into(),
            "lm_head.weight" => "output.weight".into(),
            k if k.starts_with("model.layers.") => map_layer_key(k),
            k if k.starts_with("layers.") => k.to_string(),
            other => other.to_string(),
        }
    }
}

fn map_layer_key(key: &str) -> String {
    let rest = key.strip_prefix("model.layers.").unwrap_or(key);
    let Some(dot) = rest.find('.') else {
        return key.to_string();
    };
    let (idx, tail) = rest.split_at(dot);
    let tail = &tail[1..];
    let mapped = match tail {
        "input_layernorm.weight" => "attention_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attention.wq.weight",
        "self_attn.k_proj.weight" => "attention.wk.weight",
        "self_attn.v_proj.weight" => "attention.wv.weight",
        "self_attn.o_proj.weight" => "attention.wo.weight",
        "mlp.gate_proj.weight" => "feed_forward.w1.weight",
        "mlp.up_proj.weight" => "feed_forward.w3.weight",
        "mlp.down_proj.weight" => "feed_forward.w2.weight",
        "gate_proj.weight" => "feed_forward.w1.weight",
        "up_proj.weight" => "feed_forward.w3.weight",
        "down_proj.weight" => "feed_forward.w2.weight",
        other => other,
    };
    format!("layers.{idx}.{mapped}")
}

impl WeightLoader for BackbonePrefixLoader<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn take(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take(&Self::map_key(key))
    }

    fn take_transposed(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take_transposed(&Self::map_key(key))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
}

/// Serves checkpoint tensors by name (used to rebuild graphs without reloading safetensors).
pub struct CheckpointParamLoader {
    params: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl CheckpointParamLoader {
    pub fn new(params: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self { params }
    }
}

impl WeightLoader for CheckpointParamLoader {
    fn len(&self) -> usize {
        self.params.len()
    }

    fn take(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        self.params
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing weight {key}"))
    }

    fn take_transposed(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        self.take(key)
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.params.keys().cloned().collect()
    }
}

/// Maps flow keys (`layers.*`) → checkpoint keys (`acoustic_transformer.*`).
pub struct AcousticPrefixLoader<'a> {
    inner: &'a mut rlx_core::weight_map::WeightMap,
}

impl<'a> AcousticPrefixLoader<'a> {
    pub fn new(inner: &'a mut rlx_core::weight_map::WeightMap) -> Self {
        Self { inner }
    }

    fn full_key(key: &str) -> String {
        if key.starts_with(crate::load::PREFIX_ACOUSTIC) {
            key.to_string()
        } else {
            format!("{}{key}", crate::load::PREFIX_ACOUSTIC)
        }
    }
}

impl WeightLoader for AcousticPrefixLoader<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn take(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take(&Self::full_key(key))
    }

    fn take_transposed(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take_transposed(&Self::full_key(key))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.inner
            .remaining_keys()
            .into_iter()
            .filter_map(|k| {
                k.strip_prefix(crate::load::PREFIX_ACOUSTIC)
                    .map(str::to_string)
            })
            .collect()
    }
}

/// Clone-on-take loader for one-shot compiles (acoustic snapshot).
pub struct SnapshotLoader {
    map: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl SnapshotLoader {
    pub fn new(map: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self { map }
    }
}

impl WeightLoader for SnapshotLoader {
    fn len(&self) -> usize {
        self.map.len()
    }

    fn take(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        self.map
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("missing weight {key}"))
    }

    fn take_transposed(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        self.take(key)
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
}

pub(crate) fn snapshot_backbone_params(
    store: &crate::load::VoxtralTtsWeightStore,
) -> anyhow::Result<crate::load::WeightSnapshot> {
    let mut wm = store.load_backbone()?;
    let keys: Vec<String> = wm.keys().map(str::to_string).collect();
    let mut out = HashMap::with_capacity(keys.len());
    for key in keys {
        out.insert(key.clone(), wm.take(&key)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_hf_layer_keys_to_mistral_names() {
        assert_eq!(
            BackbonePrefixLoader::map_key("model.layers.3.input_layernorm.weight"),
            "layers.3.attention_norm.weight"
        );
        assert_eq!(
            BackbonePrefixLoader::map_key("model.layers.0.mlp.gate_proj.weight"),
            "layers.0.feed_forward.w1.weight"
        );
        assert_eq!(
            BackbonePrefixLoader::map_key("model.norm.weight"),
            "norm.weight"
        );
    }
}
