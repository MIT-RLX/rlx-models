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

use rlx_voxtral_tts_train::checkpoint::{export_lora_weights, inject_weights};
use rlx_voxtral_tts_train::weights::{WeightStore, lora_param_to_hf_key};
use safetensors::SafeTensors;
use std::fs;
use tempfile::TempDir;

#[test]
fn lora_export_uses_backbone_key_prefix() {
    assert_eq!(lora_param_to_hf_key("lora.3.wq_a"), "layers.3.lora.wq_a");
    let mut weights = WeightStore::default();
    weights
        .0
        .insert("lora.0.wq_a".into(), vec![1.0, 0.0, 0.0, 0.0]);
    weights
        .0
        .insert("lora.0.wq_b".into(), vec![0.0, 1.0, 0.0, 0.0]);
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("lora.safetensors");
    export_lora_weights(&weights, &path).unwrap();
    let bytes = fs::read(&path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    assert!(st.names().contains(&"layers.0.lora.wq_a"));
}

#[test]
fn inject_lora_roundtrip_loads_bank() {
    let dir = TempDir::new().unwrap();
    let consolidated = dir.path().join("consolidated.safetensors");
    let mut base = std::collections::HashMap::new();
    let w = vec![1.0f32, 0.0, 0.0, 1.0]
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    base.insert(
        "layers.0.attention.wq.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &w).unwrap(),
    );
    safetensors::serialize_to_file(&base, None, &consolidated).unwrap();

    let mut lora = WeightStore::default();
    lora.0
        .insert("lora.0.wq_a".into(), vec![1.0, 0.0, 0.0, 0.0]);
    lora.0
        .insert("lora.0.wq_b".into(), vec![0.0, 1.0, 0.0, 0.0]);
    let lora_path = dir.path().join("lora.safetensors");
    export_lora_weights(&lora, &lora_path).unwrap();
    inject_weights(dir.path(), None, Some(&lora_path)).unwrap();

    let store = rlx_voxtral_tts::VoxtralTtsWeightStore::open(dir.path()).unwrap();
    let text = rlx_voxtral_tts::config::TextConfig {
        hidden_size: 2,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        vocab_size: 32,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 64,
        rope_theta: 1_000_000.0,
        intermediate_size: Some(8),
    };
    let bank = rlx_voxtral_tts::load_lora_bank(&store, &text)
        .unwrap()
        .expect("lora bank");
    assert!(bank.has_any());
}
