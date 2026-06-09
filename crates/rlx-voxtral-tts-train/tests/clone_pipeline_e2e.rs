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

//! Synthetic train → export → inject → embedding parity quick check.

use rlx_voxtral_tts_train::checkpoint::{
    export_encoder_weights, export_lora_weights, inject_weights,
};
use rlx_voxtral_tts_train::lm_lora_graph::build_lora_train_graph;
use rlx_voxtral_tts_train::weights::{WeightStore, lora_param_to_hf_key, test_codec_with_dims};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use tempfile::TempDir;

fn tiny_text() -> rlx_voxtral_tts::config::TextConfig {
    rlx_voxtral_tts::config::TextConfig {
        hidden_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 8,
        vocab_size: 64,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 64,
        rope_theta: 1_000_000.0,
        intermediate_size: Some(32),
    }
}

#[test]
fn clone_pipeline_export_inject_keys() {
    let dir = TempDir::new().unwrap();
    let consolidated = dir.path().join("consolidated.safetensors");
    let mut base = HashMap::new();
    let blob = vec![0.5f32; 16 * 16]
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    base.insert(
        "layers.0.attention.wq.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16, 16], &blob).unwrap(),
    );
    base.insert(
        "audio_tokenizer.input_proj.conv.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &[0u8; 16]).unwrap(),
    );
    safetensors::serialize_to_file(&base, None, &consolidated).unwrap();

    let mut enc = WeightStore::default();
    enc.0.insert("input_proj.conv.weight".into(), vec![0.1; 4]);
    let enc_path = dir.path().join("enc.safetensors");
    export_encoder_weights(&enc, &enc_path, &test_codec_with_dims(1, 2, 2)).unwrap();

    let text = tiny_text();
    let graph = build_lora_train_graph(&text, 8, 2, 1).unwrap();
    let mut lora = WeightStore::default();
    for slot in &graph.params {
        lora.0.insert(
            slot.name.clone(),
            rlx_voxtral_tts_train::weights::init_lora_param(
                &slot.name,
                2,
                text.hidden_size,
                text.num_attention_heads * text.head_dim,
                text.num_key_value_heads * text.head_dim,
                text.intermediate_size.unwrap_or(text.hidden_size * 3),
            ),
        );
    }
    let lora_path = dir.path().join("lora.safetensors");
    export_lora_weights(&lora, &lora_path).unwrap();

    inject_weights(dir.path(), Some(&enc_path), Some(&lora_path)).unwrap();
    let merged = fs::read(&consolidated).unwrap();
    let st = SafeTensors::deserialize(&merged).unwrap();
    assert!(
        st.names()
            .iter()
            .any(|k| k.starts_with("audio_tokenizer.input_proj"))
    );
    assert!(st.names().contains(&"layers.0.lora.wq_a"));
    assert_eq!(lora_param_to_hf_key("lora.0.wq_b"), "layers.0.lora.wq_b");
}

#[test]
fn rig_real_model_when_env_set() {
    let model_dir = match std::env::var("RLX_VOXTRAL_TTS_DIR") {
        Ok(v) if !v.is_empty() => std::path::PathBuf::from(v),
        _ => {
            eprintln!("skip rig_real_model_when_env_set (RLX_VOXTRAL_TTS_DIR unset)");
            return;
        }
    };
    if std::env::var("RLX_VOXTRAL_TTS_TRAIN_RIG").ok().as_deref() != Some("1") {
        eprintln!("skip rig_real_model_when_env_set (RLX_VOXTRAL_TTS_TRAIN_RIG!=1)");
        return;
    }
    let store = rlx_voxtral_tts::VoxtralTtsWeightStore::open(&model_dir).expect("open model");
    let cfg =
        rlx_voxtral_tts::config::VoxtralTtsConfig::from_model_dir(&model_dir).expect("config");
    let bank = rlx_voxtral_tts::load_lora_bank(&store, &cfg.text_config)
        .ok()
        .flatten();
    let support = rlx_voxtral_tts::voice_clone_support(&store);
    eprintln!(
        "rig: clone_support={support:?} lora={}",
        bank.as_ref()
            .map(|b| b.layers.iter().filter(|l| l.is_some()).count())
            .unwrap_or(0)
    );
}
