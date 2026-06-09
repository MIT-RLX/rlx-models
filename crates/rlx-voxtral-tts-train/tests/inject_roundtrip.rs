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

use rlx_voxtral_tts_train::checkpoint::{export_encoder_weights, inject_weights};
use rlx_voxtral_tts_train::weights::{WeightStore, test_codec_with_dims};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use tempfile::TempDir;

#[test]
fn inject_merges_encoder_keys_with_backup() {
    let dir = TempDir::new().unwrap();
    let consolidated = dir.path().join("consolidated.safetensors");
    let mut base = HashMap::new();
    let existing = vec![1.0f32, 2.0]
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    base.insert(
        "layers.0.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &existing).unwrap(),
    );
    safetensors::serialize_to_file(&base, None, &consolidated).unwrap();

    let mut enc = WeightStore::default();
    enc.0.insert(
        "input_proj.conv.weight".into(),
        vec![0.5, 0.25, 0.125, 0.0625],
    );
    let enc_path = dir.path().join("enc.safetensors");
    export_encoder_weights(&enc, &enc_path, &test_codec_with_dims(1, 2, 2)).unwrap();

    inject_weights(dir.path(), Some(&enc_path), None).unwrap();
    assert!(dir.path().join("consolidated.safetensors.backup").is_file());

    let merged = fs::read(&consolidated).unwrap();
    let st = SafeTensors::deserialize(&merged).unwrap();
    assert!(st.names().contains(&"layers.0.weight"));
    assert!(
        st.names()
            .iter()
            .any(|k| k.starts_with("audio_tokenizer.input_proj"))
    );
}
