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

use rlx_voxtral_tts_train::checkpoint::export_encoder_weights;
use rlx_voxtral_tts_train::weights::{
    WeightStore, codec_has_encoder, merge_codec_encoder_overlay, test_codec_with_dims,
};
use safetensors::SafeTensors;
use std::collections::HashMap;
use tempfile::TempDir;

#[test]
fn merge_codec_encoder_overlay_adds_input_proj() {
    let dir = TempDir::new().unwrap();
    let enc_path = dir.path().join("enc.safetensors");
    let mut enc = WeightStore::default();
    enc.0.insert("input_proj.conv.weight".into(), vec![0.5; 4]);
    export_encoder_weights(&enc, &enc_path, &test_codec_with_dims(1, 2, 2)).unwrap();

    let mut codec = HashMap::new();
    codec.insert(
        "audio_tokenizer.decoder_blocks.0.weight".into(),
        (vec![1.0; 4], vec![4]),
    );
    merge_codec_encoder_overlay(&mut codec, &enc_path).unwrap();
    assert!(codec_has_encoder(&codec));
    assert!(codec.contains_key("audio_tokenizer.input_proj.conv.weight"));
}

#[test]
fn overlay_keys_match_export_format() {
    let dir = TempDir::new().unwrap();
    let enc_path = dir.path().join("enc.safetensors");
    let mut enc = WeightStore::default();
    enc.0
        .insert("input_proj.conv.weight".into(), vec![0.1, 0.2, 0.3, 0.4]);
    export_encoder_weights(&enc, &enc_path, &test_codec_with_dims(1, 2, 2)).unwrap();
    let bytes = std::fs::read(&enc_path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    assert!(st.names().iter().any(|k| k.contains("input_proj")));
}
