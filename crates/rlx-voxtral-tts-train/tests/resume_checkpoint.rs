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

use rlx_voxtral_tts_train::checkpoint::{export_encoder_weights, load_encoder_weights};
use rlx_voxtral_tts_train::weights::{WeightStore, test_codec_with_dims};
use tempfile::TempDir;

#[test]
fn encoder_checkpoint_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("enc.safetensors");
    let mut enc = WeightStore::default();
    enc.0
        .insert("input_proj.conv.weight".into(), vec![0.5, 0.25, 0.1, 0.05]);
    export_encoder_weights(&enc, &path, &test_codec_with_dims(1, 2, 2)).unwrap();
    let loaded = load_encoder_weights(&path).unwrap();
    assert_eq!(
        loaded.get("input_proj.conv.weight"),
        Some([0.5, 0.25, 0.1, 0.05].as_slice())
    );
}
