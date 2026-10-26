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

//! The 12 Hz speech decoder (`St12HzDecoder`) decodes fixed codes as the
//! official `qwen-tts` tokenizer does (`scripts/speech_decoder_reference.py`
//! wrote `tests/fixtures/st12hz_decode`).

use std::path::PathBuf;

use rlx_qwen3_tts::speech_tokenizer::open_speech_decoder;
use rlx_runtime::Device;

#[test]
fn the_speech_decoder_decodes_as_the_reference_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("../../weights/tts/qwen3-tts-0.6b-customvoice");
    if !dir.join("speech_tokenizer/model.safetensors").is_file() {
        eprintln!("skip: no Qwen3-TTS checkpoint");
        return;
    }
    let fx = root.join("tests/fixtures/st12hz_decode");
    let codes: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read(fx.join("codes.json")).unwrap()).unwrap();
    let want: Vec<f32> = std::fs::read(fx.join("pcm.f32"))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mut dec = open_speech_decoder(&dir, Device::Cpu).unwrap();
    let got = dec.decode(&codes, Device::Cpu).unwrap();
    assert_eq!(got.len(), want.len());
    let diff = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!("{} samples, max |d| {diff} (peak {peak})", want.len());
    assert!(diff < 1e-3, "max |d| {diff}");
}
