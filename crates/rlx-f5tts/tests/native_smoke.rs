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

//! Lightweight native-F5 pipeline smoke: short reference + short text + low NFE
//! so it fits in RAM (the full 664 MB DiT over a long sequence OOMs shared
//! machines). Confirms the ort-free path compiles+runs the 3 graphs end-to-end
//! and produces audible audio. No Whisper. Set RLX_F5TTS_DIR (or weights/tts/f5tts).

use std::path::PathBuf;

use rlx_f5tts::{F5Native, InferOpts};
use rlx_runtime::Device;

fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_F5TTS_DIR") {
        let p = PathBuf::from(d);
        if p.join("F5_Transformer.onnx").is_file() {
            return Some(p);
        }
    }
    let p = PathBuf::from("weights/tts/f5tts");
    p.join("F5_Transformer.onnx").is_file().then_some(p)
}

#[test]
fn f5tts_native_pipeline_smoke() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_F5TTS_DIR");
        return;
    };
    let tts = F5Native::load_on(&dir, Device::Cpu).expect("load f5 native");
    // ~0.5 s of a sine "reference" + a very short target keeps max_duration small.
    let sr = tts.sample_rate() as f32;
    let refa: Vec<f32> = (0..(sr as usize / 2))
        .map(|i| 0.1 * (2.0 * std::f32::consts::PI * 180.0 * i as f32 / sr).sin())
        .collect();
    let opts = InferOpts { nfe: 4, speed: 1.0 };
    let wav = tts
        .synthesize("Hello.", &refa, "Hi.", &opts)
        .expect("native synthesize");
    let peak = wav.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    eprintln!(
        "native f5 smoke: {} samples ({:.2}s) peak={peak:.3}",
        wav.len(),
        wav.len() as f32 / sr
    );
    assert!(!wav.is_empty(), "no audio produced");
    assert!(peak > 0.0, "silent");
}
