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

//! An exported MOSS-TTS-Nano, opened by rlx alone, speaks as the runner
//! does from the same ONNX (`MossNative::load_loose`, `weights/tts/moss-nano`):
//! its frame loop a program, its graphs rlx graphs.
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_moss_nano::export::export_rlxp;
use rlx_moss_nano::native::{MossNative, NativeOpts};
use rlx_runtime::Device;
use serde_json::json;

#[test]
fn an_exported_moss_nano_speaks_as_the_runner_does() {
    let dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/moss-nano");
    if !dir.join("moss_tts_prefill.onnx").is_file() {
        eprintln!("skip: no MOSS-TTS-Nano checkpoint");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("moss-nano.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = MossNative::load_loose(&dir, Device::Cpu).unwrap();
    let voice = runner.voice_names()[0].clone();
    let text = "The tea is ready.";
    let opts = NativeOpts::default();

    // The frames: the package's loop draws as the runner's does, so the
    // frames agree until the two prefills' float rounding tips a sampled
    // code (a sampled decoder's ordinary divergence).
    let want = runner
        .generate_codes(text, &runner.voice_prompt_codes(&voice).unwrap(), &opts)
        .unwrap();
    let got = m.run(&[], &json!({"text": text, "voice": voice})).unwrap();
    let nvq = want[0].len();
    let codes: Vec<Vec<i32>> = got[0]
        .data
        .chunks_exact(4 * nvq)
        .map(|f| {
            f.chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect();
    let same = codes.iter().zip(&want).take_while(|(a, b)| a == b).count();
    eprintln!(
        "{} and {} frames, the first {same} the same",
        codes.len(),
        want.len()
    );
    assert!(same >= 8, "the first {same} frames agree");

    // The codec and what follows it, on the runner's frames. The runner's
    // own decode is no reference: it returns every sample NaN (its compile
    // of `moss_audio_tokenizer_decode_full`). The package's decode of these
    // frames is onnxruntime's on the same ONNX within 8.5e-6 (cosine
    // 0.99999994), checked outside this test; here, sound of the length
    // the frames make, trimmed of its pauses at most.
    let got = m
        .synthesize(text, &json!({"voice": voice, "codes": want}))
        .unwrap();
    assert_eq!(got.sample_rate, 48_000);
    let full = want.len() * 3840 * 2;
    assert!(
        got.samples.iter().all(|x| x.is_finite()),
        "non-finite audio"
    );
    let peak = got.samples.iter().fold(0f32, |a, x| a.max(x.abs()));
    eprintln!(
        "{} samples of {full} untrimmed, peak {peak}",
        got.samples.len()
    );
    assert!(got.samples.len() <= full && got.samples.len() > full / 4);
    assert!(peak > 0.05 && peak <= 0.98 + 1e-6, "peak {peak}");
}
