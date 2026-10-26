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

#![cfg(feature = "export")]

//! An exported Whisper package — its log-mel front end a graph in a package
//! of its own — transcribes what `WhisperRunner` does. Opt-in:
//! `RLX_TEST_WHISPER_DIR=<a Hugging Face Whisper checkpoint>` and
//! `RLX_TEST_WHISPER_WAV=<16 kHz mono speech>`; `RLX_TEST_DEVICE` (`cpu`).

use rlx_runtime::model::Tensor;
use rlx_whisper::export::{ExportOptions, export_rlxp};
use serde_json::json;

#[test]
fn an_exported_package_transcribes_what_whisper_runner_does() {
    let (Some(dir), Some(wav)) = (
        std::env::var_os("RLX_TEST_WHISPER_DIR"),
        std::env::var_os("RLX_TEST_WHISPER_WAV"),
    ) else {
        eprintln!("skip: set RLX_TEST_WHISPER_DIR and RLX_TEST_WHISPER_WAV");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let pcm = rlx_whisper::audio::load_wav_mono_f32(std::path::Path::new(&wav)).unwrap();
    let device = std::env::var("RLX_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());

    let cfg = rlx_whisper::WhisperConfig::from_file(&dir.join("config.json")).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let frontend = tmp.path().join("whisper-log-mel.rlxp");
    let out = tmp.path().join("whisper.rlxp");
    let opts = ExportOptions {
        frontend_package: Some(frontend.clone()),
        ..ExportOptions::default()
    };
    let report = export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!(
        "export: {report:?}, package {} bytes, front end {} bytes",
        std::fs::metadata(&out).unwrap().len(),
        std::fs::metadata(&frontend).unwrap().len()
    );

    // The front end, a model of its own, against the crate's own log-mel.
    let padded = rlx_whisper::audio::pad_or_trim_pcm(&pcm);
    let want = rlx_whisper::pcm_to_log_mel(&padded, cfg.num_mel_bins, 3000).data;
    let mut front =
        rlx_runtime::model::open_source(frontend.to_str().unwrap(), json!({"device": "cpu"}))
            .unwrap();
    let got = front
        .run(
            &[Tensor::f32("audio", vec![padded.len()], &padded)],
            &json!({}),
        )
        .unwrap()[0]
        .to_f32()
        .unwrap();
    assert_eq!(got.len(), want.len());
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("log-mel graph vs pcm_to_log_mel: max |diff| {err:.2e}");
    assert!(err < 1e-3, "log-mel graph differs by {err}");

    rlx_text::install_model_tokenizer();
    let mut m =
        rlx_runtime::model::open_source(out.to_str().unwrap(), json!({"device": device})).unwrap();
    assert_eq!(m.info().task, "speech-recognition");
    let text = m.transcribe(&pcm, 16_000, &json!({})).unwrap();
    drop(m);

    let mut runner = rlx_whisper::WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(rlx_runtime::parse_device(&device).unwrap())
        .language("en")
        .build()
        .unwrap();
    let want_text = runner.transcribe_greedy(&pcm).unwrap();
    eprintln!("package: {text:?}\nrunner:  {want_text:?}");
    assert_eq!(text.trim(), want_text.trim());
}
