// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! An exported FireRedAudio (`weights/asr/fireredaudio`), opened by rlx
//! alone, transcribes as the official code does: greedy, the reference's
//! ids (`tests/fixtures/asr_reference`, from
//! `scripts/fireredaudio_reference.py`) — the harness clip (one chunk) and
//! the clip seven times over (two chunks: the padding the encoder masks).
//! The crate's runner is held to the same ids first, then dropped (the
//! backbone in f32 is 36 GB), then the package runs.

#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use rlx_fireredaudio::FireRedRunner;
use rlx_fireredaudio::export::{ExportOptions, export_rlxp};
use serde_json::{Value, json};

fn case(root: &Path, name: &str) -> Value {
    serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/asr_reference").join(name)).unwrap(),
    )
    .unwrap()
}

#[test]
fn an_exported_fireredaudio_transcribes_as_the_reference_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("../../weights/asr/fireredaudio");
    if !dir.join("FireRedAudio/config.json").is_file() && !dir.join("config.json").is_file() {
        eprintln!("skip: no FireRedAudio checkpoint");
        return;
    }
    rlx_text::install_model_tokenizer();
    let clip =
        rlx_whisper::load_wav_mono_f32(&root.join("../../assets/harness/jfk_16k.wav")).unwrap();
    let cases: Vec<(Value, Vec<f32>)> = ["jfk_16k.json", "jfk_16k_x7.json"]
        .iter()
        .map(|n| {
            let c = case(&root, n);
            let pcm = clip.repeat(c["repeat"].as_u64().unwrap() as usize);
            (c, pcm)
        })
        .collect();
    let want = |c: &Value| -> Vec<u32> {
        let mut ids: Vec<u32> = serde_json::from_value(c["greedy_ids"].clone()).unwrap();
        // The reference's ids end with its stop token; the runner's do not.
        ids.retain(|&t| t != 248_044);
        ids
    };

    {
        let runner = FireRedRunner::builder().weights(&dir).build().unwrap();
        for (c, pcm) in &cases {
            let got = runner.asr_pcm(pcm).unwrap();
            eprintln!("runner: {:?}", got.answer);
            assert_eq!(got.token_ids, want(c), "the runner on {}", c["repeat"]);
        }
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("fireredaudio.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (c, pcm) in &cases {
        let got = m.transcribe(pcm, 16_000, &json!({})).unwrap();
        eprintln!("package: {got:?}");
        assert_eq!(
            got,
            c["greedy"].as_str().unwrap().trim(),
            "the package on {}",
            c["repeat"]
        );
    }
}
