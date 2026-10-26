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

//! An exported ChatterBox (`weights/tts/chatterbox`), opened by rlx alone,
//! speaks in its default voice as `NativeChatterBox` does with the native
//! T3 language model (`RLX_CB_NATIVE_LM`), sampled and greedy: the same
//! speech tokens (the runner's `RLX_CB_DUMP_TOKENS`), the same mel before
//! the vocoder (`RLX_DUMP_MEL`), the same speech.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_chatterbox::common::SynthOpts;
use rlx_chatterbox::export::export_rlxp;
use rlx_chatterbox::native::NativeChatterBox;
use rlx_core::export::ExportOptions;
use rlx_runtime::Device;
use serde_json::json;

const TOKENS: &str = "/tmp/cb_tokens_cpu.txt";
const MEL: &str = "/tmp/cb_mel_runner.f32";

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn an_exported_chatterbox_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/chatterbox");
    if !dir.join("native/t3_lm.safetensors").is_file() || !dir.join("onnx").is_dir() {
        eprintln!("skip: no ChatterBox ONNX or native T3");
        return;
    }
    rlx_ir::env::set("RLX_CB_NATIVE_LM", "1");
    rlx_ir::env::set("RLX_CB_DUMP_TOKENS", "1");
    rlx_ir::env::set("RLX_DUMP_MEL", MEL);
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("chatterbox.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    let runner = NativeChatterBox::load_on(&dir, Device::Cpu).unwrap();
    let mut voice = hound::WavReader::open(dir.join("default_voice.wav")).unwrap();
    let sr = voice.spec().sample_rate;
    let reference: Vec<f32> = voice.samples::<f32>().map(Result::unwrap).collect();
    let text = "Good morning, the tea is ready.";
    let frames = 40;
    for greedy in [false, true] {
        let opts = SynthOpts {
            max_frames: frames,
            greedy,
            ..SynthOpts::default()
        };
        let options = json!({"max_frames": frames, "greedy": greedy});
        let want = runner.synthesize(text, &reference, sr, &opts).unwrap();

        // The speech tokens.
        let want_tokens: Vec<i32> = std::fs::read_to_string(TOKENS)
            .unwrap()
            .split(',')
            .map(|t| t.trim().parse().unwrap())
            .collect();
        let mut run = options.clone();
        run["entry"] = json!("speak");
        run["text"] = json!(text);
        let got = m.run(&[], &run).unwrap();
        let got: Vec<i32> = got[0]
            .data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(got, want_tokens, "greedy {greedy}: the speech tokens");

        // The flow's mel.
        run["stage"] = json!("mel");
        let mel = f32s(&m.run(&[], &run).unwrap()[0].data);
        let want_mel = f32s(&std::fs::read(MEL).unwrap());
        assert_eq!(mel.len(), want_mel.len());
        let mel_cos = cosine(&mel, &want_mel);

        // The speech.
        let got = m.synthesize(text, &options).unwrap();
        assert_eq!(got.sample_rate, 24_000);
        assert_eq!(got.samples.len(), want.len(), "greedy {greedy}");
        let cos = cosine(&got.samples, &want);
        eprintln!(
            "greedy {greedy}: {} tokens the runner's, mel cosine {mel_cos}, {} samples cosine {cos}",
            want_tokens.len(),
            want.len()
        );
        assert!(mel_cos > 0.99999, "mel cosine {mel_cos}");
        assert!(cos > 0.9999, "cosine {cos}");
    }
}
