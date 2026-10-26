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

//! An exported Fish Audio Drama 3 (`weights/tts/fish-drama-3`), opened by
//! rlx alone, speaks as the crate's runner does: its codes, until a near
//! tie between the runner's graphs and host experts and the package's, and
//! the codec's speech on the runner's codes.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_fish::codec::FishCodec;
use rlx_fish::drama3::{SynthOptions, synthesize_codes};
use rlx_fish::drama3_export::export_rlxp;
use rlx_fish::{DramaModel, DramaOptions};
use rlx_runtime::Device;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_drama3_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/fish-drama-3");
    if !dir.join("codec.pth").is_file() {
        eprintln!("skip: no Drama 3 checkpoint");
        return;
    }
    rlx_text::install_model_tokenizer();
    let text = "Hello there, how are you today?";
    let frames = 8;
    let want = {
        let mut model = DramaModel::open(&dir, Device::Cpu, DramaOptions::default()).unwrap();
        let enc = rlx_fish::prompt::load_tokenizer(&dir).unwrap();
        let opts = SynthOptions {
            max_new_tokens: frames,
            ..SynthOptions::default()
        };
        synthesize_codes(&mut model, &enc, text, &[], &opts).unwrap()
    };

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("fish-drama-3.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let got = m
        .run(
            &[],
            &json!({"entry": "speak", "text": text, "max_new_tokens": frames}),
        )
        .unwrap();
    let flat: Vec<u32> = got[0]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect();
    let n = want[0].len();
    let f = flat.len() / want.len();
    // Frame by frame: all ten codes the same.
    let same = (0..n.min(f))
        .take_while(|&t| (0..want.len()).all(|i| flat[i * f + t] == want[i][t]))
        .count();
    eprintln!("{same} of {n} frames the runner's ({f} from the package)");
    assert!(same >= 4, "{same} frames the same");

    // The codec on the runner's codes.
    let samples = FishCodec::open(&dir, Device::Cpu)
        .unwrap()
        .decode(&want)
        .unwrap();
    let audio = m.synthesize(text, &json!({"codes": want})).unwrap();
    assert_eq!(audio.sample_rate, 44_100);
    let cos = cosine(&audio.samples, &samples);
    eprintln!(
        "{} samples ({} from the package), cosine {cos}",
        samples.len(),
        audio.samples.len()
    );
    assert_eq!(audio.samples.len(), samples.len());
    assert!(cos > 0.9999, "cosine {cos}");
}
