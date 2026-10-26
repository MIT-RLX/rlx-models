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

//! An exported sanoTTS voice (`weights/tts/sanotts/amy-en-1p46m`), opened by
//! rlx alone, speaks as the crate's host reference does: the same samples,
//! for two texts and a slower length scale.

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_sanotts::Synthesizer;
use rlx_sanotts::export::export_rlxp;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_sanotts_voice_speaks_as_the_reference_does() {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/sanotts/amy-en-1p46m");
    if !dir.join("manifest.json").is_file() {
        eprintln!("skip: no sanoTTS voice (`just fetch-sanotts`)");
        return;
    }
    let synth = Synthesizer::load(&dir).unwrap();
    let cases = [
        ("Hello from a two megabyte voice.", None),
        ("The quick brown fox jumps over the lazy dog!", Some(1.3f32)),
    ];
    let want: Vec<Vec<f32>> = cases
        .iter()
        .map(|(t, ls)| {
            synth
                .synthesize_with(t, ls.unwrap_or(synth.default_length_scale()))
                .unwrap()
                .samples
        })
        .collect();

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("sanotts.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for ((text, ls), want) in cases.iter().zip(&want) {
        let opts = match ls {
            Some(l) => json!({"length_scale": l}),
            None => json!({}),
        };
        let audio = m.synthesize(text, &opts).unwrap();
        let diff = audio
            .samples
            .iter()
            .zip(want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let cos = cosine(&audio.samples, want);
        eprintln!(
            "{text:?}: {} samples ({} from the package), max |d| {diff}, cosine {cos}",
            want.len(),
            audio.samples.len()
        );
        assert_eq!(audio.samples.len(), want.len());
        assert!(diff < 1e-4, "max |d| {diff}");
    }
}
