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

//! An exported TEN-VAD package, opened by rlx alone, scores a clip as
//! `TenVad::probabilities_i16` does: the fixture's 4 s of speech-like audio
//! (`tests/fixtures`), part of it, and it repeated past the LSTM reset —
//! its DSP (the pitch track's two recurrences as `Op::Scan`) and its
//! network in the package.
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_runtime::model::Tensor;
use rlx_ten_vad::export::export_rlxp;
use rlx_ten_vad::{TenVad, TenVadConfig};
use serde_json::json;

#[test]
fn an_exported_ten_vad_scores_what_the_runner_does() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let raw = std::fs::read(dir.join("synthetic_speech_16k.pcm")).unwrap();
    let pcm: Vec<i16> = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("ten-vad.rlxp");
    eprintln!(
        "{}",
        export_rlxp(
            &tmp.path().join("embedded"),
            &out,
            &ExportOptions::default()
        )
        .unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut vad = TenVad::new(TenVadConfig::default()).unwrap();
    // The clip, then 37 hops of it, then it eight times over: 2000 hops,
    // past the runner's LSTM reset at 1875.
    let long: Vec<i16> = pcm.iter().copied().cycle().take(8 * pcm.len()).collect();
    for frames in [pcm.len() / 256, 37, long.len() / 256] {
        let clip = &long[..frames * 256];
        let want = vad.probabilities_i16(clip).unwrap();
        vad.reset();
        let x: Vec<f32> = clip.iter().map(|&s| s as f32).collect();
        let got = m
            .run(&[Tensor::f32("pcm", vec![x.len()], &x)], &json!({}))
            .unwrap()[0]
            .to_f32()
            .unwrap();
        assert_eq!(got.len(), want.len());
        let diffs: Vec<f32> = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).collect();
        let worst = diffs.iter().fold(0f32, |a, &d| a.max(d));
        let off = diffs.iter().filter(|&&d| d > 1e-4).count();
        let flips = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
            .count();
        eprintln!("{frames} hops: off by {worst}; {off} above 1e-4; {flips} decisions differ");
        assert!(worst < 1e-4, "{frames} hops: off by {worst}");
        assert_eq!(flips, 0, "{frames} hops");
    }
}
