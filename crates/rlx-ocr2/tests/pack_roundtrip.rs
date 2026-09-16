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

//! `.rlxp` roundtrip: packing the asset directory and loading from the package must
//! reproduce the loose-file path *bit for bit* — same weights, same graphs, same output.
//! Env-gated so CI without assets skips. Set:
//!   OCR2_ASSETS   = dir with detector_recipe.json + {detector,recognizer}.safetensors
//!                   + codemap.txt (+ optional lexicon.tsv / ngram.bin)
//!   OCR2_FIXTURES = dir with rec_input.bin + det_input.bin

#![cfg(feature = "rlxp")]

mod common;
use common::read_f32;
use rlx_ocr2::{ContainerKind, Detector, Ocr2Pack, Recognizer, write_pack};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Self-deleting temp path for the package under test.
struct TempPath(PathBuf);

impl Drop for TempPath {
    fn drop(&mut self) {
        if self.0.is_dir() {
            let _ = std::fs::remove_dir_all(&self.0);
        } else {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

#[test]
fn pack_roundtrip_matches_loose_files() {
    let (Some(assets), Some(fixtures)) = (env_dir("OCR2_ASSETS"), env_dir("OCR2_FIXTURES")) else {
        eprintln!("OCR2_ASSETS / OCR2_FIXTURES unset — skipping");
        return;
    };
    let device = Device::Cpu;

    let rec_input = read_f32(fixtures.join("rec_input.bin"));
    let det_input = read_f32(fixtures.join("det_input.bin"));
    let width = rec_input.len() / 32;

    // Reference: the loose-file loaders.
    let rec_ref = Recognizer::load(
        &assets.join("recognizer.safetensors"),
        &assets.join("codemap.txt"),
        device,
    )
    .unwrap();
    let want_logits = rec_ref.forward_logits(&rec_input, width).unwrap();
    let det_ref = Detector::load(
        &assets.join("detector_recipe.json"),
        &assets.join("detector.safetensors"),
        device,
    )
    .unwrap();
    let want_heads = det_ref.forward(&det_input).unwrap();

    for (label, container, ext) in [
        ("flat", ContainerKind::Flat, "rlxp"),
        ("zip", ContainerKind::Zip, "zip"),
        ("dir", ContainerKind::Dir, "dir"),
    ] {
        let out = TempPath(std::env::temp_dir().join(format!(
            "rlx-ocr2-roundtrip-{}-{label}.{ext}",
            std::process::id()
        )));
        write_pack(&assets, &out.0, container).unwrap();

        let pack = Ocr2Pack::open(&out.0).unwrap();
        assert_eq!(
            pack.recipe_json,
            std::fs::read_to_string(assets.join("detector_recipe.json")).unwrap(),
            "[{label}] recipe sidecar differs"
        );

        let got_logits = pack
            .recognizer(device)
            .unwrap()
            .forward_logits(&rec_input, width)
            .unwrap();
        assert_eq!(
            got_logits, want_logits,
            "[{label}] recognizer logits differ from the loose-file path"
        );

        let got_heads = pack.detector(device, Vec::new()).unwrap();
        let got_heads = got_heads.forward(&det_input).unwrap();
        assert_eq!(got_heads.len(), want_heads.len(), "[{label}] head count");
        for ((h_got, d_got), (h_want, d_want)) in got_heads.iter().zip(&want_heads) {
            assert_eq!(h_got, h_want, "[{label}] head order");
            assert_eq!(d_got, d_want, "[{label}] head {h_got} differs");
        }
        println!(
            "[{label:4}] {} tensors, heads {} — exact",
            300,
            got_heads.len()
        );
    }
}

/// The correction assets survive the roundtrip when present.
#[test]
fn pack_carries_correction_assets() {
    let Some(assets) = env_dir("OCR2_ASSETS") else {
        eprintln!("OCR2_ASSETS unset — skipping");
        return;
    };
    let has_lm = assets.join("ngram.bin").is_file() || assets.join("lexicon.tsv").is_file();

    let out =
        TempPath(std::env::temp_dir().join(format!("rlx-ocr2-lm-{}.rlxp", std::process::id())));
    write_pack(&assets, &out.0, ContainerKind::Flat).unwrap();
    let pack = Ocr2Pack::open(&out.0).unwrap();
    assert_eq!(
        pack.has_rescorer(),
        has_lm,
        "correction assets lost/invented"
    );
    assert_eq!(pack.rescorer().unwrap().is_some(), has_lm);

    if has_lm {
        // The n-gram model must score identically whether mmap'd or unpacked from bytes.
        let want = rlx_ocr2::NgramModel::load(Path::new(&assets.join("ngram.bin"))).unwrap();
        let bytes = std::fs::read(assets.join("ngram.bin")).unwrap();
        let got = rlx_ocr2::NgramModel::from_bytes(&bytes).unwrap();
        assert_eq!(got.order, want.order);
        for tok in 0u32..80 {
            assert_eq!(got.cond(&[], tok), want.cond(&[], tok), "unigram {tok}");
            assert_eq!(
                got.cond(&[1, 2], tok),
                want.cond(&[1, 2], tok),
                "trigram {tok}"
            );
        }
    }
}
