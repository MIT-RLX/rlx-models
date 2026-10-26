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

//! The quantized release (`LiquidAI/d1-3B-GGUF`) against two references.
//!
//! * **LiquidAI's code on the dequantized GGUF** —
//!   `fixtures/d1_reference_<quant>.json`, from `scripts/d1_reference.py
//!   --gguf`: the HF model with every text weight replaced by the GGUF's,
//!   dequantized, in fp32. That is the quantized model's function with float
//!   activations, which is what rlx computes (packed weights,
//!   `DequantMatMul`, f32 activations), so it is compared tightly.
//! * **llama.cpp on the same file** — `fixtures/d1_llama_<quant>.json`, from
//!   `scripts/d1_llama_reference.py` (Metal). llama.cpp's kernels round
//!   differently (and its CPU path quantizes activations to 8 bits), and its
//!   clip.cpp preprocesses pictures on its own, so only the picks must agree
//!   and the gap is reported. On `d1-3B-Q4_K_M` rlx is within 5e-6 of the
//!   dequantized reference where llama.cpp is 3e-3 from it.
//!
//! Env-gated; skipped when unset: `RLX_D1_GGUF_DIR` (`just fetch-d1-gguf`),
//! `RLX_D1_QUANT` (default `Q4_K_M`), and as in `real_weights.rs`
//! `RLX_D1_IMAGES`, `RLX_D1_REF_DIR` (the dumps of the dequantized run — the
//! pictures as Pillow decoded them), `RLX_D1_DEVICE`.

use std::path::PathBuf;

use rlx_d1::vision::{self, RgbImage, VisionEncoder};
use rlx_d1::{D1, Json, OpenOptions};
use rlx_runtime::Device;

fn env_path(key: &str) -> Option<PathBuf> {
    rlx_ir::env::var(key)
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// `RLX_D1_WEIGHT_STORAGE` (`auto` by default): how dense weights are stored.
fn storage() -> rlx_d1::WeightStorage {
    rlx_ir::env::var("RLX_D1_WEIGHT_STORAGE")
        .map(|s| s.parse().expect("RLX_D1_WEIGHT_STORAGE"))
        .unwrap_or_default()
}

fn device() -> Device {
    rlx_ir::env::var("RLX_D1_DEVICE")
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_D1_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn quant() -> String {
    rlx_ir::env::var("RLX_D1_QUANT").unwrap_or_else(|| "Q4_K_M".into())
}

fn fixture(prefix: &str) -> Option<Json> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "tests/fixtures/{prefix}_{}.json",
        quant().to_ascii_lowercase()
    ));
    match std::fs::read_to_string(&p) {
        Ok(t) => Some(Json::parse(&t).expect("fixture parses")),
        Err(_) => {
            eprintln!("skip: no {}", p.display());
            None
        }
    }
}

fn f(v: &Json) -> f64 {
    match v {
        Json::Float(x) => *x,
        Json::Int(i) => *i as f64,
        Json::UInt(u) => *u as f64,
        other => panic!("not a number: {other:?}"),
    }
}

/// `(key, value)` for every probability in an `answers` object; a noul is
/// its P(yes).
fn probs(answers: &Json) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for (q, a) in answers.as_object().unwrap() {
        if let Some(p) = a.get("noul") {
            out.push((format!("{q}.yes"), f(p)));
        }
        if let Some(ps) = a.get("probabilities") {
            for (o, p) in ps.as_object().unwrap() {
                out.push((format!("{q}.{o}"), f(p)));
            }
        }
    }
    out
}

fn max_diff(a: &Json, b: &Json) -> f64 {
    let (a, b) = (probs(a), probs(b));
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(&b)
        .map(|((ka, x), (kb, y))| {
            assert_eq!(ka, kb);
            (x - y).abs()
        })
        .fold(0.0, f64::max)
}

fn picks(answers: &Json) -> Vec<String> {
    answers
        .as_object()
        .unwrap()
        .iter()
        .map(|(q, a)| match a.get("choice") {
            Some(c) => format!("{q}={}", c.as_str().unwrap()),
            None => format!("{q}={:.0}", f(a.get("noul").or(a.get("score")).unwrap())),
        })
        .collect()
}

/// A case's pictures; with `RLX_D1_REF_DIR`, as the reference decoded them.
fn images_of(case: &Json) -> Option<Vec<RgbImage>> {
    let names = case.get("images").and_then(Json::as_array).unwrap_or(&[]);
    if names.is_empty() {
        return Some(Vec::new());
    }
    let name = case.get("name").unwrap().as_str().unwrap();
    if names.len() == 1
        && let Some(refs) = env_path("RLX_D1_REF_DIR")
        && let Ok(bytes) = std::fs::read(refs.join(format!("vision_{name}.safetensors")))
    {
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let size: Vec<usize> = st
            .tensor("capped_size")
            .unwrap()
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as usize)
            .collect();
        let rgb = st.tensor("capped_rgb").unwrap().data().to_vec();
        return Some(vec![RgbImage::new(size[0], size[1], rgb).unwrap()]);
    }
    let dir = env_path("RLX_D1_IMAGES")?;
    Some(
        names
            .iter()
            .map(|n| RgbImage::open(&dir.join(n.as_str().unwrap())).unwrap())
            .collect(),
    )
}

/// Run every case through `d1`, yielding `(case, answers)`.
fn run_cases(d1: &mut D1, mut each: impl FnMut(&str, &Json, bool)) {
    let cases = Json::parse(include_str!("fixtures/d1_cases.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let name = case.get("name").unwrap().as_str().unwrap();
        let Some(images) = images_of(case) else {
            eprintln!("skip {name}: set RLX_D1_IMAGES");
            continue;
        };
        let out = d1
            .system_one(
                case.get("state").unwrap(),
                case.get("questions").unwrap(),
                &images,
            )
            .expect("system_one");
        each(name, out.get("answers").unwrap(), !images.is_empty());
    }
}

fn find<'a>(cases: &'a Json, name: &str) -> &'a Json {
    cases
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c.get("name").unwrap().as_str() == Some(name))
        .unwrap()
        .get("system_one")
        .unwrap()
        .get("answers")
        .unwrap()
}

#[test]
fn quantized_matches_the_dequantized_reference() {
    let Some(dir) = env_path("RLX_D1_GGUF_DIR") else {
        eprintln!("skip: set RLX_D1_GGUF_DIR to the LiquidAI/d1-3B-GGUF directory");
        return;
    };
    let Some(reference) = fixture("d1_reference") else {
        return;
    };
    let fp32 = Json::parse(include_str!("fixtures/d1_reference.json")).unwrap();
    // The F16 mmproj reproduces the HF tower (see below); the reference ran
    // the HF tower.
    let opts = OpenOptions {
        prefer: Some(quant()),
        mmproj: Some(dir.join("mmproj-d1-3B-F16.gguf")).filter(|p| p.is_file()),
        weight_storage: storage(),
    };
    let mut d1 = D1::open_with(&dir, device(), &opts).expect("open d1 GGUF");
    let (mut worst, mut worst_fp32) = (0f64, 0f64);
    run_cases(&mut d1, |name, ours, _| {
        let d = max_diff(ours, find(&reference, name));
        let dq = max_diff(ours, find(&fp32, name));
        eprintln!(
            "{name:18} vs dequantized {}: max |dp| {d:.2e}   (quantization vs fp32: {dq:.2e})",
            quant()
        );
        worst = worst.max(d);
        worst_fp32 = worst_fp32.max(dq);
    });
    eprintln!(
        "{}: worst vs dequantized reference {worst:.2e}; quantization cost vs fp32 {worst_fp32:.2e}",
        quant()
    );
    assert!(
        worst < 2e-4,
        "answers differ from the dequantized reference by {worst:.2e}"
    );
}

#[test]
fn quantized_agrees_with_llama_cpp() {
    let Some(dir) = env_path("RLX_D1_GGUF_DIR") else {
        eprintln!("skip: set RLX_D1_GGUF_DIR");
        return;
    };
    let Some(llama) = fixture("d1_llama") else {
        return;
    };
    // The tower llama.cpp ran with (recorded in the fixture).
    let opts = OpenOptions {
        prefer: Some(quant()),
        mmproj: llama
            .get("mmproj")
            .and_then(Json::as_str)
            .map(|m| dir.join(m)),
        weight_storage: storage(),
    };
    let mut d1 = D1::open_with(&dir, device(), &opts).expect("open d1 GGUF");
    let cases = llama.get("cases").unwrap().clone();
    let (mut worst_text, mut worst_img) = (0f64, 0f64);
    run_cases(&mut d1, |name, ours, pictures| {
        let theirs = find(&cases, name);
        let d = max_diff(ours, theirs);
        eprintln!("{name:18} vs llama.cpp {}: max |dp| {d:.2e}", quant());
        assert_eq!(
            picks(ours),
            picks(theirs),
            "{name}: picks differ from llama.cpp"
        );
        if pictures {
            worst_img = worst_img.max(d);
        } else {
            worst_text = worst_text.max(d);
        }
    });
    eprintln!(
        "{}: worst vs llama.cpp {worst_text:.2e} (text), {worst_img:.2e} (pictures)",
        quant()
    );
    assert!(
        worst_text < 5e-3,
        "text answers differ from llama.cpp by {worst_text:.2e}"
    );
}

/// The mmproj tower (names mapped back to HF, the conv-shaped patch
/// embedding unpermuted) against the fp32 HF features dumped by
/// `scripts/d1_reference.py` — only the F16/Q8_0 weight rounding apart.
#[test]
fn mmproj_tower_matches_the_hf_tower() {
    let (Some(dir), Some(refs)) = (env_path("RLX_D1_GGUF_DIR"), env_path("RLX_D1_REF_DIR")) else {
        eprintln!("skip: needs RLX_D1_GGUF_DIR and RLX_D1_REF_DIR");
        return;
    };
    for mm in ["mmproj-d1-3B-F16.gguf", "mmproj-d1-3B-Q8_0.gguf"] {
        let path = dir.join(mm);
        if !path.is_file() {
            eprintln!("skip {mm}: not downloaded");
            continue;
        }
        let mut enc =
            VisionEncoder::load_mmproj(&path, Some(&dir), device(), storage()).expect("mmproj");
        let cfg = enc.processor().clone();
        for name in ["cats_image", "shapes_tiled"] {
            let Ok(bytes) = std::fs::read(refs.join(format!("vision_{name}.safetensors"))) else {
                continue;
            };
            let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let size: Vec<usize> = st
                .tensor("capped_size")
                .unwrap()
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as usize)
                .collect();
            let img = RgbImage::new(
                size[0],
                size[1],
                st.tensor("capped_rgb").unwrap().data().to_vec(),
            )
            .unwrap();
            let got = enc
                .encode(&vision::process_image(&img, &cfg).unwrap())
                .unwrap();
            let want: Vec<f32> = st
                .tensor("image_features")
                .unwrap()
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(got.len(), want.len());
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (a, b) in got.iter().zip(&want) {
                dot += *a as f64 * *b as f64;
                na += (*a as f64).powi(2);
                nb += (*b as f64).powi(2);
            }
            let cos = dot / (na.sqrt() * nb.sqrt());
            eprintln!("{mm:26} {name:14} features cos vs HF fp32 {cos:.6}");
            // A wrong rename or patch permutation lands near 0, not 0.99.
            assert!(cos > 0.995, "{mm} {name}: cos {cos}");
        }
    }
}
