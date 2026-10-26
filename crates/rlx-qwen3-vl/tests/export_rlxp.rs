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

//! An exported Qwen3-VL package describes an image as `transformers` does.
//! Opt-in: `RLX_TEST_QWEN3_VL_DIR=<a dense Qwen3-VL checkpoint directory>`,
//! `RLX_TEST_VL_IMAGE=<448×448 RGB bytes>` and `RLX_TEST_VL3_REF=<file>`:
//! JSON `{prompt, text}`, the chat-templated prompt (one `<|image_pad|>`)
//! and what `generate(max_new_tokens=24, do_sample=False)` says in f32.
//! `RLX_TEST_VL_PACKAGE=<path>` keeps the package there (and reuses it).
//! `RLX_TEST_VL_IMAGE_RESIZED=<RGB bytes>`: an image of another size (the
//! reference's `resized: {width, height, text}`), which the package resizes
//! as the processor does and runs at its own grid.
//! Optional reference keys: `logits` (the prompt's last-position logits,
//! also under `resized`) and `plain_text` (greedy 4 tokens after "The
//! capital of France is"; without it the text must say Paris).

use rlx_qwen3::export::ExportOptions;
use rlx_qwen3_vl::export::{VlExportOptions, export_rlxp};
use rlx_runtime::model::Tensor;
use serde_json::{Value, json};

#[test]
fn an_exported_qwen3_vl_package_describes_an_image_as_transformers_does() {
    let (Some(dir), Some(image), Some(reference)) = (
        rlx_ir::env::var_os("RLX_TEST_QWEN3_VL_DIR"),
        rlx_ir::env::var_os("RLX_TEST_VL_IMAGE"),
        rlx_ir::env::var_os("RLX_TEST_VL3_REF"),
    ) else {
        eprintln!("skip: set RLX_TEST_QWEN3_VL_DIR, RLX_TEST_VL_IMAGE and RLX_TEST_VL3_REF");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let reference: Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let pixels = std::fs::read(image).unwrap();
    assert_eq!(pixels.len(), 448 * 448 * 3);

    let tmp = tempfile::tempdir().unwrap();
    let out = rlx_ir::env::var_os("RLX_TEST_VL_PACKAGE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| tmp.path().join("qwen3vl.rlxp"));
    if !out.exists() {
        let opts = ExportOptions {
            uppers: vec![512, 1024],
            prefill: vec![64, 256],
            prefill_batches: Vec::new(),
            speculate: 0,
            ..ExportOptions::default()
        };
        let t = std::time::Instant::now();
        let report = export_rlxp(&dir, &out, &opts, &VlExportOptions::default()).unwrap();
        eprintln!("export {:?}: {report:?}", t.elapsed());
        assert!(report.stateful_failure.is_none() && report.prefill_failure.is_none());
    }

    let device = rlx_ir::env::var_result("RLX_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    rlx_text::install_model_tokenizer();
    // `RLX_TEST_OPEN_OPTIONS='{"weight_dtype": "f16", "prefill": "release"}'`:
    // further open options (narrower weights, one executable at a time).
    let mut open = json!({"device": device});
    if let Ok(more) = std::env::var("RLX_TEST_OPEN_OPTIONS") {
        let more: Value = serde_json::from_str(&more).unwrap();
        for (k, v) in more.as_object().unwrap() {
            open[k] = v.clone();
        }
    }
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), open).unwrap();
    let image = Tensor::new("image", rlx_ir::DType::U8, vec![448, 448, 3], pixels);
    let prompt = reference["prompt"].as_str().unwrap();
    let want = reference["text"].as_str().unwrap();
    // As many tokens as the reference has (24 when it does not say).
    let n = reference["tokens"].as_array().map_or(24, Vec::len);
    let opts = json!({"max_tokens": n, "temperature": 0});
    let t = std::time::Instant::now();
    let got = m
        .generate_with_media(prompt, std::slice::from_ref(&image), &opts)
        .unwrap();
    eprintln!("{:?}: {got:?}", t.elapsed());
    assert_eq!(
        got.trim(),
        want.trim(),
        "the description differs from transformers'"
    );
    if let Some(path) = rlx_ir::env::var_os("RLX_TEST_VL_IMAGE_RESIZED") {
        let r = &reference["resized"];
        let w = r["width"].as_u64().unwrap() as usize;
        let h = r["height"].as_u64().unwrap() as usize;
        let pixels = std::fs::read(path).unwrap();
        assert_eq!(pixels.len(), w * h * 3);
        let image = Tensor::new("image", rlx_ir::DType::U8, vec![h, w, 3], pixels);
        let t = std::time::Instant::now();
        let got = m.generate_with_media(prompt, &[image], &opts).unwrap();
        eprintln!("{w}×{h}, {:?}: {got:?}", t.elapsed());
        assert_eq!(got.trim(), r["text"].as_str().unwrap().trim(), "at {w}×{h}");
    }
    // A reference with `logits` (the prompt's last-position logits): the
    // package's prefill gives them, for each image.
    let tok = rlx_text::load_tokenizer(&dir.join("tokenizer.json")).unwrap();
    let ids = tok.encode(prompt, false).unwrap();
    let mut images = vec![(image.clone(), &reference)];
    let resized_image = rlx_ir::env::var_os("RLX_TEST_VL_IMAGE_RESIZED").map(|path| {
        let r = &reference["resized"];
        let w = r["width"].as_u64().unwrap() as usize;
        let h = r["height"].as_u64().unwrap() as usize;
        Tensor::new(
            "image",
            rlx_ir::DType::U8,
            vec![h, w, 3],
            std::fs::read(path).unwrap(),
        )
    });
    if let Some(img) = &resized_image {
        images.push((img.clone(), &reference["resized"]));
    }
    for (img, r) in images {
        let Some(want) = r.get("logits").and_then(Value::as_array) else {
            continue;
        };
        let want: Vec<f32> = want.iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let (_, got) = m
            .lm()
            .unwrap()
            .prefill_with_media(&ids, std::slice::from_ref(&img))
            .unwrap();
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "{:?}: prefill logits within {diff} of transformers'",
            img.shape
        );
        assert!(diff < 2e-3, "{diff}");
    }
    // `RLX_TEST_VL_VIDEO=<frames, RGB bytes>` with `RLX_TEST_VL3_VIDEO_REF`
    // (`{prompt, text, tokens, logits, frames, height, width}`, the prompt
    // one `<|video_pad|>` per frame pair): the video path, frame pairs
    // through the tower, against transformers.
    if let (Some(path), Some(r)) = (
        rlx_ir::env::var_os("RLX_TEST_VL_VIDEO"),
        rlx_ir::env::var_os("RLX_TEST_VL3_VIDEO_REF"),
    ) {
        let r: Value = serde_json::from_slice(&std::fs::read(r).unwrap()).unwrap();
        let dim = |k: &str| r[k].as_u64().unwrap() as usize;
        let video = Tensor::new(
            "video",
            rlx_ir::DType::U8,
            vec![dim("frames"), dim("height"), dim("width"), 3],
            std::fs::read(path).unwrap(),
        );
        let prompt = r["prompt"].as_str().unwrap();
        let ids = tok.encode(prompt, false).unwrap();
        let want: Vec<f32> = r["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let (_, got) = m
            .lm()
            .unwrap()
            .prefill_with_media(&ids, std::slice::from_ref(&video))
            .unwrap();
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("video: prefill logits within {diff} of transformers'");
        assert!(diff < 2e-3, "{diff}");
        let n = r["tokens"].as_array().map_or(24, Vec::len);
        let got = m
            .generate_with_media(
                prompt,
                &[video],
                &json!({"max_tokens": n, "temperature": 0}),
            )
            .unwrap();
        eprintln!("video: {got:?}");
        assert_eq!(got.trim(), r["text"].as_str().unwrap().trim(), "the video");
    }
    let text = m
        .generate_text(
            "The capital of France is",
            &json!({"max_tokens": 4, "temperature": 0}),
        )
        .unwrap();
    eprintln!("text: {text:?}");
    // A reference may give transformers' greedy text for this prompt (a
    // random-weight checkpoint knows no capitals).
    match reference["plain_text"].as_str() {
        Some(want) => assert_eq!(text, want),
        None => assert!(text.contains("Paris"), "{text}"),
    }
}
