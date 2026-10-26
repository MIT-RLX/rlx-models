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

//! An exported Qwen2.5-VL package describes an image as `transformers`
//! does. Opt-in: `RLX_TEST_QWEN25_VL_DIR=<a Qwen2.5-VL checkpoint
//! directory>`, `RLX_TEST_VL_IMAGE=<448×448 RGB bytes>` and
//! `RLX_TEST_VL_REF=<file>`: JSON `{prompt, text}`, the chat-templated
//! prompt (one `<|image_pad|>`) and what `generate(max_new_tokens=24,
//! do_sample=False)` says in f32. Made with a still image's temporal
//! position its start, as Qwen2.5-VL's own code has it (`second_per_grid_t
//! = 0` for images): transformers 5.3.0 multiplies it by
//! `tokens_per_second`, and its text then parts from both after 10 tokens.
//! `RLX_TEST_VL_IMAGE_RESIZED=<RGB bytes>`: an image of another size (the
//! reference's `resized: {width, height, text}`), which the package resizes
//! as the processor does and runs at its own grid. `RLX_TEST_VL_PACKAGE=
//! <path>` keeps the package there (and reuses it).

use rlx_qwen3::export::ExportOptions;
use rlx_qwen25_vl::export::{VlExportOptions, export_rlxp};
use rlx_runtime::model::Tensor;
use serde_json::{Value, json};

#[test]
fn an_exported_qwen25_vl_package_describes_an_image_as_transformers_does() {
    let (Some(dir), Some(image), Some(reference)) = (
        rlx_ir::env::var_os("RLX_TEST_QWEN25_VL_DIR"),
        rlx_ir::env::var_os("RLX_TEST_VL_IMAGE"),
        rlx_ir::env::var_os("RLX_TEST_VL_REF"),
    ) else {
        eprintln!("skip: set RLX_TEST_QWEN25_VL_DIR, RLX_TEST_VL_IMAGE and RLX_TEST_VL_REF");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let reference: Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let pixels = std::fs::read(image).unwrap();
    assert_eq!(pixels.len(), 448 * 448 * 3);

    let tmp = tempfile::tempdir().unwrap();
    let out = rlx_ir::env::var_os("RLX_TEST_VL_PACKAGE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| tmp.path().join("qwen25vl.rlxp"));
    // Fewer graphs to build from the 3B checkpoint: no batched prompts, no
    // speculative check.
    let opts = ExportOptions {
        uppers: vec![512, 1024],
        prefill: vec![64, 256],
        prefill_batches: Vec::new(),
        speculate: 0,
        ..ExportOptions::default()
    };
    if !out.exists() {
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
    let opts = json!({"max_tokens": 24, "temperature": 0});
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
    // `RLX_TEST_VL_VIDEO=<frames, RGB bytes>` with `RLX_TEST_VL_VIDEO_REF`
    // (`{prompt, text, tokens, logits, frames, height, width}`, made at 2
    // fps, the prompt one `<|video_pad|>`): the video path, frame pairs
    // through the tower and one run of rows, against transformers. Made
    // with pair `g` at time position `start + g · interval`, as Qwen's own
    // code has it (transformers 5.3 gives every pair `start · interval`),
    // and its text by full forwards: transformers 5.3's cached decode
    // parts from its own full forward after a video.
    if let (Some(path), Some(r)) = (
        rlx_ir::env::var_os("RLX_TEST_VL_VIDEO"),
        rlx_ir::env::var_os("RLX_TEST_VL_VIDEO_REF"),
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
        let tok = rlx_text::load_tokenizer(&dir.join("tokenizer.json")).unwrap();
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
    // Text alone afterwards: positions are rows again.
    let text = m
        .generate_text(
            "The capital of France is",
            &json!({"max_tokens": 4, "temperature": 0}),
        )
        .unwrap();
    eprintln!("text: {text:?}");
    assert!(text.contains("Paris"), "{text}");
}
