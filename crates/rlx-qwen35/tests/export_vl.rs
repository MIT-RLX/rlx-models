// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A Qwen3.5 vision-language package answers what
//! `Qwen35Runner::generate_multimodal_with_opts` does, greedy, on a photo —
//! on Qwen3.5-0.8B where it is on disk (`weights/lm/qwen3.5-0.8b`, or
//! `RLX_QWEN35_HF_DIR`).

#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use rlx_core::export::ExportOptions;
use rlx_qwen35::{Qwen35Runner, SampleOpts};
use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use serde_json::json;

fn dir() -> Option<PathBuf> {
    let p = rlx_ir::env::var("RLX_QWEN35_HF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/lm/qwen3.5-0.8b")
        });
    p.join("config.json").exists().then_some(p)
}

#[test]
fn a_package_answers_about_a_photo_as_the_runner_does() {
    let Some(dir) = dir() else {
        eprintln!("no Qwen3.5 HF checkpoint; skipped");
        return;
    };
    let photo =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../rlx-locateanything/fixtures/sample.jpg");
    let img = image::open(&photo).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let rgb = img.into_raw();
    let prompt = "<|im_start|>user\n<__media__>Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let n = 24;
    let tok = dir.join("tokenizer.json");
    let ck = rlx_core::export::Checkpoint::open(&dir).unwrap();
    let stop: Vec<u32> = ck.stop_ids().into_iter().map(|v| v as u32).collect();
    let want = {
        let mut r = Qwen35Runner::builder()
            .weights(&dir)
            .device(Device::Cpu)
            .max_seq(2048)
            .build()
            .unwrap();
        let mut ids = r
            .generate_multimodal_with_opts(
                prompt,
                &rgb,
                w as usize,
                h as usize,
                Some(&tok),
                n,
                SampleOpts::greedy(),
                |t| !stop.contains(&t),
            )
            .unwrap();
        if ids.last().is_some_and(|t| stop.contains(t)) {
            ids.pop();
        }
        let t = tokenizers::Tokenizer::from_file(&tok).unwrap();
        t.decode(&ids, true).unwrap()
    };

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("qwen35-vl.rlxp");
    let opts = ExportOptions {
        max_context: Some(2048),
        ..ExportOptions::default()
    };
    let report = rlx_qwen35::export_vl::export_vl_rlxp(&dir, &out, &opts).unwrap();
    eprintln!("{report}");
    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let image = Tensor::new(
        "image0",
        rlx_ir::DType::U8,
        vec![h as usize, w as usize, 3],
        rgb,
    );
    let outs = m
        .run(&[image], &json!({"prompt": prompt, "max_tokens": n}))
        .unwrap();
    let got = String::from_utf8(outs[0].data.clone()).unwrap();
    eprintln!("runner:  {want:?}\npackage: {got:?}");
    assert_eq!(got, want);
}
