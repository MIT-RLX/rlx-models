// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A MiniCPM-V package answers what `MiniCpmVRunner::chat` does, greedy: a
//! photo and a question, and a question alone — on MiniCPM-V-4.7-1B where
//! it is on disk (`weights/vision/minicpm-v-4.7-1b`, or `RLX_MINICPMV_DIR`).

#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use rlx_minicpm_v::RgbImage;
use rlx_minicpm_v::export::ExportOptions;
use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use serde_json::json;

fn dir() -> Option<PathBuf> {
    let p = rlx_ir::env::var("RLX_MINICPMV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/vision/minicpm-v-4.7-1b")
        });
    p.join("model.safetensors").exists().then_some(p)
}

#[test]
fn a_package_chats_as_minicpm_v_does() {
    let Some(dir) = dir() else {
        eprintln!("no MiniCPM-V checkpoint; skipped");
        return;
    };
    let photo =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../rlx-locateanything/fixtures/sample.jpg");
    let img = image::open(&photo).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let rgb = img.into_raw();
    let picture = RgbImage {
        width: w as usize,
        height: h as usize,
        rgb: rgb.clone(),
    };
    let n = 24;
    let turns: [(&str, bool); 2] = [
        ("Describe this image in one sentence.", true),
        ("What is the capital of France?", false),
    ];
    let mut wants = Vec::new();
    {
        let mut r = rlx_minicpm_v::MiniCpmVRunner::builder()
            .model_dir(&dir)
            .device(Device::Cpu)
            .build()
            .unwrap();
        for (q, with) in &turns {
            let images: Vec<RgbImage> = if *with { vec![picture.clone()] } else { vec![] };
            wants.push(
                r.chat(
                    &images,
                    q,
                    n,
                    rlx_qwen35::SampleOpts::greedy(),
                    false,
                    |_| {},
                )
                .unwrap(),
            );
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("minicpmv.rlxp");
    let opts = ExportOptions {
        max_context: Some(2048),
        ..ExportOptions::default()
    };
    let report = rlx_minicpm_v::export::export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!("{report}");
    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for ((q, with), want) in turns.iter().zip(&wants) {
        let inputs: Vec<Tensor> = if *with {
            vec![Tensor::new(
                "image0",
                rlx_ir::DType::U8,
                vec![h as usize, w as usize, 3],
                rgb.clone(),
            )]
        } else {
            vec![]
        };
        let outs = m
            .run(&inputs, &json!({"prompt": q, "max_tokens": n}))
            .unwrap_or_else(|e| panic!("{q:?}: {e:#}"));
        let got = String::from_utf8(outs[0].data.clone()).unwrap();
        eprintln!("{q:?}\n  runner:  {want:?}\n  package: {got:?}");
        assert_eq!(&got, want, "{q:?}");
    }
}
