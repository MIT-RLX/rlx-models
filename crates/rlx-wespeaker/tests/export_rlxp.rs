// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! WeSpeaker exported as a package embeds what `WeSpeaker::embed_pcm` does:
//! the real ResNet34-LM (`weights/wespeaker-voxceleb-resnet34-LM`, or
//! `RLX_WESPEAKER_DIR`; skipped where absent), the JFK clip cut shorter
//! than the network's window, about as long and longer.

#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_runtime::model::Tensor;
use rlx_wespeaker::WeSpeaker;
use serde_json::json;

#[test]
fn an_exported_wespeaker_embeds_what_the_runner_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let Some(dir) = rlx_wespeaker::resolve_model_dir(&[&root.join("weights")]) else {
        eprintln!("skipped: no WeSpeaker model");
        return;
    };
    let mut r = hound::WavReader::open(root.join("assets/harness/jfk_16k.wav")).unwrap();
    let pcm: Vec<f32> = r
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("wespeaker.rlxp");
    let report = rlx_wespeaker::export::export_rlxp(&dir, &out, &Default::default()).unwrap();
    eprintln!("{report}");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut spk = WeSpeaker::open(&dir).unwrap();
    for n in [16_000, 24_160, 70_001] {
        let clip = &pcm[..n];
        let want = spk.embed_pcm(clip).unwrap();
        let bytes = clip.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = Tensor::new("audio", rlx_ir::DType::F32, vec![n], bytes);
        let got = m.run(&[input], &json!({})).unwrap()[0].to_f32().unwrap();
        let cos = rlx_wespeaker::cosine(&got, &want);
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("{n} samples: cosine {cos}, off by {worst}");
        assert!(cos > 0.99999, "{n} samples: cosine {cos}");
    }
}
