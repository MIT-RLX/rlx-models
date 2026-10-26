// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! A Conformer-CTC `.nemo` exported as a package transcribes what
//! `ConformerCtc` does: the real small English model
//! (`.cache/conformer-ctc`, `just fetch-conformer-ctc`, or
//! `RLX_CONFORMER_CTC_NEMO`; skipped where absent), the JFK clip and a cut of
//! it that is no whole number of frames.

#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_conformer_ctc::{ConformerCtc, wav};
use rlx_runtime::Device;
use serde_json::json;

fn nemo() -> Option<PathBuf> {
    let p = rlx_ir::env::var("RLX_CONFORMER_CTC_NEMO")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../.cache/conformer-ctc/stt_en_conformer_ctc_small.nemo")
        });
    p.is_file().then_some(p)
}

#[test]
fn an_exported_conformer_ctc_transcribes_what_the_runner_does() {
    let Some(path) = nemo() else {
        eprintln!("skipped: no Conformer-CTC checkpoint");
        return;
    };
    let clip = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/harness/jfk_16k.wav");
    let w = wav::parse(&std::fs::read(&clip).unwrap()).unwrap();
    assert_eq!(w.sample_rate, 16_000);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("conformer-ctc.rlxp");
    let report = rlx_conformer_ctc::export::export_rlxp(&path, &out, &Default::default()).unwrap();
    eprintln!("{report}");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = ConformerCtc::open(&path, Device::Cpu).unwrap();
    for n in [w.samples.len(), 77_777] {
        let pcm = &w.samples[..n];
        let want = runner.transcribe(pcm).unwrap();
        let got = m.transcribe(pcm, 16_000, &json!({})).unwrap();
        eprintln!("{n} samples:\n  runner:  {want}\n  package: {got}");
        assert_eq!(got, want, "{n} samples");
    }
}
