// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Silero VAD exported as a package scores and segments what the crate's
//! `SileroSession` and `speech_segments_silero` do: the embedded weights, the
//! JFK clip (its tail a partial frame) and the clip with a second of
//! silence spliced in.

#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_runtime::model::Tensor;
use rlx_vad::segments::{SegmentParams, speech_segments_silero};
use rlx_vad::silero::{SileroConfig, SileroSession, SileroWeights};
use serde_json::json;

#[test]
fn an_exported_silero_vad_gives_the_sessions_probabilities_and_segments() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (sr, pcm) = rlx_vad::load_wav_mono_f32(&root.join("assets/harness/jfk_16k.wav")).unwrap();
    assert_eq!(sr, 16_000);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("silero.rlxp");
    let report =
        rlx_vad::export::export_rlxp(std::path::Path::new(""), &out, &Default::default()).unwrap();
    eprintln!("{report}");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut spliced = pcm[..40_000].to_vec();
    spliced.extend(std::iter::repeat_n(0.0, 16_000));
    spliced.extend_from_slice(&pcm[40_000..]);
    for clip in [pcm.clone(), spliced] {
        let mut s = SileroSession::new(SileroWeights::embedded(), SileroConfig::default());
        let want: Vec<f32> = clip
            .chunks(s.frame_samples())
            .map(|c| s.predict_frame_padded(c).unwrap())
            .collect();
        let want_segs = speech_segments_silero(&mut s, &clip, &SegmentParams::silero()).unwrap();
        let bytes = clip.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = Tensor::new("audio", rlx_ir::DType::F32, vec![clip.len()], bytes);
        let outs = m.run(&[input], &json!({})).unwrap();
        let got = outs[0].to_f32().unwrap();
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let segs: Vec<i32> = outs[1]
            .data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let want_flat: Vec<i32> = want_segs
            .iter()
            .flat_map(|s| [s.start as i32, s.end as i32])
            .collect();
        eprintln!(
            "{} samples: {} frames, probabilities within {worst}; segments {segs:?}",
            clip.len(),
            got.len()
        );
        assert!(worst < 2e-5, "probabilities off by {worst}");
        assert_eq!(segs, want_flat);
    }
}

#[test]
fn an_exported_earshot_gives_the_detectors_probabilities_and_segments() {
    use rlx_vad::earshot::{Detector, FRAME_SAMPLES};
    use rlx_vad::segments::speech_segments_earshot;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (_, pcm) = rlx_vad::load_wav_mono_f32(&root.join("assets/harness/jfk_16k.wav")).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("earshot.rlxp");
    let report = rlx_vad::export::export_earshot(&out, &Default::default()).unwrap();
    eprintln!("{report}");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let clip = &pcm[..50_001];
    let mut det = Detector::default();
    let want: Vec<f32> = clip
        .chunks(FRAME_SAMPLES)
        .map(|c| {
            let mut f = vec![0f32; FRAME_SAMPLES];
            f[..c.len()].copy_from_slice(c);
            det.predict_f32(&f)
        })
        .collect();
    let want_segs = speech_segments_earshot(clip, &SegmentParams::earshot());
    let bytes = clip.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = Tensor::new("audio", rlx_ir::DType::F32, vec![clip.len()], bytes);
    let outs = m.run(&[input], &json!({})).unwrap();
    let got = outs[0].to_f32().unwrap();
    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(&want)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    let segs: Vec<i32> = outs[1]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let want_flat: Vec<i32> = want_segs
        .iter()
        .flat_map(|s| [s.start as i32, s.end as i32])
        .collect();
    eprintln!(
        "{} frames: probabilities within {worst}; segments {segs:?}",
        got.len()
    );
    assert!(worst < 2e-5, "probabilities off by {worst}");
    assert_eq!(segs, want_flat);
}
