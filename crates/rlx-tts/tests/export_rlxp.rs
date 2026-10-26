//! An exported RLX TTS (`weights/tts/rlx-tts`), opened by rlx alone, speaks
//! as `RlxTts` does on its portable path: the mel of a text, and the
//! vocoder's samples on the runner's mel with the BNNS AES-CTR noise and
//! greedy (until the runner's fused GRU arithmetic and the scan's separate
//! ops, ulps apart, tip a sample).
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_tts::export::export_rlxp;
use rlx_tts::frontend::TextFrontend;
use rlx_tts::wavernn::WaveRnn;
use rlx_tts::{RlxTts, VarianceControls, WaveRnnOpts, WaveRnnRng};
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn floats(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn an_exported_rlx_tts_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/rlx-tts");
    if !dir.join("rlx-tts.rlxp").is_file() {
        eprintln!("skip: no RLX TTS bundle");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("rlx-tts-package.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let tts = RlxTts::open(&dir).unwrap();

    // The mel.
    let text = "Hello there, how are you today?";
    let fe = tts.frontend();
    let ids = fe.phones_to_ids(&fe.text_to_phones(text).unwrap()).unwrap();
    let ctrl = VarianceControls {
        pause_min_frames: ((fe.pause_min_duration_ms.max(0.0) / 10.0).ceil() as usize).max(1),
        ..VarianceControls::default()
    };
    let want = tts.infer_mel(&ids, &ctrl).unwrap();
    let got = m
        .run(&[], &json!({"entry": "speak", "text": text}))
        .unwrap();
    let mel = floats(&got[0].data);
    assert_eq!(got[0].shape, vec![want.nrows(), 80], "the frame counts");
    let want_flat: Vec<f32> = want.iter().copied().collect();
    let diff = mel
        .iter()
        .zip(&want_flat)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let cos = cosine(&mel, &want_flat);
    eprintln!("mel: {} frames, max |d| {diff}, cosine {cos}", want.nrows());
    assert!(cos > 0.99999, "mel cosine {cos}");

    // The vocoder on the runner's first 30 frames.
    let frames = 30.min(want.nrows());
    let part = want.slice(ndarray::s![..frames, ..]).to_owned();
    let mut opts = WaveRnnOpts::product_default();
    opts.rng = WaveRnnRng::Bnns;
    opts.seed = Some(16_807);
    let pcm_want = WaveRnn::new(tts.wavernn_weights())
        .infer(&part, &opts)
        .unwrap();
    let rows: Vec<Vec<f32>> = part.rows().into_iter().map(|r| r.to_vec()).collect();
    let got = m.run(&[], &json!({"entry": "speak", "mel": rows})).unwrap();
    let pcm = floats(&got[0].data);
    assert_eq!(pcm.len(), pcm_want.len());
    let same = pcm
        .iter()
        .zip(&pcm_want)
        .take_while(|(a, b)| a == b)
        .count();
    let cos = cosine(&pcm, &pcm_want);
    eprintln!(
        "vocoder: {same} of {} samples the runner's, cosine {cos}",
        pcm_want.len()
    );
    // Greedy, no near tie to tip: the scan's arithmetic is the runner's.
    let mut g_opts = opts.clone();
    g_opts.greedy = true;
    let g_want = WaveRnn::new(tts.wavernn_weights())
        .infer(&part, &g_opts)
        .unwrap();
    let got = m
        .run(&[], &json!({"entry": "speak", "mel": rows, "greedy": true}))
        .unwrap();
    let g_pcm = floats(&got[0].data);
    let g_same = g_pcm
        .iter()
        .zip(&g_want)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "vocoder greedy: {g_same} of {} samples the runner's",
        g_want.len()
    );
    // The runner's GRU fuses its multiply-adds and its `exp` is vmath's; the
    // scan's are separate ops, ulps apart, and the recurrent state carries
    // that until a decision tips (greedy as sampled, a few hundred steps in).
    assert!(
        same >= 400 && g_same >= 400,
        "{same} / {g_same} samples the same"
    );

    // The whole path.
    let audio = m.synthesize(text, &json!({})).unwrap();
    assert_eq!(audio.sample_rate, 24_000);
    assert!(
        audio.samples.len() > 24_000,
        "{} samples",
        audio.samples.len()
    );
}
