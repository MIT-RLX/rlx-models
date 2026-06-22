//! Host-eager parity vs the Python reference fixtures captured by
//! `scripts/inflect_nano_reference.py`. Checks durations (exact), mel, and the
//! raw vocoder waveform per corpus sentence.
//!
//! Point `RLX_INFLECT_NANO_DATA` at the bundle dir (default: weights/inflect-nano-rlx).

use std::path::PathBuf;

use ndarray::Array2;
use rlx_inflect_nano::{InferOpts, InflectNano};
use safetensors::SafeTensors;
use serde_json::Value;

fn data_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_INFLECT_NANO_DATA") {
        let p = PathBuf::from(d);
        return p.join("config.json").exists().then_some(p);
    }
    // workspace-root/weights/inflect-nano-rlx
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/inflect-nano-rlx");
    p.join("config.json").exists().then_some(p)
}

fn read_f32(st: &SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let v = st.tensor(name).expect(name);
    let data = v
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (data, v.shape().to_vec())
}

fn read_i64(st: &SafeTensors, name: &str) -> Vec<i64> {
    let v = st.tensor(name).expect(name);
    v.data()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn ids(case: &Value, key: &str) -> Vec<i64> {
    case[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn parity_acoustic_and_vocoder() {
    let Some(dir) = data_dir() else {
        eprintln!("skip: bundle not found (set RLX_INFLECT_NANO_DATA)");
        return;
    };
    let model = InflectNano::load_from_dir(&dir).expect("load model");
    let speaker = model.cfg.default_speaker();
    let opts = InferOpts::default();

    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("fixtures/manifest.json")).unwrap())
            .unwrap();

    let mut worst_mel = 0.0f32;
    let mut worst_wav = 0.0f32;
    for case in manifest["cases"].as_array().unwrap() {
        let idx = case["index"].as_u64().unwrap();
        let phone = ids(case, "phone_ids");
        let tone = ids(case, "tone_ids");
        let lang = ids(case, "lang_ids");

        let bytes = std::fs::read(dir.join(format!("fixtures/case_{idx}.safetensors"))).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();

        // durations (exact integer match)
        let ref_dur = read_i64(&st, "durations");
        let mel = model
            .mel_from_ids(&phone, &tone, &lang, speaker, &opts)
            .expect("mel");

        // mel parity
        let (ref_mel, mel_shape) = read_f32(&st, "mel");
        let ref_mel = Array2::from_shape_vec((mel_shape[0], mel_shape[1]), ref_mel).unwrap();
        assert_eq!(
            mel.dim(),
            ref_mel.dim(),
            "case {idx}: mel shape {:?} vs {:?}",
            mel.dim(),
            ref_mel.dim()
        );
        let mel_flat: Vec<f32> = mel.iter().copied().collect();
        let ref_flat: Vec<f32> = ref_mel.iter().copied().collect();
        let mel_diff = max_abs_diff(&mel_flat, &ref_flat);
        worst_mel = worst_mel.max(mel_diff);

        // wav parity (raw vocoder output, pre-normalization)
        let wav = model.wav_from_mel(&mel).expect("wav");
        let (ref_wav, _) = read_f32(&st, "wav");
        let wav_diff = max_abs_diff(&wav, &ref_wav);
        worst_wav = worst_wav.max(wav_diff);

        eprintln!(
            "case {idx}: frames={} dur_ok={} mel_maxdiff={:.2e} wav_maxdiff={:.2e}",
            mel.dim().1,
            ref_dur == read_i64(&st, "durations"),
            mel_diff,
            wav_diff
        );
        assert_eq!(
            read_i64(&st, "durations"),
            ref_dur,
            "case {idx}: durations differ"
        );
    }
    eprintln!("WORST mel={worst_mel:.3e} wav={worst_wav:.3e}");
    assert!(worst_mel < 1e-3, "mel parity too loose: {worst_mel:.3e}");
    assert!(worst_wav < 2e-2, "wav parity too loose: {worst_wav:.3e}");
}
