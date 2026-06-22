//! End-to-end parity: raw text → wav, fully in Rust, vs the Python reference
//! fixtures (which were generated from the same texts).

use std::path::PathBuf;

use rlx_inflect_nano::{InferOpts, InflectNano};
use safetensors::SafeTensors;
use serde_json::Value;

fn data_dir() -> Option<PathBuf> {
    let base = std::env::var("RLX_INFLECT_NANO_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/inflect-nano-rlx")
        });
    base.join("config.json").exists().then_some(base)
}

fn read_f32(st: &SafeTensors, name: &str) -> Vec<f32> {
    st.tensor(name)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn e2e_text_to_wav_matches_python() {
    let Some(dir) = data_dir() else {
        eprintln!("skip: bundle not found");
        return;
    };
    let model = InflectNano::load_from_dir(&dir).expect("load");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("fixtures/manifest.json")).unwrap())
            .unwrap();
    let opts = InferOpts::default();

    let mut worst_mel = 0.0f32;
    let mut worst_wav = 0.0f32;
    for case in manifest["cases"].as_array().unwrap() {
        let idx = case["index"].as_u64().unwrap();
        let text = case["text"].as_str().unwrap();

        // text → ids must match the reference exactly
        let (p, t, l) = model.text_to_ids(text).expect("ids");
        let ep: Vec<i64> = case["phone_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(p, ep, "case {idx}: phone ids differ for {text:?}");

        let mel = model
            .mel_from_ids(&p, &t, &l, model.cfg.default_speaker(), &opts)
            .unwrap();
        let wav = model.wav_from_mel(&mel).unwrap();

        let bytes = std::fs::read(dir.join(format!("fixtures/case_{idx}.safetensors"))).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let ref_mel = read_f32(&st, "mel");
        let ref_wav = read_f32(&st, "wav");

        let mel_flat: Vec<f32> = mel.iter().copied().collect();
        let md = mel_flat
            .iter()
            .zip(&ref_mel)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let wd = wav
            .iter()
            .zip(&ref_wav)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        worst_mel = worst_mel.max(md);
        worst_wav = worst_wav.max(wd);
        eprintln!("case {idx}: e2e mel={md:.2e} wav={wd:.2e}  {text:?}");
    }
    eprintln!("E2E WORST mel={worst_mel:.3e} wav={worst_wav:.3e}");
    assert!(worst_mel < 1e-3 && worst_wav < 2e-2, "e2e parity too loose");
}
