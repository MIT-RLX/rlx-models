//! MioCodec decode vs Python fixture (`fox_fixed_ref.f32`).

use std::path::PathBuf;

use rlx_miotts::codec::{MioCodec, load_preset_embedding};
use rlx_miotts::tokens::fit_speech_len;
use rlx_runtime::Device;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_f32(p: &PathBuf) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn decoder_body_matches_fixture() {
    let codec_dir = root().join("weights/tts/miocodec");
    let fix = codec_dir.join("fixtures");
    if !codec_dir.join("decoder_body.onnx").is_file() || !fix.join("fox_fixed_ref.f32").is_file() {
        eprintln!("skip: decoder_body.onnx / fixtures missing");
        return;
    }
    let tokens: Vec<u32> = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(fix.join("fox_tokens.json")).unwrap(),
    )
    .unwrap()["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let emb = load_preset_embedding(&fix, "en_female")
        .or_else(|_| load_preset_embedding(&root().join("weights/tts/miotts/presets"), "en_female"))
        .expect("preset");
    let ref_wav = read_f32(&fix.join("fox_fixed_ref.f32"));

    let codec = MioCodec::load(&codec_dir, Device::Cpu).expect("load");
    let wav = codec
        .decode(&fit_speech_len(&tokens), &emb)
        .expect("decode");

    let n = wav.len().min(ref_wav.len());
    let dot: f64 = (0..n).map(|i| wav[i] as f64 * ref_wav[i] as f64).sum();
    let na: f64 = wav[..n]
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = ref_wav[..n]
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb + 1e-12);
    eprintln!(
        "codec parity cos={cos:.6} ep={} lenses rlx={} ref={}",
        codec.ep(),
        wav.len(),
        ref_wav.len()
    );
    assert!(cos > 0.99, "cosine {cos} too low");
}
