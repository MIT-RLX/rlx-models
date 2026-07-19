//! Native `s_encoder` vs ORT on a fixed mel fixture.
//!
//! Fixtures: `s_encoder_mel.f32` `[1,T,128]`, `s_encoder_tokens.i32` `[1,1,32]`.

use std::path::PathBuf;

use rlx_miratts::encoder::MiraSpeakerEncoder;
use rlx_runtime::Device;

#[test]
fn s_encoder_matches_ort_fixture() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("tests/fixtures");
    let mel_path = fixtures.join("s_encoder_mel.f32");
    let tok_path = fixtures.join("s_encoder_tokens.i32");
    let decoders =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/miratts/decoders");
    if !mel_path.is_file() || !tok_path.is_file() || !decoders.join("s_encoder.onnx").is_file() {
        eprintln!("skip: missing s_encoder fixtures or weights");
        return;
    }
    let mel_bytes = std::fs::read(&mel_path).unwrap();
    let mel: Vec<f32> = mel_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ref_tok: Vec<u32> = std::fs::read(&tok_path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect();
    assert_eq!(ref_tok.len(), 32);

    let enc = MiraSpeakerEncoder::load(&decoders, Device::Cpu).expect("load s_encoder");
    let got = enc.encode_mel(&mel).expect("encode_mel");
    assert_eq!(got.len(), 32);
    let matches = got
        .iter()
        .zip(ref_tok.iter())
        .filter(|(a, b)| a == b)
        .count();
    eprintln!("s_encoder token matches: {matches}/32");
    assert_eq!(
        got, ref_tok,
        "native s_encoder tokens must match ORT fixture"
    );
}
