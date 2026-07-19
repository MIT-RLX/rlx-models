// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! STEP 3 validation: the native rlx `detokenizer.onnx` decode must match the
//! onnxruntime reference. Fixtures (`tests/fixtures/codec_*`) are dumped by the
//! ort reference; skips unless weights + fixtures present.

use std::path::PathBuf;

use rlx_miratts::codec::MiraCodec;
use rlx_runtime::Device;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn read_i64(p: &PathBuf) -> Vec<i64> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn read_i32(p: &PathBuf) -> Vec<i32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn read_f32(p: &PathBuf) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

// `detokenizer.onnx` imports natively (params=554) and decodes bit-exact vs
// onnxruntime (cos 0.99999). Required three general rlx-onnx-import/rlx-cpu
// fixes: broadcast_dims leading-dim collapse, i32 Gather indices read as f32,
// and a missing CastI32ToI64. See the `miratts_port_scope` memory note.
#[test]
fn detokenizer_decode_matches_ort() {
    let dir = root().join("weights/tts/miratts/decoders");
    let fix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    if !dir.join("detokenizer.onnx").is_file() || !fix.join("codec_wav.f32").is_file() {
        eprintln!("skip: detokenizer.onnx / codec fixtures not present");
        return;
    }
    let speech: Vec<u32> = read_i64(&fix.join("codec_speech.i64"))
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let context: Vec<u32> = read_i32(&fix.join("codec_context.i32"))
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let ref_wav = read_f32(&fix.join("codec_wav.f32"));

    let codec = MiraCodec::load(&dir, Device::Cpu).expect("load codec");
    let wav = codec.decode(&speech, &context).expect("decode");

    eprintln!("ref len {} rlx len {}", ref_wav.len(), wav.len());
    assert_eq!(wav.len(), ref_wav.len(), "decoded length mismatch");
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
    let maxabs = (0..n)
        .map(|i| (wav[i] - ref_wav[i]).abs())
        .fold(0.0f32, f32::max);
    eprintln!("cos={cos:.6} max_abs={maxabs:.4}");
    assert!(cos > 0.99, "native decode cosine {cos:.4} vs ort too low");
}
