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

//! Golden codec frames → speech decode should match HF tokenizer decode (CPU eager path).

use rlx_qwen3_tts::speech_tokenizer::decode_codec_frames;
use rlx_runtime::Device;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_dir() -> Option<PathBuf> {
    std::env::var("RLX_QWEN3_TTS_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("speech_tokenizer/model.safetensors").is_file())
}

fn load_golden_hi() -> Vec<Vec<u32>> {
    let path = repo_root().join("crates/rlx-models/tests/fixtures/qwen3_tts_hi_greedy_codec.txt");
    let text = std::fs::read_to_string(&path).expect("golden fixture");
    let mut lines = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let n: usize = lines.next().expect("count").trim().parse().expect("n");
    let mut frames = Vec::with_capacity(n);
    for line in lines.take(n) {
        let vals: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.parse().expect("token"))
            .collect();
        assert_eq!(vals.len(), 16);
        frames.push(vals);
    }
    frames
}

#[test]
fn golden_codec_decode_cpu_matches_metal() {
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    if !rlx_runtime::is_available(rlx_runtime::Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let frames = load_golden_hi();
    let cpu = decode_codec_frames(&model_dir, &frames, rlx_runtime::Device::Cpu).expect("cpu");
    let metal =
        decode_codec_frames(&model_dir, &frames, rlx_runtime::Device::Metal).expect("metal");
    assert_eq!(cpu.len(), metal.len());
    let mut max_abs = 0f32;
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (a, b) in cpu.iter().zip(metal.iter()) {
        max_abs = max_abs.max((a - b).abs());
        let aa = *a as f64;
        let bb = *b as f64;
        dot += aa * bb;
        na += aa * aa;
        nb += bb * bb;
    }
    let corr = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    };
    eprintln!("golden decode cpu vs metal: max_abs={max_abs:.6} corr={corr:.4}");
    assert!(
        corr > 0.99 && max_abs < 0.05,
        "Metal GPU speech decode diverged from CPU (corr={corr:.4}, max_abs={max_abs:.4})"
    );
}

#[test]
fn golden_codec_decode_length_and_peak() {
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    let frames = load_golden_hi();
    let pcm = decode_codec_frames(&model_dir, &frames, Device::Cpu).expect("decode");
    assert_eq!(
        pcm.len(),
        22 * 1920,
        "22 frames × decode_upsample_rate 1920"
    );
    let peak = pcm.iter().map(|s| s.abs()).fold(0f32, f32::max);
    assert!(peak > 0.5, "decode peak too small: {peak}");
}

#[test]
fn golden_codec_decode_hf_pcm_parity() {
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    let hf_path = PathBuf::from("/tmp/hf_golden_pcm.bin");
    if !hf_path.is_file() {
        eprintln!("skip: run HF golden dump to /tmp/hf_golden_pcm.bin");
        return;
    }
    let frames = load_golden_hi();
    if std::env::var("RLX_QWEN3_TTS_SPEECH_DUMP").ok().as_deref() == Some("1") {
        use rlx_qwen3_tts::speech_tokenizer::St12HzDecoder;
        let mut dec = St12HzDecoder::open(&model_dir).expect("open");
        let pcm = dec.decode(&frames, Device::Cpu).expect("decode");
        let _ = std::fs::write(
            "/tmp/rlx_golden_pcm.bin",
            pcm.iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        return;
    }
    let rlx = decode_codec_frames(&model_dir, &frames, Device::Cpu).expect("decode");
    let hf_bytes = std::fs::read(&hf_path).expect("hf pcm");
    let hf: Vec<f32> = hf_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(rlx.len(), hf.len(), "pcm length");
    let rlx_peak = rlx.iter().map(|s| s.abs()).fold(0f32, f32::max);
    let hf_peak = hf.iter().map(|s| s.abs()).fold(0f32, f32::max);
    eprintln!("peak rlx={rlx_peak:.4} hf={hf_peak:.4}");
    let n = rlx.len().min(hf.len());
    let mut max_abs = 0f32;
    let mut mae = 0f64;
    for i in 0..n {
        let d = (rlx[i] - hf[i]).abs();
        max_abs = max_abs.max(d);
        mae += d as f64;
    }
    mae /= n as f64;
    eprintln!("vs HF: max_abs={max_abs:.6} mae={mae:.6}");
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let a = rlx[i] as f64;
        let b = hf[i] as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let corr = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    };
    eprintln!("corr={corr:.4}");
    let _ = std::fs::write(
        "/tmp/rlx_golden_pcm.bin",
        rlx.iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    assert!(
        corr > 0.65 && rlx_peak > hf_peak * 0.5,
        "speech decode diverged from HF (corr={corr:.4}, max_abs={max_abs:.4}, rlx_peak={rlx_peak:.4}, hf_peak={hf_peak:.4})"
    );
}
