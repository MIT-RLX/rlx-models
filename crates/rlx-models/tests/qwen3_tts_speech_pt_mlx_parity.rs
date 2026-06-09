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

//! MLX compiled speech `pre_transformer` (layer_scale baked) vs CPU eager on golden codec frames.

use rlx_qwen3_tts::speech_tokenizer::decode_codec_frames;
use rlx_runtime::{Device, is_available};
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
        frames.push(
            line.split_whitespace()
                .map(|s| s.parse().expect("tok"))
                .collect(),
        );
    }
    frames
}

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 16 {
        return 0.0;
    }
    let a = &a[..n];
    let b = &b[..n];
    let ma: f32 = a.iter().sum::<f32>() / n as f32;
    let mb: f32 = b.iter().sum::<f32>() / n as f32;
    let mut num = 0f64;
    let mut da = 0f64;
    let mut db = 0f64;
    for i in 0..n {
        let x = (a[i] - ma) as f64;
        let y = (b[i] - mb) as f64;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    (num / (da * db).sqrt()) as f32
}

#[test]
fn speech_pt_mlx_near_cpu_on_golden_frames() {
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    if !is_available(Device::Mlx) {
        eprintln!("skip: MLX not available");
        return;
    }
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    let frames = load_golden_hi();
    let cpu = decode_codec_frames(&model_dir, &frames, Device::Cpu).expect("cpu decode");
    let mlx = decode_codec_frames(&model_dir, &frames, Device::Mlx).expect("mlx decode");
    assert_eq!(cpu.len(), mlx.len());
    let c = corr(&cpu, &mlx);
    eprintln!("speech decode cpu vs mlx corr={c:.4} (len {})", cpu.len());
    assert!(
        c > 0.35,
        "MLX speech PT diverged from CPU eager (corr={c:.4}, want >0.35; use CPU eager for best quality)"
    );
}
