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

//! Greedy codec frames on MLX vs HF golden (env-gated).

use rlx_qwen3_tts::{GenerationConfig, Qwen3TtsConfig, Qwen3TtsWeightStore};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_dir() -> Option<PathBuf> {
    std::env::var("RLX_QWEN3_TTS_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("model.safetensors").is_file())
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
fn qwen3_tts_greedy_codec_frames_match_hf_on_mlx() {
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

    let golden = load_golden_hi();
    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let mut gen_cfg = GenerationConfig::greedy_for_model_dir(&model_dir).unwrap();
    gen_cfg.max_new_tokens = 32;

    let native = rlx_qwen3_tts::synthesize::synthesize_custom_voice_greedy(
        &model_dir,
        &cfg,
        &store,
        Device::Mlx,
        "Hi.",
        "vivian",
        "english",
        &gen_cfg,
        true,
    )
    .expect("mlx synth");

    let n = golden.len().min(native.codec_frames.len());
    let mut exact = 0usize;
    for i in 0..n {
        if golden[i] == native.codec_frames[i] {
            exact += 1;
        }
    }
    eprintln!(
        "mlx codec frame match: {exact}/{n} (golden {} native {})",
        golden.len(),
        native.codec_frames.len()
    );
    for i in 0..n {
        if golden[i] != native.codec_frames[i] {
            eprintln!("first mismatch frame {i}");
            eprintln!("  golden = {:?}", golden[i]);
            eprintln!("  native = {:?}", native.codec_frames[i]);
            break;
        }
    }
    assert_eq!(exact, n, "mlx greedy codec frames diverged ({exact}/{n})");
    assert_eq!(golden.len(), native.codec_frames.len());
}
