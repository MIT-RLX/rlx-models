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

//! Greedy CustomVoice parity vs committed HF golden codec frames (no Python).

use rlx_qwen3_tts::{GenerationConfig, Qwen3TtsConfig, Qwen3TtsRunnerBuilder, Qwen3TtsSession};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

fn parity_synth_device() -> Device {
    if is_available(Device::Metal) {
        Device::Metal
    } else if is_available(Device::Mlx) {
        Device::Mlx
    } else if is_available(Device::Cuda) {
        Device::Cuda
    } else if is_available(Device::Rocm) {
        Device::Rocm
    } else {
        Device::Cpu
    }
}

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
    assert_eq!(frames.len(), n);
    frames
}

#[test]
fn qwen3_tts_greedy_codec_frames_match_hf() {
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: set RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    if !model_dir
        .join("speech_tokenizer/model.safetensors")
        .is_file()
    {
        eprintln!("skip: missing speech_tokenizer — run `just fetch-qwen3-tts`");
        return;
    }

    let golden = load_golden_hi();

    let device = parity_synth_device();
    let runner = Qwen3TtsRunnerBuilder::default()
        .model_dir(&model_dir)
        .device(device)
        .build()
        .expect("runner");

    let mut gen_cfg = GenerationConfig::greedy_for_model_dir(&model_dir).expect("gen cfg");
    gen_cfg.max_new_tokens = 32;

    let native = rlx_qwen3_tts::synthesize::synthesize_custom_voice_greedy(
        &model_dir,
        runner.config(),
        &rlx_qwen3_tts::Qwen3TtsWeightStore::open(&model_dir).expect("store"),
        device,
        "Hi.",
        "vivian",
        "english",
        &gen_cfg,
        true,
    )
    .expect("native synth");

    assert!(
        !native.codec_frames.is_empty(),
        "native produced no codec frames"
    );
    let _ = runner;
    let n = golden.len().min(native.codec_frames.len());
    let mut exact = 0usize;
    for i in 0..n {
        if golden[i] == native.codec_frames[i] {
            exact += 1;
        }
    }
    eprintln!(
        "codec frame match: {exact}/{n} (golden {} native {})",
        golden.len(),
        native.codec_frames.len()
    );
    for i in 0..n {
        if golden[i] != native.codec_frames[i] {
            eprintln!("first mismatch at frame {i}");
            eprintln!("  golden = {:?}", golden[i]);
            eprintln!("  native = {:?}", native.codec_frames[i]);
            break;
        }
    }
    if native.codec_frames.len() > golden.len() {
        for (i, f) in native
            .codec_frames
            .iter()
            .enumerate()
            .skip(golden.len().saturating_sub(3))
        {
            if f[0] == 2150 {
                eprintln!("eos at native frame {i}");
                break;
            }
        }
    }
    assert!(
        exact == n && golden.len() == native.codec_frames.len(),
        "greedy codec frames diverged from golden (matched {exact}/{n})"
    );
}

/// Warmed session: two utterances on default Metal hybrid (no `METAL_DECODE_NATIVE`).
#[test]
fn qwen3_tts_session_reuse_matches_hf_golden() {
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: set RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_QWEN3_TTS_PARITY=1");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_METAL_DECODE_NATIVE")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!("skip: native Metal decode is not HF-parity yet");
        return;
    }

    let golden = load_golden_hi();
    let device = parity_synth_device();
    let mut session = Qwen3TtsSession::open(&model_dir, device).expect("session");
    let mut gen_cfg = GenerationConfig::greedy_for_model_dir(&model_dir).expect("gen cfg");
    gen_cfg.max_new_tokens = 32;

    for (utterance, text) in [("hi", "Hi."), ("hello", "Hello world.")] {
        let result = session
            .synthesize_custom_voice(text, "vivian", "english", &gen_cfg, true)
            .expect("synth");
        if utterance == "hi" {
            let n = golden.len().min(result.codec_frames.len());
            let exact = (0..n)
                .filter(|&i| golden[i] == result.codec_frames[i])
                .count();
            eprintln!("session {utterance}: codec match {exact}/{n}");
            assert_eq!(
                exact, n,
                "session reuse: {utterance} diverged from golden ({exact}/{n})"
            );
            assert_eq!(golden.len(), result.codec_frames.len());
        } else {
            assert!(
                !result.codec_frames.is_empty(),
                "session reuse: {utterance} produced no frames"
            );
            eprintln!(
                "session {utterance}: {} frames (second utterance ok)",
                result.codec_frames.len()
            );
        }
    }
}

#[test]
fn qwen3_tts_greedy_speech_decode_produces_pcm() {
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: set RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_QWEN3_TTS_PARITY=1");
        return;
    };
    if !model_dir
        .join("speech_tokenizer/model.safetensors")
        .is_file()
    {
        eprintln!("skip: missing speech_tokenizer");
        return;
    }

    let mut gen_cfg = GenerationConfig::greedy_for_model_dir(&model_dir).expect("gen cfg");
    gen_cfg.max_new_tokens = 32;

    let device = parity_synth_device();
    let result = rlx_qwen3_tts::synthesize::synthesize_custom_voice_greedy(
        &model_dir,
        &Qwen3TtsConfig::from_model_dir(&model_dir).expect("cfg"),
        &rlx_qwen3_tts::Qwen3TtsWeightStore::open(&model_dir).expect("store"),
        device,
        "Hi.",
        "vivian",
        "english",
        &gen_cfg,
        false,
    )
    .expect("synth+decode");

    assert_eq!(result.codec_frames.len(), 22);
    assert_eq!(result.sample_rate, 24_000);
    assert!(
        result.pcm.len() > 4_000,
        "expected non-trivial PCM, got {} samples",
        result.pcm.len()
    );
    let peak = result.pcm.iter().map(|s| s.abs()).fold(0f32, f32::max);
    assert!(peak > 1e-4, "PCM peak too small ({peak})");
    eprintln!(
        "speech decode ok: {} frames, {} pcm samples, peak={peak:.4}",
        result.codec_frames.len(),
        result.pcm.len()
    );
}
