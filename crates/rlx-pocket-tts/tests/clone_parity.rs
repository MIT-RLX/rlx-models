// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Voice-cloning parity (Mimi encoder + speaker projection + voice prompt)
//! against upstream `pocket_tts`, on a checkpoint whose zeroed encoder was
//! filled with seeded random weights by `scripts/pocket_tts_clone_reference.py`
//! (the ungated release has no encoder weights to test with).
//!
//! ```text
//! R=$PWD/weights/tts/pocket-tts-kyutai/reference
//! RLX_POCKET_TTS_CLONE_REFERENCE=$R/clone-english_2026-10 \
//! RLX_POCKET_TTS_CLONE_WAV=$R/clone_prompt_16bit.wav \
//! RLX_POCKET_TTS_TOKENIZER=$PWD/weights/tts/pocket-tts-kyutai/languages/english_2026-10/tokenizer.model \
//!     cargo test -p rlx-pocket-tts --release --test clone_parity -- --nocapture
//! ```
//!
//! `RLX_POCKET_TTS_DEVICE` picks the device (default `cpu`). Skips without the env.

use std::path::PathBuf;

use anyhow::Result;
use rlx_pocket_tts::weights::WeightFile;
use rlx_pocket_tts::{ModelFiles, ModelSpec, PocketTts};

fn env_path(key: &str) -> Option<PathBuf> {
    rlx_ir::env::var_result(key).ok().map(PathBuf::from)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().map(|x| x.abs()).fold(0.0, f32::max).max(1e-12);
    max_abs(a, b) / scale
}

#[test]
fn clone_matches_upstream() -> Result<()> {
    let (Some(refdir), Some(wav), Some(tokenizer)) = (
        env_path("RLX_POCKET_TTS_CLONE_REFERENCE"),
        env_path("RLX_POCKET_TTS_CLONE_WAV"),
        env_path("RLX_POCKET_TTS_TOKENIZER"),
    ) else {
        eprintln!("skip: set RLX_POCKET_TTS_CLONE_REFERENCE, _CLONE_WAV and _TOKENIZER");
        return Ok(());
    };
    let device = rlx_ir::env::var_result("RLX_POCKET_TTS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = rlx_runtime::parse_device(&device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let reference = WeightFile::open(refdir.join("clone_reference.safetensors"))?;
    let files = ModelFiles {
        weights: refdir.join("model.safetensors"),
        tokenizer,
        voices: None,
    };
    let mut tts = PocketTts::open_with(ModelSpec::builtin("english_2026-10")?, files, device)?;

    let (audio, rate) = rlx_pocket_tts::audio::read_wav_mono(&wav)?;
    let prompt = rlx_pocket_tts::audio::end_on_pause(&audio, rate as usize);
    let (want_audio, _) = reference.get_f32("audio")?;
    // Upstream pads the prompt to whole frames inside the encoder; compare the
    // part both have.
    let n = prompt.len().min(want_audio.len());
    let audio_d = max_abs(&prompt[..n], &want_audio[..n]);
    eprintln!(
        "prompt audio: {} vs {} samples, max|Δ| {audio_d:.2e}",
        prompt.len(),
        want_audio.len()
    );
    assert!(audio_d < 1e-6, "end_on_pause mismatch {audio_d}");

    let cond = tts.encode_prompt(&prompt)?;
    let (want_cond, _) = reference.get_f32("conditioning")?;
    let cond_rel = rel(&cond, &want_cond);
    eprintln!(
        "conditioning: {} values, max rel Δ {cond_rel:.2e}",
        cond.len()
    );
    assert!(cond_rel < 1e-3, "conditioning mismatch {cond_rel}");

    let state = tts.voice_from_audio(&audio, rate)?;
    let mut worst = 0f32;
    for l in 0..state.num_layers() {
        let pre = format!("transformer.layers.{l}.self_attn");
        let (k, _) = reference.get_f32(&format!("{pre}/k"))?;
        let (v, _) = reference.get_f32(&format!("{pre}/v"))?;
        worst = worst.max(rel(&state.k[l], &k)).max(rel(&state.v[l], &v));
    }
    eprintln!(
        "voice state on {device:?}: offset {}, worst layer max rel Δ {worst:.2e}",
        state.offset
    );
    assert!(worst < 1e-3, "voice state mismatch {worst}");

    // Round trip through the export format.
    let path = std::env::temp_dir().join("rlx_pocket_tts_clone_state.safetensors");
    tts.save_voice(&state, &path)?;
    let back = tts.load_voice(path.to_str().unwrap())?;
    assert_eq!(back.offset, state.offset);
    assert_eq!(back.k, state.k);
    Ok(())
}
