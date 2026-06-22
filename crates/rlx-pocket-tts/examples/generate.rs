// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Pocket TTS demo: download weights + a voice from Hugging Face, generate
//! a short utterance, write `out.wav`.
//!
//! Run with:
//! ```bash
//! cargo run -p rlx-pocket-tts --example generate --features hf-download
//! ```

use anyhow::Result;
use rlx_pocket_tts::{GenerationOptions, TtsModel};

fn main() -> Result<()> {
    let voice_name = std::env::var("POCKET_TTS_VOICE").unwrap_or_else(|_| "alba".to_string());
    let text = std::env::var("POCKET_TTS_TEXT").unwrap_or_else(|_| {
        "Hello world, this is a test of pocket TTS running in Rust.".to_string()
    });
    let out_path = std::env::var("POCKET_TTS_OUT").unwrap_or_else(|_| "pocket_tts.wav".to_string());

    eprintln!("fetching weights + tokenizer...");
    let assets = rlx_pocket_tts::download::fetch_default_assets()?;
    eprintln!("  weights:   {}", assets.weights.display());
    eprintln!("  tokenizer: {}", assets.tokenizer.display());

    eprintln!("fetching voice `{voice_name}`...");
    let voice_path = rlx_pocket_tts::download::fetch_voice(&voice_name)?;
    eprintln!("  voice: {}", voice_path.display());

    eprintln!("loading model...");
    let start = std::time::Instant::now();
    let model = TtsModel::open(&assets.weights, &assets.tokenizer)?;
    eprintln!("  loaded in {:.2}s", start.elapsed().as_secs_f32());

    let voice = model.load_voice(&voice_path)?;
    eprintln!(
        "  voice frames={} d_model={}",
        voice.num_frames(),
        voice.embed_dim()
    );

    let mut opts = GenerationOptions::default();
    if let Ok(s) = std::env::var("POCKET_TTS_SEED") {
        if let Ok(v) = s.parse::<u64>() {
            opts.seed = v;
        }
    }
    eprintln!("generating: {text:?}  seed={}", opts.seed);
    let start = std::time::Instant::now();
    let audio = model.generate(&text, &voice, opts)?;
    let dt = start.elapsed().as_secs_f32();
    eprintln!(
        "  {} samples ({:.2}s audio) in {:.2}s — {:.2}× realtime",
        audio.samples.len(),
        audio.duration_secs(),
        dt,
        audio.duration_secs() / dt.max(1e-6),
    );

    audio.write_wav(&out_path)?;
    eprintln!("wrote {out_path}");
    Ok(())
}
