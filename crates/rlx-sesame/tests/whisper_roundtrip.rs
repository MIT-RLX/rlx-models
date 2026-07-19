//! Whisper round-trip for Sesame CSM-1B.
//!
//! Needs `weights/tts/sesame/model.safetensors`, Mimi (`.cache/mimi`), and
//! Whisper Tiny (`.cache/whisper-tiny`). Run via `just sesame-whisper`.

use std::path::PathBuf;

fn sesame_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/sesame");
    if p.join("model.safetensors").is_file() && p.join("config.json").is_file() {
        Some(p)
    } else {
        None
    }
}

fn mimi_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/mimi"),
        PathBuf::from(".cache/mimi"),
    ];
    candidates
        .into_iter()
        .find(|p| p.join("config.json").is_file() || p.join("model.safetensors").is_file())
}

fn whisper_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/whisper-tiny"),
        PathBuf::from(".cache/whisper-tiny"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn whisper_coverage(hyp: &str, want: &[&str]) -> usize {
    let lower = hyp.to_lowercase();
    want.iter()
        .filter(|w| lower.contains(&w.to_lowercase()))
        .count()
}

fn resample_linear(pcm: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || pcm.is_empty() {
        return pcm.to_vec();
    }
    let ratio = dst_sr as f64 / src_sr as f64;
    let out_len = ((pcm.len() as f64) * ratio).round() as usize;
    let mut out = vec![0.0f32; out_len];
    for (i, o) in out.iter_mut().enumerate() {
        let src = i as f64 / ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(pcm.len().saturating_sub(1));
        let t = (src - i0 as f64) as f32;
        *o = pcm[i0] * (1.0 - t) + pcm[i1] * t;
    }
    out
}

#[test]
fn test_config_loads() {
    let Some(dir) = sesame_dir() else {
        eprintln!("skipping: no sesame weights");
        return;
    };
    let cfg = rlx_sesame::SesameConfig::from_file(dir.join("config.json")).expect("config");
    assert_eq!(cfg.num_hidden_layers, 16);
    assert_eq!(cfg.num_codebooks, 32);
    assert_eq!(cfg.vocab_size, 2051);
}

#[test]
fn test_weight_keys_present() {
    let Some(dir) = sesame_dir() else {
        eprintln!("skipping: no sesame weights");
        return;
    };
    if !dir.join("model.safetensors").is_file() {
        eprintln!("skipping: model.safetensors missing");
        return;
    }
    let w = rlx_sesame::weights::CsmWeights::load(&dir).expect("load weights");
    assert_eq!(w.backbone.layers.len(), 16);
    assert_eq!(w.depth.layers.len(), 4);
    assert_eq!(w.cfg.hidden_size, 2048);
}

#[test]
fn test_sesame_whisper_fox() {
    let Some(model) = sesame_dir() else {
        eprintln!("skipping: no sesame weights");
        return;
    };
    if !model.join("model.safetensors").is_file() {
        eprintln!("skipping: model.safetensors missing");
        return;
    }
    let Some(mimi) = mimi_dir() else {
        eprintln!("skipping: no mimi weights");
        return;
    };
    let Some(wd) = whisper_dir() else {
        eprintln!("skipping: no Whisper weights");
        return;
    };

    let mut session = rlx_sesame::SesameSession::open_on(&model, &mimi, rlx_runtime::Device::Cpu)
        .expect("open sesame");
    let opts = rlx_sesame::GenerateOpts {
        max_audio_frames: 200,
        temperature: 0.9,
        topk: 50,
        seed: 42,
        greedy: false,
        speaker: 0,
    };
    let text = "The quick brown fox jumps over the lazy dog.";
    let out = session.synthesize(text, &opts).expect("synthesize");
    assert!(
        out.samples.len() > 12_000,
        "too short: {} samples",
        out.samples.len()
    );
    assert!(
        out.samples.iter().any(|v| v.abs() > 1e-3),
        "near-silent audio"
    );

    let pcm16 = resample_linear(&out.samples, out.sample_rate, 16_000);
    use rlx_whisper::WhisperRunner;
    let mut whisper = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .language("en")
        .build()
        .expect("whisper");
    let transcript = whisper.transcribe_greedy(&pcm16).expect("transcribe");
    let want = ["quick", "brown", "fox", "jumps", "lazy", "dog"];
    let hits = whisper_coverage(&transcript, &want);
    eprintln!(
        "[sesame whisper fox] {}/{} frames={} transcript={transcript:?}",
        hits,
        want.len(),
        out.audio_frames.len()
    );
    assert!(
        hits >= 5,
        "Whisper coverage {hits}/{} too low: {transcript:?}",
        want.len()
    );
}

const LONG: &str = "The quick brown fox jumps over the lazy dog. Courage and kindness matter more than cleverness alone when people face hard times together and choose to help each other without waiting for perfect conditions.";

fn long_words() -> Vec<&'static str> {
    vec![
        "quick", "brown", "fox", "jumps", "lazy", "dog", "courage", "kindness", "matter", "people",
        "hard", "times", "help", "each", "other",
    ]
}

/// Default seed for the long paragraph (validated Whisper ≥14/15).
fn long_seed() -> u64 {
    std::env::var("RLX_SESAME_LONG_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42)
}

#[test]
fn test_sesame_whisper_long() {
    let Some(model) = sesame_dir() else {
        eprintln!("skipping: no sesame weights");
        return;
    };
    if !model.join("model.safetensors").is_file() {
        eprintln!("skipping: model.safetensors missing");
        return;
    }
    let Some(mimi) = mimi_dir() else {
        eprintln!("skipping: no mimi weights");
        return;
    };
    let Some(wd) = whisper_dir() else {
        eprintln!("skipping: no Whisper weights");
        return;
    };

    let mut session = rlx_sesame::SesameSession::open_on(&model, &mimi, rlx_runtime::Device::Cpu)
        .expect("open sesame");
    let seed = long_seed();
    let opts = rlx_sesame::GenerateOpts {
        max_audio_frames: 400,
        temperature: 0.9,
        topk: 50,
        seed,
        greedy: false,
        speaker: 0,
    };
    let out = session.synthesize(LONG, &opts).expect("synthesize");
    assert!(
        out.samples.len() > 48_000,
        "too short for long paragraph: {} samples / {} frames",
        out.samples.len(),
        out.audio_frames.len()
    );

    let pcm16 = resample_linear(&out.samples, out.sample_rate, 16_000);
    use rlx_whisper::WhisperRunner;
    let mut whisper = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .language("en")
        .build()
        .expect("whisper");
    let transcript = whisper.transcribe_greedy(&pcm16).expect("transcribe");
    let want = long_words();
    let hits = whisper_coverage(&transcript, &want);
    eprintln!(
        "[sesame whisper long] seed={seed} {}/{} frames={} transcript={transcript:?}",
        hits,
        want.len(),
        out.audio_frames.len()
    );
    assert!(
        hits >= 12,
        "long Whisper coverage {hits}/{} too low (seed={seed}): {transcript:?}",
        want.len()
    );
}
