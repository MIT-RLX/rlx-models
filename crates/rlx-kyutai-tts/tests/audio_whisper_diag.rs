//! Whisper + spectral diagnostics on generated WAV (env-gated).
//!
//! ```bash
//! RLX_KYUTAI_TTS_DIR=… cargo test -p rlx-kyutai-tts --test audio_whisper_diag --features all-backends --release -- --nocapture
//! ```

use anyhow::{Result, ensure};
use rlx_kyutai_tts::checkpoint::KyutaiTtsVoice;
use rlx_kyutai_tts::download::{DEFAULT_VOICE_NAME, default_kyutai_tts_dir, default_mimi_dir};
use rlx_kyutai_tts::{GenerationConfig, KyutaiTtsSession};
use rlx_mimi::audio::load_wav_mono;
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner, normalize_transcript};
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_kyutai_tts_dir());
    if dir.join("dsm_tts_1e68beda@240.safetensors").is_file() {
        Some(dir)
    } else {
        eprintln!("skip: missing Kyutai weights");
        None
    }
}

fn whisper_dir() -> Option<PathBuf> {
    let p = PathBuf::from("/Users/Shared/rlx-models/.cache/whisper-base.en");
    if p.join("model.safetensors").is_file() {
        Some(p)
    } else {
        None
    }
}

fn spectral_centroid_hz(pcm: &[f32], sr: u32) -> f64 {
    let n_fft = 1024usize;
    if pcm.len() < n_fft {
        return 0.0;
    }
    let mut cents = Vec::new();
    let mut i = 0;
    while i + n_fft <= pcm.len() {
        let mut mag = vec![0.0f64; n_fft / 2 + 1];
        for k in 0..=n_fft / 2 {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for n in 0..n_fft {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / n_fft as f64).cos();
                let x = pcm[i + n] as f64 * w;
                let ang = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / n_fft as f64;
                re += x * ang.cos();
                im += x * ang.sin();
            }
            mag[k] = (re * re + im * im).sqrt();
        }
        let sum: f64 = mag.iter().sum();
        if sum > 0.0 {
            let num: f64 = mag
                .iter()
                .enumerate()
                .map(|(k, m)| k as f64 * sr as f64 / n_fft as f64 * m)
                .sum();
            cents.push(num / sum);
        }
        i += n_fft / 2;
    }
    if cents.is_empty() {
        0.0
    } else {
        let mut s = cents.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    }
}

#[test]
fn generate_whisper_and_spectral_diag() -> Result<()> {
    let Some(model_dir) = model_dir() else {
        return Ok(());
    };
    let Some(wdir) = whisper_dir() else {
        eprintln!("skip: no whisper weights");
        return Ok(());
    };

    let mimi_dir = std::env::var("RLX_MIMI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_mimi_dir());

    let mut session = KyutaiTtsSession::open_on(&model_dir, &mimi_dir, Device::Metal)?;
    session.set_voice(KyutaiTtsVoice::new(DEFAULT_VOICE_NAME));
    let prompt = "Hello world, this is a test of the Kyutai text to speech system.";
    let result = session.generate(
        prompt,
        &GenerationConfig {
            max_steps: 200,
            ..GenerationConfig::default()
        },
    )?;

    let peak = result
        .samples
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    let rms =
        (result.samples.iter().map(|v| v * v).sum::<f32>() / result.samples.len() as f32).sqrt();
    let centroid = spectral_centroid_hz(&result.samples, result.sample_rate);

    eprintln!("frames: {}", result.audio_frames.len());
    for (i, fr) in result.audio_frames.iter().take(5).enumerate() {
        eprintln!("  frame[{i}] first8: {:?}", &fr[..8.min(fr.len())]);
    }
    eprintln!(
        "PCM: {} samples {:.2}s peak={peak:.4} rms={rms:.4} centroid={centroid:.0}Hz",
        result.samples.len(),
        result.samples.len() as f64 / result.sample_rate as f64
    );

    let tmp = std::env::temp_dir().join("rlx-kyutai-diag.wav");
    rlx_mimi::audio::write_wav_mono(&tmp, &result.samples, result.sample_rate)?;
    let pcm16 = load_wav_mono(&tmp, WHISPER_RATE as u32)?;
    let mut w = WhisperRunner::builder()
        .weights(wdir.join("model.safetensors"))
        .config_path(wdir.join("config.json"))
        .tokenizer_path(wdir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let text = normalize_transcript(&w.transcribe_greedy(&pcm16)?);
    eprintln!("whisper: {text:?}");

    ensure!(peak > 0.01, "peak too low");
    ensure!(
        centroid > 300.0 && centroid < 3500.0,
        "centroid {centroid} Hz out of speech band"
    );
    ensure!(
        text.to_lowercase().contains("hello") || text.to_lowercase().contains("world"),
        "whisper {text:?} did not match prompt {prompt:?}"
    );
    Ok(())
}
