//! Native F5 clone → write WAV → Whisper fox check (fails if not intelligible).
//!
//! Env: `RLX_F5TTS_DIR`, `RLX_WHISPER_DIR`, `NFE` (default 16), `OUT`
//! (default `tmp/f5tts_wavs/validated.wav`), `DEVICE`, `REF_SECS` (optional trim).
use std::path::PathBuf;

use rlx_f5tts::{F5Native, InferOpts, SAMPLE_RATE, write_wav};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};

const REF_TEXT: &str = "Hello from Kokoro. This is a test of speech synthesis in Rust.";
const TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const FOX: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(
        std::env::var("RLX_F5TTS_DIR").unwrap_or_else(|_| "weights/tts/f5tts".into()),
    );
    let out = PathBuf::from(
        std::env::var("OUT").unwrap_or_else(|_| "tmp/f5tts_wavs/validated.wav".into()),
    );
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let device = match std::env::var("DEVICE")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        "ane" | "coreml" => Device::Ane,
        "cuda" => Device::Cuda,
        _ => Device::Cpu,
    };
    let tts = F5Native::load_on(&dir, device)?;
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(&p)?;
    let max = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    let mut refa: Vec<f32> = r
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect();
    if let Ok(secs) = std::env::var("REF_SECS") {
        let n = (secs.parse::<f32>().unwrap_or(4.0) * tts.sample_rate() as f32) as usize;
        refa.truncate(n.min(refa.len()));
    }
    let text = std::env::var("GEN_TEXT").unwrap_or_else(|_| TEXT.to_string());
    let nfe: usize = std::env::var("NFE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let t0 = std::time::Instant::now();
    let audio = tts.synthesize(&text, &refa, REF_TEXT, &InferOpts { nfe, speed: 1.0 })?;
    let dt = t0.elapsed().as_secs_f32();
    let peak = audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let secs = audio.len() as f32 / tts.sample_rate() as f32;
    anyhow::ensure!(peak >= 0.05, "near-silent synthesis (peak={peak:.4})");
    write_wav(&audio, SAMPLE_RATE, &out)?;
    eprintln!(
        "wrote {} ({secs:.2}s, peak={peak:.3}, nfe={nfe}, exec={:?}) in {dt:.1}s → {}",
        audio.len(),
        tts.execution_device(),
        out.display()
    );

    // Only fox-gate the default short phrase.
    if text != TEXT {
        eprintln!("target:  {text}");
        eprintln!("(skip whisper fox gate for custom GEN_TEXT)");
        return Ok(());
    }
    let wd = whisper_dir().ok_or_else(|| {
        anyhow::anyhow!("Whisper weights required (RLX_WHISPER_DIR or .cache/whisper-*)")
    })?;
    let pcm = resample_linear(&audio, SAMPLE_RATE, WR as u32);
    let mut w = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let transcript = w.transcribe_greedy(&pcm)?;
    let lower = transcript.to_lowercase();
    let hits = FOX.iter().filter(|word| lower.contains(*word)).count();
    eprintln!("target:  {TEXT}");
    eprintln!("whisper: {transcript}");
    eprintln!("fox:     {hits}/6");
    anyhow::ensure!(
        hits >= 4,
        "Whisper fox coverage {hits}/6 < 4 — output is not intelligible speech"
    );
    Ok(())
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        return whisper_ready(&p).then_some(p);
    }
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    ["whisper-base.en", "whisper-tiny.en", "whisper-tiny"]
        .into_iter()
        .map(|n| cache.join(n))
        .find(|p| whisper_ready(p))
}

fn whisper_ready(dir: &std::path::Path) -> bool {
    dir.join("model.safetensors").is_file() && dir.join("tokenizer.json").is_file()
}

fn resample_linear(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}
