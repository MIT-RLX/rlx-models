//! Text-in → Zonos → Whisper text-out.
//!
//! ```bash
//! just zonos-whisper
//! RLX_TEXT="Hi." cargo run -p rlx-zonos --release --example whisper_roundtrip --features espeak
//! ```

use std::path::PathBuf;
use std::time::Instant;

use rlx_runtime::Device;
use rlx_zonos::{
    DEFAULT_DAC_DIR, DEFAULT_LOCAL_DIR, InferOpts, NativeZonos, SAMPLE_RATE, peak_amplitude,
};

const DEFAULT_TEXT: &str = "Hello from Zonos.";

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::var("RLX_ZONOS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOCAL_DIR));
    let dac_dir = PathBuf::from(DEFAULT_DAC_DIR);
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.into());
    let max_tokens: usize = std::env::var("RLX_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    anyhow::ensure!(
        model_dir.join("model.safetensors").is_file(),
        "missing Zonos weights — just fetch-zonos"
    );
    anyhow::ensure!(
        dac_dir.join("model.safetensors").is_file(),
        "missing DAC — just fetch-parler-dac"
    );

    let mut whisper = load_whisper().ok_or_else(|| {
        anyhow::anyhow!("Whisper weights not found (RLX_WHISPER_DIR / .cache/whisper-*)")
    })?;

    println!("== Zonos Whisper roundtrip ==");
    println!("text in: {text:?}");

    let t0 = Instant::now();
    let model = NativeZonos::open(&model_dir, &dac_dir, Device::Cpu)?;
    let opts = InferOpts {
        max_new_tokens: Some(max_tokens),
        greedy: true,
        ..InferOpts::default()
    };
    let pcm = model.synthesize(&text, &opts)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let peak = peak_amplitude(&pcm);
    println!(
        "synth: {:.2}s audio peak={:.3} in {:.0}ms",
        pcm.len() as f64 / SAMPLE_RATE as f64,
        peak,
        ms
    );

    let (cov, hyp) = coverage(&mut whisper, &pcm, SAMPLE_RATE, &text);
    println!("whisper-out: {hyp:?}");
    println!("coverage: {cov:.0}%");
    Ok(())
}

fn load_whisper() -> Option<rlx_whisper::WhisperRunner> {
    let dir = std::env::var("RLX_WHISPER_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
            ["whisper-base.en", "whisper-tiny.en", "whisper-tiny"]
                .into_iter()
                .map(|n| c.join(n))
                .find(|p| p.join("model.safetensors").is_file())
        })?;
    rlx_whisper::WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .ok()
}

fn coverage(
    w: &mut rlx_whisper::WhisperRunner,
    pcm: &[f32],
    sr: u32,
    expect: &str,
) -> (f64, String) {
    let pcm16 = resample(pcm, sr, rlx_whisper::SAMPLE_RATE as u32);
    let Ok(t) = w.transcribe_greedy(&pcm16) else {
        return (0.0, String::new());
    };
    let want: Vec<_> = expect
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|x| x.len() > 2)
        .map(str::to_string)
        .collect();
    let got = t.to_lowercase();
    let hit = want.iter().filter(|w| got.contains(w.as_str())).count();
    (
        100.0 * hit as f64 / want.len().max(1) as f64,
        t.trim().to_string(),
    )
}

fn resample(pcm: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return pcm.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n = ((pcm.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let src = i as f64 * ratio;
        let j = src.floor() as usize;
        let frac = (src - j as f64) as f32;
        let a = pcm[j];
        let b = pcm.get(j + 1).copied().unwrap_or(a);
        out.push(a * (1.0 - frac) + b * frac);
    }
    out
}
