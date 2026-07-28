//! F5-TTS native (ort-free) across RLX backends → Whisper fox coverage.
//!
//! ```bash
//! just f5tts-backends
//! ```
//!
//! Env: `RLX_F5TTS_DIR`, `RLX_TEXT`, `RLX_REF_TEXT`, `RLX_REF`, `RLX_NFE`, `RLX_DEVICES`.

use std::path::PathBuf;
use std::time::Instant;

use rlx_f5tts::{DEFAULT_LOCAL_DIR, F5Native, InferOpts, SAMPLE_RATE, peak_amplitude};
use rlx_runtime::{Device, is_available};
use rlx_whisper::WhisperRunner;

const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const DEFAULT_REF_TEXT: &str = "Hello from Kokoro. This is a test of speech synthesis in Rust.";
const FOX_WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn main() -> anyhow::Result<()> {
    let model = std::env::var("RLX_F5TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOCAL_DIR));
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.into());
    let ref_text = std::env::var("RLX_REF_TEXT").unwrap_or_else(|_| DEFAULT_REF_TEXT.into());
    let ref_path = std::env::var("RLX_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav")
        });
    // Official F5 / DakeQQ ONNX default is NFE=32. NFE=16 is Whisper-OK but
    // spectrally hissy vs PyTorch; 32 matches the clean Vocos noise floor.
    let nfe: usize = std::env::var("RLX_NFE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let opts = InferOpts { nfe, speed: 1.0 };
    let (reference, _) = read_wav(&ref_path)?;
    // Keep the full fixture (~4.3 s). Truncating under ~4 s breaks clone quality.

    println!("== F5-TTS native cross-backend ==");
    println!("text: {text:?}  nfe={nfe}");
    println!("model={}  ref={}", model.display(), ref_path.display());
    println!(
        "{:<8} {:>8} {:>7} {:>8} {:>8}  exec",
        "backend", "ms", "peak", "fox", "cos"
    );

    let mut cpu_wav: Option<Vec<f32>> = None;
    let mut failed = false;
    let mut whisper: Option<WhisperRunner> = None;
    let mut ordered = parse_devices();
    if let Some(i) = ordered.iter().position(|(d, _)| *d == Device::Cpu) {
        let cpu = ordered.remove(i);
        ordered.insert(0, cpu);
    }

    for (dev, label) in &ordered {
        let available = *dev == Device::Cpu
            || is_available(*dev)
            || (matches!(*dev, Device::Metal | Device::Mlx) && is_available(Device::Gpu));
        if !available {
            println!(
                "{label:<8} {:>8} {:>7} {:>8} {:>8}  (n/a)",
                "-", "-", "-", "-"
            );
            continue;
        }
        let t0 = Instant::now();
        let (pcm, exec) = match synth_one(&model, *dev, &text, &reference, &ref_text, &opts) {
            Ok(v) => v,
            Err(msg) => {
                println!("{label:<8}  {msg}");
                failed = true;
                continue;
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let peak = peak_amplitude(&pcm);
        let cos = cpu_wav
            .as_ref()
            .map(|ref_wav| cosine(ref_wav, &pcm))
            .unwrap_or(1.0);
        if cpu_wav.is_none() {
            cpu_wav = Some(pcm.clone());
        }
        let fox = {
            let w = whisper.get_or_insert_with(load_whisper);
            whisper_hits(w, &pcm)
        };
        println!("{label:<8} {ms:>8.1} {peak:>7.3} {fox:>3}/6 {cos:>8.5}  {exec}");
        let wav_path = PathBuf::from(format!("tmp/f5tts_wavs/matrix_{label}.wav"));
        if let Some(parent) = wav_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = rlx_f5tts::write_wav(&pcm, SAMPLE_RATE, &wav_path) {
            eprintln!("warn: write {}: {e:#}", wav_path.display());
        }
        if fox < 4 || peak < 0.02 || !peak.is_finite() {
            failed = true;
        }
        let cos_min = match *dev {
            Device::Gpu | Device::Metal | Device::Mlx | Device::Cuda => 0.3,
            _ => 0.9,
        };
        if cos < cos_min {
            failed = true;
        }
    }

    if failed {
        anyhow::bail!("F5-TTS native backend matrix failed");
    }
    Ok(())
}

fn synth_one(
    model: &std::path::Path,
    dev: Device,
    text: &str,
    reference: &[f32],
    ref_text: &str,
    opts: &InferOpts,
) -> Result<(Vec<f32>, String), String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        F5Native::load_on(model, dev).and_then(|tts| {
            let exec = format!(
                "pre/dec={:?} dit={:?}",
                tts.execution_device(),
                tts.dit_device()
            );
            let pcm = tts.synthesize(text, reference, ref_text, opts)?;
            Ok((pcm, exec))
        })
    })) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(format!("synth err: {e:#}")),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("panic");
            Err(format!("panic: {msg}"))
        }
    }
}

fn parse_devices() -> Vec<(Device, &'static str)> {
    if let Ok(raw) = std::env::var("RLX_DEVICES") {
        return raw
            .split(',')
            .filter_map(|s| {
                let s = s.trim();
                let d = match s.to_ascii_lowercase().as_str() {
                    "cpu" => Device::Cpu,
                    "metal" => Device::Metal,
                    "mlx" => Device::Mlx,
                    "gpu" | "wgpu" => Device::Gpu,
                    "cuda" => Device::Cuda,
                    "vulkan" => Device::Vulkan,
                    "coreml" | "ane" => Device::Ane,
                    _ => return None,
                };
                Some((d, Box::leak(s.to_string().into_boxed_str()) as &str))
            })
            .collect();
    }
    vec![
        (Device::Cpu, "cpu"),
        (Device::Metal, "metal"),
        (Device::Mlx, "mlx"),
        (Device::Gpu, "gpu"),
        (Device::Vulkan, "vulkan"),
        (Device::Ane, "CoreML"),
    ]
}

fn load_whisper() -> WhisperRunner {
    let dir = whisper_dir().expect("Whisper weights");
    WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper")
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        return whisper_ready(&p).then_some(p);
    }
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    for name in ["whisper-base.en", "whisper-tiny.en", "whisper-tiny"] {
        let p = cache.join(name);
        if whisper_ready(&p) {
            return Some(p);
        }
    }
    None
}

fn whisper_ready(dir: &std::path::Path) -> bool {
    dir.join("model.safetensors").is_file() && dir.join("tokenizer.json").is_file()
}

fn whisper_hits(runner: &mut WhisperRunner, pcm_24k: &[f32]) -> usize {
    let pcm = resample_linear(pcm_24k, SAMPLE_RATE, 16_000);
    let Ok(transcript) = runner.transcribe_greedy(&pcm) else {
        return 0;
    };
    let lower = transcript.to_lowercase();
    FOX_WORDS.iter().filter(|w| lower.contains(*w)).count()
}

fn read_wav(path: &std::path::Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path)?;
    let sr = r.spec().sample_rate;
    let ch = r.spec().channels as usize;
    let raw: Vec<f32> = match r.spec().sample_format {
        hound::SampleFormat::Int => {
            let m = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / m)
                .collect()
        }
        hound::SampleFormat::Float => r.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };
    let mono = if ch > 1 {
        raw.chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        raw
    };
    Ok((mono, sr))
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let n = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from_hz as f64 / to_hz as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    d / (na.sqrt() * nb.sqrt() + 1e-12)
}
