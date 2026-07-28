//! ChatterBox native (ort-free) across RLX backends → Whisper fox coverage.
//!
//! ```bash
//! just chatterbox-backends
//! ```

use std::path::PathBuf;
use std::time::Instant;

use rlx_chatterbox::{DEFAULT_LOCAL_DIR, NativeChatterBox, SAMPLE_RATE, SynthOpts, peak_amplitude};
use rlx_runtime::{Device, is_available};
use rlx_whisper::WhisperRunner;

const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const FOX_WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn main() -> anyhow::Result<()> {
    let model = std::env::var("RLX_CHATTERBOX_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOCAL_DIR));
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.into());
    let ref_path = std::env::var("RLX_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../rlx-luxtts/tests/fixtures/prompt.wav")
        });
    let (reference, ref_sr) = read_wav(&ref_path)?;
    let opts = SynthOpts {
        greedy: true,
        max_frames: std::env::var("RLX_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128),
        ..Default::default()
    };
    let devices = parse_devices();

    println!("== ChatterBox native cross-backend ==");
    println!("text: {text:?}");
    println!("model={}  ref={}", model.display(), ref_path.display());
    println!(
        "{:<8} {:>8} {:>7} {:>8} {:>8}  exec",
        "backend", "ms", "peak", "fox", "cos"
    );

    let mut cpu_wav: Option<Vec<f32>> = None;
    let mut failed = false;
    let mut whisper: Option<WhisperRunner> = None;

    // Prefer CPU as the cosine reference when present.
    let mut ordered = devices.clone();
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
        let (pcm, exec) = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            NativeChatterBox::load_on(&model, *dev).and_then(|tts| {
                let exec = format!("{dev:?}");
                let pcm = tts.synthesize(&text, &reference, ref_sr, &opts)?;
                Ok((pcm, exec))
            })
        })) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                println!("{label:<8}  synth err: {e:#}");
                failed = true;
                continue;
            }
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("panic");
                println!("{label:<8}  panic: {msg}");
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
        if fox < 5 || peak < 0.03 || !peak.is_finite() {
            failed = true;
        }
        let cos_min = match *dev {
            Device::Gpu | Device::Metal | Device::Mlx => 0.5,
            _ => 0.95,
        };
        if cos < cos_min {
            failed = true;
        }
    }

    if failed {
        anyhow::bail!("ChatterBox native backend matrix failed");
    }
    Ok(())
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
    // CoreML omitted from the default sweep: large HiFT/CFM MIL graphs still
    // fail to load (KernelChannels mismatch). Opt in with RLX_DEVICES=…,ane.
    vec![
        (Device::Cpu, "cpu"),
        (Device::Metal, "metal"),
        (Device::Mlx, "mlx"),
        (Device::Gpu, "gpu"),
        (Device::Vulkan, "vulkan"),
    ]
}

fn load_whisper() -> WhisperRunner {
    let dir = whisper_dir().expect("Whisper weights (RLX_WHISPER_DIR or .cache/whisper-*)");
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
    for name in [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
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
    let transcript = match runner.transcribe_greedy(&pcm) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[whisper] transcribe failed: {e:#}");
            return 0;
        }
    };
    let lower = transcript.to_lowercase();
    let hits = FOX_WORDS.iter().filter(|w| lower.contains(*w)).count();
    eprintln!("[whisper] {transcript:?} → {hits}/6");
    hits
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
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * from_hz as f64 / to_hz as f64;
            let idx = src.floor() as usize;
            let frac = (src - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}
