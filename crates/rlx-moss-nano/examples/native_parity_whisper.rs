//! Full cross-backend validation: synthesize the SAME utterance NATIVELY on CPU
//! and on DEVICE, then report (a) time-domain cosine, (b) spectral convergence +
//! log-spectral distance, and (c) Whisper transcripts for BOTH renders vs the
//! input text (round-trip: same text in → same text out on every backend).
//! `RLX_MOSS_DIR=... RLX_WHISPER_DIR=... DEVICE=metal TEXT="..." MAXF=32 \
//!  cargo run -p rlx-moss-nano --example native_parity_whisper --no-default-features --features metal`
use rlx_moss_nano::{MossNative, NativeOpts};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};
use std::path::{Path, PathBuf};

fn parse_device(s: &str) -> Device {
    match s.to_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        "ane" | "coreml" => Device::Ane,
        "cuda" => Device::Cuda,
        _ => Device::Cpu,
    }
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    d / (na.sqrt() * nb.sqrt())
}

fn mono(inter: &[f32], ch: usize) -> Vec<f32> {
    if ch <= 1 {
        return inter.to_vec();
    }
    let n = inter.len() / ch;
    (0..n)
        .map(|i| (0..ch).map(|c| inter[i * ch + c]).sum::<f32>() / ch as f32)
        .collect()
}

/// Naive real-input DFT magnitude spectrogram (Hann, nfft, hop). Fine for a short
/// validation clip; avoids pulling an FFT dep into the example.
fn stft_mag(x: &[f32], nfft: usize, hop: usize) -> Vec<Vec<f64>> {
    let win: Vec<f64> = (0..nfft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / nfft as f64).cos())
        .collect();
    let bins = nfft / 2 + 1;
    let mut frames = Vec::new();
    let mut start = 0;
    while start + nfft <= x.len() {
        let mut mag = vec![0f64; bins];
        for (k, m) in mag.iter_mut().enumerate() {
            let (mut re, mut im) = (0f64, 0f64);
            let w = 2.0 * std::f64::consts::PI * k as f64 / nfft as f64;
            for n in 0..nfft {
                let s = x[start + n] as f64 * win[n];
                re += s * (w * n as f64).cos();
                im -= s * (w * n as f64).sin();
            }
            *m = (re * re + im * im).sqrt();
        }
        frames.push(mag);
        start += hop;
    }
    frames
}

/// Spectral convergence  = ‖|A|−|B|‖_F / ‖|A|‖_F  and
/// log-spectral distance = RMS over all bins of |log|A|−log|B||.
fn spectral_metrics(a: &[f32], b: &[f32], rate: usize) -> (f64, f64) {
    // downsample to ~16k mono for a stable, rate-independent comparison
    let target = 16_000usize;
    let ds = |x: &[f32]| -> Vec<f32> {
        let n = (x.len() as u64 * target as u64 / rate as u64).max(1) as usize;
        (0..n)
            .map(|i| {
                let s = i as f64 * rate as f64 / target as f64;
                let idx = s.floor() as usize;
                let f = (s - idx as f64) as f32;
                let p = x[idx.min(x.len() - 1)];
                let q = x[(idx + 1).min(x.len() - 1)];
                p + (q - p) * f
            })
            .collect()
    };
    let (a, b) = (ds(a), ds(b));
    let (sa, sb) = (stft_mag(&a, 512, 128), stft_mag(&b, 512, 128));
    let nf = sa.len().min(sb.len());
    let (mut num, mut den, mut lsd_acc, mut lsd_n) = (0f64, 0f64, 0f64, 0usize);
    for f in 0..nf {
        for k in 0..sa[f].len().min(sb[f].len()) {
            let (x, y) = (sa[f][k], sb[f][k]);
            num += (x - y).powi(2);
            den += x * x;
            let lx = (x + 1e-8).ln();
            let ly = (y + 1e-8).ln();
            lsd_acc += (lx - ly).powi(2);
            lsd_n += 1;
        }
    }
    let sc = if den > 0.0 {
        (num.sqrt()) / (den.sqrt())
    } else {
        0.0
    };
    let lsd = if lsd_n > 0 {
        (lsd_acc / lsd_n as f64).sqrt()
    } else {
        0.0
    };
    (sc, lsd)
}

fn transcribe(wd: &Path, audio: &[f32], ch: usize, rate: u32) -> anyhow::Result<String> {
    let m = mono(audio, ch);
    let n = (m.len() as u64 * WR as u64 / rate as u64).max(1) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let s = i as f64 * rate as f64 / WR as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let p = m[idx.min(m.len() - 1)];
            let q = m[(idx + 1).min(m.len() - 1)];
            p + (q - p) * f
        })
        .collect();
    let mut w = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    w.transcribe_greedy(&pcm)
}

fn coverage(text: &str, heard: &str) -> f32 {
    let (refs, got) = (words(text), words(heard));
    if refs.is_empty() {
        return 0.0;
    }
    let hits = refs
        .iter()
        .filter(|x| got.iter().any(|h| h == *x || h.contains(x.as_str())))
        .count();
    hits as f32 / refs.len() as f32
}

fn main() -> anyhow::Result<()> {
    let dir =
        PathBuf::from(std::env::var("RLX_MOSS_DIR").unwrap_or("weights/tts/moss-nano".into()));
    let wd =
        PathBuf::from(std::env::var("RLX_WHISPER_DIR").unwrap_or(".cache/whisper-tiny".into()));
    let text =
        std::env::var("TEXT").unwrap_or("The quick brown fox jumps over the lazy dog.".into());
    let maxf: usize = std::env::var("MAXF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let seed: u64 = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let device = parse_device(&std::env::var("DEVICE").unwrap_or("metal".into()));
    let opts = NativeOpts {
        seed,
        max_frames: maxf,
        ..Default::default()
    };

    let cpu = MossNative::load_on(&dir, Device::Cpu)?;
    let gpu = MossNative::load_on(&dir, device)?;
    let voice = cpu.voice_names()[0].clone();

    let t0 = std::time::Instant::now();
    let a_cpu = cpu.synthesize(&text, &voice, &opts)?;
    let t_cpu = t0.elapsed().as_secs_f32();
    let t1 = std::time::Instant::now();
    let a_gpu = gpu.synthesize(&text, &voice, &opts)?;
    let t_gpu = t1.elapsed().as_secs_f32();

    let ch = cpu.channels() as usize;
    let rate = cpu.sample_rate();
    let secs = a_cpu.len() as f32 / (rate as f32 * ch as f32);

    let c = cos(&a_cpu, &a_gpu);
    let (sc, lsd) = spectral_metrics(&a_cpu, &a_gpu, rate as usize);
    let rtf_cpu = t_cpu / secs;
    let rtf_gpu = t_gpu / secs;

    println!("── moss-nano native parity: CPU vs {} ──", device.name());
    println!(
        "audio {secs:.2}s  |  CPU {t_cpu:.1}s (RTF {rtf_cpu:.2})  {} {t_gpu:.1}s (RTF {rtf_gpu:.2})  speedup {:.2}×",
        device.name(),
        t_cpu / t_gpu.max(1e-6)
    );
    println!("time-domain cosine      = {c:.6}   (1.0 = identical)");
    println!("spectral convergence    = {sc:.6}   (0.0 = identical)");
    println!("log-spectral distance   = {lsd:.6}   (0.0 = identical)");

    let tr_cpu = transcribe(&wd, &a_cpu, ch, rate)?;
    let tr_gpu = transcribe(&wd, &a_gpu, ch, rate)?;
    println!("── whisper round-trip ──");
    println!("input          : {text}");
    println!(
        "CPU   heard    : {tr_cpu}   (coverage {:.2})",
        coverage(&text, &tr_cpu)
    );
    println!(
        "{}  heard  : {tr_gpu}   (coverage {:.2})",
        device.name(),
        coverage(&text, &tr_gpu)
    );
    println!("transcripts identical = {}", tr_cpu.trim() == tr_gpu.trim());
    Ok(())
}
