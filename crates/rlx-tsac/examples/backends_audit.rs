//! Cross-backend audit of the correct TSAC (= Descript-DAC-44kHz) codec.
//!
//! Measures, on every available RLX backend, the four axes the codec is tuned
//! for — **speed, memory, I/O, precision** — and reports two kinds of parity:
//!   * **bit parity** — sample-wise `max|Δ|` of the decoded PCM vs a reference
//!     backend (0.0 ⇒ bit-identical output);
//!   * **cosine-distance parity** — `1 − (a·b)/(‖a‖‖b‖)` of the decoded PCM,
//!     a scale-invariant view of how far two backends' waveforms diverge.
//!
//! Two parity tests are run so encoder/quantizer divergence is separated from
//! decoder-kernel precision:
//!   1. *end-to-end* — each backend runs its own encode→decode;
//!   2. *decoder-only* — every backend decodes the **reference's** codes, so the
//!      input is identical and only the decode kernels differ.
//!
//! CPU (exact ndarray) is the precision reference when present; it is slow
//! (~0.02× real time), so pass `--trim S` to cap the clip to `S` seconds, or
//! omit `cpu` from the device list to use the first GPU backend as reference.
//!
//! ```bash
//! RLX_DAC_DIR=$PWD/.cache/dac44 cargo run -p rlx-tsac --example backends_audit \
//!   --release --features native-codec,oracle,metal,mlx,gpu -- /tmp/tcmp/in44.wav metal mlx gpu
//! # include the exact CPU reference on a 2 s clip:
//! #   ... -- /tmp/tcmp/in44.wav --trim 2 cpu metal mlx gpu
//! ```
use anyhow::Result;
use rlx_dac::DacCodes;
use rlx_tsac::{SAMPLE_RATE, audio, correct, parse_tsac_device};
use std::time::Instant;

/// Current resident set size of this process in MiB, read from `ps` (RSS in KiB
/// on both macOS and Linux). Kept dependency-free on purpose — adding a crate to
/// the manifest forces a full workspace re-resolve, which trips the pinned rlx-*
/// versions. Unlike `ru_maxrss` this is the *live* RSS, so the per-backend Δ
/// reflects that backend's compiled graphs + params actually resident.
fn rss_mb() -> f64 {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output();
    match out {
        Ok(o) => {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<f64>()
                .unwrap_or(0.0)
                / 1024.0
        }
        Err(_) => 0.0,
    }
}

/// Cosine distance `1 − (a·b)/(‖a‖‖b‖)` in f64 (0 ⇒ identical direction).
fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom <= 1e-30 {
        return 0.0;
    }
    1.0 - dot / denom
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Fraction of RVQ codes identical between two `frames` layouts (1.0 = all).
fn codes_match(a: &DacCodes, b: &DacCodes) -> f64 {
    let frames = a.frames.len().min(b.frames.len());
    let mut same = 0usize;
    let mut total = 0usize;
    for f in 0..frames {
        let q = a.frames[f].len().min(b.frames[f].len());
        for i in 0..q {
            total += 1;
            if a.frames[f][i] == b.frames[f][i] {
                same += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        same as f64 / total as f64
    }
}

struct Run {
    device: String,
    enc_warm_ms: f64,
    dec_warm_ms: f64,
    enc_cold_ms: f64,
    dec_cold_ms: f64,
    codes: DacCodes,
    pcm: Vec<f32>,
    /// Decoder-only PCM: this backend decoding the *reference* backend's codes.
    pcm_from_ref: Vec<f32>,
    container_bytes: usize,
    rss_after_mb: f64,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let in_wav = match args.peek() {
        Some(a) if !a.starts_with("--") => args.next().unwrap(),
        _ => "/tmp/tcmp/in44.wav".into(),
    };

    let mut trim_secs: Option<f64> = None;
    let mut warm_iters = 2usize;
    let mut devices: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--trim" => trim_secs = args.next().and_then(|s| s.parse().ok()),
            "--warm" => warm_iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            other => devices.push(other.to_string()),
        }
    }
    if devices.is_empty() {
        devices = ["cpu", "metal", "mlx", "gpu"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    }

    let (mut pcm, sr) = audio::load_wav_f32(std::path::Path::new(&in_wav), SAMPLE_RATE)?;
    if let Some(s) = trim_secs {
        let n = ((s * SAMPLE_RATE as f64) as usize).min(pcm.len());
        pcm.truncate(n);
    }
    let audio_s = pcm.len() as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "input: {} ({} samples = {audio_s:.2}s @ {sr} Hz)\nwarm iters: {warm_iters}\n",
        in_wav,
        pcm.len()
    );

    // First pass: establish the reference codes/PCM (cpu if requested, else the
    // first device that opens) so decoder-only parity decodes identical input.
    let mut runs: Vec<Run> = Vec::new();
    let mut ref_codes: Option<DacCodes> = None;
    let mut ref_pcm: Option<Vec<f32>> = None;
    let mut ref_device = String::new();

    for dname in &devices {
        let dev = match parse_tsac_device(dname) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("{dname}: unknown device, skipping");
                continue;
            }
        };
        if !rlx_runtime::is_available(dev) {
            eprintln!("{dname}: not available in this build, skipping");
            continue;
        }
        let codec = match correct::open(dev) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{dname}: open failed: {e}");
                continue;
            }
        };

        // Cold (includes one-time graph compile).
        let t = Instant::now();
        let codes = codec.encode_pcm(&pcm, None)?;
        let enc_cold_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let pcm_out = codec.decode_codes(&codes)?;
        let dec_cold_ms = t.elapsed().as_secs_f64() * 1e3;

        // Warm (graph cached): best-of-N.
        let mut enc_warm_ms = f64::MAX;
        let mut dec_warm_ms = f64::MAX;
        for _ in 0..warm_iters.max(1) {
            let t = Instant::now();
            let c = codec.encode_pcm(&pcm, None)?;
            enc_warm_ms = enc_warm_ms.min(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let _ = codec.decode_codes(&c)?;
            dec_warm_ms = dec_warm_ms.min(t.elapsed().as_secs_f64() * 1e3);
        }

        // Lock in the reference from the first successful backend.
        if ref_codes.is_none() {
            ref_codes = Some(clone_codes(&codes));
            ref_pcm = Some(pcm_out.clone());
            ref_device = dname.clone();
        }
        // Decoder-only parity: decode the reference codes on THIS backend.
        let pcm_from_ref = codec.decode_codes(ref_codes.as_ref().unwrap())?;

        // V2 bit-packed container: 21-byte header + codes at 10 bits each.
        let bits = (codec.config().codebook_size as f64).log2().ceil() as usize;
        let n_codes = codes.frames.len() * codes.num_quantizers;
        let container_bytes = 21 + (n_codes * bits).div_ceil(8);
        eprintln!(
            "{dname}: enc {:.0}ms ({:.1}x) dec {:.0}ms ({:.1}x) [cold enc {:.0} dec {:.0}]",
            enc_warm_ms,
            audio_s / (enc_warm_ms / 1e3),
            dec_warm_ms,
            audio_s / (dec_warm_ms / 1e3),
            enc_cold_ms,
            dec_cold_ms,
        );
        runs.push(Run {
            device: dname.clone(),
            enc_warm_ms,
            dec_warm_ms,
            enc_cold_ms,
            dec_cold_ms,
            codes,
            pcm: pcm_out,
            pcm_from_ref,
            container_bytes,
            rss_after_mb: rss_mb(),
        });
    }

    if runs.is_empty() {
        anyhow::bail!("no backends available");
    }
    let ref_pcm = ref_pcm.unwrap();
    let ref_codes = ref_codes.unwrap();

    // ---- Speed ----
    println!("\n=== SPEED (warm; RTF = audio_s / wall_s, >1 faster than real time) ===");
    println!(
        "{:<7} {:>11} {:>8} {:>11} {:>8} {:>10} {:>10}",
        "device", "enc(warm)", "encRTF", "dec(warm)", "decRTF", "encCold", "decCold"
    );
    for r in &runs {
        println!(
            "{:<7} {:>9.0}ms {:>7.1}x {:>9.0}ms {:>7.1}x {:>8.0}ms {:>8.0}ms",
            r.device,
            r.enc_warm_ms,
            audio_s / (r.enc_warm_ms / 1e3),
            r.dec_warm_ms,
            audio_s / (r.dec_warm_ms / 1e3),
            r.enc_cold_ms,
            r.dec_cold_ms,
        );
    }

    // ---- Memory ----
    println!("\n=== MEMORY (process peak RSS high-water after each backend opened) ===");
    let mut prev = 0.0;
    for r in &runs {
        println!(
            "{:<7} peak {:>8.1} MiB   (Δ since previous {:>7.1} MiB)",
            r.device,
            r.rss_after_mb,
            r.rss_after_mb - prev
        );
        prev = r.rss_after_mb;
    }

    // ---- I/O ----
    println!("\n=== I/O (.tsac TSR2 bit-packed container) ===");
    for r in &runs {
        let raw = 20 + r.codes.frames.len() * r.codes.num_quantizers * 2; // legacy u16
        let kbps = (r.container_bytes as f64 * 8.0 / 1000.0) / audio_s;
        let ratio = (pcm.len() as f64 * 4.0) / r.container_bytes as f64; // f32 PCM vs codes
        println!(
            "{:<7} {:>8} bytes  {:>6.2} kbps  {:>4} frames x {} cb  ({:.0}x vs f32 PCM, {:.0}% smaller than u16)",
            r.device,
            r.container_bytes,
            kbps,
            r.codes.frames.len(),
            r.codes.num_quantizers,
            ratio,
            (1.0 - r.container_bytes as f64 / raw as f64) * 100.0,
        );
    }

    // ---- Precision: recon quality vs input ----
    println!("\n=== PRECISION — reconstruction vs input ===");
    for r in &runs {
        let n = pcm.len().min(r.pcm.len());
        let corr = audio::correlation(&pcm[..n], &r.pcm[..n]);
        let cos = cosine_distance(&pcm[..n], &r.pcm[..n]);
        println!(
            "{:<7} recon corr {:.5}  cosDist {:.3e}",
            r.device, corr, cos
        );
    }

    // ---- Precision: cross-backend parity ----
    println!("\n=== PRECISION — cross-backend parity (reference = {ref_device}) ===");
    println!("end-to-end (each backend encodes+decodes its own):");
    println!(
        "{:<7} {:>10} {:>12} {:>12} {:>10}",
        "device", "codesMatch", "pcm max|Δ|", "cosDist", "corr"
    );
    for r in &runs {
        let cm = codes_match(&r.codes, &ref_codes);
        let ma = max_abs(&r.pcm, &ref_pcm);
        let cos = cosine_distance(&r.pcm, &ref_pcm);
        let corr = audio::correlation(&r.pcm, &ref_pcm);
        let tag = if r.device == ref_device {
            "  <ref>"
        } else {
            ""
        };
        println!(
            "{:<7} {:>9.4}% {:>12.3e} {:>12.3e} {:>10.5}{}",
            r.device,
            cm * 100.0,
            ma,
            cos,
            corr,
            tag
        );
    }
    println!(
        "\ndecoder-only (every backend decodes the reference's codes — isolates decode kernels):"
    );
    println!(
        "{:<7} {:>14} {:>14} {:>10}",
        "device", "pcm max|Δ|", "cosDist", "corr"
    );
    for r in &runs {
        let ma = max_abs(&r.pcm_from_ref, &ref_pcm);
        let cos = cosine_distance(&r.pcm_from_ref, &ref_pcm);
        let corr = audio::correlation(&r.pcm_from_ref, &ref_pcm);
        let bit = if ma == 0.0 { "  bit-exact" } else { "" };
        println!(
            "{:<7} {:>14.3e} {:>14.3e} {:>10.5}{}",
            r.device, ma, cos, corr, bit
        );
    }

    Ok(())
}

fn clone_codes(c: &DacCodes) -> DacCodes {
    DacCodes {
        frames: c.frames.clone(),
        num_quantizers: c.num_quantizers,
    }
}
