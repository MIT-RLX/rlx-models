//! Loose-dir vs packed `moss-nano.gguf` codes + PCM parity.
//!
//! ```bash
//! cargo run -p rlx-moss-nano --release --example gguf_parity
//! ```

use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};
use rlx_moss_nano::{
    gguf_bundle, MossNative, NativeOpts, DEFAULT_GGUF_NAME, DEFAULT_LOCAL_DIR,
};
use rlx_runtime::Device;

fn peak_norm_cosine(a: &[f32], b: &[f32]) -> f64 {
    let pa = a.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
    let pb = b.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = (a[i] / pa) as f64;
        let y = (b[i] / pb) as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var("RLX_MOSS_DIR").unwrap_or_else(|_| DEFAULT_LOCAL_DIR.to_string()),
    );
    ensure!(
        dir.join("moss_tts_prefill.onnx").is_file(),
        "loose moss-nano dir missing under {}",
        dir.display()
    );
    let text = std::env::var("TEXT").unwrap_or_else(|_| "Hi.".into());
    let max_frames: usize = std::env::var("MAXF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let gguf_out = std::env::var("RLX_MOSS_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.join(DEFAULT_GGUF_NAME));

    eprintln!("[pack] {} → {}", dir.display(), gguf_out.display());
    let report = gguf_bundle::pack_directory(&dir, &gguf_out)?;
    eprintln!(
        "[pack] {} bytes, {} text KV, {} blobs",
        report.bytes, report.file_kv, report.blob_count
    );

    let opts = NativeOpts {
        seed: 0,
        max_frames,
        ..Default::default()
    };
    let loose = MossNative::load_loose(&dir, Device::Cpu)
        .with_context(|| format!("load_loose {}", dir.display()))?;
    let voice = loose
        .voice_names()
        .first()
        .cloned()
        .context("no voices in manifest")?;
    let voice_codes = loose.voice_prompt_codes(&voice)?;

    eprintln!("[loose] generate_codes text={text:?} voice={voice} max_frames={max_frames}");
    let codes_a = loose.generate_codes(&text, &voice_codes, &opts)?;
    let pcm_a = loose.decode_codes(&codes_a)?;

    let from_gguf = gguf_bundle::open_gguf(&gguf_out, Device::Cpu)
        .with_context(|| format!("open_gguf {}", gguf_out.display()))?;
    let voice_codes_b = from_gguf.voice_prompt_codes(&voice)?;
    eprintln!("[gguf] generate_codes (same seed/text)");
    let codes_b = from_gguf.generate_codes(&text, &voice_codes_b, &opts)?;
    let pcm_b = from_gguf.decode_codes(&codes_b)?;

    ensure!(
        codes_a == codes_b,
        "codes mismatch: loose {} frames vs gguf {} frames",
        codes_a.len(),
        codes_b.len()
    );
    ensure!(pcm_a.len() == pcm_b.len(), "PCM length mismatch");
    let cos = peak_norm_cosine(&pcm_a, &pcm_b);
    eprintln!(
        "codes identical ({} frames); peak-norm cosine={cos:.6} (need ≥ 0.999)",
        codes_a.len()
    );
    if cos < 0.999 {
        bail!("PCM cosine {cos:.6} < 0.999");
    }
    println!("ok moss-nano gguf parity");
    Ok(())
}
