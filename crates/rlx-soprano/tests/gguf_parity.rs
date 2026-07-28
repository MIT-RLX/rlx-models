//! Loose-dir vs packed `soprano.gguf` PCM parity (short fox by default).
//!
//! ```bash
//! cargo test -p rlx-soprano --release --test gguf_parity -- --nocapture
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use rlx_runtime::Device;
use rlx_soprano::{DEFAULT_GGUF_NAME, DEFAULT_LOCAL_DIR, InferOpts, NativeSoprano, gguf_bundle};

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

fn resolve_loose_dir() -> PathBuf {
    if let Ok(p) = std::env::var("RLX_SOPRANO_DIR") {
        return PathBuf::from(p);
    }
    let candidates = [
        PathBuf::from(DEFAULT_LOCAL_DIR),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/soprano"),
    ];
    for p in candidates {
        if p.join("onnx/soprano_backbone_kv_fp32.onnx").is_file() {
            return p;
        }
    }
    PathBuf::from(DEFAULT_LOCAL_DIR)
}

#[test]
fn loose_vs_gguf_pcm_parity_short() -> Result<()> {
    let dir = resolve_loose_dir();
    if !dir.join("onnx/soprano_backbone_kv_fp32.onnx").is_file() {
        eprintln!(
            "skip: loose soprano weights missing under {}",
            dir.display()
        );
        return Ok(());
    }
    let gguf_out = std::env::var("RLX_SOPRANO_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::temp_dir().join(format!("rlx-soprano-parity-{}.gguf", std::process::id()))
        });
    let text = std::env::var("RLX_TEXT")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog.".into());
    let opts = InferOpts {
        max_new_tokens: std::env::var("RLX_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(96),
        greedy: true,
        seed: 1337,
        temperature: 0.3,
        top_p: 0.95,
        ..Default::default()
    };

    eprintln!("[pack] {} → {}", dir.display(), gguf_out.display());
    let report = gguf_bundle::pack_directory(&dir, &gguf_out)?;
    eprintln!(
        "[pack] {} bytes, {} text KV, {} blobs",
        report.bytes, report.file_kv, report.blob_count
    );

    let loose = NativeSoprano::open_loose(&dir, Device::Cpu)
        .with_context(|| format!("open_loose {}", dir.display()))?;
    let pcm_a = loose.synthesize(&text, &opts)?;

    let from_gguf = gguf_bundle::open_gguf(&gguf_out, Device::Cpu)
        .with_context(|| format!("open_gguf {}", gguf_out.display()))?;
    let pcm_b = from_gguf.synthesize(&text, &opts)?;

    ensure!(
        pcm_a.len() == pcm_b.len(),
        "PCM length mismatch: {} vs {}",
        pcm_a.len(),
        pcm_b.len()
    );
    let cos = peak_norm_cosine(&pcm_a, &pcm_b);
    eprintln!(
        "samples={} peak-norm cosine={cos:.6} (need ≥ 0.999)",
        pcm_a.len()
    );
    if cos < 0.999 {
        bail!("PCM cosine {cos:.6} < 0.999");
    }
    // Keep default pack under weights/ when writing to DEFAULT_GGUF_NAME path.
    if gguf_out.file_name().is_some_and(|n| n == DEFAULT_GGUF_NAME) {
        eprintln!("kept {}", gguf_out.display());
    } else {
        let _ = std::fs::remove_file(&gguf_out);
    }
    Ok(())
}
