//! packed_decode_parity — validate the packed K-quant decode path.
//!
//! The F32 decode dequants K-quant weights to f32 at load; the packed path keeps
//! them U8-packed and dequants inside `Op::DequantMatMul` at runtime. Both feed
//! the same Q4_K values through the same decode graph, so the logits should match
//! to ~float precision (cosine ≈ 1). This runs one decode step each way from the
//! same prefill state and reports cosine + max-abs-diff.
//!
//! Run:
//!   cargo run --release -p rlx-qwen3 --example packed_decode_parity -- \
//!       /Users/Shared/rlx-models/weights/lm/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf

use rlx_qwen3::Qwen3Runner;
use rlx_runtime::Device;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        dot += (a[i] * b[i]) as f64;
        na += (a[i] * a[i]) as f64;
        nb += (b[i] * b[i]) as f64;
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32
}

fn main() -> anyhow::Result<()> {
    let gguf = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf".to_string()
    });
    eprintln!("[parity] GGUF: {gguf}");

    // F32 generator from the GGUF (dequant-at-load); packed_weights(false) keeps
    // the streaming decode generator (packed(true) would build the prefill-only path).
    let runner = Qwen3Runner::builder()
        .weights(&gguf)
        .device(Device::Cpu)
        .packed_weights(false)
        .max_seq(256)
        .build()?;
    let mut generator = runner
        .into_generator()
        .ok_or_else(|| anyhow::anyhow!("no F32 generator (packed build?)"))?;

    // A short arbitrary prompt + a next token — parity is about the logits, not text.
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let next: u32 = 9;

    // --- F32 decode ---
    let _ = generator.prefill_get_last_logits(&prompt)?;
    let l_f32 = generator.decode_get_logits(next)?;

    // --- packed decode (same prefill state) ---
    let _ = generator.prefill_get_last_logits(&prompt)?;
    let n = generator.enable_packed_decode_from_gguf(&gguf)?;
    eprintln!("[parity] packed linears: {n}");
    if n == 0 {
        anyhow::bail!("no linears packed — is this a K-quant GGUF?");
    }
    assert!(generator.has_packed_decode(), "packed decode not enabled");
    let l_packed = generator.decode_get_logits(next)?;

    // --- compare ---
    let cos = cosine(&l_f32, &l_packed);
    let maxabs = l_f32
        .iter()
        .zip(&l_packed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let am_f32 = l_f32.iter().cloned().fold(f32::MIN, f32::max);
    let am_pk = l_packed.iter().cloned().fold(f32::MIN, f32::max);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    eprintln!(
        "[parity] logits: {} dims | cosine={cos:.6} | max|Δ|={maxabs:.4} | argmax f32={} packed={} (max {am_f32:.3}/{am_pk:.3})",
        l_f32.len(),
        argmax(&l_f32),
        argmax(&l_packed),
    );
    if cos > 0.999 && argmax(&l_f32) == argmax(&l_packed) {
        eprintln!("[parity] PASS ✓ (packed decode matches F32; same next-token)");
        Ok(())
    } else {
        anyhow::bail!(
            "PARITY FAIL: cosine {cos:.6}, argmax f32={} packed={}",
            argmax(&l_f32),
            argmax(&l_packed)
        );
    }
}
