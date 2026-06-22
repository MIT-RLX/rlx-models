// Focused diagnostic for task #50: load the GGUF, dequant
// `token_embd.weight` (Q4_K_M for Gemma 4 12B), and report
// NaN/inf/finite/min/max/mean across the dequanted f32 buffer.
//
// If the embed alone is NaN we know the bug is in Q4K dequant for this
// specific tensor (or its f16 scale/dmin fields). If the embed is finite
// but the model still outputs all-NaN logits (task #50), the bug is
// downstream — sliding-window mask, GQA, softcap, etc.

use anyhow::{Context, Result};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use std::path::PathBuf;

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .context("usage: diag_embed_dequant <gguf>")?
        .into();
    let path_s = path.to_string_lossy().into_owned();

    let mut loader = GgufLoader::from_file(&path_s).context("load gguf")?;
    println!(
        "Loaded {} ({:.1} MB on disk)",
        path_s,
        std::fs::metadata(&path)
            .map(|m| m.len() as f64 / 1.0e6)
            .unwrap_or(0.0)
    );

    let key = "model.embed_tokens.weight";
    println!("Taking {key}...");
    let (data, shape) = loader
        .take(key)
        .with_context(|| format!("loader.take({key})"))?;

    println!(
        "  shape = {shape:?}, len = {} elements ({:.2} GB f32)",
        data.len(),
        data.len() as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0)
    );

    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut n_zero = 0usize;
    let mut n_finite = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0f64;
    let mut abs_max = 0f32;
    for &v in &data {
        if v.is_nan() {
            n_nan += 1;
            continue;
        }
        if v.is_infinite() {
            n_inf += 1;
            continue;
        }
        n_finite += 1;
        if v == 0.0 {
            n_zero += 1;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        abs_max = abs_max.max(v.abs());
        sum += v as f64;
    }
    let mean = if n_finite > 0 {
        sum / n_finite as f64
    } else {
        0.0
    };

    println!("  NaN     = {n_nan}");
    println!("  Inf     = {n_inf}");
    println!("  zero    = {n_zero}");
    println!("  finite  = {n_finite}");
    println!("  min     = {min:.6e}");
    println!("  max     = {max:.6e}");
    println!("  abs_max = {abs_max:.6e}");
    println!("  mean    = {mean:.6e}");

    // Show first row's first 16 values — should be diverse non-zero
    // floats for a meaningful embed.
    let row0_take = 16.min(data.len());
    print!("  row[0][0..{row0_take}] =");
    for v in &data[..row0_take] {
        print!(" {v:+.4}");
    }
    println!();

    if n_nan > 0 || n_inf > 0 {
        println!(
            "\n  >>> Q4K dequant of `{key}` is producing NaN/Inf — Gemma 4 forward path will propagate NaN through to LM head."
        );
        std::process::exit(1);
    } else if n_zero == data.len() {
        println!(
            "\n  >>> Embed is all zeros — token lookup returns zeros, downstream norms blow up."
        );
        std::process::exit(2);
    } else {
        println!("\n  >>> Embed looks finite and nonzero. NaN is introduced downstream.");
    }
    Ok(())
}
