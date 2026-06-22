// Host-side scalar forward of the first Gemma 4 layer — print stats
// (nan/inf/min/mean/max) at every stage so we can localize where the
// rlx-gemma graph diverges from llama.cpp (task #50).
//
// We replicate llama.cpp/src/models/gemma4.cpp lines 215-365 in plain
// Rust, using the actual GGUF weights. If a step here produces a
// finite tensor that the rlx graph path produces NaN for, we know the
// rlx graph builder is mis-emitting that op. If a step produces NaN
// here too, the bug is in our understanding of the math (e.g. wrong
// formula or wrong tensor layout).
//
// Build:
//   cargo run -p rlx-gemma --release --example diag_layer_trace -- \
//     <gguf>

use anyhow::{Context, Result};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use std::path::PathBuf;

fn stats(label: &str, x: &[f32]) {
    let mut nan = 0;
    let mut inf = 0;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut absmax = 0f32;
    let mut n_finite = 0;
    for &v in x {
        if v.is_nan() {
            nan += 1;
            continue;
        }
        if v.is_infinite() {
            inf += 1;
            continue;
        }
        n_finite += 1;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        absmax = absmax.max(v.abs());
        sum += v as f64;
        sumsq += (v as f64) * (v as f64);
    }
    let mean = if n_finite > 0 {
        sum / n_finite as f64
    } else {
        0.0
    };
    let _var = if n_finite > 0 {
        sumsq / n_finite as f64 - mean * mean
    } else {
        0.0
    };
    let rms = (sumsq / n_finite.max(1) as f64).sqrt();
    println!(
        "{label:35} len={:>9} nan={nan:>3} inf={inf:>3} min={min:+.3e} max={max:+.3e} mean={mean:+.3e} rms={rms:.3e} absmax={absmax:+.3e}",
        x.len()
    );
}

// RMS norm with optional learnable gamma (gemma adds 1.0 to weight).
fn rms_norm(x: &[f32], gamma: Option<&[f32]>, gemma_add_one: bool, eps: f32) -> Vec<f32> {
    let h = x.len();
    let sumsq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = (sumsq / h as f32 + eps).sqrt().recip();
    let mut out = vec![0f32; h];
    for i in 0..h {
        let g = match gamma {
            Some(g) if gemma_add_one => 1.0 + g[i],
            Some(g) => g[i],
            None => 1.0,
        };
        out[i] = x[i] * inv_rms * g;
    }
    out
}

// Matmul x[k] @ W^T where W is [n, k] row-major. Output [n].
fn matvec_t(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for j in 0..n {
        let row = &w[j * k..(j + 1) * k];
        let mut acc = 0f32;
        for i in 0..k {
            acc += row[i] * x[i];
        }
        out[j] = acc;
    }
    out
}

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .context("usage: diag_layer_trace <gguf>")?
        .into();
    let path_s = path.to_string_lossy().into_owned();

    println!("Loading {path_s}");
    let mut loader = GgufLoader::from_file(&path_s).context("load gguf")?;

    // Constants for Gemma 4 12B from inspect_gguf
    let hidden = 3840;
    let n_heads = 16;
    let n_kv_heads_swa = 8;
    let head_dim_swa = 256; // SWA layers (blk.0 is SWA)
    let eps = 1e-6f32;

    // Use token id 2 = <bos> (Gemma 4)
    let token_id: usize = 2;

    println!("\n=== STAGE 0: embed lookup for token_id={token_id} ===");
    let (embed_table, embed_shape) = loader
        .take("model.embed_tokens.weight")
        .context("take embed")?;
    println!(
        "embed_table shape = {embed_shape:?}, len = {}",
        embed_table.len()
    );
    // GGUF stores embed as [hidden, vocab] in GGML order — innermost
    // dim is hidden. So row token_id is at offset token_id * hidden.
    // Earlier diag_embed_dequant showed shape [262144, 3840] (in HF
    // order [vocab, hidden] → GGML innermost is hidden=3840).
    let mut x: Vec<f32> = embed_table[token_id * hidden..(token_id + 1) * hidden].to_vec();
    stats("0a. embed[token=2]", &x);

    // Gemma 4 scales embed by sqrt(n_embd)
    let scale = (hidden as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
    stats("0b. embed * sqrt(hidden)", &x);

    println!("\n=== STAGE 1: input_layernorm (blk.0.attn_norm) ===");
    let (attn_norm_w, _) = loader
        .take("model.layers.0.input_layernorm.weight")
        .context("take attn_norm")?;
    stats("1a. attn_norm.weight", &attn_norm_w);
    let x_normed = rms_norm(&x, Some(&attn_norm_w), true, eps);
    stats("1b. input_layernorm(x)", &x_normed);

    println!("\n=== STAGE 2: Q projection (blk.0.attn_q) ===");
    let (q_w, q_shape) = loader
        .take("model.layers.0.self_attn.q_proj.weight")
        .context("take q_proj")?;
    println!("q_proj shape = {q_shape:?}, len = {}", q_w.len());
    // GGUF [hidden, q_dim] in GGML order → matvec expects W as [n_out, n_in]
    // q_w is stored as `[q_dim, hidden]` row-major (n_out=q_dim, n_in=hidden).
    let q_dim = n_heads * head_dim_swa; // 4096
    let q = matvec_t(&x_normed, &q_w, hidden, q_dim);
    stats("2a. Q = x_normed @ Wq^T", &q);

    println!("\n=== STAGE 3: Q reshape + per-head Q-norm (blk.0.attn_q_norm) ===");
    let (q_norm_w, q_norm_shape) = loader
        .take("model.layers.0.self_attn.q_norm.weight")
        .context("take q_norm")?;
    println!("q_norm shape = {q_norm_shape:?}");
    stats("3a. q_norm.weight", &q_norm_w);
    let mut q_normed = vec![0f32; q.len()];
    for h_idx in 0..n_heads {
        let head_slice = &q[h_idx * head_dim_swa..(h_idx + 1) * head_dim_swa];
        let head_normed = rms_norm(head_slice, Some(&q_norm_w), true, eps);
        q_normed[h_idx * head_dim_swa..(h_idx + 1) * head_dim_swa].copy_from_slice(&head_normed);
    }
    stats("3b. Q after per-head q_norm", &q_normed);

    println!("\n=== STAGE 4: K projection + K-norm ===");
    let (k_w, k_shape) = loader
        .take("model.layers.0.self_attn.k_proj.weight")
        .context("take k_proj")?;
    println!("k_proj shape = {k_shape:?}");
    let kv_dim = n_kv_heads_swa * head_dim_swa; // 2048
    let k = matvec_t(&x_normed, &k_w, hidden, kv_dim);
    stats("4a. K = x_normed @ Wk^T", &k);
    let (k_norm_w, _) = loader
        .take("model.layers.0.self_attn.k_norm.weight")
        .context("take k_norm")?;
    let mut k_normed = vec![0f32; k.len()];
    for h_idx in 0..n_kv_heads_swa {
        let head_slice = &k[h_idx * head_dim_swa..(h_idx + 1) * head_dim_swa];
        let head_normed = rms_norm(head_slice, Some(&k_norm_w), true, eps);
        k_normed[h_idx * head_dim_swa..(h_idx + 1) * head_dim_swa].copy_from_slice(&head_normed);
    }
    stats("4b. K after per-head k_norm", &k_normed);

    println!("\n=== STAGE 5: V projection + V-norm (no learnable gamma) ===");
    let (v_w, v_shape) = loader
        .take("model.layers.0.self_attn.v_proj.weight")
        .context("take v_proj")?;
    println!("v_proj shape = {v_shape:?}");
    let v = matvec_t(&x_normed, &v_w, hidden, kv_dim);
    stats("5a. V = x_normed @ Wv^T", &v);
    let mut v_normed = vec![0f32; v.len()];
    for h_idx in 0..n_kv_heads_swa {
        let head_slice = &v[h_idx * head_dim_swa..(h_idx + 1) * head_dim_swa];
        let head_normed = rms_norm(head_slice, None, false, eps);
        v_normed[h_idx * head_dim_swa..(h_idx + 1) * head_dim_swa].copy_from_slice(&head_normed);
    }
    stats("5b. V after V-norm (no scale)", &v_normed);

    println!("\nAll stages of layer 0 attention pre-RoPE computed.");
    println!("If every line above shows nan=0 inf=0 with reasonable magnitudes,");
    println!("the bug is in steps after RoPE: SDPA mask/softmax, o_proj, or");
    println!("post-attention residual add. If any line shows nan/inf, that's");
    println!("the first divergence point.");

    Ok(())
}
