// RLX — versatile ML compiler + runtime. GPLv3.
//! Numeric cross-check of the mlx-affine `GemmaQatLoader` path: dequantize a few
//! representative gemma-3n tensors and dump them, so a Python mlx reference
//! (`scripts/gemma3n_loader_ref.py`) can confirm the affine dequant + tensor
//! naming + bf16 scales/biases decode are bit-exact vs `mlx.core.dequantize`.
//!
//! Run:  cargo run -p rlx-gemma --example mlx_loader_check -- .mlx-test/gemma3n-e2b-4bit

use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;
use rlx_gemma::qat_loader::GemmaQatLoader;
use std::path::PathBuf;

fn dump(tag: &str, v: &[f32]) {
    let head: Vec<f32> = v.iter().take(8).copied().collect();
    let sum: f64 = v.iter().map(|x| *x as f64).sum();
    let finite = v.iter().all(|x| x.is_finite());
    println!(
        "{tag}: len={} finite={finite} sum={sum:.6} head8={head:?}",
        v.len()
    );
}

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| ".mlx-test/gemma3n-e2b-4bit".to_string()),
    );
    let mut ld = GemmaQatLoader::open(&dir)?;

    // 1) A quantized attention projection [out=2048, in=2048].
    let (q, qs) = ld.take("model.layers.0.self_attn.q_proj.weight")?;
    println!("q_proj shape={qs:?}");
    dump("q_proj.row0", &q[..qs[1]]);

    // 2) A quantized MLP down_proj [out=2048, in=8192].
    let (d, ds) = ld.take("model.layers.0.mlp.down_proj.weight")?;
    println!("down_proj shape={ds:?}");
    dump("down_proj.row0", &d[..ds[1]]);

    // 3) The quantized per_layer_model_projection [out=7680, in=2048].
    let (p, ps) = ld.float_tensor("model.per_layer_model_projection.weight")?;
    println!("per_layer_model_projection shape={ps:?}");
    dump("plmp.row0", &p[..ps[1]]);

    // 4) A plain bf16 norm — the loader returns the builder-facing delta gain
    //    `w − 1` (the builder's `gemma_rms` re-adds 1 to match mlx's plain
    //    `nn.RMSNorm(w)`), so this dumps ~1.0 below the raw mlx reference.
    let (n, ns) = ld.take("model.norm.weight")?;
    println!("norm shape={ns:?}");
    dump("norm", &n);

    // 5) Embedding rows for two tokens (grouped-scale affine row-gather).
    let (rows, edim) = ld.dequant_embedding_rows("model.embed_tokens.weight", &[2, 818])?;
    println!("embed dim={edim}");
    dump("embed.tok2", &rows[..edim]);
    dump("embed.tok818", &rows[edim..2 * edim]);

    Ok(())
}
