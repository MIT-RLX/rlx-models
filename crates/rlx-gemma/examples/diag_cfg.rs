// Diagnostic: what config does rlx-gemma actually load from this GGUF?
//
// The Gemma 4 12B Q4_K_M GGUF has `gemma4.embedding_length = 3840`. If
// `GemmaConfig::hidden_size` ends up anything other than 3840, the graph
// builder will try to RMS-norm a 5376-wide vector over a 3840-wide buffer
// and propagate NaN through every layer (task #50).

use anyhow::{Context, Result};
use rlx_gemma::config::gemma_cfg_from_gguf;
use rlx_gguf::GgufFile;
use std::path::PathBuf;

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .context("usage: diag_cfg <gguf>")?
        .into();
    let f = GgufFile::from_path(&path).context("load gguf")?;
    let cfg = gemma_cfg_from_gguf(&f).context("derive cfg")?;
    println!("derived GemmaConfig:");
    println!("  arch                  = {:?}", cfg.arch);
    println!("  hidden_size           = {}", cfg.hidden_size);
    println!("  intermediate_size     = {}", cfg.intermediate_size);
    println!("  num_hidden_layers     = {}", cfg.num_hidden_layers);
    println!("  num_attention_heads   = {}", cfg.num_attention_heads);
    println!("  num_key_value_heads   = {}", cfg.num_key_value_heads);
    println!("  vocab_size            = {}", cfg.vocab_size);
    println!("  max_position_embedd.  = {}", cfg.max_position_embeddings);
    println!("  head_dim              = {:?}", cfg.head_dim);
    println!("  rms_norm_eps          = {}", cfg.rms_norm_eps);
    println!("  rope_theta            = {}", cfg.rope_theta);
    println!("  tie_word_embeddings   = {}", cfg.tie_word_embeddings);
    println!(
        "  final_logit_softcap   = {:?}",
        cfg.final_logit_softcapping
    );
    println!("  sliding_window        = {:?}", cfg.sliding_window);
    println!("  global_head_dim       = {:?}", cfg.global_head_dim);
    println!(
        "  num_global_kv_heads   = {:?}",
        cfg.num_global_key_value_heads
    );
    for l in [0usize, 4, 5, 6, 11, 17, 23, 47] {
        println!(
            "  layer {l:>2}: is_full={} layer_head_dim={} layer_kv={} k_eq_v={}",
            cfg.is_full_attention_layer(l),
            cfg.layer_head_dim(l),
            cfg.layer_num_kv_heads(l),
            cfg.layer_k_eq_v(l),
        );
    }

    println!(
        "\nactual GGUF tensor shape: token_embd.weight = {:?}",
        f.tensors.get("token_embd.weight").map(|t| &t.shape)
    );
    println!(
        "                          blk.0.ffn_up.weight = {:?}",
        f.tensors.get("blk.0.ffn_up.weight").map(|t| &t.shape)
    );
    println!(
        "                          blk.0.attn_q.weight = {:?}",
        f.tensors.get("blk.0.attn_q.weight").map(|t| &t.shape)
    );
    println!(
        "                          blk.0.attn_k.weight = {:?}",
        f.tensors.get("blk.0.attn_k.weight").map(|t| &t.shape)
    );

    Ok(())
}
