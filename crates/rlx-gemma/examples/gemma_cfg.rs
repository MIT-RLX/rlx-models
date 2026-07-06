//! Dump the GemmaConfig rlx parses from a GGUF, to compare vs the reference
//! architecture (e.g. Gemma-2-2B: 26 layers, hidden 2304, 8 heads, 4 kv,
//! head_dim 256, query_pre_attn_scalar 256, sliding_window 4096, attn_softcap
//! 50, final_softcap 30, alternating sliding/full).
//!
//!   RLX_GEMMA3_GGUF=<gguf> cargo run -p rlx-gemma --release --example gemma_cfg

use anyhow::Result;

fn main() -> Result<()> {
    let path = std::env::var("RLX_GEMMA3_GGUF").expect("RLX_GEMMA3_GGUF");
    let raw = rlx_gguf::GgufFile::from_path(&path)?;
    let cfg = rlx_gemma::gemma_cfg_from_gguf(&raw)?;
    println!("arch                  = {:?}", cfg.arch);
    println!("vocab_size            = {}", cfg.vocab_size);
    println!("hidden_size           = {}", cfg.hidden_size);
    println!("intermediate_size     = {}", cfg.intermediate_size);
    println!("num_hidden_layers     = {}", cfg.num_hidden_layers);
    println!("num_attention_heads   = {}", cfg.num_attention_heads);
    println!("num_key_value_heads   = {}", cfg.num_key_value_heads);
    println!("head_dim()            = {}", cfg.head_dim());
    println!("query_pre_attn_scalar = {:?}", cfg.query_pre_attn_scalar);
    println!("attn_score_scale()    = {:?}", cfg.attn_score_scale());
    println!("attn_logit_softcap    = {:?}", cfg.attn_logit_softcapping);
    println!("final_logit_softcap   = {:?}", cfg.final_logit_softcapping);
    println!("sliding_window        = {:?}", cfg.sliding_window);
    println!("rope_theta            = {}", cfg.rope_theta);
    println!("rms_norm_eps          = {}", cfg.rms_norm_eps);
    println!("tie_word_embeddings   = {}", cfg.tie_word_embeddings);
    println!("has_ple={} ple_width={}", cfg.has_ple(), cfg.ple_width());
    println!("=== PLE-related tensors present in GGUF ===");
    for (name, t) in raw.tensors.iter() {
        let lname = name.to_lowercase();
        if lname.contains("per_layer")
            || lname.contains("per-layer")
            || lname.contains("_layer_input")
        {
            println!("  {name}  shape={:?} dtype={:?}", t.shape, t.dtype);
        }
    }
    let n = cfg.num_hidden_layers;
    let full: Vec<usize> = (0..n).filter(|&l| cfg.is_full_attention_layer(l)).collect();
    println!("full_attention_layers = {full:?}");
    println!(
        "layer0 head_dim/kv    = {}/{}",
        cfg.layer_head_dim(0),
        cfg.layer_num_kv_heads(0)
    );
    Ok(())
}
