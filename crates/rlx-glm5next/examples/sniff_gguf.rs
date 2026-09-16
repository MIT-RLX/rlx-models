// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Read a `glm5next` GGUF's metadata and print the resolved config.
//!
//! Point it at the *first* shard of a split set — that shard carries every
//! metadata key and no tensor data, so this reads ~9 MB rather than 93 GB:
//!
//! ```text
//! cargo run --release -p rlx-glm5next --example sniff_gguf -- \
//!     GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf
//! ```

use anyhow::{Result, bail};
use rlx_glm5next::Glm5NextConfig;
use rlx_glm5next::config::AttnKind;

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        bail!("usage: sniff_gguf <glm5next.gguf>");
    };
    let c = Glm5NextConfig::from_gguf_path(&path)?;

    println!(
        "hidden {} · vocab {} · ctx {}",
        c.hidden_size, c.vocab_size, c.max_position_embeddings
    );
    println!(
        "{} decoder layers (+{} MTP) = {} blocks",
        c.num_hidden_layers,
        c.num_nextn_predict_layers,
        c.block_count()
    );
    let kda = c
        .layer_types
        .iter()
        .filter(|k| **k == AttnKind::Kda)
        .count();
    println!(
        "attention: {kda} KDA ({}×{}, conv {}, lower bound {:?}) + {} MLA/DSA",
        c.linear_num_heads,
        c.linear_head_dim,
        c.linear_conv_kernel_dim,
        c.linear_lower_bound,
        c.num_hidden_layers - kda,
    );
    println!(
        "MLA: q_lora {} · kv_lora {} · nope {} · v {} · rope {} ({})",
        c.q_lora_rank,
        c.kv_lora_rank,
        c.qk_nope_head_dim,
        c.v_head_dim,
        c.qk_rope_head_dim,
        if c.qk_rope_head_dim == 0 {
            "NoPE"
        } else {
            "RoPE"
        },
    );
    println!(
        "DSA: {} heads × {} · topk {} · kpool {} · dense below {} tokens",
        c.index_n_heads, c.index_head_dim, c.index_topk, c.index_kpool, c.index_topk,
    );
    println!(
        "mHC: {} streams · {} Sinkhorn iters · eps {:e}",
        c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps
    );
    println!(
        "MoE: {} experts, {} active + {} shared · inter {} · scale {} · {} leading dense",
        c.n_routed_experts,
        c.num_experts_per_tok,
        c.n_shared_experts,
        c.moe_intermediate_size,
        c.routed_scaling_factor,
        c.first_k_dense_replace,
    );
    println!("swiglu clamp: {}", c.swiglu_limit);
    Ok(())
}
