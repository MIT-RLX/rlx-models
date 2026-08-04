//! Confirm + identify the MLA weight sparsity the dataflow recording flagged
//! (48 sites, density 0.190 = 81% zero, 2 per MLA layer). Loads a real MLA layer
//! and prints the zero-fraction of each projection weight.
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::mla::{MlaDims, MlaWeights};
use std::path::Path;

fn dens(v: &[f32]) -> (f64, usize) {
    let nz = v.iter().filter(|&&x| x != 0.0).count();
    (nz as f64 / v.len().max(1) as f64, v.len())
}

fn main() -> Result<(), String> {
    let md = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Volumes/FOUR/kimi".into());
    if !Path::new(&md).join("config.json").exists() {
        eprintln!("skip: not mounted");
        return Ok(());
    }
    let kc = KimiK3Config::load(Path::new(&md).join("config.json")).map_err(|e| e.to_string())?;
    let hidden = kc.text_config.hidden_size;
    let d = MlaDims {
        hidden,
        num_heads: 96,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        eps: 1e-5,
        batch: 1,
        seq: 1,
    };
    let mut ck = CheckpointLoader::open(&md).map_err(|e| e.to_string())?;
    // layer 3 is the first MLA layer.
    let w: MlaWeights = ck
        .load_mla("language_model.model.layers.3", d)
        .map_err(|e| e.to_string())?;
    eprintln!("\nMLA layer-3 per-projection weight DENSITY (nonzero fraction):");
    for (name, v) in [
        ("q_a_proj", &w.q_a_proj),
        ("q_b_proj", &w.q_b_proj),
        ("kv_a_proj_with_mqa", &w.kv_a_proj_with_mqa),
        ("kv_b_proj", &w.kv_b_proj),
        ("g_proj", &w.g_proj),
        ("o_proj", &w.o_proj),
    ] {
        let (dn, n) = dens(v);
        eprintln!(
            "  {name:<20} density {dn:.3}  ({:.0}% zero, {n} elems)",
            100.0 * (1.0 - dn)
        );
    }
    Ok(())
}
