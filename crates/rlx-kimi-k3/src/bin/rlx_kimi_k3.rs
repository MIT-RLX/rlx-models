// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `rlx-kimi-k3` — inspect / run Kimi-K3 (Moonshot AI).

use anyhow::Result;
use rlx_kimi_k3::config::KimiK3Config;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("");
    if path.is_empty() {
        eprintln!("usage: rlx-kimi-k3 <model-dir-or-config.json>");
        std::process::exit(2);
    }
    let cfg = KimiK3Config::load(path)?;
    let t = &cfg.text_config;
    let kda = (0..t.num_hidden_layers)
        .filter(|&i| t.is_kda_layer(i))
        .count();
    println!(
        "Kimi-K3: {} layers ({kda} KDA + {} MLA), hidden {}, {} experts ({} active + {} shared), vocab {}",
        t.num_hidden_layers,
        t.num_hidden_layers - kda,
        t.hidden_size,
        t.num_experts.unwrap_or(0),
        t.num_experts_per_token,
        t.num_shared_experts,
        t.vocab_size,
    );
    println!(
        "  vision tower: {}",
        if cfg.vision_config.is_some() {
            "yes"
        } else {
            "no"
        }
    );
    Ok(())
}
