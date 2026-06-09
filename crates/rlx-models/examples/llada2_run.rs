// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! LLaDA2 / TIDE block-diffusion inference CLI.

use rlx_cli::parse_llada2_device;
use rlx_models::llada2::{GenerateConfig, LLaDA2Runner, load_llada2_from_dir};
use rlx_models::tide::TideRunner;
use rlx_runtime::Device;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_dir = env::var("LLADA2_MODEL_DIR").or_else(|_| {
        args.iter()
            .position(|a| a == "--model-dir")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .ok_or(env::VarError::NotPresent)
    })?;
    let device = args
        .iter()
        .position(|a| a == "--device")
        .and_then(|i| args.get(i + 1))
        .map(|s| parse_llada2_device(s))
        .transpose()?
        .unwrap_or(Device::Cpu);
    let max_seq: usize = args
        .iter()
        .position(|a| a == "--max-seq")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let prompt: Vec<u32> = args
        .iter()
        .position(|a| a == "--prompt-ids")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or(vec![1, 2, 3]);

    let offload = args.iter().any(|a| a == "--offload");
    let jump_steps: usize = args
        .iter()
        .position(|a| a == "--jump-steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let (cfg, weights) = load_llada2_from_dir(std::path::Path::new(&model_dir))?;
    let mut builder = LLaDA2Runner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(device)
        .batch_seq(1, max_seq);
    if offload {
        builder = builder
            .enable_predictive_expert_offload(128)
            .jump_steps(jump_steps)
            .moe_collect_stats(true);
    }
    let mut runner = TideRunner::from_llada2(builder.build()?);

    let gen_cfg = GenerateConfig::from_model(runner.config());
    let t0 = std::time::Instant::now();
    let (tokens, stats) = runner.generate(&prompt, &gen_cfg)?;
    eprintln!(
        "generated {} tokens in {:.2?} ({} denoise steps recorded)",
        tokens.len(),
        t0.elapsed(),
        stats.len()
    );
    if offload {
        let s = runner.get_offload_stats();
        eprintln!(
            "offload: promotions={} demotions={} gpu_tokens={}",
            s.promotions, s.demotions, s.gpu_tokens
        );
    }
    println!("{:?}", tokens);
    Ok(())
}
