// RLX — GPLv3. Compare post-final-norm hidden states and lm_head logits
// across CPU / Metal / MLX to localize Metal drift.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

const IDS: &[u32] = &[818, 5279, 529, 7001, 563, 7001];

fn dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME")?;
    Ok(std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots")
        .read_dir()?
        .flatten()
        .next()
        .context("snapshot")?
        .path())
}

fn l2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn run(
    dev: Device,
    lm_head: bool,
    dir: &std::path::Path,
    cfg: &GemmaConfig,
    ple: &[f32],
    idf: &[f32],
    seq: usize,
) -> Result<Vec<f32>> {
    let mut bld = GemmaQatLoader::open(dir)?;
    let mut packed = HashMap::new();
    let (g, p) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        cfg,
        &mut bld,
        1,
        seq,
        lm_head,
        false,
        false,
        &mut packed,
        None,
        None,
    )?;
    let mut c = compile_graph_gemma_prefill_with_params(dev, g, p)?;
    Ok(c.run(&[("input_ids", idf), ("per_layer_inputs", ple)])
        .into_iter()
        .next()
        .unwrap())
}

fn main() -> Result<()> {
    let dir = dir()?;
    let cfg = GemmaConfig::from_file(&dir.join("config.json"))?;
    let loader = GemmaQatLoader::open(&dir)?;
    let bucket = std::env::var("BUCKET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(IDS.len());
    let mut ids = vec![0u32; bucket];
    ids[..IDS.len()].copy_from_slice(IDS);
    let ple = loader.compute_per_layer_inputs(&cfg, &ids)?;
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let h = cfg.hidden_size;

    println!("→ bucket={bucket}, valid tokens={}", IDS.len());

    let cpu_pre = run(Device::Cpu, false, &dir, &cfg, &ple, &ids_f32, bucket)?;

    if std::env::var("RLX_TAP_ALL").ok().is_some() {
        let mut bld = GemmaQatLoader::open(&dir)?;
        let mut packed = HashMap::new();
        let (g, p) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
            &cfg,
            &mut bld,
            1,
            bucket,
            false,
            false,
            false,
            &mut packed,
            None,
            None,
        )?;
        let mut cpu_c = compile_graph_gemma_prefill_with_params(Device::Cpu, g, p)?;
        let cpu_layers = cpu_c.run(&[
            ("input_ids", ids_f32.as_slice()),
            ("per_layer_inputs", ple.as_slice()),
        ]);
        let mut bld = GemmaQatLoader::open(&dir)?;
        let mut packed = HashMap::new();
        let (g, p) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
            &cfg,
            &mut bld,
            1,
            bucket,
            false,
            false,
            false,
            &mut packed,
            None,
            None,
        )?;
        let mut metal_c = compile_graph_gemma_prefill_with_params(Device::Metal, g, p)?;
        let metal_layers = metal_c.run(&[
            ("input_ids", ids_f32.as_slice()),
            ("per_layer_inputs", ple.as_slice()),
        ]);
        let mut bld = GemmaQatLoader::open(&dir)?;
        let mut packed = HashMap::new();
        let (g, p) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
            &cfg,
            &mut bld,
            1,
            bucket,
            false,
            false,
            false,
            &mut packed,
            None,
            None,
        )?;
        let mut mlx_c = compile_graph_gemma_prefill_with_params(Device::Mlx, g, p)?;
        let mlx_layers = mlx_c.run(&[
            ("input_ids", ids_f32.as_slice()),
            ("per_layer_inputs", ple.as_slice()),
        ]);
        println!("→ per-layer hidden L2 vs CPU (last token only, RLX_TAP_ALL)");
        println!("   layer │  Metal L2  │   MLX L2");
        for layer in 0..cfg.num_hidden_layers {
            let ti = IDS.len() - 1;
            let cpu_row = &cpu_layers[1 + layer][ti * h..(ti + 1) * h];
            let metal_row = &metal_layers[1 + layer][ti * h..(ti + 1) * h];
            let mlx_row = &mlx_layers[1 + layer][ti * h..(ti + 1) * h];
            println!(
                "   {layer:>5} │ {:>9.3e} │ {:>9.3e}",
                l2(cpu_row, metal_row),
                l2(cpu_row, mlx_row)
            );
        }
    }

    let mut metal_pre = None;
    let mut mlx_pre = None;
    if rlx_runtime::is_available(Device::Metal) {
        metal_pre = Some(run(
            Device::Metal,
            false,
            &dir,
            &cfg,
            &ple,
            &ids_f32,
            bucket,
        )?);
    }
    if rlx_runtime::is_available(Device::Mlx) {
        mlx_pre = Some(run(Device::Mlx, false, &dir, &cfg, &ple, &ids_f32, bucket)?);
    }

    println!("→ post-final-norm hidden L2 vs CPU (per token); Metal-vs-MLX column");
    for ti in 0..IDS.len() {
        print!("   tok{ti}:");
        let cpu_row = &cpu_pre[ti * h..(ti + 1) * h];
        if let Some(pre) = &metal_pre {
            print!(
                "  Metal↔CPU={:.3e}",
                l2(&pre[ti * h..(ti + 1) * h], cpu_row)
            );
        }
        if let Some(pre) = &mlx_pre {
            print!("  Mlx↔CPU={:.3e}", l2(&pre[ti * h..(ti + 1) * h], cpu_row));
        }
        if let (Some(m), Some(x)) = (&metal_pre, &mlx_pre) {
            print!(
                "  Metal↔Mlx={:.3e}",
                l2(&m[ti * h..(ti + 1) * h], &x[ti * h..(ti + 1) * h])
            );
        }
        println!();
    }
    Ok(())
}
