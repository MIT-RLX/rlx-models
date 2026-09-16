//! Dump rlx's `J_l` for given token ids, for comparison against the reference.
//!
//! Takes explicit token ids rather than text so a tokenizer difference cannot be
//! mistaken for an estimator difference — `scripts/reference_jacobian.py` prints
//! the ids it used.
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen3 --release --example dump_qwen3_jacobians -- \
//!     --ids 785,6722,315,9625,374,12095,323,279,6722,315,6323,374,26194,13 \
//!     --layers 3,4,5,6 --target 6 --dim-batch 8 --skip-first 1 --out /tmp/rlx.safetensors
//! ```

#![cfg(feature = "qwen3")]

use anyhow::{Context, Result, bail};
use rlx_jlens::models::qwen3::Qwen3LensModel;
use rlx_jlens::{FitConfig, JacobianLens, LensModel, StackLens};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let mut ids: Vec<f32> = Vec::new();
    let mut layers: Vec<usize> = vec![3, 4, 5, 6];
    let mut target = 6usize;
    let mut dim_batch = 8usize;
    let mut skip_first = 1usize;
    let mut out = "/tmp/rlx.safetensors".to_string();
    let mut dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../weights/Qwen3-0.6B")
        .to_string_lossy()
        .to_string();
    let mut device = Device::Cpu;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String> {
            argv.get(i + 1)
                .cloned()
                .with_context(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--ids" => {
                ids = need(i)?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().map(|v| v as f32))
                    .collect::<std::result::Result<_, _>>()?;
                i += 2;
            }
            "--layers" => {
                layers = need(i)?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<std::result::Result<_, _>>()?;
                i += 2;
            }
            "--target" => {
                target = need(i)?.parse()?;
                i += 2;
            }
            "--dim-batch" => {
                dim_batch = need(i)?.parse()?;
                i += 2;
            }
            "--skip-first" => {
                skip_first = need(i)?.parse()?;
                i += 2;
            }
            "--out" => {
                out = need(i)?;
                i += 2;
            }
            "--model" => {
                dir = need(i)?;
                i += 2;
            }
            "--device" => {
                device = rlx_runtime::parse_device(&need(i)?)?;
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
    }
    if ids.is_empty() {
        bail!("pass --ids");
    }

    let model = Qwen3LensModel::open(&dir)?.with_name("qwen3");
    let seq = ids.len();
    eprintln!(
        "{} layers, d_model {}, seq {seq}, target {target}, dim_batch {dim_batch}, skip_first {skip_first}",
        model.n_layers(),
        model.d_model()
    );

    let cfg = FitConfig {
        dim_batch,
        skip_first,
        device,
    };
    let mut lens = StackLens::new(&model, &layers, target, seq, cfg)?;
    let batched = lens.replicate_tokens(&ids)?;
    let start = std::time::Instant::now();
    let js = lens.jacobians(&batched)?;
    eprintln!(
        "fitted in {:.1}s | {}",
        start.elapsed().as_secs_f64(),
        lens.timing().summary()
    );

    let map: std::collections::BTreeMap<usize, _> =
        layers.iter().copied().zip(js.iter().cloned()).collect();
    JacobianLens::new(map, 1, target)?.save(&out)?;
    eprintln!("wrote {out}");
    Ok(())
}
