//! Generate stories from a trained rlx-tinystories checkpoint.
//!
//! ```text
//! cargo run --release -p rlx-tinystories --bin rlx-tinystories-generate -- \
//!     --model weights/tinystories/tinystories.rlxts --prompt "Once upon a time" --tokens 400
//! ```

use std::path::PathBuf;

use anyhow::{Result, bail};
use rlx_tensor::{Device, is_available};

use rlx_tinystories::checkpoint;
use rlx_tinystories::sample::{GenOptions, generate};

fn main() -> Result<()> {
    let mut model = PathBuf::from("weights/tinystories/tinystories.rlxts");
    let mut prompt = String::from("Once upon a time");
    let mut device: Option<String> = None;
    let mut opts = GenOptions::default();

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let next = |i: &mut usize, argv: &[String], flag: &str| -> Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => model = PathBuf::from(next(&mut i, &argv, "--model")?),
            "--prompt" => prompt = next(&mut i, &argv, "--prompt")?,
            "--tokens" => opts.max_new_tokens = next(&mut i, &argv, "--tokens")?.parse()?,
            "--temp" => opts.temperature = next(&mut i, &argv, "--temp")?.parse()?,
            "--top-k" => opts.top_k = next(&mut i, &argv, "--top-k")?.parse()?,
            "--seed" => opts.seed = next(&mut i, &argv, "--seed")?.parse()?,
            "--device" => device = Some(next(&mut i, &argv, "--device")?),
            "-h" | "--help" => {
                println!(
                    "rlx-tinystories-generate [--model FILE] [--prompt STR] [--tokens N]\n\
                     [--temp F] [--top-k N] [--seed N] [--device cpu|metal]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?} (try --help)"),
        }
        i += 1;
    }

    let dev = match device.as_deref() {
        Some("cpu") => Device::Cpu,
        Some("metal") => Device::Metal,
        _ => {
            if is_available(Device::Metal) {
                Device::Metal
            } else {
                Device::Cpu
            }
        }
    };

    let (cfg, params, bpe) = checkpoint::load(&model)?;
    eprintln!(
        "loaded {} ({} params, ctx {}, {} layers, vocab {}{}) — generating on {dev:?}…",
        model.display(),
        params.len(),
        cfg.block_size,
        cfg.n_layer,
        cfg.vocab,
        if bpe.is_some() {
            ", BPE"
        } else {
            ", byte-level"
        },
    );
    let out = generate(&cfg, &params, &prompt, dev, &opts, bpe.as_ref());
    println!("{out}");
    rlx_tensor::clear_cache();
    Ok(())
}
