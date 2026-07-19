// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

use std::path::PathBuf;

use anyhow::{Context, Result};
use rlx_metavoice::{
    DEFAULT_ENCODEC_PATH, DEFAULT_LOCAL_DIR, InferOpts, MetaVoice, parse_device, peak_amplitude,
};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let mut text: Option<String> = None;
    let mut reference: Option<PathBuf> = None;
    let mut output = PathBuf::from("metavoice.wav");
    let mut model_dir = PathBuf::from(DEFAULT_LOCAL_DIR);
    let mut encodec = PathBuf::from(DEFAULT_ENCODEC_PATH);
    let mut device = Device::Cpu;
    let mut opts = InferOpts::default();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--text" => text = Some(args.next().context("missing --text")?),
            "--reference" => {
                reference = Some(PathBuf::from(args.next().context("missing --reference")?))
            }
            "--output" | "-o" => output = PathBuf::from(args.next().context("missing --output")?),
            "--model-dir" | "--weights" => {
                model_dir = PathBuf::from(args.next().context("missing --model-dir")?)
            }
            "--encodec" => encodec = PathBuf::from(args.next().context("missing --encodec")?),
            "--max-tokens" => {
                opts.max_new_tokens = args
                    .next()
                    .context("missing --max-tokens")?
                    .parse()
                    .context("--max-tokens")?
            }
            "--guidance" => {
                opts.guidance_scale = args
                    .next()
                    .context("missing --guidance")?
                    .parse()
                    .context("--guidance")?
            }
            "--temperature" | "--temp" => {
                opts.temperature = args
                    .next()
                    .context("missing --temperature")?
                    .parse()
                    .context("--temperature")?;
                opts.greedy = false;
            }
            "--top-p" => {
                opts.top_p = args
                    .next()
                    .context("missing --top-p")?
                    .parse()
                    .context("--top-p")?;
                opts.greedy = false;
            }
            "--seed" => {
                opts.seed = args
                    .next()
                    .context("missing --seed")?
                    .parse()
                    .context("--seed")?
            }
            "--greedy" => opts.greedy = true,
            "--sample" => opts.greedy = false,
            "--device" => {
                device = parse_device(&args.next().context("missing --device")?)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            "-h" | "--help" => {
                println!("Usage: rlx-metavoice --text TEXT [--output FILE]");
                println!("  --model-dir DIR   default: {DEFAULT_LOCAL_DIR}");
                println!("  --encodec PATH    default: {DEFAULT_ENCODEC_PATH}");
                println!("  --reference WAV   speaker clone (default: bria_16k.wav)");
                println!("  --max-tokens N    first-stage AR steps (default 864)");
                println!("  --guidance F      CFG scale (default 3.0)");
                println!("  --greedy          argmax decode (default)");
                println!("  --sample          top-p sampling (disables greedy)");
                println!("  --temperature F   sampling temperature (implies --sample)");
                println!("  --top-p F         nucleus (implies --sample, default 0.95)");
                println!("  --seed N          RNG seed when sampling (default 1337)");
                println!("  --device NAME     cpu|metal|mlx|…");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }
    let text = text.context("--text is required")?;

    let t0 = std::time::Instant::now();
    let mv = MetaVoice::open_with_encodec(&model_dir, &encodec, device)
        .with_context(|| format!("open MetaVoice at {}", model_dir.display()))?;
    eprintln!(
        "[load {:?}] layers={} spk_tensors≈{} greedy={} max_tokens={} seed={}",
        t0.elapsed(),
        mv.first_args().n_layer,
        mv.weight_counts().1,
        opts.greedy,
        opts.max_new_tokens,
        opts.seed
    );

    let t1 = std::time::Instant::now();
    let pcm = mv.synthesize(&text, reference.as_deref(), &opts)?;
    let secs = pcm.len() as f32 / mv.sample_rate() as f32;
    eprintln!(
        "[synth {:?}] {} samples = {:.2}s @ {}Hz, peak {:.3}",
        t1.elapsed(),
        pcm.len(),
        secs,
        mv.sample_rate(),
        peak_amplitude(&pcm)
    );
    mv.write_wav(&pcm, &output)?;
    println!("Wrote {} samples to {}", pcm.len(), output.display());
    Ok(())
}
