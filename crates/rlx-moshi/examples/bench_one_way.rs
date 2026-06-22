//! One-way Moshi bench — load + per-frame LM timing.
//!
//! ```sh
//! cargo run -p rlx-moshi --example bench_one_way --release -- \
//!   --prompt "Hello." --max-steps 8 --warmup 2
//! ```

use anyhow::{Context, Result};
use rlx_moshi::{
    GenerationConfig, MoshiCheckpoint, MoshiSession, MoshiVariant, parse_moshi_device,
};
use rlx_runtime::Device;
use std::time::Instant;

fn main() -> Result<()> {
    let mut prompt = "Hello.".to_string();
    let mut max_steps = 8usize;
    let mut warmup = 2usize;
    let mut moshi_dir = rlx_moshi::default_moshi_dir();
    let mut mimi_dir = rlx_moshi::default_mimi_dir();
    let mut device = Device::Cpu;
    let mut checkpoint = MoshiCheckpoint::from_env_or_default();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--max-steps" => {
                i += 1;
                max_steps = args[i].parse()?;
            }
            "--warmup" => {
                i += 1;
                warmup = args[i].parse()?;
            }
            "--model-dir" => {
                i += 1;
                moshi_dir = std::path::PathBuf::from(&args[i]);
            }
            "--mimi-dir" => {
                i += 1;
                mimi_dir = std::path::PathBuf::from(&args[i]);
            }
            "--device" => {
                i += 1;
                device = parse_moshi_device(&args[i]).context("--device")?;
            }
            "--checkpoint" => {
                i += 1;
                checkpoint = MoshiCheckpoint::parse(&args[i]).context("--checkpoint")?;
            }
            other => anyhow::bail!("unknown arg {other}"),
        }
        i += 1;
    }

    let t_load = Instant::now();
    let mut session = MoshiSession::open_with_checkpoint(
        &moshi_dir,
        &mimi_dir,
        MoshiVariant::MoshikoOneWay,
        device,
        checkpoint,
    )?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let cfg = GenerationConfig {
        max_steps,
        text_seed: 42,
        audio_seed: 43,
        ..GenerationConfig::default()
    };

    if warmup > 0 {
        let mut warm = session;
        let wcfg = GenerationConfig {
            max_steps: warmup,
            text_seed: 42,
            audio_seed: 43,
            ..GenerationConfig::default()
        };
        let _ = warm.generate_one_way(&prompt, &wcfg)?;
        session = warm;
    }

    let t_gen = Instant::now();
    let result = session.generate_one_way(&prompt, &cfg)?;
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

    let audio_s = result.samples.len() as f64 / result.sample_rate as f64;
    let out_frames = result.audio_frames.len().max(1);
    let ms_per_frame = gen_ms / out_frames as f64;
    let rtf = if audio_s > 0.0 {
        (gen_ms / 1000.0) / audio_s
    } else {
        0.0
    };

    let backend = if session.device() == Device::Cpu {
        "rlx-moshi-cpu-eager"
    } else {
        "rlx-moshi-gpu-candle"
    };
    println!("backend={backend}");
    println!("device={:?}", session.device());
    println!("load_ms={load_ms:.1}");
    println!("gen_ms={gen_ms:.1}");
    println!("max_steps={max_steps}");
    println!("out_frames={out_frames}");
    println!("out_samples={}", result.samples.len());
    println!("audio_s={audio_s:.3}");
    println!("ms_per_frame={ms_per_frame:.1}");
    println!("rtf={rtf:.2}");
    println!("transcript={}", result.transcript);
    Ok(())
}
