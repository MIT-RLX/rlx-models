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

//! `rlx-vlash` CLI — run a VLASH π₀ / π₀.₅ policy on an image + robot state +
//! language instruction and print the predicted action chunk.
//!
//! ```text
//!   rlx-vlash --variant pi05 --model <checkpoint_dir> \
//!             --image obs.png --prompt "pick up the cube" \
//!             --state 0.1,0.2,0.0,... [--prompt-len 200]
//! ```

use anyhow::{Result, anyhow};
use rlx_runtime::Device;
use rlx_vlash::VlashVariant;
use rlx_vlash::preprocess::rgb8_to_nchw_normalized;
use rlx_vlash::runner::VlashRunner;

struct Args {
    variant: VlashVariant,
    model: Option<String>,
    image: Option<String>,
    prompt: String,
    state: Vec<f32>,
    prompt_len: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        variant: VlashVariant::Pi05,
        model: None,
        image: None,
        prompt: "do the task".to_string(),
        state: Vec::new(),
        prompt_len: 200,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--variant" => {
                a.variant = match it.next().as_deref() {
                    Some("pi0") => VlashVariant::Pi0,
                    Some("pi05") => VlashVariant::Pi05,
                    other => return Err(anyhow!("--variant must be pi0|pi05 (got {other:?})")),
                }
            }
            "--model" => a.model = it.next(),
            "--image" => a.image = it.next(),
            "--prompt" => a.prompt = it.next().unwrap_or_default(),
            "--prompt-len" => a.prompt_len = it.next().and_then(|s| s.parse().ok()).unwrap_or(200),
            "--state" => {
                a.state = it
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            }
            "-h" | "--help" => {
                println!(
                    "rlx-vlash --variant pi0|pi05 --model <dir> --image <path> \\\n         --prompt <text> --state <c,c,...> [--prompt-len N]"
                );
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown flag {other}")),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let Some(model) = args.model.clone() else {
        eprintln!(
            "rlx-vlash: VLASH π₀ / π₀.₅ VLA policy runner.\n\
             Provide --model <checkpoint_dir> (with model.safetensors + tokenizer.json),\n\
             --image <path>, --prompt <text>, and --state <comma-separated floats>.\n\
             Run with -h for usage."
        );
        return Ok(());
    };

    let image_path = args.image.ok_or_else(|| anyhow!("--image is required"))?;
    let img = image::open(&image_path)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let nchw = rgb8_to_nchw_normalized(img.as_raw(), h, w, 224);

    let state = if args.state.is_empty() {
        eprintln!("warning: no --state given; using zeros");
        vec![0.0f32; 14]
    } else {
        args.state
    };

    println!(
        "Loading VLASH {} from {model} (this loads the full backbone; needs ~13GB RAM in f32)…",
        args.variant.as_str()
    );
    let mut runner = VlashRunner::builder(args.variant)
        .device(Device::Cpu)
        .num_images(1)
        .prompt_tokens(args.prompt_len)
        .model_dir(&model)
        .build()?;

    let actions = runner.predict_action_chunk(&[nchw.as_slice()], &state, &args.prompt, None)?;
    let cfg = runner.config();
    let dim = actions.len() / cfg.chunk_size;
    println!(
        "Predicted action chunk: {} steps × {} dims",
        cfg.chunk_size, dim
    );
    for (i, step) in actions.chunks(dim).take(5).enumerate() {
        let preview: Vec<String> = step.iter().take(8).map(|v| format!("{v:+.3}")).collect();
        println!(
            "  step {i:2}: [{}{}]",
            preview.join(", "),
            if dim > 8 { ", …" } else { "" }
        );
    }
    Ok(())
}
