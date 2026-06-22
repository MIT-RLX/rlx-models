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

//! `rlx-florence2` command-line interface.

use crate::session::Florence2Session;
use anyhow::{Result, anyhow, bail};
use rlx_runtime::Device;
use std::path::PathBuf;

struct Args {
    weights: PathBuf,
    image: PathBuf,
    task: String,
    device: Device,
    max_new_tokens: usize,
}

fn parse_args() -> Result<Args> {
    let mut weights = None;
    let mut image = None;
    let mut task = "<CAPTION>".to_string();
    let mut device = Device::Cpu;
    let mut max_new_tokens = 64usize;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" | "--model-dir" => weights = it.next().map(PathBuf::from),
            "--image" => image = it.next().map(PathBuf::from),
            "--task" => task = it.next().ok_or_else(|| anyhow!("--task needs a value"))?,
            "--device" => {
                let d = it.next().ok_or_else(|| anyhow!("--device needs a value"))?;
                device = rlx_cli::parse_device(&d)?;
            }
            "--max-new-tokens" => {
                max_new_tokens = it
                    .next()
                    .ok_or_else(|| anyhow!("--max-new-tokens needs a value"))?
                    .parse()?;
            }
            "-h" | "--help" => {
                println!(
                    "rlx-florence2 --weights DIR --image IMG [--task <CAPTION>] [--device cpu|metal|mlx|cuda] [--max-new-tokens N]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
    }
    Ok(Args {
        weights: weights.ok_or_else(|| anyhow!("--weights DIR required"))?,
        image: image.ok_or_else(|| anyhow!("--image IMG required"))?,
        task,
        device,
        max_new_tokens,
    })
}

pub fn run() -> Result<()> {
    let args = parse_args()?;

    #[cfg(feature = "image-io")]
    let (rgb, h, w) = {
        let img = image::open(&args.image)
            .map_err(|e| anyhow!("open {}: {e}", args.image.display()))?
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        (img.into_raw(), h, w)
    };
    #[cfg(not(feature = "image-io"))]
    let (rgb, h, w): (Vec<u8>, usize, usize) =
        bail!("rebuild with --features image-io to load images");

    let mut session = Florence2Session::from_dir(&args.weights, args.device)?;
    let result = session.run(&rgb, h, w, &args.task, args.max_new_tokens)?;
    println!("{}: {:#?}", args.task, result);
    Ok(())
}
