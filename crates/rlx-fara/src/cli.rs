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

//! `rlx-fara` CLI — screenshot + goal → Fara tool call.

use crate::config::FaraSize;
use crate::download::resolve_or_download;
use crate::runner::FaraRunner;
use anyhow::{Context, Result, bail};
use rlx_cli::parse_device;
use rlx_runtime::Device;
use std::path::PathBuf;

const USAGE: &str = "\
rlx-fara — Microsoft Fara1.5 computer-use agent (Qwen3.5 multimodal)

Usage:
  rlx-fara --model-dir DIR --image PATH --goal TEXT [options]
  rlx-fara --size 4b|9b --image PATH --goal TEXT [options]

Options:
  --model-dir DIR     Local HF snapshot (config.json + safetensors)
  --size 4b|9b        Select preset / default cache under .cache/fara/
  --image PATH        Browser screenshot (PNG/JPEG)
  --goal TEXT         User task
  --device NAME       cpu|metal|mlx|cuda|… (default: cpu)
  --max-tokens N      Generation length (default: 512)
  --max-seq N         Decode capacity (default: 2048; 1440x900 shots need ~1.6k)
  --tokenizer PATH    Override tokenizer.json
  --download          Fetch from HuggingFace when model-dir missing
  -h, --help
";

pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(());
    }

    let mut model_dir: Option<PathBuf> = None;
    let mut size = FaraSize::B4;
    let mut image: Option<PathBuf> = None;
    let mut goal: Option<String> = None;
    let mut device = Device::Cpu;
    let mut max_tokens: usize = 512;
    let mut max_seq: usize = 2048;
    let mut tokenizer: Option<PathBuf> = None;
    let mut download = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    it.next().context("--model-dir needs a path")?,
                ));
            }
            "--size" => {
                size = FaraSize::parse(it.next().context("--size needs 4b|9b")?)?;
            }
            "--image" => {
                image = Some(PathBuf::from(it.next().context("--image needs a path")?));
            }
            "--goal" => {
                goal = Some(it.next().context("--goal needs text")?.clone());
            }
            "--device" => {
                device = parse_device(it.next().context("--device")?)?;
            }
            "--max-tokens" => {
                max_tokens = it
                    .next()
                    .context("--max-tokens")?
                    .parse()
                    .context("--max-tokens")?;
            }
            "--max-seq" => {
                max_seq = it.next().context("--max-seq")?.parse().context("--max-seq")?;
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(it.next().context("--tokenizer")?));
            }
            "--download" => download = true,
            other => bail!("unknown arg `{other}` (see --help)"),
        }
    }

    let image = image.ok_or_else(|| anyhow::anyhow!("--image is required"))?;
    let goal = goal.ok_or_else(|| anyhow::anyhow!("--goal is required"))?;

    let dir = match model_dir {
        Some(d) => d,
        None if download => resolve_or_download(size, None)?,
        None => {
            let def = crate::config::default_model_dir(size);
            if crate::config::is_model_dir(&def) {
                def
            } else {
                bail!(
                    "no model dir at {}; pass --model-dir or --download (size={})",
                    def.display(),
                    size.label()
                );
            }
        }
    };

    #[cfg(feature = "qwen35-vlm")]
    let (rgb, w, h) = {
        let path = image
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 image path"))?;
        let (rgb, w, h) = rlx_qwen35::load_rgb_image(path)
            .with_context(|| format!("load image {}", image.display()))?;
        (rgb, w, h)
    };
    #[cfg(not(feature = "qwen35-vlm"))]
    {
        let _ = (image, dir, goal, device, max_tokens, max_seq, tokenizer, size);
        bail!("rlx-fara: rebuild with `--features qwen35-vlm` for --image");
    }
    #[cfg(feature = "qwen35-vlm")]
    {
        eprintln!(
            "[rlx-fara] size={} model_dir={} image={}x{} device={device:?} max_tokens={max_tokens}",
            size.label(),
            dir.display(),
            w,
            h
        );
        let mut runner = FaraRunner::builder()
            .model_dir(&dir)
            .size(size)
            .device(device)
            .max_seq(max_seq)
            .build()?;
        let step = runner.step(
            &goal,
            &rgb,
            w,
            h,
            max_tokens,
            tokenizer.as_deref(),
        )?;
        println!("{}", step.raw_text);
        if !step.tool_calls.is_empty() {
            eprintln!("[rlx-fara] parsed {} tool call(s):", step.tool_calls.len());
            for (i, tc) in step.tool_calls.iter().enumerate() {
                eprintln!(
                    "  [{i}] name={} action={:?} args={}",
                    tc.name,
                    tc.action(),
                    tc.arguments
                );
            }
        }
        Ok(())
    }
}
