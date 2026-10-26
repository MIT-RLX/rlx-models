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

//! `rlx-flux3` command line.

use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

use crate::bbox::{self, Json};
use crate::pipeline::{Flux3Options, Flux3Paths, Flux3Pipeline, Flux3Sample};
use crate::pup::{Upsampler, UpsamplerPaths};

pub const HELP: &str = "\
rlx-flux3 --model-dir DIR --prompt TEXT [--output out.png]

model:    --model-dir DIR (model.safetensors + text_encoder/ + ae.safetensors)
          --device cpu|metal|mlx|cuda|rocm|gpu|vulkan
upsample: --pup DIR|FILE.gguf (flux-3-image-klein-pup: rewrite the prompt into a full caption first)
          --no-pup (send the prompt as typed; the default when --pup is absent)
          --caption-out FILE (write the caption the DiT received)
          --upsample-only (stop after writing the caption)
editing:  --image FILE (repeatable, up to 10 references)
layout:   --boxes FILE (JSON list of objects with id, bbox [y_min, x_min, y_max, x_max] in 0-1, desc;
          edits use from, src_bbox, tgt_bbox); a prompt ending in a 0-1000 box table works too
sampling: --width PX --height PX --steps N --guidance G --alpha A --seed N
memory:   --weight-quant q8_0|q6_k|q5_k|q4_k|none --text-quant (same) --blocks-per-graph N";

pub fn run(args: &[String]) -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut prompt: Option<String> = None;
    let mut out = PathBuf::from("flux3.png");
    let mut pup: Option<PathBuf> = None;
    let mut caption_out: Option<PathBuf> = None;
    let mut upsample_only = false;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut boxes_file: Option<PathBuf> = None;
    let mut opts = Flux3Options::default();
    let mut sample = Flux3Sample::default();
    let num = |s: String, flag: &str| -> Result<usize> {
        s.parse().with_context(|| format!("{flag}: `{s}`"))
    };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        match flag.as_str() {
            "--model-dir" | "--weights" => model_dir = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--prompt" | "-p" => prompt = Some(req(args, &mut i)?),
            "--output" | "-o" => out = req(args, &mut i)?.into(),
            "--pup" => pup = Some(req(args, &mut i)?.into()),
            "--no-pup" => {
                pup = None;
                i += 1;
            }
            "--caption-out" => caption_out = Some(req(args, &mut i)?.into()),
            "--upsample-only" => {
                upsample_only = true;
                i += 1;
            }
            "--image" => images.push(req(args, &mut i)?.into()),
            "--boxes" => boxes_file = Some(req(args, &mut i)?.into()),
            "--width" => sample.width = Some(num(req(args, &mut i)?, &flag)?),
            "--height" => sample.height = Some(num(req(args, &mut i)?, &flag)?),
            "--steps" => sample.steps = Some(num(req(args, &mut i)?, &flag)?),
            "--guidance" => {
                sample.guidance = Some(req(args, &mut i)?.parse().context("--guidance")?)
            }
            "--alpha" => sample.alpha = Some(req(args, &mut i)?.parse().context("--alpha")?),
            "--seed" => sample.seed = req(args, &mut i)?.parse().context("--seed")?,
            "--weight-quant" => {
                opts.weight_quant = rlx_flux1::weights::WeightQuant::parse(&req(args, &mut i)?)?
            }
            "--text-quant" => {
                opts.text_quant = rlx_flux1::weights::WeightQuant::parse(&req(args, &mut i)?)?
            }
            "--blocks-per-graph" => opts.blocks_per_graph = num(req(args, &mut i)?, &flag)?,
            "--help" | "-h" => {
                eprintln!("{HELP}");
                return Ok(());
            }
            other => bail!("unknown flag: {other} (see --help)"),
        }
    }
    let prompt = prompt.ok_or_else(|| anyhow!("--prompt is required"))?;
    opts.device = parse_standard_device("flux3", &device)?;
    let refs: Vec<image::RgbImage> = images
        .iter()
        .map(|p| {
            image::open(p)
                .with_context(|| format!("reading {}", p.display()))
                .map(|i| i.to_rgb8())
        })
        .collect::<Result<_>>()?;

    let boxes: Option<Vec<Json>> = match &boxes_file {
        Some(p) => match Json::parse(
            &std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?,
        )
        .with_context(|| format!("parsing {}", p.display()))?
        {
            Json::List(l) => Some(l),
            _ => bail!("{}: expected a JSON list of boxes", p.display()),
        },
        None => None,
    };
    // An upsampler answer (its whole box table cited as <id>s) goes to the
    // DiT as it is, as in the reference.
    let finished = boxes.is_none() && bbox::is_finished(&prompt);
    let caption = match &pup {
        Some(p) if !finished => {
            let t0 = std::time::Instant::now();
            let mut up =
                Upsampler::open_for(&UpsamplerPaths::discover(p)?, opts.device, !refs.is_empty())?;
            let c = up.upsample_request(&prompt, &refs, boxes.as_deref())?;
            drop(up);
            eprintln!(
                "[rlx-flux3] upsampled in {:.1}s:\n{c}",
                t0.elapsed().as_secs_f32()
            );
            c
        }
        _ => match &boxes {
            Some(b) => format!(
                "{} {}",
                prompt.trim_end(),
                Json::List(bbox::to_canvas(b)?).dumps()
            ),
            None => prompt,
        },
    };
    if let Some(p) = &caption_out {
        std::fs::write(p, format!("{caption}\n"))
            .with_context(|| format!("writing {}", p.display()))?;
    }
    if upsample_only {
        if pup.is_none() {
            bail!("--upsample-only needs --pup");
        }
        println!("{caption}");
        return Ok(());
    }
    let model_dir = model_dir.ok_or_else(|| anyhow!("--model-dir is required"))?;

    let mut pipe = Flux3Pipeline::open(Flux3Paths::discover(&model_dir)?, opts.clone())?;
    let d = pipe.sampler_defaults();
    eprintln!(
        "[rlx-flux3] {} ({} per-mode + {} single blocks), {} steps, guidance {}, alpha {}, device {:?}",
        if pipe.cfg.flow_map {
            "distilled flow map"
        } else {
            "base"
        },
        pipe.cfg.depth,
        pipe.cfg.depth_single,
        sample.steps.unwrap_or(d.steps),
        sample.guidance.unwrap_or(d.guidance),
        sample.alpha.unwrap_or(d.alpha),
        opts.device
    );
    let t0 = std::time::Instant::now();
    let img = pipe.generate(&caption, &refs, &sample)?;
    img.save(&out)?;
    eprintln!(
        "[rlx-flux3] wrote {} ({}x{}) in {:.1}s",
        out.display(),
        img.width,
        img.height,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
