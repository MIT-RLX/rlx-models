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

//! `rlx-flux1` command line. [`FluxArgs`] is public so front-ends built on
//! the pipeline (e.g. `rlx-genmoji`) accept the same model flags.

use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

use crate::pipeline::{Flux1Options, Flux1Paths, Flux1Pipeline, Flux1Sample};

/// Model + sampling flags shared by every FLUX.1 front-end.
#[derive(Debug, Clone, Default)]
pub struct FluxArgs {
    pub model_dir: Option<PathBuf>,
    pub transformer: Option<PathBuf>,
    pub t5: Option<PathBuf>,
    pub clip: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    pub device: Option<String>,
    pub opts: Flux1Options,
    pub sample: Flux1Sample,
    pub noise: Option<PathBuf>,
    /// `--number-format` (f32 when absent).
    pub number_format: Option<rlx_runtime::NumberFormat>,
}

pub const FLUX_FLAGS_HELP: &str = "\
model:    --model-dir DIR (flux1-dev.safetensors + t5/ + clip/ + vae/) | --transformer F --t5 D --clip D --vae F
          --lora FILE[:SCALE] (repeatable)  --device cpu|metal|mlx|cuda|rocm|gpu|vulkan
sampling: --width PX --height PX --steps N --guidance G --seed N --noise FILE (raw f32 [16,h/8,w/8])
format:   --number-format f32|f16|bf16|f8|f6|f4|fNeXmY|… (weights + compute; see crate README)
memory:   --weight-quant q8_0|q6_k|q5_k|q4_k|none --t5-quant (same) --t5-len N --doubles-per-graph N --singles-per-graph N --t5-layers-per-graph N --keep-text-encoders";

impl FluxArgs {
    /// Consume `args[*i]` (and its value) if it is a FLUX flag.
    pub fn parse_flag(&mut self, args: &[String], i: &mut usize) -> Result<bool> {
        let num = |s: String, flag: &str| -> Result<usize> {
            s.parse().with_context(|| format!("{flag}: `{s}`"))
        };
        let flag = args[*i].clone();
        match flag.as_str() {
            "--model-dir" | "--weights" => self.model_dir = Some(req(args, i)?.into()),
            "--transformer" => self.transformer = Some(req(args, i)?.into()),
            "--t5" => self.t5 = Some(req(args, i)?.into()),
            "--clip" => self.clip = Some(req(args, i)?.into()),
            "--vae" => self.vae = Some(req(args, i)?.into()),
            "--device" => self.device = Some(req(args, i)?),
            "--lora" => {
                let v = req(args, i)?;
                let (path, scale) = match v.rsplit_once(':') {
                    Some((p, s)) if s.parse::<f32>().is_ok() => (p.to_string(), s.parse()?),
                    _ => (v, 1.0),
                };
                self.opts.loras.push((path.into(), scale));
            }
            "--width" => self.sample.width = num(req(args, i)?, &flag)?,
            "--height" => self.sample.height = num(req(args, i)?, &flag)?,
            "--steps" => self.sample.steps = num(req(args, i)?, &flag)?,
            "--guidance" => self.sample.guidance = req(args, i)?.parse().context("--guidance")?,
            "--seed" => self.sample.seed = req(args, i)?.parse().context("--seed")?,
            "--noise" => self.noise = Some(req(args, i)?.into()),
            "--t5-len" => self.opts.t5_len = Some(num(req(args, i)?, &flag)?),
            "--weight-quant" => {
                self.opts.weight_quant = crate::weights::WeightQuant::parse(&req(args, i)?)?
            }
            "--t5-quant" => {
                self.opts.t5_quant = crate::weights::WeightQuant::parse(&req(args, i)?)?
            }
            "--doubles-per-graph" => self.opts.doubles_per_graph = num(req(args, i)?, &flag)?,
            "--singles-per-graph" => self.opts.singles_per_graph = num(req(args, i)?, &flag)?,
            "--t5-layers-per-graph" => self.opts.t5_layers_per_graph = num(req(args, i)?, &flag)?,
            "--number-format" => {
                let v = req(args, i)?;
                self.number_format =
                    Some(v.parse().map_err(|e| anyhow!("--number-format {v}: {e}"))?);
            }
            "--keep-text-encoders" => {
                self.opts.keep_text_encoders = true;
                *i += 1;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Resolve component paths: explicit flags override `--model-dir` discovery.
    pub fn paths(&self) -> Result<Flux1Paths> {
        let discovered = self
            .model_dir
            .as_deref()
            .map(Flux1Paths::discover)
            .transpose()?;
        let pick =
            |explicit: &Option<PathBuf>, d: Option<&PathBuf>, what: &str| -> Result<PathBuf> {
                explicit
                    .clone()
                    .or_else(|| d.cloned())
                    .ok_or_else(|| anyhow!("{what}: pass --model-dir or --{what}"))
            };
        let d = discovered.as_ref();
        Ok(Flux1Paths {
            transformer: pick(&self.transformer, d.map(|d| &d.transformer), "transformer")?,
            t5: pick(&self.t5, d.map(|d| &d.t5), "t5")?,
            clip: pick(&self.clip, d.map(|d| &d.clip), "clip")?,
            vae: pick(&self.vae, d.map(|d| &d.vae), "vae")?,
        })
    }

    /// Open the pipeline and finalize the sample settings (`--noise`).
    pub fn open(&mut self, family: &str) -> Result<Flux1Pipeline> {
        self.opts.device = parse_standard_device(family, self.device.as_deref().unwrap_or("cpu"))?;
        crate::precision::apply(self.number_format)?;
        if let Some(p) = &self.noise {
            let raw = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
            self.sample.noise = Some(
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
            );
        }
        Flux1Pipeline::open(self.paths()?, self.opts.clone())
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let mut fa = FluxArgs::default();
    let mut prompt: Option<String> = None;
    let mut out = PathBuf::from("flux1.png");
    let mut i = 0;
    while i < args.len() {
        if fa.parse_flag(args, &mut i)? {
            continue;
        }
        match args[i].as_str() {
            "--prompt" | "-p" => prompt = Some(req(args, &mut i)?),
            "--output" | "-o" => out = req(args, &mut i)?.into(),
            "--help" | "-h" => {
                eprintln!("rlx-flux1 --prompt TEXT [--output out.png]\n{FLUX_FLAGS_HELP}");
                return Ok(());
            }
            other => bail!("unknown flag: {other} (see --help)"),
        }
    }
    let prompt = prompt.ok_or_else(|| anyhow!("--prompt is required"))?;
    let mut pipe = fa.open("flux1")?;
    eprintln!(
        "[rlx-flux1] {} double + {} single blocks, guidance-embed {}, {} LoRA file(s), device {:?}",
        pipe.cfg.depth,
        pipe.cfg.depth_single,
        pipe.cfg.guidance_embed,
        fa.opts.loras.len(),
        fa.opts.device
    );
    let t0 = std::time::Instant::now();
    let img = pipe.generate(&prompt, &fa.sample)?;
    img.save(&out)?;
    eprintln!(
        "[rlx-flux1] wrote {} ({}x{}) in {:.1}s",
        out.display(),
        img.width,
        img.height,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
