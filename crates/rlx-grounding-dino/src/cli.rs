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

//! Command-line interface for `rlx-grounding-dino`.

use crate::GroundingDinoConfig;
use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_sam_device, req};
use std::path::PathBuf;

/// Parsed CLI arguments.
#[derive(Debug, Clone)]
pub struct Args {
    pub weights: Option<PathBuf>,
    pub config: Option<PathBuf>,
    pub device: String,
    pub image: Option<PathBuf>,
    pub text: String,
    pub box_threshold: f32,
    pub text_threshold: f32,
    pub dry: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            weights: None,
            config: None,
            device: "cpu".to_string(),
            image: None,
            text: String::new(),
            box_threshold: 0.3,
            text_threshold: 0.25,
            dry: false,
        }
    }
}

const HELP: &str = "rlx-grounding-dino — open-vocabulary detection (Grounding DINO)
flags:
  --weights PATH       safetensors/gguf checkpoint (required unless --dry)
  --config PATH        config.json (defaults to grounding-dino-base config)
  --device DEV         cpu|metal|mps|mlx|cuda|rocm|hip|gpu|wgpu|vulkan (default cpu)
  --image PATH         input image (jpeg/png)
  --text PROMPT        text prompt, e.g. \"a cat. a remote control.\"
  --box-threshold F    box confidence threshold (default 0.3)
  --text-threshold F   token-grounding threshold (default 0.25)
  --dry                parse/validate config + args only, no inference
  -h, --help           show this help";

pub fn parse(args: &[String]) -> Result<Args> {
    let mut out = Args::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => out.weights = Some(req(args, &mut i)?.into()),
            "--config" => out.config = Some(req(args, &mut i)?.into()),
            "--device" => out.device = req(args, &mut i)?,
            "--image" => out.image = Some(req(args, &mut i)?.into()),
            "--text" => out.text = req(args, &mut i)?,
            "--box-threshold" => out.box_threshold = req(args, &mut i)?.parse()?,
            "--text-threshold" => out.text_threshold = req(args, &mut i)?.parse()?,
            "--dry" => {
                out.dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("{HELP}");
                std::process::exit(0);
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    Ok(out)
}

pub fn run(args: &[String]) -> Result<()> {
    let parsed = parse(args)?;
    let device = parse_sam_device("grounding-dino", &parsed.device)?;

    let cfg = match &parsed.config {
        Some(p) => GroundingDinoConfig::from_file(p)?,
        None => GroundingDinoConfig::base(),
    };

    if parsed.dry {
        eprintln!(
            "[rlx-grounding-dino] dry-run ok: device={device:?} d_model={} enc/dec={}/{} queries={} swin_depths={:?}",
            cfg.d_model,
            cfg.encoder_layers,
            cfg.decoder_layers,
            cfg.num_queries,
            cfg.backbone_config.depths,
        );
        return Ok(());
    }

    // Resolve weights + tokenizer (explicit path, or HF cache when enabled).
    #[cfg(feature = "hf-cache")]
    let (weights_path, tokenizer_path, cfg) = match &parsed.weights {
        Some(p) => (
            p.clone(),
            parsed
                .config
                .as_ref()
                .map(|c| c.with_file_name("tokenizer.json")),
            cfg,
        ),
        None => {
            let r = crate::download::resolve(crate::download::DEFAULT_REPO)?;
            let cfg = GroundingDinoConfig::from_file(&r.config)?;
            (r.weights, Some(r.tokenizer), cfg)
        }
    };
    #[cfg(not(feature = "hf-cache"))]
    let (weights_path, tokenizer_path, cfg): (PathBuf, Option<PathBuf>, GroundingDinoConfig) = (
        parsed
            .weights
            .clone()
            .ok_or_else(|| anyhow!("--weights is required (hf-cache feature disabled)"))?,
        None,
        cfg,
    );

    let image_path = parsed
        .image
        .as_ref()
        .ok_or_else(|| anyhow!("--image is required"))?;
    let (rgb, h, w) = load_rgb(image_path)?;

    // Tokenize the prompt.
    let tokens = build_tokens(&parsed, tokenizer_path.as_deref())?;

    eprintln!(
        "[rlx-grounding-dino] device={device:?} image={}x{} prompt={:?}",
        w, h, parsed.text
    );
    eprintln!(
        "[rlx-grounding-dino] note: heavy compute runs on-device ({device:?}); set RLX_GDINO_PROFILE=1 for a per-stage breakdown."
    );

    let model = crate::GroundingDino::from_checkpoint_on(&weights_path, cfg, device)?;
    let mut dets = model.detect(
        &rgb,
        h,
        w,
        &tokens,
        parsed.box_threshold,
        parsed.text_threshold,
    );

    #[cfg(feature = "tokenizer")]
    if let Some(tp) = &tokenizer_path {
        if tp.exists() {
            let _ = crate::postprocess::label_detections(&mut dets, &tokens.input_ids, tp);
        }
    }

    println!("detections: {}", dets.len());
    for (i, d) in dets.iter().enumerate() {
        println!(
            "  #{i} score={:.3} box=[{:.1},{:.1},{:.1},{:.1}] label={:?} tokens={:?}",
            d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.label, d.token_indices
        );
    }
    Ok(())
}

/// Load an image as interleaved RGB `u8`.
fn load_rgb(path: &std::path::Path) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::open(path)
        .map_err(|e| anyhow!("open image {}: {e}", path.display()))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok((img.into_raw(), h, w))
}

/// Build text tokens from the prompt (tokenizer feature) or fail with guidance.
fn build_tokens(
    parsed: &Args,
    tokenizer_path: Option<&std::path::Path>,
) -> Result<crate::TextTokens> {
    if parsed.text.trim().is_empty() {
        bail!("--text prompt is required, e.g. --text \"a cat. a remote control.\"");
    }
    #[cfg(feature = "tokenizer")]
    {
        let tp = tokenizer_path.ok_or_else(|| {
            anyhow!(
                "tokenizer.json not found; pass --config beside a tokenizer.json or enable hf-cache"
            )
        })?;
        crate::tokenizer::tokenize_prompt(tp, &parsed.text)
    }
    #[cfg(not(feature = "tokenizer"))]
    {
        let _ = tokenizer_path;
        bail!("the `tokenizer` feature is disabled; cannot tokenize the prompt");
    }
}
