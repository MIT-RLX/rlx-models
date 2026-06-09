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

// RLX CLI
use crate::Vjepa2Runner;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;
pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut batch = 1usize;
    let mut dry = false;
    let mut predict = false;
    let mut pool = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch: usize")?,
            "--predict" => {
                predict = true;
                i += 1;
            }
            "--pool" => {
                pool = true;
                i += 1;
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("rlx-vjepa2 — see README for flags");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;

    eprintln!(
        "[rlx-vjepa2] vjepa2: weights={weights:?} config={config:?} device={device} batch={batch}"
    );
    let mut builder = Vjepa2Runner::builder().weights(&weights).batch(batch);
    if device != "cpu" {
        builder = builder.device(parse_standard_device("vjepa2", &device)?);
    }
    if let Some(cfg) = config {
        builder = builder.config_path(cfg);
    }
    let mut runner = builder.build()?;
    let cfg = runner.config();
    eprintln!(
        "[rlx-vjepa2] loaded — hidden={} layers={} patches={} predictor={} pooler={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_patches(),
        runner.has_predictor(),
        runner.has_pooler()
    );

    if dry {
        eprintln!("[rlx-vjepa2] --dry set; skipping forward pass");
        return Ok(());
    }

    let crop = cfg.crop_size;
    let frames = cfg.frames_per_clip;
    let rgb = vec![128u8; frames * crop * crop * 3];

    let t0 = std::time::Instant::now();
    let enc = runner.encode_video_hwc(&rgb)?;
    let dt = t0.elapsed();
    eprintln!(
        "[rlx-vjepa2] vjepa2 encode in {dt:?} — batch={} seq={} hidden={}",
        enc.per_batch.len(),
        enc.seq,
        enc.hidden
    );

    if predict {
        if !runner.has_predictor() {
            bail!("--predict requested but checkpoint has no predictor weights");
        }
        let seq = enc.seq;
        let masks = crate::Vjepa2Masks {
            context: (0..seq / 2).collect(),
            target: (seq / 2..seq).collect(),
            mask_index: 0,
        };
        let pred = runner.predict(&enc, &masks)?;
        eprintln!(
            "[rlx-vjepa2] predictor — target_tokens={} hidden={}",
            pred.num_target, pred.hidden
        );
    }

    if pool {
        if !runner.has_pooler() {
            bail!("--pool requested but checkpoint has no pooler weights");
        }
        let pooled = runner.pool(&enc)?;
        eprintln!(
            "[rlx-vjepa2] pooler — embedding_dim={}",
            pooled.embedding.len() / batch.max(1)
        );
        if let Some(logits) = pooled.logits {
            eprintln!("[rlx-vjepa2] classifier logits len={}", logits.len());
        }
    }

    Ok(())
}
