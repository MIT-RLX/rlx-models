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
use crate::Wav2Vec2BertRunner;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut batch = 1usize;
    let mut seq = 128usize;
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--wav" => wav = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch: usize")?,
            "--seq" => seq = req(args, &mut i)?.parse().context("--seq: usize")?,
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("rlx-wav2vec2-bert — see README for flags");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_standard_device("wav2vec2-bert", &device)?;

    eprintln!(
        "[rlx-wav2vec2-bert] wav2vec2-bert: weights={weights:?} device={device:?} batch={batch} seq={seq} wav={wav:?}"
    );
    let mut builder = Wav2Vec2BertRunner::builder()
        .weights(&weights)
        .device(device)
        .batch(batch)
        .seq(seq);
    if let Some(cfg) = config {
        builder = builder.config_path(cfg);
    }
    let mut runner = builder.build()?;
    let cfg = runner.config().clone();
    eprintln!(
        "[rlx-wav2vec2-bert] compiled — hidden={} layers={} feat_dim={} sample_rate={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.feature_projection_input_dim,
        runner.preprocess_config().sampling_rate,
    );

    if dry {
        eprintln!("[rlx-wav2vec2-bert] --dry set; skipping forward pass");
        return Ok(());
    }

    let t0 = std::time::Instant::now();
    let hidden = if let Some(wav_path) = wav {
        runner.encode_wav(&wav_path)?
    } else {
        // ~1 s synthetic tone @ 16 kHz when no --wav is supplied.
        let sr = runner.preprocess_config().sampling_rate as usize;
        let waveform: Vec<f32> = (0..sr)
            .map(|i| (440.0 * 2.0 * std::f32::consts::PI * i as f32 / sr as f32).sin() * 0.2)
            .collect();
        runner.encode_waveform(&waveform)?
    };
    let dt = t0.elapsed();
    let h = cfg.hidden_size;
    eprintln!("[rlx-wav2vec2-bert] encoder out in {dt:?} — shape=[{batch}, {seq}, {h}]");
    let norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mean = hidden.iter().sum::<f32>() / hidden.len() as f32;
    eprintln!("  ||hidden||₂ = {norm:.3} mean = {mean:.6}");
    Ok(())
}
