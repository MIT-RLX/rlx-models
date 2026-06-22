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

use crate::audio::write_wav_mono_16;
use crate::session::{AecConfig, AecSession};
use anyhow::{Result, bail};
use rlx_cli::req;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut mic_wav: Option<PathBuf> = None;
    let mut ref_wav: Option<PathBuf> = None;
    let mut out_wav: Option<PathBuf> = None;
    let mut step_size: Option<f32> = None;
    let mut no_residual = false;
    let mut max_delay_ms = 300u32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mic-wav" => mic_wav = Some(req(args, &mut i)?.into()),
            "--ref-wav" => ref_wav = Some(req(args, &mut i)?.into()),
            "--out-wav" | "-o" => out_wav = Some(req(args, &mut i)?.into()),
            "--step-size" => {
                step_size = Some(
                    req(args, &mut i)?
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--step-size: f32"))?,
                );
            }
            "--max-delay-ms" => {
                max_delay_ms = req(args, &mut i)?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--max-delay-ms: u32"))?;
            }
            "--no-residual" => {
                no_residual = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-aec — acoustic echo cancellation (16 kHz mono)\n\
                     Flags: --mic-wav PATH --ref-wav PATH --out-wav PATH\n\
                       [--step-size F] [--max-delay-ms N] [--no-residual]"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let mic_path = mic_wav.ok_or_else(|| anyhow::anyhow!("--mic-wav required"))?;
    let ref_path = ref_wav.ok_or_else(|| anyhow::anyhow!("--ref-wav required"))?;
    let out_path = out_wav.ok_or_else(|| anyhow::anyhow!("--out-wav required"))?;

    let cfg = AecConfig {
        step_size: step_size.unwrap_or(0.02),
        max_delay_ms,
        residual: !no_residual,
        ..AecConfig::default()
    };

    let (out, delay) = AecSession::process_wav_files(&mic_path, &ref_path, &cfg)?;
    write_wav_mono_16(&out_path, &out)?;
    println!(
        "aec: {} samples, delay={} samples ({:.1} ms) → {}",
        out.len(),
        delay,
        delay as f32 * 1000.0 / crate::audio::SAMPLE_RATE as f32,
        out_path.display()
    );
    Ok(())
}
