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
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

use crate::OcrRunner;

pub fn run(args: &[String]) -> Result<()> {
    let mut detection: Option<PathBuf> = None;
    let mut recognition: Option<PathBuf> = None;
    let mut model_dir: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--detection-model" => detection = Some(req(args, &mut i)?.into()),
            "--recognition-model" => recognition = Some(req(args, &mut i)?.into()),
            "--model-dir" => model_dir = Some(req(args, &mut i)?.into()),
            "--image" => image = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-ocr — ocrs U-Net + CRNN/GRU (native RLX, safetensors weights)\n\
                     Flags:\n\
                       --model-dir DIR          directory with ocr-*-full.safetensors\n\
                       --detection-model PATH   detection .safetensors\n\
                       --recognition-model PATH recognition .safetensors\n\
                       --image PATH             input JPEG/PNG\n\
                       [--device cpu|metal|cuda|…] [--dry]"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let device = parse_standard_device("ocr", &device)?;

    let mut builder = OcrRunner::builder().device(device);
    builder = match (detection, recognition, model_dir) {
        (Some(d), Some(r), _) => builder.detection_model(d).recognition_model(r),
        (_, _, Some(dir)) => builder.model_dir(dir),
        _ => {
            return Err(anyhow!(
                "provide --model-dir DIR or both --detection-model and --recognition-model"
            ));
        }
    };

    eprintln!("[rlx-ocr] device={device:?}");

    let runner = builder.build()?;
    eprintln!(
        "[rlx-ocr] engine ready — det_threshold={}",
        runner.engine().detection_threshold()
    );

    if dry {
        eprintln!("[rlx-ocr] --dry set; skipping inference");
        return Ok(());
    }

    let image = image.ok_or_else(|| anyhow!("--image is required unless --dry"))?;
    let t0 = std::time::Instant::now();
    let out = runner.predict_path(&image).context("OCR inference")?;
    eprintln!(
        "[rlx-ocr] extracted {} lines, {} words in {:?}",
        out.lines.len(),
        out.words.len(),
        t0.elapsed()
    );
    if !out.text.is_empty() {
        println!("{}", out.text);
    }
    Ok(())
}
