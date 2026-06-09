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

//! Convert ocrs `.rten` checkpoints to `.safetensors` for RLX compile / vmap.

use anyhow::{Result, bail};
use rlx_ocr::weights::{HF_DETECTION_RTEN, HF_RECOGNITION_RTEN, export_rten_to_safetensors};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 {
        let dir = PathBuf::from(&args[1]);
        let det_in = dir.join(HF_DETECTION_RTEN);
        let rec_in = dir.join(HF_RECOGNITION_RTEN);
        export_rten_to_safetensors(&det_in, &dir.join("ocr-detection.safetensors"))?;
        export_rten_to_safetensors(&rec_in, &dir.join("ocr-recognition.safetensors"))?;
        return Ok(());
    }
    if args.len() == 3 {
        export_rten_to_safetensors(
            PathBuf::from(&args[1]).as_path(),
            PathBuf::from(&args[2]).as_path(),
        )?;
        return Ok(());
    }
    bail!(
        "usage: rlx-ocr-convert <model-dir>\n\
         or:   rlx-ocr-convert <input.rten> <output.safetensors>"
    );
}
