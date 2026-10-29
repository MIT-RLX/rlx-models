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

//! `rlx-indra`: images in, markdown (or text, or JSON lines) out.
//!
//! ```text
//! rlx-indra [--model-dir DIR] [--device cpu|metal|...] [--format md|text|json]
//!           [--out DIR] [--keep-furniture] [--table-heads] [--no-turn] [--timings] IMAGE...
//! ```
//!
//! `--model-dir` defaults to `$INDRA_MODEL_DIR` (a `rlx-indra-convert` output).

use anyhow::{Context, Result, bail};
use rlx_indra::{Indra, Options, load_gray};
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model_dir = std::env::var_os("INDRA_MODEL_DIR").map(PathBuf::from);
    let mut opts = Options::default();
    let (mut format, mut out, mut turn, mut timings) =
        ("md".to_string(), None::<PathBuf>, true, false);
    let mut inputs = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model-dir" => model_dir = Some(args.next().context("--model-dir DIR")?.into()),
            "--device" => {
                opts.device = rlx_runtime::parse_device(&args.next().context("--device NAME")?)?
            }
            "--format" => format = args.next().context("--format md|text|json")?,
            "--out" => out = Some(args.next().context("--out DIR")?.into()),
            "--keep-furniture" => opts.furniture = true,
            "--table-heads" => opts.table_heads = true,
            "--no-turn" => turn = false,
            "--timings" => timings = true,
            "-h" | "--help" => {
                println!(
                    "usage: rlx-indra [--model-dir DIR] [--device NAME] [--format md|text|json] [--out DIR] [--keep-furniture] [--table-heads] [--no-turn] [--timings] IMAGE..."
                );
                return Ok(());
            }
            s if s.starts_with('-') => bail!("unknown flag {s}"),
            s => inputs.push(PathBuf::from(s)),
        }
    }
    if !["md", "text", "json"].contains(&format.as_str()) {
        bail!("--format must be md, text or json");
    }
    if inputs.is_empty() {
        bail!("no input images");
    }
    let model_dir = model_dir.context("pass --model-dir or set INDRA_MODEL_DIR")?;
    let ocr = Indra::load(&model_dir, opts)?;
    if let Some(d) = &out {
        std::fs::create_dir_all(d)?;
    }
    let mut failed = false;
    for path in &inputs {
        let page = match load_gray(path).and_then(|g| ocr.read(&g, turn)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("rlx-indra: {}: {e:#}", path.display());
                failed = true;
                continue;
            }
        };
        let body = match format.as_str() {
            "json" => serde_json::to_string(&serde_json::json!({
                "file": path.display().to_string(),
                "page": 0,
                "rotation": page.rotation,
                "skew": page.skew,
                "vertical": page.vertical,
                "note": page.note,
                "lines": page.lines,
                "markdown": page.markdown,
                "blocks": page.blocks,
                "timings": page.timings,
            }))?,
            "md" => page.markdown.clone().unwrap_or_else(|| page.text()),
            _ => page.text(),
        };
        if timings {
            eprintln!("{}: {:?}", path.display(), page.timings);
        }
        match &out {
            Some(d) => {
                let ext = match format.as_str() {
                    "json" => "jsonl",
                    "md" => "md",
                    _ => "txt",
                };
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
                std::fs::write(d.join(format!("{stem}.{ext}")), format!("{body}\n"))?;
            }
            None => writeln!(std::io::stdout(), "{body}")?,
        }
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}
