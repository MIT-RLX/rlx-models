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

//! Prepare VLASH weights into an RLX-native bundle (`.gguf` or `.rlxp`) with the
//! canonical key names baked in, so the runtime loads them with no remap.
//!
//! ```text
//!   cargo run --release -p rlx-vlash --example prep_weights -- \
//!       --variant pi05 --model <checkpoint_dir> \
//!       --out model.gguf --format gguf --scheme f16
//! ```
//! `--model` is a directory with `model.safetensors` (OpenPI naming) or a single
//! `.safetensors` file. `--scheme` (gguf only): f32 | f16 (default) | q8_0.

use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};

use rlx_vlash::VlashVariant;
use rlx_vlash::prep::{QuantScheme, read_gguf, write_gguf, write_rlxp};
use rlx_vlash::weights::load_remapped;

fn find_safetensors(dir: &Path) -> Result<String> {
    if dir.is_file() {
        return Ok(dir.to_string_lossy().into_owned());
    }
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(single.to_string_lossy().into_owned());
    }
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            return Ok(p.to_string_lossy().into_owned());
        }
    }
    Err(anyhow!("no .safetensors in {}", dir.display()))
}

fn main() -> Result<()> {
    let mut variant = VlashVariant::Pi05;
    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut format = String::from("gguf");
    let mut scheme = QuantScheme::F16;
    let mut all_formats = false;
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--all-formats" => all_formats = true,
            "--variant" => {
                variant = match it.next().as_deref() {
                    Some("pi0") => VlashVariant::Pi0,
                    Some("pi05") => VlashVariant::Pi05,
                    o => return Err(anyhow!("--variant pi0|pi05 (got {o:?})")),
                }
            }
            "--model" => model = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            "--format" => format = it.next().unwrap_or_default(),
            "--scheme" => {
                scheme = it
                    .next()
                    .as_deref()
                    .and_then(QuantScheme::parse)
                    .ok_or_else(|| anyhow!("--scheme f32|f16|q8_0"))?
            }
            "-h" | "--help" => {
                println!(
                    "prep_weights --variant pi0|pi05 --model <dir> --out <path> \\\n             --format gguf|rlxp --scheme f32|f16|q8_0"
                );
                return Ok(());
            }
            o => return Err(anyhow!("unknown flag {o}")),
        }
    }
    let model = model.ok_or_else(|| anyhow!("--model required"))?;
    let out = out.ok_or_else(|| anyhow!("--out required"))?;

    let st = find_safetensors(&model)?;
    println!("Loading + remapping {st} …");
    let wm = load_remapped(&st)?;
    println!("  {} canonical tensors", wm.len());

    // Write every format from a single load (--out is a directory).
    if all_formats {
        std::fs::create_dir_all(&out)?;
        // (filename, is_rlxp, scheme). GGUF and RLX package each in f16 / f32 / q8_0.
        let jobs: [(&str, bool, QuantScheme); 8] = [
            ("model.gguf", false, QuantScheme::F16),
            ("model-q8_0.gguf", false, QuantScheme::Q8_0),
            ("model-q4_k.gguf", false, QuantScheme::Q4K),
            ("model-f32.gguf", false, QuantScheme::F32),
            ("model.rlxp", true, QuantScheme::F16),
            ("model-q8_0.rlxp", true, QuantScheme::Q8_0),
            ("model-q4_k.rlxp", true, QuantScheme::Q4K),
            ("model-f32.rlxp", true, QuantScheme::F32),
        ];
        for (fname, is_rlxp, s) in jobs {
            let p = out.join(fname);
            let kind = if is_rlxp { "RLX package" } else { "GGUF" };
            println!("Writing {fname} ({kind}, {s:?}) …");
            if is_rlxp {
                write_rlxp(&wm, &p, variant, s)?;
            } else {
                write_gguf(&wm, &p, s, variant)?;
            }
            let sz = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            println!(
                "  {} ({:.1} MiB)",
                p.display(),
                sz as f64 / (1024.0 * 1024.0)
            );
        }
        // Sanity: reload the f16 bundles and confirm the canonical key set matches.
        let g = read_gguf(&out.join("model.gguf"))?;
        let r = rlx_vlash::prep::read_rlxp(&out.join("model.rlxp"))?;
        assert_eq!(
            g.len(),
            wm.len(),
            "gguf f16 tensor count mismatch on reload"
        );
        assert_eq!(
            r.len(),
            wm.len(),
            "rlxp f16 tensor count mismatch on reload"
        );
        println!(
            "verified: model.gguf + model.rlxp reload with {} tensors",
            g.len()
        );
        return Ok(());
    }

    match format.as_str() {
        "gguf" => {
            println!("Writing GGUF ({scheme:?}) → {}", out.display());
            write_gguf(&wm, &out, scheme, variant)?;
        }
        "rlxp" | "rlxpack" => {
            println!("Writing RLX package ({scheme:?}) → {}", out.display());
            write_rlxp(&wm, &out, variant, scheme)?;
        }
        other => return Err(anyhow!("--format must be gguf|rlxp (got {other})")),
    }
    let sz = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "done → {} ({:.1} MiB)",
        out.display(),
        sz as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}
