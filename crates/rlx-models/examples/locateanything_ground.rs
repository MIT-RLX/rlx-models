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

//! Minimal grounding — HF Hub cache + bundled sample (no paths required).
//!
//! ```bash
//! huggingface-cli download nvidia/LocateAnything-3B   # once
//! # or: just fetch-locateanything
//!
//! cargo run -p rlx-models --example locateanything_ground --release -- --phrase person
//! ```

use anyhow::Result;
use rlx_locateanything::{
    InferenceOptions, LocateAnythingSession, PromptStyle, fixtures, resolve_device,
};

fn main() -> Result<()> {
    let mut image = fixtures::sample_image_path();
    let mut phrase = "person".to_string();
    let mut device = "auto".to_string();
    let mut max_side: Option<u32> = Some(640);
    let mut model_dir: Option<std::path::PathBuf> = None;

    let mut args = std::env::args().skip(1).peekable();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model-dir" | "--weights" => {
                let raw = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))?;
                model_dir = Some(raw.into());
            }
            "--image" => {
                image = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))?
                    .into();
            }
            "--phrase" => {
                phrase = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))?;
            }
            "--device" => {
                device = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))?;
            }
            "--max-image-side" => {
                let s = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))?;
                max_side = Some(
                    s.parse()
                        .map_err(|e| anyhow::anyhow!("invalid --max-image-side: {e}"))?,
                );
            }
            "--help" | "-h" => {
                eprintln!(
                    "locateanything_ground — [--model-dir hf|PATH] [--image PATH] [--phrase TEXT]\n\
                     Default weights: Hugging Face cache ({})\n\
                     Default image: {}",
                    rlx_locateanything::default_hf_cache_dir().display(),
                    fixtures::sample_image_path().display()
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }

    let dev = resolve_device(Some(&device))?;
    let mut options = InferenceOptions::for_grounding()
        .device(dev)
        .prompt_style(PromptStyle::Processor);
    if let Some(side) = max_side {
        options = options.max_image_side(side);
    }

    let mut session = match model_dir {
        Some(dir) => LocateAnythingSession::open_with_options(dir, options)?,
        None => LocateAnythingSession::open_default()?,
    };

    eprintln!(
        "device={:?} model={} image={}",
        session.device(),
        session.model_dir().display(),
        image.display()
    );

    let prep = session.preprocess_file(&image)?;
    session.warmup(&prep, &phrase)?;
    let out = session.ground(&prep, &phrase)?;

    println!("{}", out.text);
    for b in &out.boxes {
        println!("box: ({:.0},{:.0})-({:.0},{:.0})", b.x1, b.y1, b.x2, b.y2);
    }
    Ok(())
}
