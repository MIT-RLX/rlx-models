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

//! Accuracy of each `--number-format` against the f32 run, per component.
//!
//! ```text
//! cargo run -p rlx-flux1 --release --features metal --example format_report -- \
//!   --model-dir weights/image/flux1-dev --device metal --formats f32,bf16,f16,f8,f6 \
//!   [--doubles 4 --singles 8] [--components clip,t5,dit,vae] [--lora FILE] [--keep a,b]
//! ```
//!
//! The f32 run is the reference (it matches diffusers to ~1e-6; see
//! `tests/reference_parity.rs`). The denoiser is measured on a slice of the first
//! `--doubles` / `--singles` blocks plus the final layer, on the f32 text
//! conditioning, so its number is the denoiser's own error.

use anyhow::{Result, bail};
use rlx_flux1::config::{ClipTextConfig, T5Config};
use rlx_flux1::precision::{Component, with_component};
use rlx_flux1::{DitGeometry, DitSource, Flux1Config, Flux1Dit, Flux1Paths, LoraSet, TensorFile};
use rlx_runtime::{Device, NumberFormat};
use std::path::PathBuf;
use std::str::FromStr;

fn rel(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut d2, mut b2, mut dm, mut bm) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = if x.is_finite() {
            (x - y) as f64
        } else {
            f64::INFINITY
        };
        d2 += d * d;
        b2 += (y as f64).powi(2);
        dm = dm.max(d.abs());
        bm = bm.max((y as f64).abs());
    }
    ((d2 / b2).sqrt(), dm / bm)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model_dir =
        PathBuf::from(get("--model-dir").unwrap_or_else(|| "weights/image/flux1-dev".into()));
    let device = Device::from_str(&get("--device").unwrap_or_else(|| "cpu".into()))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let formats: Vec<String> = get("--formats")
        .unwrap_or_else(|| "f32,bf16,f16,f8".into())
        .split(',')
        .map(str::to_string)
        .collect();
    if formats.first().map(String::as_str) != Some("f32") {
        bail!("--formats must start with f32 (the reference)");
    }
    let comps = get("--components").unwrap_or_else(|| "clip,t5,dit,vae".into());
    let want = |c: &str| comps.split(',').any(|x| x == c);
    let doubles: usize = get("--doubles").map_or(Ok(4), |v| v.parse())?;
    let singles: usize = get("--singles").map_or(Ok(8), |v| v.parse())?;
    let prompt = get("--prompt").unwrap_or_else(|| {
        "emoji of flying pink pig. enlarged head in cartoon style. cute. white wings. head is turned towards viewer. 3D lighting. no cast shadows.".into()
    });
    if let Some(k) = get("--keep") {
        rlx_ir::env::set(rlx_flux1::precision::KEEP_PARAMS_ENV, k);
    }
    let paths = Flux1Paths::discover(&model_dir)?;

    let t5_tok = tokenizers::Tokenizer::from_file(paths.t5.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let clip_tok = tokenizers::Tokenizer::from_file(paths.clip.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let t5_ids = rlx_flux1::t5::t5_ids(&t5_tok, &prompt, 512)?;
    let clip_cfg =
        ClipTextConfig::from_json(&std::fs::read_to_string(paths.clip.join("config.json"))?)?;
    let clip_ids = rlx_flux1::clip::clip_ids(&clip_tok, &prompt, 77, clip_cfg.eos_token_id)?;
    let t5_cfg: T5Config =
        serde_json::from_str(&std::fs::read_to_string(paths.t5.join("config.json"))?)?;

    let file = TensorFile::open(&paths.transformer)?;
    let mut cfg = Flux1Config::from_shapes(|k| file.shape(k))?;
    cfg.depth = doubles.min(cfg.depth);
    cfg.depth_single = singles.min(cfg.depth_single);
    let mut lora = LoraSet::default();
    if let Some(l) = get("--lora") {
        let modules: Vec<String> = file
            .keys()
            .into_iter()
            .filter_map(|k| k.strip_suffix(".weight").map(str::to_string))
            .collect();
        lora.add_file(l.as_ref(), 1.0, &cfg, &modules)?;
    }
    let geo = DitGeometry {
        txt_len: 512,
        grid_h: 10,
        grid_w: 10,
    };
    let img = rlx_flux1::sampler::gaussian_noise(geo.img_len() * cfg.in_channels, 1);
    let latents = rlx_flux1::sampler::gaussian_noise(16 * 20 * 20, 2);
    let (vae_cfg, vae_w) = rlx_flux1::vae::load_flux1_vae(&paths.vae)?;

    let mut refs: Option<[Vec<f32>; 4]> = None;
    println!(
        "format     clip pooled         t5 hidden           denoiser ({doubles}+{singles})       vae rgb"
    );
    for f in &formats {
        let fmt: NumberFormat = f.parse().map_err(|e| anyhow::anyhow!("{f}: {e}"))?;
        rlx_flux1::precision::apply(Some(fmt))?;
        if args.iter().any(|a| a == "--no-policy") {
            // Compare against the generic rules alone.
            rlx_ir::env::set("RLX_FLUX1_NO_POLICY", "1");
        }
        // The f32 run always produces the text conditioning the denoiser needs.
        let need_text = refs.is_none() && want("dit");
        let pooled = if want("clip") || need_text {
            let mut c = with_component(Component::Clip, || {
                rlx_flux1::clip::ClipTextEncoder::compile(&paths.clip, clip_cfg.clone(), device)
            })?;
            c.pooled(&clip_ids)?
        } else {
            Vec::new()
        };
        let t5 = if want("t5") || need_text {
            let mut t = with_component(Component::T5, || {
                rlx_flux1::t5::T5Encoder::compile(&paths.t5, t5_cfg.clone(), 512, device, 6)
            })?;
            t.encode(&t5_ids)?
        } else {
            Vec::new()
        };
        // The denoiser always sees the f32 conditioning.
        let (txt, pool) = match &refs {
            Some(r) => (r[1].clone(), r[0].clone()),
            None => (t5.clone(), pooled.clone()),
        };
        let dit = if want("dit") && !txt.is_empty() && !pool.is_empty() {
            let mut src = DitSource {
                file: &file,
                lora: &lora,
                cfg: &cfg,
                quant: None,
            };
            let mut d = with_component(Component::Denoiser, || {
                Flux1Dit::compile(&cfg, geo, &mut src, device, 4, 8)
            })?;
            d.forward(&img, &txt, &pool, 0.7, 5.0)?
        } else {
            Vec::new()
        };
        let vae = if want("vae") {
            let mut v = with_component(Component::Vae, || {
                rlx_flux1::vae::Flux1VaeDecoder::compile(&vae_cfg, &vae_w, 20, 20, device)
            })?;
            v.decode(&latents)?
        } else {
            Vec::new()
        };
        let cur = [pooled, t5, dit, vae];
        match &refs {
            None => {
                println!("{f:<8}   (reference)");
                refs = Some(cur);
            }
            Some(r) => {
                let cell = |a: &[f32], b: &[f32]| -> String {
                    if a.is_empty() {
                        return "-".into();
                    }
                    let (l2, mx) = rel(a, b);
                    format!("{l2:.2e} (max {mx:.1e})")
                };
                println!(
                    "{f:<8}   {:<19} {:<19} {:<24} {}",
                    cell(&cur[0], &r[0]),
                    cell(&cur[1], &r[1]),
                    cell(&cur[2], &r[2]),
                    cell(&cur[3], &r[3])
                );
            }
        }
    }
    rlx_flux1::precision::apply(None)?;
    Ok(())
}
