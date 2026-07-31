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

//! Dump rlx-vlash intermediates on the SAME fixed inputs the Python reference
//! used, so `scripts/run_parity.py` can compare the two implementations.
//!
//! Reads a fixture directory (written by `scripts/vlash_ref_dump.py`) for the
//! shared inputs — `pixel_values.bin`, `token_ids.bin`, `token_mask.bin`,
//! `state_padded.bin`, `noise.bin` — plus a checkpoint dir, then writes the rlx
//! outputs as `rlx_<stage>.bin` (`image_features_raw`, `prefix_embeds`,
//! `velocity_step0`, `actions_padded`).
//!
//! ```text
//!   cargo run --release -p rlx-vlash --example dump_intermediates -- \
//!       --variant pi05 --model <ckpt_dir> --fixture <dump_dir>
//! ```

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

use rlx_core::flow_util::compile_built;
use rlx_runtime::Device;
use rlx_vlash::config::VlashVariant;
use rlx_vlash::prefix::{assemble_prefix, build_attn_inputs};
use rlx_vlash::sample::{sample_actions, time_input};
use rlx_vlash::vision::{build_vision_flow, extract_vision_embed};
use rlx_vlash::{VlashConfig, build_denoise_flow, weights};

fn read_bin(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let p = dir.join(format!("{name}.bin"));
    let bytes = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn write_bin(dir: &Path, name: &str, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join(format!("rlx_{name}.bin")), bytes)?;
    println!("  wrote rlx_{name}.bin ({} f32)", data.len());
    Ok(())
}

fn main() -> Result<()> {
    let mut variant = VlashVariant::Pi05;
    let mut model: Option<PathBuf> = None;
    let mut fixture: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--variant" => {
                variant = match it.next().as_deref() {
                    Some("pi0") => VlashVariant::Pi0,
                    Some("pi05") => VlashVariant::Pi05,
                    o => return Err(anyhow!("--variant pi0|pi05 (got {o:?})")),
                }
            }
            "--model" => model = it.next().map(PathBuf::from),
            "--fixture" => fixture = it.next().map(PathBuf::from),
            o => return Err(anyhow!("unknown flag {o}")),
        }
    }
    let model = model.ok_or_else(|| anyhow!("--model required"))?;
    let fixture = fixture.ok_or_else(|| anyhow!("--fixture required"))?;

    let cfg = VlashConfig::for_variant(variant);
    let st = {
        let single = model.join("model.safetensors");
        if single.is_file() {
            single.to_string_lossy().into_owned()
        } else {
            model.to_string_lossy().into_owned()
        }
    };

    // Shared inputs from the Python dump.
    let pixel_values = read_bin(&fixture, "pixel_values")?;
    let token_ids: Vec<i64> = read_bin(&fixture, "token_ids")?.iter().map(|&x| x as i64).collect();
    let token_mask = read_bin(&fixture, "token_mask").unwrap_or_else(|_| vec![1.0; token_ids.len()]);
    let state = read_bin(&fixture, "state_padded")?;
    let noise = read_bin(&fixture, "noise")?;

    println!("Loading rlx-vlash {} from {st}…", variant.as_str());
    let mut wm = weights::load_remapped(&st)?;
    let vision_embed = extract_vision_embed(&mut wm, &cfg.vision)?;
    let (embed_tokens, embed_shape) = wm.take("vlm.embed_tokens.weight").context("embed_tokens")?;
    let vocab = embed_shape[0];

    // Vision.
    let vision_built = build_vision_flow(&cfg.vision, &mut wm, 1)?;
    let mut vision = compile_built(vision_built, Device::Cpu)?;
    let hidden = rlx_siglip2::assemble_vision_hidden(
        &vision_embed,
        &pixel_values,
        1,
        cfg.vision.patch_size,
        cfg.image_size,
    )?;
    let image_features = vision.run(&[("hidden", hidden.as_slice())]).remove(0);
    write_bin(&fixture, "image_features_raw", &image_features)?;

    // Prefix.
    let prefix = assemble_prefix(
        &image_features,
        1,
        cfg.vision.num_patches(),
        cfg.vlm.hidden,
        &embed_tokens,
        vocab,
        &token_ids,
        &token_mask,
    );
    write_bin(&fixture, "prefix_embeds", &prefix.emb)?;

    // Denoise.
    let attn = build_attn_inputs(&cfg, &prefix.pad);
    let denoise_built = build_denoise_flow(&cfg, &mut wm, prefix.len)?;
    let mut denoise = compile_built(denoise_built, Device::Cpu)?;

    let time_emb = time_input(&cfg, 1.0);
    let v0 = denoise
        .run(&[
            ("prefix_emb", prefix.emb.as_slice()),
            ("state", state.as_slice()),
            ("actions", noise.as_slice()),
            ("time_emb", time_emb.as_slice()),
            ("cos", attn.cos.as_slice()),
            ("sin", attn.sin.as_slice()),
            ("attn_bias", attn.bias.as_slice()),
        ])
        .remove(0);
    write_bin(&fixture, "velocity_step0", &v0)?;

    let actions = sample_actions(&mut denoise, &cfg, &prefix.emb, &state, &attn, &noise);
    write_bin(&fixture, "actions_padded", &actions)?;

    println!("done → rlx_*.bin in {}", fixture.display());
    Ok(())
}
