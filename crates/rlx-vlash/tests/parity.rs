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

//! Staged CPU parity vs the Python VLASH reference (cosine > 0.999).
//!
//! Fixtures are produced by `scripts/vlash_ref_dump.py` (raw f32 `.bin` +
//! `manifest.json`). Point the tests at them with:
//! ```text
//!   RLX_VLASH_PI0_FIXTURE  / RLX_VLASH_PI05_FIXTURE   → dump directory
//!   RLX_VLASH_PI0_MODEL    / RLX_VLASH_PI05_MODEL      → checkpoint dir
//! ```
//! Tests skip gracefully (print + return) when the env vars or files are
//! absent, so the suite passes without the multi-GB checkpoints. Run with:
//! ```text
//!   RLX_VLASH_PI05_FIXTURE=… RLX_VLASH_PI05_MODEL=… \
//!     cargo test -p rlx-vlash --test parity -- --nocapture
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rlx_core::flow_util::compile_built;
use rlx_runtime::Device;
use rlx_vlash::config::VlashVariant;
use rlx_vlash::prefix::{assemble_prefix, build_attn_inputs};
use rlx_vlash::sample::{sample_actions, time_input};
use rlx_vlash::vision::{build_vision_flow, extract_vision_embed};
use rlx_vlash::{VlashConfig, build_denoise_flow, weights};

// ---------------------------------------------------------------- fixtures ----

struct Fixture {
    dir: PathBuf,
    manifest: serde_json::Value,
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Fixture {
    fn load(dir: &Path) -> Option<Self> {
        let mpath = dir.join("manifest.json");
        if !mpath.is_file() {
            return None;
        }
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&mpath).ok()?).ok()?;
        let mut tensors = HashMap::new();
        if let Some(obj) = manifest.as_object() {
            for (name, meta) in obj {
                let Some(file) = meta.get("file").and_then(|v| v.as_str()) else {
                    continue;
                };
                let shape: Vec<usize> = meta
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
                    .unwrap_or_default();
                let bytes = std::fs::read(dir.join(file)).ok()?;
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                tensors.insert(name.clone(), (data, shape));
            }
        }
        Some(Fixture {
            dir: dir.to_path_buf(),
            manifest,
            tensors,
        })
    }

    fn get(&self, name: &str) -> Option<&(Vec<f32>, Vec<usize>)> {
        self.tensors.get(name)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn assert_parity(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length {} != {}", got.len(), want.len());
    let cos = cosine(got, want);
    let mx = max_abs(got, want);
    println!("  [{label}] cos={cos:.6} max|Δ|={mx:.3e} n={}", got.len());
    assert!(cos > 0.999, "{label}: cosine {cos:.6} <= 0.999 (max|Δ|={mx:.3e})");
}

fn fixture_dir(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from).filter(|p| p.is_dir())
}

fn model_safetensors(var: &str) -> Option<String> {
    let dir = std::env::var(var).ok()?;
    let p = Path::new(&dir);
    let single = p.join("model.safetensors");
    if single.is_file() {
        return Some(single.to_string_lossy().into_owned());
    }
    if p.is_file() {
        return Some(dir);
    }
    None
}

// ------------------------------------------------------------------ stages ----

/// Stage 1 (M1): preprocessing — needs only the fixture (no weights).
fn run_preprocess_parity(fix: &Fixture) {
    let (Some((img, ishape)), Some((pv, _))) = (fix.get("image_chw01"), fix.get("pixel_values"))
    else {
        println!("  (image_chw01/pixel_values absent — skip preprocess)");
        return;
    };
    let (h, w) = (ishape[ishape.len() - 2], ishape[ishape.len() - 1]);
    let got = rlx_vlash::resize_with_pad_normalize(img, h, w, 224);
    assert_parity("preprocess", &got, pv);
}

/// Stages 2–5 (M2–M7): vision → prefix → velocity → actions. Needs weights.
fn run_graph_parity(variant: VlashVariant, fix: &Fixture, model_st: &str) {
    let cfg = VlashConfig::for_variant(variant);

    // num_images == 1 supported by the harness dump.
    let num_images = fix
        .manifest
        .get("num_images")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    if num_images != 1 {
        println!("  (num_images={num_images} != 1 — skip graph parity)");
        return;
    }

    let mut wm = weights::load_remapped(model_st).expect("load weights");
    let vision_embed = extract_vision_embed(&mut wm, &cfg.vision).expect("vision embed");
    let (embed_tokens, embed_shape) = wm.take("vlm.embed_tokens.weight").expect("embed_tokens");
    let vocab = embed_shape[0];

    // ---- vision (stage 2) ----
    let vision_built = build_vision_flow(&cfg.vision, &mut wm, 1).expect("build vision");
    let mut vision = compile_built(vision_built, Device::Cpu).expect("compile vision");
    let (pv, _) = fix.get("pixel_values").expect("pixel_values");
    let hidden = rlx_siglip2::assemble_vision_hidden(
        &vision_embed,
        pv,
        1,
        cfg.vision.patch_size,
        cfg.image_size,
    )
    .expect("assemble vision hidden");
    let image_features = vision.run(&[("hidden", hidden.as_slice())]).remove(0);
    if let Some((ref_feat, _)) = fix.get("image_features_raw") {
        assert_parity("vision.image_features_raw", &image_features, ref_feat);
    }

    // ---- prefix (stage 3) ----
    let (tok_f, tshape) = fix.get("token_ids").expect("token_ids");
    let token_ids: Vec<i64> = tok_f.iter().map(|&x| x as i64).collect();
    let token_mask: Vec<f32> = match fix.get("token_mask") {
        Some((m, _)) => m.clone(),
        None => vec![1.0; token_ids.len()],
    };
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
    if let Some((ref_prefix, _)) = fix.get("prefix_embeds") {
        assert_parity("prefix_embeds", &prefix.emb, ref_prefix);
    }
    let _ = tshape;

    // ---- denoise graph ----
    let attn = build_attn_inputs(&cfg, &prefix.pad);
    let denoise_built = build_denoise_flow(&cfg, &mut wm, prefix.len).expect("build denoise");
    let mut denoise = compile_built(denoise_built, Device::Cpu).expect("compile denoise");

    let (state, _) = fix.get("state_padded").expect("state_padded");
    let (noise, _) = fix.get("noise").expect("noise");

    // ---- velocity at time=1 (stage 4) ----
    if let Some((ref_v0, _)) = fix.get("velocity_step0") {
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
        assert_parity("velocity_step0", &v0, ref_v0);
    }

    // ---- full sample_actions (stage 5) ----
    if let Some((ref_actions, _)) = fix.get("actions_padded") {
        let actions = sample_actions(&mut denoise, &cfg, &prefix.emb, state, &attn, noise);
        assert_parity("actions_padded", &actions, ref_actions);
    }
}

fn run_all(variant: VlashVariant, fix_var: &str, model_var: &str) {
    let Some(dir) = fixture_dir(fix_var) else {
        println!("{fix_var} unset/missing — skipping {variant:?} parity");
        return;
    };
    let Some(fix) = Fixture::load(&dir) else {
        println!("no manifest.json in {} — skipping", dir.display());
        return;
    };
    println!("== {variant:?} parity from {} ==", fix.dir.display());
    run_preprocess_parity(&fix);
    match model_safetensors(model_var) {
        Some(st) => run_graph_parity(variant, &fix, &st),
        None => println!("  {model_var} unset/missing — skipping graph stages (preprocess only)"),
    }
}

#[test]
fn pi0_parity_cpu() {
    run_all(VlashVariant::Pi0, "RLX_VLASH_PI0_FIXTURE", "RLX_VLASH_PI0_MODEL");
}

#[test]
fn pi05_parity_cpu() {
    run_all(VlashVariant::Pi05, "RLX_VLASH_PI05_FIXTURE", "RLX_VLASH_PI05_MODEL");
}
