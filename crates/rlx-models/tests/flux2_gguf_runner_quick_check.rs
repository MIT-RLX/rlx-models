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

//! Env-gated: [`Flux2Runner`] with a denoiser `.gguf` (no text encoder / VAE).
//!
//! ```text
//! FLUX_GGUF_PATH=/path/to/denoiser.gguf \
//!   cargo test -p rlx-models --test flux2_gguf_runner_quick_check --release -- --nocapture
//!
//! FLUX_GGUF_PATH=... FLUX_GGUF_COMPILE=1 \
//!   cargo test -p rlx-models --test flux2_gguf_runner_quick_check compiled --release --features metal -- --nocapture
//! ```
//!
//! Layout: pass a `.gguf` file or a model root directory (auto-picks GGUF via
//! `load_weight_map`). Optional sibling `vae/` (`FLUX_VAE_DIR` or walk-up from
//! weights path) enables VAE encode quick check. Text encoder stays off unless you load
//! a full tree and clear `skip_text_encoder`.

#[path = "vision_gguf/support.rs"]
mod support;

use rlx_models::flux2::{
    flow_match_euler_step, flow_match_sigmas, init_latent_noise, prepare_latent_ids,
    resolve_vae_dir,
};
use rlx_models::{Flux2Runner, Flux2RunnerBuilder};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

use support::{compile_gate, env_gguf_path, env_path, flux_model_root};

fn flux_gguf_weights_path() -> Option<PathBuf> {
    flux_model_root("FLUX_GGUF_PATH", "FLUX_GGUF_PATH").or_else(|| env_gguf_path("FLUX_GGUF_PATH"))
}

fn flux_vae_dir(weights: &Path) -> Option<PathBuf> {
    env_path("FLUX_VAE_DIR").or_else(|| resolve_vae_dir(weights))
}

fn small_seq_env(default_img: usize, default_txt: usize) -> (usize, usize) {
    let img = std::env::var("FLUX_GGUF_IMG_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_img);
    let txt = std::env::var("FLUX_GGUF_TXT_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_txt);
    (img, txt)
}

fn build_runner(
    device: Device,
    compiled_denoiser: bool,
    vae_dir: Option<PathBuf>,
    dual_time_embedder: bool,
) -> Flux2Runner {
    let path = flux_gguf_weights_path().expect("FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
    let (img_seq, txt_seq) = small_seq_env(4, 3);
    let mut b = Flux2RunnerBuilder::default()
        .weights(&path)
        .skip_text_encoder(true)
        .device(device)
        .compiled_denoiser(compiled_denoiser)
        .dual_time_embedder(dual_time_embedder)
        .batch(1)
        .img_seq(img_seq)
        .txt_seq(txt_seq);
    if let Some(v) = vae_dir {
        b = b.vae_dir(v);
    }
    b.build()
        .unwrap_or_else(|e| panic!("Flux2Runner::build {path:?}: {e:#}"))
}

#[test]
fn flux2_runner_build_from_gguf() {
    let Some(path) = flux_gguf_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let runner = build_runner(Device::Cpu, false, None, false);
    let cfg = runner.config();
    assert!(cfg.num_layers > 0 && cfg.joint_attention_dim > 0);
    eprintln!(
        "flux2 runner ok from {:?}: layers={} joint_dim={} in_ch={}",
        path, cfg.num_layers, cfg.joint_attention_dim, cfg.in_channels
    );
}

#[test]
fn flux2_runner_forward_noise_native() {
    let Some(_) = flux_gguf_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let runner = build_runner(Device::Cpu, false, None, false);
    let cfg = runner.config();
    let b = runner.batch();
    let img_seq = runner.img_seq();
    let txt_seq = runner.txt_seq();
    let latent_h = 2usize;
    let latent_w = (img_seq / latent_h).max(1);
    let img_ids = prepare_latent_ids(b, latent_h, latent_w);
    let txt_ids = vec![0.0f32; txt_seq * 4];
    let hidden = vec![0.0f32; b * img_seq * cfg.in_channels];
    let encoder = vec![0.0f32; b * txt_seq * cfg.joint_attention_dim];
    let timestep = vec![0.5f32; b];
    let guidance = vec![3.5f32; b];
    let noise = runner
        .forward_noise_native(
            &hidden,
            &encoder,
            &timestep,
            Some(&guidance),
            &img_ids,
            &txt_ids,
        )
        .unwrap_or_else(|e| panic!("forward_noise_native: {e:#}"));
    assert_eq!(noise.len(), b * img_seq * cfg.in_channels);
    assert!(
        noise.iter().any(|v| v.is_finite() && v.abs() > 0.0) || noise.iter().all(|v| v.is_finite())
    );
    eprintln!(
        "flux2 native denoise ok: img_seq={img_seq} txt_seq={txt_seq} noise[0..4]={:?}",
        &noise[..noise.len().min(4)]
    );
}

#[test]
fn flux2_runner_forward_noise_compiled_cpu() {
    if !compile_gate() {
        eprintln!("skip flux2_runner_forward_noise_compiled_cpu: set FLUX_GGUF_COMPILE=1");
        return;
    }
    let Some(_) = flux_gguf_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let runner = build_runner(Device::Cpu, true, None, false);
    let cfg = runner.config();
    let b = runner.batch();
    let img_seq = runner.img_seq();
    let txt_seq = runner.txt_seq();
    let latent_h = 2usize;
    let latent_w = (img_seq / latent_h).max(1);
    let img_ids = prepare_latent_ids(b, latent_h, latent_w);
    let txt_ids = vec![0.0f32; txt_seq * 4];
    let hidden = vec![0.0f32; b * img_seq * cfg.in_channels];
    let encoder = vec![0.0f32; b * txt_seq * cfg.joint_attention_dim];
    let noise = runner
        .forward_noise(
            &hidden,
            &encoder,
            &vec![0.5f32; b],
            Some(&vec![3.5f32; b]),
            &img_ids,
            &txt_ids,
        )
        .expect("forward_noise compiled cpu");
    assert_eq!(noise.len(), b * img_seq * cfg.in_channels);
    eprintln!("flux2 compiled cpu denoise ok (len={})", noise.len());
}

#[cfg(feature = "metal")]
#[test]
fn flux2_runner_forward_noise_compiled_metal() {
    if !compile_gate() {
        eprintln!("skip metal: set FLUX_GGUF_COMPILE=1");
        return;
    }
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let Some(_) = flux_gguf_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH");
        return;
    };
    let runner = build_runner(Device::Metal, true, None, false);
    let cfg = runner.config();
    let b = 1usize;
    let img_seq = runner.img_seq().min(16);
    let txt_seq = runner.txt_seq().min(8);
    let latent_h = 2usize;
    let latent_w = (img_seq / latent_h).max(1);
    let img_ids = prepare_latent_ids(b, latent_h, latent_w);
    let txt_ids = vec![0.0f32; txt_seq * 4];
    let hidden = vec![0.0f32; b * img_seq * cfg.in_channels];
    let encoder = vec![0.0f32; b * txt_seq * cfg.joint_attention_dim];
    let noise = runner
        .forward_noise(
            &hidden,
            &encoder,
            &vec![0.5f32; b],
            Some(&vec![3.5f32; b]),
            &img_ids,
            &txt_ids,
        )
        .expect("forward_noise metal");
    assert_eq!(noise.len(), b * img_seq * cfg.in_channels);
    eprintln!("flux2 compiled metal denoise ok");
}

#[test]
fn flux2_runner_dual_forward_native() {
    let Some(_) = flux_gguf_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let runner = build_runner(Device::Cpu, false, None, true);
    let cfg = runner.config();
    let b = runner.batch();
    let img_seq = runner.img_seq();
    let txt_seq = runner.txt_seq();
    let latent_h = 2usize;
    let latent_w = (img_seq / latent_h).max(1);
    let img_ids = prepare_latent_ids(b, latent_h, latent_w);
    let txt_ids = vec![0.0f32; txt_seq * 4];
    let hidden = vec![0.01f32; b * img_seq * cfg.in_channels];
    let encoder = vec![0.0f32; b * txt_seq * cfg.joint_attention_dim];
    let t = vec![0.5f32; b];
    let t2 = vec![0.25f32; b];
    let guidance = vec![3.5f32; b];
    let single = runner
        .forward_noise_native(&hidden, &encoder, &t, Some(&guidance), &img_ids, &txt_ids)
        .expect("single-time");
    let dual = runner
        .forward_noise_dual_native(
            &hidden,
            &encoder,
            &t,
            &t2,
            Some(&guidance),
            &img_ids,
            &txt_ids,
        )
        .expect("dual-time");
    let max_diff = single
        .iter()
        .zip(&dual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-5, "dual-time velocity max_diff={max_diff}");
    eprintln!("flux2 gguf dual forward ok: max_diff={max_diff:.6}");
}

#[test]
fn flux2_runner_two_step_euler_native() {
    let Some(weights_path) = flux_gguf_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let runner = build_runner(Device::Cpu, false, None, false);
    let cfg = runner.config();
    let b = runner.batch();
    let img_seq = runner.img_seq();
    let txt_seq = runner.txt_seq();
    let latent_h = 2usize;
    let latent_w = (img_seq / latent_h).max(1);
    let img_ids = prepare_latent_ids(b, latent_h, latent_w);
    let txt_ids = vec![0.0f32; txt_seq * 4];
    let encoder = vec![0.0f32; b * txt_seq * cfg.joint_attention_dim];
    let guidance = vec![3.5f32; b];

    let steps = 2usize;
    let mut latents = init_latent_noise(b, img_seq, cfg.in_channels, 42);
    let sigmas = flow_match_sigmas(steps);
    for i in 0..steps {
        let timestep = vec![sigmas[i]; b];
        let noise = runner
            .forward_noise_native(
                &latents,
                &encoder,
                &timestep,
                Some(&guidance),
                &img_ids,
                &txt_ids,
            )
            .unwrap_or_else(|e| panic!("euler step {i}: {e:#}"));
        flow_match_euler_step(&mut latents, &noise, sigmas[i], sigmas[i + 1]);
    }
    assert_eq!(latents.len(), b * img_seq * cfg.in_channels);
    eprintln!("flux2 gguf two-step euler ok from {weights_path:?}");
}

#[test]
fn flux2_runner_vae_encode_quick_check() {
    let Some(weights_path) = flux_gguf_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let vae = flux_vae_dir(&weights_path)
        .or_else(|| flux_vae_dir(weights_path.parent().unwrap_or_else(|| Path::new("."))));
    let Some(vae_dir) = vae else {
        eprintln!(
            "skip flux2_runner_vae_encode: no vae/ next to {weights_path:?} (set FLUX_VAE_DIR)"
        );
        return;
    };
    let runner = build_runner(Device::Cpu, false, Some(vae_dir.clone()), false);
    let pixel_h = 64usize;
    let pixel_w = 64usize;
    let rgb_len = runner.batch() * 3 * pixel_h * pixel_w;
    let rgb = vec![0.0f32; rgb_len];
    let latent = runner
        .vae_encode_rgb(&rgb, pixel_h, pixel_w)
        .unwrap_or_else(|e| panic!("vae_encode_rgb from {vae_dir:?}: {e:#}"));
    assert!(!latent.is_empty());
    assert!(latent.iter().all(|v| v.is_finite()));
    eprintln!(
        "flux2 vae encode ok: {vae_dir:?} latent_len={} sample={:?}",
        latent.len(),
        &latent[..latent.len().min(4)]
    );
}

#[test]
fn flux2_runner_img2img_blend_quick_check() {
    if std::env::var("FLUX_IMG2IMG").ok().as_deref() != Some("1") {
        eprintln!("skip flux2_runner_img2img_blend_quick_check: set FLUX_IMG2IMG=1");
        return;
    }
    let Some(weights_path) = flux_gguf_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let vae = flux_vae_dir(&weights_path)
        .or_else(|| flux_vae_dir(weights_path.parent().unwrap_or(Path::new("."))));
    let Some(vae_dir) = vae else {
        eprintln!("skip: set FLUX_VAE_DIR or place vae/ next to weights");
        return;
    };
    let runner = build_runner(Device::Cpu, false, Some(vae_dir), false);
    let cfg = runner.config();
    let b = runner.batch();
    let img_seq = runner.img_seq();
    let latent_h = 2usize;
    let latent_w = (img_seq / latent_h).max(1);
    let pixel_h = latent_h * 8;
    let pixel_w = latent_w * 8;
    let rgb = vec![0.5f32; b * 3 * pixel_h * pixel_w];
    let noise = init_latent_noise(b, img_seq, cfg.in_channels, 7);
    let blended = runner
        .prepare_img2img_packed(
            &rgb, pixel_h, pixel_w, latent_h, latent_w, latent_h, latent_w, &noise, 0.75, 4,
        )
        .unwrap_or_else(|e| panic!("prepare_img2img_packed: {e:#}"));
    assert_eq!(blended.len(), noise.len());
    assert!(blended.iter().any(|v| v.is_finite()));
    eprintln!("flux2 gguf img2img blend ok (len={})", blended.len());
}
