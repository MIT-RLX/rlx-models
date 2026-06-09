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

//! Env-gated Diamond Maps on real FLUX denoiser weights.
//!
//! ```text
//! FLUX_GGUF_PATH=/path/to/flux.gguf \
//!   cargo test -p rlx-models --test flux2_diamond_guidance --release -- --nocapture
//!
//! Optional flow-map LoRA (safetensors base weights only; not GGUF):
//! FLUX_FLOW_MAP_LORA=/path/to/pytorch_lora_weights.safetensors
//!
//! Compare guided vs unguided blueness (may be noisy on small seq):
//! FLUX_DIAMOND=1 FLUX_GGUF_PATH=... cargo test -p rlx-models --test flux2_diamond_guidance blueness --release
//! ```

#[path = "vision_gguf/support.rs"]
mod support;

use rlx_models::diamond::LatentReward;
use rlx_models::flux2::{
    BluenessReward, DiamondGuidanceParams, DiamondMethod, Flux2SampleParams, init_latent_noise,
    prepare_latent_ids, sample_rectified_flow, sample_rectified_flow_diamond,
};
use rlx_models::{Flux2Runner, Flux2RunnerBuilder};
use rlx_runtime::Device;
use std::path::PathBuf;

use support::{env_path, flux_model_root};

fn flux_weights_path() -> Option<PathBuf> {
    flux_model_root("FLUX_GGUF_PATH", "FLUX_GGUF_PATH")
        .or_else(|| support::env_gguf_path("FLUX_GGUF_PATH"))
}

fn flow_map_lora_path() -> Option<PathBuf> {
    env_path("FLUX_FLOW_MAP_LORA")
}

fn small_seq() -> (usize, usize) {
    let img = std::env::var("FLUX_GGUF_IMG_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let txt = std::env::var("FLUX_GGUF_TXT_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    (img, txt)
}

fn build_runner() -> Flux2Runner {
    let path = flux_weights_path().expect("FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
    let (img_seq, txt_seq) = small_seq();
    let mut b = Flux2RunnerBuilder::default()
        .weights(&path)
        .skip_text_encoder(true)
        .device(Device::Cpu)
        .compiled_denoiser(false)
        .dual_time_embedder(true)
        .batch(1)
        .img_seq(img_seq)
        .txt_seq(txt_seq);
    if let Some(lora) = flow_map_lora_path() {
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            eprintln!(
                "skip FLUX_FLOW_MAP_LORA: LoRA merge requires safetensors base weights, not GGUF ({path:?})"
            );
        } else {
            b = b.lora(lora, 1.0);
        }
    }
    b.build()
        .unwrap_or_else(|e| panic!("Flux2Runner::build {path:?}: {e:#}"))
}

type SampleLayout = (usize, usize, usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn sample_layout(runner: &Flux2Runner) -> SampleLayout {
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
    (
        latent_h, latent_w, img_seq, img_ids, txt_ids, encoder, guidance,
    )
}

#[test]
fn flux2_dual_forward_native_gguf() {
    let Some(_) = flux_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let runner = build_runner();
    let cfg = runner.config();
    let b = runner.batch();
    let (_, _, img_seq, img_ids, txt_ids, encoder, guidance) = sample_layout(&runner);
    let hidden = vec![0.01f32; b * img_seq * cfg.in_channels];
    let t = vec![0.5f32; b];
    let t2 = vec![0.25f32; b];
    let single = runner
        .forward_noise_native(&hidden, &encoder, &t, Some(&guidance), &img_ids, &txt_ids)
        .expect("single-time forward");
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
        .expect("dual-time forward");
    assert_eq!(single.len(), dual.len());
    let max_diff = single
        .iter()
        .zip(&dual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-5,
        "dual-time velocity should differ from single-time on real weights (max_diff={max_diff})"
    );
    eprintln!("flux2 dual forward ok: max_abs_diff={max_diff:.6}");
}

#[test]
fn flux2_diamond_glass_sample_quick_check() {
    let Some(path) = flux_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let runner = build_runner();
    let (latent_h, latent_w, _, _img_ids, txt_ids, encoder, guidance) = sample_layout(&runner);
    let steps = 3usize;
    let params = Flux2SampleParams {
        encoder_hidden_states: &encoder,
        encoder_negative: None,
        neg_txt_ids: None,
        txt_ids: &txt_ids,
        num_inference_steps: steps,
        cfg_scale: 1.0,
        guidance: Some(&guidance),
        latent_h,
        latent_w,
        seed: 7,
        init_timestep: 0,
        initial_latents: None,
        reference: None,
    };
    let diamond = DiamondGuidanceParams {
        method: DiamondMethod::Glass,
        mc_samples: 1,
        inner_steps: 2,
        guidance_steps: 1,
        seed: 7,
        ..DiamondGuidanceParams::default()
    };
    let reward = BluenessReward { scale: 1.0 };
    let out = sample_rectified_flow_diamond(&runner, &params, &diamond, &reward)
        .unwrap_or_else(|e| panic!("glass diamond sample from {path:?}: {e:#}"));
    assert_eq!(
        out.latents.len(),
        runner.batch() * runner.img_seq() * runner.config().in_channels
    );
    assert!(out.latents.iter().all(|v| v.is_finite()));
    eprintln!(
        "flux2 diamond glass ok: ||lat||₂={:.4}",
        out.latents.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
}

#[test]
fn flux2_diamond_weighted_sample_quick_check() {
    let Some(path) = flux_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let runner = build_runner();
    let (latent_h, latent_w, _, _, txt_ids, encoder, guidance) = sample_layout(&runner);
    let steps = 3usize;
    let params = Flux2SampleParams {
        encoder_hidden_states: &encoder,
        encoder_negative: None,
        neg_txt_ids: None,
        txt_ids: &txt_ids,
        num_inference_steps: steps,
        cfg_scale: 1.0,
        guidance: Some(&guidance),
        latent_h,
        latent_w,
        seed: 11,
        init_timestep: 0,
        initial_latents: None,
        reference: None,
    };
    let diamond = DiamondGuidanceParams {
        method: DiamondMethod::Weighted,
        mc_samples: 1,
        guidance_steps: 1,
        use_flow_map: true,
        seed: 11,
        ..DiamondGuidanceParams::default()
    };
    let reward = BluenessReward { scale: 1.0 };
    let out = sample_rectified_flow_diamond(&runner, &params, &diamond, &reward)
        .unwrap_or_else(|e| panic!("weighted diamond sample from {path:?}: {e:#}"));
    assert!(out.latents.iter().all(|v| v.is_finite()));
    eprintln!("flux2 diamond weighted ok (len={})", out.latents.len());
}

#[test]
fn flux2_diamond_blueness_vs_baseline() {
    if std::env::var("FLUX_DIAMOND").ok().as_deref() != Some("1") {
        eprintln!("skip flux2_diamond_blueness_vs_baseline: set FLUX_DIAMOND=1");
        return;
    }
    let Some(_) = flux_weights_path() else {
        eprintln!("skip: set FLUX_GGUF_PATH or FLUX_MODEL_ROOT");
        return;
    };
    let runner = build_runner();
    let (latent_h, latent_w, img_seq, _, txt_ids, encoder, guidance) = sample_layout(&runner);
    let cfg = runner.config();
    let b = runner.batch();
    let steps = std::env::var("FLUX_DIAMOND_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4usize);
    let seed = 99u64;
    let init = init_latent_noise(b, img_seq, cfg.in_channels, seed);
    let base_params = Flux2SampleParams {
        encoder_hidden_states: &encoder,
        encoder_negative: None,
        neg_txt_ids: None,
        txt_ids: &txt_ids,
        num_inference_steps: steps,
        cfg_scale: 1.0,
        guidance: Some(&guidance),
        latent_h,
        latent_w,
        seed,
        init_timestep: 0,
        initial_latents: Some(&init),
        reference: None,
    };
    let reward = BluenessReward { scale: 1.0 };
    let baseline = sample_rectified_flow(&runner, &base_params).expect("baseline sample");
    let diamond_params = DiamondGuidanceParams {
        method: DiamondMethod::Glass,
        mc_samples: 2,
        inner_steps: 3,
        guidance_steps: steps.saturating_sub(1).max(1),
        reward_scale: 2.0,
        seed,
        ..DiamondGuidanceParams::default()
    };
    let guided = sample_rectified_flow_diamond(&runner, &base_params, &diamond_params, &reward)
        .expect("guided sample");
    let r_base = reward.reward(&baseline.latents);
    let r_guided = reward.reward(&guided.latents);
    eprintln!("blueness baseline={r_base:.6} guided={r_guided:.6}");
    assert!(
        r_guided >= r_base - 1e-3,
        "guided blueness {r_guided} should not be below baseline {r_base}"
    );
}
