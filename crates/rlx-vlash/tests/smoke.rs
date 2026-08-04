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

//! Arch smoke test: build + compile + run the vision and denoise graphs on CPU
//! with a tiny synthetic checkpoint, for both π₀ and π₀.₅. Validates the graph
//! plumbing (shapes, RoPE, joint two-stream attention, GQA, adaRMS, flow
//! output) without the real multi-GB weights.

use rlx_core::flow_util::compile_built;
use rlx_runtime::Device;
use rlx_vlash::config::VlashVariant;
use rlx_vlash::prefix::build_attn_inputs;
use rlx_vlash::sample::sample_actions;
use rlx_vlash::testkit::{synth_weights, tiny_config};
use rlx_vlash::{build_denoise_flow, build_vision_flow};

fn run_variant(variant: VlashVariant) {
    run_variant_on(variant, Device::Cpu, "cpu");
}

fn run_variant_on(variant: VlashVariant, device: Device, label: &str) {
    let cfg = tiny_config(variant);
    let mut wm = synth_weights(&cfg);

    // ---- vision tower ----
    let batch_img = 1usize;
    let patches = cfg.vision.num_patches();
    let vision_built = build_vision_flow(&cfg.vision, &mut wm, batch_img).expect("build vision");
    let mut vision = compile_built(vision_built, device).expect("compile vision");

    let hidden_len = batch_img * patches * cfg.vision.width;
    let hidden: Vec<f32> = (0..hidden_len)
        .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
        .collect();
    let vout = vision.run(&[("hidden", hidden.as_slice())]);
    let image_features = &vout[0];
    assert_eq!(
        image_features.len(),
        batch_img * patches * cfg.vision.projection_dim
    );
    assert!(
        image_features.iter().all(|v| v.is_finite()),
        "[{label}/{variant:?}] vision output finite"
    );

    // ---- prefix (host) ----
    let prefix_len = patches + 4; // image tokens + 4 pretend text tokens
    let prefix_pad = vec![true; prefix_len];
    let attn = build_attn_inputs(&cfg, &prefix_pad);
    let hidden_dim = cfg.vlm.hidden;
    let prefix_emb: Vec<f32> = (0..prefix_len * hidden_dim)
        .map(|i| ((i % 19) as f32) * 0.01 - 0.09)
        .collect();

    // ---- denoise step ----
    let denoise_built = build_denoise_flow(&cfg, &mut wm, prefix_len).expect("build denoise");
    let mut denoise = compile_built(denoise_built, device).expect("compile denoise");

    let state: Vec<f32> = (0..cfg.max_state_dim).map(|i| i as f32 * 0.1).collect();
    let actions: Vec<f32> = (0..cfg.chunk_size * cfg.max_action_dim)
        .map(|i| (i as f32).sin() * 0.3)
        .collect();
    let time_emb: Vec<f32> = match variant {
        VlashVariant::Pi0 => (0..cfg.chunk_size * cfg.expert.hidden)
            .map(|i| (i as f32 * 0.01).cos())
            .collect(),
        VlashVariant::Pi05 => (0..cfg.expert.hidden)
            .map(|i| (i as f32 * 0.01).cos())
            .collect(),
    };

    let out = denoise.run(&[
        ("prefix_emb", prefix_emb.as_slice()),
        ("state", state.as_slice()),
        ("actions", actions.as_slice()),
        ("time_emb", time_emb.as_slice()),
        ("cos", attn.cos.as_slice()),
        ("sin", attn.sin.as_slice()),
        ("attn_bias", attn.bias.as_slice()),
    ]);
    let velocity = &out[0];
    assert_eq!(
        velocity.len(),
        cfg.chunk_size * cfg.max_action_dim,
        "velocity shape [chunk, action_dim]"
    );
    assert!(
        velocity.iter().all(|v| v.is_finite()),
        "[{label}/{variant:?}] velocity must be finite, got {velocity:?}"
    );
    println!("[{label}/{variant:?}] vision + denoise graphs ran; velocity finite");
}

/// Run both variants on `device` when the backend is present at runtime.
/// Only invoked by the GPU-feature-gated tests below.
#[cfg_attr(
    not(any(
        feature = "metal",
        feature = "mlx",
        feature = "gpu",
        feature = "vulkan",
        feature = "cuda"
    )),
    allow(dead_code)
)]
fn run_backend(device: Device, label: &str) {
    if !rlx_runtime::device_ext::is_available(device) {
        println!("{label} backend unavailable — skipping");
        return;
    }
    run_variant_on(VlashVariant::Pi0, device, label);
    run_variant_on(VlashVariant::Pi05, device, label);
}

#[test]
fn pi0_graphs_build_and_run_cpu() {
    run_variant(VlashVariant::Pi0);
}

#[test]
fn pi05_graphs_build_and_run_cpu() {
    run_variant(VlashVariant::Pi05);
}

#[cfg(feature = "metal")]
#[test]
fn graphs_run_metal() {
    run_backend(Device::Metal, "metal");
}

#[cfg(feature = "mlx")]
#[test]
fn graphs_run_mlx() {
    run_backend(Device::Mlx, "mlx");
}

#[cfg(feature = "gpu")]
#[test]
fn graphs_run_gpu() {
    run_backend(Device::Gpu, "gpu");
}

#[cfg(feature = "vulkan")]
#[test]
fn graphs_run_vulkan() {
    run_backend(Device::Vulkan, "vulkan");
}

#[cfg(feature = "cuda")]
#[test]
fn graphs_run_cuda() {
    run_backend(Device::Cuda, "cuda");
}

/// Prep → load → run: write a GGUF bundle, load it back with `load_prepped`,
/// and confirm the graphs build + run finite from the prepared weights.
#[test]
fn prepped_gguf_bundle_runs_cpu() {
    use rlx_vlash::prep::{QuantScheme, load_prepped, write_gguf};
    let cfg = tiny_config(VlashVariant::Pi05);
    let wm = synth_weights(&cfg);
    let mut path = std::env::temp_dir();
    path.push(format!("rlx_vlash_bundle_{}.gguf", std::process::id()));
    write_gguf(&wm, &path, QuantScheme::F16, VlashVariant::Pi05).expect("write gguf");

    // Load the prepared bundle (canonical keys, dequantized to f32).
    let mut wm2 = load_prepped(path.to_str().unwrap()).expect("load_prepped");

    let patches = cfg.vision.num_patches();
    let prefix_len = patches + 4;
    let denoise_built = build_denoise_flow(&cfg, &mut wm2, prefix_len).expect("build denoise");
    let mut denoise = compile_built(denoise_built, Device::Cpu).expect("compile denoise");

    let prefix_pad = vec![true; prefix_len];
    let attn = build_attn_inputs(&cfg, &prefix_pad);
    let prefix_emb: Vec<f32> = (0..prefix_len * cfg.vlm.hidden)
        .map(|i| ((i % 23) as f32) * 0.01 - 0.11)
        .collect();
    let state: Vec<f32> = (0..cfg.max_state_dim).map(|i| i as f32 * 0.05).collect();
    let noise: Vec<f32> = (0..cfg.chunk_size * cfg.max_action_dim)
        .map(|i| (i as f32 * 0.3).sin())
        .collect();
    let actions = sample_actions(&mut denoise, &cfg, &prefix_emb, &state, &attn, &noise);
    assert!(
        actions.iter().all(|v| v.is_finite()),
        "prepped-bundle actions finite"
    );
    let _ = std::fs::remove_file(&path);
}

/// Exercise the full host denoise loop (`sample_actions`) end-to-end.
#[test]
fn sample_loop_runs_cpu() {
    let cfg = tiny_config(VlashVariant::Pi05);
    let mut wm = synth_weights(&cfg);
    let patches = cfg.vision.num_patches();
    // Build + compile denoise for a fixed prefix.
    let prefix_len = patches + 4;
    let denoise_built = build_denoise_flow(&cfg, &mut wm, prefix_len).expect("build denoise");
    let mut denoise = compile_built(denoise_built, Device::Cpu).expect("compile denoise");

    let prefix_pad = vec![true; prefix_len];
    let attn = build_attn_inputs(&cfg, &prefix_pad);
    let prefix_emb: Vec<f32> = (0..prefix_len * cfg.vlm.hidden)
        .map(|i| ((i % 23) as f32) * 0.01 - 0.11)
        .collect();
    let state: Vec<f32> = (0..cfg.max_state_dim).map(|i| i as f32 * 0.05).collect();
    let noise: Vec<f32> = (0..cfg.chunk_size * cfg.max_action_dim)
        .map(|i| (i as f32 * 0.3).sin())
        .collect();

    let actions = sample_actions(&mut denoise, &cfg, &prefix_emb, &state, &attn, &noise);
    assert_eq!(actions.len(), cfg.chunk_size * cfg.max_action_dim);
    assert!(
        actions.iter().all(|v| v.is_finite()),
        "sampled actions finite"
    );
    // Euler integration should move x_t away from the pure noise input.
    let moved = actions
        .iter()
        .zip(noise.iter())
        .any(|(a, n)| (a - n).abs() > 1e-6);
    assert!(moved, "denoise loop should update x_t");
}
