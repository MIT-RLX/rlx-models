// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: the Pixtral vision flow (ViT + patch-merger + MLP projector) with
//! tiny synthetic weights, compiled on the `RLX_TEST_DEVICE` backend; finite out.
//! Set `RLX_TEST_DEVICE=metal|mlx|gpu|coreml|cuda|vulkan` (default CPU) and build
//! the matching cargo feature to exercise a backend.

use rlx_core::flow_util::compile_built;
use rlx_mistral_vl::config::PixtralVisionConfig;
use rlx_mistral_vl::encoder::{PixtralLayerWeights, PixtralWeights};
use rlx_mistral_vl::flow::build_pixtral_vision;
use rlx_runtime::Device;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.1
        })
        .collect()
}

fn layer(h: usize, n_ff: usize, seed: u64) -> PixtralLayerWeights {
    PixtralLayerWeights {
        ln1: fill(h, seed + 1),
        ln2: fill(h, seed + 2),
        // q/k/v/o are square [h, h] (row-major, gguf convention).
        q: fill(h * h, seed + 3),
        k: fill(h * h, seed + 4),
        v: fill(h * h, seed + 5),
        o: fill(h * h, seed + 6),
        // gate/up are [n_ff, h]; down is [h, n_ff].
        gate: fill(n_ff * h, seed + 7),
        up: fill(n_ff * h, seed + 8),
        down: fill(h * n_ff, seed + 9),
    }
}

#[test]
fn pixtral_vision_flow_compiles_and_runs() {
    // Tiny config; spatial_merge_size = 1 so no patch-merger is required.
    let cfg = PixtralVisionConfig {
        hidden_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        intermediate_size: 16,
        projector_output_dim: 8,
        spatial_merge_size: 1,
        layer_norm_eps: 1e-5,
        use_silu: true,
        ..Default::default()
    };

    let h = cfg.hidden_size;
    let n_ff = cfg.intermediate_size;
    let proj = cfg.projector_output_dim;
    let half = cfg.head_dim() / 2;
    let (grid_x, grid_y) = (2usize, 2usize);
    let n_pos = grid_x * grid_y;

    let weights = PixtralWeights {
        patch_embd: Vec::new(), // patch embed is host-side; unused by the flow.
        pre_ln: Vec::new(),
        layers: (0..cfg.num_hidden_layers)
            .map(|i| layer(h, n_ff, 100 + i as u64 * 20))
            .collect(),
        mm_input_norm: None,
        mm_patch_merger: None,
        mm1_w: fill(proj * h, 11), // [proj, h]
        mm1_b: Some(fill(proj, 12)),
        mm2_w: fill(proj * proj, 13), // [proj, proj]
        mm2_b: Some(fill(proj, 14)),
        img_break: None,
    };

    let built = build_pixtral_vision(&cfg, &weights, grid_x, grid_y).expect("build pixtral vision");
    let n_merged = built.n_merged;
    let mut compiled = compile_built(built.model, dev()).expect("compile pixtral vision flow");

    let hidden = fill(n_pos * h, 21);
    let cos = fill(n_pos * half, 22);
    let sin = fill(n_pos * half, 23);
    let out = compiled
        .run(&[
            ("hidden", hidden.as_slice()),
            ("vision_rope_cos", cos.as_slice()),
            ("vision_rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("pixtral vision forward returned output");

    assert_eq!(out.len(), n_merged * proj);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "vision embeds must be finite"
    );
}
