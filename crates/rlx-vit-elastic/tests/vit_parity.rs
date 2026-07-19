// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Phase-1 gate: the generic ViT forward (a) runs and pools a finite CLS
//! embedding, and (b) **matches the proven `rlx-uni2` reference** — which is
//! bit-exact vs timm — on identical UNI2-shaped weights. That transitively
//! validates attention/LayerScale/packed-SwiGLU/final-norm wiring.

use rlx_runtime::Device;
use rlx_vit_elastic::vit::runner::VitRunner;
use rlx_vit_elastic::vit::{
    VitConfig, assemble_hidden, prepare_from_weightmap, synthetic_checkpoint,
};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

fn fake_image(cfg: &VitConfig, seed: u32) -> Vec<u8> {
    let n = cfg.img_size * cfg.img_size * 3;
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(7);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect()
}

#[cfg(feature = "metal")]
#[test]
fn cpu_metal_forward_parity_and_local_scores() {
    use rlx_vit_elastic::snapvit::{CalibImage, SnapVitConfig, compute_local_scores};

    let cfg = VitConfig::synthetic();

    // Forward parity: CPU vs Metal (inference works on all backends).
    let lc = prepare_from_weightmap(synthetic_checkpoint(&cfg, 5), &cfg).unwrap();
    let lm = prepare_from_weightmap(synthetic_checkpoint(&cfg, 5), &cfg).unwrap();
    let mut rc = VitRunner::from_loaded(cfg.clone(), lc, Device::Cpu, 1).unwrap();
    let mut rm = VitRunner::from_loaded(cfg.clone(), lm, Device::Metal, 1).unwrap();
    let rgb = fake_image(&cfg, 7);
    let (ec, _) = rc.predict_image(&rgb, cfg.img_size, cfg.img_size).unwrap();
    let (em, _) = rm.predict_image(&rgb, cfg.img_size, cfg.img_size).unwrap();
    assert!(em[0].iter().all(|v| v.is_finite()), "Metal forward NaN");
    assert!(cosine(&ec[0], &em[0]) > 0.999, "CPU/Metal forward mismatch");

    // SnapViT with device=Metal succeeds: the backward auto-routes to CPU (the
    // Metal autodiff transpose/narrow-backward NaN), forward/fitness use Metal.
    let ll = prepare_from_weightmap(synthetic_checkpoint(&cfg, 5), &cfg).unwrap();
    let mut sc = SnapVitConfig::new(cfg.img_size);
    sc.crops.n_local = 2;
    sc.crops.blur_prob = 0.0;
    let imgs: Vec<CalibImage> = (0..2)
        .map(|i| CalibImage {
            rgb: fake_image(&cfg, 20 + i),
            h: cfg.img_size,
            w: cfg.img_size,
        })
        .collect();
    let ls = compute_local_scores(&cfg, &ll, &imgs, &sc, Device::Metal).unwrap();
    assert!(
        ls.head.iter().all(|v| v.is_finite()),
        "local-scores NaN: {:?}",
        ls.head
    );
    assert!(ls.ffn.iter().all(|v| v.is_finite()), "FFN local-scores NaN");

    // Full SnapViT run with device=Metal succeeds end-to-end (pipeline routes to
    // CPU internally; the requested device is honored for batch-1 deployment).
    use rlx_vit_elastic::snapvit::{self, SnapVitParams};
    let l2 = prepare_from_weightmap(synthetic_checkpoint(&cfg, 5), &cfg).unwrap();
    let mut p = SnapVitParams::new(cfg.img_size);
    p.ssl.crops.n_local = 2;
    p.ssl.crops.blur_prob = 0.0;
    p.pca_dim = 0;
    p.xnes.population = 6;
    p.xnes.iterations = 2;
    p.xnes.sparsities = vec![0.3];
    p.elastic_sparsities = vec![0.4];
    let res = snapvit::run(&cfg, &l2, &imgs, &imgs, &p, Device::Metal).unwrap();
    assert!(
        res.best_fitness.is_finite() && res.elastic[0].fitness.is_finite(),
        "Metal SnapViT run NaN"
    );
}

#[cfg(any(feature = "mlx", feature = "gpu", feature = "vulkan", feature = "cuda"))]
fn forward_parity_on(dev: Device) {
    for cfg in [VitConfig::synthetic(), VitConfig::synthetic_uni2()] {
        let lc = prepare_from_weightmap(synthetic_checkpoint(&cfg, 5), &cfg).unwrap();
        let ld = prepare_from_weightmap(synthetic_checkpoint(&cfg, 5), &cfg).unwrap();
        let mut rc = VitRunner::from_loaded(cfg.clone(), lc, Device::Cpu, 1).unwrap();
        let mut rd = VitRunner::from_loaded(cfg.clone(), ld, dev, 1).unwrap();
        let rgb = fake_image(&cfg, 7);
        let (ec, _) = rc.predict_image(&rgb, cfg.img_size, cfg.img_size).unwrap();
        let (ed, _) = rd.predict_image(&rgb, cfg.img_size, cfg.img_size).unwrap();
        assert!(
            ed[0].iter().all(|v| v.is_finite()),
            "{dev:?} forward NaN ({:?})",
            cfg.ffn_kind
        );
        let c = cosine(&ec[0], &ed[0]);
        assert!(
            c > 0.999,
            "{dev:?} forward mismatch vs CPU: cos={c} ({:?})",
            cfg.ffn_kind
        );
    }
}

#[cfg(feature = "mlx")]
#[test]
fn forward_parity_mlx() {
    forward_parity_on(Device::Mlx);
}

#[cfg(feature = "gpu")]
#[test]
fn forward_parity_wgpu() {
    forward_parity_on(Device::Gpu);
}

// CUDA forward parity (both ViT topologies). CUDA *forward* inference works
// (incl. exported pruned models via `VitRunner`); the autodiff *backward* is
// unsupported (rlx-cuda `AttentionBackward` panics "unfuse should have promoted
// to rank-4"), just as Metal/MLX NaN — so SnapViT/GLARE route the gradient to
// CPU (see `snapvit::local::backward_device`).
#[cfg(feature = "cuda")]
#[test]
fn forward_parity_cuda() {
    forward_parity_on(Device::Cuda);
}

#[cfg(feature = "vulkan")]
#[test]
fn forward_parity_vulkan() {
    forward_parity_on(Device::Vulkan);
}

// Probe whether a device can run the autodiff backward of the ViT head
// (transpose/narrow layout ops) without NaN/panic. Prints; asserts only the loss.
#[cfg(any(
    feature = "vulkan",
    feature = "cuda",
    feature = "gpu",
    feature = "mlx",
    feature = "metal"
))]
fn backward_probe_on(dev: Device, tag: &str) {
    use rlx_ir::NodeId;
    use rlx_ir::infer::GraphExt;
    use rlx_runtime::{CompileOptions, Session};
    use rlx_vit_elastic::vit::build_vit_graph;
    let cfg = VitConfig::synthetic();
    let ld = prepare_from_weightmap(synthetic_checkpoint(&cfg, 5), &cfg).unwrap();
    let mut vg = build_vit_graph(&cfg, 1);
    let cls = rlx_vit_elastic::vit::extract_cls(&mut vg.graph, vg.output, 1, cfg.hidden_size);
    let n = rlx_vit_elastic::dino::l2_normalize(&mut vg.graph, cls, cfg.hidden_size);
    let loss = vg.graph.mean(n, vec![0, 1], false);
    vg.graph.set_outputs(vec![loss]);
    let wrt: Vec<NodeId> = vg.params.iter().map(|p| p.node).collect();
    let bw = rlx_autodiff::grad_with_loss(&vg.graph, &wrt);
    let mut c = Session::new(dev).compile_with(bw, &CompileOptions::new());
    for p in &vg.params {
        c.set_param(&p.name, &ld.params[&p.name]);
    }
    let hm = vec![1.0f32; cfg.num_hidden_layers * cfg.hidden_size];
    let fm = vec![1.0f32; cfg.num_hidden_layers * cfg.ffn_inner()];
    let im = fake_image(&cfg, 31);
    let nchw = rlx_vit_elastic::vit::rgb_u8_to_imagenet_nchw(
        &im,
        cfg.img_size,
        cfg.img_size,
        cfg.img_size,
    );
    let hid = rlx_vit_elastic::vit::assemble_hidden(&ld.preprocess, &nchw, 1).unwrap();
    let outs = c.run(&[
        ("hidden", hid.as_slice()),
        ("head_mask", hm.as_slice()),
        ("ffn_mask", fm.as_slice()),
        ("d_output", &[1.0f32]),
    ]);
    let grad_finite = outs.iter().skip(1).all(|g| g.iter().all(|v| v.is_finite()));
    eprintln!(
        "{tag}_BACKWARD grad_finite={grad_finite} loss={}",
        outs[0][0]
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_backward_probe() {
    backward_probe_on(Device::Vulkan, "VULKAN");
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_backward_probe() {
    backward_probe_on(Device::Gpu, "WGPU");
}

#[cfg(feature = "mlx")]
#[test]
fn mlx_backward_probe() {
    backward_probe_on(Device::Mlx, "MLX");
}

#[cfg(feature = "metal")]
#[test]
fn metal_backward_probe() {
    backward_probe_on(Device::Metal, "METAL");
}

// Check every sequence position of the Metal forward output for NaN.
#[cfg(feature = "metal")]
#[test]
fn metal_full_output_finite() {
    let cfg = VitConfig::synthetic();
    let lm = prepare_from_weightmap(synthetic_checkpoint(&cfg, 5), &cfg).unwrap();
    let mut rm = VitRunner::from_loaded(cfg.clone(), lm, Device::Metal, 1).unwrap(); // Metal only, no prior CPU runner
    let im = fake_image(&cfg, 31);
    let nchw = rlx_vit_elastic::vit::rgb_u8_to_imagenet_nchw(
        &im,
        cfg.img_size,
        cfg.img_size,
        cfg.img_size,
    );
    let hid = rlx_vit_elastic::vit::assemble_hidden(rm.preprocess(), &nchw, 1).unwrap();
    let om = rm.forward_hidden(&hid).unwrap();
    let h = cfg.hidden_size;
    let all = om.iter().all(|v| v.is_finite());
    for s in 0..cfg.seq_len() {
        let mf = om[s * h..(s + 1) * h].iter().all(|v| v.is_finite());
        eprintln!("pos {s}: metal_finite={mf}");
    }
    eprintln!("METAL_FULL all_finite={all}");
    assert!(
        all,
        "Metal forward NaN on seed-31 input (LayerNorm variance clamp regression)"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_backward_probe() {
    backward_probe_on(Device::Cuda, "CUDA");
}

// The FULL SnapViT pipeline (local-scores backward + xNES fitness forward) run
// natively on a GPU device — finite end-to-end.
#[cfg(any(
    feature = "mlx",
    feature = "gpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "metal"
))]
fn snapvit_native_on(dev: Device) {
    use rlx_vit_elastic::snapvit::{self, CalibImage, SnapVitParams};
    let cfg = VitConfig::synthetic();
    let l = prepare_from_weightmap(synthetic_checkpoint(&cfg, 5), &cfg).unwrap();
    let imgs: Vec<CalibImage> = (0..4)
        .map(|i| CalibImage {
            rgb: fake_image(&cfg, 20 + i),
            h: cfg.img_size,
            w: cfg.img_size,
        })
        .collect();
    let mut p = SnapVitParams::new(cfg.img_size);
    p.ssl.crops.n_local = 2;
    p.ssl.crops.blur_prob = 0.0;
    p.pca_dim = 0;
    p.xnes.population = 6;
    p.xnes.iterations = 2;
    p.xnes.sparsities = vec![0.3];
    p.elastic_sparsities = vec![0.4];
    let res = snapvit::run(&cfg, &l, &imgs, &imgs, &p, dev).unwrap();
    assert!(
        res.best_fitness.is_finite() && res.elastic[0].fitness.is_finite(),
        "{dev:?} snapvit::run NaN"
    );
    eprintln!(
        "{dev:?}_SNAPVIT baseline={:.4} best={:.4}",
        res.baseline_fitness, res.best_fitness
    );
}

#[cfg(feature = "mlx")]
#[test]
fn snapvit_native_mlx() {
    snapvit_native_on(Device::Mlx);
}

#[cfg(feature = "metal")]
#[test]
fn snapvit_native_metal() {
    snapvit_native_on(Device::Metal);
}

#[cfg(feature = "gpu")]
#[test]
fn snapvit_native_wgpu() {
    snapvit_native_on(Device::Gpu);
}

#[cfg(feature = "cuda")]
#[test]
fn snapvit_native_cuda() {
    snapvit_native_on(Device::Cuda);
}

#[cfg(feature = "vulkan")]
#[test]
fn snapvit_native_vulkan() {
    snapvit_native_on(Device::Vulkan);
}

#[test]
fn synthetic_plain_vit_forward_runs() {
    let cfg = VitConfig::synthetic();
    let wm = synthetic_checkpoint(&cfg, 11);
    let loaded = prepare_from_weightmap(wm, &cfg).unwrap();
    let mut runner = VitRunner::from_loaded(cfg.clone(), loaded, Device::Cpu, 1).unwrap();

    let rgb = fake_image(&cfg, 3);
    let (emb, tokens) = runner
        .predict_image(&rgb, cfg.img_size, cfg.img_size)
        .unwrap();
    assert_eq!(emb.len(), 1);
    assert_eq!(emb[0].len(), cfg.hidden_size);
    assert_eq!(tokens.len(), cfg.seq_len() * cfg.hidden_size);
    assert!(emb[0].iter().all(|v| v.is_finite()));
    // A non-degenerate embedding (weights are pseudo-random, not zero).
    let norm: f32 = emb[0].iter().map(|v| v * v).sum();
    assert!(norm > 1e-6, "CLS embedding collapsed to ~0 (norm {norm})");
}

#[test]
fn uni2_forward_matches_rlx_uni2_reference() {
    // Tiny UNI2-shaped topology (LayerScale + packed SwiGLU + 8 registers).
    let cfg = VitConfig::synthetic_uni2();
    let batch = 1;

    // Two identical checkpoints (synthetic_checkpoint is deterministic).
    let wm_mine = synthetic_checkpoint(&cfg, 42);
    let wm_ref = synthetic_checkpoint(&cfg, 42);

    // ---- mine: hand-built Graph forward ----
    let loaded = prepare_from_weightmap(wm_mine, &cfg).unwrap();
    // Assemble the shared "hidden" input from a fake image (my preprocess).
    let rgb = fake_image(&cfg, 5);
    let nchw = rlx_vit_elastic::vit::rgb_u8_to_imagenet_nchw(
        &rgb,
        cfg.img_size,
        cfg.img_size,
        cfg.img_size,
    );
    let hidden = assemble_hidden(&loaded.preprocess, &nchw, batch).unwrap();
    let mut runner = VitRunner::from_loaded(cfg.clone(), loaded, Device::Cpu, batch).unwrap();
    let mine = runner.forward_hidden(&hidden).unwrap();

    // ---- reference: rlx-uni2's proven flow ----
    let uni2_cfg = rlx_uni2::Uni2Config {
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        img_size: cfg.img_size,
        patch_size: cfg.patch_size,
        mlp_hidden_dim: cfg.mlp_hidden_dim,
        layer_norm_eps: cfg.layer_norm_eps,
        num_register_tokens: cfg.num_register_tokens,
    };
    let mut wm_ref = wm_ref;
    let built = rlx_uni2::build_uni2_built(&uni2_cfg, &mut wm_ref, batch).unwrap();
    let typed = built.model.typed_params.clone();
    let (graph, params) = rlx_core::flow_util::graph_from_built(built.model).unwrap();
    let opts = rlx_core::flow_bridge::compile_options_for_profile(
        &rlx_flow::CompileProfile::encoder(),
        Device::Cpu,
    );
    let mut compiled = rlx_runtime::Session::new(Device::Cpu).compile_with(graph, &opts);
    rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
    let reference = compiled
        .run(&[("hidden", hidden.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    assert_eq!(mine.len(), reference.len());
    let c = cosine(&mine, &reference);
    let max_abs = mine
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        c > 0.9999,
        "hand-built UNI2 forward diverges from rlx-uni2 reference: cos={c}, max_abs={max_abs}"
    );
}
