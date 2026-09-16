//! The lens on **real Qwen3.5 weights**.
//!
//! Everything else in this crate runs on synthetic weights, which answer "is
//! the derivative of this graph right" but not "does this hold on a trained
//! model". A trained model has structure synthetic weights do not: residual
//! blocks that are genuinely near-identity, attention that actually attends,
//! and a gated delta-net whose recurrence carries real signal across positions.
//!
//! Skipped unless the weights are present, so the suite still runs on a machine
//! without them. Point `RLX_JLENS_QWEN35_GGUF` at a `.gguf`, or drop one in
//! `weights/Qwen3.5-0.8B-gguf/`.
//!
//! Note what "real" means here: the checkpoint on hand is K-quantized, and
//! `Qwen35Weights::from_loader` dequantizes to f32. The Jacobian fitted is
//! therefore the Jacobian *of the dequantized model* — the right object for
//! this crate's purposes, but not bit-comparable to a lens fitted on the
//! original bf16 release.

#![cfg(feature = "qwen35")]

use rlx_core::weight_loader::GgufLoader;
use rlx_jlens::models::qwen35::{BlockKind, Qwen35LensModel};
use rlx_jlens::{BlockLens, FitConfig, LensModel, StackLens};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::Device;

/// Short enough to keep the per-block VJP sweep quick; long enough to leave
/// interior positions after the sink skip.
const SEQ: usize = 24;
const SKIP_FIRST: usize = 4;

fn weights_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("RLX_JLENS_QWEN35_GGUF") {
        let p = std::path::PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/Qwen3.5-0.8B-gguf");
    let mut found: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    // Highest-fidelity quantization available, for the least dequantization error.
    found.sort();
    found.into_iter().next_back()
}

fn load() -> Option<Qwen35LensModel> {
    let path = weights_path()?;
    eprintln!("loading {} on {:?}", path.display(), device());
    let mut loader = GgufLoader::from_file(path.to_str().unwrap()).expect("open gguf");
    let cfg = Qwen35Config::from_gguf(loader.file()).expect("parse config");
    let weights = Qwen35Weights::from_loader(&mut loader, &cfg).expect("load weights");
    Some(Qwen35LensModel::new(cfg, weights).with_name("qwen35-0.8b"))
}

macro_rules! model_or_skip {
    () => {
        match load() {
            Some(m) => m,
            None => {
                eprintln!("SKIP: no Qwen3.5 gguf found (set RLX_JLENS_QWEN35_GGUF)");
                return;
            }
        }
    };
}

/// Device under test. `RLX_JLENS_DEVICE=metal` (or `mlx`, `cpu`) selects it;
/// the crate must be built with the matching feature or the backend is not
/// compiled in.
fn device() -> Device {
    match std::env::var("RLX_JLENS_DEVICE") {
        Ok(s) if !s.is_empty() => {
            rlx_runtime::parse_device(&s).unwrap_or_else(|e| panic!("RLX_JLENS_DEVICE: {e}"))
        }
        _ => Device::Cpu,
    }
}

fn config() -> FitConfig {
    FitConfig {
        dim_batch: 16,
        skip_first: SKIP_FIRST,
        device: device(),
    }
}

/// `Op::GatedDeltaNetBackward` has CPU and Metal kernels. A backend without one
/// has to fall back to the unrolled decomposition, which is built from
/// primitives every backend runs — slower, but correct, and verified equivalent
/// in `rlx-autodiff/tests/gated_delta_net_fused_backward.rs`.
///
/// Set as a process-wide override rather than plumbed through, because the
/// unfuse decision lives in `prepare_graph_for_ad`, which is backend-agnostic.
/// Making it target-aware is the proper fix.
fn select_gdn_backward_path() {
    let has_kernel = matches!(device(), Device::Cpu | Device::Metal);
    if has_kernel {
        rlx_ir::env::unset("RLX_GDN_UNFUSE_FOR_AD");
    } else {
        rlx_ir::env::set("RLX_GDN_UNFUSE_FOR_AD", "1");
    }
}

/// Deterministic residual with realistic scale. A trained model's residual
/// stream is roughly unit-RMS per position after the first few layers.
fn residual(d_model: usize) -> Vec<f32> {
    (0..SEQ * d_model)
        .map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 29;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 32;
            ((x >> 40) as f32) / 8_388_608.0 - 1.0
        })
        .collect()
}

#[test]
fn reports_the_real_model_shape() {
    let m = model_or_skip!();
    assert_eq!(m.d_model(), 1024, "Qwen3.5-0.8B embedding_length");
    assert_eq!(m.n_layers(), 24, "block_count");
    // full_attention_interval = 4 ⇒ every 4th block is attention.
    assert_eq!(m.block_kind(0), BlockKind::GatedDeltaNet);
    assert_eq!(m.block_kind(3), BlockKind::FullAttention);
    assert_eq!(m.block_kind(7), BlockKind::FullAttention);
    let attention = (0..m.n_layers())
        .filter(|&l| m.block_kind(l) == BlockKind::FullAttention)
        .count();
    eprintln!(
        "{}: {} layers, d_model {}, {attention} attention / {} gated-delta-net",
        m.name(),
        m.n_layers(),
        m.d_model(),
        m.n_layers() - attention
    );
}

/// Fit a real per-block Jacobian and check it looks like one.
fn fit_block(layer: usize) {
    let m = model_or_skip!();
    let kind = m.block_kind(layer);
    select_gdn_backward_path();
    let mut lens = BlockLens::new(&m, layer, SEQ, config()).expect("build block lens");
    let h = residual(m.d_model());
    let batched = lens.replicate(&h).expect("replicate");

    let start = std::time::Instant::now();
    let j = lens.jacobian(&batched).expect("fit jacobian");
    let elapsed = start.elapsed();

    let d = j.d_model;
    assert_eq!(j.values.len(), d * d);
    assert!(
        j.values.iter().all(|v| v.is_finite()),
        "layer {layer} ({kind:?}): Jacobian has non-finite entries"
    );

    // A residual block is near-identity: J ≈ I + (block's own derivative).
    // The diagonal should sit near 1 and dominate its row.
    let diag: Vec<f32> = (0..d).map(|i| j.values[i * d + i]).collect();
    let mean_diag = diag.iter().sum::<f32>() / d as f32;
    let mut off_diag_max = 0.0f32;
    for i in 0..d {
        for k in 0..d {
            if i != k {
                off_diag_max = off_diag_max.max(j.values[i * d + k].abs());
            }
        }
    }
    eprintln!(
        "layer {layer:2} ({kind:?}): {} passes in {:.1}s, mean diagonal {mean_diag:.4}, \
         max off-diagonal {off_diag_max:.4}, ||J||/sqrt(d) {:.4}",
        lens.passes(),
        elapsed.as_secs_f64(),
        j.scaled_norm()
    );
    // The split should show one forward against `passes()` replays. If the
    // forward count tracked the pass count, `split_vjp` would not be working.
    let t = lens.timing();
    eprintln!(
        "           {} saved activations | {}",
        lens.saved_activations(),
        t.summary()
    );
    assert_eq!(
        t.forward_runs, 1,
        "the forward should run once per residual, not once per pass"
    );
    assert_eq!(t.replay_runs, lens.passes());
    assert!(
        mean_diag > 0.5 && mean_diag < 2.0,
        "layer {layer} ({kind:?}): mean diagonal {mean_diag} — a residual block's \
         Jacobian should sit near the identity"
    );
    assert!(
        off_diag_max > 1e-4,
        "layer {layer} ({kind:?}): Jacobian is purely diagonal (max off-diagonal \
         {off_diag_max}) — the block appears to contribute nothing"
    );
}

#[test]
fn fits_a_gated_delta_net_block_on_real_weights() {
    fit_block(0);
}

#[test]
fn fits_an_attention_block_on_real_weights() {
    fit_block(3);
}

/// The Jacobian fitted on real weights, against finite differences.
///
/// The plausibility checks above say `J` *looks* like a residual block's
/// Jacobian. This says it is the right one, against an oracle that shares no
/// code with the VJP: perturb one `(source position, input dim)` at a time,
/// sum the response over target positions, average over source positions —
/// the estimator's own reduction — and compare a full column of `J`.
///
/// One column is `n_positions × 2` forward passes, so this checks two columns
/// rather than all 1024. Each central difference is computed at two step sizes
/// and skipped where they disagree: RMSNorm's `1/rms` factor makes some points
/// ill-conditioned, and there a finite difference is not measuring a
/// derivative at all.
#[test]
fn real_weight_jacobian_matches_finite_differences() {
    let m = model_or_skip!();
    // Attention block: the fast one, and the one with the most off-diagonal
    // structure to get wrong.
    let layer = 3;
    let d = m.d_model();
    let cfg = config();

    select_gdn_backward_path();
    let mut lens = BlockLens::new(&m, layer, SEQ, cfg).expect("build block lens");
    let positions: Vec<usize> = lens.positions().to_vec();
    let h = residual(d);
    let j = lens
        .jacobian(&lens.replicate(&h).expect("replicate"))
        .expect("fit jacobian");

    // Batch-1 graph for the oracle: one sequence, not the dim_batch replicas.
    let block = m.block(layer, 1, SEQ).expect("batch-1 block");
    let mut fwd = rlx_runtime::Session::new(Device::Cpu).compile(block.graph);
    for (name, data) in &block.params {
        fwd.set_param(name, data);
    }
    let input = block.residual_input.as_str();

    let eps = 5e-3f32;
    let mut compared = 0usize;
    let mut skipped = 0usize;
    let mut worst = 0.0f64;

    for &col in &[17usize, 613] {
        // reference[i] = mean over source positions of the summed response.
        let mut reference = vec![0.0f64; d];
        let mut usable_positions = 0usize;
        for &src in &positions {
            let idx = src * d + col;
            let mut at = |step: f32| -> Vec<f32> {
                let mut hp = h.clone();
                hp[idx] += step;
                fwd.run(&[(input, &hp[..])])[0].clone()
            };
            let coarse_p = at(eps);
            let coarse_m = at(-eps);
            let fine_p = at(eps * 0.5);
            let fine_m = at(-eps * 0.5);

            // Summed over target positions, per output dim, at two step sizes.
            let summed = |plus: &[f32], minus: &[f32], step: f32| -> Vec<f64> {
                (0..d)
                    .map(|i| {
                        let acc: f64 = positions
                            .iter()
                            .map(|&t| (plus[t * d + i] - minus[t * d + i]) as f64)
                            .sum();
                        acc / (2.0 * step as f64)
                    })
                    .collect()
            };
            let coarse = summed(&coarse_p, &coarse_m, eps);
            let fine = summed(&fine_p, &fine_m, eps * 0.5);

            let disagreement = coarse
                .iter()
                .zip(&fine)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if disagreement > 5e-2 {
                skipped += 1;
                continue;
            }
            usable_positions += 1;
            for i in 0..d {
                reference[i] += fine[i];
            }
        }
        assert!(
            usable_positions * 2 >= positions.len(),
            "column {col}: only {usable_positions}/{} source positions gave a usable \
             finite difference",
            positions.len()
        );
        // The fitted J averaged over ALL positions; the oracle over the usable
        // ones. With most positions usable the two agree to the tolerance below.
        for r in reference.iter_mut() {
            *r /= usable_positions as f64;
        }

        for i in 0..d {
            let got = j.values[i * d + col] as f64;
            let want = reference[i];
            let delta = (want - got).abs();
            worst = worst.max(delta);
            compared += 1;
            assert!(
                delta < 5e-2,
                "J[{i}][{col}]: estimator {got} vs finite-difference {want}"
            );
        }
        eprintln!(
            "column {col}: {usable_positions}/{} positions usable",
            positions.len()
        );
    }

    eprintln!(
        "real-weight Jacobian vs finite differences: {compared} entries, \
         {skipped} positions skipped, worst delta {worst:.2e}"
    );
    assert!(compared > 0, "nothing was compared");
}

/// The regression that motivated the fusion fix, on real weights: a block must
/// carry gradient from a later position back to an earlier one.
///
/// On synthetic weights the cross-position signal is tiny; on a trained model
/// it is the whole point of the layer. This is the strongest available check
/// that `Rewriter::copy_node`'s duplicate-Param bug is really gone.
#[test]
fn cross_position_gradient_survives_on_real_weights() {
    let m = model_or_skip!();
    let d = m.d_model();

    for layer in [0usize, 3] {
        let kind = m.block_kind(layer);
        select_gdn_backward_path();
        let mut lens = BlockLens::new(&m, layer, SEQ, config()).expect("build block lens");
        let j = lens
            .jacobian(&lens.replicate(&residual(d)).expect("replicate"))
            .expect("fit jacobian");

        // `write_rows` averages each row over source positions, so a Jacobian
        // whose off-diagonal mass is non-trivial means gradient crossed
        // positions somewhere. Compare the off-diagonal energy to the diagonal.
        let mut diag_energy = 0.0f64;
        let mut off_energy = 0.0f64;
        for i in 0..d {
            for k in 0..d {
                let v = (j.values[i * d + k] as f64).powi(2);
                if i == k {
                    diag_energy += v;
                } else {
                    off_energy += v;
                }
            }
        }
        let ratio = off_energy / diag_energy.max(1e-12);
        eprintln!("layer {layer:2} ({kind:?}): off-diagonal / diagonal energy = {ratio:.4}");
        assert!(
            ratio > 1e-3,
            "layer {layer} ({kind:?}): off-diagonal energy ratio {ratio} — the block's \
             own contribution to the Jacobian has vanished, which is what the \
             duplicate-Param fusion bug used to do"
        );
    }
}

/// The lens proper on real weights: `J_l` for every layer, from one forward.
///
/// This is the object the paper defines — the residual at the final layer
/// differentiated with respect to the residual entering each layer, over a real
/// prompt — rather than the per-block Jacobians the tests above fit.
///
/// What it asserts is the *shape of the answer*, which is strong here because
/// it is not something a broken fit would produce by accident: `J` must
/// approach the identity as the source layer approaches the target (nothing
/// left to transport), and must carry more mass the further back it reaches.
#[test]
fn fits_the_whole_stack_on_real_weights() {
    let m = model_or_skip!();
    select_gdn_backward_path();
    let target = m.n_layers() - 1;
    // Every 4th layer keeps the sweep quick while spanning the depth.
    let layers: Vec<usize> = (0..=target).step_by(4).collect();

    let start = std::time::Instant::now();
    let mut lens = StackLens::new(&m, &layers, target, SEQ, config()).expect("stack lens");
    let tokens = lens
        .replicate_tokens(
            &(0..SEQ)
                .map(|i| ((i * 37 + 11) % 1000) as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let js = lens.jacobians(&tokens).expect("fit jacobians");
    let elapsed = start.elapsed();

    assert_eq!(js.len(), layers.len());
    assert_eq!(
        lens.timing().forward_runs,
        1,
        "one forward for the whole sweep"
    );
    eprintln!(
        "whole stack: {} layers, {} passes, {:.1}s | {} saved | {}",
        layers.len(),
        lens.passes(),
        elapsed.as_secs_f64(),
        lens.saved_activations(),
        lens.timing().summary()
    );

    let d = m.d_model();
    let mut norms = Vec::new();
    for (&l, j) in layers.iter().zip(&js) {
        assert!(
            j.values.iter().all(|v| v.is_finite()),
            "layer {l}: non-finite entries"
        );
        let diag: f32 = (0..d).map(|i| j.values[i * d + i]).sum::<f32>() / d as f32;
        let off: f64 = (0..d)
            .flat_map(|i| (0..d).map(move |k| (i, k)))
            .filter(|(i, k)| i != k)
            .map(|(i, k)| (j.values[i * d + k] as f64).powi(2))
            .sum();
        eprintln!(
            "  layer {l:2} -> {target}: ||J||/sqrt(d) = {:.4}, mean diagonal = {diag:.4}, \
             off-diagonal energy = {:.4}",
            j.scaled_norm(),
            off / (d as f64)
        );
        norms.push((l, j.scaled_norm(), diag));
    }

    // The target layer's own entry tap transports through exactly one block, so
    // its Jacobian must sit near the identity.
    let (last_l, last_norm, last_diag) = *norms.last().unwrap();
    assert_eq!(last_l, *layers.last().unwrap());
    assert!(
        (last_diag - 1.0).abs() < 0.5,
        "layer {last_l} is one block from the target, so its mean diagonal should be \
         near 1, got {last_diag}"
    );

    // Transporting further should not shrink the Jacobian toward nothing.
    let first_norm = norms[0].1;
    assert!(
        first_norm >= last_norm * 0.5,
        "layer {} transports through the whole stack but has a smaller norm ({first_norm}) \
         than the one-block transport ({last_norm}) — gradient is being lost",
        norms[0].0
    );
    assert!(
        norms.iter().all(|(_, n, _)| *n > 1e-3),
        "some layer produced a ~zero Jacobian"
    );
}
