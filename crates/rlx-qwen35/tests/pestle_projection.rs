//! Pestle-factorized projections must compute exactly what the dense
//! linear they replace computes.
//!
//! `Doses-AI/Pestle-27B-Ternary-GGUF` ships every transformer linear as a
//! pair of ternary matrices around a rank-`r` waist plus three
//! per-channel scales, so `builder::emit_linear` expands one `MatMul`
//! into `mul → matmul → mul → matmul → mul`. Four independent
//! conventions have to line up for that to be the same function:
//!
//!   1. factor order — `V` (in → rank) before `U` (rank → out);
//!   2. the `[out, in]` → `[in, out]` transpose `proj_mat` applies, now
//!      applied twice with *different* dims;
//!   3. which axis each scale broadcasts along — `scale_pre` is `[in]`,
//!      `scale_mid` is `[rank]`, `scale_post` is `[out]`, and all three
//!      are per-channel, not per-token;
//!   4. `scale_mid` landing between the factors rather than folded into
//!      either end.
//!
//! Every one of those is silent when wrong: the shapes still check out
//! (Pestle's ranks are close to `in`/`out`), the model still runs, and it
//! still emits fluent text — just not this model's text. So the test
//! doesn't eyeball an output, it builds the *same model twice* —
//! [`synth_weights_pestle`] returns a factorized bundle and its
//! algebraically exact dense equivalent — and requires the logits to
//! agree.
//!
//! Slot→tensor mapping (which of the 8 slots is `ssm_beta` vs
//! `ssm_alpha`, etc.) is a property of the checkpoint, not of the graph,
//! so it can't be covered here; `weights.rs` pins it against the
//! `mortar.cpp` fork, `tests/pestle_real_weights.rs` checks the layout
//! against a real checkpoint, and a greedy run vs `mortar.cpp` itself is
//! what actually confirms the three same-shaped pairs.

use rlx_qwen35::synth::{synth_weights_pestle, tiny_cfg};
use rlx_runtime::{Device, Session};

const BATCH: usize = 1;
const SEQ: usize = 4;
const INPUT_IDS: [f32; SEQ] = [5.0, 11.0, 2.0, 19.0];

/// RLX's full standard backend set (MODELS.md "All 7").
const DEVICES: &[Device] = &[
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Cuda,
    Device::Rocm,
    Device::Gpu, // wgpu
    Device::Vulkan,
];

fn logits_on(device: Device, pestle: bool) -> Vec<f32> {
    let cfg = tiny_cfg();
    let weights = synth_weights_pestle(&cfg, pestle);
    let (hir, params, packed) =
        rlx_qwen35::build_qwen35_prefill_flow(&cfg, &weights, BATCH, SEQ, true, false, false)
            .expect("build prefill flow");
    assert!(packed.is_empty(), "synthetic weights should not be packed");

    let mut compiled = Session::new(device)
        .compile_hir(hir)
        .expect("compile prefill");
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    compiled.run(&[("input_ids", &INPUT_IDS[..])]).remove(0)
}

fn logits(pestle: bool) -> Vec<f32> {
    logits_on(Device::Cpu, pestle)
}

#[test]
fn pestle_factorization_matches_its_dense_equivalent() {
    let dense = logits(false);
    let factored = logits(true);
    assert_eq!(dense.len(), factored.len(), "logit shape changed");

    // Guard the guard: if the model collapsed to constants, "they match"
    // would be vacuous.
    let spread = dense.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - dense.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    assert!(
        spread > 1e-3,
        "dense reference logits are nearly constant (spread {spread}); \
         this fixture cannot distinguish a correct factorization"
    );

    let max_abs = dense
        .iter()
        .zip(&factored)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Same arithmetic, different association order across 3 trunk
    // layers — f32 reassociation, not a modelling difference.
    assert!(
        max_abs < 2e-3,
        "Pestle projections diverge from the dense linear they factorize: \
         max |Δlogit| = {max_abs} over {} values (logit spread {spread})",
        dense.len()
    );
}

/// A Pestle bundle must actually take the factorized path — otherwise
/// the equivalence test above passes by construction.
#[test]
fn pestle_bundle_reports_factorized_projections() {
    use rlx_qwen35::{Qwen35LayerFfn, Qwen35TrunkLayer};
    let cfg = tiny_cfg();
    let w = synth_weights_pestle(&cfg, true);
    let mut seen_pestle = 0usize;
    for layer in &w.trunk_layers {
        let ffn = match layer {
            Qwen35TrunkLayer::Linear(l) => {
                for p in [
                    &l.attn_qkv,
                    &l.attn_gate,
                    &l.ssm_beta,
                    &l.ssm_alpha,
                    &l.ssm_out,
                ] {
                    assert!(p.is_pestle(), "linear-attn projection stayed dense");
                    seen_pestle += 1;
                }
                &l.ffn
            }
            Qwen35TrunkLayer::FullAttn(f) => {
                for p in [&f.attn_q_gate, &f.attn_k, &f.attn_v, &f.attn_output] {
                    assert!(p.is_pestle(), "full-attn projection stayed dense");
                    seen_pestle += 1;
                }
                &f.ffn
            }
        };
        let Qwen35LayerFfn::Dense { gate, up, down } = ffn else {
            panic!("tiny_cfg is dense-FFN");
        };
        for p in [gate, up, down] {
            assert!(p.is_pestle(), "FFN projection stayed dense");
            seen_pestle += 1;
        }
    }
    assert!(
        seen_pestle >= 8,
        "expected a factorized trunk, saw {seen_pestle}"
    );

    // The dense half must be the mirror image, or the comparison above
    // is between two identical graphs.
    let d = synth_weights_pestle(&cfg, false);
    for layer in &d.trunk_layers {
        if let Qwen35TrunkLayer::Linear(l) = layer {
            assert!(!l.attn_qkv.is_pestle(), "dense half is factorized");
        }
    }
}

/// The factorized graph must give the same answer as the dense one on every
/// backend that can run this architecture at all — not just on CPU.
///
/// `emit_linear` adds three broadcast multiplies per projection: a `[in]`,
/// `[rank]` and `[out]` parameter each multiplied against a `[tokens, ·]`
/// activation. Broadcasting a per-channel vector across the token axis is the
/// kind of thing backends disagree about (which operand leads, whether the
/// rank-1 × rank-2 form is supported), and it sits inside every projection of
/// the model — so CPU-only coverage would leave it unverified exactly where
/// it is most likely to differ.
///
/// The comparison is **pestle vs dense on the same device**, not vs the CPU
/// reference. That way a backend which cannot run qwen35 at all (e.g. Vulkan
/// currently has no `GatedDeltaNet` kernel and its unfuse path does not fire
/// for this graph — a pre-existing gap, nothing to do with Pestle) is
/// reported as unsupported rather than counted as a Pestle regression.
#[test]
fn pestle_graph_agrees_with_dense_on_every_backend() {
    let mut checked = Vec::new();
    let mut unavailable = Vec::new();
    let mut cannot_run_arch = Vec::new();

    for &dev in DEVICES {
        if !rlx_runtime::is_available(dev) {
            unavailable.push(dev);
            continue;
        }
        // If the dense graph cannot run here, the architecture is unsupported
        // on this backend independently of the factorization.
        let dense = match std::panic::catch_unwind(|| logits_on(dev, false)) {
            Ok(v) => v,
            Err(_) => {
                cannot_run_arch.push(dev);
                continue;
            }
        };
        let factored = logits_on(dev, true);
        assert_eq!(factored.len(), dense.len(), "{dev:?}: logit shape");
        let worst = dense
            .iter()
            .zip(&factored)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 5e-3,
            "Pestle projections diverge from the dense equivalent on {dev:?}: \
             max |Δlogit| = {worst}"
        );
        checked.push(dev);
    }

    eprintln!(
        "pestle == dense on {checked:?}; \
         arch unsupported (pre-existing): {cannot_run_arch:?}; \
         device unavailable here: {unavailable:?}"
    );
    assert!(
        checked.contains(&rlx_runtime::Device::Cpu),
        "CPU must always be checked"
    );
}
