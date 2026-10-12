//! LoRA dropout: a mask input on the adapter branch only.
//!
//! Dropout is emitted as `x * mask` rather than a new op, so it needs no
//! backend support and its backward is `mul`'s. The risks that buys are all
//! about *where* the mask lands and what an unfed input does — which is what
//! these check.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_tune::adapter::{LoraInit, LoraSpec};
use rlx_tune::inject::{AdapterParam, FuseMode, inject_lora_dropout, sample_dropout_mask};

fn spec(dropout: f32) -> LoraSpec {
    LoraSpec {
        rank: 2,
        alpha: 2.0,
        dropout,
        target_modules: vec!["w".to_string()],
        init: LoraInit::Gaussian,
    }
}

/// `y = x·W` with `W` a param named `w`, so the spec targets it.
fn base_graph(m: usize, k: usize, n: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("t");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.param("w", Shape::new(&[k, n], f));
    let y = g.matmul(x, w, Shape::new(&[m, n], f));
    g.set_outputs(vec![y]);
    g
}

fn input_names(g: &Graph) -> Vec<String> {
    g.nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn no_dropout_declares_no_mask() {
    let (g, params, masks) =
        inject_lora_dropout(&base_graph(2, 3, 4), &spec(0.0), FuseMode::Unfused);
    assert_eq!(masks.len(), 0, "p = 0 must not add an input");
    assert_eq!(params.len(), 2, "A and B");
    assert!(!input_names(&g).iter().any(|n| n.contains("dropout")));
}

#[test]
fn dropout_declares_one_mask_per_target_shaped_like_its_input() {
    let (g, _p, masks) = inject_lora_dropout(&base_graph(5, 3, 4), &spec(0.05), FuseMode::Unfused);
    assert_eq!(masks.len(), 1, "one target, one mask");
    assert_eq!(masks[0].name, "lora.w.dropout");
    assert_eq!(masks[0].shape, vec![5, 3], "mask covers the LoRA input `x`");
    assert!(input_names(&g).contains(&"lora.w.dropout".to_string()));
}

/// The mask must touch the adapter branch only. If it multiplied the frozen
/// base path too, dropout would corrupt the pretrained model's output rather
/// than regularize the adapter.
#[test]
fn the_mask_leaves_the_frozen_base_path_alone() {
    let (g, params, masks) =
        inject_lora_dropout(&base_graph(1, 2, 2), &spec(0.5), FuseMode::Unfused);
    let mut c = Session::new(Device::Cpu).compile(g);
    // B = 0 => the adapter contributes nothing, whatever the mask is.
    for p in &params {
        let n: usize = p.shape.iter().product();
        let v = if p.name.ends_with(".b") {
            vec![0.0f32; n]
        } else {
            vec![0.5f32; n]
        };
        c.set_param(&p.name, &v);
    }
    c.set_param("w", &[1.0, 2.0, 3.0, 4.0]);
    let x = [1.0f32, 1.0];

    let all_ones = vec![1.0f32; 2];
    let all_zero = vec![0.0f32; 2];
    let a = c.run(&[("x", &x[..]), (masks[0].name.as_str(), &all_ones)])[0].clone();
    let b = c.run(&[("x", &x[..]), (masks[0].name.as_str(), &all_zero)])[0].clone();
    assert_eq!(a, b, "with B = 0 the mask must not change the output");
    assert_eq!(a, vec![4.0, 6.0], "and that output is the bare base matmul");
}

/// A zero mask must kill the adapter delta exactly; a ones mask must
/// reproduce the no-dropout forward.
#[test]
fn the_mask_gates_the_adapter_delta() {
    let g0 = base_graph(1, 2, 2);
    let (gd, params, masks) = inject_lora_dropout(&g0, &spec(0.5), FuseMode::Unfused);
    let (gn, params_n, _) = inject_lora_dropout(&g0, &spec(0.0), FuseMode::Unfused);

    fn setup(c: &mut rlx_runtime::CompiledGraph, ps: &[AdapterParam]) {
        for p in ps {
            let n: usize = p.shape.iter().product();
            c.set_param(&p.name, &vec![0.5f32; n]);
        }
        c.set_param("w", &[1.0, 2.0, 3.0, 4.0]);
    }
    let mut cd = Session::new(Device::Cpu).compile(gd);
    setup(&mut cd, &params);
    let mut cn = Session::new(Device::Cpu).compile(gn);
    setup(&mut cn, &params_n);

    let x = [1.0f32, 1.0];
    let base_only = vec![4.0f32, 6.0];
    let zero_mask = [0.0f32, 0.0];
    let ones_mask = [1.0f32, 1.0];
    let with_zero = cd.run(&[("x", &x[..]), (masks[0].name.as_str(), &zero_mask)])[0].clone();
    assert_eq!(
        with_zero, base_only,
        "a zero mask must remove the adapter delta entirely"
    );

    let with_ones = cd.run(&[("x", &x[..]), (masks[0].name.as_str(), &ones_mask)])[0].clone();
    let without = cn.run(&[("x", &x[..])])[0].clone();
    assert_eq!(
        with_ones, without,
        "an all-ones mask must match the undropped graph exactly"
    );
    assert_ne!(with_ones, base_only, "the adapter must actually contribute");
}

/// Inverted dropout: expectation preserved, so inference needs no rescaling.
#[test]
fn sampled_masks_are_inverted_and_unbiased() {
    assert_eq!(sample_dropout_mask(8, 0.0, &mut || 0.0), vec![1.0; 8]);

    // Deterministic sweep over [0,1) so the rate is exact, not statistical.
    let n = 10_000usize;
    let mut i = 0usize;
    let m = sample_dropout_mask(n, 0.25, &mut || {
        let v = i as f32 / n as f32;
        i += 1;
        v
    });
    let kept = m.iter().filter(|x| **x != 0.0).count();
    assert_eq!(kept, 7500, "keep rate must be 1 - p");
    for v in m.iter().filter(|x| **x != 0.0) {
        assert!(
            (v - 1.0 / 0.75).abs() < 1e-6,
            "kept entries scale by 1/(1-p)"
        );
    }
    let mean: f32 = m.iter().sum::<f32>() / n as f32;
    assert!(
        (mean - 1.0).abs() < 1e-3,
        "expectation must be 1, got {mean}"
    );
}

/// `FuseMode::Fused` has nowhere to put the mask, so it must fall back to the
/// explicit form rather than silently dropping the dropout.
#[test]
fn fused_mode_falls_back_instead_of_ignoring_dropout() {
    let (g, _p, masks) = inject_lora_dropout(&base_graph(2, 3, 4), &spec(0.1), FuseMode::Fused);
    assert_eq!(masks.len(), 1, "the mask must still exist under Fused");
    assert!(
        !g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::LoraMatMul { .. })),
        "the fused op cannot carry a mask; it must decompose"
    );
    // Without dropout the fused op is still used.
    let (g2, _, m2) = inject_lora_dropout(&base_graph(2, 3, 4), &spec(0.0), FuseMode::Fused);
    assert!(m2.is_empty());
    assert!(
        g2.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::LoraMatMul { .. })),
        "p = 0 must keep the fused fast path"
    );
}
