//! The `LensModel` seam, driven by a *second* implementation.
//!
//! `lens_model_api.rs` drives the trait against Qwen3.5 with synthetic weights.
//! This drives it against a structurally different model — dense attention-only,
//! loaded from HF safetensors rather than GGUF — which is the only way to know
//! the interface abstracts anything. Skips itself when the checkpoint is absent.

#![cfg(feature = "qwen3")]

use rlx_jlens::models::qwen3::Qwen3LensModel;
use rlx_jlens::{FitConfig, LensModel, Readout, StackLens};
use rlx_runtime::{Device, Session};

fn model() -> Option<Qwen3LensModel> {
    let dir = match std::env::var("RLX_JLENS_QWEN3_DIR") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/Qwen3-0.6B"),
    };
    if !dir.join("config.json").exists() {
        eprintln!("no Qwen3 checkpoint at {}; skipping", dir.display());
        return None;
    }
    Qwen3LensModel::open(&dir)
        .ok()
        .map(|m| m.with_name("qwen3"))
}

#[test]
fn reports_its_own_shape() {
    let Some(m) = model() else { return };
    eprintln!(
        "{}: {} layers, d_model {}",
        m.name(),
        m.n_layers(),
        m.d_model()
    );
    assert!(m.n_layers() > 0);
    assert!(m.d_model() > 0);
    assert!(
        m.block(m.n_layers(), 1, 4).is_err(),
        "layer past the end must be rejected"
    );
}

#[test]
fn unembed_decodes_a_residual() {
    let Some(m) = model() else { return };
    let mut readout = Readout::new(&m, 1, Device::Cpu).expect("unembed");
    let logits = readout.logits(&vec![0.02f32; m.d_model()]).expect("decode");
    assert_eq!(logits.len(), readout.vocab());
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "unembed produced non-finite logits"
    );
}

/// The whole point of the trait: the estimator does not know what model this is.
#[test]
fn fits_whole_stack_jacobians() {
    let Some(m) = model() else { return };
    let (seq, dim_batch) = (8usize, 4usize);
    let target = 3;
    let layers = vec![0usize, 2];
    let cfg = FitConfig {
        dim_batch,
        skip_first: 1,
        device: Device::Cpu,
    };

    let mut lens = StackLens::new(&m, &layers, target, seq, cfg).expect("stack lens");
    let tokens: Vec<f32> = (0..seq).map(|i| (1000 + i * 13) as f32).collect();
    let batched = lens.replicate_tokens(&tokens).expect("replicate");
    let js = lens.jacobians(&batched).expect("jacobians");

    assert_eq!(js.len(), layers.len());
    let d = m.d_model();
    for (j, layer) in js.iter().zip(&layers) {
        assert_eq!(j.d_model, d);
        assert!(
            j.values.iter().all(|v| v.is_finite()),
            "layer {layer}: non-finite J"
        );
        // A residual block's Jacobian is `I` plus what the block itself did, so
        // the diagonal should sit near 1 — a J of all zeros (the failure mode
        // when a tap lands on the wrong node) would not.
        let diag: f32 = (0..d).map(|i| j.values[i * d + i]).sum::<f32>() / d as f32;
        eprintln!("layer {layer}: mean diagonal {diag:.4}");
        assert!(
            diag > 0.2,
            "layer {layer}: mean diagonal {diag:.4} — the transport looks degenerate"
        );
    }
}

/// The forward must also run, or the Jacobians above are of nothing.
#[test]
fn stack_forward_runs() {
    let Some(m) = model() else { return };
    let (seq, layers, target) = (8usize, vec![0usize, 2], 3usize);
    let stack = m.stack(&layers, target, 1, seq).expect("stack");
    let mut fwd = Session::new(Device::Cpu).compile(stack.tapped.graph().clone());
    for (name, data) in &stack.params {
        fwd.set_param(name, data);
    }
    let tokens: Vec<f32> = (0..seq).map(|i| (1000 + i * 13) as f32).collect();
    let outs = fwd.run(&[(stack.token_input.as_str(), &tokens[..])]);
    assert_eq!(
        outs.len(),
        1 + layers.len(),
        "expected h_target plus one tap per layer"
    );
    for (i, o) in outs.iter().enumerate() {
        assert_eq!(o.len(), seq * m.d_model(), "output {i} has the wrong shape");
        assert!(o.iter().all(|v| v.is_finite()), "output {i} is non-finite");
    }
}

/// What shape *is* the Qwen3 trunk graph? `--ignored`.
#[test]
#[ignore = "diagnostic"]
fn dump_trunk_shape() {
    use rlx_core::SafetensorsMmapLoader;
    let Some(m) = model() else { return };
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/Qwen3-0.6B");
    let mut loader = SafetensorsMmapLoader::open(&dir).expect("loader");
    let (g, _p) = rlx_qwen3::build_qwen3_graph_sized(m.config(), &mut loader, 1, 8, false, false)
        .expect("graph");
    eprintln!("{} nodes, outputs {:?}", g.len(), g.outputs);
    let mut hist = std::collections::BTreeMap::new();
    for n in g.nodes() {
        let d = format!("{:?}", n.op);
        *hist
            .entry(d.split(['{', '(', ' ']).next().unwrap_or("?").to_string())
            .or_insert(0) += 1;
    }
    eprintln!("ops: {hist:?}");
    eprintln!("--- last 14 nodes ---");
    for n in g
        .nodes()
        .iter()
        .rev()
        .take(14)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let d = format!("{:?}", n.op);
        eprintln!("  %{} {:.60} inputs {:?}", n.id.0, d, n.inputs);
    }
}
