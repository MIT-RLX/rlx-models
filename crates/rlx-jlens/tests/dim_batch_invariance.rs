//! `dim_batch` is a scheduling knob, not a modelling one.
//!
//! It only decides how many rows of `J` one backward pass fills in, so every
//! setting must produce the same matrix. CPU is the arbiter, and the bound is
//! `CPU_INVARIANCE_TOL` — tight enough that the bug this was written for (a
//! 0.13 relative difference) misses it by five orders of magnitude, but not
//! bit-exact. It *is* bit-exact on aarch64; on x86_64 changing `dim_batch`
//! changes the GEMM's M, BLAS picks a different tiling, and the summation order
//! moves with it. That is reassociation, not a dependence on `dim_batch`.
//!
//! Metal is also checked, and also bit-exactly: it *was* nondeterministic
//! run-to-run, which turned out to be missing barriers in several Metal
//! threadgroup reductions rather than anything about the lens — see
//! `metal_reproducibility_at_the_default_dim_batch`.

#![cfg(feature = "qwen35")]

use rlx_jlens::models::qwen35::Qwen35LensModel;
use rlx_jlens::{BlockLens, FitConfig, LensModel, StackLens};
use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_runtime::Device;

fn jacobian_of(
    model: &Qwen35LensModel,
    device: Device,
    layer: usize,
    seq: usize,
    dim_batch: usize,
) -> Vec<f32> {
    let mut lens = BlockLens::new(
        model,
        layer,
        seq,
        FitConfig {
            dim_batch,
            skip_first: 2,
            device,
        },
    )
    .expect("lens");
    let d = model.d_model();
    // A deterministic, non-degenerate residual stream, replicated across the
    // batch exactly as the estimator requires — every batch element carries the
    // same activation and differs only in which cotangent it seeds.
    let one: Vec<f32> = (0..seq * d)
        .map(|i| ((i as f32 * 0.7).sin() + (i as f32 * 0.013).cos()) * 0.5)
        .collect();
    let residual: Vec<f32> = one.repeat(dim_batch);
    lens.jacobian(&residual).expect("jacobian").values
}

fn rel_frobenius(a: &[f32], b: &[f32]) -> (f32, f32) {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = a.iter().map(|x| x * x).sum();
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    ((num / den).sqrt(), worst)
}

#[allow(clippy::too_many_arguments)]
fn compare_reported(
    model: &Qwen35LensModel,
    device: Device,
    layer: usize,
    seq: usize,
    small: usize,
    large: usize,
    label: &str,
) -> f32 {
    let a = jacobian_of(model, device, layer, seq, small);
    let b = jacobian_of(model, device, layer, seq, large);
    let (rel, worst) = rel_frobenius(&a, &b);
    eprintln!("{label} layer {layer}: {small} vs {large} -> relF = {rel:.3e}  worst = {worst:.3e}");
    rel
}

/// Bit-exact on aarch64; x86_64 BLAS reassociates with the GEMM shape, which
/// `dim_batch` changes. Five orders of magnitude below anything that would
/// indicate a real `dim_batch` dependence.
const CPU_INVARIANCE_TOL: f32 = 1e-5;

fn assert_invariant(model: &Qwen35LensModel, layer: usize, seq: usize, small: usize, large: usize) {
    let rel = compare_reported(model, Device::Cpu, layer, seq, small, large, "cpu");
    assert!(
        rel < CPU_INVARIANCE_TOL,
        "layer {layer}: J depends on dim_batch on CPU (relF {rel:.3e})"
    );
}

#[test]
fn cpu_is_dim_batch_invariant() {
    let model = Qwen35LensModel::new(tiny_cfg(), synth_weights(&tiny_cfg()));
    for layer in 0..model.n_layers() {
        assert_invariant(&model, layer, 8, 4, 16);
    }
}

/// The synthetic config is 16-wide with one head; the real one is 1024-wide with
/// 16 heads and a 128-wide delta-net state, which is where per-`(batch, head)`
/// scratch sizing and dispatch grids can go wrong. Skips when absent.
fn real_model() -> Option<(Qwen35LensModel, std::path::PathBuf)> {
    use rlx_core::weight_loader::GgufLoader;
    let path: std::path::PathBuf = match std::env::var("RLX_JLENS_QWEN35_GGUF") {
        Ok(p) => p.into(),
        Err(_) => {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/Qwen3.5-0.8B-gguf");
            let mut found: Vec<_> = std::fs::read_dir(&dir)
                .ok()?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
                .collect();
            found.sort();
            found.pop()?
        }
    };
    let mut loader = GgufLoader::from_file(path.to_str()?).ok()?;
    let cfg = rlx_qwen35::Qwen35Config::from_gguf(loader.file()).ok()?;
    let weights = rlx_qwen35::Qwen35Weights::from_loader(&mut loader, &cfg).ok()?;
    Some((Qwen35LensModel::new(cfg, weights), path))
}

#[test]
fn cpu_real_weights_are_dim_batch_invariant() {
    let Some((model, _)) = real_model() else {
        eprintln!("no Qwen3.5 checkpoint; skipping");
        return;
    };
    // Layer 0 is a gated-delta-net block, layer 3 a full-attention one.
    for layer in [0usize, 3] {
        assert_invariant(&model, layer, 24, 16, 64);
    }
}

/// The whole-stack transport.
///
/// `StackLens` composes every block from the source layer to the target, so it
/// *multiplies* whatever each block's error is — this is the test that would
/// catch a `dim_batch` bug too small to see on a single block.
fn stack_jacobians(model: &Qwen35LensModel, device: Device, dim_batch: usize) -> Vec<f32> {
    let (seq, target) = (24usize, 2usize);
    let layers: Vec<usize> = (0..=target).collect();
    let mut lens = StackLens::new(
        model,
        &layers,
        target,
        seq,
        FitConfig {
            dim_batch,
            skip_first: 2,
            device,
        },
    )
    .expect("stack lens");
    let tokens: Vec<f32> = (0..seq).map(|i| (1000 + i * 37) as f32).collect();
    let batched = lens.replicate_tokens(&tokens).expect("replicate");
    let js = lens.jacobians(&batched).expect("jacobians");
    js.iter().flat_map(|j| j.values.clone()).collect()
}

#[test]
fn cpu_stack_is_dim_batch_invariant() {
    let Some((model, _)) = real_model() else {
        eprintln!("no checkpoint; skipping");
        return;
    };
    let a = stack_jacobians(&model, Device::Cpu, 16);
    let b = stack_jacobians(&model, Device::Cpu, 64);
    let (rel, _) = rel_frobenius(&a, &b);
    eprintln!("cpu stack: dim_batch 16 vs 64 -> relF = {rel:.3e}");
    assert!(
        rel < CPU_INVARIANCE_TOL,
        "stack J depends on dim_batch on CPU (relF {rel:.3e})"
    );
}

/// Metal must reproduce a fit exactly.
///
/// It did not, until the cause was found: several Metal kernels ran a
/// threadgroup reduction, had every thread read `partial[0]`, then reused
/// `partial` for a second reduction *without a barrier between*. A threadgroup
/// spans several SIMD groups, which diverge freely, so one group could clobber
/// slot 0 while another was still reading it. `rms_norm_bwd` and
/// `softmax_lastax_causal` are the two on this path; the bug is intermittent, so
/// a fit was reproducible to ~1e-3 rather than exactly.
///
/// This now asserts **bit-exact**, which is the whole point — anything looser
/// would have passed before the fix.
#[cfg(feature = "metal")]
#[test]
fn metal_reproducibility_at_the_default_dim_batch() {
    let Some((model, _)) = real_model() else {
        eprintln!("no checkpoint; skipping");
        return;
    };
    for (layer, what) in [(0usize, "delta-net"), (3, "attention")] {
        let a = jacobian_of(&model, Device::Metal, layer, 24, 8);
        let b = jacobian_of(&model, Device::Metal, layer, 24, 8);
        let (rel, worst) = rel_frobenius(&a, &b);
        eprintln!("metal {what}: run vs run -> relF = {rel:.3e}  worst = {worst:.3e}");
        assert!(
            worst == 0.0,
            "metal {what} block is not bit-reproducible (relF {rel:.3e}, worst {worst:.3e}) \
             — check for a threadgroup reduction reusing its scratch without a barrier"
        );
    }
}

/// How Metal's nondeterminism scales with `dim_batch`. `--ignored`: it is slow,
/// and it is the evidence behind the default rather than a pass/fail property.
#[cfg(feature = "metal")]
#[test]
#[ignore = "diagnostic sweep"]
fn metal_noise_grows_with_dim_batch() {
    let Some((model, _)) = real_model() else {
        return;
    };
    let per_elem = 16 * ((48 + 2) * 128 * 128 + 48 * 128) * 4;
    for b in [16usize, 32, 48, 64] {
        eprintln!("  scratch {:.2} GB", (b * per_elem) as f64 / 1e9);
        compare_reported(&model, Device::Metal, 0, 48, b, b, "metal run-vs-run");
    }
}

/// Where does Metal's nondeterminism actually come from? `--ignored`.
///
/// Run under different backend flags to bisect it — it is absent at seq 24 and
/// present at seq 48, so it tracks a size threshold somewhere in the stack.
#[cfg(feature = "metal")]
#[test]
#[ignore = "diagnostic"]
fn metal_nondeterminism_source() {
    let Some((model, _)) = real_model() else {
        return;
    };
    for seq in [24usize, 48] {
        for layer in [0usize, 3] {
            let a = jacobian_of(&model, Device::Metal, layer, seq, 8);
            let b = jacobian_of(&model, Device::Metal, layer, seq, 8);
            let (rel, _) = rel_frobenius(&a, &b);
            eprintln!("seq {seq} layer {layer}: run-vs-run relF = {rel:.3e}");
        }
    }
}

/// Forward or backward? `--ignored`.
#[cfg(feature = "metal")]
#[test]
#[ignore = "diagnostic"]
fn metal_forward_determinism() {
    use rlx_runtime::Session;
    let Some((model, _)) = real_model() else {
        return;
    };
    for layer in [0usize, 3] {
        let seq = 48;
        let blk = model.block(layer, 8, seq).expect("block");
        let run = || -> Vec<f32> {
            let mut g = Session::new(Device::Metal).compile(blk.graph.clone());
            for (name, data) in &blk.params {
                g.set_param(name, data);
            }
            let x: Vec<f32> = (0..8 * seq * model.d_model())
                .map(|i| ((i as f32 * 0.7).sin() + (i as f32 * 0.013).cos()) * 0.5)
                .collect();
            g.run(&[(blk.residual_input.as_str(), &x[..])])[0].clone()
        };
        let (a, b) = (run(), run());
        let (rel, worst) = rel_frobenius(&a, &b);
        eprintln!("layer {layer} FORWARD only: relF = {rel:.3e}  worst = {worst:.3e}");
    }
}

/// Is the *graph* the same twice? `--ignored`.
///
/// If autodiff collects a node's gradient contributions through a `HashMap`,
/// the order they are summed in varies per map instance, so two runs in one
/// process compile structurally different backward graphs and f32 rounding
/// differs — which looks exactly like a nondeterministic kernel.
#[cfg(feature = "metal")]
#[test]
#[ignore = "diagnostic"]
fn backward_graph_is_stable_across_builds() {
    use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
    let Some((model, _)) = real_model() else {
        return;
    };
    let sig = |layer: usize| -> Vec<String> {
        let blk = model.block(layer, 8, 24).expect("block");
        let out = blk.graph.outputs[0];
        let bwd = grad_with_loss_wrt(&blk.graph, &[Wrt::Node(out)], GradWithLossOptions::TRAINING);
        bwd.nodes()
            .iter()
            .map(|n| format!("{:?}|{:?}", n.op, n.inputs))
            .collect()
    };
    for layer in [0usize, 3] {
        let (a, b) = (sig(layer), sig(layer));
        let same = a == b;
        let first_diff = a.iter().zip(&b).position(|(x, y)| x != y);
        eprintln!(
            "layer {layer}: {} nodes vs {} nodes, identical = {same}, first difference at {:?}",
            a.len(),
            b.len(),
            first_diff
        );
        if let Some(i) = first_diff {
            eprintln!("   run A: {}", a[i]);
            eprintln!("   run B: {}", b[i]);
        }
    }
}

/// Runtime race, or compile-stage nondeterminism? `--ignored`.
///
/// Every earlier probe rebuilt and recompiled between runs, so it could not tell
/// "the same kernels produced different numbers" from "we compiled a different
/// program". This runs one *already compiled* graph twice, then compiles the
/// same graph twice and runs each once.
#[cfg(feature = "metal")]
#[test]
#[ignore = "diagnostic"]
fn metal_race_or_compile() {
    use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
    use rlx_runtime::Session;

    let Some((model, _)) = real_model() else {
        return;
    };
    for layer in [0usize, 3] {
        let (batch, seq) = (8usize, 48usize);
        let blk = model.block(layer, batch, seq).expect("block");
        let out = blk.graph.outputs[0];
        let bwd = grad_with_loss_wrt(&blk.graph, &[Wrt::Node(out)], GradWithLossOptions::TRAINING);

        let d = model.d_model();
        let x: Vec<f32> = (0..batch * seq * d)
            .map(|i| ((i as f32 * 0.7).sin() + (i as f32 * 0.013).cos()) * 0.5)
            .collect();
        let dy: Vec<f32> = (0..batch * seq * d)
            .map(|i| ((i as f32 * 0.011).cos()) * 0.25)
            .collect();

        let compile = || {
            let mut g = Session::new(Device::Metal).compile(bwd.clone());
            for (name, data) in &blk.params {
                g.set_param(name, data);
            }
            g
        };
        let run = |g: &mut rlx_runtime::CompiledGraph| -> Vec<f32> {
            g.run(&[(blk.residual_input.as_str(), &x[..]), ("d_output", &dy[..])])
                .concat()
        };

        // (a) one compiled graph, run twice.
        let mut g = compile();
        let (a1, a2) = (run(&mut g), run(&mut g));
        let (rel_run, worst_run) = rel_frobenius(&a1, &a2);

        // (b) two compiles of the identical graph, one run each.
        let (mut g1, mut g2) = (compile(), compile());
        let (b1, b2) = (run(&mut g1), run(&mut g2));
        let (rel_comp, worst_comp) = rel_frobenius(&b1, &b2);

        eprintln!(
            "layer {layer}: same compiled graph twice -> relF {rel_run:.3e} (worst {worst_run:.3e}) | \
             two compiles -> relF {rel_comp:.3e} (worst {worst_comp:.3e})"
        );
    }
}

/// Construction, or the per-pass loop? `--ignored`.
#[cfg(feature = "metal")]
#[test]
#[ignore = "diagnostic"]
fn metal_lens_construction_or_loop() {
    let Some((model, _)) = real_model() else {
        return;
    };
    for layer in [0usize, 3] {
        let (seq, dim_batch) = (48usize, 8usize);
        let d = model.d_model();
        let one: Vec<f32> = (0..seq * d)
            .map(|i| ((i as f32 * 0.7).sin() + (i as f32 * 0.013).cos()) * 0.5)
            .collect();
        let residual: Vec<f32> = one.repeat(dim_batch);

        // One BlockLens, two fits: isolates the per-pass loop from construction.
        let mut lens = BlockLens::new(
            &model,
            layer,
            seq,
            FitConfig {
                dim_batch,
                skip_first: 2,
                device: Device::Metal,
            },
        )
        .expect("lens");
        // Four fits: if only the first differs it is warm-up / stale arena on
        // the first run; if every pair differs it is ongoing nondeterminism.
        let runs: Vec<Vec<f32>> = (0..4)
            .map(|_| lens.jacobian(&residual).expect("j").values)
            .collect();
        for i in 1..runs.len() {
            let (rel, worst) = rel_frobenius(&runs[0], &runs[i]);
            eprintln!("layer {layer}: fit 0 vs fit {i} -> relF {rel:.3e} (worst {worst:.3e})");
        }
        let (rel, worst) = rel_frobenius(&runs[1], &runs[2]);
        eprintln!("layer {layer}: fit 1 vs fit 2 -> relF {rel:.3e} (worst {worst:.3e})");
    }
}

/// Which *output* of the replay half diverges, and from which op? `--ignored`.
///
/// The divergence is intermittent, so this retries until it sees one rather
/// than reporting a clean run as evidence of nothing.
#[cfg(feature = "metal")]
#[test]
#[ignore = "diagnostic"]
fn metal_replay_divergent_output() {
    use rlx_autodiff::split_vjp;
    use rlx_jlens::{Tap, TappedGraph};
    use rlx_runtime::Session;

    let Some((model, _)) = real_model() else {
        return;
    };
    for layer in [0usize, 3] {
        let (batch, seq) = (8usize, 48usize);
        let d = model.d_model();
        let block = model.block(layer, batch, seq).expect("block");
        let residual_input = block.residual_input.clone();
        let tapped = TappedGraph::new(
            block.graph.clone(),
            vec![Tap::at_input(layer, residual_input.clone())],
        )
        .expect("tap");
        let split = split_vjp(&tapped.vjp_graph()).expect("split");

        let mut save = {
            let mut c = Session::new(Device::Metal).compile(split.save.clone());
            for (name, data) in &block.params {
                c.set_param(name, data);
            }
            c
        };
        let mut replay = {
            let mut c = Session::new(Device::Metal).compile(split.replay.clone());
            for (name, data) in &block.params {
                c.set_param(name, data);
            }
            c
        };
        let x: Vec<f32> = (0..batch * seq * d)
            .map(|i| ((i as f32 * 0.7).sin() + (i as f32 * 0.013).cos()) * 0.5)
            .collect();
        let saved = save.run(&[(residual_input.as_str(), &x[..])]);
        for sv in &split.saved {
            replay.set_param(&sv.name, &saved[sv.save_output]);
        }
        let cot: Vec<f32> = (0..batch * seq * d)
            .map(|i| if i % 977 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut feed: Vec<(&str, &[f32])> = vec![("d_output", &cot[..])];
        if split.replay.input_id(&residual_input).is_some() {
            feed.push((residual_input.as_str(), &x[..]));
        }

        let base = replay.run(&feed);
        let mut reported = false;
        for attempt in 0..12 {
            let now = replay.run(&feed);
            let mut any = false;
            for (i, (a, b)) in base.iter().zip(&now).enumerate() {
                let (rel, worst) = rel_frobenius(a, b);
                if worst > 0.0 {
                    let node = split.replay.outputs[i];
                    let op = format!("{:?}", split.replay.node(node).op);
                    eprintln!(
                        "layer {layer} attempt {attempt}: output {i} DIFFERS relF {rel:.3e} \
                         worst {worst:.3e} <- %{} {:.70}",
                        node.0, op
                    );
                    any = true;
                    reported = true;
                }
            }
            if any {
                break;
            }
        }
        if !reported {
            eprintln!("layer {layer}: 12 attempts, no divergence seen");
        }
    }
}

/// The first node in the replay half whose value is not reproducible. `--ignored`.
///
/// Publishes every node as a graph output, runs twice, and reports the earliest
/// divergence in topological order — everything after it is downstream fallout.
#[cfg(feature = "metal")]
#[test]
#[ignore = "diagnostic"]
fn metal_first_divergent_node() {
    use rlx_autodiff::split_vjp;
    use rlx_jlens::{Tap, TappedGraph};
    use rlx_runtime::Session;

    let Some((model, _)) = real_model() else {
        return;
    };
    for layer in [3usize, 0] {
        let (batch, seq) = (8usize, 48usize);
        let d = model.d_model();
        let block = model.block(layer, batch, seq).expect("block");
        let residual_input = block.residual_input.clone();
        let tapped = TappedGraph::new(
            block.graph.clone(),
            vec![Tap::at_input(layer, residual_input.clone())],
        )
        .expect("tap");
        let split = split_vjp(&tapped.vjp_graph()).expect("split");

        let mut save = {
            let mut c = Session::new(Device::Metal).compile(split.save.clone());
            for (name, data) in &block.params {
                c.set_param(name, data);
            }
            c
        };
        let x: Vec<f32> = (0..batch * seq * d)
            .map(|i| ((i as f32 * 0.7).sin() + (i as f32 * 0.013).cos()) * 0.5)
            .collect();
        let saved = save.run(&[(residual_input.as_str(), &x[..])]);

        // Every node becomes an output, so nothing can be optimized away and
        // each intermediate is readable.
        let mut probe = split.replay.clone();
        let ids: Vec<rlx_ir::NodeId> = probe.nodes().iter().map(|n| n.id).collect();
        probe.set_outputs(ids.clone());

        let mut replay = {
            let mut c = Session::new(Device::Metal).compile(probe.clone());
            for (name, data) in &block.params {
                c.set_param(name, data);
            }
            c
        };
        for sv in &split.saved {
            replay.set_param(&sv.name, &saved[sv.save_output]);
        }
        let cot: Vec<f32> = (0..batch * seq * d)
            .map(|i| if i % 977 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut feed: Vec<(&str, &[f32])> = vec![("d_output", &cot[..])];
        if probe.input_id(&residual_input).is_some() {
            feed.push((residual_input.as_str(), &x[..]));
        }

        let base = replay.run(&feed);
        let mut found = false;
        for attempt in 0..10 {
            let now = replay.run(&feed);
            for (i, (a, b)) in base.iter().zip(&now).enumerate() {
                if a.iter().zip(b).any(|(p, q)| p != q) {
                    let id = ids[i];
                    let n = split.replay.node(id);
                    let (rel, worst) = rel_frobenius(a, b);
                    eprintln!(
                        "layer {layer} attempt {attempt}: FIRST divergence at %{} {:.60}",
                        id.0,
                        format!("{:?}", n.op)
                    );
                    eprintln!(
                        "    relF {rel:.3e} worst {worst:.3e}, shape {:?}",
                        n.shape.dims()
                    );
                    for &inp in &n.inputs {
                        eprintln!(
                            "    <- %{} {:.60}",
                            inp.0,
                            format!("{:?}", split.replay.node(inp).op)
                        );
                    }
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        if !found {
            eprintln!("layer {layer}: 10 attempts, fully reproducible");
        }
    }
}

/// Can MLX run these graphs at all? `--ignored`.
///
/// It could not when the lens was written — the qwen35 layer-probe *forward*
/// failed before any autodiff was involved. Several fusion bugs have been fixed
/// since, so this re-checks forward, then backward, then a full fit, and reports
/// the first thing that breaks rather than one opaque failure.
#[cfg(feature = "mlx")]
#[test]
#[ignore = "diagnostic"]
fn mlx_status() {
    use rlx_runtime::Session;
    let model = Qwen35LensModel::new(tiny_cfg(), synth_weights(&tiny_cfg()));
    let (seq, batch) = (8usize, 4usize);
    let d = model.d_model();
    for layer in 0..model.n_layers() {
        let blk = match model.block(layer, batch, seq) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("layer {layer}: block graph FAILED: {e}");
                continue;
            }
        };
        let x: Vec<f32> = (0..batch * seq * d)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let fwd = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g = Session::new(Device::Mlx).compile(blk.graph.clone());
            for (name, data) in &blk.params {
                g.set_param(name, data);
            }
            g.run(&[(blk.residual_input.as_str(), &x[..])])
        }));
        match fwd {
            Ok(o) => eprintln!(
                "layer {layer}: forward OK ({} outputs, finite = {})",
                o.len(),
                o[0].iter().all(|v| v.is_finite())
            ),
            Err(_) => {
                eprintln!("layer {layer}: forward PANICKED");
                continue;
            }
        }
        let fit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut lens = BlockLens::new(
                &model,
                layer,
                seq,
                FitConfig {
                    dim_batch: batch,
                    skip_first: 1,
                    device: Device::Mlx,
                },
            )
            .expect("lens");
            let one: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.01).cos()).collect();
            lens.jacobian(&one.repeat(batch)).expect("jacobian").values
        }));
        match fit {
            Ok(j) => {
                let diag: f32 = (0..d).map(|i| j[i * d + i]).sum::<f32>() / d as f32;
                eprintln!("layer {layer}: BACKWARD OK, mean diagonal {diag:.4}");
            }
            Err(_) => eprintln!("layer {layer}: backward PANICKED"),
        }
    }
}

/// CPU is the arbiter, and every accelerator must match it.
///
/// Self-consistency is a different claim from correctness: a backend can repeat
/// itself perfectly and still be wrong. CPU is what the Python reference was
/// checked against (`tests/reference_parity.rs`), so agreement with CPU is the
/// property worth asserting. Two real MLX bugs surfaced exactly here — a rank
/// reconciliation gap in `Op::Transpose` and a partial-RoPE gradient that
/// ignored head structure — which is why this is a test and not a remark.
#[allow(dead_code)]
fn assert_matches_cpu(device: Device, label: &str) {
    // CUDA and ROCm lack the fused `Op::GatedDeltaNetBackward` and differentiate
    // the unrolled scan instead. That is the same mathematics but a different
    // rounding path, so comparing a fused CPU fit against an unfused GPU one
    // measures the decomposition, not the backend. Pin both sides to the same
    // path — the library honours an explicit setting and leaves it alone.
    if !rlx_jlens::gdn_backward_is_fused_on(device) {
        // SAFETY: single-threaded test setup, before any graph is built.
        unsafe { std::env::set_var("RLX_GDN_UNFUSE_FOR_AD", "1") };
        eprintln!("{label} has no fused delta-net backward; pinning CPU to the same unfused path");
    }
    let model = Qwen35LensModel::new(tiny_cfg(), synth_weights(&tiny_cfg()));
    for layer in 0..model.n_layers() {
        let cpu = jacobian_of(&model, Device::Cpu, layer, 8, 4);
        let gpu = jacobian_of(&model, device, layer, 8, 4);
        let (rel, worst) = rel_frobenius(&cpu, &gpu);
        eprintln!("layer {layer}: {label} vs CPU relF = {rel:.3e}  worst = {worst:.3e}");
        assert!(
            rel < 1e-5,
            "layer {layer}: {label} disagrees with CPU (relF {rel:.3e}, worst {worst:.3e})"
        );
    }

    // The tiny config is 16-wide with one head. Real weights are 1024-wide with
    // 16 heads and a 128-wide delta-net state — the shapes where head packing
    // and rank reconciliation actually bite.
    let Some((real, _)) = real_model() else {
        eprintln!("no Qwen3.5 checkpoint; skipping the real-weight half");
        return;
    };
    for layer in [0usize, 3] {
        // Qwen3.5's attention is head_dim 256, and the CUDA/ROCm backward
        // kernels tile head-dim accumulators at 128. They now refuse it
        // explicitly (they used to return silent zeros), so there is nothing to
        // compare — skip with the reason rather than assert into a panic.
        if layer == 3 && matches!(device, Device::Cuda | Device::Rocm) {
            eprintln!(
                "real layer {layer}: skipped on {label} — AttentionBackward head_dim 256 \
                 exceeds the kernel's supported 128 (see the guard in its compile.rs)"
            );
            continue;
        }
        let cpu = jacobian_of(&real, Device::Cpu, layer, 24, 8);
        let gpu = jacobian_of(&real, device, layer, 24, 8);
        let (rel, worst) = rel_frobenius(&cpu, &gpu);
        eprintln!("real layer {layer}: {label} vs CPU relF = {rel:.3e}  worst = {worst:.3e}");
        assert!(
            rel < 1e-5,
            "real layer {layer}: {label} disagrees with CPU (relF {rel:.3e}, worst {worst:.3e})"
        );
    }
}

#[cfg(feature = "mlx")]
#[test]
fn mlx_matches_cpu() {
    assert_matches_cpu(Device::Mlx, "MLX");
}

#[cfg(feature = "metal")]
#[test]
fn metal_matches_cpu() {
    assert_matches_cpu(Device::Metal, "Metal");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_cpu() {
    assert_matches_cpu(Device::Cuda, "CUDA");
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_matches_cpu() {
    assert_matches_cpu(Device::Rocm, "ROCm");
}

/// First node where `RLX_JLENS_DIAG_DEVICE` disagrees with CPU. `--ignored`.
///
/// Device and layer come from the environment so one binary serves every remote:
///   RLX_JLENS_DIAG_DEVICE=cuda RLX_JLENS_DIAG_LAYER=0 cargo test ... -- --ignored
#[test]
#[ignore = "diagnostic"]
fn first_divergent_node_vs_cpu() {
    use rlx_autodiff::split_vjp;
    use rlx_jlens::{Tap, TappedGraph};
    use rlx_runtime::Session;

    let Ok(dev_name) = std::env::var("RLX_JLENS_DIAG_DEVICE") else {
        eprintln!("set RLX_JLENS_DIAG_DEVICE (cuda|rocm|metal|mlx)");
        return;
    };
    let device = rlx_runtime::parse_device(&dev_name).expect("device");
    let layer: usize = std::env::var("RLX_JLENS_DIAG_LAYER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let Some((model, _)) = real_model() else {
        eprintln!("no checkpoint; skipping");
        return;
    };
    if !rlx_jlens::gdn_backward_is_fused_on(device) {
        unsafe { std::env::set_var("RLX_GDN_UNFUSE_FOR_AD", "1") };
    }

    let (batch, seq) = (8usize, 24usize);
    let d = model.d_model();
    let block = model.block(layer, batch, seq).expect("block");
    let residual_input = block.residual_input.clone();
    let tapped = TappedGraph::new(
        block.graph.clone(),
        vec![Tap::at_input(layer, residual_input.clone())],
    )
    .expect("tap");
    let split = split_vjp(&tapped.vjp_graph()).expect("split");

    let x: Vec<f32> = (0..batch * seq * d)
        .map(|i| ((i as f32 * 0.7).sin() + (i as f32 * 0.013).cos()) * 0.5)
        .collect();
    let cot: Vec<f32> = (0..batch * seq * d)
        .map(|i| if i % 977 == 0 { 1.0 } else { 0.0 })
        .collect();

    // The save half first: its outputs are bound into the replay as
    // `__rlx_vjp_saved.*`, so a forward difference shows up there as a
    // "divergent Param" and looks misleadingly like a backward bug.
    let mut save_probe = split.save.clone();
    let save_ids: Vec<rlx_ir::NodeId> = save_probe.nodes().iter().map(|n| n.id).collect();
    save_probe.set_outputs(save_ids.clone());
    let run_save = |dev: Device| -> Vec<Vec<f32>> {
        let mut g = Session::new(dev).compile(save_probe.clone());
        for (name, data) in &block.params {
            g.set_param(name, data);
        }
        g.run(&[(residual_input.as_str(), &x[..])])
    };
    let (save_cpu, save_gpu) = (run_save(Device::Cpu), run_save(device));
    let mut save_bad = 0;
    for (i, (a, b)) in save_cpu.iter().zip(&save_gpu).enumerate() {
        if a.len() != b.len() {
            continue;
        }
        let (rel, worst) = rel_frobenius(a, b);
        if rel > 1e-5 {
            let id = save_ids[i];
            let n = split.save.node(id);
            eprintln!(
                "SAVE-HALF DIVERGENCE #{save_bad} at %{} {:.70}\n   relF {rel:.3e} worst {worst:.3e} shape {:?}",
                id.0,
                format!("{:?}", n.op),
                n.shape.dims()
            );
            for &inp in &n.inputs {
                eprintln!(
                    "   <- %{} {:.62}",
                    inp.0,
                    format!("{:?}", split.save.node(inp).op)
                );
            }
            save_bad += 1;
            if save_bad >= 3 {
                break;
            }
        }
    }
    if save_bad == 0 {
        eprintln!("save half agrees with CPU to 1e-5");
    }

    let mut probe = split.replay.clone();
    let ids: Vec<rlx_ir::NodeId> = probe.nodes().iter().map(|n| n.id).collect();
    probe.set_outputs(ids.clone());

    let run = |dev: Device| -> Vec<Vec<f32>> {
        let mut save = Session::new(dev).compile(split.save.clone());
        for (name, data) in &block.params {
            save.set_param(name, data);
        }
        let saved = save.run(&[(residual_input.as_str(), &x[..])]);
        let mut rep = Session::new(dev).compile(probe.clone());
        for (name, data) in &block.params {
            rep.set_param(name, data);
        }
        for sv in &split.saved {
            rep.set_param(&sv.name, &saved[sv.save_output]);
        }
        let mut feed: Vec<(&str, &[f32])> = vec![("d_output", &cot[..])];
        if probe.input_id(&residual_input).is_some() {
            feed.push((residual_input.as_str(), &x[..]));
        }
        rep.run(&feed)
    };
    let (cpu, gpu) = (run(Device::Cpu), run(device));
    eprintln!("layer {layer} on {dev_name}: {} nodes probed", ids.len());
    let mut shown = 0;
    for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
        if a.len() != b.len() {
            eprintln!("node {i}: LENGTH cpu {} vs gpu {}", a.len(), b.len());
            shown += 1;
            if shown >= 3 {
                break;
            }
            continue;
        }
        let (rel, worst) = rel_frobenius(a, b);
        if rel > 1e-5 {
            let id = ids[i];
            let n = split.replay.node(id);
            eprintln!(
                "DIVERGENCE #{shown} at %{} {:.70}\n   relF {rel:.3e} worst {worst:.3e} shape {:?}",
                id.0,
                format!("{:?}", n.op),
                n.shape.dims()
            );
            for &inp in &n.inputs {
                eprintln!(
                    "   <- %{} {:.62}",
                    inp.0,
                    format!("{:?}", split.replay.node(inp).op)
                );
            }
            shown += 1;
            if shown >= 3 {
                break;
            }
        }
    }
    if shown == 0 {
        eprintln!("no divergence above 1e-5");
    }
}

/// Per-layer residuals and readouts, this backend vs CPU. `--ignored`.
///
/// The Jacobian is asserted against CPU elsewhere; this asks the downstream
/// question — do the *representations* a lens reads, and the decisions it makes
/// from them, come out the same? Residuals are produced on each device and then
/// transported and decoded on CPU, so any difference is attributable to the
/// forward alone. Set `RLX_JLENS_DIAG_DEVICE`.
#[test]
#[ignore = "diagnostic"]
fn layer_representations_vs_cpu() {
    use rlx_jlens::{JacobianLens, Readout, rank_of};
    use rlx_runtime::Session;

    let Ok(dev_name) = std::env::var("RLX_JLENS_DIAG_DEVICE") else {
        eprintln!("set RLX_JLENS_DIAG_DEVICE");
        return;
    };
    let device = rlx_runtime::parse_device(&dev_name).expect("device");
    let lens_path = std::env::var("RLX_JLENS_DIAG_LENS")
        .unwrap_or_else(|_| "/tmp/big.lens.safetensors".to_string());
    let Ok(lens) = JacobianLens::load(&lens_path) else {
        eprintln!("no lens at {lens_path}; skipping");
        return;
    };
    let Some((model, path)) = real_model() else {
        return;
    };

    let ids = rlx_qwen35::encode_prompt_from_gguf(
        &path,
        "Fact: The capital of Japan is Tokyo. Fact: The capital city of France is",
    )
    .expect("tokenize");
    let seq = ids.len();
    let tokens: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
    let layers = lens.layers();
    let target = lens.target_layer;
    let d = model.d_model();
    let read_pos = seq - 1;

    let forward = |dev: Device| -> Vec<Vec<f32>> {
        let stack = model.stack(&layers, target, 1, seq).expect("stack");
        let mut g = Session::new(dev).compile(stack.tapped.graph().clone());
        for (name, data) in &stack.params {
            g.set_param(name, data);
        }
        g.run(&[(stack.token_input.as_str(), &tokens[..])])
    };
    let cpu = forward(Device::Cpu);
    let gpu = forward(device);

    // Transport + decode both on CPU, so only the forward differs.
    let mut readout = Readout::new(&model, 1, Device::Cpu).expect("readout");
    let vocab = readout.vocab();
    let final_row = |o: &[f32]| -> Vec<f32> { o[read_pos * d..(read_pos + 1) * d].to_vec() };
    let final_logits = readout.logits(&final_row(&cpu[0])).expect("logits");
    let answer = (0..vocab)
        .max_by(|&a, &b| final_logits[a].partial_cmp(&final_logits[b]).unwrap())
        .expect("argmax") as u32;

    eprintln!("\n{dev_name} vs CPU — residual at the read position, then the readout from it");
    eprintln!(
        "{:<7} {:<12} {:<12} {:<22} rank(answer)",
        "layer", "relF", "cos", "top-1 (cpu / dev)"
    );
    let mut worst_rel = 0.0f32;
    let mut disagreements = 0;
    for (slot, &layer) in layers.iter().enumerate() {
        let (a, b) = (final_row(&cpu[slot + 1]), final_row(&gpu[slot + 1]));
        let (rel, _) = rel_frobenius(&a, &b);
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb).max(f32::MIN_POSITIVE);
        worst_rel = worst_rel.max(rel);

        let j = lens.get(layer).expect("layer");
        let la = readout.logits(&j.transport(&a)).expect("logits");
        let lb = readout.logits(&j.transport(&b)).expect("logits");
        let top = |l: &[f32]| -> u32 {
            (0..vocab)
                .max_by(|&x, &y| l[x].partial_cmp(&l[y]).unwrap())
                .unwrap() as u32
        };
        let (ta, tb) = (top(&la), top(&lb));
        let same = ta == tb;
        if !same {
            disagreements += 1;
        }
        let dec = |t: u32| rlx_qwen35::decode_ids_from_gguf(&path, &[t], false).unwrap_or_default();
        eprintln!(
            "{:<7} {:<12.3e} {:<12.9} {:<22} {} / {}{}",
            layer,
            rel,
            cos,
            if same {
                format!("{:?}", dec(ta))
            } else {
                format!("{:?} / {:?}", dec(ta), dec(tb))
            },
            rank_of(&la, answer),
            rank_of(&lb, answer),
            if same { "" } else { "   <-- DIFFERENT" }
        );
    }
    eprintln!(
        "worst residual relF = {worst_rel:.3e}; top-1 disagreements: {disagreements}/{}",
        layers.len()
    );
}

/// First node of the *stack forward* where this backend disagrees with CPU.
/// `--ignored`; set `RLX_JLENS_DIAG_DEVICE`.
///
/// `layer_representations_vs_cpu` showed ROCm matching CPU to 2e-6 through
/// layer 20 and then diverging to cosine 0.93 at layer 22, deterministically.
/// This publishes every node of the tapped trunk as an output and reports the
/// earliest disagreement in topological order — everything after it is fallout.
#[test]
#[ignore = "diagnostic"]
fn stack_forward_first_divergent_node() {
    use rlx_runtime::Session;

    let Ok(dev_name) = std::env::var("RLX_JLENS_DIAG_DEVICE") else {
        eprintln!("set RLX_JLENS_DIAG_DEVICE");
        return;
    };
    let device = rlx_runtime::parse_device(&dev_name).expect("device");
    let Some((model, path)) = real_model() else {
        return;
    };

    let ids = rlx_qwen35::encode_prompt_from_gguf(
        &path,
        "Fact: The capital of Japan is Tokyo. Fact: The capital city of France is",
    )
    .expect("tokenize");
    let seq = ids.len();
    let tokens: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
    // Shrinking the trunk moves the *end* of the graph without moving any
    // individual layer, which separates "this layer is miscomputed" from
    // "whatever sits last in the arena is".
    let target: usize = std::env::var("RLX_JLENS_DIAG_TARGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(model.n_layers() - 1);
    let layers: Vec<usize> = (0..=target).step_by(2).collect();

    let stack = model.stack(&layers, target, 1, seq).expect("stack");
    let mut probe = stack.tapped.graph().clone();
    let ids_all: Vec<rlx_ir::NodeId> = probe.nodes().iter().map(|n| n.id).collect();
    probe.set_outputs(ids_all.clone());

    let run = |dev: Device| -> Vec<Vec<f32>> {
        let mut g = Session::new(dev).compile(probe.clone());
        for (name, data) in &stack.params {
            g.set_param(name, data);
        }
        g.run(&[(stack.token_input.as_str(), &tokens[..])])
    };
    let cpu = run(Device::Cpu);
    let gpu = run(device);
    eprintln!("{dev_name}: {} nodes probed", ids_all.len());

    // Every GatedDeltaNet / Attention node with its own relF, so the pattern
    // across layers is visible rather than just the first casualty.
    for (i, id) in ids_all.iter().enumerate() {
        let n = probe.node(*id);
        let interesting = matches!(
            n.op,
            rlx_ir::Op::GatedDeltaNet { .. } | rlx_ir::Op::Attention { .. }
        );
        if !interesting || cpu[i].len() != gpu[i].len() || cpu[i].is_empty() {
            continue;
        }
        let (rel, _) = rel_frobenius(&cpu[i], &gpu[i]);
        let all_zero = gpu[i].iter().all(|v| *v == 0.0);
        let kind = if matches!(n.op, rlx_ir::Op::Attention { .. }) {
            "Attention"
        } else {
            "GatedDeltaNet"
        };
        eprintln!(
            "  {kind:<14} %{:<6} relF {rel:.3e}{}",
            id.0,
            if all_zero {
                "   [GPU OUTPUT ALL ZERO]"
            } else {
                ""
            }
        );
    }

    let mut shown = 0;
    for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
        if a.len() != b.len() || a.is_empty() {
            continue;
        }
        let (rel, worst) = rel_frobenius(a, b);
        if rel > 1e-4 {
            let id = ids_all[i];
            let n = probe.node(id);
            eprintln!(
                "DIVERGENCE #{shown} at %{} {:.72}\n   relF {rel:.3e} worst {worst:.3e} shape {:?}",
                id.0,
                format!("{:?}", n.op),
                n.shape.dims()
            );
            for &inp in &n.inputs {
                let m = probe.node(inp);
                let idx = ids_all.iter().position(|&x| x == inp);
                let inp_rel = idx
                    .filter(|&k| cpu[k].len() == gpu[k].len() && !cpu[k].is_empty())
                    .map(|k| rel_frobenius(&cpu[k], &gpu[k]).0)
                    .unwrap_or(f32::NAN);
                eprintln!(
                    "   <- %{} {:.56} (its own relF {inp_rel:.2e})",
                    inp.0,
                    format!("{:?}", m.op)
                );
            }
            shown += 1;
            if shown >= 2 {
                break;
            }
        }
    }
    if shown == 0 {
        eprintln!("no divergence above 1e-4");
    }
}

/// What does the gradient half actually spend its work on? `--ignored`.
///
/// The lens asks only for `dResidual`, never for weight gradients, and it runs
/// the replay `d_model / dim_batch` times. Anything in there that is not on the
/// path to the tap's gradient is multiplied by that count.
#[test]
#[ignore = "diagnostic"]
fn replay_work_breakdown() {
    use rlx_autodiff::split_vjp;
    use rlx_jlens::{Tap, TappedGraph};

    let Some((model, _)) = real_model() else {
        return;
    };
    let (batch, seq, layer) = (8usize, 24usize, 3usize);
    let d = model.d_model();
    let block = model.block(layer, batch, seq).expect("block");
    let tapped = TappedGraph::new(
        block.graph.clone(),
        vec![Tap::at_input(layer, block.residual_input.clone())],
    )
    .expect("tap");
    let split = split_vjp(&tapped.vjp_graph()).expect("split");

    // A tensor carrying the batch axis is an activation; a 2-D one the size of
    // a weight is a parameter or a parameter-shaped gradient.
    let classify = |g: &rlx_ir::Graph| {
        let (mut act, mut wt, mut act_n, mut wt_n) = (0u64, 0u64, 0usize, 0usize);
        for n in g.nodes() {
            let dims: Vec<usize> = n.shape.dims().iter().map(|x| x.unwrap_static()).collect();
            let bytes = dims.iter().product::<usize>() as u64 * 4;
            let is_act = dims.first().copied() == Some(batch);
            if is_act {
                act += bytes;
                act_n += 1;
            } else if dims.len() == 2 && dims[0] > 1 && dims[1] > 1 {
                wt += bytes;
                wt_n += 1;
            }
        }
        (act, act_n, wt, wt_n)
    };
    for (name, g) in [("save", &split.save), ("replay", &split.replay)] {
        let (a, an, w, wn) = classify(g);
        eprintln!(
            "{name:<7} {} nodes | activation-shaped {an:4} ({:.2} GiB) | weight-shaped {wn:4} ({:.2} GiB)",
            g.len(),
            a as f64 / (1u64 << 30) as f64,
            w as f64 / (1u64 << 30) as f64
        );
    }
    // Which ops produce weight-shaped tensors in the gradient half?
    let mut hist = std::collections::BTreeMap::new();
    for n in split.replay.nodes() {
        let dims: Vec<usize> = n.shape.dims().iter().map(|x| x.unwrap_static()).collect();
        if dims.len() == 2 && dims[0] > 1 && dims[1] > 1 && dims.first().copied() != Some(batch) {
            let k = format!("{:?}", n.op);
            let k = k.split(['{', '(', ' ']).next().unwrap_or("?").to_string();
            let e = hist.entry(k).or_insert((0usize, 0u64));
            e.0 += 1;
            e.1 += dims.iter().product::<usize>() as u64 * 4;
        }
    }
    eprintln!("weight-shaped nodes in the replay half, by op:");
    let mut rows: Vec<_> = hist.into_iter().collect();
    rows.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
    for (op, (n, b)) in rows.into_iter().take(8) {
        eprintln!(
            "   {op:<22} x{n:<4} {:.2} GiB",
            b as f64 / (1u64 << 30) as f64
        );
    }
    // Are they even reachable from the outputs? A dead weight-gradient matmul
    // costs nothing if the compiler drops it, and a great deal if it does not —
    // the replay runs `d_model / dim_batch` times.
    let g = &split.replay;
    let mut live = std::collections::HashSet::new();
    let mut stack_ids: Vec<rlx_ir::NodeId> = g.outputs.clone();
    while let Some(id) = stack_ids.pop() {
        if !live.insert(id) {
            continue;
        }
        for &i in &g.node(id).inputs {
            stack_ids.push(i);
        }
    }
    let (mut dead_n, mut dead_flops) = (0usize, 0u128);
    let (mut live_n, mut live_flops) = (0usize, 0u128);
    for n in g.nodes() {
        if !matches!(n.op, rlx_ir::Op::MatMul) {
            continue;
        }
        let dims: Vec<usize> = n.shape.dims().iter().map(|x| x.unwrap_static()).collect();
        let k = g.node(n.inputs[1]).shape.dims();
        let kdim = k.first().map(|x| x.unwrap_static()).unwrap_or(1);
        let flops = dims.iter().product::<usize>() as u128 * kdim as u128 * 2;
        if live.contains(&n.id) {
            live_n += 1;
            live_flops += flops;
        } else {
            dead_n += 1;
            dead_flops += flops;
        }
    }
    eprintln!(
        "replay matmuls: {live_n} live ({:.2} GFLOP), {dead_n} DEAD ({:.2} GFLOP) — dead work is \
         multiplied by the pass count",
        live_flops as f64 / 1e9,
        dead_flops as f64 / 1e9
    );
    eprintln!(
        "replay nodes reachable from outputs: {}/{}",
        live.len(),
        g.len()
    );
    let _ = d;
}
