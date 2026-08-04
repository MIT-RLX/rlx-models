//! `--synth-stages N` is GRAPH-LIVE: the `rlx! { }` block sums exactly
//! `cfg.synth_stages` residual-VQ codebook stages per projection (via a runtime
//! `repeat s` stage loop), so the number of `Op::SynthMatMul` nodes in the built
//! graph scales linearly with `synth_stages`. This is the proof that the CLI
//! knob actually drives the graph (it used to only affect PQ-init scaling while
//! the block hardcoded exactly two stages).

use rlx_tiny::config::GptConfig;
use rlx_tiny::model;

/// Count `Op::SynthMatMul` nodes in the model's forward graph (logits mode).
/// Matched on the derived `Debug` prefix so no `rlx-ir` dependency is needed.
fn synth_matmul_nodes(cfg: &GptConfig) -> usize {
    let f = model::build(cfg, 1, false);
    f.graph()
        .nodes()
        .iter()
        .filter(|n| format!("{:?}", n.op).starts_with("SynthMatMul"))
        .count()
}

#[test]
fn synth_stages_drives_synthmatmul_node_count() {
    let with_stages = |s: usize| {
        let mut c = GptConfig::smoke();
        c.synth_stages = s;
        c
    };
    let base = GptConfig::smoke();
    // Six synth projections (wq wk wv wo w1 w2) per layer, each summed over
    // `synth_stages` codebook stages → 6·n_layer synth-matmuls PER STAGE.
    let per_stage = 6 * base.n_layer;

    let n1 = synth_matmul_nodes(&with_stages(1));
    let n2 = synth_matmul_nodes(&with_stages(2));
    let n3 = synth_matmul_nodes(&with_stages(3));

    // Exact per-stage scaling.
    assert_eq!(n1, per_stage, "stages=1 → one synth-matmul per projection");
    assert_eq!(
        n2,
        2 * per_stage,
        "stages=2 reproduces the old fixed-2 behavior"
    );
    assert_eq!(
        n3,
        3 * per_stage,
        "stages=3 → three synth-matmuls per projection"
    );

    // The headline relationship: stages=3 builds 3× the per-projection
    // SynthMatMul nodes of stages=1 — `--synth-stages N` is now live.
    assert_eq!(
        n3,
        3 * n1,
        "synth_stages must scale the graph linearly (3× at 3 vs 1)"
    );
    assert_eq!(n2, 2 * n1);
}
