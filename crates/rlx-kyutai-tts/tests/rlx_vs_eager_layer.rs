//! The RLX temporal graph must compute what the eager backbone computes.
//!
//! `rlx_backend_parity.rs` only ever compares the RLX graph to *itself* (`assert_logits_match_cpu
//! (label, &cpu, &cpu)` on CPU, then the same graph on another device), so a uniformly wrong
//! graph passes it. This is the missing direction: same synthetic weights, same input, RLX graph
//! vs the eager `StreamingTransformer`. It needs no checkpoint, so it runs everywhere, and it is
//! parameterized by depth so a failure says *which* layer first diverges.

mod backend_common;

use backend_common::{cosine, synthetic_weights, tiny_cfg};
use ndarray::{Array1, Array2};
use rlx_kyutai_tts::config::KyutaiTtsConfig;
use rlx_kyutai_tts::nn::rms_norm;
use rlx_kyutai_tts::rlx_lm::{TtsDims, temporal_decode_bucketed_rlx};
use rlx_kyutai_tts::transformer::StreamingTransformer;
use rlx_runtime::Device;
use std::collections::HashMap;

type Weights = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Post-`out_norm` hidden from the eager stack — the same tensor the RLX graph returns.
fn eager_hidden(cfg: &KyutaiTtsConfig, w: &Weights, emb: &[f32], cross_ctx: &[f32]) -> Vec<f32> {
    let d = cfg.dim;
    let tcfg = cfg.backbone_runtime();
    // `false, 1.0`: no cross positional embedding, so both sides see the raw context and the
    // comparison isolates the transformer rather than `prepare_cross_ctx`.
    let layers = StreamingTransformer::load_layers(&tcfg, w, false, 1.0).expect("eager layers");
    let mut tr = StreamingTransformer::new(tcfg, layers).expect("eager transformer");
    let t_cross = cross_ctx.len() / d;
    let ctx = Array2::from_shape_vec((t_cross, d), cross_ctx.to_vec()).expect("ctx");
    tr.set_cross_context(Some(&ctx)).expect("cross ctx");
    let x = Array2::from_shape_vec((1, d), emb.to_vec()).expect("emb");
    let h = tr.forward(&x).expect("eager forward");
    let alpha = Array1::from_vec(w["out_norm.alpha"].0[..d].to_vec());
    rms_norm(h.view(), &alpha).row(0).to_vec()
}

fn rlx_hidden(
    dims: &TtsDims,
    w: &Weights,
    emb: &[f32],
    cross_ctx: &[f32],
    cross_len: usize,
    upper: usize,
) -> Vec<f32> {
    temporal_decode_bucketed_rlx(
        dims,
        w,
        emb,
        cross_ctx,
        cross_len,
        &[],
        0,
        upper,
        Device::Cpu,
    )
    .expect("rlx decode")
    .1
}

/// One decode step at `n_layers` depth; returns `(cosine, max|Δ|)` of the backbone output.
fn compare_at_depth(n_layers: usize) -> (f64, f32) {
    let mut cfg = tiny_cfg();
    cfg.num_layers = n_layers;
    let d = cfg.dim;
    let w = synthetic_weights(&cfg);
    let emb: Vec<f32> = (0..d).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    // A single, non-zero cross frame: cross-attention must actually contribute, or the check
    // would silently skip the block that conditions on the speaker.
    let cross_ctx = vec![0.02f32; d];
    let dims = TtsDims::from_cfg_and_weights(&cfg, 1, &w).expect("dims");

    let eager = eager_hidden(&cfg, &w, &emb, &cross_ctx);
    let rlx = rlx_hidden(&dims, &w, &emb, &cross_ctx, dims.t_cross, cfg.context);
    assert_eq!(
        eager.len(),
        rlx.len(),
        "hidden width differs at {n_layers} layers"
    );
    cosine(&eager, &rlx)
}

#[test]
fn rlx_graph_matches_eager_backbone_layer_by_layer() {
    let mut first_bad = None;
    for n in 1..=3 {
        let (cos, maxd) = compare_at_depth(n);
        eprintln!("{n} layer(s): cosine={cos:.6} max|Δ|={maxd:.3e}");
        if cos <= 0.999 && first_bad.is_none() {
            first_bad = Some((n, cos, maxd));
        }
    }
    if let Some((n, cos, maxd)) = first_bad {
        panic!(
            "RLX temporal graph diverges from the eager backbone starting at {n} layer(s): \
             cosine {cos:.6}, max|Δ| {maxd:.3e}"
        );
    }
}

/// Eager `set_cross_context(None)` **skips** the cross-attention block entirely. The RLX graph
/// has no such switch — it always attends, over whatever `cross_ctx` buffer it was given, which
/// `RlxKyutaiTtsModel::set_generation_conditions` zero-fills when there is no speaker. The two
/// only agree if attending an all-zero context is exactly a no-op.
#[test]
fn zero_cross_context_is_equivalent_to_skipping_cross_attention() {
    let mut cfg = tiny_cfg();
    cfg.num_layers = 2;
    let d = cfg.dim;
    let w = synthetic_weights(&cfg);
    let emb: Vec<f32> = (0..d).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let zero_ctx = vec![0.0f32; d];
    let dims = TtsDims::from_cfg_and_weights(&cfg, 1, &w).expect("dims");

    // Eager with no speaker: cross-attention skipped.
    let tcfg = cfg.backbone_runtime();
    let layers = StreamingTransformer::load_layers(&tcfg, &w, false, 1.0).expect("layers");
    let mut tr = StreamingTransformer::new(tcfg, layers).expect("transformer");
    tr.set_cross_context(None).expect("no cross ctx");
    let x = Array2::from_shape_vec((1, d), emb.clone()).expect("emb");
    let h = tr.forward(&x).expect("eager forward");
    let alpha = Array1::from_vec(w["out_norm.alpha"].0[..d].to_vec());
    let eager = rms_norm(h.view(), &alpha).row(0).to_vec();

    let rlx = rlx_hidden(&dims, &w, &emb, &zero_ctx, 0, cfg.context);
    let (cos, maxd) = cosine(&eager, &rlx);
    eprintln!("zero-cross: cosine={cos:.6} max|Δ|={maxd:.3e}");
    assert!(
        cos > 0.999,
        "attending an all-zero cross context is not a no-op: cosine {cos:.6}, max|Δ| {maxd:.3e} \
         — the RLX graph needs the same skip/mask the eager path has"
    );
}

/// `RlxKyutaiTtsModel::set_generation_conditions` pads the conditioner's cross context out to a
/// fixed `MAX_SPEAKER_CROSS_FRAMES` buffer with zeros, and the graph attends over all of it with
/// `MaskKind::None`. Eager attends only the real frames. Zero *keys* score 0 against any query,
/// so after the softmax each padding slot carries weight `exp(0)` — padding does not drop out,
/// it competes, and with enough of it the real speaker frames are washed out.
#[test]
fn zero_padded_cross_context_must_not_dilute_the_real_frames() {
    let mut cfg = tiny_cfg();
    cfg.num_layers = 2;
    let d = cfg.dim;
    let w = synthetic_weights(&cfg);
    let emb: Vec<f32> = (0..d).map(|i| 0.01 * (i as f32 + 1.0)).collect();

    // One real conditioning frame …
    let real: Vec<f32> = (0..d).map(|i| 0.02 + 0.001 * i as f32).collect();
    // … as eager sees it (exactly one frame).
    let tcfg = cfg.backbone_runtime();
    let layers = StreamingTransformer::load_layers(&tcfg, &w, false, 1.0).expect("layers");
    let mut tr = StreamingTransformer::new(tcfg, layers).expect("transformer");
    let ctx = Array2::from_shape_vec((1, d), real.clone()).expect("ctx");
    tr.set_cross_context(Some(&ctx)).expect("cross ctx");
    let x = Array2::from_shape_vec((1, d), emb.clone()).expect("emb");
    let h = tr.forward(&x).expect("eager forward");
    let alpha = Array1::from_vec(w["out_norm.alpha"].0[..d].to_vec());
    let eager = rms_norm(h.view(), &alpha).row(0).to_vec();

    // … and as the RLX model builds it: the same frame followed by zero padding.
    const PADDED_FRAMES: usize = 8;
    let mut padded = real.clone();
    padded.resize(PADDED_FRAMES * d, 0.0);
    let dims = TtsDims::from_cfg_and_weights(&cfg, PADDED_FRAMES, &w).expect("dims");
    let rlx = rlx_hidden(&dims, &w, &emb, &padded, 1, cfg.context);

    let (cos, maxd) = cosine(&eager, &rlx);
    eprintln!("padded-cross ({PADDED_FRAMES} frames, 1 real): cosine={cos:.6} max|Δ|={maxd:.3e}");
    assert!(
        cos > 0.999,
        "zero padding is diluting cross-attention: cosine {cos:.6}, max|Δ| {maxd:.3e} \
         — the graph needs a cross-attention mask over the real frame count"
    );
}

/// Pins the SwiGLU width convention that the RLX graph got wrong.
///
/// `hidden_scale` is not the hidden width. The 1.6B checkpoint stores
/// `gating.linear_in.weight = [11264, 2048]` — hidden 5632 — for
/// `dim_feedforward = 2048 · 4.125 = 8448`, i.e. the usual SwiGLU ⅔ adjustment.
/// `TtsDims::from_cfg` computed `dim_feedforward / 2 = 4224` and every gate/up/down slice in
/// the graph was taken at the wrong offset, which is why the RLX path produced fluent speech
/// that ignored the script.
#[test]
fn swiglu_width_comes_from_the_checkpoint_not_hidden_scale() {
    let cfg = tiny_cfg();
    let w = synthetic_weights(&cfg);
    let d = cfg.dim;
    let from_weights = TtsDims::from_cfg_and_weights(&cfg, 1, &w)
        .expect("dims")
        .ffn;
    let from_cfg_only = TtsDims::from_cfg(&cfg, 1).ffn;

    let stored = &w["transformer.layers.0.gating.linear_in.weight"].1;
    assert_eq!(
        stored,
        &vec![2 * from_weights, d],
        "fixture is not checkpoint-shaped"
    );
    assert_ne!(
        from_weights, from_cfg_only,
        "fixture no longer distinguishes the two conventions, so it cannot catch the bug"
    );

    // And the binder must refuse the config-derived width rather than mis-slice silently.
    let bad = TtsDims::from_cfg(&cfg, 1);
    let mut compiled = rlx_runtime::Session::new(Device::Cpu)
        .compile(rlx_kyutai_tts::rlx_lm::build_temporal_decode_graph_bucketed(&bad, cfg.context));
    let err = rlx_kyutai_tts::rlx_lm::set_temporal_params(&mut compiled, &bad, &w)
        .expect_err("binding a config-derived ffn must fail loudly");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("gating.linear_in.weight"),
        "expected a shape complaint naming the packed projection, got: {msg}"
    );
}
