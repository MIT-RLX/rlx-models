//! End-to-end forward on a synthetic Qwen3.5 trunk.
//!
//! This is where the architectural claim gets checked: **asking a question
//! alongside others must give exactly the answer it gives alone.** In the row
//! form that holds by construction, so the test is really pinning the things
//! that could break it — row batching, right-padding, and the readout
//! indexing — none of which announce themselves when they go wrong. A
//! mis-gathered hidden state still yields a valid probability distribution.
//!
//! The companion check is liveness: editing an option must move *that*
//! question's numbers. Without it, a readout that returned a constant would
//! pass the isolation test perfectly.

use rlx_kev::api::{Record, RecordQuestion};
use rlx_kev::encode::{Encoder, SPECIAL, TokenizerLike};
use rlx_kev::model::{DecisionModel, ShapePolicy};
use rlx_kev::pointer::PointerHead;
use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_qwen35::{MatWeight, Proj, Qwen35LayerFfn, Qwen35TrunkLayer, Qwen35Weights};
use rlx_runtime::Device;

/// One token per character, ids kept well inside `tiny_cfg`'s 32-token
/// vocabulary, delimiters at the top of the range.
struct FakeTok;

impl TokenizerLike for FakeTok {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        SPECIAL
            .iter()
            .position(|s| *s == token)
            .map(|i| 27 + i as u32)
    }
    fn encode_no_special(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(text.chars().map(|c| (c as u32) % 20 + 1).collect())
    }
}

/// Deterministic pseudo-random scalar.
fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

/// `synth_weights` builds every projection from `ramp()`, a strictly linear
/// sequence. The resulting trunk is so smooth that hidden states at different
/// positions agree to ~1e-3 and the readout comes out uniform to within f32
/// rounding — at which point *every* assertion below holds vacuously, and a
/// readout that returned a constant would pass the isolation test perfectly.
///
/// So give the trunk full-rank projections. `ramp()` is a fixture convenience,
/// not part of the architecture, and randomizing it makes the checks strictly
/// sharper: leakage between rows is amplified along with signal.
///
/// `ssm_a` and `ssm_dt_bias` are left alone — they parameterize the DeltaNet
/// recurrence's decay, and random values there make the scan diverge rather
/// than merely un-smooth.
fn randomize_projections(w: &mut Qwen35Weights) {
    let fill = |v: &mut Vec<f32>, seed: u64| {
        for (i, x) in v.iter_mut().enumerate() {
            *x = 0.5 * hashed(seed, i);
        }
    };
    let fill_mat = |m: &mut MatWeight, seed: u64| {
        if let MatWeight::F32(v) = m {
            fill(v, seed);
        }
    };
    let fill_proj = |p: &mut Proj, seed: u64| {
        for (j, m) in p.mats_mut().enumerate() {
            fill_mat(m, seed.wrapping_add(j as u64 * 7919));
        }
    };
    // Norm gammas stay near 1: a random gain there rescales the residual
    // stream instead of decorrelating it.
    let fill_norm = |v: &mut Vec<f32>, seed: u64| {
        for (i, x) in v.iter_mut().enumerate() {
            *x = 1.0 + 0.3 * hashed(seed, i);
        }
    };

    for (li, layer) in w.trunk_layers.iter_mut().enumerate() {
        let s = |k: u64| (li as u64 + 1).wrapping_mul(0x9E37_79B1).wrapping_add(k);
        let ffn = match layer {
            Qwen35TrunkLayer::Linear(l) => {
                fill_norm(&mut l.attn_norm, s(1));
                fill_norm(&mut l.attn_post_norm, s(2));
                fill_proj(&mut l.attn_qkv, s(3));
                fill_proj(&mut l.attn_gate, s(4));
                fill_proj(&mut l.ssm_beta, s(5));
                fill_proj(&mut l.ssm_alpha, s(6));
                fill_proj(&mut l.ssm_out, s(7));
                fill_norm(&mut l.ssm_norm, s(8));
                &mut l.ffn
            }
            Qwen35TrunkLayer::FullAttn(l) => {
                fill_norm(&mut l.attn_norm, s(1));
                fill_norm(&mut l.attn_post_norm, s(2));
                fill_proj(&mut l.attn_q_gate, s(3));
                fill_proj(&mut l.attn_k, s(4));
                fill_proj(&mut l.attn_v, s(5));
                fill_proj(&mut l.attn_output, s(6));
                fill_norm(&mut l.attn_q_norm, s(7));
                fill_norm(&mut l.attn_k_norm, s(8));
                &mut l.ffn
            }
        };
        if let Qwen35LayerFfn::Dense { gate, up, down } = ffn {
            fill_proj(gate, s(20));
            fill_proj(up, s(21));
            fill_proj(down, s(22));
        }
    }
    fill_norm(&mut w.output_norm, 0xFEED);
}

fn random_head(hidden: usize, head_dim: usize) -> PointerHead {
    let w = |seed: u64| -> Vec<f32> {
        (0..head_dim * hidden)
            .map(|i| 0.5 * hashed(seed, i))
            .collect()
    };
    let b = |seed: u64| -> Vec<f32> { (0..head_dim).map(|i| 0.1 * hashed(seed, i)).collect() };
    PointerHead::new(w(1), b(2), w(3), b(4), hidden, head_dim, 1.0).expect("head")
}

fn model() -> DecisionModel<FakeTok> {
    let cfg = tiny_cfg();
    let mut weights = synth_weights(&cfg);
    randomize_projections(&mut weights);
    let head = random_head(cfg.hidden_size, 8);
    DecisionModel::new(
        Encoder::new(FakeTok).expect("delimiters"),
        head,
        cfg,
        weights,
        Device::Cpu,
        // tiny_cfg has max_position_embeddings = 64, which is exactly one
        // SHAPE_BUCKET, so every record here compiles to a single shape.
        ShapePolicy::Bucketed {
            max_rows: 8,
            max_seq: 64,
        },
    )
    .expect("model")
}

fn q(instr: &str, options: &[&str]) -> RecordQuestion {
    RecordQuestion {
        instr: instr.into(),
        options: options.iter().map(|s| (*s).to_string()).collect(),
        label: 0,
    }
}

fn record(state: &str, questions: Vec<RecordQuestion>) -> Record {
    Record {
        state: state.into(),
        questions,
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn probabilities_are_a_distribution_per_question() {
    let mut m = model();
    let rec = record(
        "late order",
        vec![
            q("dept", &["ret", "ship", "bill"]),
            q("urgent", &["no", "yes"]),
        ],
    );
    let probs = m.probs(&rec).expect("forward");

    assert_eq!(probs.len(), 2, "one vector per question");
    assert_eq!(probs[0].len(), 3);
    assert_eq!(probs[1].len(), 2);
    for (i, p) in probs.iter().enumerate() {
        assert!(
            p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "question {i} produced {p:?}"
        );
        assert!(
            (p.iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "question {i} sums to {}",
            p.iter().sum::<f32>()
        );
    }
}

/// The isolation claim. Adding a sibling question changes the batch shape,
/// the padding, and every readout index — and must change no answer.
#[test]
fn a_question_answers_the_same_alone_as_alongside_siblings() {
    let mut m = model();
    let state = "shoes arrived late";
    let target = q("dept", &["ret", "ship", "bill"]);

    let alone = m
        .probs(&record(state, vec![target.clone()]))
        .expect("forward");

    // Non-vacuity: if the distribution were uniform, equality below would
    // prove nothing.
    let spread = alone[0].iter().copied().fold(f32::MIN, f32::max)
        - alone[0].iter().copied().fold(f32::MAX, f32::min);
    assert!(
        spread > 1e-4,
        "the lone answer is uniform ({:?}); this test cannot distinguish a \
         correct readout from a constant one",
        alone[0]
    );

    for extra in 1..=3 {
        let mut qs = vec![target.clone()];
        for i in 0..extra {
            qs.push(q(
                &format!("sibling{i}"),
                &["alpha", "beta", "gamma", "delta"],
            ));
        }
        let together = m.probs(&record(state, qs)).expect("forward");
        assert_eq!(together.len(), 1 + extra);
        let diff = max_abs_diff(&alone[0], &together[0]);
        assert!(
            diff < 1e-5,
            "adding {extra} sibling question(s) moved the first answer by {diff}: \
             {:?} vs {:?}",
            alone[0],
            together[0]
        );
    }
}

/// Order among siblings must not matter either: the target question answers
/// the same whether it is asked first or last.
#[test]
fn sibling_order_does_not_change_an_answer() {
    let mut m = model();
    let state = "double charge on my card";
    let target = q("dept", &["ret", "bill"]);
    let other = q("tone", &["calm", "cross"]);

    let first = m
        .probs(&record(state, vec![target.clone(), other.clone()]))
        .expect("forward");
    let last = m
        .probs(&record(state, vec![other, target]))
        .expect("forward");

    let diff = max_abs_diff(&first[0], &last[1]);
    assert!(
        diff < 1e-5,
        "the target answer moved by {diff} when reordered: {:?} vs {:?}",
        first[0],
        last[1]
    );
}

/// Liveness: the readout must actually be reading the option it claims to.
#[test]
fn editing_an_option_moves_that_questions_answer() {
    let mut m = model();
    let state = "shoes arrived late";

    let before = m
        .probs(&record(state, vec![q("dept", &["ret", "ship"])]))
        .expect("forward");
    let after = m
        .probs(&record(state, vec![q("dept", &["ret", "zzzz"])]))
        .expect("forward");

    let diff = max_abs_diff(&before[0], &after[0]);
    assert!(
        diff > 1e-4,
        "rewriting an option left the distribution unchanged ({diff}); the \
         readout is not reading the option spans"
    );
}

/// And the state must reach the answer — otherwise the branch is being read
/// in isolation from the document it is supposed to be about.
#[test]
fn editing_the_state_moves_the_answer() {
    let mut m = model();
    let qs = vec![q("dept", &["ret", "ship"])];
    let a = m
        .probs(&record("shoes arrived late", qs.clone()))
        .expect("forward");
    let b = m
        .probs(&record("charged me twice over", qs))
        .expect("forward");
    let diff = max_abs_diff(&a[0], &b[0]);
    assert!(
        diff > 1e-4,
        "changing the state left the answer unchanged ({diff}); the branch is \
         not attending to the state"
    );
}

#[test]
fn repeated_runs_are_bit_identical() {
    let mut m = model();
    let rec = record("same input", vec![q("dept", &["ret", "ship", "bill"])]);
    let a = m.probs(&rec).expect("forward");
    let b = m.probs(&rec).expect("forward");
    assert_eq!(a, b, "the forward pass is not deterministic");
}

#[test]
fn a_request_that_exceeds_the_pinned_shape_is_refused() {
    let cfg = tiny_cfg();
    let mut weights = synth_weights(&cfg);
    randomize_projections(&mut weights);
    let head = random_head(cfg.hidden_size, 8);
    let mut m = DecisionModel::new(
        Encoder::new(FakeTok).expect("delimiters"),
        head,
        cfg,
        weights,
        Device::Cpu,
        ShapePolicy::Fixed { rows: 1, seq: 64 },
    )
    .expect("model");

    // One question fits the pinned 1x64 shape.
    assert!(
        m.probs(&record("s", vec![q("i", &["a", "b"])])).is_ok(),
        "a fitting request should run"
    );
    assert_eq!(m.compiled_shape(), Some((1, 64)));

    // Two questions need two rows, which the pinned shape cannot serve.
    let err = m
        .probs(&record("s", vec![q("i", &["a"]), q("j", &["b"])]))
        .expect_err("must refuse rather than truncate");
    assert!(
        err.to_string().contains("pinned"),
        "the error should explain the pinned shape, got: {err}"
    );
}
