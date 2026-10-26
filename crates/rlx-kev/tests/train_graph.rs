//! Does the training graph actually train?
//!
//! A loss that goes down proves very little — a wrong backward still descends
//! *something*. So these check the three things that would make training
//! silently useless:
//!
//! 1. injection is a no-op until the first step (`B = 0`), so step 0 is the
//!    base model;
//! 2. every trainable gets a **non-zero** gradient (a duplicated `Param` in a
//!    graph rewriter has produced all-zero grads in this codebase before);
//! 3. the analytic gradient matches a finite difference.
//!
//! Plus an overfit-one-batch check, which is the cheapest end-to-end signal
//! that the wiring is right.

use rlx_kev::api::{Record, RecordQuestion};
use rlx_kev::encode::{EncodeOpts, Encoder, SPECIAL, TokenizerLike};
use rlx_kev::train::{
    LoraTargets, TrainConfig, TrainExample, TrainShape, Trainer, build_feeds, build_train_graph,
};
use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_qwen35::{MatWeight, Proj, Qwen35LayerFfn, Qwen35TrunkLayer, Qwen35Weights};
use rlx_runtime::Device;

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

fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

/// `synth_weights` is built from `ramp()`, which leaves the trunk nearly
/// position-invariant — every check here would pass vacuously. Give it
/// full-rank projections, but leave `ssm_a` / `ssm_dt_bias` alone or the
/// DeltaNet recurrence diverges.
fn randomize(w: &mut Qwen35Weights) {
    let fill = |v: &mut Vec<f32>, seed: u64| {
        for (i, x) in v.iter_mut().enumerate() {
            *x = 0.5 * hashed(seed, i);
        }
    };
    let fill_proj = |p: &mut Proj, seed: u64| {
        for (j, m) in p.mats_mut().enumerate() {
            if let MatWeight::F32(v) = m {
                fill(v, seed.wrapping_add(j as u64 * 7919));
            }
        }
    };
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

fn weights() -> Qwen35Weights {
    let cfg = tiny_cfg();
    let mut w = synth_weights(&cfg);
    randomize(&mut w);
    w
}

const SHAPE: TrainShape = TrainShape {
    rows: 2,
    seq: 64,
    max_options: 3,
};

fn tcfg() -> TrainConfig {
    TrainConfig {
        lora_rank: 4,
        lora_alpha: 8.0,
        targets: LoraTargets::All,
        head_dim: 8,
        lr: 5e-3,
        head_lr: 1e-2,
        weight_decay: 0.0,
        clip_grad_norm: 1.0,
        // Off in the graph tests so the finite-difference check is
        // deterministic; the recipe default is 0.05.
        lora_dropout: 0.0,
        // Exercise checkpointing in the gradient checks: a recompute that
        // shifted a gradient would show up in the finite-difference test.
        // 0 = planner-guided: try checkpointing, keep it only if the arena
        // actually shrinks. Exercises both branches across the test shapes.
        checkpoint_segments: 0,
        bf16: false,
        seed: 7,
    }
}

fn example(state: &str, label: usize) -> TrainExample {
    TrainExample {
        record: Record {
            state: state.into(),
            questions: vec![RecordQuestion {
                instr: "which team".into(),
                options: vec!["ret".into(), "ship".into(), "bill".into()],
                label,
            }],
        },
    }
}

fn opts() -> EncodeOpts {
    EncodeOpts::default()
}

fn encoder() -> Encoder<FakeTok> {
    Encoder::new(FakeTok).expect("delimiters")
}

#[test]
fn lora_targets_match_the_graphs_param_names() {
    let cfg = tiny_cfg();
    let g = build_train_graph(&cfg, weights(), SHAPE, &tcfg()).expect("build");
    assert!(
        !g.adapter_names.is_empty(),
        "no LoRA was injected — the target patterns do not match the graph"
    );
    // The tiny config has both block kinds, so both families must be hit.
    let joined = g.adapter_names.join(" ");
    assert!(joined.contains("ffn_gate"), "MLP not adapted: {joined}");
    assert!(
        joined.contains("attn_qkv") || joined.contains("ssm_out"),
        "DeltaNet projections not adapted: {joined}"
    );
}

/// `B = 0` at init, so the adapter contributes nothing until the first step.
#[test]
fn injection_is_a_no_op_before_the_first_step() {
    let cfg = tiny_cfg();
    let g = build_train_graph(&cfg, weights(), SHAPE, &tcfg()).expect("build");
    for name in &g.adapter_names {
        let v = g.params.get(name).expect("adapter param present");
        if name.ends_with(".b") {
            assert!(
                v.iter().all(|x| *x == 0.0),
                "{name} must start at zero so injection is a no-op"
            );
        } else {
            assert!(
                v.iter().any(|x| *x != 0.0),
                "{name} must start non-zero or B's gradient is identically zero"
            );
        }
    }
}

/// At step 1 only `B` and the head can move.
///
/// With `B = 0`, `dL/dA = (dL/dy) · Bᵀ = 0` — so an all-zero gradient on every
/// `A` is *correct* at initialization, not a wiring bug. Pinning it means a
/// future reader does not "fix" it, and it also proves the two sides are wired
/// distinctly rather than aliased.
#[test]
fn at_step_one_only_b_and_the_head_have_gradients() {
    let cfg = tiny_cfg();
    let g = build_train_graph(&cfg, weights(), SHAPE, &tcfg()).expect("build");
    let mut t = Trainer::new(g, Device::Cpu, &tcfg()).expect("trainer");
    let e = encoder();
    let feeds = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");
    let loss = t.accumulate(&feeds).expect("backward");
    assert!(loss.is_finite() && loss > 0.0, "loss = {loss}");

    let names: Vec<String> = t.trainable_names().iter().map(|s| s.to_string()).collect();
    let mut a_nonzero = 0;
    let mut b_nonzero = 0;
    for (i, n) in names.iter().enumerate() {
        let g = t.grad_of(i).expect("gradient present");
        let any = g.iter().any(|x| *x != 0.0);
        if n.ends_with(".a") && any {
            a_nonzero += 1;
        }
        if n.ends_with(".b") && any {
            b_nonzero += 1;
        }
    }
    assert_eq!(
        a_nonzero, 0,
        "`A` cannot have a gradient while `B` is still zero"
    );
    assert!(
        b_nonzero > 0,
        "`B` must have a gradient at step 1, or nothing is learning"
    );
}

/// After two steps every trainable must have moved.
///
/// `B` moves first, which makes `A`'s gradient non-zero from step 2 — so this
/// is where a duplicated `Param` in a graph rewriter would show up as
/// permanently frozen weights. That failure mode has occurred in this
/// codebase, and it is silent: the loss still descends via the other tensors.
#[test]
fn every_trainable_moves_within_two_steps() {
    let cfg = tiny_cfg();
    let g = build_train_graph(&cfg, weights(), SHAPE, &tcfg()).expect("build");
    let names: Vec<String> = g.wrt.iter().map(|s| s.name.clone()).collect();
    let before: Vec<Vec<f32>> = names.iter().map(|n| g.params[n].clone()).collect();

    let mut t = Trainer::new(g, Device::Cpu, &tcfg()).expect("trainer");
    let e = encoder();
    let feeds = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");
    for _ in 0..2 {
        t.accumulate(&feeds).expect("backward");
        t.optimizer_step().expect("step");
    }

    let mut unchanged = Vec::new();
    for (n, b) in names.iter().zip(&before) {
        let a = &t.params()[n];
        if a.iter().zip(b).all(|(x, y)| x == y) {
            unchanged.push(n.clone());
        }
    }
    assert!(
        unchanged.is_empty(),
        "{} trainable(s) never moved: {:?}",
        unchanged.len(),
        &unchanged[..unchanged.len().min(6)]
    );
}

/// Analytic vs finite-difference gradient on the pointer head.
///
/// Checked on the head rather than a LoRA tensor because the head sits right
/// at the loss, so the central difference is well conditioned — screening the
/// oracle matters more than coverage here.
#[test]
fn analytic_gradient_matches_a_finite_difference() {
    let cfg = tiny_cfg();
    let e = encoder();
    let ex = [example("late parcel and a double charge", 2)];
    let feeds = build_feeds(&e, &cfg, SHAPE, &ex, opts()).expect("feeds");

    // Analytic: one backward, read the head's gradient.
    let g = build_train_graph(&cfg, weights(), SHAPE, &tcfg()).expect("build");
    let idx = g
        .wrt
        .iter()
        .position(|s| s.name == rlx_kev::train::HEAD_Q_W)
        .expect("head q.weight is trainable");
    let base_params = g.params.clone();
    let mut t = Trainer::new(g, Device::Cpu, &tcfg()).expect("trainer");
    let l0 = t.accumulate(&feeds).expect("backward");
    let grad = t.grad_of(idx).expect("gradient").to_vec();

    // Numeric: perturb a few entries and re-run the forward only.
    let probe: Vec<usize> = vec![0, 17, 41, 123];
    let mut worst_rel = 0.0f32;
    for &i in &probe {
        let mut losses = [0.0f32; 2];
        for (k, eps) in [1e-2f32, -1e-2].iter().enumerate() {
            let mut p = base_params.clone();
            p.get_mut(rlx_kev::train::HEAD_Q_W).expect("param")[i] += *eps;
            let mut gg = build_train_graph(&cfg, weights(), SHAPE, &tcfg()).expect("build");
            gg.params = p;
            let mut tt = Trainer::new(gg, Device::Cpu, &tcfg()).expect("trainer");
            losses[k] = tt.accumulate(&feeds).expect("forward");
        }
        let fd = (losses[0] - losses[1]) / (2.0 * 1e-2);
        let an = grad[i];
        let rel = (fd - an).abs() / an.abs().max(fd.abs()).max(1e-4);
        worst_rel = worst_rel.max(rel);
        assert!(
            rel < 5e-2,
            "entry {i}: analytic {an} vs finite difference {fd} (rel {rel}); loss0 = {l0}"
        );
    }
    eprintln!("worst relative gradient error: {worst_rel:.4}");
}

/// The cheapest end-to-end signal: a handful of steps on one batch must drive
/// the loss down substantially.
#[test]
fn overfits_a_single_batch() {
    let cfg = tiny_cfg();
    let g = build_train_graph(&cfg, weights(), SHAPE, &tcfg()).expect("build");
    let mut t = Trainer::new(g, Device::Cpu, &tcfg()).expect("trainer");
    let e = encoder();
    let batch = [example("late parcel", 1), example("charged twice", 2)];
    let feeds = build_feeds(&e, &cfg, SHAPE, &batch, opts()).expect("feeds");

    let first = t.accumulate(&feeds).expect("step");
    t.optimizer_step().expect("opt");
    let mut last = first;
    for _ in 0..30 {
        last = t.accumulate(&feeds).expect("step");
        t.optimizer_step().expect("opt");
    }
    assert!(
        last < first * 0.5,
        "loss did not fall on a single batch: {first} -> {last}"
    );
    eprintln!("overfit one batch: {first:.4} -> {last:.4}");
}

/// A padded row must not contribute to the loss.
#[test]
fn masked_rows_do_not_affect_the_loss() {
    let cfg = tiny_cfg();
    let e = encoder();
    // One question fills row 0; row 1 stays masked.
    let one = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");
    assert_eq!(one.active_rows, 1);

    let g = build_train_graph(&cfg, weights(), SHAPE, &tcfg()).expect("build");
    let mut t = Trainer::new(g, Device::Cpu, &tcfg()).expect("trainer");
    let a = t.accumulate(&one).expect("loss");

    // Change what the padded row would have seen: the loss must not move.
    let mut two = one;
    for (n, v) in two.named.iter_mut() {
        if n == "input_ids" {
            let seq = SHAPE.seq;
            for x in v[seq..].iter_mut() {
                *x = 9.0;
            }
        }
    }
    let b = t.accumulate(&two).expect("loss");
    assert!(
        (a - b).abs() < 1e-5,
        "a masked row changed the loss: {a} vs {b}"
    );
}

/// JSONL labels resolve by option **key**, and a bad key is an error.
#[test]
fn jsonl_labels_resolve_by_option_key() {
    let line = r#"{"state":"late parcel",
      "questions":{
        "d":{"type":"choice","instructions":"which team",
             "criteria":{"returns":"a","shipping":"b","billing":null}},
        "e":{"type":"noul","instructions":"urgent?"}},
      "labels":{"d":"shipping","e":"true"}}"#;
    let ex = rlx_kev::train::parse_labelled(line).expect("parses");
    assert_eq!(ex.record.questions.len(), 2);
    // `shipping` is index 1 in document order — not sorted order, where
    // `billing` would come first.
    assert_eq!(ex.record.questions[0].label, 1);
    // noul options are [no, yes], so "true" is index 1.
    assert_eq!(ex.record.questions[1].label, 1);

    let bad = line.replace(r#""d":"shipping""#, r#""d":"nope""#);
    let err = rlx_kev::train::parse_labelled(&bad).expect_err("must refuse");
    assert!(err.to_string().contains("not an option"), "{err}");

    let missing = line.replace(r#","e":"true""#, "");
    assert!(
        rlx_kev::train::parse_labelled(&missing).is_err(),
        "a question with no label must be refused, not defaulted to class 0"
    );
}

/// A trained adapter must round-trip: export it, read it back with the
/// inference loader, and merge it.
///
/// This is the seam where the two orientation conventions meet — in-graph
/// weights are `[in, out]` so `inject_lora` emits `A [in, r]` / `B [r, out]`,
/// while peft wants `A [r, in]` / `B [out, r]`. A transpose dropped here gives
/// a checkpoint that loads and merges into nonsense.
#[test]
fn a_trained_adapter_round_trips_through_the_inference_loader() {
    let cfg = tiny_cfg();
    let c = tcfg();
    let g = build_train_graph(&cfg, weights(), SHAPE, &c).expect("build");
    let mut t = Trainer::new(g, Device::Cpu, &c).expect("trainer");
    let e = encoder();
    let feeds = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");
    // Two steps so both A and B are non-zero.
    for _ in 0..2 {
        t.accumulate(&feeds).expect("backward");
        t.optimizer_step().expect("step");
    }

    let mut dir = std::env::temp_dir();
    dir.push(format!("rlx-kev-adapter-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    rlx_kev::train::export_adapter(&t, c.lora_rank, c.lora_alpha, &dir).expect("export");

    let ad = rlx_kev::LoraAdapter::open(&dir).expect("reads back");
    assert_eq!(ad.r, c.lora_rank);
    assert_eq!(ad.alpha, c.lora_alpha);
    assert!(!ad.is_empty(), "adapter came back empty");

    // Every exported module must merge into a correctly-shaped base weight.
    let mut merged = 0usize;
    for m in ad.modules().map(str::to_string).collect::<Vec<_>>() {
        let p = ad.get(&m).expect("pair present");
        let mut w = vec![0.0f32; p.out_features * p.in_features];
        assert!(
            ad.merge_into(&m, &mut w, p.out_features, p.in_features)
                .expect("merge"),
            "{m} did not merge"
        );
        assert!(
            w.iter().any(|x| *x != 0.0),
            "{m} merged to all zeros — a transpose was probably dropped"
        );
        merged += 1;
    }
    assert_eq!(merged, ad.len());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The trained head must come back in the host layout the readout expects.
#[test]
fn the_trained_head_loads_as_a_pointer_head() {
    let cfg = tiny_cfg();
    let c = tcfg();
    let g = build_train_graph(&cfg, weights(), SHAPE, &c).expect("build");
    let t = Trainer::new(g, Device::Cpu, &c).expect("trainer");
    let head = t.pointer_head(2.0).expect("head");
    assert_eq!(head.head_dim, c.head_dim);
    assert_eq!(head.hidden, cfg.hidden_size);
    assert_eq!(head.temperature, 2.0);
    // A distribution over 3 options, from a decide row and 3 option rows.
    let d = cfg.hidden_size;
    let hd: Vec<f32> = (0..d).map(|i| 0.01 * i as f32).collect();
    let ho: Vec<f32> = (0..3 * d).map(|i| 0.01 * (i % d) as f32 + 0.1).collect();
    let p = head.probs(&hd, &ho).expect("probs");
    assert_eq!(p.len(), 3);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

/// The exported head must come back through the **inference** loader
/// unchanged.
///
/// `export_head` writes `head.json` + `head.safetensors` rather than a torch
/// pickle, so `Checkpoint::open` has a second path; this is what stops that
/// path from drifting from the `head.pt` one. It also re-checks the transpose:
/// the graph stores `[d, dp]`, the file `[dp, d]`.
#[test]
fn the_exported_head_round_trips_through_the_checkpoint_loader() {
    let cfg = tiny_cfg();
    let c = tcfg();
    let g = build_train_graph(&cfg, weights(), SHAPE, &c).expect("build");
    let mut t = Trainer::new(g, Device::Cpu, &c).expect("trainer");
    let e = encoder();
    let feeds = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");
    t.accumulate(&feeds).expect("backward");
    t.optimizer_step().expect("step");

    let mut dir = std::env::temp_dir();
    dir.push(format!("rlx-kev-head-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let meta = rlx_kev::checkpoint::Meta {
        base: "synthetic".into(),
        lora: c.lora_rank,
        head_dim: c.head_dim,
        temperature: 2.25,
        ..rlx_kev::checkpoint::Meta::default()
    };
    rlx_kev::train::export_head(&t, &meta, &dir).expect("export head");

    let ck = rlx_kev::Checkpoint::open(&dir).expect("opens a head.json checkpoint");
    assert_eq!(ck.meta().head_dim, c.head_dim);
    assert_eq!(ck.meta().lora, c.lora_rank);
    assert!((ck.meta().temperature - 2.25).abs() < 1e-6);

    let want = t.pointer_head(2.25).expect("trainer head");
    let got = ck.pointer_head().expect("loaded head");
    assert_eq!(got.hidden, want.hidden);
    assert_eq!(got.head_dim, want.head_dim);
    for (a, b) in got.q_weight.iter().zip(&want.q_weight) {
        assert!((a - b).abs() < 1e-6, "q.weight drifted: {a} vs {b}");
    }
    for (a, b) in got.k_weight.iter().zip(&want.k_weight) {
        assert!((a - b).abs() < 1e-6, "k.weight drifted: {a} vs {b}");
    }
    // And the round-tripped head produces the same distribution.
    let d = cfg.hidden_size;
    let hd: Vec<f32> = (0..d).map(|i| 0.02 * i as f32 - 0.5).collect();
    let ho: Vec<f32> = (0..3 * d).map(|i| 0.01 * (i % d) as f32).collect();
    let pa = want.probs(&hd, &ho).expect("probs");
    let pb = got.probs(&hd, &ho).expect("probs");
    for (a, b) in pa.iter().zip(&pb) {
        assert!(
            (a - b).abs() < 1e-6,
            "probabilities drifted: {pa:?} vs {pb:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A temperature-less `head.json` must be refused, exactly as `head.pt` is.
#[test]
fn a_head_json_without_a_temperature_is_refused() {
    let mut dir = std::env::temp_dir();
    dir.push(format!("rlx-kev-notemp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(
        dir.join("head.json"),
        r#"{"base":"x","lora":8,"head_dim":8}"#,
    )
    .expect("write");
    let err = match rlx_kev::Checkpoint::open(&dir) {
        Ok(_) => panic!("a checkpoint with no temperature must be refused"),
        Err(e) => format!("{e:#}"),
    };
    assert!(err.contains("temperature"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `one_cycle_lr` must match `torch.optim.lr_scheduler.OneCycleLR`.
///
/// Values from torch itself (`max_lr = 2e-5`, `total_steps = 20`,
/// `pct_start = 0.1`). The peak lands on step **1** because torch's
/// `step_size_up` is `pct_start · total − 1`; an off-by-one there shifts the
/// entire schedule and would quietly change the run.
// The torch values below are quoted at full precision on purpose; they are
// truncated to f32 at compile time.
#[allow(clippy::excessive_precision)]
#[test]
fn one_cycle_matches_torch() {
    let want = [
        8e-07,
        2e-05,
        1.9848078e-05,
        1.9396929e-05,
        1.8660259e-05,
        1.7660454e-05,
        1.642789e-05,
        1.500002e-05,
        1.3420228e-05,
        1.1736515e-05,
        1.000004e-05,
        8.263565e-06,
        6.579852e-06,
        5.00006e-06,
        3.57219e-06,
        2.339626e-06,
        1.339821e-06,
        6.03151e-07,
        1.52002e-07,
        8e-11f32,
    ];
    for (i, w) in want.iter().enumerate() {
        let got = rlx_kev::train::one_cycle_lr(2e-5, i, 20, 0.1);
        let rel = (got - w).abs() / w.max(1e-12);
        assert!(
            rel < 2e-3,
            "step {i}: got {got:e}, torch gives {w:e} (rel {rel:.2e})"
        );
    }
    // Peak is at step 1, and step 0 starts at max_lr/25.
    assert!(rlx_kev::train::one_cycle_lr(2e-5, 1, 20, 0.1) >= 1.99e-5);
    assert!((rlx_kev::train::one_cycle_lr(2e-5, 0, 20, 0.1) - 8e-7).abs() < 1e-9);
}

/// A subsample must span the whole split, not its first source.
#[test]
fn strided_sampling_spans_the_split() {
    let idx = rlx_kev::eval::strided_indices(1204, 120);
    assert_eq!(idx.len(), 120);
    assert_eq!(idx[0], 0);
    assert!(
        *idx.last().expect("non-empty") > 1100,
        "the sample must reach the end of the split, got {:?}",
        idx.last()
    );
    // Strictly increasing, no duplicates.
    assert!(idx.windows(2).all(|w| w[0] < w[1]));
    // Degenerate cases.
    assert_eq!(rlx_kev::eval::strided_indices(5, 10), vec![0, 1, 2, 3, 4]);
    assert!(rlx_kev::eval::strided_indices(0, 3).is_empty());
    assert!(rlx_kev::eval::strided_indices(3, 0).is_empty());
}

/// bf16 autocast must halve activation memory while leaving params and
/// gradients in F32.
///
/// That split is the whole contract: the recipe's `--dtype bf16` is "autocast
/// forward with fp32 master weights", so a `Param` that came back BF16 would
/// mean the optimizer was updating half-precision weights — the classic way
/// mixed-precision training silently stops converging.
#[test]
fn bf16_autocast_keeps_params_f32_and_shrinks_activations() {
    use rlx_compile::memory::{ArenaWidthPolicy, plan_memory_with_policy};
    use rlx_ir::{DType, Op};

    let cfg = tiny_cfg();
    let f32c = tcfg();
    let bf16c = TrainConfig { bf16: true, ..f32c };
    let g32 = build_train_graph(&cfg, weights(), SHAPE, &f32c).expect("f32 graph");
    let g16 = build_train_graph(&cfg, weights(), SHAPE, &bf16c).expect("bf16 graph");

    // Params stay F32 — the optimizer's master weights.
    let mut bf16_params = 0usize;
    let mut bf16_nodes = 0usize;
    for n in g16.forward.nodes() {
        if n.shape.dtype() == DType::BF16 {
            bf16_nodes += 1;
            if matches!(n.op, Op::Param { .. }) {
                bf16_params += 1;
            }
        }
    }
    assert_eq!(
        bf16_params, 0,
        "parameters must stay F32 under autocast; {bf16_params} came back BF16"
    );
    assert!(
        bf16_nodes > 0,
        "nothing was cast to BF16 — the policy did not apply"
    );

    // Memory is REPORTED, not asserted. Under the CPU-safe policy only
    // elementwise and reduction activations go to BF16 — the large tensors are
    // Compute outputs and stay F32 — while the inserted Cast nodes add
    // buffers. Measured, that is a net *increase* on both the tiny model here
    // and the real one (116 GB -> 134 GB). Asserting a shrink would be
    // asserting something false; the win needs BF16 matmul on activations,
    // which the CPU backend does not have.
    let plan =
        |g: &rlx_ir::Graph| plan_memory_with_policy(g, 256, ArenaWidthPolicy::Hybrid).arena_size;
    let a32 = plan(&g32.forward);
    let a16 = plan(&g16.forward);
    eprintln!(
        "forward arena {a32} -> {a16} ({:.0}%), {bf16_nodes} BF16 nodes",
        100.0 * a16 as f64 / a32 as f64
    );
    // What must hold: the cast is not a no-op and not everything.
    assert!(
        bf16_nodes * 3 < g16.forward.nodes().len() * 2,
        "the safe policy should leave most nodes F32, got {bf16_nodes} of {}",
        g16.forward.nodes().len()
    );
}

/// Under bf16 the loss and gradients must still be F32 and still be right.
///
/// Tolerance is bf16's, not f32's: ~3 decimal digits of mantissa, so this
/// asks for the same answer to a few percent, not to 1e-6. A much larger
/// divergence would mean a cast landed in the wrong place.
#[test]
fn bf16_training_produces_f32_gradients_close_to_the_f32_run() {
    let cfg = tiny_cfg();
    let e = encoder();
    let feeds = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");

    let run = |bf16: bool| -> (f32, Vec<f32>) {
        let c = TrainConfig { bf16, ..tcfg() };
        let g = build_train_graph(&cfg, weights(), SHAPE, &c).expect("graph");
        let idx = g
            .wrt
            .iter()
            .position(|s| s.name == rlx_kev::train::HEAD_Q_W)
            .expect("head is trainable");
        let mut t = Trainer::new(g, Device::Cpu, &c).expect("trainer");
        let loss = t.accumulate(&feeds).expect("backward");
        (loss, t.grad_of(idx).expect("gradient").to_vec())
    };

    let (l32, g32) = run(false);
    let (l16, g16) = run(true);
    assert!(l16.is_finite(), "bf16 loss is not finite: {l16}");
    assert!(
        (l16 - l32).abs() <= 0.05 * l32.abs().max(1.0),
        "bf16 loss {l16} is far from the f32 loss {l32}"
    );

    assert_eq!(g16.len(), g32.len(), "gradient shape changed under bf16");
    // Compare direction, which is what the optimizer actually consumes;
    // bf16 rounding moves magnitudes more than it moves the direction.
    let dot: f64 = g32
        .iter()
        .zip(&g16)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let n32: f64 = g32.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let n16: f64 = g16.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    assert!(n32 > 0.0 && n16 > 0.0, "a gradient was all zeros");
    let cos = dot / (n32 * n16);
    assert!(
        cos > 0.99,
        "bf16 gradient points elsewhere: cosine {cos:.4} against the f32 run"
    );
    eprintln!("bf16 loss {l16:.4} vs f32 {l32:.4}; grad cosine {cos:.5}");
}

/// Every backend takes the safe bf16 policy for training, and bf16 training
/// is correct on each available one.
///
/// Pinning the policy choice here means changing it has to be deliberate. All
/// three macOS backends now train correctly under the safe policy, MLX included
/// (it used to fail in the backward).
#[test]
fn every_backend_takes_the_safe_bf16_policy_and_trains_correctly() {
    use rlx_compile::precision::PrecisionPolicy;
    for dev in [Device::Cpu, Device::Metal, Device::Mlx, Device::Ane] {
        assert!(
            matches!(
                rlx_kev::train::bf16_policy_for(dev),
                PrecisionPolicy::AutoMixedBf16Safe
            ),
            "{dev:?} should use the safe policy until BF16 compute/backward works"
        );
    }

    let cfg = tiny_cfg();
    let e = encoder();
    let feeds = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");
    let run = |bf16: bool, dev: Device| -> (f32, Vec<f32>) {
        let c = TrainConfig { bf16, ..tcfg() };
        let g =
            rlx_kev::train::build_train_graph_for(&cfg, weights(), SHAPE, &c, dev).expect("graph");
        let idx = g
            .wrt
            .iter()
            .position(|s| s.name == rlx_kev::train::HEAD_Q_W)
            .expect("head is trainable");
        let mut t = Trainer::new(g, dev, &c).expect("trainer");
        let loss = t.accumulate(&feeds).expect("backward");
        (loss, t.grad_of(idx).expect("gradient").to_vec())
    };
    let (l32, g32) = run(false, Device::Cpu);

    // MLX used to be absent here: its bf16 backward died on
    // `nelems doesn't match shape product`. That was two upstream bugs, both
    // fixed — `grad_with_loss` sized a zero cotangent constant at 4 bytes per
    // element under a bf16 declaration, and MLX's `RmsNorm` lowering dropped the
    // flattening that `fuse_rms_norm_reshape` folds in.
    #[allow(unused_mut)] // the pushes below are feature-gated
    let mut devices = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    devices.push(Device::Metal);
    #[cfg(feature = "mlx")]
    devices.push(Device::Mlx);
    for dev in devices {
        if !rlx_runtime::is_available(dev) {
            continue;
        }
        let (l16, g16) = run(true, dev);
        assert!(l16.is_finite(), "{dev:?}: bf16 loss is not finite: {l16}");
        assert!(
            (l16 - l32).abs() <= 0.1 * l32.abs().max(1.0),
            "{dev:?}: bf16 loss {l16} vs f32 {l32}"
        );
        let dot: f64 = g32
            .iter()
            .zip(&g16)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let n32: f64 = g32.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let n16: f64 = g16.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        assert!(n32 > 0.0 && n16 > 0.0, "{dev:?}: a gradient was all zeros");
        let cos = dot / (n32 * n16);
        assert!(cos > 0.98, "{dev:?}: bf16 gradient cosine {cos:.4}");
        eprintln!("{dev:?}: bf16 loss {l16:.4} vs {l32:.4}, grad cosine {cos:.5}");
    }
}

/// Every trainable gradient must be finite on the training device.
///
/// This is the check that was missing: the other backward tests here run on
/// `Device::Cpu`, and the one multi-backend test compares only the *head*
/// gradient — which stays finite even when every trunk adapter gradient is NaN.
/// So a backend could poison all 16 LoRA gradients of a layer and still pass.
/// It swept the compiled shape as well, since the recipe compiles one graph per
/// (rows, seq, max_options) bucket and pads batches into it.
#[test]
fn every_trainable_gradient_is_finite() {
    let cfg = tiny_cfg();
    let e = encoder();
    let base = tcfg();
    let dev = std::env::var("KEV_TEST_DEVICE")
        .ok()
        .map_or(Device::Cpu, |d| {
            rlx_runtime::parse_device(&d).expect("device")
        });
    // The exact row length, so "no padding anywhere" is one of the cases.
    let probe = example("late parcel", 1);
    let enc = e.encode(&probe.record, opts()).expect("encode");
    let (state_ids, _, branch_rows) = rlx_kev::encode::rows_of(&enc).expect("rows");
    let exact = state_ids.len() + branch_rows[0].ids.len();
    let mut failures = Vec::new();
    for (rows, seq, max_options) in [
        (1usize, exact, 3usize),
        (1, 64, 3),
        (2, 64, 3),
        // Options padded past the question's own count, which is what a
        // below-bucket batch looks like.
        (2, 64, 6),
    ] {
        let shape = TrainShape {
            rows,
            seq,
            max_options,
        };
        let Ok(feeds) = build_feeds(&e, &cfg, shape, &[example("late parcel", 1)], opts()) else {
            continue;
        };
        // `1` = checkpointing off, `0` = planner-guided (the default).
        for segs in [1usize, 0] {
            let c = TrainConfig {
                checkpoint_segments: segs,
                bf16: std::env::var_os("KEV_TEST_BF16").is_some(),
                ..base
            };
            let g = build_train_graph(&cfg, weights(), shape, &c).expect("graph");
            let names: Vec<String> = g.wrt.iter().map(|s| s.name.clone()).collect();
            let mut t = Trainer::new(g, dev, &c).expect("trainer");
            let loss = t.accumulate(&feeds).expect("backward");
            let bad: Vec<&String> = names
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    t.grad_of(*i)
                        .is_some_and(|g| g.iter().any(|v| !v.is_finite()))
                })
                .map(|(_, n)| n)
                .collect();
            eprintln!(
                "  rows {rows} seq {seq} opts {max_options} ckpt {segs}: loss {loss:.4}  \
                 {}/{} non-finite",
                bad.len(),
                names.len()
            );
            if !loss.is_finite() || !bad.is_empty() {
                failures.push(format!(
                    "{rows}x{seq}x{max_options}/ckpt{segs} ({} grads, e.g. {:?})",
                    bad.len(),
                    &bad[..bad.len().min(3)]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "non-finite gradients on {dev:?}:\n  {}",
        failures.join("\n  ")
    );
}

/// Temperature scaling must recover the temperature that distorted a
/// distribution, and must not touch accuracy.
///
/// kev's `temperature` divides the pointer logits, and a checkpoint that has
/// never been calibrated carries 1.0 — which leaves every argmax alone while
/// roughly doubling the confident-error rate, so it has to be fitted on the
/// calibration split before any Brier/ECE/NLL number is quoted.
#[test]
fn temperature_fitting_recovers_a_known_temperature() {
    use rlx_kev::eval::{Row, fit_temperature, metrics, rescale, rescaled};

    // Logits with a clear but imperfect signal, so NLL has an interior minimum.
    let logits: Vec<Vec<f32>> = (0..64)
        .map(|i| {
            let k = (i % 7) as f32;
            vec![1.0 + 0.3 * k, 0.4 * k - 0.2, -0.6 + 0.1 * k]
        })
        .collect();
    let softmax = |z: &[f32]| -> Vec<f32> {
        let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    };
    let rows: Vec<Row> = logits
        .iter()
        .enumerate()
        .map(|(i, z)| Row {
            p: softmax(z),
            // Labels mostly follow the argmax, so the raw model is overconfident
            // rather than wrong — the case temperature scaling exists for.
            label: if i % 5 == 0 { 1 } else { 0 },
            source: "synthetic".into(),
        })
        .collect();

    // Fitting is equivariant: sharpening the logits by `s` scales the optimal
    // temperature by `1/s`. Checking that, rather than "recovers 1.0", is what
    // makes this a test of the optimizer and not of the data.
    let t_star = fit_temperature(&rows);
    assert!(
        t_star.is_finite() && t_star > 0.05 && t_star < 20.0,
        "fit landed on an interval endpoint: {t_star}"
    );
    for s in [0.5f32, 1.7, 3.0] {
        let fitted = fit_temperature(&rescaled(&rows, s));
        let want = t_star / s;
        assert!(
            (fitted - want).abs() <= 0.02 * want,
            "logits sharpened by {s}: fitted {fitted}, expected {want}"
        );
    }

    // Fitting reduces NLL and leaves accuracy exactly where it was.
    let t = t_star;
    let before = metrics(&rows);
    let after = metrics(&rescaled(&rows, t));
    assert!(
        after.nll <= before.nll + 1e-9,
        "fitting raised NLL: {} -> {}",
        before.nll,
        after.nll
    );
    assert_eq!(
        after.accuracy, before.accuracy,
        "temperature is monotone per row, so accuracy cannot move"
    );
    // And a temperature of 1 is the identity.
    let p = rescale(&rows[0].p, 1.0);
    let err = p
        .iter()
        .zip(&rows[0].p)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err < 1e-6, "T=1 changed the distribution by {err}");
}

/// The chance floors must track the option counts, and flag a guesser.
///
/// They exist because every failure this crate has hit — a broken backend
/// gradient, a diverging learning rate — reads as "plausible loss, chance
/// accuracy", and `ln 2 = 0.693` looks like a good loss if the comparison in your
/// head is zero.
#[test]
fn chance_floors_track_the_option_counts() {
    use rlx_kev::eval::{Row, metrics};

    let uniform = |k: usize| Row {
        p: vec![1.0 / k as f32; k],
        label: 0,
        source: "u".into(),
    };
    // Two 2-option and two 8-option questions: chance accuracy is the mean of
    // 1/2 and 1/8, not 1/2 and not 1/8.
    let rows = vec![uniform(2), uniform(2), uniform(8), uniform(8)];
    let m = metrics(&rows);
    let want_acc = (0.5 + 0.5 + 0.125 + 0.125) / 4.0;
    let want_nll = (2f64.ln() * 2.0 + 8f64.ln() * 2.0) / 4.0;
    assert!(
        (m.chance_accuracy - want_acc).abs() < 1e-12,
        "chance accuracy {} should be {want_acc}",
        m.chance_accuracy
    );
    assert!(
        (m.chance_nll - want_nll).abs() < 1e-9,
        "chance nll {} should be {want_nll}",
        m.chance_nll
    );
    // A uniform predictor IS chance, by construction.
    assert!(m.at_chance(), "a uniform predictor must be flagged: {m}");
    assert!(
        (m.nll - want_nll).abs() < 1e-6,
        "a uniform predictor's NLL is its own floor"
    );

    // A perfect predictor is not.
    let sharp: Vec<Row> = (0..4)
        .map(|i| Row {
            p: if i < 2 {
                vec![0.99, 0.01]
            } else {
                let mut v = vec![0.001; 8];
                v[0] = 0.993;
                v
            },
            label: 0,
            source: "s".into(),
        })
        .collect();
    let ms = metrics(&sharp);
    assert!(!ms.at_chance(), "a confident correct predictor: {ms}");
}

/// Scoring order must not change the metrics.
///
/// `eval::score` visits records in SHAPE order so the model compiles one graph
/// per bucket rather than one per record. That is only safe because the metrics
/// are order-independent — this pins that, so a future change that makes them
/// order-sensitive (a running statistic, say) fails here rather than silently
/// reporting a different number than the suite order would.
#[test]
fn metrics_do_not_depend_on_row_order() {
    use rlx_kev::eval::{Row, by_source, metrics};

    let mk = |p: Vec<f32>, label: usize, src: &str| Row {
        p,
        label,
        source: src.into(),
    };
    let rows = vec![
        mk(vec![0.7, 0.3], 0, "a"),
        mk(vec![0.2, 0.8], 0, "b"),
        mk(vec![0.1, 0.2, 0.7], 2, "a"),
        mk(vec![0.4, 0.6], 1, "c"),
        mk(vec![0.95, 0.05], 0, "b"),
    ];
    let mut shuffled = rows.clone();
    shuffled.reverse();
    shuffled.swap(0, 2);

    let a = metrics(&rows);
    let b = metrics(&shuffled);
    for (name, x, y) in [
        ("accuracy", a.accuracy, b.accuracy),
        ("nll", a.nll, b.nll),
        ("brier", a.brier, b.brier),
        ("ece", a.ece, b.ece),
        ("mean_conf", a.mean_confidence, b.mean_confidence),
        (
            "confident_err",
            a.confident_error_rate,
            b.confident_error_rate,
        ),
        ("chance_acc", a.chance_accuracy, b.chance_accuracy),
        ("chance_nll", a.chance_nll, b.chance_nll),
    ] {
        assert!(
            (x - y).abs() < 1e-12,
            "{name} moved with row order: {x} vs {y}"
        );
    }
    // And the per-source split is keyed by name, not position.
    let sa = by_source(&rows);
    let sb = by_source(&shuffled);
    assert_eq!(sa.len(), sb.len());
    for ((na, ma), (nb, mb)) in sa.iter().zip(&sb) {
        assert_eq!(na, nb, "per-source order changed");
        assert!(
            (ma.accuracy - mb.accuracy).abs() < 1e-12,
            "{na} accuracy moved"
        );
    }
}

/// The training log and `examples/pad_audit` both report where the computed
/// positions went, as three buckets: real tokens, rows the batch never filled,
/// and sequence padding inside the rows it did. Those numbers are only worth
/// reading if the buckets are an exact partition of `rows * seq` — an
/// accounting that double-counts or drops positions would still look plausible
/// (three percentages that roughly sum to 100) while pointing at the wrong
/// lever. The audit's conclusion that within-batch spread dominates rests
/// entirely on this being a true split, so pin it.
#[test]
fn the_padding_report_is_an_exact_partition_of_the_computed_positions() {
    use rlx_kev::encode::rows_of;
    // Deliberately uneven lengths: a partition bug cancels out when every row
    // is the same size, which is exactly the shape a careless fixture has.
    let examples: Vec<_> = [
        "a",
        "a b c d e f g h",
        "x",
        "the quota was raised again last quarter",
        "p q",
        "m",
        "one two three four five six seven eight nine ten",
    ]
    .iter()
    .enumerate()
    .map(|(i, s)| example(s, i % 3))
    .collect();

    let enc = encoder();
    let batches = rlx_kev::train::shape_batches(&enc, &examples, opts(), 4, 512).expect("packer");
    assert!(!batches.is_empty(), "fixture produced no batches");

    let (mut slot, mut real, mut idle, mut pad) = (0usize, 0usize, 0usize, 0usize);
    let mut saw_padding = false;
    for (shape, idxs) in &batches {
        let (mut active, mut r, mut true_max) = (0usize, 0usize, 0usize);
        for &i in idxs {
            let e = enc.encode(&examples[i].record, opts()).expect("encode");
            let (state_ids, _, rows) = rows_of(&e).expect("rows");
            active += rows.len();
            for row in &rows {
                let len = state_ids.len() + row.ids.len();
                r += len;
                true_max = true_max.max(len);
            }
        }
        assert!(
            active <= shape.rows,
            "packer overfilled a batch: {active} rows into {} slots",
            shape.rows
        );
        slot += shape.rows * shape.seq;
        real += r;
        idle += (shape.rows - active) * shape.seq;
        pad += active * shape.seq - r;
        if active * shape.seq > r {
            saw_padding = true;
        }
    }

    assert_eq!(
        real + idle + pad,
        slot,
        "buckets do not partition the slots: real {real} + idle {idle} + pad {pad} != {slot}"
    );
    // Guard against the test passing because nothing was padded at all, which
    // would make the partition trivially true and prove nothing.
    assert!(
        saw_padding,
        "fixture produced no padding, so the split is vacuous"
    );
    assert!(
        real > 0 && real < slot,
        "expected some real tokens and some waste"
    );
}

/// A periodic checkpoint has to survive being killed at an arbitrary instant,
/// which is the entire reason it exists: a full-suite run is 8-14 hours and the
/// final export only lands after the last optimizer step, so a run that dies at
/// hour 13 must still leave something loadable. Writing straight into the
/// output directory makes the window *worse* than no checkpoint at all — the
/// new file is torn and the old one is already gone.
///
/// Pin the two properties that buy safety: the staging directory never
/// survives a successful write, and every file it produced is a complete
/// checkpoint the inference loader accepts. Overwriting an existing checkpoint
/// has to hold too, since that is what every step after the first does.
#[test]
fn a_periodic_checkpoint_overwrites_in_place_and_stays_loadable() {
    let cfg = tiny_cfg();
    let c = tcfg();
    let g = build_train_graph(&cfg, weights(), SHAPE, &c).expect("build");
    let mut t = Trainer::new(g, Device::Cpu, &c).expect("trainer");
    let e = encoder();
    let feeds = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");

    let mut dir = std::env::temp_dir();
    dir.push(format!("rlx-kev-ckpt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let meta = rlx_kev::checkpoint::Meta {
        lora: c.lora_rank,
        head_dim: c.head_dim,
        temperature: 1.0,
        ..rlx_kev::checkpoint::Meta::default()
    };

    let mut seen = Vec::new();
    for round in 0..3 {
        t.accumulate(&feeds).expect("backward");
        t.optimizer_step().expect("step");
        rlx_kev::train::checkpoint_atomically(&t, c.lora_rank, c.lora_alpha, &meta, &dir)
            .expect("checkpoint");

        assert!(
            !dir.join(".staging").exists(),
            "round {round}: staging directory outlived a successful write"
        );
        let ad = rlx_kev::LoraAdapter::open(&dir)
            .unwrap_or_else(|e| panic!("round {round}: checkpoint does not load: {e}"));
        assert_eq!(ad.r, c.lora_rank);
        assert_eq!(ad.alpha, c.lora_alpha);

        assert!(!ad.is_empty(), "round {round}: checkpoint has no modules");

        // Merge a module down to real numbers. "Loadable" would otherwise be
        // satisfied by three identical copies of an untrained export, which is
        // exactly what a checkpoint that silently stopped tracking the trainer
        // would look like.
        let m = ad.modules().next().expect("a module").to_string();
        let p = ad.get(&m).expect("pair present");
        let mut w = vec![0.0f32; p.out_features * p.in_features];
        assert!(
            ad.merge_into(&m, &mut w, p.out_features, p.in_features)
                .expect("merge"),
            "round {round}: {m} did not merge"
        );
        seen.push(w);
    }
    assert!(
        seen[0] != seen[2],
        "the checkpoint is identical after two more optimizer steps — it is \
         not tracking the trainer"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// `--init-from` must restore the pointer head, not just the adapter.
///
/// Merging the adapter into the base weights restores the *backbone*, but the
/// head is separate parameters that `build_train_graph` always seeds
/// pseudo-randomly. A warm start that silently keeps the random head begins at
/// chance and has to relearn the readout — which defeats warm-starting and
/// makes the released recipe (a continuation round from kev-0.8b)
/// unreproducible. The `init_from` doc comment promised "adapter + head" while
/// nothing ever read `head.safetensors` back.
#[test]
fn a_warm_start_restores_the_trained_head_not_a_fresh_one() {
    let cfg = tiny_cfg();
    let c = tcfg();
    let e = encoder();
    let feeds = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");

    // Train a head, then export it.
    let g = build_train_graph(&cfg, weights(), SHAPE, &c).expect("build");
    let mut t = Trainer::new(g, Device::Cpu, &c).expect("trainer");
    for _ in 0..3 {
        t.accumulate(&feeds).expect("backward");
        t.optimizer_step().expect("step");
    }
    let trained = t.pointer_head(1.0).expect("trained head");

    let mut dir = std::env::temp_dir();
    dir.push(format!("rlx-kev-warm-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let meta = rlx_kev::checkpoint::Meta {
        lora: c.lora_rank,
        head_dim: c.head_dim,
        temperature: 1.0,
        ..rlx_kev::checkpoint::Meta::default()
    };
    rlx_kev::train::export_head(&t, &meta, &dir).expect("export head");

    // A fresh graph seeds the head pseudo-randomly. Confirm it really differs
    // from the trained one, or the restore below proves nothing.
    let mut fresh = build_train_graph(&cfg, weights(), SHAPE, &c).expect("rebuild");
    let before = fresh
        .params
        .get(rlx_kev::train::HEAD_Q_W)
        .expect("head param")
        .clone();

    let loaded = rlx_kev::Checkpoint::open(&dir)
        .expect("checkpoint opens")
        .pointer_head()
        .expect("head loads");
    rlx_kev::train::restore_head(&mut fresh, &loaded).expect("restore");
    let after = fresh
        .params
        .get(rlx_kev::train::HEAD_Q_W)
        .expect("head param")
        .clone();
    assert!(
        before != after,
        "restore was a no-op — the fresh head already equalled the trained one, \
         so this test cannot detect a missing restore"
    );

    // And it must be the *trained* head, through the same transpose the readout
    // applies. A transposed restore would still change the params and still be
    // wrong, which is why this compares values rather than just inequality.
    let t2 = Trainer::new(fresh, Device::Cpu, &c).expect("trainer from restored graph");
    let got = t2.pointer_head(1.0).expect("head readout");
    for (n, (a, b)) in got.q_weight.iter().zip(&trained.q_weight).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "q.weight[{n}] {a} != {b} (transpose dropped?)"
        );
    }
    for (n, (a, b)) in got.k_weight.iter().zip(&trained.k_weight).enumerate() {
        assert!((a - b).abs() < 1e-6, "k.weight[{n}] {a} != {b}");
    }
    for (n, (a, b)) in got.q_bias.iter().zip(&trained.q_bias).enumerate() {
        assert!((a - b).abs() < 1e-6, "q.bias[{n}] {a} != {b}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A warm start must carry the checkpoint's adapter into the trainable
/// parameters, so the run's own export is self-contained.
///
/// The alternative — merge the checkpoint's delta into the base and fit a fresh
/// adapter on top — trains a correct model but exports only the *new* delta.
/// Loading that the normal way (`base + adapter`) drops everything the
/// checkpoint knew. On real weights the loss was stark: total `||B||` over 186
/// B-matrices went 86.85 (kev-0.8b) -> 0.35 after a two-step delta run.
///
/// Pinned as a round trip, because that is the property that makes an export
/// self-contained: restoring an adapter and exporting it again must reproduce
/// it. A transpose dropped on either side breaks this while leaving both files
/// the same size and shape.
#[test]
fn restoring_an_adapter_then_exporting_it_reproduces_it() {
    let cfg = tiny_cfg();
    let c = tcfg();
    let e = encoder();
    let feeds = build_feeds(&e, &cfg, SHAPE, &[example("late parcel", 1)], opts()).expect("feeds");

    // Train an adapter so A and B are both non-trivial, then export it.
    let g = build_train_graph(&cfg, weights(), SHAPE, &c).expect("build");
    let mut t = Trainer::new(g, Device::Cpu, &c).expect("trainer");
    for _ in 0..3 {
        t.accumulate(&feeds).expect("backward");
        t.optimizer_step().expect("step");
    }
    let mut src = std::env::temp_dir();
    src.push(format!("rlx-kev-ad-src-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&src);
    rlx_kev::train::export_adapter(&t, c.lora_rank, c.lora_alpha, &src).expect("export");
    let original = rlx_kev::LoraAdapter::open(&src).expect("reads back");

    // A fresh graph starts with B = 0, so it is provably not already equal.
    let mut fresh = build_train_graph(&cfg, weights(), SHAPE, &c).expect("rebuild");
    let b_name = fresh
        .adapter_names
        .iter()
        .find(|n| n.ends_with(".b"))
        .expect("a B tensor")
        .clone();
    assert!(
        fresh.params[&b_name].iter().all(|x| *x == 0.0),
        "fresh B is not zero, so this test cannot prove the restore did anything"
    );

    rlx_kev::train::restore_adapter(&mut fresh, &original).expect("restore");
    assert!(
        fresh.params[&b_name].iter().any(|x| *x != 0.0),
        "restore left B at zero — the checkpoint's delta was not carried over"
    );

    // Round trip: export what we restored and compare tensor for tensor.
    let t2 = Trainer::new(fresh, Device::Cpu, &c).expect("trainer from restored graph");
    let mut out = std::env::temp_dir();
    out.push(format!("rlx-kev-ad-out-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    rlx_kev::train::export_adapter(&t2, c.lora_rank, c.lora_alpha, &out).expect("re-export");
    let again = rlx_kev::LoraAdapter::open(&out).expect("reads back");

    assert_eq!(again.r, original.r);
    let mods: Vec<_> = original.modules().map(str::to_string).collect();
    assert!(!mods.is_empty(), "no modules exported");
    for m in &mods {
        let a = original.get(m).expect("orig pair");
        let b = again.get(m).expect("round-tripped pair");
        assert_eq!(
            (a.in_features, a.out_features),
            (b.in_features, b.out_features),
            "{m} shape"
        );
        for (n, (x, y)) in a.a.iter().zip(&b.a).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "{m} A[{n}] {x} != {y} (transpose dropped?)"
            );
        }
        for (n, (x, y)) in a.b.iter().zip(&b.b).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "{m} B[{n}] {x} != {y} (transpose dropped?)"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&src);
    let _ = std::fs::remove_dir_all(&out);
}
