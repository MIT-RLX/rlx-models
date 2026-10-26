//! Shared pieces for building a **pre-training objective** onto a model trunk.
//!
//! Most ports in this workspace carry a `train.rs` that fits a supervised head
//! over frozen features and a note saying the model's real pre-training objective
//! is "out of scope". Implementing those objectives needs the same three things
//! every time — select the masked positions, build a loss into the graph, and work
//! out which parameters may be differentiated — so they live here rather than
//! being copied per crate.
//!
//! # Two things that are not obvious and cost a debugging session each
//!
//! **Not every `Op::Param` is trainable.** `ModelFlow::attn_mask_ones` declares the
//! all-ones attention mask as a *parameter*. It is reachable from the loss (through
//! `Op::Attention`) yet categorically non-differentiable, and
//! `rlx_autodiff::grad_with_loss` **panics** on a request for it — "no gradient
//! flowed to %N" — rather than returning zero. [`trainable_params`] filters it out.
//!
//! **Masked positions must be distinguishable from one another.** If every masked
//! slot receives the same mask token and the trunk carries no positional signal,
//! those slots are identical inputs to a permutation-equivariant attention stack,
//! so the best possible reconstruction is the per-feature *mean* of the targets.
//! That is a hard floor no learning rate moves: measured on `rlx-brant`, the loss
//! sat at exactly `0.037616` for a 3-position mask — the mean-predictor MSE to six
//! decimals — identically at `1e-3` and `1e-2`. [`add_sinusoidal_positions`] is the
//! cheap fix when the trunk does not add positions itself.
//!
//! **A masked objective needs a real batch.** This is the one that cost the most.
//! Reconstructing a masked slot means routing another position's content through
//! attention, and that is a structural solution the optimizer has to find; a
//! per-token one is always available and much easier. With one window per step the
//! gradient noise hides the first and leaves the second, and the run *underfits* —
//! on real EEG (1600 Bonn windows, `seq=64`, 16-sample patches, `dim=64`) the train
//! loss stalled at 0.98 against a mean-predictor 1.007, i.e. 2% of the variance,
//! while the identical graph autoencoded a **visible** patch to 0.0018 (99.8%).
//! Stacking 8 windows per step and changing nothing else took it to 17% within 15
//! epochs and still falling. Before blaming the objective, the trunk or the port:
//!
//! * Run the **unmasked control** — score the same positions without blanking them.
//!   If that does not approach zero the problem is the path, not the routing, and no
//!   amount of masked-objective tuning will find it.
//! * Compare against a **least-squares map from the neighbouring positions** fit on
//!   the same split. Z-scored data puts the mean predictor at ~1.0, so any loss just
//!   under 1.0 reads as progress; the linear number says how much is there to get.
//!   On the Bonn split it is 0.590 — 41% of the variance — so 2% was not "EEG is
//!   unpredictable", it was a bug-shaped result.

use std::collections::HashSet;

use rlx_ir::infer::GraphExt as _;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const F32: DType = DType::F32;

/// A parameter that may be differentiated: its name and its graph node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trainable {
    pub name: String,
    pub node: NodeId,
}

/// The parameters `loss` actually depends on and that autodiff can differentiate.
///
/// Two filters, both necessary:
///
/// * **reachability** — a parameter outside the loss cone gets no gradient, and
///   asking for one panics;
/// * **attention mask operands** — see the module docs; a mask declared as a
///   `Param` is reachable but has no VJP.
///
/// Sorted and de-duplicated by name so a caller's parameter order is stable.
pub fn trainable_params(graph: &Graph, loss: NodeId) -> Vec<Trainable> {
    let mut reachable = HashSet::new();
    let mut stack = vec![loss];
    while let Some(id) = stack.pop() {
        if !reachable.insert(id) {
            continue;
        }
        for &i in &graph.node(id).inputs {
            stack.push(i);
        }
    }
    let mask_operands: HashSet<NodeId> = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Attention { .. }))
        .flat_map(|n| n.inputs.iter().copied())
        .filter(|id| matches!(graph.node(*id).op, Op::Param { .. }))
        .collect();

    let mut out: Vec<Trainable> = graph
        .nodes()
        .iter()
        .filter(|n| reachable.contains(&n.id) && !mask_operands.contains(&n.id))
        .filter_map(|n| match &n.op {
            Op::Param { name, .. } => Some(Trainable {
                name: name.clone(),
                node: n.id,
            }),
            _ => None,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out.dedup_by(|a, b| a.name == b.name);
    // `RLX_PRETRAIN_DEBUG=1` lists what will actually be updated. The failure this
    // exists for is silent: a parameter excluded here still has a value, still runs
    // in the forward, and the loss still goes down a little, so a frozen attention
    // stack looks like a weak objective rather than a bug.
    if rlx_ir::env::var_result("RLX_PRETRAIN_DEBUG").is_ok_and(|v| v != "0") {
        let excluded: Vec<&str> = graph
            .nodes()
            .iter()
            .filter(|n| reachable.contains(&n.id) && mask_operands.contains(&n.id))
            .filter_map(|n| match &n.op {
                Op::Param { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        eprintln!(
            "trainable_params: {} trainable {:?}",
            out.len(),
            out.iter().map(|t| t.name.as_str()).collect::<Vec<_>>()
        );
        eprintln!(
            "trainable_params: {} excluded (Attention operands) {excluded:?}",
            excluded.len()
        );
    }
    out
}

/// A one-hot `[m, n_pos]` matrix selecting `positions` out of a flattened trunk.
///
/// Used with [`select_rows`] instead of a gather: a selection matmul is
/// differentiable with the ops a forward already uses, and does not depend on
/// gather's backward for index tensors.
pub fn selection_matrix(positions: &[usize], n_pos: usize) -> Vec<f32> {
    let mut v = vec![0f32; positions.len() * n_pos];
    for (row, &p) in positions.iter().enumerate() {
        if p < n_pos {
            v[row * n_pos + p] = 1.0;
        }
    }
    v
}

/// `sel [m, n_pos] @ x [n_pos, width]` — the rows named by a [`selection_matrix`].
///
/// `x` is reshaped to `[n_pos, width]` first, so a `[batch, seq, width]` trunk
/// output can be passed directly.
pub fn select_rows(
    g: &mut Graph,
    x: NodeId,
    sel_input: &str,
    m: usize,
    n_pos: usize,
    width: usize,
) -> NodeId {
    let flat = g.reshape_(x, vec![n_pos as i64, width as i64]);
    let sel = g.input(sel_input, Shape::new(&[m, n_pos], F32));
    g.mm(sel, flat)
}

/// L2-normalise along `axis`, with a floor so an all-zero row cannot divide by 0.
pub fn l2_normalize(g: &mut Graph, x: NodeId, axis: usize) -> NodeId {
    let sq = g.mul(x, x);
    let s = g.sum(sq, vec![axis], true);
    let n = g.sqrt(s);
    let floor = g.full(&[1], 1e-12, F32);
    let n = g.add(n, floor);
    g.div(x, n)
}

/// InfoNCE over a candidate set, as a scalar loss node.
///
/// `query` is `[m, width]`; `cands_input` names an `[m, k1, width]` graph input
/// whose **index 0 is the positive** and whose remaining `k1 - 1` entries are
/// distractors. Similarity is cosine, scaled by `1 / temperature`, and the loss is
/// the mean negative log-softmax of the positive.
///
/// Chance level is `ln(k1)`; a loss that starts far from it means the candidate set
/// or the temperature is wired wrong, which is worth asserting in a test.
pub fn infonce_loss(
    g: &mut Graph,
    query: NodeId,
    cands_input: &str,
    m: usize,
    k1: usize,
    width: usize,
    temperature: f32,
) -> NodeId {
    let cands = g.input(cands_input, Shape::new(&[m, k1, width], F32));
    infonce_loss_nodes(g, query, cands, m, k1, width, temperature)
}

/// [`infonce_loss`] against a candidate **node** rather than a host-supplied input.
///
/// Needed when the candidates are produced inside the graph — as they are once the feature
/// encoder is differentiated and the latents no longer exist on the host. The caller is
/// responsible for putting `cands` through `stop_gradient` if the targets should not be
/// trained towards the query, which is what wav2vec 2.0 does: without it the cheapest way
/// to cut the loss is to collapse the targets onto each other rather than to predict them.
pub fn infonce_loss_nodes(
    g: &mut Graph,
    query: NodeId,
    cands: NodeId,
    m: usize,
    k1: usize,
    width: usize,
    temperature: f32,
) -> NodeId {
    // `k1` is not needed to build the graph — it is checked, because a candidate tensor
    // laid out `[m, width, k1]` or carrying the wrong candidate count still multiplies and
    // reduces to a finite loss that trains towards the wrong thing.
    let got: Vec<usize> = g
        .shape(cands)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    assert_eq!(
        got,
        vec![m, k1, width],
        "infonce candidates must be [m, k1, width]"
    );
    let q = l2_normalize(g, query, 1);
    let c = l2_normalize(g, cands, 2);

    // sum(c * q[:, None, :], -1) -> [m, k1]
    let qb = g.reshape_(q, vec![m as i64, 1, width as i64]);
    let prod = g.mul(c, qb);
    let sim = g.sum(prod, vec![2], false);

    let inv_t = g.full(&[1, 1], 1.0 / temperature, F32);
    let logits = g.mul(sim, inv_t);

    // Max-subtracted logsumexp: the exponential must not overflow.
    let mx = g.add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes: vec![1],
            keep_dim: true,
        },
        vec![logits],
        Shape::new(&[m, 1], F32),
    );
    let shifted = g.sub(logits, mx);
    let ex = g.exp(shifted);
    let sum_ex = g.sum(ex, vec![1], true);
    let lse = g.log(sum_ex);
    let pos = g.narrow_(shifted, 1, 0, 1);
    let nll = g.sub(lse, pos);
    let flat = g.reshape_(nll, vec![m as i64]);
    g.mean(flat, vec![0], false)
}

/// Mean squared error between `pred [m, width]` and a `target_input` of the same
/// shape, as a scalar loss node.
pub fn masked_mse_loss(
    g: &mut Graph,
    pred: NodeId,
    target_input: &str,
    m: usize,
    width: usize,
) -> NodeId {
    let target = g.input(target_input, Shape::new(&[m, width], F32));
    let d = g.sub(pred, target);
    let sq = g.mul(d, d);
    let flat = g.reshape_(sq, vec![(m * width) as i64]);
    g.mean(flat, vec![0], false)
}

/// Build the candidate tensor for [`infonce_loss`] from a bank of vectors.
///
/// Index 0 of each row is the vector at that masked position; the rest are drawn
/// from **other** positions. Sampling the same position as a distractor would put
/// the answer in the denominator twice and cap the achievable loss.
pub fn infonce_candidates(
    bank: &[f32],
    n_pos: usize,
    width: usize,
    positions: &[usize],
    n_distractors: usize,
    seed: u64,
) -> Vec<f32> {
    let k1 = n_distractors + 1;
    let mut out = vec![0f32; positions.len() * k1 * width];
    let mut st = seed | 1;
    let mut next = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    for (row, &p) in positions.iter().enumerate() {
        let base = row * k1 * width;
        out[base..base + width].copy_from_slice(&bank[p * width..(p + 1) * width]);
        for k in 1..k1 {
            let mut q = (next() as usize) % n_pos.max(1);
            while q == p && n_pos > 1 {
                q = (next() as usize) % n_pos;
            }
            let dst = base + k * width;
            out[dst..dst + width].copy_from_slice(&bank[q * width..(q + 1) * width]);
        }
    }
    out
}

/// Add a sinusoidal positional encoding in place to a `[n_pos, width]` buffer.
///
/// `seq` is the period (positions repeat per batch item). See the module docs for
/// why this is load-bearing for a masked objective rather than a refinement.
pub fn add_sinusoidal_positions(x: &mut [f32], n_pos: usize, seq: usize, width: usize, scale: f32) {
    let half = (width / 2).max(1);
    for p in 0..n_pos {
        let t = (p % seq.max(1)) as f32;
        for d in 0..width {
            let (i, phase) = if d < half {
                (d, 0.0f32)
            } else {
                (d - half, std::f32::consts::FRAC_PI_2)
            };
            let freq = 1.0f32 / 10_000f32.powf(2.0 * i as f32 / width as f32);
            x[p * width + d] += scale * (t * freq + phase).sin();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_picks_the_named_rows() {
        let sel = selection_matrix(&[0, 2], 3);
        assert_eq!(sel, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_distractor_is_never_the_positive_itself() {
        // Index 0 is the positive; with 3 positions and 2 distractors, neither may
        // equal it — otherwise the answer sits in the denominator twice.
        let bank: Vec<f32> = (0..3 * 2).map(|i| i as f32).collect();
        let c = infonce_candidates(&bank, 3, 2, &[1], 2, 99);
        assert_eq!(&c[0..2], &[2.0, 3.0], "positive must be position 1");
        for k in 1..3 {
            assert_ne!(
                &c[k * 2..k * 2 + 2],
                &[2.0, 3.0],
                "distractor {k} is the positive"
            );
        }
    }

    /// A trunk that mixes across positions but cannot tell them apart.
    ///
    /// `out[p] = x[p] + mean(x)`, which is permutation-equivariant — exactly the
    /// property that makes identical masked rows unrecoverable. No attention op is
    /// needed to reproduce the floor, and building one here would test rlx-flow rather
    /// than this driver.
    fn mean_mixer(n_pos: usize, width: usize) -> (Graph, NodeId) {
        let mut g = Graph::new("mean-mixer");
        let x = g.input("hidden", Shape::new(&[n_pos, width], F32));
        let m = g.mean(x, vec![0], true);
        let out = g.add(x, m);
        g.set_outputs(vec![out]);
        (g, out)
    }

    /// Rows with a strong per-position profile, plus per-window jitter, so predicting
    /// *which* row is masked is worth much more than predicting the average row.
    fn positional_windows(n: usize, n_pos: usize, width: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|w| {
                let mut v = vec![0f32; n_pos * width];
                for p in 0..n_pos {
                    for d in 0..width {
                        let jitter = ((w * 31 + p * 7 + d) % 17) as f32 / 17.0 - 0.5;
                        v[p * width + d] = (p as f32 * 0.9 + d as f32 * 0.3).sin() + 0.05 * jitter;
                    }
                }
                v
            })
            .collect()
    }

    /// `position_scale: None` floors the objective; `Some` removes the floor.
    ///
    /// This is the defect the field exists for, and it is not hypothetical — the
    /// driver shipped without it and `rlx-brainrvq` floored at a held-out 0.178
    /// against a mean predictor of 0.221, with 2.7x the data and 2.5x the epochs
    /// moving it by 0.016. Zeroed masked rows are identical inputs, so a
    /// permutation-equivariant trunk can only emit one average row for all of them.
    #[test]
    fn without_positions_the_masked_loss_floors() {
        let (n_pos, width) = (8usize, 4usize);
        let train = positional_windows(12, n_pos, width);
        let held = positional_windows(4, n_pos, width);
        let run = |scale: Option<f32>| -> f32 {
            let (g, out) = mean_mixer(n_pos, width);
            let cfg = BatchedConfig {
                batch: 1,
                seq: n_pos,
                width,
                position_scale: scale,
                mask_positions: vec![1, 3, 5],
                input_name: "hidden".to_string(),
                epochs: 150,
                lr: 5e-2,
                seed: 3,
                eval_every: 25,
            };
            // Plain SGD: `rlx-optim` is deliberately not a dependency here (the model
            // crates patch it to a path while this crate resolves the registry copy),
            // and the claim under test is about the input, not the optimizer.
            let mut update = |_: &str, _: &[usize], p: &mut [f32], grad: &[f32]| {
                for (v, g) in p.iter_mut().zip(grad) {
                    *v -= cfg.lr * g;
                }
            };
            let rep = train_masked_objective(
                g,
                out,
                std::collections::HashMap::new(),
                &cfg,
                Objective::MaskedMse,
                &train,
                &held,
                None,
                &mut update,
            )
            .expect("driver run");
            rep.best_heldout().expect("a held-out curve").1
        };
        let floored = run(None);
        let with_pos = run(Some(1.0));
        eprintln!("[driver] held-out without positions {floored:.5}, with {with_pos:.5}");
        assert!(
            with_pos < floored * 0.6,
            "positions did not remove the floor: {floored:.5} -> {with_pos:.5}"
        );
    }

    /// The codebook objective refuses to invent labels.
    ///
    /// HuBERT's targets come from clustering features the trunk is not fed. Deriving them
    /// from the input instead would make the task solvable by copying, so the driver has
    /// no fallback and must say so rather than guess.
    #[test]
    fn codebook_ce_requires_labels() {
        let (n_pos, width) = (8usize, 4usize);
        let batches = positional_windows(4, n_pos, width);
        let (g, out) = mean_mixer(n_pos, width);
        let cfg = BatchedConfig {
            batch: 1,
            seq: n_pos,
            width,
            position_scale: Some(1.0),
            mask_positions: vec![1, 3],
            input_name: "hidden".to_string(),
            epochs: 1,
            lr: 1e-2,
            seed: 3,
            eval_every: 0,
        };
        let mut update = |_: &str, _: &[usize], _: &mut [f32], _: &[f32]| {};
        let err = train_masked_objective(
            g,
            out,
            std::collections::HashMap::new(),
            &cfg,
            Objective::MaskedCodebookCe {
                n_classes: 4,
                proj_dim: 4,
                temperature: 0.1,
            },
            &batches,
            &batches,
            None,
            &mut update,
        )
        .expect_err("missing labels must be refused");
        assert!(err.contains("class_targets"), "unhelpful error: {err}");
    }

    /// k-means seeding must not leave an entry unused.
    ///
    /// Random seeding can put two centres in the same cluster and leave another with no
    /// members, and an empty cluster means one class is unpredictable for the whole run —
    /// or, for a VQ codebook, a dead entry. Farthest-point seeding is what prevents it.
    #[test]
    fn kmeans_uses_every_cluster_it_is_given() {
        // Four well-separated blobs of three points each.
        let width = 2usize;
        let mut data = Vec::new();
        for (cx, cy) in [(0.0f32, 0.0f32), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)] {
            for j in 0..3 {
                data.push(cx + j as f32 * 0.05);
                data.push(cy - j as f32 * 0.05);
            }
        }
        let ids = kmeans_labels(&data, width, 4, 20, 7);
        let distinct: HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            4,
            "only {} of 4 clusters used: {ids:?}",
            distinct.len()
        );
        let centres = kmeans_centroids(&data, width, 4, 20, 7);
        assert_eq!(centres.len(), 4 * width);
        assert!(centres.iter().all(|v| v.is_finite()));
    }

    /// The three constant-predictor floors, each against its own definition.
    ///
    /// These exist because the cross-entropy one was wrong in shipped code:
    /// `-ln(p_majority)` was reported as a floor, and it is not any predictor's
    /// cross-entropy — it read 0.626 where the true floor was 1.288, which turned a
    /// 34%-better result into "does not beat the floor".
    #[test]
    fn constant_predictor_floors_are_the_optimal_constants() {
        // Cross-entropy: H(p), strictly between -ln(p_max) and ln(k) when imbalanced.
        let labels = vec![
            vec![0u32; 130],
            vec![1; 59],
            vec![2; 535],
            vec![3; 79],
            vec![4; 197],
        ];
        let h = prior_predictor_ce(&labels, 5);
        let total: usize = labels.iter().map(|b| b.len()).sum();
        let p_max = labels.iter().map(|b| b.len()).max().unwrap() as f32 / total as f32;
        assert!((h - 1.2879).abs() < 1e-3, "H(p) = {h}");
        assert!(
            -p_max.ln() < h && h < 5f32.ln(),
            "H(p) {h} not between the two wrong bars"
        );
        // An absent class must not contribute `0·ln 0` and make the whole thing NaN.
        assert!(prior_predictor_ce(&[vec![2u32; 8]], 5).abs() < 1e-6);
        assert!(prior_predictor_ce(&[], 5).is_nan());

        // MSE: the variance about the mean, not `mean(y²)`. Shifting the data by a
        // constant must leave it unchanged, which is exactly what predicting 0 fails.
        let y: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let shifted: Vec<f32> = y.iter().map(|v| v + 10.0).collect();
        let (a, b) = (constant_predictor_mse(&y), constant_predictor_mse(&shifted));
        assert!(
            (a - b).abs() < 1e-3,
            "MSE floor moved under a shift: {a} vs {b}"
        );
        let naive: f32 = shifted.iter().map(|v| v * v).sum::<f32>() / shifted.len() as f32;
        assert!(naive > b * 10.0, "predicting zero should be far worse here");

        // MAE: about the median. On a skewed sample the median beats the mean.
        let skew: Vec<f32> = (0..64).map(|i| if i < 60 { 0.0 } else { 100.0 }).collect();
        let mae = constant_predictor_mae(&skew);
        let mean = skew.iter().sum::<f32>() / skew.len() as f32;
        let mae_about_mean: f32 =
            skew.iter().map(|x| (x - mean).abs()).sum::<f32>() / skew.len() as f32;
        assert!(
            mae < mae_about_mean,
            "median MAE {mae} should beat mean MAE {mae_about_mean}"
        );
        assert!(constant_predictor_mae(&[]).is_nan());
    }

    #[test]
    fn positions_become_distinguishable() {
        // The floor this exists to remove: identical slots -> identical encodings.
        let (n, seq, w) = (4usize, 4usize, 8usize);
        let mut x = vec![0f32; n * w];
        add_sinusoidal_positions(&mut x, n, seq, w, 0.1);
        for a in 0..n {
            for b in (a + 1)..n {
                let same = (0..w).all(|d| (x[a * w + d] - x[b * w + d]).abs() < 1e-9);
                assert!(!same, "positions {a} and {b} encode identically");
            }
        }
    }
}

// ───────────────────────── multi-batch pre-training ─────────────────────────

/// A named tensor bound to a graph input for one step.
pub type Binding = (String, Vec<f32>);

/// Pre-train over **several batches**, cycling them for `epochs` passes.
///
/// The per-crate `train_loop`s in this workspace bind one fixed set of inputs and
/// repeat it, which proves an objective differentiates but trains on a single
/// example — a loss reaching zero there is memorisation, not learning. This
/// rebinds the inputs each step so a model sees a corpus, and returns the mean
/// loss per epoch.
///
/// Every batch must carry the same input names and lengths: the graph is compiled
/// once for a fixed shape, so a ragged batch is a caller error rather than
/// something to pad silently.
///
/// `update` applies one optimizer step — `(name, shape, param, grad)`. It is a
/// closure rather than an `&mut dyn Optimizer` on purpose: this crate resolves
/// `rlx-optim` from the registry while the model crates patch it to a local path,
/// so a trait in the signature makes the two `Optimizer`s different types and the
/// caller cannot pass its own. A closure sidesteps the identity entirely.
pub fn train_batches(
    forward: rlx_ir::Graph,
    wrt: &[Trainable],
    params: &mut std::collections::HashMap<String, Vec<f32>>,
    param_shapes: &std::collections::HashMap<String, Vec<usize>>,
    batches: &[Vec<Binding>],
    update: &mut dyn FnMut(&str, &[usize], &mut [f32], &[f32]),
    epochs: usize,
) -> Result<Vec<f32>, String> {
    train_batches_with(
        forward,
        wrt,
        params,
        param_shapes,
        batches,
        update,
        epochs,
        &mut |_, _| {},
    )
}

/// [`train_batches`] with an end-of-epoch hook, which is how a held-out curve gets
/// measured.
///
/// Scoring held-out data only before and after says whether training helped overall
/// and nothing about *when* it stopped helping. On real EEG that gap is the whole
/// story: `rlx-brant` on 1600 Bonn windows ended at train 0.294 / held-out 0.593,
/// so the run was memorising for an unknown number of its last epochs and the
/// before/after pair cannot say which ones. `on_epoch` receives the epoch index
/// (0-based) and the current parameters, so the caller can run its own evaluation
/// graph — the parameters are the full state, nothing else is needed to reproduce a
/// score.
#[allow(clippy::too_many_arguments)]
pub fn train_batches_with(
    forward: rlx_ir::Graph,
    wrt: &[Trainable],
    params: &mut std::collections::HashMap<String, Vec<f32>>,
    param_shapes: &std::collections::HashMap<String, Vec<usize>>,
    batches: &[Vec<Binding>],
    update: &mut dyn FnMut(&str, &[usize], &mut [f32], &[f32]),
    epochs: usize,
    on_epoch: &mut dyn FnMut(usize, &std::collections::HashMap<String, Vec<f32>>),
) -> Result<Vec<f32>, String> {
    if batches.is_empty() {
        return Err("no batches to train on".into());
    }
    let names: Vec<&str> = batches[0].iter().map(|(n, _)| n.as_str()).collect();
    for (i, b) in batches.iter().enumerate() {
        if b.len() != names.len() || b.iter().zip(&names).any(|((n, _), w)| n != w) {
            return Err(format!("batch {i} binds different inputs from batch 0"));
        }
        for ((n, v), (n0, v0)) in b.iter().zip(&batches[0]) {
            if v.len() != v0.len() {
                return Err(format!(
                    "batch {i} input `{n}` has {} values, batch 0's `{n0}` has {}",
                    v.len(),
                    v0.len()
                ));
            }
        }
    }

    let wrt_ids: Vec<rlx_ir::NodeId> = wrt.iter().map(|s| s.node).collect();
    let backward = rlx_autodiff::grad_with_loss(&forward, &wrt_ids);
    let mut compiled = rlx_runtime::Session::new(rlx_runtime::Device::Cpu)
        .compile_with(backward, &rlx_runtime::CompileOptions::new());
    for (name, data) in params.iter() {
        compiled.set_param(name, data);
    }

    let cotangent = [1.0f32];
    let mut epoch_losses = Vec::with_capacity(epochs);
    for epoch in 0..epochs {
        let mut sum = 0.0f32;
        for b in batches {
            let mut run: Vec<(&str, &[f32])> =
                b.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
            run.push(("d_output", &cotangent));
            let outs = compiled.run(&run);
            if outs.is_empty() || outs[0].is_empty() {
                return Err("backward graph produced no loss output".into());
            }
            sum += outs[0][0];

            // outs = [loss, grad(wrt[0]), ...]
            for (i, slot) in wrt.iter().enumerate() {
                let grad = outs.get(1 + i).cloned().unwrap_or_default();
                let p = params
                    .get_mut(&slot.name)
                    .ok_or_else(|| format!("missing param value for {}", slot.name))?;
                let shape = param_shapes
                    .get(&slot.name)
                    .cloned()
                    .unwrap_or_else(|| vec![p.len()]);
                update(&slot.name, &shape, p, &grad);
                compiled.set_param(&slot.name, p);
            }
        }
        epoch_losses.push(sum / batches.len() as f32);
        on_epoch(epoch, params);
    }
    Ok(epoch_losses)
}

/// The mean loss of `batches` under the current `params`, without updating them.
///
/// Held-out evaluation is the only thing that separates pre-training from
/// memorising the training batch, so it belongs next to the loop rather than in a
/// caller's test.
pub fn eval_batches(
    forward: &rlx_ir::Graph,
    params: &std::collections::HashMap<String, Vec<f32>>,
    batches: &[Vec<Binding>],
) -> Result<f32, String> {
    if batches.is_empty() {
        return Err("no batches to evaluate".into());
    }
    let mut compiled = rlx_runtime::Session::new(rlx_runtime::Device::Cpu)
        .compile_with(forward.clone(), &rlx_runtime::CompileOptions::new());
    for (name, data) in params.iter() {
        compiled.set_param(name, data);
    }
    compiled.finalize_params();
    let mut sum = 0.0f32;
    for b in batches {
        let run: Vec<(&str, &[f32])> = b.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        let outs = compiled.run(&run);
        if outs.is_empty() || outs[0].is_empty() {
            return Err("forward graph produced no loss output".into());
        }
        sum += outs[0][0];
    }
    Ok(sum / batches.len() as f32)
}

// ---------------------------------------------------------------------------
// The batched driver: one implementation of "train this objective on a trunk".
// ---------------------------------------------------------------------------

/// Which objective to attach to a trunk's output.
#[derive(Debug, Clone, Copy)]
pub enum Objective {
    /// Reconstruct the masked rows of the input from the trunk output, through a
    /// trained linear head `[width, width]`.
    ///
    /// The target is the input row itself, so there is no separate output width here.
    /// A token that carries raw samples of a different length than the trunk is wide
    /// needs its own input projection as well, which is the caller's graph to build —
    /// see `rlx-brant`'s `pretrain_masked_patches`.
    MaskedMse,
    /// Pick the true latent of each masked position out of `n_distractors + 1`
    /// candidates by cosine similarity (wav2vec2 / BENDR style).
    InfoNce {
        n_distractors: usize,
        temperature: f32,
    },
    /// Predict a discrete cluster id per masked position: cross-entropy over cosine
    /// logits against a learned codebook (HuBERT style).
    ///
    /// The trunk output is projected to `proj_dim` by a trained `final_proj`, both it
    /// and each of the `n_classes` rows of `label_embedding` are L2-normalised, and the
    /// scaled cosine similarities are the logits. Targets come from
    /// [`train_masked_objective`]'s `class_targets`, not from the input, because HuBERT's
    /// labels are produced by clustering *separate* features — see [`kmeans_labels`].
    MaskedCodebookCe {
        n_classes: usize,
        proj_dim: usize,
        temperature: f32,
    },
}

/// Geometry and schedule for [`train_masked_objective`].
#[derive(Debug, Clone)]
pub struct BatchedConfig {
    pub batch: usize,
    pub seq: usize,
    /// Trunk width.
    pub width: usize,
    /// Amplitude of the sinusoidal positional signal added to the trunk input, or
    /// `None` when the trunk adds positions itself.
    ///
    /// **Not optional in practice, and there is no safe default.** Masked rows are
    /// zeroed, so without positions they are *identical* inputs to a
    /// permutation-equivariant attention stack and the best available reconstruction
    /// is one average row for all of them — a hard floor no learning rate or epoch
    /// count moves. Measured: `rlx-brainrvq` on this driver reached 0.178 against a
    /// mean predictor of 0.221, and 2.7x the data with 2.5x the epochs moved it by
    /// 0.016. Pass `Some(scale)` unless the trunk's own graph adds a positional
    /// embedding, and make `scale` comparable to the data's own scale — a signal ten
    /// times smaller than the content is one attention has to dig for.
    pub position_scale: Option<f32>,
    /// Rows to mask, indices into `0..n_pos` (so they span every batch row).
    pub mask_positions: Vec<usize>,
    /// Name of the trunk's input, as the flow declared it — usually `"hidden"`.
    pub input_name: String,
    pub epochs: usize,
    pub lr: f32,
    pub seed: u32,
    /// Score the held-out set every N epochs (0 = only before and after).
    pub eval_every: usize,
}

/// What a [`train_masked_objective`] run measured.
#[derive(Debug, Clone)]
pub struct BatchedReport {
    /// Mean training loss per epoch.
    pub train: Vec<f32>,
    /// `(epoch, held-out loss)` at each [`BatchedConfig::eval_every`] point.
    pub heldout: Vec<(usize, f32)>,
    /// Held-out loss before the first step, and after the last.
    pub before: f32,
    pub after: f32,
}

impl BatchedReport {
    /// The best held-out point, `(epoch, loss)` — `None` without a held-out curve.
    ///
    /// Not the last epoch: these runs overfit, and reporting the final score
    /// understates a model that was better earlier. On Bonn the gap was 18%.
    pub fn best_heldout(&self) -> Option<(usize, f32)> {
        self.heldout
            .iter()
            .copied()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// Attach `obj` to `trunk_out`, then train it over `train` and score it on `heldout`.
///
/// Every port that pre-trains needs the same eleven steps after its trunk graph
/// exists — select the masked rows, build the loss, find the differentiable
/// parameters, bind per-batch host inputs, evaluate before, loop, evaluate after —
/// and the interesting part is none of them. This is that middle, once.
///
/// `graph` must already contain the trunk with `trunk_out` as the hidden state it
/// produces for `cfg.input_name`; `params` are its initial values. `train` and
/// `heldout` are `[batch · seq · width]` buffers.
///
/// Targets and InfoNCE candidates are taken from the **raw** buffer, before masking
/// and before any positional signal is added. Only the trunk's input carries
/// positions: adding them to the target would make part of the answer a constant the
/// head can memorise, and adding them to the candidates would let position alone
/// identify the true one.
pub fn train_masked_objective(
    mut graph: Graph,
    trunk_out: NodeId,
    mut params: std::collections::HashMap<String, Vec<f32>>,
    cfg: &BatchedConfig,
    obj: Objective,
    train: &[Vec<f32>],
    heldout: &[Vec<f32>],
    class_targets: Option<(&[Vec<u32>], &[Vec<u32>])>,
    update: &mut dyn FnMut(&str, &[usize], &mut [f32], &[f32]),
) -> Result<BatchedReport, String> {
    let (n_pos, width) = (cfg.batch * cfg.seq, cfg.width);
    let m = cfg.mask_positions.len();
    if train.is_empty() {
        return Err("no training batches".into());
    }
    if heldout.is_empty() {
        return Err("no held-out batches".into());
    }
    if m == 0 {
        return Err("nothing is masked".into());
    }
    if m >= n_pos {
        return Err("every position is masked; there is no context to reconstruct from".into());
    }
    if cfg.mask_positions.iter().any(|&p| p >= n_pos) {
        return Err(format!("a mask position is outside 0..{n_pos}"));
    }
    for (i, b) in train.iter().chain(heldout).enumerate() {
        if b.len() != n_pos * width {
            return Err(format!(
                "batch {i} has {} values, expected n_pos·width = {}",
                b.len(),
                n_pos * width
            ));
        }
    }

    let picked = select_rows(&mut graph, trunk_out, "sel", m, n_pos, width);
    let loss = match obj {
        Objective::MaskedMse => {
            let rw = graph.param("recon.weight".to_string(), Shape::new(&[width, width], F32));
            let rb = graph.param("recon.bias".to_string(), Shape::new(&[1, width], F32));
            let y = graph.mm(picked, rw);
            let y = graph.add(y, rb);
            params.insert(
                "recon.weight".to_string(),
                pseudo_values(width * width, cfg.seed, (1.0 / width as f32).sqrt()),
            );
            params.insert("recon.bias".to_string(), vec![0f32; width]);
            masked_mse_loss(&mut graph, y, "target", m, width)
        }
        Objective::InfoNce {
            n_distractors,
            temperature,
        } => {
            if n_distractors < 1 {
                return Err("InfoNCE needs at least one distractor".into());
            }
            if n_pos <= n_distractors {
                return Err(format!(
                    "cannot draw {n_distractors} distractors from {n_pos} positions"
                ));
            }
            if temperature <= 0.0 {
                return Err("temperature must be positive".into());
            }
            infonce_loss(
                &mut graph,
                picked,
                "cands",
                m,
                n_distractors + 1,
                width,
                temperature,
            )
        }
        Objective::MaskedCodebookCe {
            n_classes,
            proj_dim,
            temperature,
        } => {
            let (l, ps) = codebook_ce_loss(
                &mut graph,
                picked,
                m,
                width,
                n_classes,
                proj_dim,
                temperature,
                cfg.seed,
            )?;
            params.extend(ps);
            l
        }
    };
    graph.set_outputs(vec![loss]);

    let wrt = trainable_params(&graph, loss);
    if wrt.is_empty() {
        return Err("no trainable parameter found".into());
    }
    let mut shapes: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for slot in &wrt {
        shapes.insert(
            slot.name.clone(),
            graph
                .shape(slot.node)
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect(),
        );
    }

    let sel = selection_matrix(&cfg.mask_positions, n_pos);
    let bind =
        |idx: u64, raw: &Vec<f32>, labels: Option<&Vec<u32>>| -> Result<Vec<Binding>, String> {
            let mut masked = raw.clone();
            for &p in &cfg.mask_positions {
                masked[p * width..(p + 1) * width].fill(0.0);
            }
            if let Some(scale) = cfg.position_scale {
                add_sinusoidal_positions(&mut masked, n_pos, cfg.seq, width, scale);
            }
            let mut v = vec![
                (cfg.input_name.clone(), masked),
                ("sel".to_string(), sel.clone()),
            ];
            match obj {
                Objective::MaskedMse => {
                    let mut tgt = vec![0f32; m * width];
                    for (row, &p) in cfg.mask_positions.iter().enumerate() {
                        tgt[row * width..(row + 1) * width]
                            .copy_from_slice(&raw[p * width..(p + 1) * width]);
                    }
                    v.push(("target".to_string(), tgt));
                }
                Objective::InfoNce { n_distractors, .. } => {
                    // Distractors are drawn from THIS batch. A shared pool would make
                    // them separable by which window they came from rather than by
                    // position, and the trunk would score well without using context.
                    v.push((
                        "cands".to_string(),
                        infonce_candidates(
                            raw,
                            n_pos,
                            width,
                            &cfg.mask_positions,
                            n_distractors,
                            (cfg.seed as u64).wrapping_add(idx).wrapping_mul(2654435761) | 1,
                        ),
                    ));
                }
                Objective::MaskedCodebookCe { n_classes, .. } => {
                    let ids = labels.ok_or_else(|| {
                        "MaskedCodebookCe needs class_targets; its labels come from clustering \
                     separate features, not from the input"
                            .to_string()
                    })?;
                    if ids.len() != m {
                        return Err(format!(
                            "class_targets has {} ids for {m} masked positions",
                            ids.len()
                        ));
                    }
                    let mut onehot = vec![0f32; m * n_classes];
                    for (row, &c) in ids.iter().enumerate() {
                        let c = c as usize;
                        if c >= n_classes {
                            return Err(format!("class id {c} is outside 0..{n_classes}"));
                        }
                        onehot[row * n_classes + c] = 1.0;
                    }
                    v.push(("target_onehot".to_string(), onehot));
                }
            }
            Ok(v)
        };
    let (train_lab, held_lab) = match class_targets {
        Some((t, h)) => {
            if t.len() != train.len() || h.len() != heldout.len() {
                return Err(format!(
                    "class_targets must match the batch counts ({} vs {}, {} vs {})",
                    t.len(),
                    train.len(),
                    h.len(),
                    heldout.len()
                ));
            }
            (Some(t), Some(h))
        }
        None => (None, None),
    };
    let train_b: Vec<Vec<Binding>> = train
        .iter()
        .enumerate()
        .map(|(i, b)| bind(i as u64, b, train_lab.map(|t| &t[i])))
        .collect::<Result<_, String>>()?;
    // A separate seed stream, so a held-out score can never be helped by having seen
    // the same distractor draw during training.
    let held_b: Vec<Vec<Binding>> = heldout
        .iter()
        .enumerate()
        .map(|(i, b)| bind(0x5000_0000 + i as u64, b, held_lab.map(|h| &h[i])))
        .collect::<Result<_, String>>()?;

    let before = eval_batches(&graph, &params, &held_b)?;
    let eval_g = graph.clone();
    let eval_every = cfg.eval_every;
    let mut held_curve: Vec<(usize, f32)> = Vec::new();
    let mut eval_err: Option<String> = None;
    let mut on_epoch = |epoch: usize, ps: &std::collections::HashMap<String, Vec<f32>>| {
        if eval_every == 0 || !(epoch + 1).is_multiple_of(eval_every) || eval_err.is_some() {
            return;
        }
        match eval_batches(&eval_g, ps, &held_b) {
            Ok(v) => held_curve.push((epoch + 1, v)),
            // Recorded rather than ignored: a failure mid-run would otherwise leave
            // a curve with a silent hole in it.
            Err(e) => eval_err = Some(e),
        }
    };
    let curve = train_batches_with(
        graph.clone(),
        &wrt,
        &mut params,
        &shapes,
        &train_b,
        update,
        cfg.epochs,
        &mut on_epoch,
    )?;
    if let Some(e) = eval_err {
        return Err(format!("held-out evaluation failed during training: {e}"));
    }
    let after = eval_batches(&graph, &params, &held_b)?;
    Ok(BatchedReport {
        train: curve,
        heldout: held_curve,
        before,
        after,
    })
}

/// Deterministic small values for a head this module introduces itself.
///
/// Its own, rather than a crate's `weights::pseudo`: the head does not exist in any
/// checkpoint, so nothing downstream can depend on matching a particular
/// initialisation, and reaching into a caller's weight module for one would couple
/// this driver to every crate that uses it.
fn pseudo_values(n: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut st = (seed as u64) | 1;
    (0..n)
        .map(|_| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            ((st >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 2.0 * amp
        })
        .collect()
}

/// k-means cluster ids for every row of `data`, the way HuBERT's first-iteration
/// targets are made.
///
/// HuBERT does not predict its own input: iteration 1 predicts k-means clusters of
/// *separate* acoustic features, and later iterations re-cluster an earlier model's
/// hidden states. That indirection is the objective, not an implementation detail —
/// predicting a quantisation of the very rows the trunk is fed makes the task
/// solvable by copying, which is why [`Objective::MaskedCodebookCe`] takes its labels
/// as an argument instead of deriving them.
///
/// Lloyd's algorithm, k-means++-style seeding by farthest point (deterministic from
/// `seed`), `iters` passes. Returns one id per row of `data`, so
/// `data.len() / width` ids.
pub fn kmeans_labels(data: &[f32], width: usize, k: usize, iters: usize, seed: u64) -> Vec<u32> {
    kmeans(data, width, k, iters, seed).0
}

/// The cluster centres [`kmeans_labels`] converged to, `[k · width]` row-major.
///
/// The reason this is public: a VQ codebook initialised from random noise **collapses**.
/// Its entries land nowhere near the encoder's output distribution, two of them end up
/// nearest to everything, and the commitment loss falls beautifully while the codebook
/// is dead — measured on `rlx-dewave` at 25% of 8 entries in use with the loss down
/// 75x. Seeding the codebook from centres of the encoder's own outputs is the standard
/// fix and the cheap one.
pub fn kmeans_centroids(data: &[f32], width: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    kmeans(data, width, k, iters, seed).1
}

fn kmeans(data: &[f32], width: usize, k: usize, iters: usize, seed: u64) -> (Vec<u32>, Vec<f32>) {
    let n = data.len() / width.max(1);
    if n == 0 || k == 0 {
        return (Vec::new(), Vec::new());
    }
    let k = k.min(n);
    let row = |i: usize| &data[i * width..(i + 1) * width];
    let d2 =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

    // Seed: one row chosen by `seed`, then repeatedly the row farthest from every
    // centre already chosen. Random seeding can leave a centre with no members, and an
    // empty cluster makes one class unpredictable for the whole run.
    let mut centres: Vec<Vec<f32>> = Vec::with_capacity(k);
    centres.push(row((seed as usize) % n).to_vec());
    while centres.len() < k {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..n {
            let nearest = centres
                .iter()
                .map(|c| d2(row(i), c))
                .fold(f32::INFINITY, f32::min);
            if nearest > best.1 {
                best = (i, nearest);
            }
        }
        centres.push(row(best.0).to_vec());
    }

    let mut ids = vec![0u32; n];
    for _ in 0..iters.max(1) {
        let mut moved = false;
        for i in 0..n {
            let (mut bi, mut bd) = (0usize, f32::INFINITY);
            for (c, centre) in centres.iter().enumerate() {
                let d = d2(row(i), centre);
                if d < bd {
                    (bi, bd) = (c, d);
                }
            }
            if ids[i] != bi as u32 {
                ids[i] = bi as u32;
                moved = true;
            }
        }
        let mut sums = vec![vec![0f32; width]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = ids[i] as usize;
            counts[c] += 1;
            for (d, v) in row(i).iter().enumerate() {
                sums[c][d] += v;
            }
        }
        for c in 0..k {
            // An emptied cluster keeps its previous centre rather than collapsing to
            // the origin, which would pull every point towards it on the next pass.
            if counts[c] == 0 {
                continue;
            }
            for d in 0..width {
                centres[c][d] = sums[c][d] / counts[c] as f32;
            }
        }
        if !moved {
            break;
        }
    }
    (ids, centres.concat())
}

/// Cross-entropy of the best **constant** predictor on a set of class labels: the entropy
/// `H(p)` of their distribution.
///
/// The floor any classification objective has to beat, and the one that is easy to get
/// wrong in two directions:
///
/// * `ln(num_classes)` is right only when the classes are balanced. Clustered targets
///   rarely are — k-means on real features leaves uneven clusters, and sleep stages are
///   roughly `13/6/54/8/20`.
/// * `−ln(p_majority)` is **not any predictor's cross-entropy**. The predictor it appears
///   to describe — always answer the majority class, with probability 1 — assigns
///   probability 0 to every other class, so its cross-entropy is *infinite*. Using it as a
///   floor sets the bar below what anything can reach: measured on a Sleep-EDF split it
///   reads 0.626 where `H(p)` is 1.288, which turned a 34%-better result into "does not
///   beat the floor".
///
/// The best a constant predictor can do is emit the label distribution itself, scoring
/// `H(p) = −Σ p_c ln p_c`. Absent classes are skipped rather than contributing `0·ln 0`,
/// which would make the result `NaN` — and a `NaN` floor turns every comparison against it
/// into a silent `false`.
///
/// For **accuracy** the majority-class rate is the right comparison; it is only in
/// cross-entropy that a constant predictor has to hedge.
pub fn prior_predictor_ce(labels: &[Vec<u32>], num_classes: usize) -> f32 {
    let mut counts = vec![0usize; num_classes];
    let mut total = 0usize;
    for b in labels {
        for &c in b {
            if (c as usize) < num_classes {
                counts[c as usize] += 1;
                total += 1;
            }
        }
    }
    if total == 0 {
        return f32::NAN;
    }
    counts
        .iter()
        .filter(|&&n| n > 0)
        .map(|&n| {
            let p = n as f32 / total as f32;
            -p * p.ln()
        })
        .sum()
}

/// Mean-squared error of the best **constant** predictor for `targets`: their variance
/// about their own mean.
///
/// Predicting **zero** is not it, even for z-scored data. Windows are normalised
/// individually, so the masked positions pooled across a corpus have a mean that is only
/// near zero, and `mean(y²)` therefore reports the MSE of the wrong constant. It is an
/// over-estimate of the floor, which flatters the model.
pub fn constant_predictor_mse(targets: &[f32]) -> f32 {
    if targets.is_empty() {
        return f32::NAN;
    }
    let n = targets.len() as f64;
    let mean = targets.iter().map(|&v| v as f64).sum::<f64>() / n;
    (targets
        .iter()
        .map(|&v| (v as f64 - mean) * (v as f64 - mean))
        .sum::<f64>()
        / n) as f32
}

/// Mean absolute error of the best **constant** predictor for `targets`: the MAE about
/// their **median**, not their mean.
///
/// The constant minimising absolute error is the median; using the mean reports a larger
/// number and so understates what the model has to beat. This is the L1 counterpart of
/// [`constant_predictor_mse`] and the baseline an L1-trained regressor should be read
/// against.
pub fn constant_predictor_mae(targets: &[f32]) -> f32 {
    if targets.is_empty() {
        return f32::NAN;
    }
    let mut v: Vec<f32> = targets.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
    };
    (targets
        .iter()
        .map(|&x| (x - median).abs() as f64)
        .sum::<f64>()
        / targets.len() as f64) as f32
}

/// HuBERT's masked-prediction head: scaled cosine logits against a learned codebook, and
/// the cross-entropy against a one-hot target input named `"target_onehot"`.
///
/// Returns the loss node and the parameter values the head introduces. Shared by
/// [`train_masked_objective`]'s [`Objective::MaskedCodebookCe`] arm and by ports that build
/// their own graph — a port with an in-graph feature front-end cannot use the driver, whose
/// masking is host-side, and a second copy of this construction would drift from the first.
///
/// `label_embedding.weight` is stored `[proj_dim, n_classes]` so the similarity is a plain
/// matmul; a HuBERT checkpoint ships `[n_classes, proj_dim]` and must be transposed on load.
#[allow(clippy::too_many_arguments)]
pub fn codebook_ce_loss(
    g: &mut Graph,
    query: NodeId,
    m: usize,
    width: usize,
    n_classes: usize,
    proj_dim: usize,
    temperature: f32,
    seed: u32,
) -> Result<(NodeId, Vec<(String, Vec<f32>)>), String> {
    if n_classes < 2 {
        return Err("a codebook of fewer than 2 classes has nothing to predict".into());
    }
    if temperature <= 0.0 {
        return Err("temperature must be positive".into());
    }
    let pw = g.param(
        "final_proj.weight".to_string(),
        Shape::new(&[width, proj_dim], F32),
    );
    let pb = g.param(
        "final_proj.bias".to_string(),
        Shape::new(&[1, proj_dim], F32),
    );
    let proj = g.mm(query, pw);
    let proj = g.add(proj, pb);
    let proj_n = l2_normalize(g, proj, 1);
    let le = g.param(
        "label_embedding.weight".to_string(),
        Shape::new(&[proj_dim, n_classes], F32),
    );
    let le_n = l2_normalize(g, le, 0);
    let sim = g.mm(proj_n, le_n);
    let inv_t = g.full(&[1, 1], 1.0 / temperature, F32);
    let logits = g.mul(sim, inv_t);

    let mx = g.add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes: vec![1],
            keep_dim: true,
        },
        vec![logits],
        Shape::new(&[m, 1], F32),
    );
    let shifted = g.sub(logits, mx);
    let ex = g.exp(shifted);
    let sum_ex = g.sum(ex, vec![1], true);
    let lse = g.log(sum_ex);
    let onehot = g.input("target_onehot", Shape::new(&[m, n_classes], F32));
    let picked_logit = g.mul(shifted, onehot);
    let pos = g.sum(picked_logit, vec![1], true);
    let nll = g.sub(lse, pos);
    let nll_flat = g.reshape_(nll, vec![m as i64]);
    let loss = g.mean(nll_flat, vec![0], false);

    let params = vec![
        (
            "final_proj.weight".to_string(),
            pseudo_values(width * proj_dim, seed ^ 0x1234, (1.0 / width as f32).sqrt()),
        ),
        ("final_proj.bias".to_string(), vec![0f32; proj_dim]),
        (
            "label_embedding.weight".to_string(),
            pseudo_values(proj_dim * n_classes, seed ^ 0x9abc, 0.5),
        ),
    ];
    Ok((loss, params))
}
