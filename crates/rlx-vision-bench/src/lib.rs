// RLX models — vision training + benchmark harness.
// SPDX-License-Identifier: GPL-3.0-only

//! Vision training + a configuration-sweep **benchmark harness** on RLX:
//! MNIST / Fashion-MNIST / CIFAR-10 / CIFAR-100 / ImageNet / COCO (see
//! [`datasets`]) × MLP / CNN, on any device, single-machine or distributed.
//!
//! - [`harness`] sweeps the `(dataset × model)` matrix on one device
//!   ([`train_local`]) and prints a comparison table — `--suite`.
//! - The distributed **data-parallel** path shards the data across ranks and
//!   averages the per-parameter gradients each step (`all_reduce(Mean)`) before
//!   a momentum-SGD update, so every replica stays bit-identical and the
//!   effective batch is `world_size ×` the local one.
//!
//! Models adapt to the dataset via [`DataSpec`]: the MLP is `pixels → hidden →
//! classes`; the CNN is two 3×3 convs (each + 2× pool) then extra pools down to
//! a bounded spatial size, so one architecture spans 28² up to 640² inputs.
//!
//! Two gradient-sync paths ([`train_report`]):
//! * **sync** — an in-graph `collective.all_reduce(Mean)` (from `rlx-collectives`)
//!   is baked into the compiled graph; each rank's `run()` blocks at it.
//! * **async** — the graph emits rank-local grads; they're flattened into one
//!   bucket (DDP-style) and reduced with a non-blocking
//!   [`rlx_collectives::start_all_reduce`] overlapped with the next batch's prep.
//!
//! Two run modes:
//! * single-machine: [`run_distributed`] spawns `world` ranks as threads over a
//!   loopback transport.
//! * multi-node: [`run_node_from_env`] makes each *process* one rank, joined
//!   across machines via `rlx_driver::node::Node::from_env` (`RANK`/`WORLD`/
//!   `PEERS`/`TOPOLOGY`).
//!
//! Every rank reports its identity (host/pid/shard) + throughput + comm/compute
//! timing, so a multi-node run shows exactly which node does what.

pub mod datasets;
pub mod harness;

pub use datasets::{Data, DataSpec, DatasetKind, Split};
use rlx_collectives::{ReduceKind, ReduceMode};
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::time::Instant;

/// Which model to train.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// `784 → hidden → 10` MLP (fast; ~98%).
    Mlp,
    /// `conv(1→16)·pool·conv(16→32)·pool·fc(→128)·fc(→10)` — with `--augment`
    /// reaches >99.3%.
    Cnn,
}

impl ModelKind {
    pub fn name(self) -> &'static str {
        match self {
            ModelKind::Mlp => "mlp",
            ModelKind::Cnn => "cnn",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mlp" => Some(ModelKind::Mlp),
            "cnn" => Some(ModelKind::Cnn),
            _ => None,
        }
    }
    pub fn all() -> &'static [ModelKind] {
        &[ModelKind::Mlp, ModelKind::Cnn]
    }
}

/// Total trainable-parameter count for `cfg` (sum of every param's element
/// count) — reported by the harness.
pub fn param_count(cfg: &Config) -> usize {
    params_spec(cfg)
        .iter()
        .map(|(_, s)| s.iter().product::<usize>())
        .sum()
}

/// Training hyper-parameters.
#[derive(Clone, Copy)]
pub struct Config {
    pub model: ModelKind,
    /// Input geometry + class count of the dataset being trained.
    pub spec: DataSpec,
    pub hidden: usize,
    pub batch: usize,
    pub epochs: usize,
    pub lr: f32,
    pub momentum: f32,
    pub seed: u64,
    /// Overlap the gradient all-reduce with host-side work (async path).
    pub async_overlap: bool,
    /// Random ±2px translation of training images (regularization).
    pub augment: bool,
    /// Reduce gradients **deterministically**: the sync (in-graph) path bakes
    /// [`ReduceMode::Deterministic`] into the collective, so the cross-rank
    /// gradient is the f64-accumulate ring — reproducible run-to-run and across
    /// node counts, correctly-rounded, at ring bandwidth (~1.5× vs f32). The
    /// async path honors the same mode via `RLX_DETERMINISTIC_REDUCE`. On by
    /// default (the whole model stays bit-identical across replicas); set false
    /// for the plain f32 ring if you want the last drop of reduce bandwidth.
    pub deterministic: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelKind::Mlp,
            spec: DataSpec {
                h: 28,
                w: 28,
                c: 1,
                classes: 10,
            },
            hidden: 256,
            batch: 64,
            epochs: 8,
            lr: 0.02, // effective lr ≈ lr/(1-momentum) = 0.2
            momentum: 0.9,
            seed: 1,
            async_overlap: false,
            augment: false,
            deterministic: true,
        }
    }
}

/// Per-rank training metrics (returned by [`train_report`]).
#[derive(Clone, Copy, Debug)]
pub struct Report {
    pub accuracy: f64,
    pub samples: usize,
    pub wall_s: f64,
    /// Time in backward compute (`run()`).
    pub compute_s: f64,
    /// Time in the gradient all-reduce (0 on the synchronous in-graph path,
    /// where comm is folded into `compute_s`).
    pub comm_s: f64,
}

/// `(name, shape)` of every trainable parameter, in the order the forward
/// declares them.
/// CNN channel widths + the pool target that bounds the FC input regardless of
/// input resolution (extra 2× pools run until the spatial size ≤ this).
const CNN_C1: usize = 32;
const CNN_C2: usize = 64;
const CNN_POOL_TARGET: usize = 8;

/// Spatial dims of the CNN. **Same-padding** 3×3 convs (pad 1) preserve size, so
/// only the 2× max-pools reduce it: after conv1+pool it's `p1`, after conv2+pool
/// `p2`; conv3 keeps `p2`, then extra 2× pools run until ≤ [`CNN_POOL_TARGET`]
/// so the flatten (`CNN_C2 · fh · fw`) stays bounded from 28² to 640² inputs.
struct CnnGeom {
    p1: (usize, usize),
    p2: (usize, usize),
    extra: Vec<(usize, usize)>,
    flat: usize,
}

fn cnn_geom(spec: &DataSpec) -> CnnGeom {
    let p1 = (spec.h / 2, spec.w / 2);
    let p2 = (p1.0 / 2, p1.1 / 2);
    let mut hw = p2;
    let mut extra = Vec::new();
    while hw.0.min(hw.1) > CNN_POOL_TARGET {
        hw = (hw.0 / 2, hw.1 / 2);
        extra.push(hw);
    }
    CnnGeom {
        p1,
        p2,
        extra,
        flat: CNN_C2 * hw.0 * hw.1,
    }
}

fn params_spec(cfg: &Config) -> Vec<(&'static str, Vec<usize>)> {
    let (classes, hidden, pixels) = (cfg.spec.classes, cfg.hidden, cfg.spec.pixels());
    match cfg.model {
        ModelKind::Mlp => vec![
            ("fc1_w", vec![pixels, hidden]),
            ("fc1_b", vec![hidden]),
            ("fc2_w", vec![hidden, classes]),
            ("fc2_b", vec![classes]),
        ],
        ModelKind::Cnn => {
            let geom = cnn_geom(&cfg.spec);
            // Three same-padding conv blocks, each conv → bias → SiLU.
            vec![
                ("conv1_w", vec![CNN_C1, cfg.spec.c, 3, 3]),
                ("conv1_b", vec![CNN_C1]),
                ("conv2_w", vec![CNN_C2, CNN_C1, 3, 3]),
                ("conv2_b", vec![CNN_C2]),
                ("conv3_w", vec![CNN_C2, CNN_C2, 3, 3]),
                ("conv3_b", vec![CNN_C2]),
                ("fc1_w", vec![geom.flat, hidden]),
                ("fc1_b", vec![hidden]),
                ("fc2_w", vec![hidden, classes]),
                ("fc2_b", vec![classes]),
            ]
        }
    }
}

/// Build the model forward + softmax-CE loss. Returns `(loss, logits, params)`.
fn build_forward(g: &mut Graph, cfg: &Config) -> (NodeId, NodeId, Vec<NodeId>) {
    match cfg.model {
        ModelKind::Mlp => build_mlp(g, cfg),
        ModelKind::Cnn => build_cnn(g, cfg),
    }
}

fn softmax_ce_mean(g: &mut Graph, logits: NodeId, labels: NodeId) -> NodeId {
    let loss_per = g.softmax_cross_entropy_with_logits(logits, labels);
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0],
            keep_dim: false,
        },
        vec![loss_per],
        Shape::from_dims(&[], DType::F32),
    )
}

fn build_mlp(g: &mut Graph, cfg: &Config) -> (NodeId, NodeId, Vec<NodeId>) {
    let f = DType::F32;
    let (b, hidden) = (cfg.batch, cfg.hidden);
    let (pixels, classes) = (cfg.spec.pixels(), cfg.spec.classes);
    let x = g.input("x", Shape::new(&[b, pixels], f));
    let labels = g.input("labels", Shape::new(&[b], f));
    let fc1_w = g.param("fc1_w", Shape::new(&[pixels, hidden], f));
    let fc1_b = g.param("fc1_b", Shape::new(&[hidden], f));
    let fc2_w = g.param("fc2_w", Shape::new(&[hidden, classes], f));
    let fc2_b = g.param("fc2_b", Shape::new(&[classes], f));

    let h = g.matmul(x, fc1_w, Shape::new(&[b, hidden], f));
    let h = g.binary(BinaryOp::Add, h, fc1_b, Shape::new(&[b, hidden], f));
    let h = g.activation(Activation::Relu, h, Shape::new(&[b, hidden], f));
    let logits = g.matmul(h, fc2_w, Shape::new(&[b, classes], f));
    let logits = g.binary(BinaryOp::Add, logits, fc2_b, Shape::new(&[b, classes], f));
    let loss = softmax_ce_mean(g, logits, labels);
    (loss, logits, vec![fc1_w, fc1_b, fc2_w, fc2_b])
}

fn build_cnn(g: &mut Graph, cfg: &Config) -> (NodeId, NodeId, Vec<NodeId>) {
    let f = DType::F32;
    let b = cfg.batch;
    let spec = &cfg.spec;
    let (classes, hid) = (spec.classes, cfg.hidden);
    let geom = cnn_geom(spec);
    // Flat pixels reinterpret as [B, C, H, W] row-major.
    let x = g.input("x", Shape::new(&[b, spec.c, spec.h, spec.w], f));
    let labels = g.input("labels", Shape::new(&[b], f));
    let conv1_w = g.param("conv1_w", Shape::new(&[CNN_C1, spec.c, 3, 3], f));
    let conv1_b = g.param("conv1_b", Shape::new(&[CNN_C1], f));
    let conv2_w = g.param("conv2_w", Shape::new(&[CNN_C2, CNN_C1, 3, 3], f));
    let conv2_b = g.param("conv2_b", Shape::new(&[CNN_C2], f));
    let conv3_w = g.param("conv3_w", Shape::new(&[CNN_C2, CNN_C2, 3, 3], f));
    let conv3_b = g.param("conv3_b", Shape::new(&[CNN_C2], f));
    let fc1_w = g.param("fc1_w", Shape::new(&[geom.flat, hid], f));
    let fc1_b = g.param("fc1_b", Shape::new(&[hid], f));
    let fc2_w = g.param("fc2_w", Shape::new(&[hid, classes], f));
    let fc2_b = g.param("fc2_b", Shape::new(&[classes], f));

    // Three same-padding conv blocks (conv → GroupNorm → SiLU); pools after the
    // first two. GroupNorm normalizes activations (batch-size-independent) → far
    // better accuracy + rock-solid training; SiLU keeps gradients alive.
    let h1 = conv_bias_silu(g, x, conv1_w, conv1_b, b, CNN_C1, spec.h, spec.w);
    let (p1h, p1w) = geom.p1;
    let p1 = maxpool(g, h1, b, CNN_C1, p1h, p1w);
    let h2 = conv_bias_silu(g, p1, conv2_w, conv2_b, b, CNN_C2, p1h, p1w);
    let (p2h, p2w) = geom.p2;
    let p2 = maxpool(g, h2, b, CNN_C2, p2h, p2w);
    let mut p = conv_bias_silu(g, p2, conv3_w, conv3_b, b, CNN_C2, p2h, p2w);
    // Extra 2× pools (no params) down to ≤ CNN_POOL_TARGET, bounding the FC head.
    for &(eh, ew) in &geom.extra {
        p = maxpool(g, p, b, CNN_C2, eh, ew);
    }

    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![b as i64, geom.flat as i64],
        },
        vec![p],
        Shape::new(&[b, geom.flat], f),
    );
    let h = g.matmul(flat, fc1_w, Shape::new(&[b, hid], f));
    let h = g.binary(BinaryOp::Add, h, fc1_b, Shape::new(&[b, hid], f));
    let h = g.activation(Activation::Silu, h, Shape::new(&[b, hid], f));
    let logits = g.matmul(h, fc2_w, Shape::new(&[b, classes], f));
    let logits = g.binary(BinaryOp::Add, logits, fc2_b, Shape::new(&[b, classes], f));
    let loss = softmax_ce_mean(g, logits, labels);
    (
        loss,
        logits,
        vec![
            conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, fc1_w, fc1_b, fc2_w, fc2_b,
        ],
    )
}

/// One conv block: same-padding 3×3 conv → per-channel bias → SiLU. Output is
/// `[b, c_out, h, wd]` (same-padding preserves the spatial size).
fn conv_bias_silu(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    bias: NodeId,
    b: usize,
    c_out: usize,
    h: usize,
    wd: usize,
) -> NodeId {
    let f = DType::F32;
    let shape = Shape::new(&[b, c_out, h, wd], f);
    let c = conv2d(g, x, w, b, c_out, h, wd);
    let c = bias_add_4d(g, c, bias, b, c_out, h, wd);
    g.activation(Activation::Silu, c, shape)
}

/// Add a per-channel bias to a `[b, c, h, w]` feature map.
fn bias_add_4d(
    g: &mut Graph,
    x: NodeId,
    bias: NodeId,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
) -> NodeId {
    let f = DType::F32;
    let bias_4d = g.add_node(
        Op::Reshape {
            new_shape: vec![1, c as i64, 1, 1],
        },
        vec![bias],
        Shape::new(&[1, c, 1, 1], f),
    );
    g.binary(BinaryOp::Add, x, bias_4d, Shape::new(&[b, c, h, w], f))
}

fn conv2d(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    b: usize,
    c_out: usize,
    h: usize,
    wd: usize,
) -> NodeId {
    g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![1, 1], // same-padding for 3×3 → preserves spatial size
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[b, c_out, h, wd], DType::F32),
    )
}

fn maxpool(g: &mut Graph, x: NodeId, b: usize, c: usize, h: usize, w: usize) -> NodeId {
    g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[b, c, h, w], DType::F32),
    )
}

/// The backward graph with each per-parameter gradient wrapped in an in-graph
/// `all_reduce(Mean)` over `group_id`. Outputs: `[loss, logits, avg_grad_0…]`.
pub fn build_dp_graph(cfg: &Config, group_id: u64) -> Graph {
    let mut g = Graph::new("mnist_fwd");
    let (loss, logits, params) = build_forward(&mut g, cfg);
    g.set_outputs(vec![loss, logits]); // [0]=loss (seeded), [1]=logits (aux)

    let mut bwd = rlx_autodiff::grad_with_loss(&g, &params);
    // bwd.outputs = [loss, logits, grad_fc1_w, grad_fc1_b, grad_fc2_w, grad_fc2_b].
    let outs = bwd.outputs.clone();
    let (head, grads) = outs.split_at(2);
    let mut new_outs = head.to_vec();
    for &gr in grads {
        // Bake the reduction mode into the graph. `Deterministic` = the
        // f64-accumulate ring, so every replica's averaged gradient is identical
        // run-to-run and across node counts (no drift between one machine and the
        // cluster); plain `all_reduce_op` follows the runtime env default.
        new_outs.push(if cfg.deterministic {
            rlx_collectives::all_reduce_op_mode(
                &mut bwd,
                gr,
                group_id,
                ReduceKind::Mean,
                ReduceMode::Deterministic,
            )
        } else {
            rlx_collectives::all_reduce_op(&mut bwd, gr, group_id, ReduceKind::Mean)
        });
    }
    bwd.set_outputs(new_outs);
    bwd
}

/// The backward graph WITHOUT any in-graph collective — outputs the rank-local
/// gradients `[loss, logits, local_grad_0…]`. The async training path reduces
/// these host-side with [`rlx_collectives::start_all_reduce`], so the comm can
/// be timed separately and overlapped with host work.
pub fn build_local_grad_graph(cfg: &Config) -> Graph {
    let mut g = Graph::new("mnist_fwd_local");
    let (loss, logits, params) = build_forward(&mut g, cfg);
    g.set_outputs(vec![loss, logits]);
    rlx_autodiff::grad_with_loss(&g, &params)
}

/// A forward-only graph (`x → logits`, no collective) for standalone accuracy
/// evaluation on a single rank.
pub fn build_eval_graph(cfg: &Config) -> Graph {
    let mut g = Graph::new("mnist_eval");
    let (_loss, logits, _params) = build_forward(&mut g, cfg);
    g.set_outputs(vec![logits]);
    g
}

/// Deterministic Xavier-uniform init — identical on every rank so replicas
/// start in sync. Weights ∼ U(-√(1/fan_in), +√(1/fan_in)); biases 0.
pub fn init_params(cfg: &Config) -> Vec<Vec<f32>> {
    let mut state = cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = || -> f32 {
        // splitmix64 → f32 in [-1, 1).
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    params_spec(cfg)
        .iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            if name.ends_with("_g") {
                return vec![1.0f32; n]; // GroupNorm γ starts at 1 (identity scale)
            }
            if shape.len() == 1 {
                return vec![0.0f32; n]; // biases + GroupNorm β
            }
            // fan_in: rank-2 fc weight [in, out] → in; rank-4 conv weight
            // [out, in, kh, kw] → in·kh·kw. He scaling (√(2/fan_in)) suits the
            // ReLU-family (SiLU) activations better than Xavier.
            let fan_in = if shape.len() == 4 {
                shape[1] * shape[2] * shape[3]
            } else {
                shape[0]
            };
            let scale = (2.0 / fan_in as f32).sqrt();
            (0..n).map(|_| next() * scale).collect()
        })
        .collect()
}

/// Global-L2-norm gradient clip. The deep CNN's conv gradients can spike and,
/// amplified by momentum, blow the weights out in one step → dead ReLUs →
/// logits collapse to a constant → loss pins at `ln(classes)` (random). Capping
/// the summed gradient norm is the standard cure and lets the CNN actually
/// train at the same lr the MLP uses. Kept tight (1.0) because momentum (0.9)
/// amplifies the clipped gradient ~10× into the velocity — a looser clip (5.0)
/// still let the velocity reach ~50 and blow up after a couple good epochs.
const GRAD_CLIP: f32 = 1.0;

/// Scale factor to apply to every gradient so their combined L2 norm ≤
/// [`GRAD_CLIP`] (1.0 when already under the cap).
fn clip_scale(grads: &[Vec<f32>]) -> f32 {
    let sq: f32 = grads.iter().flat_map(|g| g.iter()).map(|&x| x * x).sum();
    let norm = sq.sqrt();
    if norm.is_finite() && norm > GRAD_CLIP {
        GRAD_CLIP / norm
    } else {
        1.0
    }
}

/// Run one rank's full data-parallel training and return the test accuracy
/// (0..=1) — a thin wrapper over [`train_report`].
pub fn train_rank(
    cfg: &Config,
    group_id: u64,
    train_shard: &Split,
    test: &Split,
    report: bool,
) -> f64 {
    train_report(cfg, group_id, train_shard, test, report).accuracy
}

/// Run one rank's full data-parallel training and return a [`Report`] (accuracy
/// + throughput + comm/compute timing).
///
/// `group_id` must already have a registered `ProcessGroup` (the caller wires
/// the transport); `train_shard` is this rank's slice of the training set.
///
/// Two gradient-sync paths (selected by `cfg.async_overlap`):
/// * **sync** — the all-reduce is baked *into* the graph ([`build_dp_graph`]);
///   `run()` blocks at it. Comm time is folded into `compute_s`.
/// * **async** — the graph emits rank-local grads ([`build_local_grad_graph`]);
///   each bucket is reduced with a non-blocking [`rlx_collectives::start_all_reduce`]
///   while the *next* mini-batch is gathered, then joined. `comm_s` is measured.
pub fn train_report(
    cfg: &Config,
    group_id: u64,
    train_shard: &Split,
    test: &Split,
    report: bool,
) -> Report {
    let graph = if cfg.async_overlap {
        build_local_grad_graph(cfg)
    } else {
        build_dp_graph(cfg, group_id)
    };
    let mut sess = Session::new(training_device()).compile(graph);

    let names: Vec<&str> = params_spec(cfg).into_iter().map(|(n, _)| n).collect();
    let mut params = init_params(cfg);
    let mut vel: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();
    for (name, p) in names.iter().zip(&params) {
        sess.set_param(name, p);
    }

    let n = train_shard.len();
    let batches = n / cfg.batch;
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = cfg.seed.wrapping_add(0xABCD);
    let mut aug = cfg.seed.wrapping_add(0x51ED);

    let (mut compute_s, mut comm_s) = (0.0f64, 0.0f64);
    let mut samples = 0usize;
    let wall = Instant::now();

    for epoch in 0..cfg.epochs {
        // Step-decay LR: halve every third of the run — better final accuracy.
        let lr = cfg.lr * 0.5f32.powi((epoch * 3 / cfg.epochs.max(1)) as i32);
        shuffle(&mut order, &mut rng);
        let mut epoch_loss = 0.0f64;

        // Software-pipelined batches: the next mini-batch is gathered while the
        // current step's all-reduce is in flight (async path), so host-side data
        // prep overlaps the gradient communication.
        let mut cur = (batches > 0)
            .then(|| gather_batch(train_shard, &order[0..cfg.batch], cfg.augment, &mut aug));

        for b in 0..batches {
            let (x, y) = cur.take().unwrap();

            let t = Instant::now();
            let outs = sess.run(&[
                ("x", x.as_slice()),
                ("labels", y.as_slice()),
                ("d_output", &[1.0f32]),
            ]);
            compute_s += t.elapsed().as_secs_f64();
            epoch_loss += outs[0][0] as f64;
            samples += cfg.batch;

            // outs = [loss, logits, grad_0, …].
            let grads: Vec<Vec<f32>> = if cfg.async_overlap {
                let t = Instant::now();
                // Gradient bucketing (à la PyTorch DDP): flatten every bucket
                // into one contiguous buffer and issue a *single* non-blocking
                // all-reduce — one tag, one round-trip, minimal overhead.
                let sizes: Vec<usize> = (0..params.len()).map(|i| outs[2 + i].len()).collect();
                let mut flat: Vec<f32> = Vec::with_capacity(sizes.iter().sum());
                for i in 0..params.len() {
                    flat.extend_from_slice(&outs[2 + i]);
                }
                let handle = rlx_collectives::start_all_reduce(group_id, flat, ReduceKind::Mean)
                    .expect("registered group");
                // Overlap: gather the next batch while the reduce is in flight.
                if b + 1 < batches {
                    cur = Some(gather_batch(
                        train_shard,
                        &order[(b + 1) * cfg.batch..(b + 2) * cfg.batch],
                        cfg.augment,
                        &mut aug,
                    ));
                }
                let reduced = handle.wait();
                comm_s += t.elapsed().as_secs_f64();
                // Scatter the flat buffer back into per-parameter gradients.
                let mut off = 0usize;
                sizes
                    .iter()
                    .map(|&sz| {
                        let g = reduced[off..off + sz].to_vec();
                        off += sz;
                        g
                    })
                    .collect()
            } else {
                if b + 1 < batches {
                    cur = Some(gather_batch(
                        train_shard,
                        &order[(b + 1) * cfg.batch..(b + 2) * cfg.batch],
                        cfg.augment,
                        &mut aug,
                    ));
                }
                // Sync path: grads already all-reduced inside the graph.
                (0..params.len()).map(|i| outs[2 + i].clone()).collect()
            };

            let cs = clip_scale(&grads);
            for i in 0..params.len() {
                let (p, v) = (&mut params[i], &mut vel[i]);
                for j in 0..p.len() {
                    v[j] = cfg.momentum * v[j] + cs * grads[i][j];
                    p[j] -= lr * v[j];
                }
                sess.set_param(names[i], p);
            }
        }
        if report {
            eprintln!(
                "  epoch {}/{}: mean loss {:.4}",
                epoch + 1,
                cfg.epochs,
                epoch_loss / batches.max(1) as f64
            );
        }
    }
    let wall_s = wall.elapsed().as_secs_f64();

    // Standalone forward-only eval (no collective) with the trained params.
    let mut eval = Session::new(Device::Cpu).compile(build_eval_graph(cfg));
    for (name, p) in names.iter().zip(&params) {
        eval.set_param(name, p);
    }
    let accuracy = evaluate(&mut eval, test, cfg.batch);
    Report {
        accuracy,
        samples,
        wall_s,
        compute_s,
        comm_s,
    }
}

/// **Single-machine** training on one `device` — no collective (the harness
/// path). Builds the local-gradient graph, runs momentum SGD directly on the
/// rank-local gradients, evaluates on `test`, and returns a [`Report`]
/// (`comm_s = 0`). This is what the model/dataset [`harness`] uses to sweep
/// configurations on one box; the distributed paths ([`train_report`],
/// [`run_distributed`], [`run_node_from_env`]) add the gradient all-reduce.
pub fn train_local(
    cfg: &Config,
    train: &Split,
    test: &Split,
    device: Device,
    report: bool,
) -> Report {
    let mut sess = Session::new(device).compile(build_local_grad_graph(cfg));
    let names: Vec<&str> = params_spec(cfg).into_iter().map(|(n, _)| n).collect();
    let mut params = init_params(cfg);
    let mut vel: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();
    for (name, p) in names.iter().zip(&params) {
        sess.set_param(name, p);
    }

    let n = train.len();
    let batches = n / cfg.batch;
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = cfg.seed.wrapping_add(0xABCD);
    let mut aug = cfg.seed.wrapping_add(0x51ED);
    let mut compute_s = 0.0f64;
    let mut samples = 0usize;
    let wall = Instant::now();

    for epoch in 0..cfg.epochs {
        let lr = cfg.lr * 0.5f32.powi((epoch * 3 / cfg.epochs.max(1)) as i32);
        shuffle(&mut order, &mut rng);
        let mut epoch_loss = 0.0f64;
        for b in 0..batches {
            let (x, y) = gather_batch(
                train,
                &order[b * cfg.batch..(b + 1) * cfg.batch],
                cfg.augment,
                &mut aug,
            );
            let t = Instant::now();
            let outs = sess.run(&[
                ("x", x.as_slice()),
                ("labels", y.as_slice()),
                ("d_output", &[1.0f32]),
            ]);
            compute_s += t.elapsed().as_secs_f64();
            epoch_loss += outs[0][0] as f64;
            samples += cfg.batch;
            // outs = [loss, logits, local_grad_0…] — clip then apply (no reduce).
            let cs = clip_scale(&outs[2..2 + names.len()]);
            for i in 0..names.len() {
                let (p, v) = (&mut params[i], &mut vel[i]);
                let grad = &outs[2 + i];
                for j in 0..p.len() {
                    v[j] = cfg.momentum * v[j] + cs * grad[j];
                    p[j] -= lr * v[j];
                }
                sess.set_param(names[i], p);
            }
        }
        if report {
            eprintln!(
                "  epoch {}/{}: mean loss {:.4}",
                epoch + 1,
                cfg.epochs,
                epoch_loss / batches.max(1) as f64
            );
        }
    }
    let wall_s = wall.elapsed().as_secs_f64();

    let mut eval = Session::new(Device::Cpu).compile(build_eval_graph(cfg));
    for (name, p) in names.iter().zip(&params) {
        eval.set_param(name, p);
    }
    let accuracy = evaluate(&mut eval, test, cfg.batch);
    Report {
        accuracy,
        samples,
        wall_s,
        compute_s,
        comm_s: 0.0,
    }
}

/// Classification accuracy on `test` in full batches (drops a final partial batch).
fn evaluate(sess: &mut CompiledGraph, test: &Split, batch: usize) -> f64 {
    let n = test.len();
    let classes = test.spec.classes;
    let batches = n / batch;
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut no_aug = 0u64;
    for b in 0..batches {
        let idx: Vec<usize> = (b * batch..(b + 1) * batch).collect();
        let (x, y) = gather_batch(test, &idx, false, &mut no_aug);
        let outs = sess.run(&[("x", x.as_slice()), ("labels", y.as_slice())]);
        let logits = &outs[0];
        for (i, &row) in idx.iter().enumerate() {
            let pred = argmax(&logits[i * classes..(i + 1) * classes]);
            if pred == test.label(row) {
                correct += 1;
            }
            total += 1;
        }
    }
    correct as f64 / total.max(1) as f64
}

/// Gather `idx` into a dense `(images, labels)` batch. When `augment`, each
/// image is randomly translated by up to ±2px (padding with the `-1.0`
/// background) — cheap regularization that meaningfully lifts CNN accuracy.
fn gather_batch(
    split: &Split,
    idx: &[usize],
    augment: bool,
    rng: &mut u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(idx.len() * split.pixels());
    let mut y = Vec::with_capacity(idx.len());
    for &i in idx {
        let img = split.image(i);
        if augment {
            let (dy, dx) = (rand_shift(rng), rand_shift(rng));
            // Horizontal flip only for natural color images (CIFAR/ImageNet/COCO);
            // flipping MNIST/Fashion glyphs would corrupt the label.
            let flip = split.spec.c == 3 && rand_bool(rng);
            shift_into(img, dy, dx, flip, &split.spec, &mut x);
        } else {
            x.extend_from_slice(img);
        }
        y.push(split.labels[i]);
    }
    (x, y)
}

/// A fair coin from a splitmix64 stream.
fn rand_bool(state: &mut u64) -> bool {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 31;
    z & 1 == 1
}

/// A translation offset in `[-2, 2]` from a splitmix64 stream.
fn rand_shift(state: &mut u64) -> i32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 31;
    (z % 5) as i32 - 2
}

/// Append a `[C,H,W]` image shifted by `(dy, dx)` (same shift on every channel)
/// into `out` (background = -1.0, the normalized black used everywhere).
fn shift_into(img: &[f32], dy: i32, dx: i32, flip: bool, spec: &DataSpec, out: &mut Vec<f32>) {
    let (h, w) = (spec.h as i32, spec.w as i32);
    for ch in 0..spec.c {
        let plane = &img[ch * spec.h * spec.w..(ch + 1) * spec.h * spec.w];
        for r in 0..h {
            for c in 0..w {
                let src_c = if flip { w - 1 - c } else { c };
                let (sr, sc) = (r - dy, src_c - dx);
                let v = if (0..h).contains(&sr) && (0..w).contains(&sc) {
                    plane[sr as usize * spec.w + sc as usize]
                } else {
                    -1.0
                };
                out.push(v);
            }
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

fn shuffle(order: &mut [usize], state: &mut u64) {
    // Fisher–Yates with a splitmix64 stream.
    for i in (1..order.len()).rev() {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 31;
        let j = (z % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

/// Split `full` into `world` contiguous shards and return shard `rank`.
pub fn shard(full: &Split, rank: u32, world: u32) -> Split {
    let n = full.len();
    let per = n / world as usize;
    let start = rank as usize * per;
    let end = if rank == world - 1 { n } else { start + per };
    let px = full.pixels();
    Split {
        images: full.images[start * px..end * px].to_vec(),
        labels: full.labels[start..end].to_vec(),
        spec: full.spec,
    }
}

/// This machine's hostname (for the "which node does what" banner).
pub fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "localhost".into())
}

/// The device this rank trains on: `RLX_DEVICE` (e.g. `metal`, `cuda`, `mlx`,
/// `cpu`) if set, else the fastest backend compiled in and live on this host —
/// so a heterogeneous cluster has each rank pick its own best GPU for maximum
/// speed. The gradient all-reduce host-delegates the collective to the CPU
/// (and uses the deterministic f64 ring when `RLX_DETERMINISTIC_REDUCE=1`), so
/// the precision of the cross-rank sync is independent of the per-rank compute
/// backend: GPUs for speed, CPU f64 reduction for precision.
pub fn training_device() -> Device {
    match std::env::var("RLX_DEVICE") {
        Ok(s) if !s.trim().is_empty() => rlx_runtime::parse_device(s.trim()).unwrap_or_else(|e| {
            eprintln!("  (RLX_DEVICE={s:?} invalid: {e}; using fastest_device)");
            rlx_runtime::fastest_device()
        }),
        _ => rlx_runtime::fastest_device(),
    }
}

/// Print the per-rank identity banner: which node/pid owns this rank, the
/// device it computes on, and how much of the data it holds.
pub fn node_banner(rank: u32, world: u32, shard_len: usize, cfg: &Config) {
    let path = if cfg.async_overlap {
        "async-overlap"
    } else {
        "sync in-graph"
    };
    eprintln!(
        "  [rank {rank}/{world}] host={} pid={} device={} shard={shard_len} samples · {path} all-reduce",
        hostname(),
        std::process::id(),
        rlx_runtime::full_name(training_device()),
    );
}

/// Spawn `world` ranks as threads over a loopback `NetTransport`, run
/// data-parallel training (gradients averaged across ranks each step), and
/// return every rank's [`Report`] (indexed by rank). `train` is sharded
/// internally. This is the single-machine demo path; for true multi-node use
/// [`run_node_from_env`] (one process per rank).
pub fn run_distributed(cfg: &Config, world: u32, train: &Split, test: &Split) -> Vec<Report> {
    use rlx_driver::{NetTransport, ProcessGroup};
    use std::net::{Ipv4Addr, SocketAddr, TcpListener};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    // Unique group-id base per call: the collective registry is process-global,
    // so two concurrent `run_distributed` calls (e.g. parallel tests) must not
    // clobber each other's `ProcessGroup`s.
    static GID_BASE: AtomicU64 = AtomicU64::new(5000);
    let base = GID_BASE.fetch_add(world as u64 + 1, Ordering::Relaxed);

    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    let shards: Vec<Split> = (0..world).map(|r| shard(train, r, world)).collect();
    let test = Arc::new(Split {
        images: test.images.clone(),
        labels: test.labels.clone(),
        spec: test.spec,
    });

    let handles: Vec<_> = listeners
        .into_iter()
        .zip(shards)
        .enumerate()
        .map(|(rank, (listener, my_shard))| {
            let addrs = addrs.clone();
            let cfg = *cfg;
            let test = test.clone();
            thread::spawn(move || {
                let rank = rank as u32;
                rlx_collectives::register();
                node_banner(rank, world, my_shard.len(), &cfg);
                let t = NetTransport::from_listener(rank, world, listener, addrs, 1 << 22).unwrap();
                let gid = base + rank as u64;
                rlx_collectives::register_group(gid, Arc::new(ProcessGroup::new(Arc::new(t))));
                let rep = train_report(&cfg, gid, &my_shard, &test, rank == 0);
                rlx_collectives::unregister_group(gid);
                rep
            })
        })
        .collect();

    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

/// **Multi-node** entry: this process is exactly one rank. Reads
/// `RANK`/`WORLD`/`PEERS` (or `DISCOVER=…`) from the environment via
/// [`rlx_driver::node::Node::from_env`], connects the real (cross-machine)
/// transport, shards `train` by rank, trains, and returns this rank's
/// [`Report`]. Every process prints its own node banner, so a multi-machine run
/// shows exactly which host owns which rank + shard.
/// Whether to build the multi-node group over iroh (NAT-traversing: relays +
/// pkarr discovery) rather than the default TCP/Thunderbolt mesh — selected
/// with `TOPOLOGY=iroh` (or `RLX_TRANSPORT=iroh`). The iroh path needs the
/// `iroh` cargo feature (`cargo run -p rlx-vision-bench --features iroh`).
fn iroh_selected() -> bool {
    matches!(std::env::var("TOPOLOGY").as_deref(), Ok("iroh"))
        || matches!(std::env::var("RLX_TRANSPORT").as_deref(), Ok("iroh"))
}

/// Build the multi-node [`ProcessGroup`] from the environment: over iroh when
/// [`iroh_selected`], else the TCP/Thunderbolt path via `Node::from_env`. The
/// gradient all-reduce is transport-agnostic, so training is identical either
/// way — only the wire under the collectives changes.
fn build_multinode_group() -> Result<(u32, u32, std::sync::Arc<rlx_driver::ProcessGroup>), String> {
    if iroh_selected() {
        #[cfg(feature = "iroh")]
        {
            let g =
                rlx_driver::process_group_from_env().map_err(|e| format!("iroh transport: {e}"))?;
            return Ok((g.rank(), g.world_size(), g));
        }
        #[cfg(not(feature = "iroh"))]
        {
            return Err("TOPOLOGY=iroh but rlx-vision-bench was built without the `iroh` feature — rebuild with `--features iroh`".to_string());
        }
    }
    let node = rlx_driver::node::Node::from_env()?;
    let (rank, world) = (node.rank(), node.world());
    let group = node.connect().map_err(|e| format!("connect: {e}"))?;
    Ok((rank, world, group))
}

pub fn run_node_from_env(
    cfg: &Config,
    train: &Split,
    test: &Split,
) -> Result<(u32, u32, Report), String> {
    let (rank, world, group) = build_multinode_group()?;

    rlx_collectives::register();
    let gid = 7000u64; // one process = one group; a fixed id is fine per-process.
    rlx_collectives::register_group(gid, group);

    let my_shard = shard(train, rank, world);
    node_banner(rank, world, my_shard.len(), cfg);
    let rep = train_report(cfg, gid, &my_shard, test, rank == 0);
    rlx_collectives::unregister_group(gid);
    Ok((rank, world, rep))
}

/// Write an f32 array to a node-local `.bin` and return its `file://` URI.
fn write_bin(dir: &std::path::Path, name: &str, vals: &[f32]) -> String {
    let path = dir.join(format!("{name}.bin"));
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&path, &bytes).expect("write data bin");
    format!("file://{}", path.display())
}

/// **Master / coordinator role** — the *only* process that has MNIST code.
///
/// It builds the training job for `cfg`, stages the data + initial params to
/// node-local files (only the `file://` URIs cross the wire, never the tensors),
/// ships a [`rlx_runtime::dist::TrainSpec`] to each **generic worker** (ranks
/// ≥1, running `rlx-collectives`'s `dist_node` example in `MODE=trainserve` with
/// zero model code), runs rank 0's own shard through the same generic
/// `dist::run_train`, then evaluates the trained model.
///
/// The gradient reduce is host-side (`start_all_reduce(Mean)`) so the shipped
/// graph carries no in-graph collective — the workers stay fully model-agnostic.
pub fn run_coordinate(
    cfg: &Config,
    group: std::sync::Arc<rlx_driver::ProcessGroup>,
    train: &Split,
    test: &Split,
) -> Report {
    use rlx_runtime::dist::{self, DataRef, TrainSpec, WeightRef};

    let grad_group = 0u64;
    let (rank, world) = (group.rank(), group.world_size());

    // Stage data + initial params node-locally.
    let dir = std::env::temp_dir();
    let x_uri = write_bin(&dir, "rlx_mnist_x", &train.images);
    let y_uri = write_bin(&dir, "rlx_mnist_y", &train.labels);
    let names: Vec<&str> = params_spec(cfg).into_iter().map(|(n, _)| n).collect();
    let inits = init_params(cfg);
    let weights: Vec<WeightRef> = names
        .iter()
        .zip(&inits)
        .map(|(n, vals)| WeightRef {
            name: n.to_string(),
            uri: write_bin(&dir, &format!("rlx_mnist_p_{n}"), vals),
            packed: false,
        })
        .collect();

    // Per-epoch LR schedule (same step-decay as `train_report`).
    let lr_per_epoch: Vec<f32> = (0..cfg.epochs)
        .map(|e| cfg.lr * 0.5f32.powi((e * 3 / cfg.epochs.max(1)) as i32))
        .collect();

    // Equal shards for every rank (drop the remainder) so all ranks run the same
    // number of batches → their per-step gradient all-reduces stay in lock-step.
    let per = train.len() / world as usize;
    let make_spec = |r: u32| TrainSpec {
        graph: build_local_grad_graph(cfg), // [loss, logits, grad_0…] — no collective
        params: weights.clone(),
        grad_start: 2,
        loss_index: 0,
        data: vec![
            DataRef {
                input: "x".into(),
                uri: x_uri.clone(),
                elem: cfg.spec.pixels(),
                shard_start: r as usize * per,
                shard_len: per,
            },
            DataRef {
                input: "labels".into(),
                uri: y_uri.clone(),
                elem: 1,
                shard_start: r as usize * per,
                shard_len: per,
            },
        ],
        seed_input: Some("d_output".into()),
        momentum: cfg.momentum,
        lr_per_epoch: lr_per_epoch.clone(),
        batch: cfg.batch,
        // Each worker's device directive: `auto` (its fastest) or `all`
        // (intra-node data-parallel across every backend). RLX_WORKER_DEVICE
        // overrides; must be uniform across ranks so their reduces stay
        // lock-step (equal shards → equal batch counts).
        device: std::env::var("RLX_WORKER_DEVICE").unwrap_or_else(|_| "auto".into()),
        grad_group,
        // Push shards over the wire when workers have no shared filesystem
        // (required cross-machine); default off (shared FS / pre-staged).
        push_data: std::env::var("RLX_PUSH_DATA").is_ok(),
    };
    let push = std::env::var("RLX_PUSH_DATA").is_ok();

    rlx_collectives::register();
    rlx_collectives::register_group(grad_group, group.clone());
    group.barrier().expect("initial barrier"); // mirror the worker's post-connect barrier

    // Ship each generic worker its job (they're blocked in `recv_train`), then
    // push its data shard if it has no shared filesystem.
    for r in 1..world {
        let spec_r = make_spec(r);
        dist::ship_train(&group, r, &spec_r).expect("ship_train");
        if push {
            dist::push_shards(&group, r, &spec_r, dist::uri_resolver).expect("push_shards");
        }
    }
    eprintln!(
        "  [master rank {rank}] host={} — shipped a {}-param TrainSpec to {} generic worker(s); \
         training my own shard on this node…",
        hostname(),
        weights.len(),
        world.saturating_sub(1),
    );

    // Rank 0 runs its own shard through the SAME generic loop the workers use.
    let (m, final_params) = dist::run_train(
        &make_spec(0),
        world,
        dist::uri_resolver,
        |flat| {
            rlx_collectives::start_all_reduce(grad_group, flat.to_vec(), ReduceKind::Mean)
                .expect("gradient group")
                .wait()
        },
        true,
    )
    .expect("run_train");

    group.barrier().expect("post-train barrier");
    rlx_collectives::unregister_group(grad_group);
    group.barrier().expect("final barrier");

    // Evaluate the trained model (rank 0's params == every rank's, kept synced).
    let mut eval = Session::new(Device::Cpu).compile(build_eval_graph(cfg));
    for (n, v) in &final_params {
        eval.set_param(n, v);
    }
    let accuracy = evaluate(&mut eval, test, cfg.batch);
    eprintln!(
        "  [master rank {rank}] trained on {} — {} samples, compute {:.2}s / comm {:.2}s, loss {:.4}→{:.4}",
        rlx_runtime::full_name(m.device),
        m.samples,
        m.compute_s,
        m.comm_s,
        m.first_loss,
        m.last_loss,
    );
    Report {
        accuracy,
        samples: m.samples,
        wall_s: m.wall_s,
        compute_s: m.compute_s,
        comm_s: m.comm_s,
    }
}

/// **Intra-node all-backends** demo (single node, no networking). Trains via the
/// generic `dist::run_train` with `device: "all"`, which spins up one lane per
/// backend this node supports (CPU + GPU + …) and runs full mini-batches
/// concurrently, averaging their gradients (effective batch = `batch × lanes`).
/// Prints which backends were actually used. The cross-worker reduce is the
/// identity (world = 1), so this isolates the intra-node speedup.
pub fn run_alldev(cfg: &Config, train: &Split, test: &Split) -> Report {
    use rlx_runtime::dist::{self, DataRef, TrainSpec, WeightRef};

    let dir = std::env::temp_dir();
    let x_uri = write_bin(&dir, "rlx_mnist_x", &train.images);
    let y_uri = write_bin(&dir, "rlx_mnist_y", &train.labels);
    let names: Vec<&str> = params_spec(cfg).into_iter().map(|(n, _)| n).collect();
    let weights: Vec<WeightRef> = names
        .iter()
        .zip(&init_params(cfg))
        .map(|(n, vals)| WeightRef {
            name: n.to_string(),
            uri: write_bin(&dir, &format!("rlx_mnist_p_{n}"), vals),
            packed: false,
        })
        .collect();
    let lr_per_epoch: Vec<f32> = (0..cfg.epochs)
        .map(|e| cfg.lr * 0.5f32.powi((e * 3 / cfg.epochs.max(1)) as i32))
        .collect();
    let n = train.len();
    let spec = TrainSpec {
        graph: build_local_grad_graph(cfg),
        params: weights,
        grad_start: 2,
        loss_index: 0,
        data: vec![
            DataRef {
                input: "x".into(),
                uri: x_uri,
                elem: cfg.spec.pixels(),
                shard_start: 0,
                shard_len: n,
            },
            DataRef {
                input: "labels".into(),
                uri: y_uri,
                elem: 1,
                shard_start: 0,
                shard_len: n,
            },
        ],
        seed_input: Some("d_output".into()),
        momentum: cfg.momentum,
        lr_per_epoch,
        batch: cfg.batch,
        device: "all".into(),
        grad_group: 0,
        push_data: false,
    };

    // Single node → world 1, so the cross-worker reduce is the identity.
    let (m, final_params) =
        dist::run_train(&spec, 1, dist::uri_resolver, |flat| flat.to_vec(), true)
            .expect("run_train");

    let mut eval = Session::new(Device::Cpu).compile(build_eval_graph(cfg));
    for (nm, v) in &final_params {
        eval.set_param(nm, v);
    }
    let accuracy = evaluate(&mut eval, test, cfg.batch);
    let used: Vec<&str> = m
        .lanes
        .iter()
        .map(|d| rlx_runtime::device_label(*d))
        .collect();
    eprintln!(
        "  intra-node: trained across {used:?} concurrently — {} samples, compute {:.2}s, loss {:.4}→{:.4}",
        m.samples, m.compute_s, m.first_loss, m.last_loss,
    );
    Report {
        accuracy,
        samples: m.samples,
        wall_s: m.wall_s,
        compute_s: m.compute_s,
        comm_s: m.comm_s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivially-separable 10-class task (class `c` lights up its pixel block),
    /// interleaved by class so each contiguous shard sees every class.
    fn synthetic(total: usize) -> Split {
        let spec = DataSpec {
            h: 28,
            w: 28,
            c: 1,
            classes: 10,
        };
        let (px, classes) = (spec.pixels(), spec.classes);
        let block = px / classes;
        let mut images = Vec::with_capacity(total * px);
        let mut labels = Vec::with_capacity(total);
        for i in 0..total {
            let c = i % classes;
            let mut img = vec![-1.0f32; px];
            for p in c * block..(c + 1) * block {
                img[p] = 1.0;
            }
            images.extend_from_slice(&img);
            labels.push(c as f32);
        }
        Split {
            images,
            labels,
            spec,
        }
    }

    #[test]
    fn distributed_dp_training_learns() {
        // 2-rank data-parallel training on the synthetic task: the in-graph
        // gradient all-reduce keeps replicas in sync and it learns to ~100%.
        let train = synthetic(640);
        let test = synthetic(160);
        let cfg = Config {
            model: ModelKind::Mlp,
            spec: DataSpec {
                h: 28,
                w: 28,
                c: 1,
                classes: 10,
            },
            hidden: 64,
            batch: 32,
            epochs: 6,
            lr: 0.02, // effective lr ≈ lr/(1-momentum) = 0.2
            momentum: 0.9,
            seed: 1,
            async_overlap: false,
            augment: false,
            deterministic: true, // exercises the deterministic in-graph reduce
        };
        let reps = run_distributed(&cfg, 2, &train, &test);
        let acc = reps[0].accuracy;
        assert!(acc > 0.9, "2-rank DP training should learn, got {acc}");
    }

    #[test]
    fn async_overlap_path_learns() {
        // Same task via the async host-side all-reduce path — must also learn,
        // and it records a non-zero comm time.
        let train = synthetic(640);
        let test = synthetic(160);
        let cfg = Config {
            model: ModelKind::Mlp,
            spec: DataSpec {
                h: 28,
                w: 28,
                c: 1,
                classes: 10,
            },
            hidden: 64,
            batch: 32,
            epochs: 6,
            lr: 0.02,
            momentum: 0.9,
            seed: 1,
            async_overlap: true,
            augment: true,
            deterministic: true,
        };
        let reps = run_distributed(&cfg, 2, &train, &test);
        assert!(
            reps[0].accuracy > 0.9,
            "async DP training should learn, got {}",
            reps[0].accuracy
        );
        assert!(reps[0].comm_s > 0.0, "async path must record comm time");
    }
}
