//! Fitting Jacobians — model-independent.
//!
//! Everything here is written against [`LensModel`], so it runs unchanged on
//! any model that can hand back a residual-to-residual graph.

use std::time::{Duration, Instant};

use anyhow::Context as _;
use rlx_autodiff::{SavedActivation, split_vjp};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::estimator::{
    ResidualShape, SKIP_FIRST_N_POSITIONS, fill_onehot_cotangent, scaled_frobenius_norm,
    valid_positions, write_rows,
};
use crate::model::{LensModel, Params, Result};
use crate::vjp::{Tap, TappedGraph};

/// How the Jacobian is estimated.
#[derive(Debug, Clone, Copy)]
pub struct FitConfig {
    /// Output dimensions covered per VJP pass. Each is carried by one batch
    /// element, so the graph is built at `batch = dim_batch`. Higher means
    /// fewer passes and more memory; total work is unchanged.
    pub dim_batch: usize,
    /// Leading positions excluded from the average — attention sinks with
    /// atypical residual statistics.
    pub skip_first: usize,
    /// Device to run on.
    pub device: Device,
}

/// Select the gated-delta-net backward the target device can actually run.
///
/// `Op::GatedDeltaNetBackward` is a fused kernel that CPU, Metal and MLX
/// implement and CUDA and ROCm do not. A backend without it must instead
/// differentiate the unrolled scan, which every backend can run: identical
/// mathematics, but slower and needing 327 saved activations instead of 42.
/// That choice has to be made *before* autodiff runs, so it cannot be a
/// post-hoc graph rewrite, and upstream exposes it as the process-global
/// `RLX_GDN_UNFUSE_FOR_AD`.
///
/// Deciding it from the device beats making every caller know which backends
/// have the kernel — getting it wrong is not graceful degradation, it is
/// `backend "cuda" doesn't claim support for GatedDeltaNetBackward` at compile
/// time.
///
/// Because the switch is process-global it is re-selected on *every* call,
/// including back to the fused path: fitting on CUDA and then on CPU in one
/// process must not leave the CPU fit on the slow path. An explicit
/// `RLX_GDN_UNFUSE_FOR_AD` in the environment at startup is honoured and never
/// touched, so a caller can still pin either path — which is also how a
/// like-for-like cross-backend comparison is set up, since otherwise CPU and
/// CUDA would be running different decompositions.
pub fn select_gdn_backward_for(device: Device) {
    const FLAG: &str = "RLX_GDN_UNFUSE_FOR_AD";
    // Whether the *caller* pinned it, sampled once before we ever write to it.
    static PINNED_BY_CALLER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PINNED_BY_CALLER.get_or_init(|| rlx_ir::env::var(FLAG).is_some()) {
        return;
    }
    if gdn_backward_is_fused_on(device) {
        rlx_ir::env::unset(FLAG);
    } else {
        rlx_ir::env::set(FLAG, "1");
    }
}

/// Whether `device` implements the fused `Op::GatedDeltaNetBackward`.
pub fn gdn_backward_is_fused_on(device: Device) -> bool {
    rlx_runtime::supports(
        device,
        &rlx_ir::Op::GatedDeltaNetBackward {
            state_size: 128,
            carry_state: false,
            gate_per_channel: false,
        },
    )
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            dim_batch: 8,
            skip_first: SKIP_FIRST_N_POSITIONS,
            device: Device::Cpu,
        }
    }
}

/// Where the time went.
///
/// The estimator sweeps `ceil(d_model / dim_batch)` cotangents over one
/// forward, so the interesting number is not total time but how it divides:
/// `forward` should be paid once per residual, `replay` once per pass. If
/// `forward` scales with the pass count, the split is not doing its job.
#[derive(Debug, Clone, Copy, Default)]
pub struct LensTiming {
    /// Building and compiling the save/replay graphs — once per `BlockLens`.
    pub compile: Duration,
    /// Running the forward half.
    pub forward: Duration,
    pub forward_runs: usize,
    /// Binding saved activations onto the replay graph.
    pub bind: Duration,
    /// Running the gradient half.
    pub replay: Duration,
    pub replay_runs: usize,
}

impl LensTiming {
    pub fn total(&self) -> Duration {
        self.compile + self.forward + self.bind + self.replay
    }

    /// One-line summary for logs.
    pub fn summary(&self) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        format!(
            "compile {:.0} ms | forward {:.0} ms over {} run(s) | bind {:.0} ms | \
             replay {:.0} ms over {} run(s) | total {:.0} ms",
            ms(self.compile),
            ms(self.forward),
            self.forward_runs,
            ms(self.bind),
            ms(self.replay),
            self.replay_runs,
            ms(self.total()),
        )
    }
}

/// A fitted Jacobian for one cut point.
#[derive(Debug, Clone)]
pub struct Jacobian {
    /// Row-major `[d_model, d_model]`, `J[i][j] = ∂h_out_i / ∂h_in_j`. A
    /// readout is `J·h`.
    pub values: Vec<f32>,
    pub d_model: usize,
}

impl Jacobian {
    pub fn zeros(d_model: usize) -> Self {
        Self {
            values: vec![0.0; d_model * d_model],
            d_model,
        }
    }

    /// `‖J‖_F / √d` — the magnitude diagnostic. Heavy-tailed prompts show up
    /// here before they show up in a fitted lens.
    pub fn scaled_norm(&self) -> f32 {
        scaled_frobenius_norm(&self.values, self.d_model)
    }

    /// Transport a residual into the target-layer basis: `h @ Jᵀ`.
    pub fn transport(&self, residual: &[f32]) -> Vec<f32> {
        let d = self.d_model;
        assert_eq!(
            residual.len() % d,
            0,
            "residual is not a multiple of d_model"
        );
        let rows = residual.len() / d;
        let mut out = vec![0.0f32; rows * d];
        for r in 0..rows {
            for i in 0..d {
                let mut acc = 0.0f32;
                for j in 0..d {
                    acc += self.values[i * d + j] * residual[r * d + j];
                }
                out[r * d + i] = acc;
            }
        }
        out
    }

    /// Accumulate `other` into `self` (running sum across prompts).
    pub fn add(&mut self, other: &Jacobian) {
        assert_eq!(self.d_model, other.d_model, "d_model mismatch");
        for (a, b) in self.values.iter_mut().zip(&other.values) {
            *a += b;
        }
    }

    /// Divide by `n` — turning a running sum into a mean.
    pub fn scale(&mut self, n: f32) {
        for v in self.values.iter_mut() {
            *v /= n;
        }
    }
}

/// Compile both halves of a split VJP, freeing each parameter as it lands.
///
/// Binding from a borrowed map keeps the entire host-side parameter set alive
/// until *both* arenas have been filled, so the trunk is resident three times at
/// the peak: once on the host, once in the save arena, once in the replay arena.
/// For a 3B model in f32 that is ~33 GB before a single activation exists, and
/// on a unified-memory machine it is the difference between fitting and being
/// killed. Consuming the map lets each tensor drop as soon as both graphs have
/// copied it, which caps the peak at the two arenas plus one tensor.
///
/// Both halves are subsets of the same parameter set and `set_param` ignores
/// names a graph does not have, so binding everything to each is correct.
fn compile_pair(
    save_graph: rlx_ir::Graph,
    replay_graph: rlx_ir::Graph,
    params: &mut Params,
    device: Device,
) -> (CompiledGraph, CompiledGraph) {
    let mut save = Session::new(device).compile(save_graph);
    let mut replay = Session::new(device).compile(replay_graph);
    for (name, data) in params.drain() {
        save.set_param(&name, &data);
        replay.set_param(&name, &data);
    }
    (save, replay)
}

/// A compiled VJP over one residual block, reusable across inputs.
///
/// The backward graph is split into a forward half and a gradient half
/// (`rlx_autodiff::split_vjp`), so a Jacobian costs **one** forward pass plus
/// `ceil(d_model / dim_batch)` gradient passes — not one forward per pass.
/// `grad_with_loss` mirrors the whole forward into the backward graph, which is
/// right for training and wrong for sweeping many cotangents over one input.
///
/// Saved activations cross as parameters, so they are bound once per residual
/// rather than fed on every pass.
pub struct BlockLens {
    save: CompiledGraph,
    replay: CompiledGraph,
    saved: Vec<SavedActivation>,
    residual_input: String,
    /// Whether the gradient half reads the residual directly (it does when a
    /// weight gradient needs the layer input).
    replay_needs_residual: bool,
    /// Index into `replay`'s outputs holding the tap's gradient.
    grad_output: usize,
    shape: ResidualShape,
    positions: Vec<usize>,
    dim_batch: usize,
    timing: LensTiming,
}

impl BlockLens {
    /// Build and compile the VJP for `layer` of `model`.
    pub fn new(model: &dyn LensModel, layer: usize, seq: usize, cfg: FitConfig) -> Result<Self> {
        let start = Instant::now();
        select_gdn_backward_for(cfg.device);
        model.check_layer(layer)?;
        let mut block = model.block(layer, cfg.dim_batch, seq)?;
        let d_model = model.d_model();
        let positions = valid_positions(seq, cfg.skip_first)
            .context("choosing positions to average the Jacobian over")?;

        let tapped = TappedGraph::new(
            block.graph,
            vec![Tap::at_input(layer, block.residual_input.clone())],
        )
        .context("tapping the block's residual input")?;
        // Index of the tap's gradient among the monolithic backward's outputs.
        let grad_in_bwd = tapped.grad_output_index(0);

        let split = split_vjp(&tapped.vjp_graph()).context("splitting the backward graph")?;
        let grad_output = split
            .replay_output_indices
            .iter()
            .position(|&idx| idx == grad_in_bwd)
            .context("the tap's gradient is not produced by the replay half")?;
        let replay_needs_residual = split.replay.input_id(&block.residual_input).is_some();

        let (save, replay) = compile_pair(split.save, split.replay, &mut block.params, cfg.device);

        Ok(Self {
            save,
            replay,
            saved: split.saved,
            residual_input: block.residual_input,
            replay_needs_residual,
            grad_output,
            shape: ResidualShape::new(cfg.dim_batch, seq, d_model),
            positions,
            dim_batch: cfg.dim_batch,
            timing: LensTiming {
                compile: start.elapsed(),
                ..Default::default()
            },
        })
    }

    /// Sequence positions the Jacobian is averaged over.
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// Number of gradient passes per Jacobian.
    pub fn passes(&self) -> usize {
        self.shape.d_model.div_ceil(self.dim_batch)
    }

    /// Activations carried from the forward half to the gradient half.
    pub fn saved_activations(&self) -> usize {
        self.saved.len()
    }

    /// Accumulated timing.
    pub fn timing(&self) -> LensTiming {
        self.timing
    }

    /// Fit `J` for one input residual.
    ///
    /// `residual` is `[dim_batch, seq, d_model]` — the same sequence replicated
    /// across the batch, since batch element `b` carries output dimension
    /// `dim_start + b`'s cotangent. Use [`Self::replicate`] to build it.
    pub fn jacobian(&mut self, residual: &[f32]) -> Result<Jacobian> {
        if residual.len() != self.shape.elements() {
            return Err(anyhow::anyhow!(
                "residual is {} elements, need {} ([{}, {}, {}])",
                residual.len(),
                self.shape.elements(),
                self.shape.batch,
                self.shape.seq,
                self.shape.d_model
            )
            .into());
        }

        // Forward, once.
        let t0 = Instant::now();
        let saved_values = self.save.run(&[(self.residual_input.as_str(), residual)]);
        self.timing.forward += t0.elapsed();
        self.timing.forward_runs += 1;

        let t1 = Instant::now();
        for s in &self.saved {
            self.replay.set_param(&s.name, &saved_values[s.save_output]);
        }
        self.timing.bind += t1.elapsed();

        let d_model = self.shape.d_model;
        let mut jacobian = Jacobian::zeros(d_model);
        let mut cotangent = vec![0.0f32; self.shape.elements()];

        let mut dim_start = 0;
        while dim_start < d_model {
            let n_dims = self.dim_batch.min(d_model - dim_start);
            fill_onehot_cotangent(
                &mut cotangent,
                self.shape,
                dim_start,
                n_dims,
                &self.positions,
            )?;
            let t2 = Instant::now();
            let mut feed: Vec<(&str, &[f32])> = vec![("d_output", &cotangent[..])];
            if self.replay_needs_residual {
                feed.push((self.residual_input.as_str(), residual));
            }
            let outs = self.replay.run(&feed);
            self.timing.replay += t2.elapsed();
            self.timing.replay_runs += 1;

            write_rows(
                &outs[self.grad_output],
                self.shape,
                dim_start,
                n_dims,
                &self.positions,
                &mut jacobian.values,
            )?;
            dim_start += self.dim_batch;
        }
        Ok(jacobian)
    }

    /// Replicate a single `[seq, d_model]` residual across the batch axis.
    pub fn replicate(&self, single: &[f32]) -> Result<Vec<f32>> {
        let per = self.shape.seq * self.shape.d_model;
        if single.len() != per {
            return Err(anyhow::anyhow!(
                "residual is {} elements, need {per} ([{}, {}])",
                single.len(),
                self.shape.seq,
                self.shape.d_model
            )
            .into());
        }
        Ok(single.repeat(self.shape.batch))
    }

    /// Mean Jacobian over a corpus of `[seq, d_model]` residuals.
    ///
    /// Returns `None` for an empty corpus. `observe` is called with each
    /// prompt's own Jacobian before it is folded in — the hook for the
    /// convergence and outlier diagnostics the reference implementation logs.
    pub fn fit(
        &mut self,
        residuals: &[Vec<f32>],
        mut observe: impl FnMut(usize, &Jacobian),
    ) -> Result<Option<Jacobian>> {
        if residuals.is_empty() {
            return Ok(None);
        }
        let mut sum = Jacobian::zeros(self.shape.d_model);
        for (i, residual) in residuals.iter().enumerate() {
            let batched = self.replicate(residual)?;
            let j = self.jacobian(&batched)?;
            observe(i, &j);
            sum.add(&j);
        }
        sum.scale(residuals.len() as f32);
        Ok(Some(sum))
    }
}

/// The lens proper: `J_l` for **every** tapped layer, from one forward.
///
/// [`BlockLens`] differentiates a single block in isolation. This differentiates
/// the residual at the target layer with respect to the residual at each source
/// layer, over a real prompt — the `J_l = E[∂h_final/∂h_l]` the lens is defined
/// by.
///
/// The economy is that `grad_with_loss_wrt` takes *many* `wrt` targets, so one
/// backward yields every layer's gradient at once. Fitting `L` layers costs the
/// same VJP sweep as fitting one; only the row-accumulation is per layer. On top
/// of that, [`rlx_autodiff::split_vjp()`] means the forward runs once per prompt
/// rather than once per cotangent.
pub struct StackLens {
    save: CompiledGraph,
    replay: CompiledGraph,
    saved: Vec<SavedActivation>,
    token_input: String,
    /// Elements the trunk's input expects, from its declared shape.
    input_elems: usize,
    /// Auxiliary inputs bound on every run (see `StackGraph::extra_feeds`).
    extra_feeds: Vec<(String, Vec<f32>)>,
    /// Indices of `extra_feeds` the gradient half still reads. Resolved once,
    /// against the split graph, because a `CompiledGraph` cannot be queried.
    replay_extra: Vec<usize>,
    replay_needs_tokens: bool,
    /// Index into `replay`'s outputs per tap, parallel to `layers`.
    grad_outputs: Vec<usize>,
    layers: Vec<usize>,
    shape: ResidualShape,
    positions: Vec<usize>,
    dim_batch: usize,
    timing: LensTiming,
}

impl StackLens {
    /// Build and compile the tapped VJP over `model`'s trunk.
    ///
    /// `source_layers` are tapped at their *entry* residual; `target_layer`
    /// supplies `h_final`.
    pub fn new(
        model: &dyn LensModel,
        source_layers: &[usize],
        target_layer: usize,
        seq: usize,
        cfg: FitConfig,
    ) -> Result<Self> {
        let start = Instant::now();
        select_gdn_backward_for(cfg.device);
        let mut stack = model.stack(source_layers, target_layer, cfg.dim_batch, seq)?;
        let input_elems = {
            let g = stack.tapped.graph();
            g.input_id(&stack.token_input)
                .and_then(|id| g.node(id).shape.num_elements())
                .unwrap_or(cfg.dim_batch * seq)
        };
        let d_model = model.d_model();
        let positions = valid_positions(seq, cfg.skip_first)
            .context("choosing positions to average the Jacobian over")?;

        // Gradient output indices in the monolithic backward, before the split.
        let grads_in_bwd: Vec<usize> = (0..stack.tapped.taps().len())
            .map(|i| stack.tapped.grad_output_index(i))
            .collect();

        let split = split_vjp(&stack.tapped.vjp_graph()).context("splitting the backward graph")?;
        let grad_outputs: Vec<usize> = grads_in_bwd
            .iter()
            .map(|&idx| {
                split
                    .replay_output_indices
                    .iter()
                    .position(|&o| o == idx)
                    .context("a tap's gradient is not produced by the replay half")
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let save_graph = split.save;
        let replay_graph = split.replay;
        let replay_needs_tokens = replay_graph.input_id(&stack.token_input).is_some();
        let replay_extra: Vec<usize> = stack
            .extra_feeds
            .iter()
            .enumerate()
            .filter(|(_, (name, _))| replay_graph.input_id(name).is_some())
            .map(|(i, _)| i)
            .collect();

        let (save, replay) = compile_pair(save_graph, replay_graph, &mut stack.params, cfg.device);

        Ok(Self {
            save,
            replay,
            saved: split.saved,
            token_input: stack.token_input,
            input_elems,
            extra_feeds: stack.extra_feeds,
            replay_extra,
            replay_needs_tokens,
            grad_outputs,
            layers: stack.layers,
            shape: ResidualShape::new(cfg.dim_batch, seq, d_model),
            positions,
            dim_batch: cfg.dim_batch,
            timing: LensTiming {
                compile: start.elapsed(),
                ..Default::default()
            },
        })
    }

    /// Layer index per fitted Jacobian, in order.
    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    pub fn passes(&self) -> usize {
        self.shape.d_model.div_ceil(self.dim_batch)
    }

    pub fn saved_activations(&self) -> usize {
        self.saved.len()
    }

    pub fn timing(&self) -> LensTiming {
        self.timing
    }

    /// Replicate one prompt's `[seq]` token ids across the batch axis.
    ///
    /// Every replica is the same prompt: the batch axis carries *output
    /// dimensions*, not different prompts.
    pub fn replicate_tokens(&self, tokens: &[f32]) -> Result<Vec<f32>> {
        if tokens.len() != self.shape.seq {
            return Err(anyhow::anyhow!(
                "prompt is {} tokens, need {}",
                tokens.len(),
                self.shape.seq
            )
            .into());
        }
        Ok(tokens.repeat(self.shape.batch))
    }

    /// Fit `J_l` for every tapped layer on one prompt.
    ///
    /// `tokens` is `[dim_batch · seq]` — the same prompt replicated; use
    /// [`Self::replicate_tokens`].
    pub fn jacobians(&mut self, tokens: &[f32]) -> Result<Vec<Jacobian>> {
        // Validated against the graph's *declared* input, not `batch · seq`.
        // A language model's trunk takes `[batch, seq]` token ids, but a vision
        // trunk takes pixels or an already-embedded `[batch, tokens, d]`
        // sequence, and assuming the former is one more place the interface
        // quietly means "language model".
        if tokens.len() != self.input_elems {
            return Err(anyhow::anyhow!(
                "input buffer is {} elements, need {} for `{}`",
                tokens.len(),
                self.input_elems,
                self.token_input
            )
            .into());
        }

        let t0 = Instant::now();
        let mut save_feed: Vec<(&str, &[f32])> = vec![(self.token_input.as_str(), tokens)];
        for (name, data) in &self.extra_feeds {
            save_feed.push((name.as_str(), data.as_slice()));
        }
        let saved_values = self.save.run(&save_feed);
        self.timing.forward += t0.elapsed();
        self.timing.forward_runs += 1;

        let t1 = Instant::now();
        for s in &self.saved {
            self.replay.set_param(&s.name, &saved_values[s.save_output]);
        }
        self.timing.bind += t1.elapsed();

        let d_model = self.shape.d_model;
        let mut jacobians: Vec<Jacobian> = (0..self.layers.len())
            .map(|_| Jacobian::zeros(d_model))
            .collect();
        let mut cotangent = vec![0.0f32; self.shape.elements()];

        let mut dim_start = 0;
        while dim_start < d_model {
            let n_dims = self.dim_batch.min(d_model - dim_start);
            fill_onehot_cotangent(
                &mut cotangent,
                self.shape,
                dim_start,
                n_dims,
                &self.positions,
            )?;
            let t2 = Instant::now();
            let mut feed: Vec<(&str, &[f32])> = vec![("d_output", &cotangent[..])];
            if self.replay_needs_tokens {
                feed.push((self.token_input.as_str(), tokens));
            }
            // The gradient half re-reads whatever auxiliary inputs survived the
            // split — mRoPE tables, masks — on every pass.
            for i in &self.replay_extra {
                let (name, data) = &self.extra_feeds[*i];
                feed.push((name.as_str(), data.as_slice()));
            }
            let outs = self.replay.run(&feed);
            self.timing.replay += t2.elapsed();
            self.timing.replay_runs += 1;

            // One backward, every layer's rows.
            for (j, &slot) in self.grad_outputs.iter().enumerate() {
                write_rows(
                    &outs[slot],
                    self.shape,
                    dim_start,
                    n_dims,
                    &self.positions,
                    &mut jacobians[j].values,
                )?;
            }
            dim_start += self.dim_batch;
        }
        Ok(jacobians)
    }

    /// Mean `J_l` per layer over a corpus of prompts.
    ///
    /// `observe` sees each prompt's own Jacobians before they are folded in.
    pub fn fit(
        &mut self,
        prompts: &[Vec<f32>],
        mut observe: impl FnMut(usize, &[Jacobian]),
    ) -> Result<Option<Vec<Jacobian>>> {
        if prompts.is_empty() {
            return Ok(None);
        }
        let d_model = self.shape.d_model;
        let mut sums: Vec<Jacobian> = (0..self.layers.len())
            .map(|_| Jacobian::zeros(d_model))
            .collect();
        for (i, prompt) in prompts.iter().enumerate() {
            let batched = self.replicate_tokens(prompt)?;
            let js = self.jacobians(&batched)?;
            observe(i, &js);
            for (sum, j) in sums.iter_mut().zip(&js) {
                sum.add(j);
            }
        }
        let n = prompts.len() as f32;
        for sum in sums.iter_mut() {
            sum.scale(n);
        }
        Ok(Some(sums))
    }
}
