//! The training loop, lifted out of `bin/train.rs` so `main` reads as *parse →
//! resolve config/corpus → build+init → [`Trainer::run`]*. A [`Trainer`] owns
//! the model, optimizer, schedule, and loop policy; each iteration is a short,
//! named sequence — `Trainer::train_step` → `Trainer::maybe_eval`
//! (keep-best checkpoint) → `Trainer::maybe_sample` — instead of one long
//! block. Behavior is identical to the hand-inlined loop: same distill / QAT /
//! plain step selection, same eval-every / sample-every cadence, same keep-best
//! checkpointing and bits/byte report.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use rlx_tensor::{Device, Func, LrSchedule};

use crate::bpe::Bpe;
use crate::checkpoint;
use crate::config::GptConfig;
use crate::data::{Batcher, Tokens};
use crate::model;
use crate::optim::HybridOptimizer;
use crate::progress::Progress;
use crate::rng::Rng;
use crate::sample::{GenOptions, generate};

/// Loop policy derived from the CLI: cadence, gradient clipping, checkpoint
/// path, sampling, the tokenizer, and optional emulated-precision QAT.
pub struct TrainOpts {
    /// Total training steps.
    pub steps: usize,
    /// Global-L2-norm gradient clip (`<= 0` disables).
    pub grad_clip: f32,
    /// Held-out eval / keep-best cadence (`0` disables).
    pub eval_every: usize,
    /// Sample-a-story cadence (`0` disables).
    pub sample_every: usize,
    /// Keep-best checkpoint path.
    pub out: PathBuf,
    /// Trained BPE tokenizer, embedded in the checkpoint and reused for samples.
    pub bpe: Option<Bpe>,
    /// Corpus bytes per token — renormalizes per-token loss to true bits/byte.
    pub bytes_per_tok: f32,
    /// Generation knobs for the periodic sample.
    pub gen_opts: GenOptions,
    /// Emulated-precision QAT `(exp_bits, man_bits, max_normal)`, if `--fake-quant`.
    pub fake_quant: Option<(u32, u32, f32)>,
}

/// Owns the model + optimizer + schedule + policy and runs training to
/// completion. Construct with [`Trainer::new`] after build+init, then call
/// [`Trainer::run`] (or a one-shot diagnostic: [`Trainer::bench_backward`] /
/// [`Trainer::diagnose_grads`]).
pub struct Trainer {
    cfg: GptConfig,
    dev: Device,
    model: Func,
    opt: HybridOptimizer,
    sched: LrSchedule,
    opts: TrainOpts,
    /// Distillation teacher (dense forward → soft targets), if `--distill`.
    teacher: Option<Func>,
    // ── mutable run state ──
    best_val: f32,
    saved_any: bool,
}

impl Trainer {
    /// Assemble a trainer from an already-built+initialized model, its optimizer
    /// and schedule, an optional distillation teacher, and the CLI-derived policy.
    pub fn new(
        cfg: GptConfig,
        dev: Device,
        model: Func,
        opt: HybridOptimizer,
        sched: LrSchedule,
        teacher: Option<Func>,
        opts: TrainOpts,
    ) -> Self {
        Self {
            cfg,
            dev,
            model,
            opt,
            sched,
            opts,
            teacher,
            best_val: f32::INFINITY,
            saved_any: false,
        }
    }

    /// Whether a distillation teacher is attached (changes the step + eval path).
    fn distilling(&self) -> bool {
        self.teacher.is_some()
    }

    /// Snapshot the current weights (name → data) for checkpointing / generation.
    fn params(&self) -> Vec<(String, Vec<f32>)> {
        params_of(&self.model)
    }

    /// One optimizer step on `(tok, tgt)`; returns the batch loss. Selects the
    /// distillation / QAT / plain path exactly as the CLI requested.
    fn train_step(&mut self, step: usize, tok: &[f32], tgt: &[f32]) -> f32 {
        let (next, loss) = if let Some(t) = self.teacher.as_ref() {
            // Distillation: run the dense teacher forward for this batch and feed
            // its logits as the soft target (the graph softmaxes + adds the KD term).
            let tlogits = t.run_on(self.dev, &[("tok_ids", tok)]).remove(0);
            let feed: &[(&str, &[f32])] = &[
                ("tok_ids", tok),
                ("tgt_ids", tgt),
                ("teacher_logits", &tlogits),
            ];
            self.model.train_step_all_at_on_clipped(
                self.dev,
                &mut self.opt,
                &self.sched,
                step,
                self.opts.grad_clip,
                feed,
            )
        } else if let Some((e, mb, mx)) = self.opts.fake_quant {
            // Quantization-aware step: weights → fXmYeZ grid on the forward,
            // straight-through gradient to the f32 masters.
            let feed: &[(&str, &[f32])] = &[("tok_ids", tok), ("tgt_ids", tgt)];
            self.model.train_step_all_at_on_qat(
                self.dev,
                &mut self.opt,
                &self.sched,
                step,
                self.opts.grad_clip,
                |w| rlx_tensor::lowp::quantize_slice_scaled(w, e, mb, mx),
                feed,
            )
        } else {
            let feed: &[(&str, &[f32])] = &[("tok_ids", tok), ("tgt_ids", tgt)];
            self.model.train_step_all_at_on_clipped(
                self.dev,
                &mut self.opt,
                &self.sched,
                step,
                self.opts.grad_clip,
                feed,
            )
        };
        self.model = next;
        loss[0]
    }

    /// On the `eval_every` cadence, estimate held-out loss and keep the best
    /// checkpoint (so a later divergence can never overwrite good weights).
    fn maybe_eval(
        &mut self,
        step: usize,
        is_last: bool,
        val: &Tokens,
        batcher: &Batcher,
        rng: &mut Rng,
        progress: &mut Progress,
    ) -> Result<()> {
        let do_eval = self.opts.eval_every > 0
            && ((step + 1).is_multiple_of(self.opts.eval_every) || is_last)
            && val.len() > self.cfg.block_size + 1;
        if !do_eval {
            return Ok(());
        }
        let (vtok, vtgt) = batcher.sample(val, rng);
        let vfeed: &[(&str, &[f32])] = &[("tok_ids", &vtok), ("tgt_ids", &vtgt)];
        // The distill graph carries a teacher-logits input, so eval its pure-CE
        // loss on a plain graph bound with the current params (comparable
        // bits/byte across runs); otherwise eval the model directly.
        let vloss = if self.distilling() {
            let mut em = model::build(&self.cfg, self.cfg.batch, true);
            for (n, d) in self.params() {
                em = em.with_param(n, d);
            }
            em.run_on(self.dev, vfeed)[0][0]
        } else {
            self.model.run_on(self.dev, vfeed)[0][0]
        };
        // Keep-best: only persist when the held-out loss improves.
        let mark = if vloss.is_finite() && vloss < self.best_val {
            self.best_val = vloss;
            match checkpoint::save(
                &self.opts.out,
                &self.cfg,
                &self.params(),
                self.opts.bpe.as_ref(),
            ) {
                Ok(()) => {
                    self.saved_any = true;
                    " ★ best (saved)"
                }
                Err(e) => {
                    progress.note(&format!("  !! checkpoint save failed: {e}"));
                    ""
                }
            }
        } else {
            ""
        };
        // Per-token loss → true bits/byte (÷ bytes-per-token), so byte-level and
        // BPE runs are on the same axis: fewer bits/byte = better model.
        progress.note(&format!(
            "  ├─ step {}  val loss {vloss:.4}  (bits/byte {:.3}){mark}",
            step + 1,
            vloss / std::f32::consts::LN_2 / self.opts.bytes_per_tok
        ));
        Ok(())
    }

    /// On the `sample_every` cadence, generate a short story from the prompt.
    fn maybe_sample(&self, step: usize, is_last: bool, progress: &mut Progress) {
        if self.opts.sample_every > 0
            && ((step + 1).is_multiple_of(self.opts.sample_every) || is_last)
        {
            let story = generate(
                &self.cfg,
                &self.params(),
                "Once upon a time",
                self.dev,
                &self.opts.gen_opts,
                self.opts.bpe.as_ref(),
            );
            progress.note(&format!("  └─ sample: {}", story.replace('\n', "⏎")));
        }
    }

    /// Train to completion over `train`, estimating held-out loss on `val`. The
    /// loop body is a short, obvious sequence; the objective itself lives in the
    /// model's `rlx! { }` block.
    pub fn run(&mut self, train: Tokens, val: Tokens, rng: &mut Rng) -> Result<()> {
        let batcher = Batcher::new(&self.cfg);
        let t0 = Instant::now();
        let mut ema = f32::NAN;
        let mut progress = Progress::new(self.opts.steps);
        for step in 0..self.opts.steps {
            let (tok, tgt) = batcher.sample(&train, rng);
            let loss = self.train_step(step, &tok, &tgt);
            ema = if ema.is_nan() {
                loss
            } else {
                0.95 * ema + 0.05 * loss
            };
            let is_last = step + 1 == self.opts.steps;
            progress.tick(step + 1, ema, self.sched.lr_at(step));

            // Diverged? Stop — the best checkpoint on disk is retained.
            if !loss.is_finite() {
                progress.note(&format!(
                    "  !! train loss became {loss} at step {} — stopping (best checkpoint kept)",
                    step + 1
                ));
                break;
            }

            self.maybe_eval(step, is_last, &val, &batcher, rng, &mut progress)?;
            self.maybe_sample(step, is_last, &mut progress);
        }
        progress.finish();

        // Fallback save if keep-best never fired (e.g. --eval-every 0 or no val data).
        if !self.saved_any {
            checkpoint::save(
                &self.opts.out,
                &self.cfg,
                &self.params(),
                self.opts.bpe.as_ref(),
            )?;
        }
        println!(
            "done in {:.1}s → best val loss {:.4}, checkpoint {}",
            t0.elapsed().as_secs_f64(),
            self.best_val,
            self.opts.out.display()
        );
        rlx_tensor::clear_cache();
        Ok(())
    }

    /// `RLX_BENCH_BWD=1` one-shot: warm the compile cache, then time forward-only
    /// vs fwd+bwd (the difference is the backward). Prints a `BWD_BENCH` line.
    pub fn bench_backward(&self, train: Tokens, rng: &mut Rng) {
        let batcher = Batcher::new(&self.cfg);
        let (tok, tgt) = batcher.sample(&train, rng);
        let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
        let vg = self.model.value_and_grad_all();
        for _ in 0..3 {
            let _ = self.model.run_on(self.dev, feed);
            let _ = vg.run_on(self.dev, feed);
        }
        let n = 10;
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = self.model.run_on(self.dev, feed);
        }
        let fwd = t0.elapsed().as_secs_f64() / n as f64 * 1e3;
        let t1 = Instant::now();
        for _ in 0..n {
            let _ = vg.run_on(self.dev, feed);
        }
        let fwdbwd = t1.elapsed().as_secs_f64() / n as f64 * 1e3;
        println!(
            "BWD_BENCH forward={fwd:.1}ms fwd+bwd={fwdbwd:.1}ms backward={:.1}ms ratio={:.2}x",
            fwdbwd - fwd,
            (fwdbwd - fwd) / fwd
        );
    }

    /// `--diag-grads` one-shot: report each parameter's gradient finiteness /
    /// max-abs on a single batch, to localize a NaN.
    pub fn diagnose_grads(&self, train: Tokens, rng: &mut Rng) {
        let batcher = Batcher::new(&self.cfg);
        let (tok, tgt) = batcher.sample(&train, rng);
        let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
        let names = self.model.param_names();
        let out = self.model.value_and_grad_all().run_on(self.dev, feed);
        println!("diag: loss = {}", out[0][0]);
        let mut bad = 0;
        for (i, n) in names.iter().enumerate() {
            let g = &out[i + 1];
            let finite = g.iter().all(|x| x.is_finite());
            let maxabs = g.iter().fold(0f32, |a, &x| a.max(x.abs()));
            let flag = if !finite { " <-- NON-FINITE" } else { "" };
            println!("  {n:10}  finite={finite}  maxabs={maxabs:.3e}{flag}");
            if !finite {
                bad += 1;
            }
        }
        println!("diag: {bad}/{} params with non-finite grads", names.len());
        rlx_tensor::clear_cache();
    }
}

/// Snapshot a model's params as `(name, data)` for checkpointing / generation.
fn params_of(model: &Func) -> Vec<(String, Vec<f32>)> {
    model
        .param_names()
        .into_iter()
        .map(|n| {
            let d = model.param_binding(&n).unwrap().to_vec();
            (n, d)
        })
        .collect()
}
