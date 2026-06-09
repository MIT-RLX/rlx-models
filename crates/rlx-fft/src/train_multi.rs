// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Multi-n_fft encoder–decoder training study and train×eval matrix.

use crate::butterfly::butterfly_train_step_encdec;
use crate::config::{FftLearnConfig, MultiTrainConfig, MultiTrainSchedule};
use crate::fused_train::fused_encdec_train_step;
use crate::second_order::{TwiddleOptState, TwiddleOptimizer};
use crate::train::random_batch;
use crate::train_phased::precision_encdec;
use crate::twiddle::exact_twiddles;
use crate::weights::{EncDecWeights, export_safetensors};
use anyhow::{Result, ensure};
use rand::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn null_as_nan<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(f32::NAN))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTrainEvalRow {
    /// Training regime label (`single_64`, `mixed_round_robin`, `exact`, …).
    pub regime: String,
    pub schedule: String,
    /// n_fft sizes included in this training run.
    pub train_sizes: Vec<usize>,
    pub eval_n_fft: usize,
    pub train_steps_total: usize,
    pub train_elapsed_ms: f64,
    #[serde(deserialize_with = "null_as_nan")]
    pub encoder_spectrum_mse: f32,
    #[serde(deserialize_with = "null_as_nan")]
    pub encoder_spectrum_max_err: f32,
    #[serde(deserialize_with = "null_as_nan")]
    pub decoder_time_mse: f32,
    #[serde(deserialize_with = "null_as_nan")]
    pub decoder_time_max_err: f32,
    #[serde(deserialize_with = "null_as_nan")]
    pub roundtrip_mse: f32,
    #[serde(deserialize_with = "null_as_nan")]
    pub roundtrip_max_err: f32,
    pub converged: bool,
    #[serde(deserialize_with = "null_as_nan")]
    pub final_holdout_mse: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTrainReport {
    pub batch: usize,
    pub n_ffts: Vec<usize>,
    pub max_steps: usize,
    pub min_steps: usize,
    pub until_converged: bool,
    pub eval_batches: usize,
    pub seed: u64,
    #[serde(default = "default_grad_clip_report")]
    pub grad_clip: f32,
    #[serde(default)]
    pub project_twiddles: bool,
    #[serde(default = "default_use_fused")]
    pub use_fused_train: bool,
    #[serde(default = "default_optimizer_label")]
    pub optimizer: String,
    pub elapsed_ms: f64,
    pub rows: Vec<MultiTrainEvalRow>,
}

fn default_grad_clip_report() -> f32 {
    1.0
}

fn default_use_fused() -> bool {
    true
}

fn default_optimizer_label() -> String {
    "sgd".into()
}

struct SizeTwiddles {
    encoder: Vec<f32>,
    decoder: Vec<f32>,
    opt: TwiddleOptState,
}

fn new_size_twiddles(model: &FftLearnConfig, optimizer: TwiddleOptimizer) -> SizeTwiddles {
    let stages = model.n_fft.trailing_zeros() as usize;
    let half = model.n_fft / 2;
    let tw_len = stages * half * 2;
    SizeTwiddles {
        encoder: exact_twiddles(model),
        decoder: exact_twiddles(model),
        opt: TwiddleOptState::new(optimizer, tw_len, tw_len),
    }
}
struct ConvergenceTracker {
    patience: usize,
    rel_delta: f32,
    abs_delta: f32,
    best: f32,
    stale: usize,
}

impl ConvergenceTracker {
    fn new(cfg: &MultiTrainConfig) -> Self {
        Self {
            patience: cfg.converge_patience,
            rel_delta: cfg.converge_delta,
            abs_delta: cfg.converge_delta * 1e-4,
            best: f32::INFINITY,
            stale: 0,
        }
    }

    /// Returns true when training should stop (loss plateau).
    fn observe(&mut self, loss: f32) -> bool {
        if !loss.is_finite() {
            self.stale = 0;
            return false;
        }
        let improved = if !self.best.is_finite() {
            true
        } else {
            let drop = self.best - loss;
            drop > self.abs_delta || drop / self.best.max(1e-12) > self.rel_delta
        };
        if improved {
            self.best = loss;
            self.stale = 0;
        } else {
            self.stale += 1;
        }
        self.stale >= self.patience
    }
}

pub fn run_multi_train(cfg: &MultiTrainConfig) -> Result<MultiTrainReport> {
    ensure!(!cfg.n_ffts.is_empty(), "n_ffts must not be empty");
    ensure!(cfg.steps >= 1, "steps must be >= 1");
    for &n in &cfg.n_ffts {
        FftLearnConfig::new(n, cfg.batch)?;
    }

    let started = Instant::now();
    let mut rows = Vec::new();

    rows.extend(eval_exact_baseline(cfg)?);

    for &schedule in &cfg.schedules {
        eprintln!("[train-multi] schedule={}", schedule.label());
        let regime_rows = train_schedule(cfg, schedule)?;
        rows.extend(regime_rows);
    }

    let report = MultiTrainReport {
        batch: cfg.batch,
        n_ffts: cfg.n_ffts.clone(),
        max_steps: cfg.steps,
        min_steps: cfg.min_steps,
        until_converged: cfg.until_converged,
        eval_batches: cfg.eval_batches,
        seed: cfg.seed,
        grad_clip: cfg.grad_clip,
        project_twiddles: cfg.project_twiddles,
        use_fused_train: cfg.use_fused_train,
        optimizer: cfg.optimizer.label().to_string(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        rows,
    };

    if let Some(out) = &cfg.out_dir {
        std::fs::create_dir_all(out)?;
        let path = out.join("multi_train_report.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
        eprintln!("wrote {}", path.display());
    }

    Ok(report)
}

fn eval_exact_baseline(cfg: &MultiTrainConfig) -> Result<Vec<MultiTrainEvalRow>> {
    let mut rows = Vec::new();
    for &n in &cfg.n_ffts {
        let model = FftLearnConfig::new(n, cfg.batch)?;
        let enc = exact_twiddles(&model);
        let dec = exact_twiddles(&model);
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let (enc_mse, enc_max, dec_mse, dec_max, rt_mse, rt_max) =
            precision_encdec(&enc, &dec, &model, cfg.eval_batches, &mut rng)?;
        rows.push(row_from_metrics(
            "exact",
            "exact",
            vec![n],
            n,
            0,
            0.0,
            enc_mse,
            enc_max,
            dec_mse,
            dec_max,
            rt_mse,
            rt_max,
            true,
            rt_mse,
            None,
        ));
    }
    Ok(rows)
}

fn train_schedule(
    cfg: &MultiTrainConfig,
    schedule: MultiTrainSchedule,
) -> Result<Vec<MultiTrainEvalRow>> {
    match schedule {
        MultiTrainSchedule::Single => train_single_per_size(cfg),
        MultiTrainSchedule::RoundRobin
        | MultiTrainSchedule::Random
        | MultiTrainSchedule::Balanced => train_mixed(cfg, schedule),
    }
}

fn train_single_per_size(cfg: &MultiTrainConfig) -> Result<Vec<MultiTrainEvalRow>> {
    let mut all_rows = Vec::new();

    for &n in &cfg.n_ffts {
        let model = FftLearnConfig::new(n, cfg.batch)?;
        let tw = new_size_twiddles(&model, cfg.optimizer);
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.wrapping_add(n as u64));
        let regime = format!("single_{n}");
        let label = regime.clone();

        let outcome = train_until_converged(
            cfg,
            &label,
            &mut rng,
            move |_step, tw_map, rng| {
                let st = tw_map.get_mut(&n).expect("twiddles");
                train_encdec_step_on(cfg, st, n, rng)
            },
            HashMap::from([(n, tw)]),
            |tw_map, rng| holdout_mse(cfg, tw_map, &[n], rng),
        )?;

        let tw = outcome.tw;
        let st = tw.get(&n).expect("twiddles");
        let weights = EncDecWeights::from_twiddles(&st.encoder, &st.decoder, n);
        let checkpoint = save_multi_checkpoint(cfg, &regime, &weights, n)?;

        all_rows.extend(eval_twiddles_matrix(
            cfg,
            &regime,
            schedule_label(MultiTrainSchedule::Single),
            &[n],
            outcome.steps,
            outcome.elapsed_ms,
            outcome.converged,
            outcome.holdout_mse,
            &tw,
            checkpoint,
        )?);
    }
    Ok(all_rows)
}

fn train_mixed(
    cfg: &MultiTrainConfig,
    schedule: MultiTrainSchedule,
) -> Result<Vec<MultiTrainEvalRow>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let mut tw: HashMap<usize, SizeTwiddles> = HashMap::new();
    for &n in &cfg.n_ffts {
        let model = FftLearnConfig::new(n, cfg.batch)?;
        tw.insert(n, new_size_twiddles(&model, cfg.optimizer));
    }

    let regime = format!("mixed_{}", schedule.label());
    let n_sizes = cfg.n_ffts.len();
    let eval_sizes = cfg.n_ffts.clone();

    let outcome = match schedule {
        MultiTrainSchedule::Balanced => {
            let per = cfg.steps / n_sizes;
            ensure!(
                per >= 1,
                "steps={} too small for {} sizes in balanced mode",
                cfg.steps,
                n_sizes
            );
            train_balanced_until_converged(cfg, &regime, per, &mut tw, &mut rng)?
        }
        MultiTrainSchedule::RoundRobin | MultiTrainSchedule::Random => train_until_converged(
            cfg,
            &regime,
            &mut rng,
            move |step, tw_map, rng| {
                let pick = match schedule {
                    MultiTrainSchedule::RoundRobin => cfg.n_ffts[step % n_sizes],
                    MultiTrainSchedule::Random => cfg.n_ffts[rng.gen_range(0..n_sizes)],
                    _ => unreachable!(),
                };
                let st = tw_map.get_mut(&pick).expect("twiddles");
                train_encdec_step_on(cfg, st, pick, rng)
            },
            tw,
            {
                let eval_sizes = eval_sizes.clone();
                move |tw_map, rng| holdout_mse(cfg, tw_map, &eval_sizes, rng)
            },
        )?,
        MultiTrainSchedule::Single => unreachable!(),
    };

    let tw = outcome.tw;
    let mut checkpoint = None;
    if let Some(out_dir) = &cfg.out_dir {
        let dir = out_dir.join(&regime);
        std::fs::create_dir_all(&dir)?;
        for &n in &cfg.n_ffts {
            let st = tw.get(&n).expect("twiddles");
            let weights = EncDecWeights::from_twiddles(&st.encoder, &st.decoder, n);
            let path = dir.join(format!("n{n}_encdec.safetensors"));
            export_safetensors(&path, &weights.merged())?;
        }
        checkpoint = Some(dir);
    }

    eval_twiddles_matrix(
        cfg,
        &regime,
        schedule.label().to_string(),
        &cfg.n_ffts,
        outcome.steps,
        outcome.elapsed_ms,
        outcome.converged,
        outcome.holdout_mse,
        &tw,
        checkpoint,
    )
}

struct ConvergeOutcome {
    tw: HashMap<usize, SizeTwiddles>,
    steps: usize,
    elapsed_ms: f64,
    converged: bool,
    holdout_mse: f32,
}

fn train_until_converged<R: Rng>(
    cfg: &MultiTrainConfig,
    label: &str,
    rng: &mut R,
    mut step_fn: impl FnMut(usize, &mut HashMap<usize, SizeTwiddles>, &mut R) -> Result<()>,
    mut tw: HashMap<usize, SizeTwiddles>,
    mut holdout_fn: impl FnMut(&HashMap<usize, SizeTwiddles>, &mut R) -> Result<f32>,
) -> Result<ConvergeOutcome> {
    let started = Instant::now();
    let mut tracker = ConvergenceTracker::new(cfg);
    let mut step = 0usize;
    let mut converged = false;
    let mut holdout_mse = f32::INFINITY;

    while step < cfg.steps {
        step_fn(step, &mut tw, rng)?;
        step += 1;

        if cfg.until_converged && step >= cfg.min_steps && step.is_multiple_of(cfg.converge_every) {
            holdout_mse = holdout_fn(&tw, rng)?;
            eprintln!(
                "  [{label}] step {step} holdout_mse={holdout_mse:.6e} best={:.6e}",
                tracker.best
            );
            if tracker.observe(holdout_mse) {
                converged = true;
                eprintln!("  [{label}] converged at step {step} holdout_mse={holdout_mse:.6e}");
                break;
            }
        } else if cfg.log_every > 0 && step.is_multiple_of(cfg.log_every) {
            eprintln!("  [{label}] step {step}/{}", cfg.steps);
        }
    }

    if !holdout_mse.is_finite() {
        holdout_mse = holdout_fn(&tw, rng)?;
    }

    Ok(ConvergeOutcome {
        tw,
        steps: step,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        converged: converged && holdout_mse.is_finite(),
        holdout_mse,
    })
}

fn train_balanced_until_converged<R: Rng>(
    cfg: &MultiTrainConfig,
    label: &str,
    per_size: usize,
    tw: &mut HashMap<usize, SizeTwiddles>,
    rng: &mut R,
) -> Result<ConvergeOutcome> {
    let started = Instant::now();
    let mut tracker = ConvergenceTracker::new(cfg);
    let mut step = 0usize;
    let mut converged = false;
    let mut final_holdout = f32::INFINITY;
    let eval_sizes = cfg.n_ffts.clone();

    'outer: while step < cfg.steps {
        for &n in &cfg.n_ffts {
            if step >= cfg.steps {
                break 'outer;
            }
            let st = tw.get_mut(&n).expect("twiddles");
            train_encdec_step_on(cfg, st, n, rng)?;
            step += 1;

            if cfg.until_converged
                && step >= cfg.min_steps
                && step.is_multiple_of(cfg.converge_every)
            {
                let loss = holdout_mse(cfg, tw, &eval_sizes, rng)?;
                eprintln!(
                    "  [{label}] step {step} holdout_mse={loss:.6e} best={:.6e}",
                    tracker.best
                );
                if tracker.observe(loss) {
                    converged = true;
                    final_holdout = loss;
                    eprintln!("  [{label}] converged at step {step} holdout_mse={loss:.6e}");
                    break 'outer;
                }
                final_holdout = loss;
            } else if cfg.log_every > 0 && step.is_multiple_of(cfg.log_every) {
                eprintln!(
                    "  [{label}] step {step}/{} (balanced ~{per_size}/size)",
                    cfg.steps
                );
            }
        }
    }

    if !final_holdout.is_finite() {
        final_holdout = holdout_mse(cfg, tw, &eval_sizes, rng)?;
    }

    Ok(ConvergeOutcome {
        tw: std::mem::take(tw),
        steps: step,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        converged: converged && final_holdout.is_finite(),
        holdout_mse: final_holdout,
    })
}

fn holdout_mse(
    cfg: &MultiTrainConfig,
    tw: &HashMap<usize, SizeTwiddles>,
    sizes: &[usize],
    rng: &mut impl Rng,
) -> Result<f32> {
    let mut acc = 0f32;
    let mut n = 0f32;
    for &size in sizes {
        let Some(st) = tw.get(&size) else {
            continue;
        };
        let model = FftLearnConfig::new(size, cfg.batch)?;
        let (_, _, _, _, rt_mse, _) =
            precision_encdec(&st.encoder, &st.decoder, &model, cfg.eval_batches, rng)?;
        acc += rt_mse;
        n += 1.0;
    }
    Ok(if n > 0.0 { acc / n } else { f32::INFINITY })
}

fn train_encdec_step_on(
    cfg: &MultiTrainConfig,
    st: &mut SizeTwiddles,
    n: usize,
    rng: &mut impl Rng,
) -> Result<()> {
    let signal = random_batch(rng, cfg.batch, n);
    if cfg.use_fused_train {
        fused_encdec_train_step(
            &signal,
            &mut st.encoder,
            &mut st.decoder,
            cfg.batch,
            n,
            cfg.lr,
            cfg.spectrum_weight,
            cfg.grad_clip,
            cfg.project_twiddles,
            Some(&mut st.opt),
        )?;
    } else {
        butterfly_train_step_encdec(
            &signal,
            &mut st.encoder,
            &mut st.decoder,
            cfg.batch,
            n,
            cfg.lr as f32,
            cfg.spectrum_weight,
        )?;
    }
    Ok(())
}

fn eval_twiddles_matrix(
    cfg: &MultiTrainConfig,
    regime: &str,
    schedule: String,
    train_sizes: &[usize],
    train_steps: usize,
    train_elapsed_ms: f64,
    converged: bool,
    holdout_mse: f32,
    tw: &HashMap<usize, SizeTwiddles>,
    checkpoint: Option<PathBuf>,
) -> Result<Vec<MultiTrainEvalRow>> {
    let mut rows = Vec::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.wrapping_add(17));

    for &eval_n in &cfg.n_ffts {
        let Some(st) = tw.get(&eval_n) else {
            continue;
        };
        let model = FftLearnConfig::new(eval_n, cfg.batch)?;
        let (enc_mse, enc_max, dec_mse, dec_max, rt_mse, rt_max) =
            precision_encdec(&st.encoder, &st.decoder, &model, cfg.eval_batches, &mut rng)?;
        rows.push(row_from_metrics(
            regime,
            &schedule,
            train_sizes.to_vec(),
            eval_n,
            train_steps,
            train_elapsed_ms,
            enc_mse,
            enc_max,
            dec_mse,
            dec_max,
            rt_mse,
            rt_max,
            converged,
            holdout_mse,
            checkpoint.clone(),
        ));
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn row_from_metrics(
    regime: &str,
    schedule: &str,
    train_sizes: Vec<usize>,
    eval_n_fft: usize,
    train_steps_total: usize,
    train_elapsed_ms: f64,
    encoder_spectrum_mse: f32,
    encoder_spectrum_max_err: f32,
    decoder_time_mse: f32,
    decoder_time_max_err: f32,
    roundtrip_mse: f32,
    roundtrip_max_err: f32,
    converged: bool,
    final_holdout_mse: f32,
    checkpoint: Option<PathBuf>,
) -> MultiTrainEvalRow {
    MultiTrainEvalRow {
        regime: regime.to_string(),
        schedule: schedule.to_string(),
        train_sizes,
        eval_n_fft,
        train_steps_total,
        train_elapsed_ms,
        encoder_spectrum_mse,
        encoder_spectrum_max_err,
        decoder_time_mse,
        decoder_time_max_err,
        roundtrip_mse,
        roundtrip_max_err,
        converged,
        final_holdout_mse,
        checkpoint,
    }
}

fn save_multi_checkpoint(
    cfg: &MultiTrainConfig,
    regime: &str,
    weights: &EncDecWeights,
    n: usize,
) -> Result<Option<PathBuf>> {
    let Some(out) = &cfg.out_dir else {
        return Ok(None);
    };
    let dir = out.join(regime);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("n{n}_encdec.safetensors"));
    export_safetensors(&path, &weights.merged())?;
    Ok(Some(path))
}

fn schedule_label(s: MultiTrainSchedule) -> String {
    s.label().to_string()
}

pub fn write_multi_train_json(path: &Path, report: &MultiTrainReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

pub fn best_regime_per_eval(report: &MultiTrainReport) -> Vec<(usize, String, f32)> {
    let mut out = Vec::new();
    for &n in &report.n_ffts {
        let best = report
            .rows
            .iter()
            .filter(|r| r.eval_n_fft == n && r.regime != "exact")
            .min_by(|a, b| {
                a.roundtrip_max_err
                    .partial_cmp(&b.roundtrip_max_err)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(r) = best {
            out.push((n, r.regime.clone(), r.roundtrip_max_err));
        }
    }
    out
}

pub fn print_multi_train_table(report: &MultiTrainReport) {
    eprintln!(
        "\n=== Multi-n_fft training study (batch={}, max_steps={}, min_steps={}, until_converged={}) ===\n",
        report.batch, report.max_steps, report.min_steps, report.until_converged
    );

    for &eval_n in &report.n_ffts {
        eprintln!("--- eval n_fft={eval_n} ---");
        eprintln!(
            "{:<22} {:>10} {:>6} {:>10} {:>10} {:>10}",
            "regime", "steps", "conv", "rt_max", "enc_max", "train_ms"
        );
        let mut subset: Vec<&MultiTrainEvalRow> = report
            .rows
            .iter()
            .filter(|r| r.eval_n_fft == eval_n)
            .collect();
        subset.sort_by(|a, b| {
            a.roundtrip_max_err
                .partial_cmp(&b.roundtrip_max_err)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for r in &subset {
            eprintln!(
                "{:<22} {:>10} {:>6} {:>10.3e} {:>10.3e} {:>10.1}",
                r.regime,
                r.train_steps_total,
                if r.converged { "yes" } else { "no" },
                r.roundtrip_max_err,
                r.encoder_spectrum_max_err,
                r.train_elapsed_ms
            );
        }
        if let Some(best) = subset.first() {
            eprintln!(
                "  → best: {} (rt_max={:.3e}, steps={})\n",
                best.regime, best.roundtrip_max_err, best.train_steps_total
            );
        }
    }

    eprintln!("--- train×eval roundtrip max_err matrix ---");
    let regimes: Vec<String> = report
        .rows
        .iter()
        .map(|r| r.regime.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    eprint!("{:>22}", "regime \\ eval");
    for &n in &report.n_ffts {
        eprint!(" {:>10}", n);
    }
    eprintln!();
    for regime in &regimes {
        eprint!("{regime:>22}");
        for &n in &report.n_ffts {
            let cell = report
                .rows
                .iter()
                .find(|r| r.regime == *regime && r.eval_n_fft == n);
            if let Some(r) = cell {
                eprint!(" {:>10.2e}", r.roundtrip_max_err);
            } else {
                eprint!(" {:>10}", "—");
            }
        }
        eprintln!();
    }
    eprintln!("\nTotal study time: {:.1} ms\n", report.elapsed_ms);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MultiTrainSchedule;

    fn test_cfg(steps: usize, schedules: Vec<MultiTrainSchedule>) -> MultiTrainConfig {
        MultiTrainConfig {
            n_ffts: vec![64, 128],
            batch: 4,
            steps,
            schedules,
            lr: 5e-4,
            spectrum_weight: 1.0,
            seed: 1,
            log_every: 0,
            eval_batches: 2,
            out_dir: None,
            until_converged: false,
            min_steps: 300,
            converge_every: 25,
            converge_patience: 5,
            converge_delta: 1e-4,
            grad_clip: 1.0,
            project_twiddles: true,
            use_fused_train: true,
            optimizer: TwiddleOptimizer::Sgd,
        }
    }

    #[test]
    fn multi_train_single_schedule() {
        let report = run_multi_train(&test_cfg(40, vec![MultiTrainSchedule::Single])).unwrap();
        assert!(report.rows.iter().any(|r| r.regime == "single_64"));
        assert!(report.rows.iter().any(|r| r.regime == "single_128"));
        for &n in &[64usize, 128] {
            let best = report
                .rows
                .iter()
                .filter(|r| r.eval_n_fft == n && r.regime.starts_with("single_"))
                .map(|r| r.roundtrip_max_err)
                .fold(f32::INFINITY, f32::min);
            assert!(best < 0.5, "n={n} single train rt_max={best}");
        }
    }

    #[test]
    fn mixed_round_robin_runs() {
        let report = run_multi_train(&test_cfg(20, vec![MultiTrainSchedule::RoundRobin])).unwrap();
        assert!(report.rows.iter().any(|r| r.regime == "mixed_round_robin"));
    }

    #[test]
    fn convergence_stops_early() {
        let mut cfg = test_cfg(2000, vec![MultiTrainSchedule::Single]);
        cfg.n_ffts = vec![64];
        cfg.until_converged = true;
        cfg.min_steps = 20;
        cfg.converge_every = 10;
        cfg.converge_patience = 2;
        cfg.converge_delta = 1e-2;
        let report = run_multi_train(&cfg).unwrap();
        let row = report
            .rows
            .iter()
            .find(|r| r.regime == "single_64")
            .expect("single_64");
        assert!(row.converged, "expected early convergence");
        assert!(
            row.train_steps_total < cfg.steps,
            "expected fewer than max steps"
        );
    }

    #[test]
    fn fused_single_1024_stays_finite() {
        let mut cfg = test_cfg(80, vec![MultiTrainSchedule::Single]);
        cfg.n_ffts = vec![1024];
        cfg.until_converged = false;
        cfg.lr = 1e-4;
        cfg.use_fused_train = true;
        cfg.optimizer = TwiddleOptimizer::Adam;
        cfg.project_twiddles = true;
        let report = run_multi_train(&cfg).unwrap();
        let row = report
            .rows
            .iter()
            .find(|r| r.regime == "single_1024")
            .expect("single_1024");
        assert!(
            row.roundtrip_max_err.is_finite(),
            "rt_max={}",
            row.roundtrip_max_err
        );
        assert!(row.encoder_spectrum_max_err.is_finite());
    }
}
