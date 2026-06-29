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

//! CLI for training, evaluation, and benchmarking learned FFT / IFFT.

use crate::ablation::{print_ablation_table, run_ablation, tier_summary, write_ablation_json};
use crate::ablation_html::{read_ablation_json, write_ablation_html};
use crate::ablation_ternary::{
    TernaryAblationOpts, print_ternary_ablation_table, quick_ablation_opts, run_ternary_ablation,
    write_ternary_ablation_csv, write_ternary_ablation_json,
};
use crate::ablation_ternary_html::write_ternary_ablation_html;
use crate::bench::bench_all;
use crate::bench_encdec::{
    bench_exact_baseline, bench_phased_dir, print_encdec_bench_table, write_encdec_bench_json,
};
use crate::bench_sweep::{
    available_devices, parse_batch_spec, parse_csv_usize, parse_k_spec, print_sweep_chart,
    run_sweep, sweep_markdown_chart, write_sweep_json,
};
use crate::bench_sweep_html::{read_sweep_json, write_sweep_html};
use crate::config::{
    EncDecTrainConfig, FftLearnConfig, MultiTrainConfig, MultiTrainSchedule, PhasedTrainConfig,
    SUPPORTED_N_FFT, TrainConfig, TransformDir, parse_n_fft, parse_transform_dir,
};
use crate::e2e_bench_html::{read_e2e_json, write_e2e_html};
use crate::learned_model::FastLearnedFftModel;
use crate::runner::FftLearnRunner;
use crate::study_html::write_study_html;
use crate::train::{evaluate_weights_dir, random_complex_batch, train_butterfly_dir, train_encdec};
use crate::train_multi::{print_multi_train_table, run_multi_train};
use crate::train_multi_html::write_multi_train_html;
use crate::train_phased::train_phased_encdec;
use crate::weights::load_safetensors;
use anyhow::{Context, Result, bail, ensure};
use rand::prelude::*;
use rlx_cli::{parse_device, req};
use rlx_runtime::Device;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let args = apply_global_precision(args)?;
    let args = args.as_slice();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print_help();
        return Ok(());
    }
    match args[0].as_str() {
        "train" => cmd_train(&args[1..]),
        "eval" => cmd_eval(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        "compare" => cmd_compare(&args[1..]),
        "roundtrip" => cmd_roundtrip(&args[1..]),
        "train-encdec" => cmd_train_encdec(&args[1..]),
        "train-phased" => cmd_train_phased(&args[1..]),
        "train-multi" => cmd_train_multi(&args[1..]),
        "bench-phased" => cmd_bench_phased(&args[1..]),
        "bench-sweep" => cmd_bench_sweep(&args[1..]),
        "max-batch" => cmd_max_batch(&args[1..]),
        "report-html" => cmd_report_html(&args[1..]),
        "study-report" => cmd_study_report(&args[1..]),
        "ablation" => cmd_ablation(&args[1..]),
        "ablation-ternary" => cmd_ablation_ternary(&args[1..]),
        "train-e2e" => cmd_train_e2e(&args[1..]),
        "train-matryoshka" => cmd_train_matryoshka(&args[1..]),
        "precision-sweep" => cmd_precision_sweep(&args[1..]),
        "train-distill" => cmd_train_distill(&args[1..]),
        "train-distill-ternary" => cmd_train_distill_ternary(&args[1..]),
        "bench-e2e" => cmd_bench_e2e(&args[1..]),
        "bench-welch-peaks" => cmd_bench_welch_peaks(&args[1..]),
        #[cfg(feature = "dev")]
        "bench-fusion-phases" => cmd_bench_fusion_phases(&args[1..]),
        #[cfg(not(feature = "dev"))]
        "bench-fusion-phases" => {
            bail!("bench-fusion-phases requires --features dev (e.g. --features dev,apple-silicon)")
        }
        other => bail!("unknown subcommand: {other} (try --help)"),
    }
}

/// Strip a global `--precision f32|f16|bf16` flag (accepted anywhere in the
/// arg list) and apply it to the FFT compile path. Every subcommand that
/// compiles a graph through `try_compile_graph*` then honors it — `f16` is what
/// pushes the butterfly/mask/denoiser graph onto fp16 silicon (ANE/Metal/MLX).
fn apply_global_precision(args: &[String]) -> Result<Vec<String>> {
    use crate::compile::{parse_precision, set_compile_precision};
    let mut out = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--precision" {
            let val = args
                .get(i + 1)
                .context("--precision needs a value (f32|f16|bf16)")?;
            set_compile_precision(parse_precision(val)?);
            i += 2;
        } else if let Some(val) = args[i].strip_prefix("--precision=") {
            set_compile_precision(parse_precision(val)?);
            i += 1;
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    Ok(out)
}

#[cfg(feature = "dev")]
const FUSION_PHASES_HELP: &str = "           bench-fusion-phases [--n-fft N] [--batch B] [--k K] [--iters N] [--device auto|cpu|metal|…] [--json PATH]\n";

#[cfg(not(feature = "dev"))]
const FUSION_PHASES_HELP: &str = "";

fn print_help() {
    eprintln!(
        "rlx-fft — learned butterfly FFT / IFFT\n\
         \n\
         Subcommands:\n\
           train     --n-fft N [--batch B] [--steps N] [--lr F] [--device auto|cpu|metal|…] [--out DIR] [--ifft]\n\
           eval      --weights PATH --n-fft N [--batch B] [--batches N] [--ifft]\n\
           bench     --n-fft N [--batch B] [--iters N] [--weights PATH] [--device auto|cpu|metal|…] [--with-butterfly-compiled] [--ifft]\n\
           compare   --n-fft N [--batch B] [--weights PATH] [--device cpu|metal|…] [--ifft]\n\
           roundtrip --n-fft N [--batch B]   # fft → ifft roundtrip vs n_fft scaling\n\
           train-encdec --n-fft N [--batch B] [--steps N] [--lr F] [--device auto|cpu|metal|…] [--spectrum-weight F] [--out DIR]\n\
           train-phased --n-fft N [--batch B] [--encoder-steps N] [--decoder-steps N] [--joint-steps N] [--out DIR]\n\
           train-multi --n-fft 64,128,256 [--batch B] [--steps MAX] [--min-steps N] [--until-converged|--fixed-steps] [--schedules …] [--optimizer adam|sgd|diag_precond] [--grad-clip F] [--no-project-twiddles] [--eager-train] [--out DIR] [--json PATH] [--html PATH]\n\
           bench-phased --dir DIR --n-fft N [--batch B] [--iters N] [--device auto|cpu|metal|…] [--json PATH]\n\
           bench-sweep [--n-fft 64,256,…] [--batch 1,8,32,64,…] [--devices cpu,metal] [--all] [--iters N] [--both-dirs] [--with-butterfly-compiled] [--json PATH] [--md PATH] [--html PATH]\n\
           max-batch [--n-fft N] [--devices cpu,metal,wgpu] [--dtype f16|f32|f64] [--limbs K]   # largest safe batch per device (memory + dispatch bounded)\n\
           report-html --json PATH --html PATH   # HTML from sweep, ablation, multi-train, or e2e JSON\n\
           study-report [--ablation-csv-dir DIR | --ablation-json PATH] [--ablation-csv-out DIR] [--train-json PATH] [--html PATH] [--run-ablation|--limit-sweep …] [--run-model-studies]\n\
           ablation [--n-fft 256,1024] [--batch 8,64,256] [--devices cpu,metal] [--iters N] [--train-steps N] [--with-compiled] [--both-dirs] [--forward-only] [--with-welch|--no-welch] [--json PATH] [--csv-dir DIR] [--html PATH]\n\
           ablation-ternary [--quick] [--n-fft 128,256] [--batch 8,32] [--devices auto|cpu,metal,…] [--iters N] [--teacher-steps N] [--distill-steps N] [--ternary-steps N] [--json PATH] [--csv PATH] [--html PATH]\n\
           train-e2e [--n-fft N] [--batch B] [--n-mels M] [--steps N] [--lr F] [--gate-lr F] [--sparsity-weight F] [--mel-weight F] [--welch-weight F] [--peak-weight F] [--peak-k K] [--spectrum-weight F] [--log-every N] [--no-q8] [--json PATH]\n\
           train-matryoshka [--n-fft N] [--batch B] [--steps N] [--lr F] [--log-every N] [--f16-weight F] [--perturb F] [--seed N] [--json PATH] [--html PATH]   # QAT nested f16/f32 twiddles + reconstruction curve\n\
           precision-sweep [--n-fft 256,1024] [--batch B] [--max-iters N] [--seed N] [--json PATH]   # f16-only precision recovery via iterative refinement\n\
           train-distill [--n-fft N] [--batch B] [--n-mels M] [--steps N] [--lr F] [--teacher-steps N] [--json PATH]\n\
           train-distill-ternary [--n-fft N] [--batch B] [--n-mels M] [--steps N] [--compute-weight F] [--teacher-steps N] [--json PATH]\n\
           bench-e2e [--n-fft N] [--batch B[,B2…]|1-1024] [--n-mels M] [--peak-k K] [--iters N] [--device all|cpu,metal,mlx,wgpu,wgu|apple-silicon|…] [--train-first] [--distill-first] [--ternary-distill] [--steps N] [--distill-steps N] [--compute-weight F] [--no-hard-gates] [--no-compiled] [--no-distilled] [--no-ternary-distilled] [--with-eager-learned] [--json PATH] [--html PATH]\n\
           bench-welch-peaks [--n-fft N] [--batch B[,B2…]|32-8192] [--k K[,K2…]|4-64] [--iters N] [--device auto|cpu|metal|…] [--strategy auto|ultra|fast|rlx|learned] [--train-steps N] [--seed N] [--no-compiled] [--no-ultra-fast] [--json PATH]\n\
           {FUSION_PHASES_HELP}\
         \n\
         Global flags:\n\
           --precision f32|f16|bf16   # float dtype for compiled graphs (default f32; f16 → ANE/half silicon)\n\
         \n\
         Supported n_fft: {:?}",
        crate::config::SUPPORTED_N_FFT
    );
}

fn parse_dir_flag(args: &[String], i: &mut usize) -> Result<Option<TransformDir>> {
    if *i < args.len() && (args[*i] == "--ifft" || args[*i] == "--inverse") {
        *i += 1;
        return Ok(Some(TransformDir::Inverse));
    }
    Ok(None)
}

fn cmd_precision_sweep(args: &[String]) -> Result<()> {
    let mut n_ffts = vec![256usize];
    let mut batch = 16usize;
    let mut max_iters = 3usize;
    let mut seed = 7u64;
    let mut json: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_ffts = parse_csv_usize(&req(args, &mut i)?, "--n-fft")?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--max-iters" => max_iters = req(args, &mut i)?.parse().context("--max-iters")?,
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--json" => json = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let mut reports = Vec::new();
    for &n in &n_ffts {
        let report = crate::precision_fft::precision_sweep(n, batch, max_iters, seed)?;
        println!("\nn_fft={n} batch={batch}  (precision recovery on f16-only hardware)");
        println!(
            "  {:<18} {:>12} {:>8}  {}",
            "scheme", "max_err", "passes", "output"
        );
        for row in &report.rows {
            let passes = if row.passes.is_nan() {
                "  —".to_string()
            } else {
                format!("{:.0}", row.passes)
            };
            println!(
                "  {:<18} {:>12} {:>8}  {}",
                row.scheme,
                format!("{:.3e}", row.max_err),
                passes,
                if row.two_limb_out {
                    "f16×2 (pair)"
                } else {
                    "f16/f32"
                }
            );
        }
        reports.push(report);
    }

    if let Some(path) = json {
        std::fs::write(&path, serde_json::to_string_pretty(&reports)?)
            .with_context(|| format!("write {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn cmd_train_matryoshka(args: &[String]) -> Result<()> {
    let mut n_fft = 256usize;
    let mut batch = 16usize;
    let mut steps = 400usize;
    let mut lr = 5e-3f32;
    let mut log_every = 20usize;
    let mut f16_weight = 1.0f32;
    let mut perturb = 0.0f32;
    let mut seed = 42u64;
    let mut json: Option<PathBuf> = None;
    let mut html: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--steps" => steps = req(args, &mut i)?.parse().context("--steps")?,
            "--lr" => lr = req(args, &mut i)?.parse().context("--lr")?,
            "--log-every" => log_every = req(args, &mut i)?.parse().context("--log-every")?,
            "--f16-weight" => f16_weight = req(args, &mut i)?.parse().context("--f16-weight")?,
            "--perturb" => perturb = req(args, &mut i)?.parse().context("--perturb")?,
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--json" => json = Some(req(args, &mut i)?.into()),
            "--html" => html = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let result = crate::matryoshka::train_matryoshka(
        n_fft, batch, steps, lr, log_every, f16_weight, perturb, seed,
    )?;
    let base = &result.points[0];
    let last = result.points.last().unwrap();
    eprintln!(
        "matryoshka n_fft={n_fft} batch={batch} steps={steps} f16_weight={f16_weight}\n\
         {:<14} {:>12} -> {:>12} ({:>5.1}x)\n\
         {:<14} {:>12} -> {:>12} ({:>5.1}x)\n\
         {:<14} {:>12} -> {:>12} ({:>5.1}x)\n\
         {:<14} {:>12} -> {:>12} ({:>5.1}x)",
        "encoder f16",
        format!("{:.3e}", base.enc_err_f16),
        format!("{:.3e}", last.enc_err_f16),
        base.enc_err_f16 / last.enc_err_f16.max(1e-12),
        "decoder f16",
        format!("{:.3e}", base.recon_err_f16),
        format!("{:.3e}", last.recon_err_f16),
        base.recon_err_f16 / last.recon_err_f16.max(1e-12),
        "encoder f32",
        format!("{:.3e}", base.enc_err_f32),
        format!("{:.3e}", last.enc_err_f32),
        base.enc_err_f32 / last.enc_err_f32.max(1e-12),
        "decoder f32",
        format!("{:.3e}", base.recon_err_f32),
        format!("{:.3e}", last.recon_err_f32),
        base.recon_err_f32 / last.recon_err_f32.max(1e-12),
    );

    if let Some(path) = json {
        std::fs::write(&path, serde_json::to_string_pretty(&result)?)
            .with_context(|| format!("write {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = html {
        std::fs::write(&path, crate::matryoshka::render_curve_html(&result))
            .with_context(|| format!("write {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn cmd_train(args: &[String]) -> Result<()> {
    let mut n_fft = 64usize;
    let mut batch = 8usize;
    let mut steps = 500usize;
    let mut lr = 1e-3f64;
    let mut seed = 42u64;
    let mut log_every = 50usize;
    let mut device = "cpu".to_string();
    let mut out: Option<PathBuf> = None;
    let mut dir = TransformDir::Forward;

    let mut i = 0;
    while i < args.len() {
        if let Some(d) = parse_dir_flag(args, &mut i)? {
            dir = d;
            continue;
        }
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--steps" => steps = req(args, &mut i)?.parse().context("--steps")?,
            "--lr" => lr = req(args, &mut i)?.parse().context("--lr")?,
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--log-every" => log_every = req(args, &mut i)?.parse().context("--log-every")?,
            "--device" => device = req(args, &mut i)?,
            "--direction" => dir = parse_transform_dir(&req(args, &mut i)?)?,
            "--out" => out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let cfg = TrainConfig {
        model: FftLearnConfig::new(n_fft, batch)?,
        direction: dir,
        steps,
        lr,
        seed,
        log_every,
        device,
        out_dir: out,
        ..TrainConfig::default()
    };
    let report = train_butterfly_dir(&cfg, dir)?;
    eprintln!(
        "train {:?} done: mse={:.6e} max_err={:.6e} steps={} elapsed_ms={:.1}",
        dir, report.final_mse, report.max_error, report.steps, report.elapsed_ms
    );
    Ok(())
}

fn cmd_eval(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut n_fft = 64usize;
    let mut batch = 8usize;
    let mut batches = 32usize;
    let mut dir = TransformDir::Forward;

    let mut i = 0;
    while i < args.len() {
        if let Some(d) = parse_dir_flag(args, &mut i)? {
            dir = d;
            continue;
        }
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--batches" => batches = req(args, &mut i)?.parse().context("--batches")?,
            "--direction" => dir = parse_transform_dir(&req(args, &mut i)?)?,
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights_path = weights.context("--weights PATH required")?;
    let store = load_safetensors(&weights_path)?;
    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let (mse, max_err) = evaluate_weights_dir(&store, &cfg, batches, dir)?;
    eprintln!(
        "eval {:?}: mse={mse:.6e} max_err={max_err:.6e} ({batches} batches)",
        dir
    );
    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<()> {
    let mut n_fft = 256usize;
    let mut batch = 32usize;
    let mut iters = 200usize;
    let mut device = "auto".to_string();
    let mut with_butterfly_compiled = false;
    let mut weights: Option<PathBuf> = None;
    let mut dir = TransformDir::Forward;

    let mut i = 0;
    while i < args.len() {
        if let Some(d) = parse_dir_flag(args, &mut i)? {
            dir = d;
            continue;
        }
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--iters" => iters = req(args, &mut i)?.parse().context("--iters")?,
            "--device" => device = req(args, &mut i)?,
            "--with-butterfly-compiled" => {
                with_butterfly_compiled = true;
                i += 1;
            }
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--direction" => dir = parse_transform_dir(&req(args, &mut i)?)?,
            other => bail!("unknown flag: {other}"),
        }
    }
    let report = bench_all(
        n_fft,
        batch,
        iters,
        dir,
        &device,
        with_butterfly_compiled,
        weights.as_deref(),
    )?;
    let compiled_line = if with_butterfly_compiled {
        format!(
            "\n\t butterfly rlx    {:>8.4} ms  max_err={:.3e}",
            report.butterfly_compiled_ms, report.butterfly_compiled_err
        )
    } else {
        String::new()
    };
    eprintln!(
        "bench {:?} n_fft={} batch={} iters={} device={:?} butterfly=[{}]\n\
         \t rustfft          {:>8.4} ms  (reference)\n\
         \t rlx Op::Fft      {:>8.4} ms  max_err={:.3e}\n\
         \t butterfly eager  {:>8.4} ms  max_err={:.3e}{compiled_line}",
        report.direction,
        report.n_fft,
        report.batch,
        report.iters,
        report.device,
        report.butterfly_weights,
        report.rustfft_ms,
        report.rlx_fft_ms,
        report.rlx_fft_err,
        report.butterfly_eager_ms,
        report.butterfly_eager_err,
    );
    Ok(())
}

fn cmd_compare(args: &[String]) -> Result<()> {
    let mut n_fft = 64usize;
    let mut batch = 4usize;
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut compile = false;
    let mut dir = TransformDir::Forward;

    let mut i = 0;
    while i < args.len() {
        if let Some(d) = parse_dir_flag(args, &mut i)? {
            dir = d;
            continue;
        }
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--direction" => dir = parse_transform_dir(&req(args, &mut i)?)?,
            "--compile" => {
                compile = true;
                i += 1;
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let mut runner = if let Some(path) = weights {
        let store = load_safetensors(&path)?;
        FftLearnRunner::with_weights_dir(cfg, &store, dir)?
    } else {
        FftLearnRunner::new_dir(cfg, dir)?
    };

    if compile {
        let dev: Device = parse_device(&device)?;
        runner.load_compiled(dev)?;
        eprintln!("compiled on {dev:?}");
    }

    let input: Vec<f32> = if dir.is_forward() {
        (0..runner.config().batch * runner.config().n_fft)
            .map(|i| (i as f32 * 0.13).sin())
            .collect()
    } else {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        random_complex_batch(&mut rng, runner.config().batch, runner.config().n_fft)
    };
    let (mse, max_err) = runner.compare_reference(&input)?;
    eprintln!("compare {:?}: mse={mse:.6e} max_err={max_err:.6e}", dir);
    Ok(())
}

fn cmd_train_encdec(args: &[String]) -> Result<()> {
    let mut n_fft = 64usize;
    let mut batch = 8usize;
    let mut steps = 500usize;
    let mut lr = 1e-3f64;
    let mut seed = 42u64;
    let mut log_every = 50usize;
    let mut spectrum_weight = 1.0f32;
    let mut device = "auto".to_string();
    let mut out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--steps" => steps = req(args, &mut i)?.parse().context("--steps")?,
            "--lr" => lr = req(args, &mut i)?.parse().context("--lr")?,
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--log-every" => log_every = req(args, &mut i)?.parse().context("--log-every")?,
            "--device" => device = req(args, &mut i)?,
            "--spectrum-weight" => {
                spectrum_weight = req(args, &mut i)?.parse().context("--spectrum-weight")?
            }
            "--out" => out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let cfg = EncDecTrainConfig {
        model: FftLearnConfig::new(n_fft, batch)?,
        steps,
        lr,
        spectrum_weight,
        seed,
        log_every,
        device,
        out_dir: out,
        ..EncDecTrainConfig::default()
    };
    let report = train_encdec(&cfg)?;
    eprintln!(
        "train-encdec done: recon_mse={:.6e} spectrum_mse={:.6e} roundtrip_max_err={:.6e} steps={} elapsed_ms={:.1}",
        report.reconstruction_mse,
        report.spectrum_mse,
        report.roundtrip_max_error,
        report.steps,
        report.elapsed_ms
    );
    Ok(())
}

fn cmd_roundtrip(args: &[String]) -> Result<()> {
    let mut n_fft = 64usize;
    let mut batch = 4usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            other => bail!("unknown flag: {other}"),
        }
    }

    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let fft = FftLearnRunner::new(cfg.clone())?;
    let ifft = FftLearnRunner::new_ifft(cfg)?;
    let signal: Vec<f32> = (0..batch * n_fft)
        .map(|i| (i as f32 * 0.11).sin())
        .collect();
    let spectrum = fft.forward_eager(&signal)?;
    let recovered = ifft.forward_eager(&spectrum)?;
    let scale = crate::reference::roundtrip_scale(n_fft);
    let mut max_err = 0f32;
    for b in 0..batch {
        for i in 0..n_fft {
            let base = b * n_fft * 2 + i * 2;
            let expected_re = signal[b * n_fft + i] * scale;
            max_err = max_err.max((recovered[base] - expected_re).abs());
            max_err = max_err.max(recovered[base + 1].abs());
        }
    }
    eprintln!("roundtrip: max_err={max_err:.6e} (expect ~0; scale={scale})");
    Ok(())
}

fn cmd_train_phased(args: &[String]) -> Result<()> {
    let mut n_fft = 64usize;
    let mut batch = 8usize;
    let mut encoder_steps = 300usize;
    let mut decoder_steps = 300usize;
    let mut joint_steps = 300usize;
    let mut lr = 5e-4f64;
    let mut spectrum_weight = 1.0f32;
    let mut seed = 42u64;
    let mut log_every = 50usize;
    let mut out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--encoder-steps" => {
                encoder_steps = req(args, &mut i)?.parse().context("--encoder-steps")?
            }
            "--decoder-steps" => {
                decoder_steps = req(args, &mut i)?.parse().context("--decoder-steps")?
            }
            "--joint-steps" => joint_steps = req(args, &mut i)?.parse().context("--joint-steps")?,
            "--lr" => lr = req(args, &mut i)?.parse().context("--lr")?,
            "--spectrum-weight" => {
                spectrum_weight = req(args, &mut i)?.parse().context("--spectrum-weight")?
            }
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--log-every" => log_every = req(args, &mut i)?.parse().context("--log-every")?,
            "--out" => out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let out_dir = out.context("--out DIR required for train-phased")?;
    let cfg = PhasedTrainConfig {
        model: FftLearnConfig::new(n_fft, batch)?,
        encoder_steps,
        decoder_steps,
        joint_steps,
        lr,
        spectrum_weight,
        seed,
        log_every,
        out_dir: Some(out_dir),
    };
    let report = train_phased_encdec(&cfg)?;
    for p in &report.phases {
        eprintln!(
            "[{}] steps={} train_ms={:.1} enc_max={:.3e} dec_max={:.3e} rt_max={:.3e} -> {}",
            p.name,
            p.steps,
            p.elapsed_ms,
            p.encoder_spectrum_max_err,
            p.decoder_time_max_err,
            p.roundtrip_max_err,
            p.checkpoint.display()
        );
    }
    eprintln!(
        "train-phased done: {} phases, total_ms={:.1}",
        report.phases.len(),
        report.total_elapsed_ms
    );
    Ok(())
}

fn cmd_train_multi(args: &[String]) -> Result<()> {
    let mut n_fft_csv = "64,128,256".to_string();
    let mut batch = 8usize;
    let mut steps = 10_000usize;
    let mut min_steps = 300usize;
    let mut until_converged = true;
    let mut converge_every = 25usize;
    let mut converge_patience = 5usize;
    let mut converge_delta = 1e-4f32;
    let mut schedules_csv = "single,round_robin,random,balanced".to_string();
    let mut lr = 1e-4f64;
    let mut spectrum_weight = 1.0f32;
    let mut seed = 42u64;
    let mut log_every = 50usize;
    let mut eval_batches = 8usize;
    let mut out: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut html_out: Option<PathBuf> = None;
    let mut grad_clip = 1.0f32;
    let mut project_twiddles = true;
    let mut use_fused_train = true;
    let mut optimizer = "sgd".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft_csv = req(args, &mut i)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--steps" => steps = req(args, &mut i)?.parse().context("--steps")?,
            "--min-steps" => min_steps = req(args, &mut i)?.parse().context("--min-steps")?,
            "--until-converged" => {
                until_converged = true;
                i += 1;
            }
            "--fixed-steps" => {
                until_converged = false;
                i += 1;
            }
            "--converge-every" => {
                converge_every = req(args, &mut i)?.parse().context("--converge-every")?
            }
            "--converge-patience" => {
                converge_patience = req(args, &mut i)?.parse().context("--converge-patience")?
            }
            "--converge-delta" => {
                converge_delta = req(args, &mut i)?.parse().context("--converge-delta")?
            }
            "--schedules" => schedules_csv = req(args, &mut i)?,
            "--lr" => lr = req(args, &mut i)?.parse().context("--lr")?,
            "--spectrum-weight" => {
                spectrum_weight = req(args, &mut i)?.parse().context("--spectrum-weight")?
            }
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--log-every" => log_every = req(args, &mut i)?.parse().context("--log-every")?,
            "--eval-batches" => {
                eval_batches = req(args, &mut i)?.parse().context("--eval-batches")?
            }
            "--out" => out = Some(req(args, &mut i)?.into()),
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--html" => html_out = Some(req(args, &mut i)?.into()),
            "--optimizer" => optimizer = req(args, &mut i)?,
            "--grad-clip" => grad_clip = req(args, &mut i)?.parse().context("--grad-clip")?,
            "--project-twiddles" => {
                project_twiddles = true;
                i += 1;
            }
            "--no-project-twiddles" => {
                project_twiddles = false;
                i += 1;
            }
            "--fused-train" => {
                use_fused_train = true;
                i += 1;
            }
            "--eager-train" => {
                use_fused_train = false;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let n_ffts = parse_csv_usize(&n_fft_csv, "--n-fft")?;
    let schedules = MultiTrainSchedule::parse_csv(&schedules_csv)?;
    let optimizer = crate::second_order::TwiddleOptimizer::parse(&optimizer)?;
    let cfg = MultiTrainConfig {
        n_ffts,
        batch,
        steps,
        schedules,
        lr,
        spectrum_weight,
        seed,
        log_every,
        eval_batches,
        out_dir: out,
        until_converged,
        min_steps,
        converge_every,
        converge_patience,
        converge_delta,
        grad_clip,
        project_twiddles,
        use_fused_train,
        optimizer,
    };

    let report = run_multi_train(&cfg)?;
    print_multi_train_table(&report);
    let winners = crate::train_multi::best_regime_per_eval(&report);
    eprintln!("Best learned regime per eval n_fft: {winners:?}");

    if let Some(path) = json_out {
        crate::train_multi::write_multi_train_json(&path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = html_out {
        write_multi_train_html(&path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn cmd_bench_phased(args: &[String]) -> Result<()> {
    let mut dir: Option<PathBuf> = None;
    let mut n_fft = 64usize;
    let mut batch = 8usize;
    let mut iters = 100usize;
    let mut device = "auto".to_string();
    let mut with_compiled = false;
    let mut json_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => dir = Some(req(args, &mut i)?.into()),
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch")?,
            "--iters" => iters = req(args, &mut i)?.parse().context("--iters")?,
            "--device" => device = req(args, &mut i)?,
            "--with-compiled" => {
                with_compiled = true;
                i += 1;
            }
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let phased_dir = dir.context("--dir PATH required")?;
    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let mut rows = vec![bench_exact_baseline(&cfg, iters, &device)?];
    rows.extend(bench_phased_dir(
        &phased_dir,
        &cfg,
        iters,
        &device,
        with_compiled,
    )?);
    print_encdec_bench_table(&rows);
    if let Some(path) = json_out {
        write_encdec_bench_json(&path, &rows)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn default_n_fft_csv() -> String {
    SUPPORTED_N_FFT
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn default_batch_csv() -> String {
    "1,8,32,64,128,256,512,1024,2048,4096".to_string()
}

fn cmd_bench_sweep(args: &[String]) -> Result<()> {
    let mut n_fft_csv = default_n_fft_csv();
    let mut batch_csv = default_batch_csv();
    let mut devices_csv: Option<String> = None;
    let mut iters = 30usize;
    let mut both_dirs = false;
    let mut sweep_all = false;
    let mut with_butterfly_compiled = false;
    let mut weights: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut md_out: Option<PathBuf> = None;
    let mut html_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft_csv = req(args, &mut i)?,
            "--batch" => batch_csv = req(args, &mut i)?,
            "--devices" => devices_csv = Some(req(args, &mut i)?),
            "--iters" => iters = req(args, &mut i)?.parse().context("--iters")?,
            "--both-dirs" => {
                both_dirs = true;
                i += 1;
            }
            "--all" => {
                sweep_all = true;
                i += 1;
            }
            "--with-butterfly-compiled" => {
                with_butterfly_compiled = true;
                i += 1;
            }
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--md" => md_out = Some(req(args, &mut i)?.into()),
            "--html" => html_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                eprintln!(
                    "\nAvailable backends on this build: {:?}",
                    available_devices()
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    if sweep_all {
        n_fft_csv = default_n_fft_csv();
        batch_csv = default_batch_csv();
        both_dirs = true;
    }

    let n_ffts = parse_csv_usize(&n_fft_csv, "--n-fft")?;
    let batches = parse_csv_usize(&batch_csv, "--batch")?;
    let device_names = if let Some(csv) = devices_csv {
        crate::device::parse_bench_device_list(&csv)?
    } else {
        available_devices()
            .into_iter()
            .map(str::to_string)
            .collect()
    };
    let device_refs: Vec<&str> = device_names.iter().map(String::as_str).collect();

    let directions = if both_dirs {
        vec![TransformDir::Forward, TransformDir::Inverse]
    } else {
        vec![TransformDir::Forward]
    };

    eprintln!(
        "bench-sweep: n_fft={n_ffts:?} batch={batches:?} devices={device_names:?} dirs={directions:?} iters={iters} compiled={with_butterfly_compiled}"
    );

    let report = run_sweep(
        &n_ffts,
        &batches,
        &device_refs,
        &directions,
        iters,
        with_butterfly_compiled,
        weights.as_deref(),
    )?;
    print_sweep_chart(&report);

    if let Some(path) = json_out {
        write_sweep_json(&path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = md_out {
        std::fs::write(&path, sweep_markdown_chart(&report))
            .with_context(|| format!("write {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = html_out {
        write_sweep_html(&path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

/// Report the largest FFT batch each device can run for a given `(n_fft, dtype,
/// limbs)`, detecting the memory + dispatch ceilings from the live machine.
fn cmd_max_batch(args: &[String]) -> Result<()> {
    use crate::max_batch::{FftProblem, auto_max_fft_batch, detect_device_caps};

    let mut n_fft = 256usize;
    let mut devices_csv: Option<String> = None;
    let mut elem_bytes = 4usize; // f32
    let mut limbs = 1usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft = req(args, &mut i)?.parse().context("--n-fft")?,
            "--devices" => devices_csv = Some(req(args, &mut i)?),
            "--dtype" => {
                elem_bytes = match req(args, &mut i)?.trim().to_ascii_lowercase().as_str() {
                    "f16" | "bf16" => 2,
                    "f32" => 4,
                    "f64" => 8,
                    other => bail!("--dtype f16|f32|f64 (got {other})"),
                }
            }
            "--limbs" => limbs = req(args, &mut i)?.parse().context("--limbs")?,
            "--help" | "-h" => {
                eprintln!(
                    "max-batch [--n-fft N] [--devices cpu,metal,wgpu] [--dtype f16|f32|f64] [--limbs K]\n\
                     Reports the largest safe batch per device (memory- and dispatch-bounded).\n\
                     Env: RLX_FFT_MEM_BUDGET_MB overrides the memory budget (discrete-GPU VRAM)."
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let device_names = if let Some(csv) = devices_csv {
        crate::device::parse_bench_device_list(&csv)?
    } else {
        available_devices()
            .into_iter()
            .map(str::to_string)
            .collect()
    };

    let problem = FftProblem::new(n_fft, elem_bytes, limbs);
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    eprintln!(
        "max-batch: n_fft={n_fft} dtype={}B limbs={limbs} row={} B/batch",
        elem_bytes,
        problem.row_footprint_bytes()
    );
    eprintln!(
        "  {:<8} {:>12} {:>11} {:>12} {:>12}  {:<10} {}",
        "device", "mem budget", "dispatch", "mem cap", "MAX BATCH", "limited by", "mem source"
    );
    for name in &device_names {
        let device = crate::device::resolve_train_device(Some(name))?;
        let caps = detect_device_caps(device);
        let cap = auto_max_fft_batch(device, problem);
        eprintln!(
            "  {:<8} {:>10.1} GB {:>11} {:>12} {:>12}  {:<10} {}",
            name,
            mb(caps.mem_budget_bytes) / 1024.0,
            caps.dispatch_cap
                .map(|d| d.to_string())
                .unwrap_or_else(|| "—".into()),
            cap.mem_cap,
            cap.max_batch,
            format!("{:?}", cap.limited_by),
            caps.mem_source,
        );
    }
    Ok(())
}

fn cmd_report_html(args: &[String]) -> Result<()> {
    let mut json_in: Option<String> = None;
    let mut html_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => json_in = Some(req(args, &mut i)?),
            "--html" => html_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let json_spec = json_in.context("--json PATH required")?;
    let html_path = html_out.context("--html PATH required")?;
    let json_paths: Vec<PathBuf> = json_spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    ensure!(!json_paths.is_empty(), "--json PATH required");
    let json_path = json_paths[0].clone();
    let bytes =
        std::fs::read(&json_path).with_context(|| format!("read {}", json_path.display()))?;
    let root: serde_json::Value =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", json_path.display()))?;
    let is_multi_train = root.get("n_ffts").is_some()
        && root
            .get("rows")
            .and_then(|r| r.as_array())
            .and_then(|rows| rows.first())
            .and_then(|r0| r0.get("regime"))
            .is_some();
    let is_e2e = root.get("n_mels").is_some()
        && root
            .get("rows")
            .and_then(|r| r.as_array())
            .and_then(|rows| rows.first())
            .and_then(|r0| r0.get("pipeline"))
            .is_some();
    let is_ablation = !is_multi_train
        && !is_e2e
        && (root.get("train_steps").is_some()
            || root
                .get("rows")
                .and_then(|r| r.as_array())
                .and_then(|rows| rows.first())
                .and_then(|r0| r0.get("variant"))
                .is_some());
    if is_multi_train {
        let report = crate::train_multi_html::read_multi_train_json(&json_path)?;
        write_multi_train_html(&html_path, &report)?;
    } else if is_ablation {
        let report = read_ablation_json(&json_path)?;
        write_ablation_html(&html_path, &report)?;
    } else if is_e2e {
        let reports: Result<Vec<_>> = json_paths.iter().map(|p| read_e2e_json(p)).collect();
        let reports = reports?;
        let report = if reports.len() == 1 {
            reports.into_iter().next().unwrap()
        } else {
            crate::e2e_bench::merge_e2e_reports(&reports)?
        };
        write_e2e_html(&html_path, &report)?;
    } else {
        let report = read_sweep_json(&json_path)?;
        write_sweep_html(&html_path, &report)?;
    }
    eprintln!("wrote {}", html_path.display());
    Ok(())
}

fn cmd_study_report(args: &[String]) -> Result<()> {
    let mut ablation_json: Option<PathBuf> = None;
    let mut ablation_csv_dir: Option<PathBuf> = None;
    let mut ablation_csv_out: Option<PathBuf> = None;
    let mut ablation_out: Option<PathBuf> = None;
    let mut train_json: Option<PathBuf> = None;
    let mut html_out: Option<PathBuf> = None;
    let mut do_ablation = false;
    let mut limit_sweep = false;
    let mut run_model_studies = false;
    let mut model_study_n_fft = 128usize;
    let mut model_study_steps = 120usize;
    let mut n_fft_csv = "64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072".to_string();
    let mut batch_csv = "1,8,32,64,128,256,512,1024,2048,4096".to_string();
    let mut devices_csv: Option<String> = None;
    let mut iters = 15usize;
    let mut train_steps = 30usize;
    let mut with_compiled = true;
    let mut both_dirs = true;
    let mut with_welch = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ablation-json" => ablation_json = Some(req(args, &mut i)?.into()),
            "--ablation-csv-dir" => ablation_csv_dir = Some(req(args, &mut i)?.into()),
            "--ablation-csv-out" => ablation_csv_out = Some(req(args, &mut i)?.into()),
            "--ablation-out" => ablation_out = Some(req(args, &mut i)?.into()),
            "--train-json" => train_json = Some(req(args, &mut i)?.into()),
            "--html" => html_out = Some(req(args, &mut i)?.into()),
            "--run-ablation" => {
                do_ablation = true;
                i += 1;
            }
            "--limit-sweep" => {
                limit_sweep = true;
                do_ablation = true;
                i += 1;
            }
            "--run-model-studies" => {
                run_model_studies = true;
                i += 1;
            }
            "--model-study-n-fft" => {
                model_study_n_fft = req(args, &mut i)?.parse().context("--model-study-n-fft")?;
            }
            "--model-study-steps" => {
                model_study_steps = req(args, &mut i)?.parse().context("--model-study-steps")?;
            }
            "--n-fft" => n_fft_csv = req(args, &mut i)?,
            "--batch" => batch_csv = req(args, &mut i)?,
            "--devices" => devices_csv = Some(req(args, &mut i)?),
            "--iters" => iters = req(args, &mut i)?.parse().context("--iters")?,
            "--train-steps" => train_steps = req(args, &mut i)?.parse().context("--train-steps")?,
            "--with-compiled" => {
                with_compiled = true;
                i += 1;
            }
            "--eager-only" => {
                with_compiled = false;
                i += 1;
            }
            "--forward-only" => {
                both_dirs = false;
                i += 1;
            }
            "--no-welch" => {
                with_welch = false;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let html_path = html_out.unwrap_or_else(|| PathBuf::from("/tmp/rlx-fft-study.html"));

    let ablation = if do_ablation {
        let device_names: Vec<String> = if let Some(csv) = devices_csv {
            csv.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        } else if limit_sweep {
            crate::ablation::limit_sweep_devices()
        } else {
            available_devices()
                .into_iter()
                .map(str::to_string)
                .collect()
        };
        ensure!(!device_names.is_empty(), "no devices selected");
        let device_refs: Vec<&str> = device_names.iter().map(String::as_str).collect();
        if limit_sweep {
            eprintln!(
                "[study-report] limit sweep n_fft={SUPPORTED_N_FFT:?} devices={device_names:?} (cpu+gpu)"
            );
            Some(crate::ablation::run_limit_sweep(
                &device_refs,
                iters,
                train_steps,
                42,
            )?)
        } else {
            let n_ffts = parse_csv_usize(&n_fft_csv, "--n-fft")?;
            let batches = parse_csv_usize(&batch_csv, "--batch")?;
            eprintln!(
                "[study-report] running ablation n_fft={n_ffts:?} batch={batches:?} devices={device_names:?}"
            );
            Some(run_ablation(
                &n_ffts,
                &batches,
                &device_refs,
                iters,
                train_steps,
                42,
                with_compiled,
                both_dirs,
                with_welch,
            )?)
        }
    } else if let Some(path) = &ablation_csv_dir {
        eprintln!(
            "[study-report] loading ablation CSV from {}",
            path.display()
        );
        Some(crate::ablation_csv::read_ablation_csv_dir(path)?)
    } else if let Some(path) = &ablation_json {
        Some(read_ablation_json(path)?)
    } else {
        None
    };

    let multi_train = if let Some(path) = &train_json {
        Some(crate::train_multi_html::read_multi_train_json(path)?)
    } else {
        None
    };

    ensure!(
        ablation.is_some() || multi_train.is_some(),
        "provide --ablation-csv-dir / --ablation-json / --train-json, or pass --run-ablation"
    );

    let telemetry = if run_model_studies {
        eprintln!(
            "[study-report] collecting model telemetry n_fft={model_study_n_fft} steps={model_study_steps}"
        );
        Some(crate::study_collect::collect_study_telemetry(
            model_study_n_fft,
            8,
            model_study_steps,
            model_study_steps,
            42,
        )?)
    } else {
        None
    };

    if let (Some(path), Some(report)) = (&ablation_out, &ablation) {
        crate::ablation::write_ablation_json(path, report)?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(report) = &ablation {
        if do_ablation || ablation_csv_out.is_some() {
            let csv_dir = ablation_csv_out
                .clone()
                .unwrap_or_else(|| PathBuf::from("/tmp/rlx-fft-study-csv"));
            crate::ablation_csv::write_ablation_csv_dir(&csv_dir, report)?;
        }
    }

    let inputs = crate::study_html::StudyInputs {
        ablation,
        multi_train,
        telemetry,
    };
    write_study_html(&html_path, &inputs)?;
    eprintln!("wrote {}", html_path.display());
    Ok(())
}

fn cmd_ablation_ternary(args: &[String]) -> Result<()> {
    let mut opts = TernaryAblationOpts::default();
    let mut json_out: Option<PathBuf> = None;
    let mut csv_out: Option<PathBuf> = None;
    let mut html_out: Option<PathBuf> = None;
    let mut n_fft_csv: Option<String> = None;
    let mut batch_csv: Option<String> = None;
    let mut devices_csv: Option<String> = None;
    let mut prune_csv: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--quick" => {
                opts = quick_ablation_opts();
                i += 1;
            }
            "--n-fft" => n_fft_csv = Some(req(args, &mut i)?),
            "--batch" => batch_csv = Some(req(args, &mut i)?),
            "--devices" => devices_csv = Some(req(args, &mut i)?),
            "--iters" => opts.iters = req(args, &mut i)?.parse().context("--iters")?,
            "--teacher-steps" => {
                opts.teacher_steps = req(args, &mut i)?.parse().context("--teacher-steps")?
            }
            "--distill-steps" => {
                opts.distill_steps = req(args, &mut i)?.parse().context("--distill-steps")?
            }
            "--ternary-steps" => {
                opts.ternary_steps = req(args, &mut i)?.parse().context("--ternary-steps")?
            }
            "--prune-targets" => prune_csv = Some(req(args, &mut i)?),
            "--seed" => opts.seed = req(args, &mut i)?.parse().context("--seed")?,
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--csv" => csv_out = Some(req(args, &mut i)?.into()),
            "--html" => html_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    if let Some(csv) = n_fft_csv {
        opts.n_ffts = parse_csv_usize(&csv, "--n-fft")?;
    }
    if let Some(csv) = batch_csv {
        opts.batches = parse_csv_usize(&csv, "--batch")?;
    }
    if let Some(csv) = prune_csv {
        opts.prune_targets = csv
            .split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();
        ensure!(!opts.prune_targets.is_empty(), "--prune-targets");
    }
    if let Some(csv) = devices_csv {
        if csv == "auto" {
            opts.devices = available_devices()
                .into_iter()
                .map(str::to_string)
                .collect();
        } else {
            opts.devices = crate::device::parse_bench_device_list(&csv)?;
        }
    }

    let report = run_ternary_ablation(&opts)?;
    print_ternary_ablation_table(&report);
    if let Some(path) = json_out {
        write_ternary_ablation_json(&path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = csv_out {
        write_ternary_ablation_csv(&path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = html_out {
        write_ternary_ablation_html(&path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn cmd_ablation(args: &[String]) -> Result<()> {
    let mut n_fft_csv = "256,1024,4096".to_string();
    let mut batch_csv = "8,64,256,512,1024,2048,4096".to_string();
    let mut devices_csv: Option<String> = None;
    let mut iters = 20usize;
    let mut train_steps = 40usize;
    let mut seed = 42u64;
    let mut with_compiled = false;
    let mut both_dirs = true;
    let mut with_welch = true;
    let mut json_out: Option<PathBuf> = None;
    let mut csv_dir_out: Option<PathBuf> = None;
    let mut html_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft_csv = req(args, &mut i)?,
            "--batch" => batch_csv = req(args, &mut i)?,
            "--devices" => devices_csv = Some(req(args, &mut i)?),
            "--iters" => iters = req(args, &mut i)?.parse().context("--iters")?,
            "--train-steps" => train_steps = req(args, &mut i)?.parse().context("--train-steps")?,
            "--with-compiled" => {
                with_compiled = true;
                i += 1;
            }
            "--both-dirs" => {
                both_dirs = true;
                i += 1;
            }
            "--forward-only" => {
                both_dirs = false;
                i += 1;
            }
            "--with-welch" => {
                with_welch = true;
                i += 1;
            }
            "--no-welch" => {
                with_welch = false;
                i += 1;
            }
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--csv-dir" => csv_dir_out = Some(req(args, &mut i)?.into()),
            "--html" => html_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let n_ffts = parse_csv_usize(&n_fft_csv, "--n-fft")?;
    let batches = parse_csv_usize(&batch_csv, "--batch")?;
    let device_names = if let Some(csv) = devices_csv {
        crate::device::parse_bench_device_list(&csv)?
    } else {
        available_devices()
            .into_iter()
            .map(str::to_string)
            .collect()
    };
    let device_refs: Vec<&str> = device_names.iter().map(String::as_str).collect();

    let report = run_ablation(
        &n_ffts,
        &batches,
        &device_refs,
        iters,
        train_steps,
        seed,
        with_compiled,
        both_dirs,
        with_welch,
    )?;
    print_ablation_table(&report);
    let wins = tier_summary(&report);
    eprintln!("Tier win counts (fastest variant per cell): {wins:?}");
    let emit_csv = json_out.is_some() || csv_dir_out.is_some() || html_out.is_some();
    if let Some(path) = &json_out {
        write_ablation_json(path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    if emit_csv {
        let csv_dir = csv_dir_out
            .clone()
            .unwrap_or_else(|| PathBuf::from("/tmp/rlx-fft-ablation-csv"));
        crate::ablation_csv::write_ablation_csv_dir(&csv_dir, &report)?;
    }
    if let Some(path) = html_out {
        write_ablation_html(&path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn cmd_train_e2e(args: &[String]) -> Result<()> {
    let mut cfg = crate::train_e2e::E2eTrainConfig::default();
    let mut json_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => cfg.n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => cfg.batch = req(args, &mut i)?.parse().context("--batch")?,
            "--n-mels" => cfg.n_mels = req(args, &mut i)?.parse().context("--n-mels")?,
            "--steps" => cfg.steps = req(args, &mut i)?.parse().context("--steps")?,
            "--lr" => cfg.lr = req(args, &mut i)?.parse().context("--lr")?,
            "--sparsity-weight" => {
                cfg.sparsity_weight = req(args, &mut i)?.parse().context("--sparsity-weight")?
            }
            "--gate-lr" => cfg.gate_lr = req(args, &mut i)?.parse().context("--gate-lr")?,
            "--mel-weight" => {
                cfg.mel_weight = req(args, &mut i)?.parse().context("--mel-weight")?
            }
            "--welch-weight" => {
                cfg.welch_weight = req(args, &mut i)?.parse().context("--welch-weight")?
            }
            "--peak-weight" => {
                cfg.peak_weight = req(args, &mut i)?.parse().context("--peak-weight")?
            }
            "--peak-k" | "--k" => cfg.peak_k = req(args, &mut i)?.parse().context("--peak-k")?,
            "--spectrum-weight" => {
                cfg.spectrum_weight = req(args, &mut i)?.parse().context("--spectrum-weight")?
            }
            "--log-every" => cfg.log_every = req(args, &mut i)?.parse().context("--log-every")?,
            "--no-q8" => {
                cfg.train_q8 = false;
                i += 1;
            }
            "--seed" => cfg.seed = req(args, &mut i)?.parse().context("--seed")?,
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let (_model, report) = crate::train_e2e::train_fast_learned_model(&cfg)?;
    eprintln!(
        "train-e2e done: spec_err={:.3e} mel_err={:.3e} welch_err={:.3e} peak_err={:.3e} mean_gate={:.3} active_gates={} q8={} ({:.1} ms)",
        report.final_spectrum_max_err,
        report.final_mel_max_err,
        report.final_welch_max_err,
        report.final_peak_max_err,
        report.mean_gate,
        report.active_gates,
        report.q8_enabled,
        report.elapsed_ms
    );
    if let Some(path) = json_out {
        std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn cmd_train_distill(args: &[String]) -> Result<()> {
    let mut cfg = crate::train_distill::DistillTrainConfig::default();
    let mut teacher_cfg = crate::train_e2e::E2eTrainConfig::default();
    let mut json_out: Option<PathBuf> = None;
    let mut teacher_steps: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => cfg.n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => cfg.batch = req(args, &mut i)?.parse().context("--batch")?,
            "--n-mels" => cfg.n_mels = req(args, &mut i)?.parse().context("--n-mels")?,
            "--steps" => cfg.steps = req(args, &mut i)?.parse().context("--steps")?,
            "--lr" => cfg.lr = req(args, &mut i)?.parse().context("--lr")?,
            "--teacher-steps" => {
                teacher_steps = Some(req(args, &mut i)?.parse().context("--teacher-steps")?)
            }
            "--seed" => cfg.seed = req(args, &mut i)?.parse().context("--seed")?,
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    teacher_cfg.n_fft = cfg.n_fft;
    teacher_cfg.batch = cfg.batch;
    teacher_cfg.n_mels = cfg.n_mels;
    if let Some(s) = teacher_steps {
        teacher_cfg.steps = s;
    }

    eprintln!(
        "[train-distill] training teacher ({} steps)…",
        teacher_cfg.steps
    );
    let (teacher, trep) = crate::train_e2e::train_fast_learned_model(&teacher_cfg)?;
    eprintln!(
        "[train-distill] teacher mel_err={:.3e} welch_err={:.3e}",
        trep.final_mel_max_err, trep.final_welch_max_err
    );

    let (student, rep) = crate::train_distill::distill_from_teacher(&teacher, &cfg)?;
    eprintln!(
        "train-distill done: mel_vs_teacher={:.3e} welch_vs_teacher={:.3e} mel_vs_ref={:.3e} ({:.1} ms)",
        rep.final_mel_err_vs_teacher,
        rep.final_welch_err_vs_teacher,
        rep.final_mel_err_vs_ref,
        rep.elapsed_ms
    );
    if let Some(path) = json_out {
        let payload = serde_json::json!({
            "teacher": trep,
            "distill": rep,
            "student_n_fft": student.n_fft,
            "student_n_mels": student.n_mels,
        });
        std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn cmd_train_distill_ternary(args: &[String]) -> Result<()> {
    let mut cfg = crate::train_distill_ternary::DistillTernaryTrainConfig::default();
    let mut teacher_cfg = crate::train_e2e::E2eTrainConfig::default();
    let mut json_out: Option<PathBuf> = None;
    let mut teacher_steps: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => cfg.n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => cfg.batch = req(args, &mut i)?.parse().context("--batch")?,
            "--n-mels" => cfg.n_mels = req(args, &mut i)?.parse().context("--n-mels")?,
            "--steps" => cfg.steps = req(args, &mut i)?.parse().context("--steps")?,
            "--lr" => cfg.lr = req(args, &mut i)?.parse().context("--lr")?,
            "--compute-weight" => {
                cfg.compute_weight = req(args, &mut i)?.parse().context("--compute-weight")?
            }
            "--teacher-steps" => {
                teacher_steps = Some(req(args, &mut i)?.parse().context("--teacher-steps")?)
            }
            "--seed" => cfg.seed = req(args, &mut i)?.parse().context("--seed")?,
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    teacher_cfg.n_fft = cfg.n_fft;
    teacher_cfg.batch = cfg.batch;
    teacher_cfg.n_mels = cfg.n_mels;
    if let Some(s) = teacher_steps {
        teacher_cfg.steps = s;
    }

    eprintln!(
        "[train-distill-ternary] training teacher ({} steps)…",
        teacher_cfg.steps
    );
    let (teacher, trep) = crate::train_e2e::train_fast_learned_model(&teacher_cfg)?;
    eprintln!(
        "[train-distill-ternary] teacher mel_err={:.3e} welch_err={:.3e}",
        trep.final_mel_max_err, trep.final_welch_max_err
    );

    let (student, rep) =
        crate::train_distill_ternary::distill_ternary_from_teacher(&teacher, &cfg)?;
    eprintln!(
        "train-distill-ternary done: mel_vs_teacher={:.3e} welch_vs_teacher={:.3e} mel_vs_ref={:.3e} compute={:.3} skip={} fwd={} rev={} ({:.1} ms)",
        rep.final_mel_err_vs_teacher,
        rep.final_welch_err_vs_teacher,
        rep.final_mel_err_vs_ref,
        rep.compute_fraction,
        rep.skip_gates,
        rep.forward_gates,
        rep.reverse_gates,
        rep.elapsed_ms
    );
    if let Some(path) = json_out {
        let payload = serde_json::json!({
            "teacher": trep,
            "distill_ternary": rep,
            "student_n_fft": student.n_fft,
            "student_n_mels": student.n_mels,
            "gates": student.gates,
        });
        std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn cmd_bench_e2e(args: &[String]) -> Result<()> {
    let mut n_fft = 256usize;
    let mut batch_csv = "8".to_string();
    let mut n_mels = 40usize;
    let mut peak_k = crate::peak::DEFAULT_PEAK_K;
    let mut iters = 20usize;
    let mut device_csv = "all".to_string();
    let mut train_first = false;
    let mut train_steps = 2000usize;
    let mut seed = 42u64;
    let mut json_out: Option<PathBuf> = None;
    let mut html_out: Option<PathBuf> = None;
    let mut with_learned_hard = true;
    let mut with_learned_compiled = true;
    let mut with_learned_distilled = true;
    let mut with_learned_distilled_ternary = true;
    let mut with_eager_learned = false;
    let mut distill_first = false;
    let mut ternary_distill_first = false;
    let mut distill_steps = 1200usize;
    let mut compute_weight = 0.10f32;
    let mut target_compute_fraction = 0.96f32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch_csv = req(args, &mut i)?,
            "--n-mels" => n_mels = req(args, &mut i)?.parse().context("--n-mels")?,
            "--peak-k" | "--k" => peak_k = req(args, &mut i)?.parse().context("--peak-k")?,
            "--iters" => iters = req(args, &mut i)?.parse().context("--iters")?,
            "--device" => device_csv = req(args, &mut i)?,
            "--train-first" => {
                train_first = true;
                i += 1;
            }
            "--distill-first" => {
                distill_first = true;
                ternary_distill_first = true;
                i += 1;
            }
            "--distill-only" => {
                distill_first = true;
                ternary_distill_first = false;
                i += 1;
            }
            "--ternary-distill" => {
                ternary_distill_first = true;
                distill_first = true;
                i += 1;
            }
            "--steps" => train_steps = req(args, &mut i)?.parse().context("--steps")?,
            "--distill-steps" => {
                distill_steps = req(args, &mut i)?.parse().context("--distill-steps")?
            }
            "--compute-weight" => {
                compute_weight = req(args, &mut i)?.parse().context("--compute-weight")?
            }
            "--target-compute-fraction" => {
                target_compute_fraction = req(args, &mut i)?
                    .parse()
                    .context("--target-compute-fraction")?
            }
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--no-hard-gates" => {
                with_learned_hard = false;
                i += 1;
            }
            "--no-compiled" => {
                with_learned_compiled = false;
                i += 1;
            }
            "--no-distilled" => {
                with_learned_distilled = false;
                i += 1;
            }
            "--no-ternary-distilled" => {
                with_learned_distilled_ternary = false;
                i += 1;
            }
            "--with-eager-learned" => {
                with_eager_learned = true;
                i += 1;
            }
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--html" => html_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let batches = parse_batch_spec(&batch_csv, "--batch")?;
    let device_names = crate::device::parse_bench_device_list(&device_csv)?;
    eprintln!("[bench-e2e] devices: {}", device_names.join(", "));

    let mut merged = crate::e2e_bench::E2eBenchReport {
        n_mels,
        iters,
        elapsed_ms: 0.0,
        rows: Vec::new(),
        meta: crate::e2e_bench::E2eBenchMeta {
            n_fft,
            seed,
            devices: device_names.clone(),
            batches: batches.clone(),
            peak_k,
            train_steps: if train_first || distill_first {
                Some(train_steps)
            } else {
                None
            },
            distill_steps: if distill_first || ternary_distill_first {
                Some(distill_steps)
            } else {
                None
            },
            teacher: None,
            distill: None,
            per_batch: Vec::new(),
        },
    };
    let started = std::time::Instant::now();

    for &batch in &batches {
        let mut batch_meta = crate::e2e_bench::E2eBatchTrainMeta {
            batch,
            teacher: None,
            distill: None,
        };

        let trained = if train_first || distill_first {
            eprintln!("[bench-e2e] training teacher batch={batch} steps={train_steps}");
            let train_cfg = crate::train_e2e::E2eTrainConfig {
                n_fft,
                batch,
                n_mels,
                steps: train_steps,
                seed: seed.wrapping_add(batch as u64),
                peak_k,
                ..crate::train_e2e::E2eTrainConfig::default()
            };
            let (m, rep) = crate::train_e2e::train_fast_learned_model(&train_cfg)?;
            eprintln!(
                "[bench-e2e] batch={batch} spec_err={:.3e} mel_err={:.3e} welch_err={:.3e} mean_gate={:.3} active_gates={}",
                rep.final_spectrum_max_err,
                rep.final_mel_max_err,
                rep.final_welch_max_err,
                rep.mean_gate,
                rep.active_gates
            );
            batch_meta.teacher = Some(rep);
            Some(m)
        } else {
            None
        };

        let distilled = if distill_first {
            if let Some(teacher) = trained.as_ref() {
                eprintln!("[bench-e2e] distilling student batch={batch} steps={distill_steps}");
                let dcfg = crate::train_distill::DistillTrainConfig {
                    n_fft,
                    batch,
                    n_mels,
                    steps: distill_steps,
                    seed: seed.wrapping_add(batch as u64).wrapping_add(1),
                    ..crate::train_distill::DistillTrainConfig::default()
                };
                let (d, rep) = crate::train_distill::distill_from_teacher(teacher, &dcfg)?;
                eprintln!(
                    "[bench-e2e] batch={batch} mel_vs_teacher={:.3e} welch_vs_teacher={:.3e} mel_vs_ref={:.3e}",
                    rep.final_mel_err_vs_teacher,
                    rep.final_welch_err_vs_teacher,
                    rep.final_mel_err_vs_ref
                );
                batch_meta.distill = Some(rep);
                Some(d)
            } else {
                None
            }
        } else if train_first {
            trained
                .as_ref()
                .map(crate::distill_model::DistilledFftModel::from_teacher)
        } else {
            None
        };

        let distilled_ternary = if ternary_distill_first {
            if let Some(teacher) = trained.as_ref() {
                eprintln!(
                    "[bench-e2e] ternary distilling batch={batch} steps={distill_steps} compute_weight={compute_weight}"
                );
                let dcfg = crate::train_distill_ternary::DistillTernaryTrainConfig {
                    n_fft,
                    batch,
                    n_mels,
                    steps: distill_steps,
                    compute_weight,
                    target_compute_fraction,
                    seed: seed.wrapping_add(batch as u64).wrapping_add(3),
                    ..crate::train_distill_ternary::DistillTernaryTrainConfig::default()
                };
                let (d, rep) = if let Some(base) = distilled.as_ref() {
                    crate::train_distill_ternary::distill_ternary_from_distilled(
                        base, teacher, &dcfg,
                    )?
                } else {
                    crate::train_distill_ternary::distill_ternary_from_teacher(teacher, &dcfg)?
                };
                eprintln!(
                    "[bench-e2e] batch={batch} ternary mel_vs_teacher={:.3e} spec_vs_ref={:.3e} compute={:.3} skip={} fwd={} rev={}",
                    rep.final_mel_err_vs_teacher,
                    rep.final_spec_err_vs_ref,
                    rep.compute_fraction,
                    rep.skip_gates,
                    rep.forward_gates,
                    rep.reverse_gates
                );
                Some(d)
            } else {
                None
            }
        } else if distill_first {
            distilled.as_ref().map(|d| {
                crate::distill_ternary_model::DistilledTernaryFftModel::from_distilled(
                    d,
                    trained.as_ref().expect("teacher"),
                )
            })
        } else if train_first {
            trained
                .as_ref()
                .map(crate::distill_ternary_model::DistilledTernaryFftModel::from_teacher)
        } else {
            None
        };

        merged.meta.per_batch.push(batch_meta);

        let fallback = if trained.is_none() {
            let cfg = FftLearnConfig::new(n_fft, batch)?;
            Some(FastLearnedFftModel::new(&cfg, n_mels, 16_000.0).with_q8())
        } else {
            None
        };
        let bench_model = trained.as_ref().or(fallback.as_ref());
        let distilled_fallback =
            bench_model.map(crate::distill_model::DistilledFftModel::from_teacher);
        let bench_distilled = distilled.as_ref().or(distilled_fallback.as_ref());
        let ternary_fallback =
            bench_model.map(crate::distill_ternary_model::DistilledTernaryFftModel::from_teacher);
        let bench_distilled_ternary = distilled_ternary.as_ref().or(ternary_fallback.as_ref());

        for dev_name in &device_names {
            let dev = parse_device(&crate::device::normalize_device_alias(dev_name))?;
            crate::device::ensure_backend_ready(dev)?;
            let inputs = crate::e2e_bench::E2eBenchInputs {
                n_fft,
                batch,
                n_mels,
                iters,
                device: dev,
                seed: seed.wrapping_add(batch as u64),
                model: bench_model,
                distilled: bench_distilled,
                distilled_ternary: bench_distilled_ternary,
                with_learned_hard,
                with_learned_compiled,
                with_learned_distilled,
                with_learned_distilled_ternary,
                with_eager_learned,
                peak_k,
            };
            let report = crate::e2e_bench::run_e2e_bench(&inputs)?;
            merged.rows.extend(report.rows);
        }
    }

    if let Some(last) = merged.meta.per_batch.last() {
        merged.meta.teacher = last.teacher.clone();
        merged.meta.distill = last.distill.clone();
    }
    merged.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    crate::e2e_bench::print_e2e_table(&merged);
    if let Some(path) = json_out {
        crate::e2e_bench::write_e2e_json(&path, &merged)?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = html_out {
        write_e2e_html(&path, &merged)?;
        eprintln!("wrote {}", path.display());
        let _ = std::process::Command::new("open").arg(&path).status();
    }
    Ok(())
}

fn cmd_bench_welch_peaks(args: &[String]) -> Result<()> {
    let mut n_fft = 256usize;
    let mut batch_csv = "32".to_string();
    let mut k_csv = "16".to_string();
    let mut device = "auto".to_string();
    let mut iters = 50usize;
    let mut train_steps = 200usize;
    let mut seed = 42u64;
    let mut with_compiled = true;
    let mut with_ultra_fast = true;
    let mut pick_mode = crate::welch_peaks_picker::WelchPeaksPickMode::Auto;
    let mut json_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch_csv = req(args, &mut i)?,
            "--k" | "--peak-k" => k_csv = req(args, &mut i)?,
            "--device" => device = req(args, &mut i)?,
            "--strategy" => {
                pick_mode =
                    crate::welch_peaks_picker::parse_welch_peaks_strategy(&req(args, &mut i)?)?;
            }
            "--iters" => iters = req(args, &mut i)?.parse().context("--iters")?,
            "--train-steps" => train_steps = req(args, &mut i)?.parse().context("--train-steps")?,
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--no-compiled" => {
                with_compiled = false;
                i += 1;
            }
            "--no-ultra-fast" => {
                with_ultra_fast = false;
                i += 1;
            }
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let batches = parse_batch_spec(&batch_csv, "--batch")?;
    let ks = parse_k_spec(&k_csv, "--k")?;
    let single = batches.len() == 1
        && ks.len() == 1
        && !batch_csv.contains(',')
        && !batch_csv.contains('-')
        && !k_csv.contains(',')
        && !k_csv.contains('-');

    let opts = crate::bench_welch_peaks::WelchPeaksBenchOpts {
        n_fft,
        batch: batches[0],
        k: ks[0],
        device_name: device,
        iters,
        train_steps,
        seed,
        with_compiled,
        with_ultra_fast,
        pick_mode,
    };
    let report = if single {
        crate::bench_welch_peaks::run_welch_peaks_bench_opts(&opts)?
    } else {
        crate::bench_welch_peaks::run_welch_peaks_sweep(&opts, &batch_csv, &k_csv)?
    };
    crate::bench_welch_peaks::print_welch_peaks_table(&report);
    if let Some(path) = json_out {
        crate::bench_welch_peaks::write_welch_peaks_json(&path, &report)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

#[cfg(feature = "dev")]
fn cmd_bench_fusion_phases(args: &[String]) -> Result<()> {
    let mut n_fft = 256usize;
    let mut batch_csv = "32".to_string();
    let mut k = 16usize;
    let mut device = "auto".to_string();
    let mut iters = 30usize;
    let mut seed = 42u64;
    let mut json_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-fft" => n_fft = parse_n_fft(&req(args, &mut i)?)?,
            "--batch" => batch_csv = req(args, &mut i)?,
            "--k" | "--peak-k" => k = req(args, &mut i)?.parse().context("--k")?,
            "--device" => device = req(args, &mut i)?,
            "--iters" => iters = req(args, &mut i)?.parse().context("--iters")?,
            "--seed" => seed = req(args, &mut i)?.parse().context("--seed")?,
            "--json" => json_out = Some(req(args, &mut i)?.into()),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let batches = parse_batch_spec(&batch_csv, "--batch")?;
    let single = batches.len() == 1 && !batch_csv.contains(',') && !batch_csv.contains('-');

    if single {
        let report = crate::bench_fusion_phases::run_fusion_phase_bench(
            n_fft, batches[0], k, &device, iters, seed,
        )?;
        crate::bench_fusion_phases::print_fusion_phase_report(&report);
        if let Some(path) = json_out {
            crate::bench_fusion_phases::write_fusion_phase_json(&path, &report)?;
            eprintln!("wrote {}", path.display());
        }
    } else {
        let sweep = crate::bench_fusion_phases::run_fusion_phase_sweep(
            n_fft, &batch_csv, k, &device, iters, seed,
        )?;
        for report in &sweep.reports {
            crate::bench_fusion_phases::print_fusion_phase_report(report);
        }
        crate::bench_fusion_phases::print_fusion_phase_sweep_summary(&sweep);
        if let Some(path) = json_out {
            crate::bench_fusion_phases::write_fusion_phase_sweep_json(&path, &sweep)?;
            eprintln!("wrote {}", path.display());
        }
    }
    Ok(())
}
