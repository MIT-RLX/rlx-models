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

//! Learned FFT — butterfly network trained to match reference FFT, compiled via RLX.
//!
//! # Overview
//!
//! This crate learns twiddle factors in a Cooley–Tukey butterfly network so the
//! transform matches [`rustfft`] on random signals. After training, the same graph
//! can be compiled to CPU/GPU backends for batched inference.
//!
//! # Module map
//!
//! - [`model`] — butterfly / Stockham graphs, twiddles, configs, model variants
//! - [`train`] — training loops (phased, multi-stage, distill, e2e, RLX-graph)
//! - [`learned`] — fast learned FFT pipeline (Tier D) and its compiled deploy
//! - [`distill`] — float / ternary distilled students and compiled deploys
//! - [`exec`] — backend compilation, device selection, inference runner
//! - [`spectral`] — mel frontend, Welch PSD, top-K peak extraction
//! - [`bench`], [`ablation`], [`study`] — benchmarks, ablations, HTML reports
//! - [`cli`] — the `rlx-fft` command-line interface
//!
//! # Quick start
//!
//! ```no_run
//! use rlx_fft::{FftLearnConfig, FftLearnRunner, TrainConfig, train_butterfly};
//!
//! # fn main() -> anyhow::Result<()> {
//! let cfg = FftLearnConfig::new(256, 8)?;
//! let report = train_butterfly(&TrainConfig {
//!     model: cfg.clone(),
//!     steps: 200,
//!     ..TrainConfig::default()
//! })?;
//! println!("mse={} max_err={}", report.final_mse, report.max_error);
//!
//! let runner = FftLearnRunner::with_weights(cfg, &report.weights)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Welch peaks
//!
//! Fast top-K spike extraction with an automatic or forced strategy picker
//! ([`AutoWelchPeaks`], `--strategy` on `bench-welch-peaks`). See `crates/rlx-fft/README.md`
//! (Welch peaks section) in this repo.

pub mod ablation;
pub mod bench;
pub mod cli;
pub mod distill;
pub mod exec;
pub mod learned;
pub mod model;
pub mod spectral;
pub mod study;
pub mod train;

// Reference FFT oracle at the crate root — the stable path other crates
// (rlx-aec, rlx-qwen3-tts) use for `rlx_fft::reference::fft_real_batch`.
pub use model::reference;

pub use ablation::csv::{
    LIMITS_CSV, META_CSV, ROWS_CSV, TOP5_CSV, read_ablation_csv_dir, read_ablation_rows_csv,
    write_ablation_csv_dir,
};
pub use ablation::html::{read_ablation_json, render_ablation_html, write_ablation_html};
pub use ablation::ternary::{
    TernaryAblationOpts, TernaryAblationReport, TernaryAblationRow, TernaryArchVariantId,
    TernaryExecMode, TernaryParetoPoint, print_ternary_ablation_table, quick_ablation_opts,
    run_ternary_ablation, ternary_ablation_row_ok, ternary_aggregate_variants,
    ternary_pareto_frontier, ternary_recommendation, write_ternary_ablation_csv,
    write_ternary_ablation_json,
};
pub use ablation::ternary_html::{
    read_ternary_ablation_json, render_ternary_ablation_html, write_ternary_ablation_html,
};
pub use ablation::{
    AblationReport, AblationRow, ablation_row_ok, ablation_winners, limit_sweep_devices,
    merge_ablation_reports, print_ablation_table, run_ablation, run_limit_sweep, tier_summary,
    top5_variants_per_n_fft, write_ablation_json,
};
pub use bench::e2e::{
    E2eBackend, E2eBatchTrainMeta, E2eBenchMeta, E2eBenchReport, E2eBenchRow, E2ePipeline,
    merge_e2e_reports, print_e2e_table, read_e2e_json, run_e2e_bench, write_e2e_json,
};
pub use bench::e2e_html::{render_e2e_html, write_e2e_html};
pub use bench::encdec::{
    EncDecBenchRow, bench_encdec_weights, bench_exact_baseline, bench_phased_dir,
    print_encdec_bench_table, write_encdec_bench_json,
};
pub use bench::sweep::{
    SweepReport, SweepRow, available_devices, parse_batch_spec, parse_csv_usize, parse_k_spec,
    print_sweep_chart, run_sweep, sweep_markdown_chart, write_sweep_json,
};
pub use bench::sweep_html::{read_sweep_json, render_sweep_html, write_sweep_html};
pub use bench::welch_peaks::{
    WelchPeaksBenchOpts, WelchPeaksBenchReport, WelchPeaksBenchRow, print_welch_peaks_table,
    run_welch_peaks_batch_sweep, run_welch_peaks_bench, run_welch_peaks_bench_opts,
    run_welch_peaks_k_sweep, run_welch_peaks_sweep, write_welch_peaks_json,
};
pub use bench::{
    BenchReport, bench_all, bench_all_dir, bench_reference_vs_learned,
    bench_reference_vs_learned_dir,
};
pub use distill::compile::{CompiledDistilledMel, compile_distilled_mel};
pub use distill::model::DistilledFftModel;
pub use distill::ternary_arch::{CorrectorKind, GateLayout, SpectrumCorrection, TernaryArchConfig};
pub use distill::ternary_compile::{CompiledDistilledTernaryMel, compile_distilled_ternary_mel};
pub use distill::ternary_gates::{GateMode, compute_fraction, gate_mode_counts};
pub use distill::ternary_model::DistilledTernaryFftModel;
pub use exec::device::{
    bench_device_label, ensure_backend_ready, normalize_device_alias, parse_bench_device_list,
    pick_auto_device, resolve_train_device,
};
pub use exec::runner::FftLearnRunner;
pub use learned::model::FastLearnedFftModel;
pub use model::config::{
    EncDecTrainConfig, FftLearnConfig, MultiTrainConfig, MultiTrainSchedule, PhasedTrainConfig,
    SUPPORTED_N_FFT, TrainConfig, TransformDir, parse_transform_dir,
};
pub use model::twiddle::{TwiddleSet, exact_twiddles, exact_twiddles_dir};
pub use model::twiddle_stability::{
    lr_for_n_fft, max_twiddle_magnitude, project_twiddles_unit_circle, twiddle_drift_from_unit,
};
pub use model::variants::{FftVariantId, VariantState};
pub use model::weights::{EncDecWeights, WeightStore, export_safetensors, load_safetensors};
pub use spectral::peak::{
    DEFAULT_PEAK_K, WelchPeakParams, WelchPeaksScratch, peak_band_mask,
    peak_loss_grad_wrt_spectrum, peak_match_loss, peak_max_err, peaks_from_psd_batch,
    peaks_from_segment_spectrum_streaming, topk_peaks_one, welch_peaks_from_segment_spectrum,
    welch_peaks_rustfft, welch_peaks_rustfft_with_scratch,
};
pub use spectral::welch_peaks_compile::{
    CompiledLearnedWelchPeaks, CompiledRlxWelchPeaks, CompiledRlxWelchPeaksExec,
    CompiledRlxWelchPeaksFused, RlxWelchPeaksExecKind, compile_learned_welch_peaks,
    compile_rlx_welch_peaks, compile_welch_peaks_fused, default_welch_peaks_hard_threshold,
    rlx_welch_peaks_exec_kind,
};
pub use spectral::welch_peaks_cost::{
    WelchPeaksCostEstimates, WelchPeaksFusionGateBreakdown, algorithm_bandwidth_gbps,
    ayala_io_cost_ns, estimate_welch_peaks_costs, fused_welch_peaks_auto_viable,
    rustfft_peaks_io_profile, useful_bytes_touched, welch_peaks_fusion_gate_breakdown,
    welch_peaks_fusion_target, welch_peaks_io_fusion_gate,
};
pub use spectral::welch_peaks_picker::{
    AutoWelchPeaks, WelchPeaksPickBreakdown, WelchPeaksPickMode, WelchPeaksStrategy,
    all_welch_peaks_strategy_names, parse_welch_peaks_strategy, pick_welch_peaks_breakdown,
    pick_welch_peaks_strategy, resolve_welch_peaks_strategy, rlx_crossover_batch,
    ultra_fast_max_batch,
};
pub use study::html::{StudyInputs, render_study_html, write_study_html};
pub use train::distill::{DistillTrainConfig, DistillTrainReport, distill_from_teacher};
pub use train::distill_ternary::{
    DistillTernaryTrainConfig, DistillTernaryTrainReport, distill_ternary_from_distilled,
    distill_ternary_from_teacher,
};
pub use train::e2e::{E2eTrainConfig, E2eTrainReport, train_fast_learned_model};
pub use train::multi::{
    MultiTrainEvalRow, MultiTrainReport, best_regime_per_eval, print_multi_train_table,
    run_multi_train, write_multi_train_json,
};
pub use train::multi_html::{
    read_multi_train_json, render_multi_train_html, write_multi_train_html,
};
pub use train::phased::{PhaseMetrics, PhasedTrainResult, precision_encdec, train_phased_encdec};
pub use train::rlx::train_encdec_rlx_pool;
pub use train::second_order::{
    TwiddleOptState, TwiddleOptimizer, diag_gn_step, hvp_twiddles_finite_diff,
};
pub use train::{
    EncDecTrainResult, TrainResult, evaluate_encdec_weights, evaluate_weights,
    evaluate_weights_dir, random_complex_batch, train_butterfly, train_butterfly_dir,
    train_butterfly_eager, train_encdec, train_encdec_eager,
};
