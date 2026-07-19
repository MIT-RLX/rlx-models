// RLX models — distributed vision training.
// SPDX-License-Identifier: GPL-3.0-only

//! A configuration-sweep harness: train every `(dataset × model)` combination
//! on a chosen device and print a comparison table (params, throughput,
//! accuracy, wall). Each cell is a single-machine [`train_local`] run — the
//! distributed transports (TCP / iroh) are an orthogonal axis exercised by the
//! `run_node_from_env` path, not swept here.
//!
//! ```text
//! rlx-vision-bench --suite                       # all datasets × all models, fastest device
//! rlx-vision-bench --suite cifar10,cifar100 --model cnn --epochs 2
//! RLX_DEVICE=cuda rlx-vision-bench --suite       # pin the device
//! ```

use crate::{
    Config, DataSpec, DatasetKind, ModelKind, datasets, param_count, train_local, training_device,
};
use rlx_runtime::Device;

/// One result cell of the sweep.
pub struct SuiteRow {
    pub dataset: &'static str,
    pub synthetic: bool,
    pub model: &'static str,
    pub device: String,
    pub spec: DataSpec,
    pub params: usize,
    pub samples: usize,
    pub accuracy: f64,
    /// Training throughput (samples / compute-second).
    pub throughput: f64,
    pub wall_s: f64,
}

/// What to sweep.
pub struct SuiteOpts {
    pub datasets: Vec<DatasetKind>,
    pub models: Vec<ModelKind>,
    pub device: Device,
    pub epochs: usize,
    pub batch: usize,
    pub hidden: usize,
    pub augment: bool,
    /// Cap on train/test samples per dataset (keeps big/synthetic runs quick).
    pub max_train: Option<usize>,
    pub max_test: Option<usize>,
}

impl Default for SuiteOpts {
    fn default() -> Self {
        Self {
            datasets: DatasetKind::all().to_vec(),
            models: ModelKind::all().to_vec(),
            device: training_device(),
            epochs: 1,
            batch: 64,
            hidden: 128,
            augment: false,
            max_train: Some(20_000),
            max_test: Some(4_000),
        }
    }
}

/// Run the `(dataset × model)` matrix and return one [`SuiteRow`] per cell.
pub fn run_suite(opts: &SuiteOpts) -> Vec<SuiteRow> {
    let mut rows = Vec::new();
    let device_name = rlx_runtime::full_name(opts.device).to_string();
    for &ds in &opts.datasets {
        eprintln!("── loading {} ──", ds.name());
        let data = match datasets::load(ds, opts.max_train, opts.max_test) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  skip {}: {e}", ds.name());
                continue;
            }
        };
        for &model in &opts.models {
            let cfg = Config {
                model,
                spec: data.spec,
                hidden: opts.hidden,
                batch: opts.batch,
                epochs: opts.epochs,
                augment: opts.augment,
                ..Config::default()
            };
            if data.train.len() < cfg.batch || data.test.len() < cfg.batch {
                eprintln!(
                    "  {} × {}: too few samples for batch {}",
                    data.name,
                    model.name(),
                    cfg.batch
                );
                continue;
            }
            let params = param_count(&cfg);
            eprintln!(
                "  ▶ {} × {} on {} — input {}×{}×{}, {} classes, {} params …",
                data.name,
                model.name(),
                device_name,
                data.spec.c,
                data.spec.h,
                data.spec.w,
                data.spec.classes,
                human(params),
            );
            let rep = train_local(&cfg, &data.train, &data.test, opts.device, false);
            rows.push(SuiteRow {
                dataset: data.name,
                synthetic: data.synthetic,
                model: model.name(),
                device: device_name.clone(),
                spec: data.spec,
                params,
                samples: rep.samples,
                accuracy: rep.accuracy,
                throughput: rep.samples as f64 / rep.compute_s.max(1e-9),
                wall_s: rep.wall_s,
            });
        }
    }
    rows
}

/// Print the results table.
pub fn print_table(rows: &[SuiteRow]) {
    println!(
        "\n{:<14} {:<5} {:<20} {:<13} {:>7} {:>8} {:>13} {:>10} {:>7}",
        "dataset",
        "model",
        "device",
        "C×H×W",
        "classes",
        "params",
        "throughput",
        "accuracy",
        "wall",
    );
    println!("{}", "─".repeat(103));
    for r in rows {
        let acc = if r.synthetic {
            "n/a (syn)".to_string()
        } else {
            format!("{:.2}%", r.accuracy * 100.0)
        };
        println!(
            "{:<14} {:<5} {:<20} {:<13} {:>7} {:>8} {:>10.0}/s {:>10} {:>6.1}s",
            r.dataset,
            r.model,
            r.device,
            format!("{}×{}×{}", r.spec.c, r.spec.h, r.spec.w),
            r.spec.classes,
            human(r.params),
            r.throughput,
            acc,
            r.wall_s,
        );
    }
    println!(
        "\nsynthetic datasets (imagenet/coco unless RLX_IMAGENET_DIR/RLX_COCO_DIR is set) show n/a accuracy — random data exercises the config, not learning."
    );
}

/// Human-readable count: `1.2M`, `51.4K`, `842`.
fn human(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}
