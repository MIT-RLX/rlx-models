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

//! Figure 2 — regression in an RKHS, adaptive trees against fixed-depth ones.
//!
//! ```text
//! cargo run -p rlx-fgd --release --example kernel_regression -- --loss mse
//! cargo run -p rlx-fgd --release --example kernel_regression -- --loss bce --nn-steps 400
//! # Section 4.1's own dataset:
//! curl -L https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna -o /tmp/cod-rna
//! cargo run -p rlx-fgd --release --example kernel_regression -- --libsvm /tmp/cod-rna --loss bce
//! # Gradient on a backend:
//! cargo run -p rlx-fgd --release --features metal --example kernel_regression -- --device metal
//! ```

use rlx_fgd::algorithm::{FgdConfig, Refinement, run};
use rlx_fgd::baseline::{FourierMlp, MlpConfig};
use rlx_fgd::rlx::DeviceRkhsRegression;
use rlx_fgd::tasks::rkhs::{Dataset, PointwiseLoss, RbfKernel, RkhsRegression};
use rlx_runtime::Device;
use std::time::Instant;

fn flag(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    flag(name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn parse_device(name: &str) -> Device {
    match name {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "gpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        other => panic!("unknown device {other:?}"),
    }
}

fn main() {
    let loss = match flag("--loss").as_deref().unwrap_or("mse") {
        "mse" => PointwiseLoss::SquaredError,
        "bce" => PointwiseLoss::BinaryCrossEntropy,
        other => panic!("unknown loss {other:?}; expected mse or bce"),
    };
    let steps: usize = arg("--steps", 60);
    let gamma: f64 = arg("--gamma", 100.0);
    let max_depth: u32 = arg("--max-depth", 14);
    let nn_steps: usize = arg("--nn-steps", 0);
    let device = flag("--device").map(|d| parse_device(&d));

    // Section 4.1 keeps the first three features of cod-rna and maps its ±1
    // labels to {0,1}; without that file, a synthetic set of the same shape.
    let raw = match flag("--libsvm") {
        Some(path) => {
            let text = std::fs::read_to_string(&path).expect("could not read the LIBSVM file");
            let full = Dataset::from_libsvm(&text, 8, |y| (y + 1.0) / 2.0)
                .expect("could not parse the LIBSVM file");
            println!("cod-rna: {} rows, keeping 3 features", full.len());
            full.truncate_features(3)
        }
        None => {
            let synthetic = Dataset::synthetic(1_000, 21);
            match loss {
                PointwiseLoss::BinaryCrossEntropy => {
                    let targets = synthetic
                        .targets()
                        .iter()
                        .map(|&y| if y > 0.0 { 1.0 } else { 0.0 })
                        .collect();
                    Dataset::new(
                        (0..synthetic.len())
                            .flat_map(|i| synthetic.row(i).to_vec())
                            .collect(),
                        targets,
                        synthetic.dim(),
                    )
                }
                PointwiseLoss::SquaredError => synthetic,
            }
        }
    };

    let (lower, upper) = raw.bounds();
    let (train, test) = raw.normalized(&lower, &upper).split(0.2, 0);
    println!(
        "{} train / {} test rows, {} features, RBF γ = {gamma}, {steps} steps\n",
        train.len(),
        test.len(),
        train.dim()
    );

    let problem = RkhsRegression::new(train, RbfKernel::new(gamma), loss);
    // Cross-entropy is 1/4-smooth against squared error's 1; scaling the step
    // by 1/K puts the two on the same footing.
    let learning_rate: f64 = arg("--lr", 10.0 / loss.smoothness());

    println!(
        "{:<28} {:>10} {:>12} {:>12} {:>10}",
        "method", "leaves", "train loss", "test loss", "time"
    );
    println!("{}", "-".repeat(76));

    let config = |refinement| FgdConfig {
        learning_rate,
        tolerance: 0.5,
        steps,
        refinement,
        seed: 0,
    };

    let row = |name: String, report: &rlx_fgd::Report<rlx_fgd::RegressionTree>, secs: f64| {
        println!(
            "{name:<28} {:>10} {:>12.4} {:>12.4} {:>9.2}s",
            report.history.last().map_or(0, |s| s.dofs),
            report.final_loss().unwrap_or(f64::NAN),
            problem.dataset_loss(&report.solution, &test),
            secs
        );
    };

    // The paper's baseline: FGD at a tree depth fixed for the whole run.
    for depth in [2u32, 4, 8, 12] {
        let started = Instant::now();
        let report = run(
            &problem,
            problem.initial_tree(depth),
            config(Refinement::Fixed),
        );
        row(
            format!("fixed FGD, depth {depth}"),
            &report,
            started.elapsed().as_secs_f64(),
        );
    }

    let started = Instant::now();
    let report = match device {
        Some(device) => {
            let on_device = DeviceRkhsRegression::new(problem.clone(), device)
                .expect("backend unavailable in this build");
            run(
                &on_device,
                problem.initial_tree(2),
                config(Refinement::Adaptive {
                    max_level: max_depth,
                }),
            )
        }
        None => run(
            &problem,
            problem.initial_tree(2),
            config(Refinement::Adaptive {
                max_level: max_depth,
            }),
        ),
    };
    let label = match device {
        Some(device) => format!("adaptive FGD, {device:?}"),
        None => "adaptive FGD (ours)".to_string(),
    };
    row(label, &report, started.elapsed().as_secs_f64());

    if nn_steps > 0 {
        let mut net = FourierMlp::new(MlpConfig {
            inputs: problem.train.dim(),
            hidden: vec![128, 128],
            learning_rate: 3e-3,
            ..MlpConfig::default()
        });
        let train_xs: Vec<f64> = (0..problem.train.len())
            .flat_map(|i| problem.train.row(i).to_vec())
            .collect();
        let test_xs: Vec<f64> = (0..test.len()).flat_map(|i| test.row(i).to_vec()).collect();
        let started = Instant::now();
        for _ in 0..nn_steps {
            net.train_batch(&train_xs, problem.train.targets(), loss);
        }
        println!(
            "{:<28} {:>10} {:>12.4} {:>12.4} {:>9.2}s",
            format!("neural network, {nn_steps} steps"),
            net.parameters(),
            net.dataset_loss(&train_xs, problem.train.targets(), loss),
            net.dataset_loss(&test_xs, test.targets(), loss),
            started.elapsed().as_secs_f64()
        );
    }

    println!(
        "\nfinal depth {}, worst RelErr {:.4}, Theorem 3.10 precondition held: {}",
        report.final_level(),
        report.worst_relative_error(),
        report.guarantees_hold()
    );
}
