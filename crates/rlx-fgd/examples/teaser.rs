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

//! Figure 1 — adaptive FGD against fixed-representation FGD on `L²` fitting.
//!
//! ```text
//! cargo run -p rlx-fgd --release --example teaser -- --steps 100 --lr 0.05 --eps 3
//! cargo run -p rlx-fgd --release --example teaser -- --nn-steps 500   # add the MLP baseline
//! ```

use rlx_fgd::algorithm::{FgdConfig, Refinement, run};
use rlx_fgd::baseline::{FourierMlp, MlpConfig, train_on_uniform_samples};
use rlx_fgd::rep::DyadicGrid;
use rlx_fgd::tasks::fit::{L2Fit, Spiral};
use rlx_fgd::theory::guaranteed_relative_error;
use std::time::Instant;

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let steps: usize = arg("--steps", 100);
    let learning_rate: f64 = arg("--lr", 0.05);
    let tolerance: f64 = arg("--eps", 3.0);
    let max_level: u32 = arg("--max-level", 10);
    let initial_level: u32 = arg("--init-level", 3);
    let nn_steps: usize = arg("--nn-steps", 0);

    let problem = L2Fit::with_max_level(Spiral::default(), max_level);

    println!("L² fitting of the Figure 1 spiral, {steps} steps at η = {learning_rate}");
    println!(
        "Algorithm 1 tolerance ε = {tolerance} ⇒ RelErr ≤ {:.4} (Theorem 3.10(i))\n",
        guaranteed_relative_error(tolerance)
    );
    println!(
        "{:<28} {:>10} {:>12} {:>14} {:>10}",
        "method", "dofs", "L(f_T)", "worst RelErr", "time"
    );
    println!("{}", "-".repeat(78));

    // The paper's baseline: FGD at a representation fixed for the whole run.
    for level in [4u32, 5, 6, 7, 8] {
        let started = Instant::now();
        let report = run(
            &problem,
            DyadicGrid::<2>::unit(level),
            FgdConfig {
                learning_rate,
                tolerance,
                steps,
                refinement: Refinement::Fixed,
                seed: 0,
            },
        );
        println!(
            "{:<28} {:>10} {:>12.3e} {:>14.4} {:>9.2}s",
            format!("fixed FGD, {0}×{0} grid", 1usize << level),
            report.history.last().map_or(0, |s| s.dofs),
            report.final_loss().unwrap_or(f64::NAN),
            report.worst_relative_error(),
            started.elapsed().as_secs_f64()
        );
    }

    let started = Instant::now();
    let report = run(
        &problem,
        DyadicGrid::<2>::unit(initial_level),
        FgdConfig {
            learning_rate,
            tolerance,
            steps,
            refinement: Refinement::Adaptive { max_level },
            seed: 0,
        },
    );
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{:<28} {:>10} {:>12.3e} {:>14.4} {:>9.2}s",
        "adaptive FGD (ours)",
        report.history.last().map_or(0, |s| s.dofs),
        report.final_loss().unwrap_or(f64::NAN),
        report.worst_relative_error(),
        elapsed
    );

    if nn_steps > 0 {
        let target = problem.target;
        let mut net = FourierMlp::new(MlpConfig::default());
        let parameters = net.parameters();
        let started = Instant::now();
        train_on_uniform_samples(&mut net, |x| target.eval(x[0], x[1]), nn_steps, 256, 1);
        let elapsed = started.elapsed().as_secs_f64();
        // Same quadrature as the FGD rows, so the losses are comparable.
        let resolution = problem.loss_quadrature;
        let cell = 1.0 / resolution as f64;
        let mut total = 0.0;
        for i in 0..resolution {
            let x = (i as f64 + 0.5) * cell;
            for j in 0..resolution {
                let y = (j as f64 + 0.5) * cell;
                let residual = net.eval(&[x, y]) - target.eval(x, y);
                total += residual * residual;
            }
        }
        println!(
            "{:<28} {:>10} {:>12.3e} {:>14} {:>9.2}s",
            format!("neural network, {nn_steps} steps"),
            parameters,
            0.5 * total * cell * cell,
            "—",
            elapsed
        );
    }

    println!("\nrefinement schedule (adaptive run):");
    let mut last = 0;
    for record in &report.history {
        if record.level != last {
            let cells = 1usize << record.level;
            println!(
                "  step {:>4}: {cells}×{cells} grid, ‖g‖ = {:.4e}, U = {:.4e}, L = {:.3e}",
                record.step,
                record.gradient_norm,
                record.error_bound,
                record.loss.unwrap_or(f64::NAN),
            );
            last = record.level;
        }
    }
    println!(
        "\nTheorem 3.10 precondition held every step: {}",
        report.guarantees_hold()
    );
}
