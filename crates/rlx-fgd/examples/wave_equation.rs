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

//! Figure 3 — solving the wave equation by descending `L_PDE` in `H²(R³)`.
//!
//! ```text
//! cargo run -p rlx-fgd --release --example wave_equation -- --steps 400
//! cargo run -p rlx-fgd --release --example wave_equation -- --steps 400 --out /tmp/wave
//! ```
//!
//! With `--out`, writes one CSV per time slice plus the initial condition, so
//! the propagating solution can be plotted.

use rlx_fgd::algorithm::{FgdConfig, Refinement, run};
use rlx_fgd::rep::FourierGrid;
use rlx_fgd::tasks::wave::{InitialCondition, SincCubeBumps, WaveEquation};
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

fn main() {
    let steps: usize = arg("--steps", 200);
    let learning_rate: f64 = arg("--lr", 0.1);
    let tolerance: f64 = arg("--eps", 0.5);
    let initial_level: u32 = arg("--init-level", 3);
    let max_level: u32 = arg("--max-level", 5);
    let speed: f64 = arg("--speed", 2.0);
    let bandwidth: f64 = arg("--bandwidth", 5.0);

    let problem = WaveEquation {
        samples_per_cell: 4,
        ..WaveEquation::with_max_level(speed, SincCubeBumps::sampled(3, 2.0, 0), max_level)
    };
    let initial = FourierGrid::zeros(-bandwidth, bandwidth, initial_level);

    println!(
        "wave equation at c = {speed} on |ω| ≤ {bandwidth}, {steps} steps at η = {learning_rate}"
    );
    let before = problem.loss_terms(&initial);
    println!(
        "  start:  L_PDE = {:.5e}  (residual {:.3e}, displacement {:.3e}, velocity {:.3e})",
        before.total(),
        before.residual,
        before.displacement,
        before.velocity
    );

    let started = Instant::now();
    let report = run(
        &problem,
        initial,
        FgdConfig {
            learning_rate,
            tolerance,
            steps,
            refinement: Refinement::Adaptive { max_level },
            seed: 0,
        },
    );
    let elapsed = started.elapsed().as_secs_f64();

    let after = problem.loss_terms(&report.solution);
    println!(
        "  end:    L_PDE = {:.5e}  (residual {:.3e}, displacement {:.3e}, velocity {:.3e})",
        after.total(),
        after.residual,
        after.displacement,
        after.velocity
    );
    println!(
        "  {} frequency cells per axis after {} refinements, {elapsed:.2}s",
        1usize << report.final_level(),
        report.history.iter().map(|s| s.refinements).sum::<u32>()
    );
    println!(
        "  worst RelErr {:.4}, Theorem 3.10 precondition held: {}",
        report.worst_relative_error(),
        report.guarantees_hold()
    );

    if let Some(out) = flag("--out") {
        std::fs::create_dir_all(&out).expect("could not create the output directory");
        let resolution = 128usize;
        let axis: Vec<f64> = (0..resolution)
            .map(|i| -3.0 + 6.0 * i as f64 / (resolution - 1) as f64)
            .collect();
        let times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let slices = report.solution.evaluate_spatial_grid(&times, &axis, &axis);

        for (t_index, t) in times.iter().enumerate() {
            let mut csv = String::new();
            for i in 0..resolution {
                let row: Vec<String> = (0..resolution)
                    .map(|j| {
                        let value = slices[(t_index * resolution + i) * resolution + j];
                        format!("{value:.6}")
                    })
                    .collect();
                csv.push_str(&row.join(","));
                csv.push('\n');
            }
            std::fs::write(format!("{out}/t{t:.1}.csv"), csv).expect("could not write a slice");
        }

        let mut csv = String::new();
        for &x in &axis {
            let row: Vec<String> = axis
                .iter()
                .map(|&y| format!("{:.6}", problem.initial.spatial(x, y)))
                .collect();
            csv.push_str(&row.join(","));
            csv.push('\n');
        }
        std::fs::write(format!("{out}/initial.csv"), csv)
            .expect("could not write the initial condition");
        println!("  wrote {} slices and initial.csv to {out}", times.len());
    }
}
