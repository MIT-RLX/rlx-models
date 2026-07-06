//! Benchmark the three NARMA-10 predictors on a synthetic series.
//!
//! See [README](../README.md) for protocol details.

use rlx_narma10::{
    LCESN_TIMESTEPS, LCESN_TRAIN_SAMPLES, LCESN_WASHOUT, QUICK_TIMESTEPS, TrainConfig,
    bench_predictors, generate, persistence_nrmse,
};

fn main() {
    let mut protocol = "lcesn".to_string();
    let mut steps = LCESN_TIMESTEPS;
    let mut steps_set = false;
    let mut seed = 42u64;
    let mut washout = LCESN_WASHOUT;

    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--protocol=") {
            protocol = v.to_string();
        } else if let Some(v) = arg.strip_prefix("--steps=") {
            steps = v.parse().expect("--steps");
            steps_set = true;
        } else if let Some(v) = arg.strip_prefix("--seed=") {
            seed = v.parse().expect("--seed");
        } else if let Some(v) = arg.strip_prefix("--washout=") {
            washout = v.parse().expect("--washout");
        }
    }

    let mut cfg = match protocol.as_str() {
        "lcesn" => {
            if !steps_set {
                steps = LCESN_TIMESTEPS;
            } else {
                steps = steps.max(LCESN_TIMESTEPS);
            }
            TrainConfig::lcesn()
        }
        "quick" => {
            if !steps_set {
                steps = QUICK_TIMESTEPS;
            }
            TrainConfig::quick()
        }
        other => {
            eprintln!("unknown --protocol={other} (use lcesn or quick)");
            std::process::exit(1);
        }
    };

    cfg.washout = washout;
    cfg.seed = seed;
    if protocol.as_str() == "lcesn" && steps != LCESN_TIMESTEPS {
        cfg.train_frac = TrainConfig::train_frac_for_collected(steps, washout, LCESN_TRAIN_SAMPLES);
    }

    let series = generate(steps, seed);
    let expected_train = cfg.expected_train_samples(steps);
    let persist = persistence_nrmse(&series.targets);

    println!(
        "NARMA-10 predictor benchmark (protocol={protocol}, steps={steps}, seed={seed}, washout={washout})"
    );
    println!("expected post-washout train samples: {expected_train}");
    println!("persistence baseline NRMSE: {persist:.4}");
    println!();
    println!(
        "{:<20} {:>12} {:>12} {:>8} {:>8}",
        "model", "train_nrmse", "test_nrmse", "train_n", "test_n"
    );
    println!("{}", "-".repeat(64));

    for row in bench_predictors(&series, &cfg).expect("bench") {
        println!(
            "{:<20} {:>12.4} {:>12.4} {:>8} {:>8}",
            row.name, row.train_nrmse, row.test_nrmse, row.train_samples, row.test_samples
        );
    }
}
