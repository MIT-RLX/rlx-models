// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Predictor quick check (5k) and LCESN paper protocol (14k).

use rlx_narma10::{
    LCESN_TIMESTEPS, LCESN_TRAIN_SAMPLES, TrainConfig, bench_predictors, generate,
    persistence_nrmse,
};

#[test]
fn predictors_beat_persistence_quick() {
    let series = generate(5_000, 7);
    let cfg = TrainConfig::quick();
    let persist = persistence_nrmse(&series.targets);
    assert!(persist > 0.5, "unexpected persistence baseline {persist}");

    for row in bench_predictors(&series, &cfg).unwrap() {
        assert!(
            row.test_nrmse < persist,
            "{} test NRMSE {:.4} not better than persistence {:.4}",
            row.name,
            row.test_nrmse,
            persist
        );
        let ceiling = if row.name == "local_esn" { 0.45 } else { 0.35 };
        assert!(
            row.test_nrmse < ceiling,
            "{} test NRMSE {:.4} above quick-check ceiling {ceiling}",
            row.name,
            row.test_nrmse
        );
    }
}

#[test]
fn local_esn_lcesn_paper_protocol() {
    let series = generate(LCESN_TIMESTEPS, 42);
    let cfg = TrainConfig::lcesn();
    assert_eq!(
        cfg.expected_train_samples(LCESN_TIMESTEPS),
        LCESN_TRAIN_SAMPLES,
        "train sample count should match LCESN paper"
    );

    let rows = bench_predictors(&series, &cfg).unwrap();
    let local = rows.iter().find(|r| r.name == "local_esn").unwrap();
    assert_eq!(local.train_samples, LCESN_TRAIN_SAMPLES);
    assert!(
        local.test_samples >= 1_000,
        "expected ~1k test samples, got {}",
        local.test_samples
    );

    let persist = persistence_nrmse(&series.targets);
    assert!(
        local.test_nrmse < persist,
        "local_esn {:.4} should beat persistence {:.4}",
        local.test_nrmse,
        persist
    );
    assert!(
        local.test_nrmse < 0.45,
        "local_esn test NRMSE {:.4} outside literature-scale band",
        local.test_nrmse
    );
}

#[test]
fn poly_readout_competitive_with_esn() {
    let series = generate(5_000, 99);
    let cfg = TrainConfig::quick();
    let rows = bench_predictors(&series, &cfg).unwrap();
    let esn = rows.iter().find(|r| r.name == "esn_ridge").unwrap();
    let poly = rows.iter().find(|r| r.name == "poly_readout_esn").unwrap();
    assert!(
        poly.test_nrmse <= esn.test_nrmse * 1.35 + 1e-6,
        "poly {:.4} much worse than esn {:.4}",
        poly.test_nrmse,
        esn.test_nrmse
    );
}
