// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use rlx_qwen25_vl::{
    AifConfig, AifProbe, VisionKeySpan, compute_mu, compute_token_entropies, distribution_entropy,
    select_adaptive_mask_ratio,
};

#[test]
fn paper_eq3_eq4_eq5_synthetic() {
    let dynamics = vec![
        vec![0.0, 0.0, 0.9, 0.0],
        vec![0.2, 0.2, 0.2, 0.2],
        vec![0.1, 0.3, 0.1, 0.3],
        vec![0.05, 0.05, 0.05, 0.05],
    ];
    let mu = compute_mu(&dynamics);
    assert_eq!(mu.len(), 4);
    assert!((mu[0] - 0.225).abs() < 1e-5);

    let ent = compute_token_entropies(&dynamics, &mu);
    assert_eq!(ent.len(), 4);
    assert!(ent[0] < ent[1] + 1e-5);

    let s0 = distribution_entropy(&mu);
    assert!(s0 > 0.0);

    let ratio = select_adaptive_mask_ratio(&mu, &ent);
    assert!((0.1..=0.9).contains(&ratio));

    let probe = AifProbe::build(dynamics);
    assert_eq!(probe.mu, mu);
    assert!((probe.s0 - s0).abs() < 1e-5);
    assert_eq!(probe.mask_ratio, ratio);
}

#[test]
fn probe_blocked_keys_use_highest_entropy() {
    let dynamics = vec![vec![0.25; 8]; 16];
    let probe = AifProbe::build(dynamics);
    let blocked = probe.blocked_keys(VisionKeySpan { start: 5, end: 21 });
    assert!(!blocked.is_empty());
    assert!(blocked.iter().all(|&k| (5..21).contains(&k)));
}

#[test]
fn aif_config_from_probe() {
    let probe = AifProbe::build(vec![vec![0.0, 1.0], vec![0.5, 0.5]]);
    let cfg = AifConfig::from(&probe);
    let span = VisionKeySpan { start: 0, end: 2 };
    assert_eq!(cfg.blocked_keys(span), probe.blocked_keys(span));
}
