// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Adaptive Information Flow (AIF) — Liu et al., CVPR 2026.
// https://arxiv.org/abs/2604.15809
//
// Paper pipeline (Fig. 6):
//   1. Probe forward → token dynamics D_v, μ, Ent  (Eq. 2–4)
//      Modes: `prefill_v2t` (default) or `decode_step` (Fig. 6 one-step decode).
//      Native path: graph Q/K side outputs or CPU replay (`native_probe`).
//   2. Adaptive mask ratio via S0 vs S  (Eq. 5, Sec. 4.3)
//   3. Modulated causal mask: block **text queries → masked visual keys** only;
//      vision→vision paths stay open (Fig. 2).

mod config;
mod dynamics;
mod mask;
mod mode;
mod native_probe;
mod probe;

pub use config::AifConfig;
pub use dynamics::{
    MASK_RATIO_CANDIDATES, compute_mu, compute_token_entropies, distribution_entropy,
    select_adaptive_mask_ratio,
};
pub use mask::{
    VisionKeySpan, block_highest_entropy_keys, block_lowest_mu_keys, decode_mask_row_causal,
};
pub use mode::AifDynamicsMode;
pub use native_probe::{
    NativePrefillProbeInputs, compute_dynamics_eq2_prefill, dynamics_from_graph_qk_decode_step,
    dynamics_from_graph_qk_layers, native_prefill_probe,
};
pub use probe::AifProbe;

/// Sec. 3.2 ablation helper — mask lowest-μ keys at a fixed ratio (not adaptive AIF).
pub fn block_lowest_scored_keys(span: VisionKeySpan, mu: &[f32], ratio: f32) -> Vec<usize> {
    block_lowest_mu_keys(span, mu, ratio)
}

#[deprecated(note = "renamed to distribution_entropy (Eq. 5)")]
pub fn mu_distribution_entropy(mu: &[f32]) -> f32 {
    distribution_entropy(mu)
}

/// Back-compat alias — prefer [`AifConfig`].
pub type AifLiteConfig = AifConfig;

#[deprecated(note = "use select_adaptive_mask_ratio with token entropies (Sec. 4.3)")]
pub fn adaptive_mask_ratio(_scores: &[f32]) -> f32 {
    0.5
}

#[deprecated(note = "removed; use AifProbe::build from layer dynamics")]
pub fn legacy_adaptive_mask_ratio(_scores: &[f32]) -> f32 {
    0.5
}
