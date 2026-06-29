// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Prefill side-output layout (KV cache + optional AIF Q/K taps).

use anyhow::{Context, Result, ensure};

/// Output tensor order from [`crate::lm_flow::build_qwen25_vl_prefill_mrope_built`].
#[derive(Debug, Clone, Copy)]
pub struct PrefillSideLayout {
    pub n_layers: usize,
    pub export_qk: bool,
}

impl PrefillSideLayout {
    pub fn expected_outputs(&self) -> usize {
        1 + 2 * self.n_layers + if self.export_qk { 2 * self.n_layers } else { 0 }
    }

    pub fn parse_kv_qk(
        &self,
        outputs: impl Iterator<Item = Vec<f32>>,
    ) -> Result<(
        Vec<f32>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    )> {
        let mut iter = outputs;
        let hidden = iter.next().context("prefill hidden missing")?;
        let mut layers_k = Vec::with_capacity(self.n_layers);
        let mut layers_v = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            layers_k.push(iter.next().context("past_k missing")?);
            layers_v.push(iter.next().context("past_v missing")?);
        }
        let qk = if self.export_qk {
            let mut q_layers = Vec::with_capacity(self.n_layers);
            let mut k_layers = Vec::with_capacity(self.n_layers);
            for _ in 0..self.n_layers {
                q_layers.push(iter.next().context("probe_q missing")?);
                k_layers.push(iter.next().context("probe_k missing")?);
            }
            Some((q_layers, k_layers))
        } else {
            None
        };
        ensure!(
            iter.next().is_none(),
            "prefill side layout: trailing outputs (expected {})",
            self.expected_outputs()
        );
        Ok((hidden, layers_k, layers_v, qk))
    }
}
