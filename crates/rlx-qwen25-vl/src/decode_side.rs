// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Decode side-output layout (KV cache + optional AIF Q/K taps).

use anyhow::{Context, Result, ensure};

/// Output tensor order from Qwen3 decode with optional Q/K export.
#[derive(Debug, Clone, Copy)]
pub struct DecodeSideLayout {
    pub n_layers: usize,
    pub export_qk: bool,
}

impl DecodeSideLayout {
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
        let logits = iter.next().context("decode logits missing")?;
        let mut layers_k = Vec::with_capacity(self.n_layers);
        let mut layers_v = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            layers_k.push(iter.next().context("decode past_k missing")?);
            layers_v.push(iter.next().context("decode past_v missing")?);
        }
        let qk = if self.export_qk {
            let mut q_layers = Vec::with_capacity(self.n_layers);
            let mut k_layers = Vec::with_capacity(self.n_layers);
            for _ in 0..self.n_layers {
                q_layers.push(iter.next().context("decode probe_q missing")?);
                k_layers.push(iter.next().context("decode probe_k missing")?);
            }
            Some((q_layers, k_layers))
        } else {
            None
        };
        ensure!(
            iter.next().is_none(),
            "decode produced extra outputs beyond layout"
        );
        Ok((logits, layers_k, layers_v, qk))
    }
}
