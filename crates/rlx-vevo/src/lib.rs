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

//! # rlx-vevo
//!
//! **Vevo2** (Amphion) — controllable zero-shot voice/style imitation on RLX. Vevo
//! **disentangles** speech into content, style (content-style), and timbre, and
//! recombines them from different references: an autoregressive content→content-
//! style transformer (conditioned on a *style* reference) feeds a flow-matching
//! content-style→mel stage (conditioned on a *timbre* reference), then a vocoder.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **AR content→style transformer** → Llama-style (`rlx-llama32`).
//! - **Flow token→mel + guidance** → [`rlx_audio_blocks::sampling`].
//! - **Vocoder** → BigVGAN (`rlx-neutts`).
//!
//! Checkpoint-free, unit-tested core: the config, **unit run-length reduction**
//! ([`collapse_repeats`]/[`expand_units`] — the reduced-unit representation the
//! content tokenizer emits), and the disentangled-control presets ([`VevoControl`]).
//! The AR + flow + vocoder graphs are the next step.

use anyhow::{Result, ensure};
use rlx_audio_blocks::sampling::{FlowMatchEuler, classifier_free_guidance};

/// Vevo2 config.
#[derive(Debug, Clone, PartialEq)]
pub struct VevoConfig {
    pub sample_rate: usize,
    /// Content (VQ) token vocabulary.
    pub content_vocab: usize,
    /// Content-style token vocabulary.
    pub style_vocab: usize,
    // AR content→content-style transformer.
    pub ar_hidden: usize,
    pub ar_layers: usize,
    pub ar_heads: usize,
    // Flow content-style→mel.
    pub flow_steps: usize,
    pub mel_dim: usize,
    pub cfg_scale: f32,
}

impl Default for VevoConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            content_vocab: 8192,
            style_vocab: 16_384,
            ar_hidden: 1024,
            ar_layers: 16,
            ar_heads: 16,
            flow_steps: 32,
            mel_dim: 100,
            cfg_scale: 2.0,
        }
    }
}

impl VevoConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.content_vocab > 0, "content_vocab must be > 0");
        ensure!(self.style_vocab > 0, "style_vocab must be > 0");
        ensure!(self.flow_steps > 0, "flow_steps must be > 0");
        Ok(())
    }

    /// The content-style→mel flow-matching sampler (noise → data).
    pub fn flow_scheduler(&self, steps: usize) -> FlowMatchEuler {
        FlowMatchEuler::ascending(steps)
    }

    /// Apply this model's classifier-free guidance to a (cond, uncond) velocity.
    pub fn guided(&self, v_cond: &[f32], v_uncond: &[f32]) -> Vec<f32> {
        classifier_free_guidance(v_cond, v_uncond, self.cfg_scale)
    }
}

/// Where a disentangled attribute is taken from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefSource {
    /// Keep the source utterance's own attribute.
    Source,
    /// Take it from a reference.
    Reference,
}

/// Which references drive style and timbre (content always comes from the source).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VevoControl {
    pub style: RefSource,
    pub timbre: RefSource,
}

impl VevoControl {
    /// Voice conversion: keep source style/prosody, swap only timbre.
    pub fn voice_conversion() -> Self {
        Self {
            style: RefSource::Source,
            timbre: RefSource::Reference,
        }
    }

    /// Style transfer: take style from a reference, keep source timbre.
    pub fn style_transfer() -> Self {
        Self {
            style: RefSource::Reference,
            timbre: RefSource::Source,
        }
    }

    /// Full imitation: style and timbre both from references.
    pub fn full_imitation() -> Self {
        Self {
            style: RefSource::Reference,
            timbre: RefSource::Reference,
        }
    }

    /// Whether the output timbre is converted (differs from source).
    pub fn converts_timbre(&self) -> bool {
        self.timbre == RefSource::Reference
    }

    /// Whether the output style is converted (differs from source).
    pub fn converts_style(&self) -> bool {
        self.style == RefSource::Reference
    }
}

/// Run-length reduce a sequence of discrete content units into `(units, durations)`
/// — the reduced-unit representation Vevo's content tokenizer emits (consecutive
/// duplicate units collapsed).
pub fn collapse_repeats(units: &[i32]) -> (Vec<i32>, Vec<usize>) {
    let mut out = Vec::new();
    let mut durs = Vec::new();
    for &u in units {
        if out.last() == Some(&u) {
            *durs.last_mut().unwrap() += 1;
        } else {
            out.push(u);
            durs.push(1);
        }
    }
    (out, durs)
}

/// Inverse of [`collapse_repeats`]: expand `(units, durations)` back to a full
/// per-frame unit sequence.
pub fn expand_units(units: &[i32], durations: &[usize]) -> Result<Vec<i32>> {
    ensure!(
        units.len() == durations.len(),
        "units/durations length mismatch"
    );
    let mut out = Vec::new();
    for (&u, &n) in units.iter().zip(durations) {
        out.extend(std::iter::repeat_n(u, n));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_and_flow() {
        let c = VevoConfig::default();
        c.validate().unwrap();
        let s = c.flow_scheduler(8);
        assert_eq!(s.sigmas[0], 0.0);
        assert_eq!(*s.sigmas.last().unwrap(), 1.0);
        assert_eq!(c.guided(&[1.0], &[0.0]), vec![2.0]);
    }

    #[test]
    fn control_presets() {
        assert!(VevoControl::voice_conversion().converts_timbre());
        assert!(!VevoControl::voice_conversion().converts_style());
        assert!(VevoControl::style_transfer().converts_style());
        assert!(!VevoControl::style_transfer().converts_timbre());
        let full = VevoControl::full_imitation();
        assert!(full.converts_style() && full.converts_timbre());
    }

    #[test]
    fn unit_rle_roundtrips() {
        let units = vec![5, 5, 5, 3, 3, 7];
        let (u, d) = collapse_repeats(&units);
        assert_eq!(u, vec![5, 3, 7]);
        assert_eq!(d, vec![3, 2, 1]);
        assert_eq!(expand_units(&u, &d).unwrap(), units);
    }

    #[test]
    fn unit_rle_edge_cases() {
        let (u, d) = collapse_repeats(&[]);
        assert!(u.is_empty() && d.is_empty());
        assert!(expand_units(&[1, 2], &[1]).is_err());
    }
}
