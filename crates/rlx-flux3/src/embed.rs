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

//! Timestep conditioning, on the host.
//!
//! Every token of a mode shares one conditioning vector, so the
//! modulations are a handful of matrix-vector products per step — far
//! cheaper on the host than as graph inputs. The single-stream modulation
//! is shared by all 28 single blocks, and the per-mode one by all early
//! blocks, so each is computed once per step.

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::config::{Flux3Config, Mode, TIME_EMBED_DIM};
use crate::weights::DitSource;

/// `bf16(x)`, round to nearest even.
pub fn round_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// BFL `timestep_embedding` as the reference evaluates it: the timesteps
/// are a bf16 tensor, so `1000·t` is formed and rounded in bf16 before the
/// f32 sinusoid (`[cos | sin]`), which is cast back to bf16 (`.to(t)`).
pub fn timestep_embedding(t: f32) -> Vec<f32> {
    let t = round_bf16(1000.0 * round_bf16(t));
    let half = TIME_EMBED_DIM / 2;
    let mut out = vec![0.0f32; TIME_EMBED_DIM];
    for i in 0..half {
        let freq = (-(10_000f32.ln()) * i as f32 / half as f32).exp();
        let arg = t * freq;
        out[i] = round_bf16(arg.cos());
        out[half + i] = round_bf16(arg.sin());
    }
    out
}

/// A bias-free linear layer on the host (`[out, in]` rows).
#[derive(Debug, Clone)]
pub struct HostLinear {
    pub w: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl HostLinear {
    pub fn load(src: &DitSource<'_>, key: &str) -> Result<Self> {
        let (w, shape) = src.load(&format!("{key}.weight"))?;
        ensure!(shape.len() == 2, "{key}: bad linear shape {shape:?}");
        Ok(Self {
            w,
            out_dim: shape[0],
            in_dim: shape[1],
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        self.w
            .par_chunks(self.in_dim)
            .map(|row| {
                row.iter()
                    .zip(x)
                    .map(|(&w, &v)| w as f64 * v as f64)
                    .sum::<f64>() as f32
            })
            .collect()
    }
}

fn silu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
}

/// `out_layer(silu(in_layer(x)))`.
#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    pub in_layer: HostLinear,
    pub out_layer: HostLinear,
}

impl MlpEmbedder {
    pub fn load(src: &DitSource<'_>, prefix: &str) -> Result<Self> {
        Ok(Self {
            in_layer: HostLinear::load(src, &format!("{prefix}.in_layer"))?,
            out_layer: HostLinear::load(src, &format!("{prefix}.out_layer"))?,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.out_layer.forward(&silu(&self.in_layer.forward(x)))
    }
}

/// Modulations of one mode: `[shift | scale | gate]` for the early and the
/// single-stream blocks, `3·D` each.
#[derive(Debug, Clone, PartialEq)]
pub struct ModeMods {
    pub early: Vec<f32>,
    pub single: Vec<f32>,
}

/// Everything the host derives from the flow time.
#[derive(Debug, Clone)]
pub struct Conditioning {
    pub time_in: MlpEmbedder,
    /// Flow map only: the time each step jumps to.
    pub ref_time_in: Option<MlpEmbedder>,
    early: Vec<(Mode, HostLinear)>,
    single: Vec<(Mode, HostLinear)>,
    /// `final_layer.image.adaLN_modulation.1`: `[shift | scale]`.
    final_mod: HostLinear,
}

impl Conditioning {
    pub fn load(src: &DitSource<'_>, cfg: &Flux3Config) -> Result<Self> {
        let mut early = Vec::new();
        let mut single = Vec::new();
        for m in Mode::ALL {
            let k = m.mod_key();
            early.push((
                m,
                HostLinear::load(src, &format!("early_stream_modulations.{k}.lin"))?,
            ));
            single.push((
                m,
                HostLinear::load(src, &format!("single_stream_modulations.{k}.lin"))?,
            ));
        }
        Ok(Self {
            time_in: MlpEmbedder::load(src, "time_in")?,
            ref_time_in: if cfg.flow_map {
                Some(MlpEmbedder::load(src, "ref_time_in")?)
            } else {
                None
            },
            early,
            single,
            final_mod: HostLinear::load(src, "final_layer.image.adaLN_modulation.1")?,
        })
    }

    /// The conditioning vector of a mode at flow time `t` (0 for text and
    /// references), plus the flow map's jump target `t_ref`.
    pub fn vec(&self, t: f32, t_ref: Option<f32>) -> Vec<f32> {
        let mut v = self.time_in.forward(&timestep_embedding(t));
        if let (Some(r), Some(tr)) = (&self.ref_time_in, t_ref) {
            let rv = r.forward(&timestep_embedding(tr));
            v.iter_mut().zip(rv).for_each(|(a, b)| *a += b);
        }
        v
    }

    /// `lin(silu(vec))` of the mode's early and single-stream modulations.
    pub fn mode_mods(&self, mode: Mode, vec: &[f32]) -> ModeMods {
        let s = silu(vec);
        let pick = |l: &[(Mode, HostLinear)]| {
            l.iter()
                .find(|(m, _)| *m == mode)
                .map(|(_, lin)| lin.forward(&s))
                .expect("every mode has a modulation")
        };
        ModeMods {
            early: pick(&self.early),
            single: pick(&self.single),
        }
    }

    /// `[shift | scale]` of the final layer, from the image mode's vector.
    pub fn final_mods(&self, vec_image: &[f32]) -> Vec<f32> {
        self.final_mod.forward(&silu(vec_image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_is_rounded_through_bf16_twice() {
        // 0.7234 → bf16 0.72265625 → ×1000 = 722.656… → bf16 (step 4 in [512, 1024)) = 724.
        let e = timestep_embedding(0.7234);
        assert_eq!(e[0], round_bf16(724f32.cos()));
        assert_eq!(e[128], round_bf16(724f32.sin()));
        assert_eq!(timestep_embedding(0.0)[0], 1.0);
    }
}
