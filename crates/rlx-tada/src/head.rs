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

//! The flow-matching prediction head (`tada.nn.vibevoice`) and its ODE solver.
//!
//! At every autoregressive step the backbone hands this head one hidden state
//! and it integrates a velocity field from noise to a `acoustic_dim + time_dim`
//! vector: the token's acoustic latent concatenated with two Gray-coded frame
//! gaps. So "how this token sounds" and "how long it lasts" are sampled
//! jointly, by the same ODE, from the same conditioning.
//!
//! The whole solve — every Euler step, both classifier-free-guidance branches,
//! the guidance blend — is emitted as **one** graph. Timesteps and guidance
//! scales are fixed by the schedule, not by the data, so they fold into
//! constants and the ten head evaluations become a single `run` per token
//! instead of ten. The head's weights are interned once and reused across the
//! unrolled steps (see [`Ctx::param_keyed`]).

use crate::builder::{Ctx, F32, compile, lower};
use crate::config::TadaConfig;
use crate::prof::trace;
use crate::weights::{Linear, TensorStore};
use anyhow::{Result, bail};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::sync::Arc;

/// How Euler timesteps are spread over `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeSchedule {
    Uniform,
    Cosine,
    /// Uniform in log-SNR — the upstream default. Packs steps near `t = 0`,
    /// where the field is still resolving which token it is generating.
    #[default]
    LogSnr,
}

/// How the guidance scale decays across the solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CfgSchedule {
    Constant,
    Linear,
    /// Upstream default: full guidance at `t = 0`, none at `t = 1`.
    #[default]
    Cosine,
}

/// Sampling settings for one token's solve. Defaults mirror
/// `tada.modules.tada.InferenceOptions`.
#[derive(Debug, Clone)]
pub struct SolveOptions {
    pub num_steps: usize,
    pub acoustic_cfg_scale: f32,
    pub duration_cfg_scale: f32,
    pub cfg_schedule: CfgSchedule,
    pub time_schedule: TimeSchedule,
    pub noise_temperature: f32,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            num_steps: 10,
            acoustic_cfg_scale: 1.6,
            duration_cfg_scale: 1.0,
            cfg_schedule: CfgSchedule::Cosine,
            time_schedule: TimeSchedule::LogSnr,
            noise_temperature: 0.9,
        }
    }
}

impl SolveOptions {
    /// Whether the solve runs the conditional and unconditional branches
    /// together. Keyed off the *base* scale, as upstream does — a cosine
    /// schedule that happens to reach 1.0 at `t = 1` does not switch paths.
    pub fn uses_guidance(&self) -> bool {
        self.acoustic_cfg_scale != 1.0
    }
}

/// `t_span` for `num_steps` Euler steps: `num_steps + 1` values in `[0, 1]`.
pub fn time_schedule(num_steps: usize, schedule: TimeSchedule) -> Vec<f32> {
    let n = num_steps;
    match schedule {
        TimeSchedule::Uniform => (0..=n).map(|i| i as f32 / n as f32).collect(),
        TimeSchedule::Cosine => (0..=n)
            .map(|i| {
                let u = i as f32 / n as f32;
                0.5 * (1.0 - (std::f32::consts::PI * u).cos())
            })
            .collect(),
        TimeSchedule::LogSnr => {
            let mut t: Vec<f32> = (0..=n)
                .map(|i| {
                    let log_snr = 5.0 - 10.0 * (i as f32 / n as f32);
                    1.0 / (1.0 + (log_snr / 2.0).exp())
                })
                .collect();
            t[0] = 0.0;
            t[n] = 1.0;
            t
        }
    }
}

/// Guidance scale at `t`, per `_scheduled_cfg`.
pub fn scheduled_cfg(base: f32, t: f32, schedule: CfgSchedule) -> f32 {
    if base == 1.0 || schedule == CfgSchedule::Constant {
        return base;
    }
    match schedule {
        CfgSchedule::Linear => 1.0 + (base - 1.0) * (1.0 - t),
        CfgSchedule::Cosine => 1.0 + (base - 1.0) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos()),
        CfgSchedule::Constant => base,
    }
}

/// Sinusoidal timestep features, `cos` half first (matching upstream's
/// `cat([cos, sin])`, which is the reverse of the more common ordering).
fn timestep_features(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Per-layer weight *names*, resolved against the mapping at build time.
struct HeadLayer {
    ada: String,
    norm: String,
    gate: String,
    up: String,
    down: String,
    ffn_dim: usize,
}

/// `VibeVoiceDiffusionHead`.
///
/// Holds names and shapes, not weights. The head is ~350 M parameters and the
/// unrolled solver interns every one of them into the graph anyway, so keeping
/// a second f32 copy on the heap for the lifetime of the model bought nothing
/// but 1.4 GB. Only the timestep MLP is materialized, because it is evaluated
/// host-side to fold the schedule into constants.
pub struct DiffusionHead {
    store: Arc<TensorStore>,
    prefix: String,
    t_mlp0: Linear,
    t_mlp2: Linear,
    layers: Vec<HeadLayer>,
    hidden: usize,
    latent: usize,
    acoustic_dim: usize,
    freq_dim: usize,
    eps: f32,
}

impl DiffusionHead {
    pub fn load(store: Arc<TensorStore>, cfg: &TadaConfig, prefix: &str) -> Result<Self> {
        if cfg.bottleneck_dim.is_some() {
            bail!(
                "bottleneck_dim is set in the config but `bottleneck_proj` is not \
                 wired — no shipped TADA checkpoint uses it"
            );
        }
        let hidden = cfg.cond_dim();
        let latent = cfg.latent_dim();
        let mut layers = Vec::with_capacity(cfg.head_layers);
        for i in 0..cfg.head_layers {
            let p = format!("{prefix}layers.{i}");
            let gate = format!("{p}.ffn.gate_proj.weight");
            let ffn_dim = store.shape(&gate)?[0];
            layers.push(HeadLayer {
                ada: format!("{p}.adaLN_modulation.1.weight"),
                norm: format!("{p}.norm.weight"),
                gate,
                up: format!("{p}.ffn.up_proj.weight"),
                down: format!("{p}.ffn.down_proj.weight"),
                ffn_dim,
            });
        }
        let t_mlp0 = store.linear(&format!("{prefix}t_embedder.mlp.0.weight"))?;
        let freq_dim = t_mlp0.in_dim;
        let out_dim = store.shape(&format!("{prefix}final_layer.linear.weight"))?[0];
        if out_dim != latent {
            bail!(
                "head emits {out_dim} values but config derives a {latent}-wide latent \
                 ({} acoustic + {} time)",
                cfg.acoustic_dim,
                cfg.time_dim()
            );
        }
        Ok(Self {
            t_mlp2: store.linear(&format!("{prefix}t_embedder.mlp.2.weight"))?,
            t_mlp0,
            store,
            prefix: prefix.to_string(),
            layers,
            hidden,
            latent,
            acoustic_dim: cfg.acoustic_dim,
            freq_dim,
            eps: 1e-5,
        })
    }

    pub fn latent_dim(&self) -> usize {
        self.latent
    }

    /// `t_embedder(t)` on the host — `t` comes from a fixed schedule, so the
    /// whole timestep MLP folds into a constant vector per step and never
    /// enters the graph.
    fn timestep_embedding(&self, t: f32) -> Vec<f32> {
        let feats = timestep_features(t, self.freq_dim);
        let mut h = vec![0f32; self.hidden];
        for (o, slot) in h.iter_mut().enumerate() {
            let row = &self.t_mlp0.weight[o * self.freq_dim..(o + 1) * self.freq_dim];
            *slot = silu(row.iter().zip(&feats).map(|(w, x)| w * x).sum::<f32>());
        }
        let mut out = vec![0f32; self.hidden];
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &self.t_mlp2.weight[o * self.hidden..(o + 1) * self.hidden];
            *slot = row.iter().zip(&h).map(|(w, x)| w * x).sum();
        }
        out
    }

    /// Emit one head evaluation over `x` `[batch, latent]` with conditioning
    /// `c` `[batch, hidden]`, returning the velocity `[batch, latent]`.
    fn build_eval(
        &self,
        ctx: &mut Ctx,
        x: HirNodeId,
        c: HirNodeId,
        batch: usize,
        zero: HirNodeId,
        ones: HirNodeId,
    ) -> Result<HirNodeId> {
        let h = self.hidden;
        let pre = &self.prefix;
        let silu_c = ctx.g.silu(c);
        let mut cur = ctx.linear_keyed_try("noisy_images_proj", x, h, self.latent, || {
            self.store.get(&format!("{pre}noisy_images_proj.weight"))
        })?;

        for (i, layer) in self.layers.iter().enumerate() {
            let m = ctx.linear_keyed_try(&format!("layers.{i}.ada"), silu_c, 3 * h, h, || {
                self.store.get(&layer.ada)
            })?;
            let shift = ctx.g.narrow_(m, 1, 0, h);
            let scale = ctx.g.narrow_(m, 1, h, h);
            let gate = ctx.g.narrow_(m, 1, 2 * h, h);

            let nw = ctx.param_keyed_try(&format!("layers.{i}.norm"), &[h], || {
                self.store.get(&layer.norm)
            })?;
            let n = ctx.g.rms_norm(cur, nw, zero, self.eps);
            // modulate(x, shift, scale) = x·(1 + scale) + shift
            let s1 = ctx.g.add(scale, ones);
            let modulated = ctx.g.mul(n, s1);
            let modulated = ctx.g.add(modulated, shift);

            let ff = layer.ffn_dim;
            let g_proj =
                ctx.linear_keyed_try(&format!("layers.{i}.gate"), modulated, ff, h, || {
                    self.store.get(&layer.gate)
                })?;
            let u_proj =
                ctx.linear_keyed_try(&format!("layers.{i}.up"), modulated, ff, h, || {
                    self.store.get(&layer.up)
                })?;
            let act = ctx.g.silu(g_proj);
            let swiglu = ctx.g.mul(act, u_proj);
            let ffn = ctx.linear_keyed_try(&format!("layers.{i}.down"), swiglu, h, ff, || {
                self.store.get(&layer.down)
            })?;
            cur = ctx.g.gated_residual(cur, ffn, gate);
        }

        let m = ctx.linear_keyed_try("final.ada", silu_c, 2 * h, h, || {
            self.store
                .get(&format!("{pre}final_layer.adaLN_modulation.1.weight"))
        })?;
        let shift = ctx.g.narrow_(m, 1, 0, h);
        let scale = ctx.g.narrow_(m, 1, h, h);
        // `norm_final` is RMSNorm(elementwise_affine=False) — unit gamma.
        let unit = ctx.param_keyed("final.norm_ones", &[h], || vec![1.0; h]);
        let n = ctx.g.rms_norm(cur, unit, zero, self.eps);
        let s1 = ctx.g.add(scale, ones);
        let modulated = ctx.g.mul(n, s1);
        let modulated = ctx.g.add(modulated, shift);
        let _ = batch;
        ctx.linear_keyed_try("final.linear", modulated, self.latent, h, || {
            self.store.get(&format!("{pre}final_layer.linear.weight"))
        })
    }

    /// Compile the full ODE solve for `opts` into one graph.
    ///
    /// Inputs: `noise` `[1, latent]`, `cond` `[1, hidden]`, and — when guidance
    /// is on — `neg_cond` `[1, hidden]`. Output: the solved latent
    /// `[1, latent]`.
    pub fn compile_solver(&self, device: Device, opts: &SolveOptions) -> Result<CompiledGraph> {
        self.compile_solver_tagged(device, opts, "tada")
    }

    /// As [`Self::compile_solver`], but keyed for the on-disk compile cache.
    ///
    /// The solver unrolls every Euler step, so its graph is large and takes
    /// ~1.9 s to lower — more than the whole decode loop for a short utterance.
    /// It also never changes shape between utterances, so it is the single
    /// best candidate for caching.
    pub fn compile_solver_tagged(
        &self,
        device: Device,
        opts: &SolveOptions,
        tag: &str,
    ) -> Result<CompiledGraph> {
        // Deliberately NOT routed through `AotCache`. Measured on `tada-1b`,
        // the cached-LIR pipeline produced a solver that ran 70 % slower
        // (176 ms → 298 ms per token) and pushed peak RSS from 12.7 GB to
        // 20.5 GB — its fusion choices differ from `Session::compile`'s, and
        // for this graph they are worse. Lowering is not the cost here anyway:
        // almost all of the ~1.9 s is interning the head's 350 M parameters,
        // which caching the LIR would not avoid.
        let _ = tag;
        let t = std::time::Instant::now();
        let (hir, params) = self.build_solver(opts)?;
        let t_build = t.elapsed();
        let t = std::time::Instant::now();
        let g = lower(hir, "diffusion head solver")?;
        let t_lower = t.elapsed();
        let t = std::time::Instant::now();
        let out = compile(device, g, params);
        trace!(
            "    solver: build {t_build:?} lower {t_lower:?} compile {:?}",
            t.elapsed()
        );
        Ok(out)
    }

    fn build_solver(
        &self,
        opts: &SolveOptions,
    ) -> Result<(HirModule, crate::builder::NamedTensors)> {
        if opts.num_steps == 0 {
            bail!("flow matching needs at least one Euler step");
        }
        let guided = opts.uses_guidance();
        let batch = if guided { 2 } else { 1 };
        let h = self.hidden;
        let lat = self.latent;

        let mut hir = HirModule::new("tada_flow_head");
        let mut g = HirMut::new(&mut hir);
        let mut ctx = Ctx::new(&mut g);

        let noise = ctx.g.input("noise", Shape::new(&[1, lat], F32));
        let cond = ctx.g.input("cond", Shape::new(&[1, h], F32));
        let cond_all = if guided {
            let neg = ctx.g.input("neg_cond", Shape::new(&[1, h], F32));
            ctx.g.concat_(vec![cond, neg], 0)
        } else {
            cond
        };
        let cond_proj = ctx.linear_keyed_try("cond_proj", cond_all, h, h, || {
            self.store.get(&format!("{}cond_proj.weight", self.prefix))
        })?;

        let zero = ctx.param_keyed("zero_beta", &[h], || vec![0.0; h]);
        let ones = ctx.param_keyed("ones", &[1, h], || vec![1.0; h]);

        let t_span = time_schedule(opts.num_steps, opts.time_schedule);
        let mut x = noise;
        for step in 0..opts.num_steps {
            let t_cur = t_span[step];
            let dt = t_span[step + 1] - t_cur;

            let t_emb = ctx.param_keyed(&format!("t_emb.{step}"), &[1, h], || {
                self.timestep_embedding(t_cur)
            });
            let c = ctx.g.add(cond_proj, t_emb);

            let x_in = if guided {
                ctx.g.concat_(vec![x, x], 0)
            } else {
                x
            };
            let v = self.build_eval(&mut ctx, x_in, c, batch, zero, ones)?;

            let velocity = if guided {
                // Per-field guidance: the acoustic half and the duration half
                // get different scales, so the blend is a constant row vector.
                let a = scheduled_cfg(opts.acoustic_cfg_scale, t_cur, opts.cfg_schedule);
                let d = scheduled_cfg(opts.duration_cfg_scale, t_cur, opts.cfg_schedule);
                let mut blend = vec![d; lat];
                blend[..self.acoustic_dim].fill(a);
                let blend = ctx.param_keyed(&format!("cfg.{step}"), &[1, lat], || blend);

                let pos = ctx.g.narrow_(v, 0, 0, 1);
                let neg = ctx.g.narrow_(v, 0, 1, 1);
                let diff = ctx.g.sub(pos, neg);
                let scaled = ctx.g.mul(diff, blend);
                ctx.g.add(neg, scaled)
            } else {
                v
            };

            let dt_p = ctx.param_keyed(&format!("dt.{step}"), &[1, 1], || vec![dt]);
            let dt_e = ctx.g.expand_(dt_p, vec![1, lat as i64]);
            let stepped = ctx.g.mul(velocity, dt_e);
            x = ctx.g.add(x, stepped);
        }

        let params = ctx.into_params();
        hir.set_outputs(vec![x]);
        Ok((hir, params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_span_the_unit_interval_monotonically() {
        for s in [
            TimeSchedule::Uniform,
            TimeSchedule::Cosine,
            TimeSchedule::LogSnr,
        ] {
            let t = time_schedule(10, s);
            assert_eq!(t.len(), 11, "{s:?}");
            assert_eq!(t[0], 0.0, "{s:?}");
            assert_eq!(t[10], 1.0, "{s:?}");
            for w in t.windows(2) {
                assert!(w[0] < w[1], "{s:?} not increasing: {t:?}");
            }
        }
    }

    /// Values from `torch.sigmoid(-torch.linspace(5, -5, 11) / 2)` with the
    /// endpoints clipped, i.e. exactly what `_build_time_schedule` returns.
    #[test]
    fn log_snr_matches_the_reference_schedule() {
        let t = time_schedule(10, TimeSchedule::LogSnr);
        let want = [
            0.0, 0.119203, 0.182426, 0.268941, 0.377541, 0.5, 0.622459, 0.731059, 0.817574,
            0.880797, 1.0,
        ];
        for (i, (&a, &b)) in t.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-5, "step {i}: {a} vs {b}");
        }
        // The sigmoid is odd about its midpoint, so the schedule is symmetric.
        for i in 1..10 {
            assert!((t[i] + t[10 - i] - 1.0).abs() < 1e-5, "step {i}");
        }
    }

    #[test]
    fn cosine_matches_the_reference_schedule() {
        let t = time_schedule(10, TimeSchedule::Cosine);
        let want = [
            0.0, 0.024472, 0.095491, 0.206107, 0.345492, 0.5, 0.654509, 0.793893, 0.904509,
            0.975528, 1.0,
        ];
        for (i, (&a, &b)) in t.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-5, "step {i}: {a} vs {b}");
        }
    }

    #[test]
    fn cosine_guidance_decays_from_base_to_one() {
        assert!((scheduled_cfg(1.6, 0.0, CfgSchedule::Cosine) - 1.6).abs() < 1e-6);
        assert!((scheduled_cfg(1.6, 1.0, CfgSchedule::Cosine) - 1.0).abs() < 1e-6);
        assert!((scheduled_cfg(1.6, 0.5, CfgSchedule::Cosine) - 1.3).abs() < 1e-5);
    }

    #[test]
    fn linear_guidance_decays_linearly() {
        assert!((scheduled_cfg(2.0, 0.25, CfgSchedule::Linear) - 1.75).abs() < 1e-6);
    }

    #[test]
    fn a_unit_base_scale_is_never_rescaled() {
        for s in [
            CfgSchedule::Constant,
            CfgSchedule::Linear,
            CfgSchedule::Cosine,
        ] {
            for t in [0.0, 0.3, 1.0] {
                assert_eq!(scheduled_cfg(1.0, t, s), 1.0);
            }
        }
    }

    #[test]
    fn timestep_features_put_cosine_first() {
        let f = timestep_features(0.0, 8);
        // At t = 0 every angle is 0 → cos half is 1, sin half is 0.
        assert_eq!(&f[..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&f[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn timestep_features_sweep_frequencies() {
        let f = timestep_features(1.0, 8);
        // Highest frequency is index 0 (freq 1.0); the last is 10000^(-3/4).
        assert!((f[0] - 1.0f32.cos()).abs() < 1e-6);
        let lowest = (-(10000f32.ln()) * 3.0 / 4.0).exp();
        assert!((f[3] - lowest.cos()).abs() < 1e-6);
    }
}
