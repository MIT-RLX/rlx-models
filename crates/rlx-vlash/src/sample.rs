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

//! Host-side flow-matching `sample_actions`: a short Euler integration of the
//! learned velocity field produced by the denoise-step graph.
//!
//! ```text
//!   x_t = noise ;  time = 1 ;  dt = -1 / num_inference_steps
//!   repeat num_inference_steps:
//!     v_t  = denoise(prefix, state, x_t, time_emb(time), rope, bias)
//!     x_t += dt · v_t ;  time += dt
//! ```
//! (reference `PI0Model.sample_actions` / `denoise_step`). The prefix is fixed
//! across steps (design A recomputes it each step but the result is identical).

use rlx_runtime::CompiledGraph;

use crate::config::{VlashConfig, VlashVariant};
use crate::prefix::AttnInputs;
use crate::util::sinusoidal_time_embedding;

/// Build the `time_emb` graph input for a given scalar `time`.
/// π₀ broadcasts the `[D]` embedding to `[chunk, D]`; π₀.₅ passes `[D]`.
pub fn time_input(cfg: &VlashConfig, time: f32) -> Vec<f32> {
    let d = cfg.expert.hidden;
    let base = sinusoidal_time_embedding(time, d, cfg.min_period, cfg.max_period);
    match cfg.variant {
        VlashVariant::Pi0 => {
            let mut out = Vec::with_capacity(cfg.chunk_size * d);
            for _ in 0..cfg.chunk_size {
                out.extend_from_slice(&base);
            }
            out
        }
        VlashVariant::Pi05 => base,
    }
}

/// Integrate the velocity field for `num_inference_steps` starting from `noise`
/// (`[chunk · max_action_dim]`). Returns the final `x_t` (`[chunk · max_action_dim]`,
/// still normalized + padded — caller truncates + unnormalizes).
///
/// `prefix_emb`, `state`, and `attn` are fixed across steps.
pub fn sample_actions(
    denoise: &mut CompiledGraph,
    cfg: &VlashConfig,
    prefix_emb: &[f32],
    state: &[f32],
    attn: &AttnInputs,
    noise: &[f32],
) -> Vec<f32> {
    let steps = cfg.num_inference_steps.max(1);
    let dt = -1.0 / steps as f32;
    let mut x_t = noise.to_vec();
    let mut time = 1.0f32;
    for _ in 0..steps {
        let time_emb = time_input(cfg, time);
        let v = {
            let out = denoise.run(&[
                ("prefix_emb", prefix_emb),
                ("state", state),
                ("actions", x_t.as_slice()),
                ("time_emb", time_emb.as_slice()),
                ("cos", attn.cos.as_slice()),
                ("sin", attn.sin.as_slice()),
                ("attn_bias", attn.bias.as_slice()),
            ]);
            out.into_iter().next().expect("denoise → velocity")
        };
        debug_assert_eq!(v.len(), x_t.len(), "velocity/x_t length mismatch");
        for (x, dv) in x_t.iter_mut().zip(v.iter()) {
            *x += dt * dv;
        }
        time += dt;
    }
    x_t
}
