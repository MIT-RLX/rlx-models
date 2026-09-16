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

//! Google **TimesFM-3** — zero-shot multivariate time-series forecasting.
//!
//! Native Rust inference port of [`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch)
//! (Stacked Mixing Transformer, RevIN, contiguous patch masking).
//!
//! # Quick start
//!
//! ```bash
//! # Synthetic tiny model (no download)
//! cargo run -p rlx-timesfm3 --release -- --synth --horizon 32
//!
//! # Official weights (non-commercial license), Metal backend
//! huggingface-cli download google/timesfm-3.0-pytorch --local-dir .cache/timesfm3
//! cargo run -p rlx-timesfm3 --release --features metal -- --weights .cache/timesfm3 --device metal --horizon 128
//! ```

pub mod cli;
pub mod config;
pub mod device;
pub mod flow;
pub mod forecaster;
pub mod host;
#[cfg(feature = "dev")]
pub mod parity;
pub mod rope;
pub mod session;
pub mod weights;

pub use config::TimesFM3Config;
pub use device::{available_device_labels, available_devices, device_label, resolve_device};
pub use forecaster::{ForecastOutput, TimesFM3Forecaster, TimesFM3ForecasterBuilder};
pub use host::TimesFM3Model;
#[cfg(feature = "dev")]
pub use parity::{compare_core, compare_resblock, max_abs_diff};
pub use session::TimesFM3Session;
pub use weights::TimesFM3Weights;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn synth_decode_runs() {
        let cfg = TimesFM3Config::synth_tiny();
        let model = TimesFM3Model::synth(cfg, 7);
        let ctx = Array3::from_shape_vec((1, 1, 64), vec![0.1; 64]).unwrap();
        let out = model.decode(ctx.view(), 16, None, None, None);
        assert_eq!(out.shape(), &[1, 1, 16, 9]);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn synth_decode_128_context() {
        let cfg = TimesFM3Config::synth_tiny();
        let model = TimesFM3Model::synth(cfg, 42);
        let ctx: Vec<f32> = (0..128).map(|t| (t as f32 * 0.1).sin()).collect();
        let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();
        let out = model.decode(arr.view(), 8, None, None, None);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
