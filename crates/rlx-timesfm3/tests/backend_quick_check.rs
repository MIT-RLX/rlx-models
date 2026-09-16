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

//! Per-backend quick check on the synthetic tiny model.

use ndarray::Array3;
use rlx_runtime::Device;
use rlx_timesfm3::{
    TimesFM3Config, TimesFM3Model, TimesFM3Session, available_devices, device_label,
};

#[test]
fn all_available_backends_decode() {
    let cfg = TimesFM3Config::synth_tiny();
    let ctx: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();

    for dev in available_devices() {
        let model = TimesFM3Model::synth(cfg.clone(), 11);
        let mut session = TimesFM3Session::from_model(model, dev);
        let out = session
            .decode(arr.view(), 4, None, None, None)
            .unwrap_or_else(|e| panic!("{} decode failed: {e}", device_label(dev)));
        assert_eq!(out.shape(), &[1, 1, 4, 9], "{}", device_label(dev));
        assert!(
            out.iter().all(|x| x.is_finite()),
            "{} produced non-finite values",
            device_label(dev)
        );
        let _ = Device::Cpu; // keep import used on all builds
    }
}
