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

//! Compile-time backend feature wiring (rlx-llama32 device pass-through).

use rlx_neutts::{
    cuda_feature_enabled, enabled_backend_labels, gpu_feature_enabled, metal_feature_enabled,
    mlx_feature_enabled, rocm_feature_enabled, vulkan_feature_enabled,
};

#[test]
fn backbone_device_features_match_cfg() {
    assert_eq!(metal_feature_enabled(), cfg!(feature = "metal"));
    assert_eq!(mlx_feature_enabled(), cfg!(feature = "mlx"));
    assert_eq!(cuda_feature_enabled(), cfg!(feature = "cuda"));
    assert_eq!(rocm_feature_enabled(), cfg!(feature = "rocm"));
    assert_eq!(vulkan_feature_enabled(), cfg!(feature = "vulkan"));
    assert_eq!(gpu_feature_enabled(), cfg!(feature = "gpu"));
}

#[test]
fn enabled_labels_non_empty_with_codec() {
    let labels = enabled_backend_labels();
    assert!(
        labels
            .iter()
            .any(|l| l.contains("codec") || l.contains("rlx-llama32"))
    );
}
