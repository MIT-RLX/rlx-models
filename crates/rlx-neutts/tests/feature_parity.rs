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

//! Feature-flag parity.

use rlx_neutts::{
    burn_feature_enabled, codec_feature_enabled, llama_feature_enabled,
    parity_llama_cpp_feature_enabled, wgpu_feature_enabled,
};

#[test]
fn codec_feature_always_on() {
    assert!(codec_feature_enabled());
}

#[test]
fn wgpu_is_alias_for_burn() {
    assert_eq!(wgpu_feature_enabled(), burn_feature_enabled());
}

#[test]
fn llama_is_production_backbone_feature() {
    assert_eq!(llama_feature_enabled(), cfg!(feature = "llama"));
}

#[test]
fn parity_llama_cpp_is_separate() {
    assert_eq!(
        parity_llama_cpp_feature_enabled(),
        cfg!(feature = "parity-llama-cpp")
    );
}
