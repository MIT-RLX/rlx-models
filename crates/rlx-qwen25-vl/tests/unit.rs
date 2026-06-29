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

//! Unit tests for Qwen2.5-VL crate scaffolding.

use rlx_qwen25_vl::{
    ACCEPTED_LM_ARCHES, ACCEPTED_MMPROJ_TYPES, build_multimodal_mrope_sections, synth,
    vision::MmProjConfig,
};

#[test]
fn accepted_arches_include_qwen2() {
    assert!(ACCEPTED_LM_ARCHES.contains(&"qwen2"));
    assert!(ACCEPTED_LM_ARCHES.contains(&"qwen2vl"));
}

#[test]
fn accepted_mmproj_includes_qwen25vl_merger() {
    assert!(ACCEPTED_MMPROJ_TYPES.contains(&"qwen2.5vl_merger"));
}

#[test]
fn mmproj_output_grid_matches_llama_cpp_formula() {
    let cfg = MmProjConfig {
        patch_size: 14,
        n_embd: 1280,
        n_head: 16,
        n_layer: 32,
        image_size: 448,
        image_min_pixels: 784,
        image_max_pixels: 1_048_576,
        n_merge: 2,
        eps: 1e-6,
        projector_type: "qwen2.5vl_merger".into(),
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        spatial_merge_size: 2,
        llm_hidden_size: 3584,
        n_ff: 5120,
        n_wa_pattern: 8,
        use_silu: true,
        use_rms_norm: true,
    };
    let (gx, gy) = cfg.output_grid(448, 448);
    assert_eq!(gx, 16);
    assert_eq!(gy, 16);
    assert_eq!(cfg.n_out_tokens(448, 448), 256);
}

#[test]
fn tiny_lm_mrope_sections_align_with_seq() {
    let _cfg = synth::tiny_lm_cfg();
    let sec = build_multimodal_mrope_sections(1, 2, 2, 1, 0);
    assert_eq!(sec.len(), 1 + 4 + 1);
}
