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

//! FLUX.2 basic tests (synthetic tiny config).

mod compile_support;

use rlx_models::flux2::synthetic_weights;
use rlx_models::{
    Flux2Config, Flux2ForwardInput, build_flux2_minimal_hir, compile_flux2_minimal,
    extract_flux2_weights, flux2_transformer_forward, prepare_weight_map,
};

#[test]
fn native_forward_tiny() {
    let cfg = Flux2Config::tiny();
    let wm = synthetic_weights(&cfg);
    let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
    let out = flux2_transformer_forward(
        &w,
        &cfg,
        Flux2ForwardInput {
            hidden_states: &vec![0.0; cfg.in_channels * 4],
            encoder_hidden_states: &vec![0.0; cfg.joint_attention_dim * 3],
            timestep: &[0.5],
            timestep_target: None,
            guidance: Some(&[3.5]),
            img_ids: &[0.0; 16],
            txt_ids: &[0.0; 12],
            batch: 1,
            img_seq: 4,
            txt_seq: 3,
        },
    )
    .unwrap();
    assert_eq!(out.len(), 4 * cfg.proj_out_dim());
}

#[test]
fn minimal_hir_lowers() {
    let cfg = Flux2Config::tiny();
    let wm = synthetic_weights(&cfg);
    let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
    let (hir, _) = build_flux2_minimal_hir(&cfg, &w, 1, 4).unwrap();
    assert_eq!(hir.outputs.len(), 1);
    assert_eq!(hir.lower_to_mir().unwrap().outputs().len(), 1);
}

#[test]
fn minimal_compiles_cpu() {
    let cfg = Flux2Config::tiny();
    let wm = synthetic_weights(&cfg);
    let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
    let (mut compiled, _) = compile_flux2_minimal(&cfg, &w, 1, 4).unwrap();
    let hidden = vec![0.0f32; cfg.in_channels * 4];
    assert_eq!(
        compiled.run(&[("hidden", hidden.as_slice())])[0].len(),
        4 * cfg.proj_out_dim()
    );
}
