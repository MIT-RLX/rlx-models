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

//! The tapped Qwen3-VL stack on a scaled-down configuration.
//!
//! These check structure — causality, GQA head grouping, the tap position and
//! the conditioning contract — not numerical agreement with the reference,
//! which needs the ~60 GB encoder this port did not fetch.

use rlx_minimax_h3::config::H3TextEncoderConfig;
use rlx_minimax_h3::qwen3vl::{compile_text_encoder, layers_to_run, synthetic_weights};
use rlx_runtime::Device;

/// The released stack depth — the tap has to actually reach layer 50 — with
/// tiny widths so it compiles in a test.
fn tiny() -> H3TextEncoderConfig {
    H3TextEncoderConfig {
        hidden_size: 32,
        num_hidden_layers: 64,
        num_attention_heads: 8,
        num_key_value_heads: 1,
        head_dim: 8,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 5e6,
        vocab_size: 64,
        mrope_section: [2, 1, 1],
        mrope_interleaved: true,
    }
}

fn build(
    seq: usize,
) -> (
    rlx_minimax_h3::qwen3vl::H3Qwen3VlEncoder,
    H3TextEncoderConfig,
) {
    let cfg = tiny();
    let mut w = synthetic_weights(&cfg, 13);
    let enc = compile_text_encoder(&cfg, &mut w, Device::Cpu, seq).expect("compile text encoder");
    (enc, cfg)
}

#[test]
fn the_tap_reaches_layer_fifty_and_stops() {
    let mut c = tiny();
    assert_eq!(layers_to_run(&c), 50);
    assert!(c.validate().is_ok());
    // A stack that cannot reach the tap is a configuration error.
    c.num_hidden_layers = 20;
    assert!(
        c.validate().is_err(),
        "a 20-layer stack cannot supply a layer-50 tap"
    );
}

#[test]
fn produces_conditioning_of_the_right_shape() {
    let seq = 7;
    let (mut enc, cfg) = build(seq);
    let ids: Vec<u32> = (0..seq as u32)
        .map(|i| (i * 5) % cfg.vocab_size as u32)
        .collect();
    let c = enc.encode_tokens(&ids).expect("encode");

    assert_eq!(c.num_tokens(), seq);
    assert_eq!(c.hidden_size, cfg.hidden_size);
    assert_eq!(c.hidden.len(), seq * cfg.hidden_size);
    c.validate().expect("conditioning must be well-formed");
    c.check_against(cfg.hidden_size).expect("width must match");
    assert!(c.hidden.iter().all(|v| v.is_finite()));
    assert!(
        c.hidden.iter().any(|v| v.abs() > 1e-6),
        "the tap collapsed to zero"
    );
}

#[test]
fn every_row_is_tagged_as_text() {
    let (mut enc, _) = build(4);
    let c = enc.encode_tokens(&[1, 2, 3, 4]).unwrap();
    assert!(
        c.token_tags
            .iter()
            .all(|&t| t == rlx_minimax_h3::config::Modality::Text.tag()),
        "a text-only prompt tags every row as text"
    );
}

#[test]
fn attention_is_causal() {
    // Changing a *later* token must not move an earlier row. A missing causal
    // mask is invisible in shapes and would silently condition on the future.
    let seq = 6;
    let (mut enc, cfg) = build(seq);
    let base: Vec<u32> = vec![3, 9, 14, 20, 25, 31];
    let mut changed = base.clone();
    changed[5] = 40;

    let a = enc.encode_tokens(&base).unwrap();
    let b = enc.encode_tokens(&changed).unwrap();
    let h = cfg.hidden_size;

    for row in 0..5 {
        let d: f32 = a.hidden[row * h..(row + 1) * h]
            .iter()
            .zip(&b.hidden[row * h..(row + 1) * h])
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(
            d < 1e-5,
            "row {row} moved when a later token changed (delta {d})"
        );
    }
    // The last row *must* move.
    let d: f32 = a.hidden[5 * h..]
        .iter()
        .zip(&b.hidden[5 * h..])
        .map(|(x, y)| (x - y).abs())
        .sum();
    assert!(d > 1e-5, "the changed token did not affect its own row");
}

#[test]
fn earlier_tokens_reach_later_rows() {
    // The converse of causality: changing the first token must move the last
    // row, or attention is not mixing at all.
    let seq = 5;
    let (mut enc, cfg) = build(seq);
    let base: Vec<u32> = vec![2, 7, 11, 19, 23];
    let mut changed = base.clone();
    changed[0] = 41;

    let a = enc.encode_tokens(&base).unwrap();
    let b = enc.encode_tokens(&changed).unwrap();
    let h = cfg.hidden_size;
    let d: f32 = a.hidden[(seq - 1) * h..]
        .iter()
        .zip(&b.hidden[(seq - 1) * h..])
        .map(|(x, y)| (x - y).abs())
        .sum();
    assert!(d > 1e-5, "the first token never reached the last row");
}

#[test]
fn encoding_is_deterministic() {
    let (mut enc, _) = build(4);
    let ids = [5u32, 11, 17, 23];
    let a = enc.encode_tokens(&ids).unwrap();
    let b = enc.encode_tokens(&ids).unwrap();
    assert_eq!(a.hidden, b.hidden);
}

#[test]
fn rejects_a_wrong_length_or_out_of_range_prompt() {
    let (mut enc, cfg) = build(4);
    assert!(
        enc.encode_tokens(&[1, 2, 3]).is_err(),
        "a length mismatch must be caught"
    );
    let bad = vec![1u32, 2, 3, cfg.vocab_size as u32];
    assert!(
        enc.encode_tokens(&bad).is_err(),
        "an out-of-vocabulary id must be caught"
    );
}

#[test]
fn gqa_grouping_runs_with_several_kv_heads() {
    // Exercise the narrow+concat widening with a real group factor rather than
    // the degenerate single-KV-head case.
    let mut cfg = tiny();
    cfg.num_attention_heads = 8;
    cfg.num_key_value_heads = 4; // group of 2
    let mut w = synthetic_weights(&cfg, 17);
    let mut enc = compile_text_encoder(&cfg, &mut w, Device::Cpu, 5).expect("compile");
    let c = enc.encode_tokens(&[1, 2, 3, 4, 5]).expect("encode");
    assert_eq!(c.num_tokens(), 5);
    assert!(c.hidden.iter().all(|v| v.is_finite()));
}

#[test]
fn conditioning_drives_the_packed_layout() {
    // The whole point of the tap: its rows become the text rows of the packed
    // sequence.
    use rlx_minimax_h3::layout::{H3Geometry, build_packed_sequence};
    let seq = 6;
    let (mut enc, _) = build(seq);
    let c = enc.encode_tokens(&[1, 2, 3, 4, 5, 6]).unwrap();
    let g = H3Geometry::resolve(768, 1344, 124, 16, 2).unwrap();
    let layout = build_packed_sequence(&c.token_tags, &g, [1, 2, 2], &[]).unwrap();
    assert_eq!(layout.text_indices.len(), seq);
    layout.validate().unwrap();
}
