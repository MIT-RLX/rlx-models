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

//! Parity vs the upstream aligner's dynamic program.
//!
//! `tests/fixtures/align_reference.json` holds the exact `token_positions`
//! that `tada.modules.aligner._align_text_tokens` (torch 2.11) returns for six
//! random `[frames, vocab]` logit tensors. The assignment is discrete, so
//! parity here is exact equality — no tolerance. It is worth pinning because
//! the recursion has two non-obvious details that a "reasonable" rewrite gets
//! wrong: the diagonal is seeded before the relaxation loop and never
//! overwritten, and ties (`use >= skip`) resolve toward consuming the frame.

use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    frames: usize,
    vocab: usize,
    logits: Vec<f32>,
    tokens: Vec<u32>,
    positions: Vec<u32>,
}

#[test]
fn alignment_matches_the_upstream_dynamic_program() {
    let raw = include_str!("fixtures/align_reference.json");
    let fixture: Fixture = serde_json::from_str(raw).expect("parse align_reference.json");
    assert!(!fixture.cases.is_empty());

    for (i, case) in fixture.cases.iter().enumerate() {
        let got = rlx_tada::align_tokens(&case.logits, case.vocab, &case.tokens, case.frames)
            .unwrap_or_else(|e| panic!("case {i}: {e}"));
        // Upstream returns 0-based frames; we store 1-based so 0 can pad.
        let got0: Vec<u32> = got.token_positions.iter().map(|p| p - 1).collect();
        assert_eq!(
            got0,
            case.positions,
            "case {i} ({} frames, vocab {}, {} tokens)",
            case.frames,
            case.vocab,
            case.tokens.len()
        );

        // The mask must mark exactly the assigned frames.
        let marked: Vec<u32> = got
            .token_mask
            .iter()
            .enumerate()
            .filter(|&(_, &m)| m == 1)
            .map(|(i, _)| i as u32)
            .collect();
        let mut expect = case.positions.clone();
        expect.sort_unstable();
        expect.dedup();
        assert_eq!(
            marked, expect,
            "case {i}: token_mask disagrees with positions"
        );
    }
}
