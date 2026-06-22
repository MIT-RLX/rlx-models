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

//! DTW word-alignment unit tests (no Python).

use rlx_whisper::alignment_heads::{load_alignment_heads, model_nickname};
use rlx_whisper::config::WhisperConfig;
use rlx_whisper::dtw::{dtw, median_filter_1d};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RefWord {
    word: String,
    start: f32,
    end: f32,
}

const FIXTURE: &str = include_str!("../../rlx-whisper/fixtures/jfk_words_dtw.json");
const COLLAR_SEC: f32 = 0.2;

fn within_collar(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= COLLAR_SEC
}

#[test]
fn alignment_heads_tiny_matches_openai() {
    let cfg = WhisperConfig::tiny();
    let name = model_nickname(&cfg, "whisper-tiny");
    assert_eq!(name, "tiny");
    let heads = load_alignment_heads(&cfg, &name).unwrap();
    assert_eq!(
        heads.pairs,
        vec![(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)]
    );
}

#[test]
fn dtw_toy_cost_path_is_monotonic() {
    let costs = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0];
    let (text_ix, frame_ix) = dtw(&costs, 2, 3);
    assert_eq!(text_ix.len(), frame_ix.len());
    for w in frame_ix.windows(2) {
        assert!(w[1] >= w[0]);
    }
}

#[test]
fn median_filter_reduces_spike() {
    let x = vec![0.0, 0.0, 10.0, 0.0, 0.0];
    let y = median_filter_1d(&x, 3);
    assert!(y[2] < 10.0);
}

#[test]
fn jfk_word_fixture_monotonic_and_collar_ready() {
    let words: Vec<RefWord> = serde_json::from_str(FIXTURE).unwrap();
    assert!(words.len() >= 2);
    for w in words.windows(2) {
        assert!(w[1].start >= w[0].start);
        assert!(w[0].end >= w[0].start);
    }
    // Reference bounds for parity checks (200 ms collar, WhisperX standard).
    let expected_starts = [0.0f32, 0.32, 0.64, 0.88, 1.28];
    for (w, &exp) in words.iter().zip(expected_starts.iter()) {
        assert!(
            within_collar(w.start, exp),
            "{} start {} vs {}",
            w.word,
            w.start,
            exp
        );
    }
}
