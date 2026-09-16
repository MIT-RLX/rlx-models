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

//! Dubbing timing fit from translator-core cue placement fields.

use serde::{Deserialize, Serialize};

/// Per-cue placement vs planned ASR/slot window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CueTiming {
    pub cue_id: usize,
    pub window_sec: f64,
    pub placed_sec: Option<f64>,
    /// placed / window (1.0 = exact fit).
    pub fill_ratio: Option<f64>,
    /// max(0, placed - window) / window.
    pub overrun_ratio: Option<f64>,
}

/// Aggregate timing across cues that have placement data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingScore {
    pub n_with_placement: usize,
    pub mean_fill_ratio: f64,
    pub mean_overrun_ratio: f64,
    pub max_overrun_ratio: f64,
    pub cues: Vec<CueTiming>,
}

/// Score cues that expose `start_sec` / `end_sec` / optional `placed_duration_sec`.
pub fn score_timing(cues: &[CueTimingInput]) -> TimingScore {
    let mut out = Vec::new();
    let mut sum_fill = 0.0;
    let mut sum_over = 0.0;
    let mut max_over: f64 = 0.0;
    let mut n = 0usize;
    for (i, c) in cues.iter().enumerate() {
        let window = (c.end_sec - c.start_sec).max(0.0);
        let placed = c.placed_duration_sec;
        let (fill, over) = if window > 1e-6 {
            if let Some(p) = placed {
                let fill = p / window;
                let over = ((p - window).max(0.0)) / window;
                (Some(fill), Some(over))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        if let (Some(f), Some(o)) = (fill, over) {
            sum_fill += f;
            sum_over += o;
            max_over = max_over.max(o);
            n += 1;
        }
        out.push(CueTiming {
            cue_id: c.id.unwrap_or(i),
            window_sec: window,
            placed_sec: placed,
            fill_ratio: fill,
            overrun_ratio: over,
        });
    }
    TimingScore {
        n_with_placement: n,
        mean_fill_ratio: if n == 0 { 0.0 } else { sum_fill / n as f64 },
        mean_overrun_ratio: if n == 0 { 0.0 } else { sum_over / n as f64 },
        max_overrun_ratio: max_over,
        cues: out,
    }
}

/// Minimal cue fields needed for timing (from `result.json`).
#[derive(Debug, Clone)]
pub struct CueTimingInput {
    pub id: Option<usize>,
    pub start_sec: f64,
    pub end_sec: f64,
    pub placed_duration_sec: Option<f64>,
}
