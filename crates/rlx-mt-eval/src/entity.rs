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

//! Glossary / named-fact F1 for dubbing (Helix, 950 HP, radial flux, …).

use serde::{Deserialize, Serialize};

/// Precision / recall over required entity substrings (case-insensitive).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntitySet {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub hit: usize,
    pub required: usize,
    pub missing: Vec<String>,
}

/// Which of `entities` appear as substrings in `hypothesis` (ASCII-lowercased).
pub fn entity_hits(hypothesis: &str, entities: &[&str]) -> (usize, Vec<String>) {
    let hay = hypothesis.to_lowercase();
    let mut missing = Vec::new();
    let mut hit = 0usize;
    for e in entities {
        let needle = e.to_lowercase();
        if needle.is_empty() {
            continue;
        }
        if hay.contains(&needle) {
            hit += 1;
        } else {
            missing.push((*e).to_string());
        }
    }
    (hit, missing)
}

/// Entity F1: for dubbing we treat every listed entity as required in the hyp,
/// so precision == recall == hit/required (no spurious-entity set).
pub fn entity_f1(hypothesis: &str, entities: &[&str]) -> EntitySet {
    let required = entities.iter().filter(|e| !e.is_empty()).count();
    if required == 0 {
        return EntitySet {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
            hit: 0,
            required: 0,
            missing: Vec::new(),
        };
    }
    let (hit, missing) = entity_hits(hypothesis, entities);
    let r = hit as f64 / required as f64;
    EntitySet {
        precision: r,
        recall: r,
        f1: r,
        hit,
        required,
        missing,
    }
}
