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

//! Cross-attention alignment-head presets (OpenAI Whisper).

use crate::config::WhisperConfig;
use anyhow::Result;

/// `(decoder_layer, head_index)` pairs used for word-level DTW alignment.
#[derive(Debug, Clone)]
pub struct AlignmentHeads {
    pub pairs: Vec<(usize, usize)>,
}

impl AlignmentHeads {
    pub fn indices(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.pairs.iter().copied()
    }
}

/// Known model nicknames for preset lookup.
pub fn model_nickname(cfg: &WhisperConfig, weights_path: &str) -> String {
    let path = weights_path.to_ascii_lowercase();
    for name in [
        "large-v3-turbo",
        "large-v3",
        "large-v2",
        "large-v1",
        "medium.en",
        "medium",
        "small.en",
        "small",
        "base.en",
        "base",
        "tiny.en",
        "tiny",
        "turbo",
        "large",
    ] {
        if path.contains(name) {
            return name.to_string();
        }
    }
    if cfg.decoder_layers <= 4 && cfg.decoder_attention_heads == 20 {
        "turbo".into()
    } else if cfg.decoder_layers >= 32 {
        "large-v3".into()
    } else if cfg.decoder_layers >= 24 {
        "medium".into()
    } else if cfg.decoder_layers >= 12 {
        "small".into()
    } else if cfg.decoder_layers >= 6 {
        "base".into()
    } else {
        "tiny".into()
    }
}

/// Decoded from OpenAI Whisper `_ALIGNMENT_HEADS` (whisper/__init__.py).
const PRESET_HEADS: &[(&str, &[(usize, usize)])] = &[
    (
        "tiny.en",
        &[
            (1, 0),
            (2, 0),
            (2, 5),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
        ],
    ),
    ("tiny", &[(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)]),
    ("base.en", &[(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)]),
    (
        "base",
        &[
            (3, 1),
            (4, 2),
            (4, 3),
            (4, 7),
            (5, 1),
            (5, 2),
            (5, 4),
            (5, 6),
        ],
    ),
    (
        "small.en",
        &[
            (6, 6),
            (7, 0),
            (7, 3),
            (7, 8),
            (8, 2),
            (8, 5),
            (8, 7),
            (9, 0),
            (9, 4),
            (9, 8),
            (9, 10),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 6),
            (10, 11),
            (11, 2),
            (11, 4),
        ],
    ),
    (
        "small",
        &[
            (5, 3),
            (5, 9),
            (8, 0),
            (8, 4),
            (8, 7),
            (8, 8),
            (9, 0),
            (9, 7),
            (9, 9),
            (10, 5),
        ],
    ),
    (
        "medium.en",
        &[
            (11, 4),
            (14, 1),
            (14, 12),
            (14, 14),
            (15, 4),
            (16, 0),
            (16, 4),
            (16, 9),
            (17, 12),
            (17, 14),
            (18, 7),
            (18, 10),
            (18, 15),
            (20, 0),
            (20, 3),
            (20, 9),
            (20, 14),
            (21, 12),
        ],
    ),
    (
        "medium",
        &[(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
    ),
    (
        "large-v1",
        &[
            (9, 19),
            (11, 2),
            (11, 4),
            (11, 17),
            (22, 7),
            (22, 11),
            (22, 17),
            (23, 2),
            (23, 15),
        ],
    ),
    (
        "large-v2",
        &[
            (10, 12),
            (13, 17),
            (16, 11),
            (16, 12),
            (16, 13),
            (17, 15),
            (17, 16),
            (18, 4),
            (18, 11),
            (18, 19),
            (19, 11),
            (21, 2),
            (21, 3),
            (22, 3),
            (22, 9),
            (22, 12),
            (23, 5),
            (23, 7),
            (23, 13),
            (25, 5),
            (26, 1),
            (26, 12),
            (27, 15),
        ],
    ),
    (
        "large-v3",
        &[
            (7, 0),
            (10, 17),
            (12, 18),
            (13, 12),
            (16, 1),
            (17, 14),
            (19, 11),
            (21, 4),
            (24, 1),
            (25, 6),
        ],
    ),
    (
        "large",
        &[
            (7, 0),
            (10, 17),
            (12, 18),
            (13, 12),
            (16, 1),
            (17, 14),
            (19, 11),
            (21, 4),
            (24, 1),
            (25, 6),
        ],
    ),
    (
        "large-v3-turbo",
        &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
    ),
    (
        "turbo",
        &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
    ),
];

pub fn load_alignment_heads(_cfg: &WhisperConfig, model_name: &str) -> Result<AlignmentHeads> {
    let pairs = PRESET_HEADS
        .iter()
        .find(|(n, _)| *n == model_name)
        .map(|(_, p)| p.to_vec())
        .ok_or_else(|| anyhow::anyhow!("no alignment preset for {model_name}"))?;
    Ok(AlignmentHeads { pairs })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_preset_loads() {
        let cfg = WhisperConfig::tiny();
        let heads = load_alignment_heads(&cfg, "tiny").unwrap();
        assert_eq!(
            heads.pairs,
            vec![(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)]
        );
    }

    #[test]
    fn turbo_preset_loads() {
        let mut cfg = WhisperConfig::tiny();
        cfg.decoder_layers = 4;
        cfg.decoder_attention_heads = 20;
        let heads = load_alignment_heads(&cfg, "turbo").unwrap();
        assert_eq!(heads.pairs.len(), 6);
    }
}
