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

//! Embedded Earshot predictor weights (f32 blob from pykeio/earshot).

use std::sync::OnceLock;

static WEIGHTS: OnceLock<Vec<f32>> = OnceLock::new();

fn all() -> &'static [f32] {
    WEIGHTS.get_or_init(|| {
        let bytes = include_bytes!("../../weights/earshot_weights.bin");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    })
}

fn slice(offset: usize, len: usize) -> &'static [f32] {
    &all()[offset..offset + len]
}

pub struct ParsedWeights {
    pub norm_weight: &'static [f32],
    pub layer1_kernel: &'static [f32],
    pub layer1_weight: &'static [f32],
    pub layer1_bias: &'static [f32],
    pub layer2_kernel: &'static [f32],
    pub layer2_weight: &'static [f32],
    pub layer2_bias: &'static [f32],
    pub layer3_kernel: &'static [f32],
    pub layer3_weight: &'static [f32],
    pub layer3_bias: &'static [f32],
    pub rnn1_weight: &'static [f32],
    pub rnn2_weight: &'static [f32],
    pub output_weight: &'static [f32],
}

static PARSED: OnceLock<ParsedWeights> = OnceLock::new();

pub fn weights() -> &'static ParsedWeights {
    PARSED.get_or_init(|| ParsedWeights {
        norm_weight: slice(0, 40),
        layer1_kernel: slice(40, 9),
        layer1_weight: slice(49, 16),
        layer1_bias: slice(65, 16),
        layer2_kernel: slice(81, 48),
        layer2_weight: slice(129, 256),
        layer2_bias: slice(385, 16),
        layer3_kernel: slice(401, 48),
        layer3_weight: slice(449, 256),
        layer3_bias: slice(705, 16),
        rnn1_weight: slice(721, 10240),
        rnn2_weight: slice(10961, 8192),
        output_weight: slice(19153, 128),
    })
}
