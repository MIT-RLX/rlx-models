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

//! Interleaved M-RoPE cos/sin tables.
//!
//! Qwen3.8-Flash-Next rotates only `rope_dim_count = 64` of its 256-wide
//! attention head (`partial_rotary_factor = 0.25`) and splits those 64 dims
//! across M-RoPE sections `[11, 11, 10, 0]` in the *interleaved* layout
//! Qwen3.5 introduced. For text-only inference the reference expands a scalar
//! position `p` into `[p, p, p, 0]`, exactly as `llm_graph_input_pos::set_input`
//! does in llama.cpp.
//!
//! One table serves both the attention heads (`head_dim = 256`) and the QSA
//! indexer (`indexer_head_dim = 128`): `Op::Rope` strides rows by the table's
//! own last dimension and reads the first `n_rot / 2 = 32` columns of each, so
//! a table built at `head_dim / 2` is correct for both widths. (It did not used
//! to be — a kernel that assumed `row_stride == n_rot / 2` read the wrong row
//! for every position past the first under partial rotation.)
//!
//! The frequency math itself lives upstream in `rlx_flow::rope`.

use crate::config::Qwen4ExpConfig;

/// Normalise `rope_dim_sections` to exactly four entries — llama.cpp always
/// passes four ints to `ggml_rope_multi`.
fn sections4(cfg: &Qwen4ExpConfig) -> [usize; 4] {
    if cfg.rope_dim_sections.is_empty() {
        [cfg.rope_dim_count / 2, 0, 0, 0]
    } else {
        rlx_flow::rope::mrope_sections4(&cfg.rope_dim_sections)
    }
}

/// Text-modality section positions: `[p, p, p, 0]`.
pub fn text_section_pos(pos: usize) -> [usize; 4] {
    [pos, pos, pos, 0]
}

/// Width a cos/sin table row must have — `head_dim / 2`, shared by the
/// attention and indexer rotations.
pub fn table_width(cfg: &Qwen4ExpConfig) -> usize {
    cfg.head_dim / 2
}

/// One cos/sin row from explicit per-section positions.
pub fn mrope_row(
    cfg: &Qwen4ExpConfig,
    section_pos: [usize; 4],
    width: usize,
) -> (Vec<f32>, Vec<f32>) {
    rlx_flow::rope::mrope_row_for_sections_ex(
        cfg.rope_theta as f64,
        cfg.rope_dim_count,
        sections4(cfg),
        section_pos,
        width,
        cfg.mrope_interleaved,
    )
}

/// Flat `[rows · width]` cos/sin tables for a prefill.
///
/// `section_positions` overrides the text default per row, which is how a
/// multimodal batch feeds its 3-D positions; `None` uses
/// [`text_section_pos`].
pub fn mrope_tables(
    cfg: &Qwen4ExpConfig,
    rows: usize,
    section_positions: Option<&[[usize; 4]]>,
) -> (Vec<f32>, Vec<f32>) {
    let width = table_width(cfg);
    let mut cos = vec![0f32; rows * width];
    let mut sin = vec![0f32; rows * width];
    for t in 0..rows {
        let sec = section_positions
            .and_then(|r| r.get(t).copied())
            .unwrap_or_else(|| text_section_pos(t));
        let (c, s) = mrope_row(cfg, sec, width);
        let off = t * width;
        cos[off..off + width].copy_from_slice(&c);
        sin[off..off + width].copy_from_slice(&s);
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Qwen4ExpConfig {
        Qwen4ExpConfig::qwen38_flash_next()
    }

    #[test]
    fn position_zero_is_the_identity_rotation() {
        let c = cfg();
        let (cos, sin) = mrope_tables(&c, 2, None);
        let w = table_width(&c);
        assert_eq!(cos.len(), 2 * w);
        for i in 0..c.rope_dim_count / 2 {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[0][{i}]");
            assert!(sin[i].abs() < 1e-6, "sin[0][{i}]");
        }
    }

    /// Both rotations read the same table, so its row stride must be
    /// `head_dim / 2` and cover at least `n_rot / 2` live columns.
    #[test]
    fn one_table_serves_the_attention_and_the_indexer() {
        let c = cfg();
        let w = table_width(&c);
        assert_eq!(w, 128);
        assert!(w >= c.rope_dim_count / 2);
        assert!(w >= c.indexer_head_dim / 2);
    }

    #[test]
    fn multimodal_positions_differ_from_text() {
        let c = cfg();
        let w = table_width(&c);
        let text = mrope_row(&c, text_section_pos(2), w);
        let mm = mrope_row(&c, [2, 2, 5, 0], w);
        assert_ne!(text.0, mm.0);
    }
}
