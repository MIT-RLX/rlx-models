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

//! Gemma mobile QAT quant codec (`quant_method: "gemma"`).
//!
//! The Gemma 4 E2B *mobile* checkpoint quantizes most weights with a custom
//! symmetric, **per-output-channel** (per-row) scheme at a per-module bit width
//! of 2, 4, or 8 bits — selected by the `module_quant_configs` regex table in
//! `config.json`. There is **no zero-point**; dequant is simply
//! `w = signed(q) * weight_scale[row]`.
//!
//! Storage in safetensors:
//! - 2-bit: `uint8`, 4 values/byte, least-significant field first; each field
//!   `u ∈ [0,3]` maps to signed `u - 2 ∈ [-2, 1]`.
//! - 4-bit: `uint8`, 2 values/byte, low nibble first; `u ∈ [0,15]` → `u - 8`.
//! - 8-bit: `int8`, one value/byte, used directly (already signed).
//!
//! `weight_scale` is `float32` of shape `(out, 1)` — one scale per output row.
//! Embeddings (`embedding_quantized` + `embedding_scale`) use the same codec.
//!
//! These are pure decode helpers. Bit layout / tensor naming are pinned by the
//! tests below and cross-checked against the real checkpoint via
//! `scripts/gemma4_e2b_dump.py --inspect`.

/// Per-module quantization bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaQuantBits {
    Two,
    Four,
    Eight,
}

impl GemmaQuantBits {
    /// Map a `num_bits` value from `module_quant_configs` to a width.
    pub fn from_num_bits(n: u32) -> Option<Self> {
        match n {
            2 => Some(Self::Two),
            4 => Some(Self::Four),
            8 => Some(Self::Eight),
            _ => None,
        }
    }

    /// Number of quantized values packed into one byte.
    pub fn values_per_byte(self) -> usize {
        match self {
            Self::Two => 4,
            Self::Four => 2,
            Self::Eight => 1,
        }
    }

    /// Bit width of one field.
    fn field_bits(self) -> u32 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    /// Symmetric offset subtracted from the unsigned field (`2^(bits-1)`).
    /// 8-bit is stored as native `int8`, so it carries no extra offset.
    fn zero_point(self) -> i32 {
        match self {
            Self::Two => 2,
            Self::Four => 8,
            Self::Eight => 0,
        }
    }

    /// Packed byte count for `n` contiguous quantized values.
    pub fn packed_len(self, n: usize) -> usize {
        n.div_ceil(self.values_per_byte())
    }
}

/// Unpack the leading `n` quantized values from a packed byte run into signed
/// integers (zero-point already removed). For 2/4-bit the bytes are `uint8`
/// with the least-significant field first; for 8-bit each byte is read as
/// native `int8`.
pub fn unpack_row(packed: &[u8], n: usize, bits: GemmaQuantBits) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    match bits {
        GemmaQuantBits::Eight => {
            for &b in packed.iter().take(n) {
                out.push(b as i8 as i32);
            }
        }
        _ => {
            let vpb = bits.values_per_byte();
            let shift = bits.field_bits();
            let mask = (1i32 << shift) - 1;
            let zp = bits.zero_point();
            'outer: for &byte in packed {
                let mut acc = byte as i32;
                for _ in 0..vpb {
                    if out.len() == n {
                        break 'outer;
                    }
                    out.push((acc & mask) - zp);
                    acc >>= shift;
                }
            }
        }
    }
    out
}

/// Dequantize a row-major `(out, inn)` weight matrix that is packed
/// independently per output row (each row occupies `bits.packed_len(inn)`
/// bytes). `scale` holds one `float32` per output row. Returns a dense
/// `out * inn` row-major `f32` buffer in the same `(out, inn)` orientation as
/// the original (HF Linear) weight.
pub fn dequantize_matrix(
    packed: &[u8],
    scale: &[f32],
    out: usize,
    inn: usize,
    bits: GemmaQuantBits,
) -> anyhow::Result<Vec<f32>> {
    if scale.len() != out {
        anyhow::bail!(
            "gemma-qat: weight_scale len {} != out rows {out}",
            scale.len()
        );
    }
    let bytes_per_row = bits.packed_len(inn);
    let expected = bytes_per_row * out;
    if packed.len() != expected {
        anyhow::bail!(
            "gemma-qat: packed len {} != out*bytes_per_row ({out}*{bytes_per_row}={expected}) \
             for {bits:?} inn={inn}",
            packed.len()
        );
    }
    let mut w = vec![0f32; out * inn];
    for o in 0..out {
        let row = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
        let q = unpack_row(row, inn, bits);
        let s = scale[o];
        let dst = &mut w[o * inn..(o + 1) * inn];
        for (d, qv) in dst.iter_mut().zip(q.iter()) {
            *d = *qv as f32 * s;
        }
    }
    Ok(w)
}

/// One ordered rule from `quantization_config.module_quant_configs`.
///
/// HF compiles all patterns into a single alternation and, for a given module
/// name, the **earliest-listed** pattern that matches wins (`replace_with_quant_layers`
/// in `transformers/integrations/gemma_quant.py`). We mirror that "first match
/// in config order wins" precedence here. The pattern grammar used by the Gemma
/// checkpoints is small and fixed, so instead of a full regex engine (whose Rust
/// crate also can't compile the `(?!…)` lookahead the audio rule uses) we map
/// each known pattern string to a purpose-built predicate.
#[derive(Debug, Clone)]
enum QuantPat {
    /// `^lit$` — whole name equals `lit`.
    Exact(String),
    /// `…lit$` — name ends with `lit` (`\.`-unescaped).
    EndsWith(String),
    /// `language_model\.layers\.(\d|1[0-4])\.mlp\.` — layer in `0..=14` + `.mlp.`.
    LayerMlpUpTo14,
    /// `language_model\.layers\.\d+\.mlp\.` — any layer's `.mlp.`.
    AnyLayerMlp,
    /// `language_model\.layers\.\d+\.self_attn\.`.
    AnyLayerSelfAttn,
    /// `audio_tower(?!.*lconv1d\.linear_start)` — in audio tower, NOT a
    /// `lconv1d.linear_start` submodule.
    AudioNotLconvStart,
    /// `audio_tower\.layers\.\d+\.lconv1d\.linear_start\.`.
    AudioLconvStart,
    /// Plain substring (fallback for `vision_tower`, etc.).
    Contains(String),
}

impl QuantPat {
    /// Translate a `module_quant_configs` regex string into a predicate.
    fn parse(pat: &str) -> Self {
        let unesc = |s: &str| s.replace("\\.", ".").replace("\\d", "");
        match pat {
            "^lm_head$" => QuantPat::Exact("lm_head".into()),
            r"language_model\.embed_tokens$" => {
                QuantPat::EndsWith("language_model.embed_tokens".into())
            }
            r"language_model\.embed_tokens_per_layer$" => {
                QuantPat::EndsWith("language_model.embed_tokens_per_layer".into())
            }
            r"language_model\.layers\.(\d|1[0-4])\.mlp\." => QuantPat::LayerMlpUpTo14,
            r"language_model\.layers\.\d+\.mlp\." => QuantPat::AnyLayerMlp,
            r"language_model\.layers\.\d+\.self_attn\." => QuantPat::AnyLayerSelfAttn,
            r"language_model\.layers\.\d+\.per_layer_input_gate$" => {
                QuantPat::EndsWith("per_layer_input_gate".into())
            }
            r"language_model\.layers\.\d+\.per_layer_projection$" => {
                QuantPat::EndsWith("per_layer_projection".into())
            }
            r"audio_tower(?!.*lconv1d\.linear_start)" => QuantPat::AudioNotLconvStart,
            r"audio_tower\.layers\.\d+\.lconv1d\.linear_start\." => QuantPat::AudioLconvStart,
            other if other.starts_with('^') && other.ends_with('$') => {
                QuantPat::Exact(unesc(&other[1..other.len() - 1]))
            }
            other if other.ends_with('$') => QuantPat::EndsWith(unesc(&other[..other.len() - 1])),
            other => QuantPat::Contains(unesc(other)),
        }
    }

    /// Specificity rank — lower wins when several patterns match one name.
    ///
    /// `serde_json` (no `preserve_order`) hands us object keys in BTreeMap
    /// order, not config order, so we can't rely on insertion order for HF's
    /// "earliest pattern wins" rule. Instead we order by specificity, which
    /// reproduces the intended precedence: the `(\d|1[0-4])\.mlp\.` override
    /// (rank 2) beats the catch-all `\d+\.mlp\.` (rank 4) for layers 0–14, and
    /// the `lconv1d.linear_start` audio override beats the broad audio rule.
    fn rank(&self) -> u8 {
        match self {
            QuantPat::Exact(_) => 0,
            QuantPat::EndsWith(_) => 1,
            QuantPat::LayerMlpUpTo14 => 2,
            QuantPat::AudioLconvStart => 2,
            QuantPat::AnyLayerSelfAttn => 3,
            QuantPat::AnyLayerMlp => 4,
            QuantPat::AudioNotLconvStart => 5,
            QuantPat::Contains(_) => 6,
        }
    }

    fn matches(&self, name: &str) -> bool {
        // Layer index immediately after `layers.`, if present.
        let layer_idx = |seg: &str| -> Option<usize> {
            let i = name.find(seg)? + seg.len();
            let rest = &name[i..];
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..end].parse().ok()
        };
        match self {
            QuantPat::Exact(s) => name == s,
            QuantPat::EndsWith(s) => name.ends_with(s),
            QuantPat::Contains(s) => name.contains(s),
            QuantPat::LayerMlpUpTo14 => {
                name.contains(".mlp.")
                    && layer_idx("language_model.layers.").is_some_and(|l| l <= 14)
            }
            QuantPat::AnyLayerMlp => {
                name.contains(".mlp.") && name.contains("language_model.layers.")
            }
            QuantPat::AnyLayerSelfAttn => {
                name.contains(".self_attn.") && name.contains("language_model.layers.")
            }
            QuantPat::AudioNotLconvStart => {
                name.contains("audio_tower") && !name.contains("lconv1d.linear_start")
            }
            QuantPat::AudioLconvStart => {
                name.contains("audio_tower") && name.contains("lconv1d.linear_start")
            }
        }
    }
}

/// Per-module quantization plan parsed from `config.json`'s
/// `quantization_config` block. Resolves the bit width (or "keep fp") for any
/// parameter/module name.
#[derive(Debug, Clone)]
pub struct GemmaQuantPlan {
    default_bits: u32,
    quantize_embeddings: bool,
    not_convert: Vec<String>,
    rules: Vec<(QuantPat, u32)>,
}

impl GemmaQuantPlan {
    /// Build from the parsed `quantization_config` JSON object.
    pub fn from_json(quant_cfg: &serde_json::Value) -> Self {
        let default_bits = quant_cfg
            .get("num_bits")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as u32;
        let quantize_embeddings = quant_cfg
            .get("quantize_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let not_convert = quant_cfg
            .get("modules_to_not_convert")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        // `module_quant_configs` is a JSON object; serde_json preserves insertion
        // order with the `preserve_order` feature. The workspace enables it, so
        // config-order precedence is honored.
        let mut rules: Vec<(QuantPat, u32)> = quant_cfg
            .get("module_quant_configs")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(pat, v)| {
                        v.get("num_bits")
                            .and_then(|b| b.as_u64())
                            .map(|b| (QuantPat::parse(pat), b as u32))
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Most-specific (lowest rank) first so `resolve_bits` can take the first
        // match without depending on JSON key ordering.
        rules.sort_by_key(|(pat, _)| pat.rank());
        Self {
            default_bits,
            quantize_embeddings,
            not_convert,
            rules,
        }
    }

    pub fn quantize_embeddings(&self) -> bool {
        self.quantize_embeddings
    }

    /// Resolve the quant width for a module name. Returns `None` when the module
    /// is in `modules_to_not_convert` (i.e. kept in floating point). The
    /// `modules_to_not_convert` entries are checked as substrings, matching HF's
    /// `should_convert_module`.
    pub fn resolve_bits(&self, name: &str) -> Option<GemmaQuantBits> {
        if self.not_convert.iter().any(|m| name.contains(m.as_str())) {
            return None;
        }
        let bits = self
            .rules
            .iter()
            .find(|(pat, _)| pat.matches(name))
            .map(|(_, b)| *b)
            .unwrap_or(self.default_bits);
        GemmaQuantBits::from_num_bits(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_map_from_num_bits() {
        assert_eq!(GemmaQuantBits::from_num_bits(2), Some(GemmaQuantBits::Two));
        assert_eq!(GemmaQuantBits::from_num_bits(4), Some(GemmaQuantBits::Four));
        assert_eq!(
            GemmaQuantBits::from_num_bits(8),
            Some(GemmaQuantBits::Eight)
        );
        assert_eq!(GemmaQuantBits::from_num_bits(3), None);
    }

    #[test]
    fn unpack_4bit_low_nibble_first_signed() {
        // byte 0x?? = high<<4 | low. low nibble is the FIRST value.
        // 0x10 -> low=0 -> -8 ; high=1 -> 1-8 = -7
        // 0xF8 -> low=8 -> 0  ; high=15-> 7
        let bytes = [0x10u8, 0xF8u8];
        let v = unpack_row(&bytes, 4, GemmaQuantBits::Four);
        assert_eq!(v, vec![-8, -7, 0, 7]);
    }

    #[test]
    fn unpack_2bit_lsb_first_signed() {
        // 0b11_10_01_00 = 0xE4. Fields LSB-first: 0,1,2,3 -> -2,-1,0,1
        let bytes = [0xE4u8];
        let v = unpack_row(&bytes, 4, GemmaQuantBits::Two);
        assert_eq!(v, vec![-2, -1, 0, 1]);
    }

    #[test]
    fn unpack_8bit_is_native_int8() {
        let bytes = [0x00u8, 0x7F, 0x80, 0xFF];
        let v = unpack_row(&bytes, 4, GemmaQuantBits::Eight);
        assert_eq!(v, vec![0, 127, -128, -1]);
    }

    #[test]
    fn unpack_respects_partial_trailing_field() {
        // inn=3 with 2 values/byte -> 2 bytes, last byte's high nibble unused.
        // 0xF8: low=8->0, high=15->7 ; 0x00: low=0->-8 (3rd value)
        let bytes = [0xF8u8, 0x00u8];
        let v = unpack_row(&bytes, 3, GemmaQuantBits::Four);
        assert_eq!(v, vec![0, 7, -8]);
    }

    #[test]
    fn dequantize_applies_per_row_scale() {
        // 2 rows x 4 cols, 4-bit. Row0 scale 0.5, row1 scale 2.0.
        // Row0 bytes: 0x10,0xF8 -> [-8,-7,0,7]
        // Row1 bytes: 0x98,0x21 -> low=8->0,high=9->1 ; low=1->-7,high=2->-6
        let packed = [0x10u8, 0xF8, 0x98, 0x21];
        let scale = [0.5f32, 2.0];
        let w = dequantize_matrix(&packed, &scale, 2, 4, GemmaQuantBits::Four).unwrap();
        assert_eq!(
            w,
            vec![
                -4.0, -3.5, 0.0, 3.5, // row0 * 0.5
                0.0, 2.0, -14.0, -12.0, // row1 * 2.0
            ]
        );
    }

    #[test]
    fn dequantize_rejects_size_mismatch() {
        let packed = [0u8; 2];
        let scale = [1.0f32; 2];
        // out=2, inn=4 needs 2 bytes/row = 4 bytes, only 2 supplied.
        assert!(dequantize_matrix(&packed, &scale, 2, 4, GemmaQuantBits::Four).is_err());
    }

    #[test]
    fn packed_len_rounds_up() {
        assert_eq!(GemmaQuantBits::Two.packed_len(10), 3); // ceil(10/4)
        assert_eq!(GemmaQuantBits::Four.packed_len(7), 4); // ceil(7/2)
        assert_eq!(GemmaQuantBits::Eight.packed_len(5), 5);
    }

    /// The real `quantization_config` from the E2B checkpoint.
    const E2B_QUANT_CFG: &str = r#"{
      "module_quant_configs": {
        "^lm_head$": {"num_bits": 2},
        "audio_tower(?!.*lconv1d\\.linear_start)": {"num_bits": 2},
        "audio_tower\\.layers\\.\\d+\\.lconv1d\\.linear_start\\.": {"num_bits": 4},
        "language_model\\.embed_tokens$": {"num_bits": 2},
        "language_model\\.embed_tokens_per_layer$": {"num_bits": 4},
        "language_model\\.layers\\.(\\d|1[0-4])\\.mlp\\.": {"num_bits": 4},
        "language_model\\.layers\\.\\d+\\.mlp\\.": {"num_bits": 2},
        "language_model\\.layers\\.\\d+\\.per_layer_input_gate$": {"num_bits": 8},
        "language_model\\.layers\\.\\d+\\.per_layer_projection$": {"num_bits": 8},
        "language_model\\.layers\\.\\d+\\.self_attn\\.": {"num_bits": 4},
        "vision_tower": {"num_bits": 8}
      },
      "modules_to_not_convert": [
        "model.vision_tower.patch_embedder",
        "model.audio_tower.subsample_conv_projection",
        "model.audio_tower.output_proj",
        "relative_k_proj",
        "model.embed_audio",
        "model.embed_vision",
        "per_layer_model_projection"
      ],
      "num_bits": 4,
      "quantize_embeddings": true
    }"#;

    fn plan() -> GemmaQuantPlan {
        GemmaQuantPlan::from_json(&serde_json::from_str(E2B_QUANT_CFG).unwrap())
    }

    #[test]
    fn resolve_bits_matches_real_config() {
        let p = plan();
        let b = |n: &str| p.resolve_bits(n);
        use GemmaQuantBits::*;
        // self_attn → 4-bit
        assert_eq!(
            b("model.language_model.layers.0.self_attn.q_proj.weight"),
            Some(Four)
        );
        assert_eq!(
            b("model.language_model.layers.34.self_attn.v_proj.weight"),
            Some(Four)
        );
        // mlp: layers 0–14 → 4-bit, 15+ → 2-bit (the tricky overlap)
        assert_eq!(
            b("model.language_model.layers.0.mlp.gate_proj.weight"),
            Some(Four)
        );
        assert_eq!(
            b("model.language_model.layers.14.mlp.down_proj.weight"),
            Some(Four)
        );
        assert_eq!(
            b("model.language_model.layers.15.mlp.gate_proj.weight"),
            Some(Two)
        );
        assert_eq!(
            b("model.language_model.layers.20.mlp.gate_proj.weight"),
            Some(Two)
        );
        // per-layer gate/projection → 8-bit
        assert_eq!(
            b("model.language_model.layers.3.per_layer_input_gate"),
            Some(Eight)
        );
        assert_eq!(
            b("model.language_model.layers.3.per_layer_projection"),
            Some(Eight)
        );
        // embeddings
        assert_eq!(b("model.language_model.embed_tokens"), Some(Two));
        assert_eq!(b("model.language_model.embed_tokens_per_layer"), Some(Four));
        // lm_head → 2-bit (exact)
        assert_eq!(b("lm_head"), Some(Two));
        // vision tower → 8-bit
        assert_eq!(
            b("model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight"),
            Some(Eight)
        );
        // audio: broad 2-bit, but lconv1d.linear_start → 4-bit
        assert_eq!(
            b("model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight"),
            Some(Two)
        );
        assert_eq!(
            b("model.audio_tower.layers.0.lconv1d.linear_start.weight"),
            Some(Four)
        );
    }

    #[test]
    fn modules_to_not_convert_stay_fp() {
        let p = plan();
        // These keep floating point (None) per modules_to_not_convert.
        assert_eq!(
            p.resolve_bits("model.language_model.per_layer_model_projection.weight"),
            None
        );
        assert_eq!(
            p.resolve_bits("model.vision_tower.patch_embedder.input_proj.weight"),
            None
        );
        assert_eq!(
            p.resolve_bits("model.embed_vision.embedding_projection.weight"),
            None
        );
        assert_eq!(p.resolve_bits("model.audio_tower.output_proj.weight"), None);
        assert!(p.quantize_embeddings());
    }
}
