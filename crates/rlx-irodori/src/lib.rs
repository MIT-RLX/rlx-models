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

//! # rlx-irodori
//!
//! **Irodori-TTS** — a Japanese voice-design TTS on RLX. An LM-style backbone
//! emits neural-codec tokens conditioned on text plus a voice-design embedding
//! (style / timbre), decoded by a neural codec.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **Backbone** → Llama-shaped LM (`rlx-llama32`).
//! - **Codec** → a neural audio codec (`rlx-dac` / `rlx-snac`).
//!
//! Japanese synthesis is timed in **morae**, not characters, so the checkpoint-free
//! core here is a correct **mora frontend** ([`count_morae`]) plus the config. The
//! LM + codec graph wiring is the next step.

use anyhow::{Result, ensure};

/// Irodori model config. Dimensional fields carry plausible values; exact widths
/// come from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct IrodoriConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    // LM backbone.
    pub backbone_hidden: usize,
    pub backbone_layers: usize,
    pub backbone_heads: usize,
    // Neural codec.
    pub num_codebooks: usize,
    pub codebook_size: usize,
    /// Voice-design (style/timbre) conditioning width.
    pub voice_design_dim: usize,
    /// Acoustic tokens emitted per mora (rough duration prior for the frontend).
    pub tokens_per_mora: f32,
}

impl Default for IrodoriConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            hop_length: 480,
            backbone_hidden: 1024,
            backbone_layers: 24,
            backbone_heads: 16,
            num_codebooks: 4,
            codebook_size: 1024,
            voice_design_dim: 256,
            tokens_per_mora: 8.0,
        }
    }
}

impl IrodoriConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_codebooks > 0, "num_codebooks must be > 0");
        ensure!(self.codebook_size > 0, "codebook_size must be > 0");
        ensure!(self.tokens_per_mora > 0.0, "tokens_per_mora must be > 0");
        Ok(())
    }

    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }

    /// A rough number of acoustic tokens to budget for `kana` text, from its mora
    /// count (Japanese duration scales with morae, not characters).
    pub fn tokens_for_kana(&self, kana: &str) -> usize {
        (count_morae(kana) as f32 * self.tokens_per_mora).round() as usize
    }
}

/// Small kana that combine with the preceding kana into a single mora
/// (yōon ゃゅょ and small vowels) — these do **not** add a mora. Note the sokuon
/// っ/ッ is *not* here: it counts as its own mora.
fn is_combining_small(c: char) -> bool {
    matches!(
        c,
        'ぁ' | 'ぃ'
            | 'ぅ'
            | 'ぇ'
            | 'ぉ'
            | 'ゃ'
            | 'ゅ'
            | 'ょ'
            | 'ゎ'
            | 'ゕ'
            | 'ゖ'
            | 'ァ'
            | 'ィ'
            | 'ゥ'
            | 'ェ'
            | 'ォ'
            | 'ャ'
            | 'ュ'
            | 'ョ'
            | 'ヮ'
            | 'ヵ'
            | 'ヶ'
    )
}

/// True for a base hiragana or katakana code point.
fn is_kana(c: char) -> bool {
    let u = c as u32;
    (0x3041..=0x3096).contains(&u) || (0x30A1..=0x30FA).contains(&u)
}

/// Count the morae in a hiragana/katakana string.
///
/// Rules: each base kana is one mora; small yōon/vowel kana (ゃゅょ, ぁぃぅぇぉ, …)
/// attach to the previous mora and count for zero; the sokuon (っ/ッ), the moraic
/// nasal (ん/ン), and the long-vowel mark (ー) each count as one mora;
/// non-kana characters are ignored.
pub fn count_morae(s: &str) -> usize {
    let mut n = 0;
    for c in s.chars() {
        if is_combining_small(c) {
            continue;
        }
        if is_kana(c) || c == 'ー' {
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = IrodoriConfig::default();
        assert_eq!(c.sample_rate, 24_000);
        assert_eq!(c.num_codebooks, 4);
        c.validate().unwrap();
    }

    #[test]
    fn mora_counting_follows_japanese_rules() {
        assert_eq!(count_morae("とうきょう"), 4); // to-u-kyo-u
        assert_eq!(count_morae("がっこう"), 4); // ga-Q-ko-u (sokuon counts)
        assert_eq!(count_morae("きゃ"), 1); // yōon → single mora
        assert_eq!(count_morae("ラーメン"), 4); // ra-ā-me-N (chōonpu + nasal count)
        assert_eq!(count_morae("しんぶん"), 4); // shi-N-bu-N
        assert_eq!(count_morae(""), 0);
    }

    #[test]
    fn non_kana_is_ignored() {
        // Latin / punctuation / spaces contribute no morae.
        assert_eq!(count_morae("ABC 123!"), 0);
        assert_eq!(count_morae("あ、い。"), 2);
    }

    #[test]
    fn token_budget_scales_with_morae() {
        let c = IrodoriConfig::default(); // 8 tokens/mora
        assert_eq!(c.tokens_for_kana("とうきょう"), 32); // 4 morae * 8
    }
}
