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

//! Text (and vision) conditioning for MiniMax-H3.
//!
//! The conditioner is a full `Qwen3VLForConditionalGeneration` — 64 decoder
//! layers, hidden 5120 — but H3 does **not** read its final hidden state. It
//! reads the *unnormalized* hidden state after decoder layer
//! [`H3TextEncoderConfig::TAP_LAYER`] (50). The last layer is post-norm and is
//! not the conditioning the released weights were trained against, so tapping
//! the wrong place is a silent quality regression rather than a crash.
//!
//! Two facts make this module mostly a contract rather than a model:
//!
//! 1. **Per-row modality tags.** Text rows are tagged `1`, *except* the rows
//!    belonging to a keyframe's or reference's vision block, which H3 tags `0`
//!    (video). [`H3TextConditioning::token_tags`] carries that, and
//!    [`crate::layout`] copies it straight into the packed sequence.
//! 2. **The tap is the only thing H3 wants.** Nothing downstream needs logits,
//!    a KV cache, or sampling — just `[num_text_tokens, 5120]`.
//!
//! ## Status
//!
//! The conditioning **contract** and the prompt assembly are implemented, and
//! [`H3TextConditioning::from_safetensors`] loads a tap dumped from any runtime,
//! which is enough to drive the rest of the pipeline. Running the 64-layer
//! Qwen3-VL stack natively is not wired here: the released encoder is ~60 GB
//! across 14 shards, it was not fetched for this port, and a layer-50 tap that
//! cannot be checked against the reference would be untested code on the
//! critical path. [`encode_with`] is the seam a native or external encoder
//! plugs into.

use crate::config::{H3TextEncoderConfig, Modality};
use anyhow::{Context, Result, bail, ensure};
use std::path::Path;

/// Conditioning handed to the DiT: one row per text token.
#[derive(Debug, Clone, PartialEq)]
pub struct H3TextConditioning {
    /// `[num_tokens * hidden_size]`, row-major.
    pub hidden: Vec<f32>,
    /// Width of one row — `text_dim` on the DiT side, 5120 for the release.
    pub hidden_size: usize,
    /// Per-row modality tag: `1` for text, `0` for a vision block's rows.
    pub token_tags: Vec<u32>,
}

impl H3TextConditioning {
    /// Build conditioning where every row is plain text.
    pub fn text_only(hidden: Vec<f32>, hidden_size: usize) -> Result<Self> {
        ensure!(hidden_size > 0, "hidden_size must be positive");
        ensure!(
            hidden.len().is_multiple_of(hidden_size),
            "hidden of {} values does not divide into rows of {hidden_size}",
            hidden.len()
        );
        let rows = hidden.len() / hidden_size;
        Ok(Self {
            hidden,
            hidden_size,
            token_tags: vec![Modality::Text.tag(); rows],
        })
    }

    /// Build conditioning with explicit per-row tags.
    pub fn new(hidden: Vec<f32>, hidden_size: usize, token_tags: Vec<u32>) -> Result<Self> {
        let c = Self {
            hidden,
            hidden_size,
            token_tags,
        };
        c.validate()?;
        Ok(c)
    }

    #[must_use]
    pub fn num_tokens(&self) -> usize {
        self.token_tags.len()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.hidden_size > 0, "hidden_size must be positive");
        ensure!(
            self.hidden.len() == self.token_tags.len() * self.hidden_size,
            "conditioning holds {} values for {} rows of {}",
            self.hidden.len(),
            self.token_tags.len(),
            self.hidden_size
        );
        ensure!(
            !self.token_tags.is_empty(),
            "conditioning must not be empty"
        );
        for (i, &t) in self.token_tags.iter().enumerate() {
            if t != Modality::Text.tag() && t != Modality::Video.tag() {
                bail!(
                    "text row {i} is tagged {t}; only text ({}) and video ({}) tags are valid in the text stream",
                    Modality::Text.tag(),
                    Modality::Video.tag()
                );
            }
        }
        ensure!(
            self.hidden.iter().all(|v| v.is_finite()),
            "conditioning contains non-finite values"
        );
        Ok(())
    }

    /// Check the conditioning against what the DiT expects.
    pub fn check_against(&self, text_dim: usize) -> Result<()> {
        self.validate()?;
        ensure!(
            self.hidden_size == text_dim,
            "conditioning is {} wide, the transformer's `text_dim` is {text_dim}",
            self.hidden_size
        );
        Ok(())
    }

    /// Load a tap dumped to `safetensors`.
    ///
    /// Reads `hidden_states` (`[num_tokens, hidden]`, or `[1, num_tokens, hidden]`)
    /// and, when present, an integer `token_tags` of `[num_tokens]`. This is the
    /// escape hatch for driving H3 from a tap produced by any runtime.
    pub fn from_safetensors(path: &Path) -> Result<Self> {
        use safetensors::SafeTensors;
        let bytes = std::fs::read(path)
            .with_context(|| format!("read conditioning tensor {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse conditioning tensor {}", path.display()))?;

        let t = st
            .tensor("hidden_states")
            .context("conditioning file has no `hidden_states` tensor")?;
        let shape = t.shape();
        let (rows, hidden_size) = match shape {
            [r, h] => (*r, *h),
            [1, r, h] => (*r, *h),
            other => bail!(
                "`hidden_states` must be [tokens, hidden] or [1, tokens, hidden], got {other:?}"
            ),
        };
        let hidden = decode_f32(t.dtype(), t.data(), rows * hidden_size)
            .context("decode `hidden_states`")?;

        let token_tags = match st.tensor("token_tags") {
            Ok(tags) => {
                let n: usize = tags.shape().iter().product();
                ensure!(
                    n == rows,
                    "`token_tags` has {n} entries for {rows} rows of `hidden_states`"
                );
                decode_u32(tags.dtype(), tags.data(), n).context("decode `token_tags`")?
            }
            Err(_) => vec![Modality::Text.tag(); rows],
        };
        Self::new(hidden, hidden_size, token_tags)
    }
}

fn decode_f32(dtype: safetensors::Dtype, data: &[u8], n: usize) -> Result<Vec<f32>> {
    use safetensors::Dtype;
    let out: Vec<f32> = match dtype {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => bail!("unsupported conditioning dtype {other:?}; use f32, f16 or bf16"),
    };
    ensure!(out.len() == n, "decoded {} values, expected {n}", out.len());
    Ok(out)
}

fn decode_u32(dtype: safetensors::Dtype, data: &[u8], n: usize) -> Result<Vec<u32>> {
    use safetensors::Dtype;
    let out: Vec<u32> = match dtype {
        Dtype::I64 | Dtype::U64 => data
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")) as u32)
            .collect(),
        Dtype::I32 | Dtype::U32 => data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::U8 | Dtype::I8 => data.iter().map(|&b| b as u32).collect(),
        other => bail!("unsupported `token_tags` dtype {other:?}"),
    };
    ensure!(out.len() == n, "decoded {} tags, expected {n}", out.len());
    Ok(out)
}

/// Produce conditioning from any encoder.
///
/// `encode` receives the assembled prompt and must return
/// `[num_tokens, hidden_size]` — the **unnormalized** hidden state after
/// decoder layer [`H3TextEncoderConfig::TAP_LAYER`].
pub fn encode_with<F>(
    cfg: &H3TextEncoderConfig,
    prompt: &str,
    mut encode: F,
) -> Result<H3TextConditioning>
where
    F: FnMut(&str) -> Result<(Vec<f32>, usize)>,
{
    cfg.validate()?;
    let assembled = assemble_prompt(prompt);
    let (hidden, hidden_size) = encode(&assembled)?;
    ensure!(
        hidden_size == cfg.hidden_size,
        "encoder produced rows of {hidden_size}, the text config declares {}",
        cfg.hidden_size
    );
    H3TextConditioning::text_only(hidden, hidden_size)
}

/// Wrap a raw prompt in the Qwen3-VL chat turn the conditioner expects.
///
/// H3 conditions on the *assistant-facing* turn, so the prompt is placed in a
/// user message and the assistant header is opened but not filled.
#[must_use]
pub fn assemble_prompt(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt.trim()
    )
}

/// Placeholder rows for wiring and shape checks.
///
/// Deterministic, small, and finite — enough to exercise the packed layout and
/// the DiT without a 60 GB encoder, and obviously not a substitute for one.
#[must_use]
pub fn placeholder_conditioning(num_tokens: usize, hidden_size: usize) -> H3TextConditioning {
    let hidden = (0..num_tokens * hidden_size)
        .map(|i| {
            let t = (i % 97) as f32 / 97.0;
            (t - 0.5) * 0.02
        })
        .collect();
    H3TextConditioning {
        hidden,
        hidden_size,
        token_tags: vec![Modality::Text.tag(); num_tokens],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> H3TextEncoderConfig {
        H3TextEncoderConfig {
            hidden_size: 8,
            num_hidden_layers: 64,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 25_600,
            rms_norm_eps: 1e-6,
            rope_theta: 5e6,
            vocab_size: 151_936,
            mrope_section: [24, 20, 20],
            mrope_interleaved: true,
        }
    }

    #[test]
    fn text_only_tags_every_row_as_text() {
        let c = H3TextConditioning::text_only(vec![0.0; 12], 4).unwrap();
        assert_eq!(c.num_tokens(), 3);
        assert!(c.token_tags.iter().all(|&t| t == Modality::Text.tag()));
        c.validate().unwrap();
    }

    #[test]
    fn vision_rows_may_carry_the_video_tag() {
        let c = H3TextConditioning::new(
            vec![0.0; 12],
            4,
            vec![
                Modality::Text.tag(),
                Modality::Video.tag(),
                Modality::Text.tag(),
            ],
        )
        .unwrap();
        assert_eq!(c.token_tags[1], Modality::Video.tag());
    }

    #[test]
    fn audio_tag_is_rejected_in_the_text_stream() {
        let err = H3TextConditioning::new(vec![0.0; 8], 4, vec![Modality::Audio.tag(); 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("tagged"), "unexpected error: {err}");
    }

    #[test]
    fn ragged_conditioning_is_rejected() {
        assert!(H3TextConditioning::text_only(vec![0.0; 7], 4).is_err());
        assert!(H3TextConditioning::new(vec![0.0; 8], 4, vec![1; 3]).is_err());
    }

    #[test]
    fn non_finite_conditioning_is_rejected() {
        let c = H3TextConditioning {
            hidden: vec![0.0, f32::NAN, 0.0, 0.0],
            hidden_size: 4,
            token_tags: vec![Modality::Text.tag()],
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn width_must_match_the_transformer_text_dim() {
        let c = H3TextConditioning::text_only(vec![0.0; 12], 4).unwrap();
        assert!(c.check_against(4).is_ok());
        let err = c.check_against(5120).unwrap_err().to_string();
        assert!(err.contains("text_dim"), "unexpected error: {err}");
    }

    #[test]
    fn encode_with_checks_the_declared_width() {
        let c = cfg();
        let ok = encode_with(&c, "a cat", |p| {
            assert!(p.contains("a cat"));
            Ok((vec![0.0; 3 * 8], 8))
        })
        .unwrap();
        assert_eq!(ok.num_tokens(), 3);

        let bad = encode_with(&c, "a cat", |_| Ok((vec![0.0; 3 * 4], 4)));
        assert!(bad.is_err(), "a width mismatch must be caught");
    }

    #[test]
    fn prompt_assembly_opens_the_assistant_turn() {
        let p = assemble_prompt("  a red balloon  ");
        assert!(p.starts_with("<|im_start|>user\n"));
        assert!(p.contains("a red balloon"));
        assert!(p.trim_end().ends_with("<|im_start|>assistant"));
        // The user turn is closed, the assistant turn is not.
        assert_eq!(p.matches("<|im_end|>").count(), 1);
    }

    #[test]
    fn placeholder_conditioning_is_small_and_finite() {
        let c = placeholder_conditioning(6, 16);
        c.validate().unwrap();
        assert_eq!(c.num_tokens(), 6);
        assert!(c.hidden.iter().all(|v| v.is_finite() && v.abs() < 0.05));
    }

    #[test]
    fn tap_layer_is_not_the_last_layer() {
        // Reading the final (post-norm) state instead of layer 50 is a silent
        // quality regression, so pin the intent here.
        let c = cfg();
        assert_eq!(H3TextEncoderConfig::TAP_LAYER, 50);
        assert!(H3TextEncoderConfig::TAP_LAYER < c.num_hidden_layers);
    }
}
