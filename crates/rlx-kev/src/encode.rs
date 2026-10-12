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

//! Packing a request into tokens: `<state> …` then one `<q> … <decide>`
//! branch per question.
//!
//! Port of `kev/model.py::encode` / `rows_of`. Nothing here touches a model,
//! so it is exactly reproducible against the Python and is where the
//! golden-vector tests live.
//!
//! # Delimiters are borrowed, not added
//!
//! The five structural tokens are **rarely-used Qwen specials reused as
//! delimiters**, so no embedding rows are added or trained and the LoRA
//! adapts their meaning. That also means the tokenizer already knows them
//! and we only need `token_to_id`.
//!
//! # Caller text can never produce a delimiter
//!
//! [`sanitize`] rewrites `<|name|>` to `<¦name¦>` (U+00A6 BROKEN BAR) before
//! tokenizing, so an option boundary cannot be forged out of user input. This
//! is a security property of the format, not a formatting nicety — kev ships
//! a "fake delimiter tokens" playground preset that tests exactly this.

use anyhow::{Result, bail};

use crate::api::Record;

/// `<state>`, `<q>`, `<opt>`, `</opt>`, `<decide>` — in that order.
pub const SPECIAL: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];

/// Training context: state tokens, tokens per question branch, whole record.
///
/// Serving raises the first two (see [`crate::model::INFER_MAX_STATE`]); these
/// are the bounds the released checkpoints were trained under and the ones a
/// training run must admit records against.
pub const MAX_STATE: usize = 384;
pub const MAX_BRANCH: usize = 1024;
pub const MAX_PACKED: usize = 2048;

/// `opt[i]` for a state or instruction token.
pub const OPT_NONE: i32 = -1;
/// `opt[i]` for the `<decide>` token.
pub const OPT_DECIDE: i32 = -2;

/// The tokenizer operations the encoder needs.
///
/// A trait rather than a concrete type so the golden-vector tests can run a
/// deterministic stand-in and stay useful without a `tokenizer.json`.
pub trait TokenizerLike {
    /// Id of an existing vocabulary token, or `None` if absent.
    fn token_to_id(&self, token: &str) -> Option<u32>;
    /// Tokenize without adding any special tokens.
    fn encode_no_special(&self, text: &str) -> Result<Vec<u32>>;
}

/// Rewrite `<|name|>` to `<¦name¦>` so caller text cannot emit a control token.
///
/// `name` is one or more of `[A-Za-z0-9_]`, matching the Python regex. On a
/// failed match we advance one byte and keep scanning, which is what
/// `re.sub` does.
pub fn sanitize(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'<' && i + 1 < b.len() && b[i + 1] == b'|' {
            let start = i + 2;
            let mut j = start;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            if j > start && j + 1 < b.len() && b[j] == b'|' && b[j + 1] == b'>' {
                out.push_str("<\u{a6}");
                // The scanned run is ASCII by construction, so this is a
                // char boundary.
                out.push_str(&text[start..j]);
                out.push_str("\u{a6}>");
                i = j + 2;
                continue;
            }
        }
        // Not a match here: copy one character and resume after it.
        let ch = text[i..].chars().next().expect("byte index on a boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// One packed record.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoding {
    pub ids: Vec<u32>,
    /// `0` = state, `k` = question `k` (1-based).
    pub seg: Vec<i32>,
    /// Branch positions restart just after the state.
    pub pos: Vec<u32>,
    /// Per-token option index: [`OPT_NONE`], `0..K-1`, or [`OPT_DECIDE`].
    pub opt: Vec<i32>,
    pub option_isolation: bool,
    /// Index of each question's `<decide>` token.
    pub decide_idx: Vec<usize>,
    /// Index of each option's `</opt>` token, per question.
    pub opt_idx: Vec<Vec<usize>>,
    pub labels: Vec<usize>,
    pub state_truncated: bool,
}

impl Encoding {
    /// Number of state tokens (`seg.count(0)`).
    pub fn state_len(&self) -> usize {
        self.seg.iter().filter(|s| **s == 0).count()
    }
    pub fn num_questions(&self) -> usize {
        self.decide_idx.len()
    }
}

/// One question's branch, as a causal row continuing from the state.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub ids: Vec<u32>,
    pub pos: Vec<u32>,
    /// `<decide>` offset **within the branch**.
    pub decide: usize,
    /// `</opt>` offsets within the branch, one per option.
    pub opts: Vec<usize>,
}

/// Knobs for [`Encoder::encode`].
#[derive(Debug, Clone, Copy)]
pub struct EncodeOpts {
    pub max_state: usize,
    pub max_branch: usize,
    /// Fail instead of truncating an over-long state.
    pub strict: bool,
    /// Make every option span its own sub-branch (packed form only).
    pub option_isolation: bool,
}

impl Default for EncodeOpts {
    fn default() -> Self {
        Self {
            max_state: MAX_STATE,
            max_branch: MAX_BRANCH,
            strict: false,
            option_isolation: false,
        }
    }
}

/// Resolves the five delimiters once, then packs records.
pub struct Encoder<T: TokenizerLike> {
    tok: T,
    /// `[state, q, opt, /opt, decide]`.
    special_ids: [u32; 5],
}

impl<T: TokenizerLike> Encoder<T> {
    /// Resolve the delimiters against `tok`.
    ///
    /// Errors when any is missing rather than substituting something: a
    /// checkpoint trained with `<|box_end|>` as `</opt>` reads a different
    /// token as the option boundary and quietly returns wrong probabilities.
    pub fn new(tok: T) -> Result<Self> {
        let mut special_ids = [0u32; 5];
        for (slot, name) in special_ids.iter_mut().zip(SPECIAL) {
            match tok.token_to_id(name) {
                Some(id) => *slot = id,
                None => bail!(
                    "tokenizer has no {name:?}; kev reuses the five Qwen specials \
                     {SPECIAL:?} as structural delimiters"
                ),
            }
        }
        Ok(Self { tok, special_ids })
    }

    pub fn tokenizer(&self) -> &T {
        &self.tok
    }

    pub fn special_ids(&self) -> [u32; 5] {
        self.special_ids
    }

    /// Tokenize caller-supplied text after [`sanitize`].
    pub fn user_tokens(&self, text: &str) -> Result<Vec<u32>> {
        self.tok.encode_no_special(&sanitize(text))
    }

    /// Pack one record: `[<state> …]` then per-question
    /// `[<q> instr <opt> o </opt>… <decide>]`.
    pub fn encode(&self, rec: &Record, opts: EncodeOpts) -> Result<Encoding> {
        let EncodeOpts {
            max_state,
            max_branch,
            strict,
            option_isolation,
        } = opts;
        if max_state == 0 {
            bail!("max_state must be positive");
        }

        let state_tokens = self.user_tokens(&rec.state)?;
        let state_truncated = state_tokens.len() + 1 > max_state;
        if strict && state_truncated {
            bail!(
                "state exceeds {max_state} tokens: {}",
                state_tokens.len() + 1
            );
        }
        let keep = (max_state - 1).min(state_tokens.len());

        let mut ids = Vec::with_capacity(max_state);
        ids.push(self.special_ids[0]);
        ids.extend_from_slice(&state_tokens[..keep]);
        let ls = ids.len();

        let mut seg = vec![0i32; ls];
        let mut pos: Vec<u32> = (0..ls as u32).collect();
        let mut opt = vec![OPT_NONE; ls];

        let [_, q_id, o_id, c_id, d_id] = self.special_ids;
        let mut decide_idx = Vec::with_capacity(rec.questions.len());
        let mut opt_idx = Vec::with_capacity(rec.questions.len());

        for (k0, q) in rec.questions.iter().enumerate() {
            let k = (k0 + 1) as i32;

            let mut instr = vec![q_id];
            instr.extend(self.user_tokens(&q.instr)?);

            let mut spans: Vec<Vec<u32>> = Vec::with_capacity(q.options.len());
            for o in &q.options {
                let mut sp = vec![o_id];
                sp.extend(self.user_tokens(o)?);
                sp.push(c_id);
                spans.push(sp);
            }

            let mut br = instr.clone();
            for sp in &spans {
                br.extend_from_slice(sp);
            }
            br.push(d_id);

            if br.len() + ls > max_branch {
                bail!("branch too long: {}", br.len());
            }

            let base = ids.len();
            let p0 = ls;

            let mut br_opt = vec![OPT_NONE; instr.len()];
            for (j, sp) in spans.iter().enumerate() {
                br_opt.extend(std::iter::repeat_n(j as i32, sp.len()));
            }
            br_opt.push(OPT_DECIDE);

            let br_pos: Vec<u32> = if option_isolation {
                // Every option span starts at the same position and
                // `<decide>` sits after the longest one, so the option
                // representations and `<decide>`'s view of them are
                // permutation-invariant by construction.
                let longest = spans.iter().map(Vec::len).max().unwrap_or(0);
                let after_instr = p0 + instr.len();
                let mut v: Vec<u32> = (p0..after_instr).map(|p| p as u32).collect();
                for sp in &spans {
                    v.extend((0..sp.len()).map(|i| (after_instr + i) as u32));
                }
                v.push((after_instr + longest) as u32);
                v
            } else {
                (p0..p0 + br.len()).map(|p| p as u32).collect()
            };

            let mut ends = Vec::with_capacity(spans.len());
            let mut cursor = instr.len();
            for sp in &spans {
                cursor += sp.len();
                ends.push(cursor - 1);
            }

            debug_assert_eq!(br.len(), br_opt.len());
            debug_assert_eq!(br.len(), br_pos.len());

            seg.extend(std::iter::repeat_n(k, br.len()));
            pos.extend(br_pos);
            opt.extend(br_opt);
            decide_idx.push(base + br.len() - 1);
            opt_idx.push(ends.iter().map(|e| base + e).collect());
            ids.extend(br);
        }

        Ok(Encoding {
            ids,
            seg,
            pos,
            opt,
            option_isolation,
            decide_idx,
            opt_idx,
            labels: rec.questions.iter().map(|q| q.label).collect(),
            state_truncated,
        })
    }
}

/// Split a packed encoding into its state and per-question branch rows.
///
/// Feeding `state ++ rows[k]` as one causal row is equivalent to the packed
/// block-causal form for question `k` **on any architecture**: the row holds
/// exactly the tokens question `k` may attend to, at the same positions. That
/// is what makes the recurrent Gated DeltaNet layers — which ignore attention
/// masks entirely — give exactly isolated answers.
pub fn rows_of(enc: &Encoding) -> Result<(Vec<u32>, Vec<u32>, Vec<Row>)> {
    let ls = enc.state_len();
    let mut rows = Vec::with_capacity(enc.decide_idx.len());
    let mut start = ls;
    for (k0, (d, oi)) in enc.decide_idx.iter().zip(&enc.opt_idx).enumerate() {
        let k = (k0 + 1) as i32;
        let end = d + 1;
        if enc.seg.get(start).copied() != Some(k) || enc.seg.get(end - 1).copied() != Some(k) {
            bail!("branch layout mismatch at question {k}");
        }
        rows.push(Row {
            ids: enc.ids[start..end].to_vec(),
            pos: enc.pos[start..end].to_vec(),
            decide: d - start,
            opts: oi.iter().map(|o| o - start).collect(),
        });
        start = end;
    }
    Ok((enc.ids[..ls].to_vec(), enc.pos[..ls].to_vec(), rows))
}

#[cfg(feature = "tokenizer")]
mod hf {
    use anyhow::{Context, Result};

    /// A `tokenizers`-backed [`super::TokenizerLike`].
    pub struct HfTokenizer(tokenizers::Tokenizer);

    impl HfTokenizer {
        /// Load `tokenizer.json`.
        pub fn from_file(path: &std::path::Path) -> Result<Self> {
            let t = tokenizers::Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("loading tokenizer {}", path.display()))?;
            Ok(Self(t))
        }

        pub fn inner(&self) -> &tokenizers::Tokenizer {
            &self.0
        }

        /// Detokenize, keeping special tokens — used only for the
        /// billing-style `usage.output_tokens` count.
        pub fn decode(&self, ids: &[u32]) -> Result<String> {
            self.0
                .decode(ids, false)
                .map_err(|e| anyhow::anyhow!("{e}"))
        }
    }

    impl super::TokenizerLike for HfTokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.0.token_to_id(token)
        }
        fn encode_no_special(&self, text: &str) -> Result<Vec<u32>> {
            let enc = self
                .0
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(enc.get_ids().to_vec())
        }
    }
}

#[cfg(feature = "tokenizer")]
pub use hf::HfTokenizer;
