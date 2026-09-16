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

//! Drives the NMT: encode a source sentence, then step the decoder.
//!
//! ```text
//!   src ids ─┬─ embedding ── input_<lang> ── encoder ── handover_<lang> ──┐
//!            │                                       (6 cross-attn K/V)  │
//!   prev id ── embedding ──┐                                             │
//!   position ──────────────┴── decoder_<lang> ◀───────────────────────────┘
//!                              ├─ final_layer_output  [1,1,512]
//!                              └─ 3 × self_attn.accum.next   (carried on)
//! ```
//!
//! # Two things that are not the usual transformer
//!
//! * **Decoder self-attention is an average-attention network.** There is no
//!   self-attention softmax; the decoder carries a running sum per layer
//!   (`self_attn.accum`) and multiplies by `1/position`. Decoding is therefore
//!   O(1) per step in the target length, and the whole recurrent state is three
//!   512-wide vectors.
//! * **`final_layer_output` is a hidden state, not logits.** Scores come from
//!   the `readout` graph, which gathers the *tied* embedding table for a set of
//!   candidate ids and dots them against the hidden state. On device that set
//!   is a shortlist of ~100 (`shortlist-cond-n`); [`Nmt::logits`] can score any
//!   candidate set, including the full vocabulary.
//!
//! The `.espresso.shape` files declare a source length of 64, but that is only
//! the export-time geometry: the graphs are length-polymorphic and take **no
//! attention mask**, so a source is encoded at its true length. Padding it out
//! would put 60 junk positions in front of the decoder's cross-attention.

use crate::espresso::Manifest;
use crate::exec::{Env, Value, run_checked};
use crate::net::Graph;
use crate::pdec::PDecParams;
use crate::shortlist::Shortlist;
use crate::spm::Vocab;
use crate::tensor::Tensor;
use crate::tuning::Tuning;
use anyhow::{Context, Result, anyhow, ensure};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;

/// The recurrent decoder state: one accumulator per layer.
#[derive(Debug, Clone)]
pub struct DecoderState {
    pub accum: Vec<Tensor>,
}

impl DecoderState {
    /// Zeroed state for `layers` layers of `width`.
    pub fn new(layers: usize, width: usize) -> Self {
        Self {
            accum: (0..layers)
                .map(|_| Tensor::zeros(vec![1, 1, width]))
                .collect(),
        }
    }
}

/// A loaded NMT for one target language.
pub struct Nmt {
    pub manifest: Manifest,
    pub lang: String,
    embedding: Graph,
    input: Graph,
    encoder: Graph,
    handover: Graph,
    decoder: Graph,
    readout: Graph,
    /// Declared source length the graphs were compiled for.
    pub source_len: usize,
    /// Model width (`StateWidth`).
    pub width: usize,
    /// Dequantized readout table, built once. The table is 168 000 x 512, so
    /// re-gathering it per decode step dominates everything else.
    /// Dequantized readout rows, built on demand.
    ///
    /// Materialising the whole tied table costs 168 000 x 512 f32 (~344 MB) to
    /// score a shortlist of a few hundred, which is what made a multi-direction
    /// sweep exhaust memory. Rows are cheap to produce and get reused across
    /// decode steps, so cache the ones actually asked for.
    readout_rows: RefCell<BTreeMap<u32, Vec<f32>>>,
    /// Raw `weights_u8` / `Q_meta` / geometry of the readout gather.
    readout_src: RefCell<Option<ReadoutSource>>,
    /// Everything that changes what the search does. See [`Tuning`].
    ///
    /// Seeded from `RLX_TRANSLATE_*` at load, so a sweep can move a lever
    /// without every caller growing a parameter; override with
    /// [`Nmt::tuning_mut`].
    tuning: Tuning,
    /// Whether the decoder's `embedding` input carries the positional term.
    /// Defaults to the manifest's `NeedsPosition`; see
    /// [`Nmt::set_decoder_uses_positions`].
    decoder_positions: bool,
    /// Which shortlist table was loaded, so a mismatch can be caught.
    shortlist_table: String,
    /// Per-source-token candidate table, when the bundle ships one.
    ///
    /// The config's `enable_shortlist` / `shortlist-lang-pair` say the decoder
    /// scores only these candidates rather than the whole vocabulary.
    pub shortlist: Option<Shortlist>,
}

/// Cuts `src` to `cap` tokens, keeping the last one.
///
/// Free-standing so it can be tested without a bundle: the property that
/// matters — the terminator survives the cut — is exactly the one that was
/// wrong, and it does not need 400 MB of weights to check.
fn clamp_source_to(src: &[u32], cap: Option<usize>) -> Vec<u32> {
    let Some(cap) = cap.map(|c| c.max(2)) else {
        return src.to_vec();
    };
    if src.len() <= cap {
        return src.to_vec();
    }
    let mut out: Vec<u32> = src[..cap - 1].to_vec();
    out.push(*src.last().expect("non-empty"));
    out
}

/// Which language each per-language graph comes from.
///
/// [`Parts::uniform`] is what every caller wants; the split exists because
/// whether `input_<lang>` follows the source or the target is an assumption
/// this port made rather than read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parts {
    pub input: String,
    pub handover: String,
    pub decoder: String,
}

impl Parts {
    /// All three graphs from one language.
    ///
    /// What this port did everywhere, and wrong: see [`Parts::for_pair`].
    pub fn uniform(lang: &str) -> Self {
        Parts {
            input: lang.to_string(),
            handover: lang.to_string(),
            decoder: lang.to_string(),
        }
    }

    /// `input_<source>`, `handover_<target>`, `decoder_<target>`.
    ///
    /// `input_<lang>` is the first four encoder blocks and reads the source, so
    /// it follows the **source** language; `handover_<lang>` builds the
    /// decoder's cross-attention and `decoder_<lang>` generates, so those
    /// follow the target. Keying all three by the target is what this port did,
    /// and it survived because the directions it was checked against are into
    /// languages whose bundles tolerate it. `en_US-tr_TR` does not:
    ///
    /// | direction | all-target | input=source |
    /// | --- | --- | --- |
    /// | `en_US-tr_TR` | `Biz yürüdük boyunca nehir boyunca başladı yağmur yağıyor` | `Yağmur yağmaya başlayana kadar nehir boyunca yürüdük` |
    /// | `en_US-zh_TW` | `一直沿河走，直到下雨` | `我們沿著河邊走，直到開始下雨` |
    /// | `tr_TR-en_US` | `The rain started until we walked the river` | `we walked along the river until it started raining` |
    /// | `en_US-fr_FR` | (correct) | (correct, unchanged) |
    ///
    /// The middle two columns for tr and zh_TW are the OS's reference verbatim.
    /// Taking the handover from the source instead produces noise in every
    /// direction, which is the control: only `input_` moves.
    ///
    /// `RLX_TRANSLATE_INPUT_GRAPH=target` restores the old keying.
    pub fn for_pair(source: &str, target: &str) -> Self {
        let by_target = std::env::var("RLX_TRANSLATE_INPUT_GRAPH")
            .is_ok_and(|v| v.eq_ignore_ascii_case("target"));
        Parts {
            input: if by_target { target } else { source }.to_string(),
            handover: target.to_string(),
            decoder: target.to_string(),
        }
    }
}

impl Nmt {
    /// Loads every graph for `lang`.
    ///
    /// `home` must be the directory holding the manifest — the shared graphs
    /// have identical file names in every bundle, so resolving them elsewhere
    /// silently mixes bundles. Per-language graphs may come from `extra`.
    pub fn load(home: &Path, extra: &[&Path], lang: &str) -> Result<Self> {
        Self::load_with_shortlist(home, extra, lang, &format!("all-{lang}"))
    }

    /// Loads every graph for `lang`, with a named shortlist table.
    ///
    /// `table` is the config's `shortlist-lang-pair`, and it is **not** always
    /// `all-<lang>`. `en_US-zh_TW` and `en_US-zh_CN` share one model, one
    /// vocabulary and one `decoder_zh`, and differ only in their direction
    /// token and their shortlist — `all-zh_TW` against `all-zh`. Deriving the
    /// table from the two-letter language handed zh_TW the Simplified table, so
    /// Traditional characters were never candidates and the direction scored
    /// chrF 0.348 against the OS with zero exact matches, the worst of the 43
    /// measured. It is the only shipped table whose name is not `all-<lang>`,
    /// which is why it went unnoticed.
    pub fn load_with_shortlist(
        home: &Path,
        extra: &[&Path],
        lang: &str,
        table: &str,
    ) -> Result<Self> {
        Self::load_parts(home, extra, &Parts::uniform(lang), table)
    }

    /// Loads the bundle for a direction, keying each graph correctly.
    ///
    /// The one to call when both languages are known; see [`Parts::for_pair`]
    /// for why that matters. `source` and `target` are the two-letter graph
    /// names, not locales.
    pub fn load_for_pair(
        home: &Path,
        extra: &[&Path],
        source: &str,
        target: &str,
        table: &str,
    ) -> Result<Self> {
        Self::load_parts(home, extra, &Parts::for_pair(source, target), table)
    }

    /// Loads a bundle choosing each per-language graph independently.
    ///
    /// The three per-language graphs are not obviously all keyed by the same
    /// language: `input_<lang>` is the first four *encoder* blocks and reads the
    /// source, while `handover_<lang>` builds the decoder's cross-attention and
    /// `decoder_<lang>` generates. This port keys all three by the target, which
    /// is right for the directions that work — but "right for the directions
    /// that work" is what a wrong assumption looks like from the inside, so the
    /// combination is reachable and `examples/graph_langs.rs` measures it.
    pub fn load_parts(home: &Path, extra: &[&Path], parts: &Parts, table: &str) -> Result<Self> {
        let lang = parts.decoder.as_str();
        let manifest = Manifest::load(home.join("pyespresso.mdl.bin"))?;
        for l in [&parts.input, &parts.handover, &parts.decoder] {
            ensure!(
                manifest.languages().contains(&l.as_str()),
                "bundle at {} does not cover {l:?} (has {:?})",
                home.display(),
                manifest.languages()
            );
        }
        let find = |net: &str| -> Result<Graph> {
            let dir = std::iter::once(home)
                .chain(extra.iter().copied())
                .find(|d| d.join(net).exists())
                .ok_or_else(|| anyhow!("graph {net} is not installed"))?;
            Graph::load(dir, net).with_context(|| format!("loading {net}"))
        };
        let per = |key: &str, l: &str| -> Result<Graph> {
            let net = manifest
                .lang_graphs
                .get(key)
                .and_then(|m| m.get(l))
                .ok_or_else(|| anyhow!("manifest has no {key} for {l}"))?;
            find(net)
        };

        let embedding = find("embedding.espresso.net")?;
        let readout = find(
            manifest
                .str("ReadoutGraph")
                .unwrap_or("readout.espresso.net"),
        )?;
        let encoder = find(
            manifest
                .str("EncoderGraph")
                .unwrap_or("encoder.espresso.net"),
        )?;
        let input = per("InputLangGraph", &parts.input)?;
        let handover = per("HandoverLangGraph", &parts.handover)?;
        let decoder = per("DecoderLangGraph", &parts.decoder)?;

        // A table that is present but unreadable must be an error. Swallowing
        // it with `.ok()` is what hid the parser bug: `candidates` falls back
        // to the whole vocabulary when there is no shortlist, so every CJK and
        // into-English direction quietly scored 48-96k candidates instead of
        // ~700, and still produced plausible text. A missing file is different
        // — some bundles ship none — and stays a `None`.
        let shortlist = match std::iter::once(home)
            .chain(extra.iter().copied())
            .map(|d| d.join("shortlists").join(format!("{table}.shortlist")))
            .find(|p| p.exists())
        {
            Some(p) => Some(Shortlist::load(&p)?),
            None => None,
        };
        let manifest_positions = manifest.flag("NeedsPosition").unwrap_or(true);
        let source_len = embedding
            .declared_shape("embedding")
            .map(|s| s.len() / s.w.max(1))
            .unwrap_or(64);
        let width = manifest.int("StateWidth").unwrap_or(512) as usize;

        Ok(Self {
            manifest,
            lang: lang.to_string(),
            embedding,
            input,
            encoder,
            handover,
            decoder,
            readout,
            source_len,
            width,
            readout_rows: RefCell::new(BTreeMap::new()),
            readout_src: RefCell::new(None),
            decoder_positions: manifest_positions,
            tuning: Tuning::from_env()?,
            shortlist_table: table.to_string(),
            shortlist,
        })
    }

    /// The source, cut only if [`crate::tuning::Tuning::max_source_tokens`]
    /// asks for it.
    ///
    /// This used to cut at [`Nmt::source_len`] — 64, the length the embedding
    /// graph was traced at — and did it *after* the terminator was appended, so
    /// a longer source lost both its tail and its `<s>`. The graphs are
    /// length-polymorphic and take 501 source pieces without complaint, so
    /// there is nothing to cut for.
    ///
    /// When a cut is asked for, it keeps room for the terminator: losing the
    /// tail costs content, but losing `<s>` leaves the encoder looking at an
    /// unfinished sentence, and the decoder answers that with repetition.
    fn clamp_source(&self, src: &[u32]) -> Vec<u32> {
        clamp_source_to(src, self.tuning.max_source_tokens)
    }

    /// The bundle's manifest, for callers that need its wiring strings.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Number of decoder layers carrying state.
    pub fn layers(&self) -> usize {
        self.manifest.decoder_layers()
    }

    /// Embeds `ids` at `positions`.
    fn embed(&self, ids: &[u32], positions: &[f32]) -> Result<Tensor> {
        ensure!(
            ids.len() == positions.len(),
            "ids and positions differ in length"
        );
        let mut env = Env::new();
        env.insert(
            self.manifest
                .str("SourceInputStr")
                .unwrap_or("src_tokens")
                .to_string(),
            Value::F32(Tensor::new(
                vec![ids.len()],
                ids.iter().map(|v| *v as f32).collect(),
            )?),
        );
        env.insert(
            "positions".to_string(),
            Value::F32(Tensor::new(vec![positions.len()], positions.to_vec())?),
        );
        let out = run_checked(&self.embedding, env)?;
        Ok(out
            .get("embedding")
            .ok_or_else(|| anyhow!("embedding graph produced no `embedding`"))?
            .f32()?
            .clone())
    }

    /// Token embedding without the positional term.
    ///
    /// The manifest's `NeedsEncoderPositions` names the *encoder*, and the
    /// decoder receives `position` separately as the divisor of its
    /// average-attention mean, so the decoder's `embedding` input is plausibly
    /// the bare token embedding. Selected by [`Nmt::decoder_uses_positions`].
    fn embed_token_only(&self, id: u32) -> Result<Tensor> {
        let gather = self
            .embedding
            .layers
            .iter()
            .filter(|l| l.kind == "quantized_gather")
            .max_by_key(|l| l.int("nRow").unwrap_or(0))
            .ok_or_else(|| anyhow!("embedding graph has no vocabulary gather"))?;
        let cols = gather
            .int("nCol")
            .ok_or_else(|| anyhow!("gather has no nCol"))? as usize;
        let rows = gather
            .int("nRow")
            .ok_or_else(|| anyhow!("gather has no nRow"))? as usize;
        ensure!(
            (id as usize) < rows,
            "token {id} is outside a {rows}-row vocabulary"
        );
        let table = self.embedding.weights.raw(gather.req_blob("weights_u8")?)?;
        let meta = self.embedding.weights.f32s(gather.req_blob("Q_meta")?)?;
        // The scale that follows the gather in the graph (sqrt(width)).
        let alpha = self
            .embedding
            .layers
            .iter()
            .find(|l| l.kind == "elementwise" && l.int("operation") == Some(3))
            .and_then(|l| l.float("alpha"))
            .unwrap_or(1.0) as f32;
        let base = id as usize * cols;
        let data: Vec<f32> = (0..cols)
            .map(|c| {
                crate::exec::dequant_gather_row(table[base + c], &meta[c * 4..c * 4 + 4]) * alpha
            })
            .collect();
        Tensor::new(vec![1, cols], data)
    }

    /// Whether the decoder's `embedding` input carries the positional term.
    pub fn decoder_uses_positions(&self) -> bool {
        self.decoder_positions
    }

    /// Candidate target tokens for a source sentence: the shortlist union plus
    /// `always`. Falls back to the whole vocabulary when no table is installed.
    pub fn candidates(&self, source: &[u32], always: &[u32], vocab: usize) -> Vec<u32> {
        match &self.shortlist {
            Some(s) => s.union(source, always),
            None => (0..vocab as u32).collect(),
        }
    }

    /// The encoder input: control tags, the sentence, and the terminator.
    ///
    /// `<s>` terminates the source as well as opening the target; `</s>` is
    /// essentially unused. Where the *second* target tag goes is a setting —
    /// see [`crate::tuning::Tuning::domain_tag_last`].
    fn tagged_source(
        &self,
        vocab: &Vocab,
        params: &PDecParams,
        text: &str,
        bos: u32,
    ) -> Result<Vec<u32>> {
        let id = |piece: &String| -> Result<u32> {
            vocab
                .id(piece)
                .ok_or_else(|| anyhow!("vocabulary has no control token {piece}"))
        };
        let target = params.target_token_pieces();
        let (lead, trail) = if self.tuning.domain_tag_last && target.len() > 1 {
            target.split_at(1)
        } else {
            (target.as_slice(), &[][..])
        };
        let mut src = Vec::new();
        for piece in params.source_token_pieces().iter().chain(lead) {
            src.push(id(piece)?);
        }
        src.extend(vocab.encode(text));
        for piece in trail {
            src.push(id(piece)?);
        }
        src.push(bos);
        Ok(src)
    }

    /// Refuses to emit a target n-gram the hypothesis already contains.
    ///
    /// **Defaults to 3.** the shipped config names no n-gram constraint, so this was
    /// off at first and measured as a trade — but that measurement used the
    /// phrasebook-derived reference dump, where the decoder never repeats and
    /// the lever therefore has nothing to do. Re-measured on non-lexicon
    /// sentences, where the OS's own text is reproduced and repetition actually
    /// happens (`avant de fermer le bureau avant de fermer le bureau`), it
    /// **moves output closer to the OS**: chrF 0.693 -> 0.720 and one more exact
    /// match over 21 pivot sentences, with the lexicon bench unchanged at 112
    /// identical / chrF 0.808. 3 and 4 score the same. Set 0 to disable.
    pub fn set_no_repeat_ngram(&mut self, n: usize) {
        self.tuning.no_repeat_ngram = n;
    }

    /// Every search setting at once.
    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    /// Mutable access to every search setting.
    ///
    /// The individual setters below predate [`Tuning`] and remain as
    /// shorthands; this is the whole surface, including the levers that used
    /// to be constants inside `translate_nbest`.
    pub fn tuning_mut(&mut self) -> &mut Tuning {
        &mut self.tuning
    }

    /// Replaces every search setting.
    pub fn set_tuning(&mut self, t: Tuning) {
        self.tuning = t;
    }

    /// Refuses to repeat a run of `n` characters within one hypothesis.
    ///
    /// Default 0 (off), matching the OS. Unlike
    /// [`Nmt::set_no_repeat_ngram`] this sees repetition that spans token
    /// boundaries, which is how a word missing from the vocabulary stutters.
    pub fn set_no_repeat_char_ngram(&mut self, n: usize) {
        self.tuning.no_repeat_char_ngram = n;
    }

    /// Scores the full vocabulary rather than the config's shortlist.
    ///
    /// Diagnostic: `enable_shortlist` is what the OS runs, and the shortlist is
    /// also what makes decoding fast. Turning it off answers whether a bad
    /// output was a *search* failure or a *reachability* one.
    pub fn set_ignore_shortlist(&mut self, v: bool) {
        self.tuning.ignore_shortlist = v;
    }

    /// Exponent in `score / len^alpha`, overriding the config's `norm-costs`.
    ///
    /// `norm-costs` is that boolean's two endpoints: true is 1.0, false is 0.0.
    /// `rlx-nllb` models the same thing as a tunable exponent, and both
    /// endpoints measured worse than they need to, so the middle is worth
    /// searching. `None` restores the config's own setting.
    pub fn set_length_penalty(&mut self, alpha: Option<f64>) {
        self.tuning.length_penalty = alpha;
    }

    /// Overrides [`Nmt::decoder_uses_positions`].
    ///
    /// Which reading is correct is not settled — the manifest's `NeedsPosition`
    /// is ambiguous between the positional embedding and the `1/position`
    /// scalar the decoder takes separately — so both are reachable and can be
    /// measured against reference output.
    pub fn set_decoder_uses_positions(&mut self, v: bool) {
        self.decoder_positions = v;
    }

    /// Encoder states for a source, before the handover projection.
    ///
    /// Exposed for diagnosis: comparing whole handover tensors across two
    /// sources is confounded, because every padded position is identical in
    /// both and dominates the similarity.
    pub fn encoder_states(&self, src: &[u32]) -> Result<Tensor> {
        let ids: Vec<u32> = self.clamp_source(src);
        let positions: Vec<f32> = (1..=ids.len()).map(|i| i as f32).collect();
        let embedded = self.embed(&ids, &positions)?;
        let mut env = Env::new();
        env.insert("embedding".to_string(), Value::F32(embedded));
        let env = run_checked(&self.input, env)?;
        let bridge = self
            .manifest
            .str("InputNetValuesStr")
            .unwrap_or("encoder.3.output");
        let mid = env
            .get(bridge)
            .ok_or_else(|| anyhow!("input net produced no {bridge}"))?
            .clone();
        let mut env = Env::new();
        env.insert(bridge.to_string(), mid);
        let env = run_checked(&self.encoder, env)?;
        let enc = self
            .manifest
            .str("EncoderValuesStr")
            .unwrap_or("encoder.15.output");
        Ok(env
            .get(enc)
            .ok_or_else(|| anyhow!("encoder produced no {enc}"))?
            .f32()?
            .clone())
    }

    /// Full encoder output for `src`: `[tokens, width]`.
    ///
    /// This is the model's own contextual representation of a sentence, before
    /// the handover projects it into per-layer cross-attention keys and values.
    /// Mean-pooled it gives a sentence embedding, which is what
    /// [`crate::score`] uses to judge whether two different wordings mean the
    /// same thing — something character overlap cannot see.
    pub fn encoder_output(&self, src: &[u32]) -> Result<Tensor> {
        let ids: Vec<u32> = self.clamp_source(src);
        let positions: Vec<f32> = (1..=ids.len()).map(|i| i as f32).collect();
        let embedded = self.embed(&ids, &positions)?;
        let mut env = Env::new();
        env.insert("embedding".to_string(), Value::F32(embedded));
        let env = run_checked(&self.input, env)?;
        let bridge = self
            .manifest
            .str("InputNetValuesStr")
            .unwrap_or("encoder.3.output");
        let mid = env
            .get(bridge)
            .ok_or_else(|| anyhow!("input net produced no {bridge}"))?
            .clone();
        let mut env = Env::new();
        env.insert(bridge.to_string(), mid);
        let env = run_checked(&self.encoder, env)?;
        let name = self
            .manifest
            .str("EncoderValuesStr")
            .unwrap_or("encoder.15.output");
        Ok(env
            .get(name)
            .ok_or_else(|| anyhow!("encoder produced no {name}"))?
            .f32()?
            .clone())
    }

    /// Mean-pooled encoder output for `text`, tagged as `locale`.
    ///
    /// The encoder is multilingual and shared, so strings in any of the
    /// bundle's languages land in one space and are directly comparable.
    pub fn sentence_embedding(&self, vocab: &Vocab, locale: &str, text: &str) -> Result<Vec<f32>> {
        let tag = vocab
            .id(&format!("<src-{locale}>"))
            .ok_or_else(|| anyhow!("vocabulary has no <src-{locale}>"))?;
        let bos = vocab
            .id("<s>")
            .ok_or_else(|| anyhow!("vocabulary has no <s>"))?;
        let mut src = vec![tag];
        src.extend(vocab.encode(text));
        src.push(bos);
        let out = self.encoder_output(&src)?;
        let w = out.width();
        let rows = out.len() / w.max(1);
        let mut v = vec![0.0f32; w];
        for r in 0..rows {
            for (i, x) in out.row(r).iter().enumerate() {
                v[i] += *x;
            }
        }
        let n = rows.max(1) as f32;
        for x in &mut v {
            *x /= n;
        }
        Ok(v)
    }

    /// Encodes a source sentence, returning the cross-attention tensors the
    /// decoder consumes. `src` is encoded at its true length (truncated to
    /// [`Nmt::source_len`]); the graphs are length-polymorphic and take no
    /// attention mask, so padding would be attended to as real content.
    pub fn encode(&self, src: &[u32]) -> Result<Env> {
        let ids: Vec<u32> = self.clamp_source(src);
        // Positional-embedding indices are 0-based. `PositionZeroBased = F`
        // refers to the decoder's `position` scalar (the divisor of the
        // average-attention mean), which is 1-based; making the embedding
        // indices 1-based too was measured and made the output strictly worse.
        let positions: Vec<f32> = (1..=ids.len()).map(|i| i as f32).collect();

        let embedded = self.embed(&ids, &positions)?;
        let mut env = Env::new();
        env.insert("embedding".to_string(), Value::F32(embedded));
        let env = run_checked(&self.input, env)?;

        let bridge = self
            .manifest
            .str("InputNetValuesStr")
            .unwrap_or("encoder.3.output");
        let mid = env
            .get(bridge)
            .ok_or_else(|| anyhow!("input net produced no {bridge}"))?
            .clone();
        let mut env = Env::new();
        env.insert(bridge.to_string(), mid);
        let env = run_checked(&self.encoder, env)?;

        let enc_name = self
            .manifest
            .str("EncoderValuesStr")
            .unwrap_or("encoder.15.output");
        let enc = env
            .get(enc_name)
            .ok_or_else(|| anyhow!("encoder produced no {enc_name}"))?
            .clone();
        let mut env = Env::new();
        env.insert(enc_name.to_string(), enc);
        let hv = run_checked(&self.handover, env)?;

        // Keep only the tensors the decoder names, so the caller cannot
        // accidentally feed a stale intermediate through.
        let mut out = Env::new();
        for name in self.manifest.csv("HandoverStrings") {
            let v = hv
                .get(&name)
                .ok_or_else(|| anyhow!("handover produced no {name}"))?
                .clone();
            out.insert(name, v);
        }
        Ok(out)
    }

    /// One decoder step. Returns the hidden state and the next recurrent state.
    ///
    /// `position` is 1-based, for two independent reasons: the decoder forms its
    /// average-attention mean as `(x + accum) * (1/position)`, so it must equal
    /// the number of tokens accumulated *including* this one; and the manifest
    /// sets `PositionZeroBased = F`, so the positional embedding is 1-based too.
    /// Greedy translation of one sentence, using the pair's own control tokens.
    ///
    /// The protocol is not stated anywhere in the config, and every part of it
    /// was settled by measurement against the OS's own output:
    ///
    /// * the source carries the wrapped **source and target** tags, in that
    ///   order ([`PDecParams::source_token_pieces`]);
    /// * **`<s>` opens the target sequence and also terminates it** — it is not
    ///   a control token to be suppressed, and suppressing it leaves the model
    ///   unable to stop, so it loops;
    /// * **`<s>` terminates the source too**, not `</s>`. Using `</s>` there
    ///   costs a factor of ~25 in the rank of the OS's next piece.
    ///
    /// Scoring is restricted to the shortlist, as `enable_shortlist` asks.
    pub fn translate_greedy(
        &self,
        vocab: &Vocab,
        params: &PDecParams,
        text: &str,
        budget: usize,
    ) -> Result<String> {
        let bos = vocab
            .id("<s>")
            .ok_or_else(|| anyhow!("vocabulary has no <s> to open the target with"))?;
        let src = self.tagged_source(vocab, params, text, bos)?;

        let handover = self.encode(&src)?;
        let cands = self.candidates(&src, &[bos], vocab.len());
        let mut state = DecoderState::new(self.layers(), self.width);
        let mut prev = bos;
        let mut pieces: Vec<u32> = Vec::new();
        for pos in 1..=budget.max(1) {
            let (hidden, next) = self.step(prev, pos, &state, &handover)?;
            let scores = self.logits(&hidden, &cands)?;
            let (best, _) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| anyhow!("no candidate scored"))?;
            let id = cands[best];
            state = next;
            prev = id;
            if id == bos {
                break;
            }
            pieces.push(id);
        }
        Ok(vocab.decode(&pieces))
    }

    /// The `n` best translations of one sentence, best first.
    ///
    /// Beam search keeps several hypotheses alive anyway, so the runners-up are
    /// free — and they are genuinely informative here, because this model often
    /// has more than one correct answer (`"J'adore l'été"` and `"J'aime
    /// l'été"` are both right for *i love the summer*). Greedy decoding throws
    /// that away; [`Nmt::translate_greedy`] is the single-answer path.
    ///
    /// `score` is the summed log-probability and `normalized_score` divides it
    /// by length when the config sets `norm-costs` — without that, a shorter
    /// hypothesis always outranks a longer one simply for having fewer terms.
    pub fn translate_nbest(
        &self,
        vocab: &Vocab,
        params: &PDecParams,
        text: &str,
        n: usize,
    ) -> Result<Vec<Variant>> {
        let bos = vocab
            .id("<s>")
            .ok_or_else(|| anyhow!("vocabulary has no <s> to open the target with"))?;
        let eos = vocab.id("</s>").unwrap_or(bos);
        let src = self.tagged_source(vocab, params, text, bos)?;

        let handover = self.encode(&src)?;
        // The shortlist is ~100 target pieces per source token. For a rare word
        // that has to be *spelled* from fragments, the fragments it needs may
        // simply not be in that union — in which case no amount of search can
        // produce the right spelling.
        // A model loaded with the wrong table still translates, just into the
        // wrong variant, so this has to be an error rather than a warning.
        // An empty `lang_pair` means the block named no table, so there is
        // nothing to disagree with.
        if params.shortlist.enabled
            && !params.shortlist.lang_pair.is_empty()
            && !self.tuning.ignore_shortlist
        {
            ensure!(
                self.shortlist_table == params.shortlist.lang_pair,
                "model was loaded with shortlist {:?} but this direction asks for {:?}; \
                 use Nmt::load_with_shortlist",
                self.shortlist_table,
                params.shortlist.lang_pair
            );
        }
        let cands = if self.tuning.ignore_shortlist {
            (0..vocab.len() as u32).collect()
        } else {
            self.candidates(&src, &[bos], vocab.len())
        };
        // Suppress what the config names, plus the *direction* tags — emitting
        // `<src-en_US>` or `<en_US-fr_FR-optimal>` mid-sentence is garbage. `<s>`
        // must stay scoreable because it terminates the sequence.
        //
        // A blanket "drop everything angle-bracketed" also removes the
        // `<STRUCT_*>` markers, which is wrong: those are how the model offers
        // gender alternatives, and this pair's `shortlist-suppress-tokens` is
        // empty, so the OS suppresses nothing at all.
        let named: std::collections::BTreeSet<u32> = params
            .shortlist
            .suppress_tokens
            .iter()
            .filter_map(|t| vocab.id(t))
            .collect();
        let suppressed: std::collections::BTreeSet<u32> = (0..vocab.len() as u32)
            .filter(|i| {
                if *i == bos || *i == eos {
                    return false;
                }
                if named.contains(i) {
                    return true;
                }
                let p = vocab.piece(*i).unwrap_or("");
                if p.starts_with("<STRUCT_") {
                    return false;
                }
                p.starts_with("<src-")
                    || p.starts_with("<tar-")
                    || p.ends_with("-optimal>")
                    || p == "<unk>"
                    || p == "<GENDER_TAG>"
            })
            .collect();
        // Ask for more than we need. Several token paths decode to the same
        // string — a different segmentation, or a case variant the sentence
        // caser would collapse anyway — and a list that repeats one answer
        // three times tells the caller nothing.
        let want = n.max(1);
        let mut params = params.clone();
        self.tuning.apply(&mut params, want);
        let options = crate::beam::SearchOptions {
            pruning: if self.tuning.rs_beam_prune {
                crate::beam::PruningPolicy::RelativeToBest {
                    factor: params.rs_beam,
                }
            } else {
                crate::beam::PruningPolicy::default()
            },
            suppressed,
            no_repeat_ngram: self.tuning.no_repeat_ngram,
            no_repeat_char_ngram: self.tuning.no_repeat_char_ngram,
            length_penalty: self.tuning.length_penalty,
        };
        let mut dec = self.beam_decoder(&handover, vocab.len(), bos, cands);
        dec.pieces = Some(vocab);
        let hyps = crate::beam::search(&mut dec, &params, &[bos], src.len(), &options)?;
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut out = Vec::with_capacity(want);
        for h in hyps {
            let text = vocab.decode(&h.tokens);
            if text.is_empty() || !seen.insert(text.to_lowercase()) {
                continue;
            }
            out.push(Variant {
                text,
                score: h.score,
                normalized_score: h.normalized_score,
            });
            if out.len() == want {
                break;
            }
        }
        Ok(out)
    }

    /// Runs the input net for `src` and returns its whole environment.
    ///
    /// Exposed for diagnosis: the encoder's own `self_attn.attn_probs` decide
    /// whether the source is being mixed sanely or attention has saturated to
    /// a hard argmax, and they are not observable from [`Nmt::encode`].
    pub fn encode_env(&self, src: &[u32]) -> Result<Env> {
        let ids: Vec<u32> = self.clamp_source(src);
        let positions: Vec<f32> = (1..=ids.len()).map(|i| i as f32).collect();
        let embedded = self.embed(&ids, &positions)?;
        let mut env = Env::new();
        env.insert("embedding".to_string(), Value::F32(embedded));
        run_checked(&self.input, env)
    }

    pub fn step(
        &self,
        prev: u32,
        position: usize,
        state: &DecoderState,
        handover: &Env,
    ) -> Result<(Tensor, DecoderState)> {
        let out = self.step_env(prev, position, state, handover)?;
        let scores = self
            .manifest
            .str("ScoresStr")
            .unwrap_or("final_layer_output");
        let hidden = out
            .get(scores)
            .ok_or_else(|| anyhow!("decoder produced no {scores}"))?
            .f32()?
            .clone();
        let mut next = Vec::with_capacity(self.layers());
        for name in self.manifest.csv("StateStrings") {
            let key = format!("{name}.next");
            next.push(
                out.get(&key)
                    .ok_or_else(|| anyhow!("decoder produced no {key}"))?
                    .f32()?
                    .clone(),
            );
        }
        Ok((hidden, DecoderState { accum: next }))
    }

    /// Adapts this model to [`crate::beam::NmtDecoder`] for a single sentence.
    ///
    /// `handover` is the cross-attention state from [`Nmt::encode`], and `stop`
    /// is the piece that terminates a hypothesis. On this model that is `<s>`,
    /// which both opens and closes the target sequence.
    pub fn beam_decoder<'a>(
        &'a self,
        handover: &'a Env,
        vocab: usize,
        stop: u32,
        candidates: Vec<u32>,
    ) -> BeamAdapter<'a> {
        BeamAdapter {
            seen: RefCell::new(std::collections::HashMap::new()),
            incremental: self.tuning.incremental,
            nmt: self,
            pieces: None,
            handover,
            vocab,
            stop,
            candidates,
        }
    }

    /// One decoder step for several hypotheses at once.
    ///
    /// Beam search advances every live hypothesis through the same position, so
    /// they differ only in their previous token and their accumulator. Run as
    /// separate calls, each re-reads every decoder weight; stacked into one
    /// call the weights are read once, which is the difference between a
    /// bandwidth-bound GEMV and a compute-bound GEMM.
    ///
    /// **Every shipped decoder graph refuses this today**, and this is not
    /// wired into the search. `examples/batch_probe.rs` asks directly and gets
    /// `reshape_2 (reshape): shape [8, 1, 64] needs 512 elements, got 1024` —
    /// heads by *one query* by head-dim, with the query count baked in. Making
    /// it `[8, n, 64]` also needs a transpose inserted around it, because a
    /// stacked `[n, 512]` is row-major where the reshape wants head-major. That
    /// is a graph rewrite in the attention layout, which is where this port's
    /// worst bugs have lived, so it is not attempted on the strength of a
    /// timing that a shared machine cannot measure.
    ///
    /// Kept because the probe needs it, and because the remaining work is that
    /// rewrite rather than anything about decoding: the probe checks row 0
    /// against a single step, so the batching itself is verifiable the moment a
    /// graph accepts it.
    pub fn step_batch(
        &self,
        prev: &[u32],
        position: usize,
        states: &[DecoderState],
        handover: &Env,
    ) -> Result<(Tensor, Vec<DecoderState>)> {
        ensure!(!prev.is_empty(), "step_batch needs at least one hypothesis");
        ensure!(
            prev.len() == states.len(),
            "{} tokens for {} states",
            prev.len(),
            states.len()
        );
        ensure!(position >= 1, "decoder positions are 1-based");
        let n = prev.len();
        let w = self.width;
        let positions = vec![position as f32; n];
        let embedded = if self.decoder_uses_positions() {
            self.embed(prev, &positions)?
        } else {
            let mut rows = Vec::with_capacity(n * w);
            for t in prev {
                rows.extend_from_slice(self.embed_token_only(*t)?.data());
            }
            Tensor::new(vec![n, w], rows)?
        };

        let mut env: Env = handover.clone();
        env.insert("embedding".to_string(), Value::F32(embedded));
        env.insert(
            "position".to_string(),
            Value::F32(Tensor::new(vec![1], vec![position as f32])?),
        );
        for (i, name) in self.manifest.csv("StateStrings").iter().enumerate() {
            let mut rows = Vec::with_capacity(n * w);
            for st in states {
                let t = st
                    .accum
                    .get(i)
                    .ok_or_else(|| anyhow!("state has no accumulator {i}"))?;
                ensure!(t.len() == w, "accumulator {i} is {} wide, not {w}", t.len());
                rows.extend_from_slice(t.data());
            }
            env.insert(name.clone(), Value::F32(Tensor::new(vec![n, w], rows)?));
        }
        let out = run_checked(&self.decoder, env)?;

        let scores = self
            .manifest
            .str("ScoresStr")
            .unwrap_or("final_layer_output");
        let hidden = out
            .get(scores)
            .ok_or_else(|| anyhow!("decoder produced no {scores}"))?
            .f32()?
            .clone();
        ensure!(
            hidden.len() == n * w,
            "batched decoder produced {} values, not {n}x{w}",
            hidden.len()
        );
        let names = self.manifest.csv("StateStrings");
        let mut nexts: Vec<Vec<Tensor>> = vec![Vec::with_capacity(names.len()); n];
        for name in &names {
            let key = format!("{name}.next");
            let t = out
                .get(&key)
                .ok_or_else(|| anyhow!("decoder produced no {key}"))?
                .f32()?;
            ensure!(t.len() == n * w, "{key} is {} values, not {n}x{w}", t.len());
            for (i, slot) in nexts.iter_mut().enumerate() {
                slot.push(Tensor::new(vec![w], t.data()[i * w..(i + 1) * w].to_vec())?);
            }
        }
        Ok((
            hidden,
            nexts
                .into_iter()
                .map(|accum| DecoderState { accum })
                .collect(),
        ))
    }

    /// Runs one decoder step and returns the whole graph environment.
    ///
    /// Exposed because whether the source is attended to at all is decided by
    /// `decoder.*.encoder_attn.attn_probs`, which [`Nmt::step`] discards.
    pub fn step_env(
        &self,
        prev: u32,
        position: usize,
        state: &DecoderState,
        handover: &Env,
    ) -> Result<Env> {
        ensure!(position >= 1, "decoder positions are 1-based");
        ensure!(
            state.accum.len() == self.layers(),
            "state has {} accumulators, model has {} layers",
            state.accum.len(),
            self.layers()
        );
        let embedded = if self.decoder_uses_positions() {
            self.embed(&[prev], &[position as f32])?
        } else {
            self.embed_token_only(prev)?
        };

        let mut env: Env = handover.clone();
        env.insert("embedding".to_string(), Value::F32(embedded));
        env.insert(
            "position".to_string(),
            Value::F32(Tensor::new(vec![1], vec![position as f32])?),
        );
        for (name, t) in self.manifest.csv("StateStrings").iter().zip(&state.accum) {
            env.insert(name.clone(), Value::F32(t.clone()));
        }
        run_checked(&self.decoder, env)
    }

    /// Scores `candidates` against `hidden` through the readout's tied
    /// embedding table, returning log-probabilities in candidate order.
    ///
    /// On device the candidate set is a shortlist; passing the whole vocabulary
    /// gives the unrestricted distribution.
    pub fn logits(&self, hidden: &Tensor, candidates: &[u32]) -> Result<Vec<f32>> {
        ensure!(!candidates.is_empty(), "no candidates to score");
        self.ensure_readout_source()?;
        let src = self.readout_src.borrow();
        let src = src.as_ref().expect("source was just built");
        ensure!(
            src.cols == hidden.width(),
            "readout width {} does not match hidden width {}",
            src.cols,
            hidden.width()
        );

        let h = hidden.data();
        let w = hidden.width();
        ensure!(h.len() >= w, "hidden state is shorter than its width");
        // Use the final row: a step produces one position.
        let hv = &h[h.len() - w..];
        let bytes = self.readout.weights.raw(src.weights_blob)?;
        let meta = self.readout.weights.f32s(src.meta_blob)?;
        let mut rows = self.readout_rows.borrow_mut();
        let mut scores: Vec<f32> = Vec::with_capacity(candidates.len());
        for c in candidates {
            let i = *c as usize;
            if i >= src.rows {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let row = rows.entry(*c).or_insert_with(|| {
                let base = i * src.cols;
                (0..src.cols)
                    .map(|k| {
                        crate::exec::dequant_gather_row(bytes[base + k], &meta[k * 4..k * 4 + 4])
                    })
                    .collect()
            });
            scores.push(row.iter().zip(hv).map(|(a, b)| a * b).sum());
        }
        // `ReadoutWithSoftmax` + `ApplyLog` — log-softmax over the candidates.
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter().map(|v| (v - max).exp()).sum();
        let log_z = max + sum.ln();
        for s in &mut scores {
            *s -= log_z;
        }
        Ok(scores)
    }

    /// Reads the readout gather's raw arrays once, without expanding them.
    ///
    /// The expansion is what costs memory, so it is deferred to
    /// [`Nmt::logits`], which only ever needs the rows in the candidate set.
    fn ensure_readout_source(&self) -> Result<()> {
        if self.readout_src.borrow().is_some() {
            return Ok(());
        }
        let gather = self
            .readout
            .layers
            .iter()
            .find(|l| l.kind == "quantized_gather")
            .ok_or_else(|| anyhow!("readout graph has no gather"))?;
        let cols = gather.int("nCol").ok_or_else(|| anyhow!("no nCol"))? as usize;
        let rows = gather.int("nRow").ok_or_else(|| anyhow!("no nRow"))? as usize;
        let weights_blob = gather.req_blob("weights_u8")?;
        let meta_blob = gather.req_blob("Q_meta")?;
        ensure!(
            self.readout.weights.raw(weights_blob)?.len() >= rows * cols
                && self.readout.weights.f32s(meta_blob)?.len() >= cols * 4,
            "readout arrays are shorter than the declared {rows}x{cols}"
        );
        *self.readout_src.borrow_mut() = Some(ReadoutSource {
            weights_blob,
            meta_blob,
            rows,
            cols,
        });
        Ok(())
    }
}

/// Drives [`crate::beam::search`] from an [`Nmt`].
///
/// Beam search asks for the continuations of an arbitrary prefix. Replaying
/// each prefix from a fresh state is the obvious implementation and it is what
/// this did first — but it is quadratic, and measured it dominated everything
/// else: 10.9 s of beam search against 0.36 s of encoding and 3.8 ms per
/// decoder step, because a beam of 8 over 15 steps replays ~960 steps to
/// perform ~120.
///
/// Beam search only ever extends a prefix by one token, so caching the state
/// each prefix leaves behind turns the replay into a single step. The cache is
/// scoped to one sentence — the adapter is built per call — and holds a state
/// plus hidden vector per live prefix, a few kilobytes each.
pub struct BeamAdapter<'a> {
    /// Decoder state and hidden vector left behind by each prefix seen so far.
    seen: RefCell<std::collections::HashMap<Vec<u32>, (DecoderState, Tensor)>>,
    /// False replays every prefix; see [`crate::tuning::Tuning::incremental`].
    incremental: bool,
    nmt: &'a Nmt,
    /// Only needed to render a hypothesis for the character-level check.
    pieces: Option<&'a Vocab>,
    handover: &'a Env,
    vocab: usize,
    stop: u32,
    /// Ids worth scoring — the shortlist union. Everything else is reported as
    /// `-inf`, which both applies the config's `enable_shortlist` and avoids
    /// dotting the hidden state against all 168 000 rows every step.
    candidates: Vec<u32>,
}

impl crate::beam::NmtDecoder for BeamAdapter<'_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn eos_id(&self) -> u32 {
        self.stop
    }

    fn surface(&self, tokens: &[u32]) -> Option<String> {
        let v = self.pieces?;
        Some(v.decode(tokens))
    }

    fn log_probs(&mut self, prefix: &[u32]) -> Result<Vec<f32>> {
        ensure!(!prefix.is_empty(), "beam search asked for an empty prefix");
        if self.incremental
            && let Some((_, h)) = self.seen.borrow().get(prefix)
        {
            let h = h.clone();
            return self.score(&h);
        }
        // The parent prefix is almost always present: beam search extends by one
        // token at a time. Fall back to a full replay only for the first call.
        let parent = self
            .incremental
            .then(|| self.seen.borrow().get(&prefix[..prefix.len() - 1]).cloned())
            .flatten();
        let (mut state, start) = match parent {
            Some((st, _)) => (st, prefix.len() - 1),
            None => (DecoderState::new(self.nmt.layers(), self.nmt.width), 0),
        };
        let mut hidden = None;
        for (i, t) in prefix.iter().enumerate().skip(start) {
            let (h, next) = self.nmt.step(*t, i + 1, &state, self.handover)?;
            state = next;
            hidden = Some(h);
        }
        let h = hidden.expect("prefix is non-empty");
        if self.incremental {
            self.seen
                .borrow_mut()
                .insert(prefix.to_vec(), (state, h.clone()));
        }
        self.score(&h)
    }
}

impl BeamAdapter<'_> {
    /// Spreads shortlist scores over a full-vocabulary vector.
    fn score(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        let scored = self.nmt.logits(hidden, &self.candidates)?;
        let mut out = vec![f32::NEG_INFINITY; self.vocab];
        for (id, s) in self.candidates.iter().zip(scored) {
            let i = *id as usize;
            ensure!(i < self.vocab, "candidate {i} is outside the vocabulary");
            out[i] = s;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::clamp_source_to;

    #[test]
    fn an_uncapped_source_is_untouched() {
        let src: Vec<u32> = (0..200).collect();
        assert_eq!(clamp_source_to(&src, None), src);
    }

    #[test]
    fn a_cut_source_keeps_its_terminator() {
        // 1 is `<s>`, which both opens the target and closes the source. The
        // cut used to be a plain `take(64)`, which dropped it — and a source
        // the encoder sees as unfinished comes back as repetition, not as a
        // truncated sentence, so the symptom pointed nowhere near the cause.
        let mut src: Vec<u32> = (10..80).collect();
        src.push(1);
        let cut = clamp_source_to(&src, Some(64));
        assert_eq!(cut.len(), 64);
        assert_eq!(*cut.last().expect("non-empty"), 1);
        assert_eq!(cut[..63], src[..63]);
    }

    #[test]
    fn a_source_shorter_than_the_cap_is_untouched() {
        let src = vec![5, 6, 7, 1];
        assert_eq!(clamp_source_to(&src, Some(64)), src);
    }

    #[test]
    fn a_degenerate_cap_still_leaves_room_for_the_terminator() {
        let src = vec![5, 6, 7, 8, 1];
        // Anything below 2 cannot hold a token and a terminator.
        assert_eq!(clamp_source_to(&src, Some(0)), vec![5, 1]);
        assert_eq!(clamp_source_to(&src, Some(3)), vec![5, 6, 1]);
    }
    use super::*;

    #[test]
    fn fresh_state_is_zeroed_and_correctly_shaped() {
        let s = DecoderState::new(3, 512);
        assert_eq!(s.accum.len(), 3);
        for a in &s.accum {
            assert_eq!(a.dims(), &[1, 1, 512]);
            assert!(a.data().iter().all(|v| *v == 0.0));
        }
    }
}

/// One candidate translation from [`Nmt::translate_nbest`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct Variant {
    pub text: String,
    /// Summed log-probability of the hypothesis.
    pub score: f64,
    /// Length-normalized score when `norm-costs` is set, else `score`.
    pub normalized_score: f64,
}

/// Where the readout gather's arrays live, without copying them.
///
/// The weights are already resident in the graph; holding a second copy costs
/// another 86 MB per model for nothing.
struct ReadoutSource {
    weights_blob: u64,
    meta_blob: u64,
    rows: usize,
    cols: usize,
}
