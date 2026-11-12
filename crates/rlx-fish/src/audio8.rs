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

//! `config.json` of the dual-AR checkpoints: S2-Pro and Audio8.
//! Audio8-TTS-Realistic-Air (`model_type: arktts`): S2's dual-AR design at
//! 0.6 B, 23 languages, zero-shot voice cloning.
//!
//! [`crate::S2Model`] runs it, [`crate::FishCodec`] reads its `codec.pth`, and
//! the prompt is S2's. What differs from S2 at this level is the generation
//! driver: the reference speaks the whole text in one pass inside a
//! 2048-row context, where S2 chunks a conversation, and its defaults come
//! from the checkpoint's `generation_config.json`.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use crate::config::S2Config;
use crate::prompt::{Row, TextEncoder};
use crate::s2::{Conversation, Reference, S2Model, frames_to_codes};
use crate::sampling::{Rng, SampleParams};

/// Settings of one utterance.
#[derive(Clone, Debug)]
pub struct SynthOptions {
    pub params: SampleParams,
    pub seed: u64,
    /// Frame budget, cut further to what the context has left.
    pub max_new_tokens: usize,
}

impl SynthOptions {
    /// The checkpoint's `generation_config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("generation_config.json");
        let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        Self::from_value(&serde_json::from_slice(&raw)?)
    }

    fn from_value(v: &Value) -> Result<Self> {
        let number = |key: &str| {
            v.get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("generation config: missing number {key}"))
        };
        Ok(Self {
            params: SampleParams {
                temperature: number("temperature")? as f32,
                top_p: number("top_p")? as f32,
                top_k: number("top_k")? as usize,
            },
            seed: 0,
            max_new_tokens: number("max_new_tokens")? as usize,
        })
    }
}

/// `ArkttsProcessor._clean_text`: trimmed, whitespace runs collapsed to one
/// space.
fn clean_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The prompt `ArkttsProcessor` and `_prepare_prompt` build for `text`, with
/// an optional reference voice.
pub fn prompt_rows(
    enc: &dyn TextEncoder,
    cfg: &S2Config,
    text: &str,
    reference: Option<&Reference>,
) -> Result<Vec<Row>> {
    let target = clean_text(text);
    ensure!(!target.is_empty(), "text must not be empty");
    let references: Vec<Reference> = reference
        .map(|r| {
            let cleaned = clean_text(&r.text);
            ensure!(
                !cleaned.is_empty(),
                "a reference voice needs its transcript"
            );
            Ok(Reference {
                text: cleaned,
                codes: r.codes.clone(),
            })
        })
        .transpose()?
        .into_iter()
        .collect();
    let mut conversation = Conversation::new(&references)?;
    conversation.push_user(&target);
    conversation.prompt_rows(enc, cfg)
}

/// The result of [`synthesize_codes`].
#[derive(Clone, Debug)]
pub struct Synthesis {
    /// `[num_codebooks][frames]`.
    pub codes: Vec<Vec<u32>>,
    /// The model chose to stop; `false` means the frame budget ran out first.
    pub finished: bool,
}

/// Generate `text` in one pass, as `ArkttsModel.generate` does.
pub fn synthesize_codes(
    model: &mut S2Model,
    enc: &dyn TextEncoder,
    text: &str,
    reference: Option<&Reference>,
    opts: &SynthOptions,
) -> Result<Synthesis> {
    synthesize_codes_with(model, enc, text, reference, opts, |_| {})
}

/// [`synthesize_codes`] with a callback per frame as it is made (its row
/// `[main, code 0, …]`): what playing while generating reads.
pub fn synthesize_codes_with(
    model: &mut S2Model,
    enc: &dyn TextEncoder,
    text: &str,
    reference: Option<&Reference>,
    opts: &SynthOptions,
    mut on_frame: impl FnMut(&[u32]),
) -> Result<Synthesis> {
    let rows = prompt_rows(enc, &model.cfg, text, reference)?;
    let context = model.cfg.max_seq_len;
    ensure!(
        rows.len() < context,
        "prompt length {} must be smaller than {context}",
        rows.len()
    );
    let budget = opts.max_new_tokens.min(context - rows.len());
    let mut rng = Rng::new(opts.seed);
    let frames = model.generate_with(&rows, budget, opts.params, &mut rng, |f| {
        let row: Vec<u32> = std::iter::once(f.main)
            .chain(f.codes.iter().copied())
            .collect();
        on_frame(&row);
    })?;
    Ok(Synthesis {
        codes: frames_to_codes(&frames, model.cfg.num_codebooks),
        finished: frames.len() < budget,
    })
}

// ── several sentences at once ───────────────────────────────────────────

/// The shortest sentence [`synthesize_batch`] generates on its own: a lone
/// "Hi." is too little text to carry a voice.
pub const SENTENCE_MIN_CHARS: usize = 24;

/// `text` cut after sentence ends (`.` `!` `?` `;` followed by whitespace,
/// or a CJK `。` `！` `？` `；`), pieces shorter than `min_chars` merged into
/// the next one ([`SENTENCE_MIN_CHARS`] is what synthesis uses).
pub fn split_sentences(text: &str, min_chars: usize) -> Vec<String> {
    let text = clean_text(text);
    let mut pieces: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        let end = matches!(c, '。' | '！' | '？' | '；')
            || (matches!(c, '.' | '!' | '?' | ';')
                && chars.peek().is_none_or(|n| n.is_whitespace()));
        if end && cur.trim().chars().count() >= min_chars {
            pieces.push(cur.trim().to_string());
            cur.clear();
        }
    }
    let rest = cur.trim();
    if !rest.is_empty() {
        match pieces.last_mut() {
            Some(last) if rest.chars().count() < min_chars => {
                last.push(' ');
                last.push_str(rest);
            }
            _ => pieces.push(rest.to_string()),
        }
    }
    pieces
}

/// [`synthesize_codes`] for a longer text, `batch` sentences at a time
/// ([`S2Model::generate_batch`]): one decode step serves the whole batch, so
/// the text takes about as many steps as its longest sentences.
///
/// Every sentence speaks in one voice: with `reference` that voice; without,
/// the first sentence is generated alone and becomes the reference of the
/// rest. Sentence `i` samples from seed `opts.seed + i`. The codes come back
/// in text order.
pub fn synthesize_batch(
    model: &mut S2Model,
    enc: &dyn TextEncoder,
    text: &str,
    reference: Option<&Reference>,
    opts: &SynthOptions,
    batch: usize,
) -> Result<Synthesis> {
    let sentences = split_sentences(text, SENTENCE_MIN_CHARS);
    ensure!(!sentences.is_empty(), "text must not be empty");
    let nc = model.cfg.num_codebooks;
    let mut codes: Vec<Vec<u32>> = vec![Vec::new(); nc];
    let mut finished = true;
    let append = |more: &[Vec<u32>], codes: &mut Vec<Vec<u32>>| {
        for (all, more) in codes.iter_mut().zip(more) {
            all.extend(more);
        }
    };
    let (anchor, rest) = match reference {
        Some(r) => (r.clone(), &sentences[..]),
        None => {
            let first = synthesize_codes(model, enc, &sentences[0], None, opts)?;
            finished &= first.finished;
            append(&first.codes, &mut codes);
            let anchor = Reference {
                text: sentences[0].clone(),
                codes: first.codes,
            };
            (anchor, &sentences[1..])
        }
    };
    let skip = sentences.len() - rest.len();
    let context = model.cfg.max_seq_len;
    for (c, chunk) in rest.chunks(batch.max(1)).enumerate() {
        let prompts: Vec<Vec<Row>> = chunk
            .iter()
            .map(|t| prompt_rows(enc, &model.cfg, t, Some(&anchor)))
            .collect::<Result<_>>()?;
        let longest = prompts.iter().map(Vec::len).max().expect("a sentence");
        ensure!(
            longest < context,
            "prompt length {longest} must be smaller than {context}"
        );
        let budget = opts.max_new_tokens.min(context - 1 - longest);
        let first = skip + c * batch.max(1);
        let seeds: Vec<u64> = (0..chunk.len())
            .map(|i| opts.seed + (first + i) as u64)
            .collect();
        let frames = model.generate_batch(&prompts, budget, opts.params, &seeds)?;
        for f in frames {
            finished &= f.len() < budget;
            append(&frames_to_codes(&f, nc), &mut codes);
        }
    }
    Ok(Synthesis { codes, finished })
}

// ── a model split into pipeline stages ──────────────────────────────────
//
// A browser holds Audio8 across Web Workers, each with its own address
// space: slow-layer stages ([`SlowStage`]) and the fast decoder with the
// sampler ([`FrameMaker`]). A frame goes row → first stage → … → last stage
// (hidden state + band logits) → frame maker (main token, its codes) → the
// next row. The same seed draws the same frames as [`S2Model::generate`]:
// one RNG takes the main-token draws and then the codes', in that order.

/// What a slow stage reads: prompt or frame rows (the first stage, which
/// embeds them), or the previous stage's output rows `[n, dim]`.
pub enum StageInput<'a> {
    Rows(&'a [Row]),
    Hidden(&'a [f32]),
}

/// One run of slow layers ([`crate::s2::Stage`]) and its KV cache.
pub struct SlowStage {
    pub model: S2Model,
    state: Option<crate::s2::SlowState>,
    /// Sequences prefilled for the next batch ([`Self::queue`]).
    queued: Vec<crate::s2::SlowState>,
    batch: Option<crate::s2::BatchState>,
}

impl SlowStage {
    pub fn new(model: S2Model) -> Self {
        Self {
            model,
            state: None,
            queued: Vec::new(),
            batch: None,
        }
    }

    fn input(&self, input: StageInput<'_>) -> Result<(Vec<f32>, usize)> {
        let dim = self.model.cfg.slow.dim;
        match input {
            StageInput::Rows(rows) => Ok((self.model.embed_rows(rows)?, rows.len())),
            StageInput::Hidden(h) => {
                ensure!(
                    h.len().is_multiple_of(dim),
                    "{} values in rows of {dim}",
                    h.len()
                );
                Ok((h.to_vec(), h.len() / dim))
            }
        }
    }

    /// Start an utterance; the cache holds `cap` rows. Returns this stage's
    /// output rows (on the last stage, the final-norm rows: see
    /// [`Self::hidden`] and [`Self::logits`] for the newest one).
    pub fn prefill(&mut self, input: StageInput<'_>, cap: usize) -> Result<Vec<f32>> {
        let (x, t) = self.input(input)?;
        let (state, out) = self.model.prefill_embeds(x, t, cap)?;
        self.state = Some(state);
        Ok(out)
    }

    /// One row more. Returns this stage's output row.
    pub fn step(&mut self, input: StageInput<'_>) -> Result<Vec<f32>> {
        let (x, t) = self.input(input)?;
        ensure!(t == 1, "a step takes one row, got {t}");
        let state = self.state.as_mut().context("step before prefill")?;
        self.model.step(state, &x)?;
        Ok(state.hidden.clone())
    }

    /// [`Self::prefill`] with async runs (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn prefill_async(&mut self, input: StageInput<'_>, cap: usize) -> Result<Vec<f32>> {
        let (x, t) = self.input(input)?;
        let (state, out) = self.model.prefill_embeds_async(x, t, cap).await?;
        self.state = Some(state);
        Ok(out)
    }

    /// [`Self::step`] with async runs.
    #[cfg(target_family = "wasm")]
    pub async fn step_async(&mut self, input: StageInput<'_>) -> Result<Vec<f32>> {
        let (x, t) = self.input(input)?;
        ensure!(t == 1, "a step takes one row, got {t}");
        let state = self.state.as_mut().context("step before prefill")?;
        self.model.step_async(state, &x).await?;
        Ok(state.hidden.clone())
    }

    // Several sequences: `queue` each prompt (prefilled alone, in order),
    // `join`, then `step_batch` with one row per sequence. Every stage of a
    // pipeline queues the same prompts in the same order.

    /// Prefill one sequence of the next batch. Returns this stage's output
    /// rows, as [`Self::prefill`] does.
    pub fn queue(&mut self, input: StageInput<'_>, cap: usize) -> Result<Vec<f32>> {
        let (x, t) = self.input(input)?;
        let (state, out) = self.model.prefill_embeds(x, t, cap)?;
        self.queued.push(state);
        Ok(out)
    }

    /// [`Self::queue`] with async runs (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn queue_async(&mut self, input: StageInput<'_>, cap: usize) -> Result<Vec<f32>> {
        let (x, t) = self.input(input)?;
        let (state, out) = self.model.prefill_embeds_async(x, t, cap).await?;
        self.queued.push(state);
        Ok(out)
    }

    /// Decode the queued sequences together from here on.
    pub fn join(&mut self) -> Result<()> {
        let states = std::mem::take(&mut self.queued);
        self.batch = Some(self.model.batch_join(states)?);
        Ok(())
    }

    /// One row more for every sequence of the batch (`Rows` with one row per
    /// sequence, or the previous stage's `[n, dim]`). Returns this stage's
    /// output rows `[n, dim]`.
    pub fn step_batch(&mut self, input: StageInput<'_>) -> Result<Vec<f32>> {
        let (x, _) = self.input(input)?;
        let st = self.batch.as_mut().context("step_batch before join")?;
        self.model.batch_step(st, &x)?;
        Ok(st.hidden.clone())
    }

    /// [`Self::step_batch`] with async runs.
    #[cfg(target_family = "wasm")]
    pub async fn step_batch_async(&mut self, input: StageInput<'_>) -> Result<Vec<f32>> {
        let (x, _) = self.input(input)?;
        let st = self.batch.as_mut().context("step_batch before join")?;
        self.model.batch_step_async(st, &x).await?;
        Ok(st.hidden.clone())
    }

    /// The batch's newest final-norm rows `[n, dim]` (last stage).
    pub fn batch_hidden(&self) -> &[f32] {
        self.batch.as_ref().map_or(&[], |b| &b.hidden)
    }

    /// The batch's newest band logits `[n, band]` (last stage).
    pub fn batch_logits(&self) -> &[f32] {
        self.batch.as_ref().map_or(&[], |b| &b.logits)
    }

    /// The newest row's final-norm hidden state (last stage).
    pub fn hidden(&self) -> &[f32] {
        self.state.as_ref().map_or(&[], |s| &s.hidden)
    }

    /// The newest row's band logits (last stage; empty elsewhere).
    pub fn logits(&self) -> &[f32] {
        self.state.as_ref().map_or(&[], |s| &s.logits)
    }
}

/// The capacity [`S2Model::generate`] gives the cache for a `prompt`-row
/// prompt and `max_new` frames.
pub fn cache_capacity(cfg: &S2Config, prompt: usize, max_new: usize) -> usize {
    let bucket = crate::s2::DECODE_BUCKET;
    ((prompt + max_new).div_ceil(bucket) * bucket).min(cfg.max_seq_len)
}

/// The fast decoder and the sampler: band logits and the hidden state in, the
/// frame's row `[main, code 0, …]` out.
pub struct FrameMaker {
    pub model: S2Model,
    sampler: crate::s2::MainSampler,
    params: SampleParams,
    rng: Rng,
    /// The samplers of a batch ([`Self::restart_batch`]).
    batch: Option<crate::s2::BatchSampler>,
}

impl FrameMaker {
    /// `model` needs only the fast stack's tensors (a stage of no slow layers).
    pub fn new(model: S2Model, params: SampleParams, seed: u64) -> Self {
        let sampler = crate::s2::MainSampler::new(&model.cfg, params);
        Self {
            model,
            sampler,
            params,
            rng: Rng::new(seed),
            batch: None,
        }
    }

    /// Start over (a new utterance) from `seed`.
    pub fn restart(&mut self, params: SampleParams, seed: u64) {
        self.sampler = crate::s2::MainSampler::new(&self.model.cfg, params);
        self.params = params;
        self.rng = Rng::new(seed);
        self.batch = None;
    }

    /// Start a batch: sequence `i` sampled from `seeds[i]`, as it would be
    /// alone.
    pub fn restart_batch(&mut self, params: SampleParams, seeds: &[u64]) {
        self.batch = Some(crate::s2::BatchSampler::new(&self.model.cfg, params, seeds));
    }

    /// The next row of every sequence of the batch, from the last stage's
    /// rows and band logits; `None` once all have ended. Rows of ended
    /// sequences are filler — check [`Self::ended`].
    pub fn next_batch(&mut self, hidden: &[f32], logits: &[f32]) -> Result<Option<Vec<Row>>> {
        let sampler = self
            .batch
            .as_mut()
            .context("next_batch before restart_batch")?;
        let rows = sampler.frame(&mut self.model, hidden, logits)?;
        sampler.close();
        Ok(rows)
    }

    /// [`Self::next_batch`] with async runs (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn next_batch_async(
        &mut self,
        hidden: &[f32],
        logits: &[f32],
    ) -> Result<Option<Vec<Row>>> {
        let sampler = self
            .batch
            .as_mut()
            .context("next_batch before restart_batch")?;
        let rows = sampler.frame_async(&mut self.model, hidden, logits).await?;
        sampler.close();
        Ok(rows)
    }

    /// Whether sequence `i` of the batch has ended.
    pub fn ended(&self, i: usize) -> bool {
        self.batch.as_ref().is_some_and(|b| b.ended(i))
    }

    /// The next frame's row, or `None` at `<|im_end|>`.
    pub fn next(&mut self, hidden: &[f32], logits: &[f32]) -> Result<Option<Row>> {
        let main = self.sampler.choose(logits, &mut self.rng);
        if main == self.model.cfg.eos {
            return Ok(None);
        }
        let codes = self
            .model
            .fast_frame_fused(hidden, main, self.params, &mut self.rng)?;
        self.sampler.accept(main);
        let mut row = vec![main];
        row.extend(codes);
        Ok(Some(row))
    }

    /// [`Self::next`] with async runs (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn next_async(&mut self, hidden: &[f32], logits: &[f32]) -> Result<Option<Row>> {
        let main = self.sampler.choose(logits, &mut self.rng);
        if main == self.model.cfg.eos {
            return Ok(None);
        }
        let codes = self
            .model
            .fast_frame_fused_async(hidden, main, self.params, &mut self.rng)
            .await?;
        self.sampler.accept(main);
        let mut row = vec![main];
        row.extend(codes);
        Ok(Some(row))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{BTreeMap, BTreeSet};

    use safetensors::{Dtype, tensor::TensorView};

    use super::*;
    use crate::S2Options;
    use rlx_runtime::Device;

    /// The published `config.json` and tensor index, verbatim.
    const CONFIG: &str = include_str!("../tests/fixtures/audio8_config.json");
    const MANIFEST: &str = include_str!("../tests/fixtures/audio8_manifest.json");

    fn real_config() -> S2Config {
        S2Config::from_value(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    /// `layers.3.ffn_norm.weight` → `layers.N.ffn_norm.weight`.
    fn pattern(key: &str) -> String {
        key.split('.')
            .map(|p| {
                if p.bytes().all(|b| b.is_ascii_digit()) {
                    "N"
                } else {
                    p
                }
            })
            .collect::<Vec<_>>()
            .join(".")
    }

    /// Records each string it is handed: one id per call, so a row's id says
    /// which literal produced it.
    #[derive(Default)]
    struct Recorder(RefCell<Vec<String>>);

    impl TextEncoder for Recorder {
        fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
            let mut parts = self.0.borrow_mut();
            parts.push(text.to_string());
            Ok(vec![parts.len() as u32])
        }

        fn special_id(&self, _token: &str) -> Option<u32> {
            None
        }
    }

    /// The literals of `ArkttsProcessor._prompt_segments`, in order.
    const SUFFIX: [&str; 3] = ["<|im_end|>\n", "<|im_start|>user\n", "Hello there."];
    const TAIL: [&str; 2] = ["<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"];

    #[test]
    fn prompt_without_a_reference_is_the_processors() {
        let enc = Recorder::default();
        let rows = prompt_rows(&enc, &real_config(), "  Hello \n  there.  ", None).unwrap();
        let mut want = vec![
            "<|im_start|>system\n".to_string(),
            "convert the provided text to speech".to_string(),
        ];
        want.extend(SUFFIX.map(String::from));
        want.extend(TAIL.map(String::from));
        assert_eq!(*enc.0.borrow(), want);
        assert_eq!(rows.len(), want.len());
        assert!(rows.iter().all(|r| r.len() == 11 && r[1..] == [0; 10]));
    }

    #[test]
    fn prompt_with_a_reference_is_the_processors() {
        let cfg = real_config();
        let enc = Recorder::default();
        let codes: Vec<Vec<u32>> = (0..10).map(|i| vec![i, i + 100]).collect();
        let reference = Reference {
            text: " The  reference\ttranscript. ".into(),
            codes,
        };
        let rows = prompt_rows(&enc, &cfg, "Hello there.", Some(&reference)).unwrap();
        let mut want = vec![
            "<|im_start|>system\n".to_string(),
            "convert the provided text to speech reference to the following:\n\nText:\n"
                .to_string(),
            "<|speaker:0|>The reference transcript.".to_string(),
            "\n\nSpeech:\n".to_string(),
        ];
        want.extend(SUFFIX.map(String::from));
        want.extend(TAIL.map(String::from));
        assert_eq!(*enc.0.borrow(), want);
        // The two audio rows sit right after the prefix.
        assert_eq!(rows.len(), want.len() + 2);
        assert_eq!(rows[4][0], cfg.semantic_begin);
        assert_eq!(rows[5][0], cfg.semantic_begin + 100);
        assert_eq!(rows[5][1..], (0..10).map(|i| i + 100).collect::<Vec<_>>());
    }

    #[test]
    fn a_tagged_reference_keeps_its_tag_and_empty_text_is_refused() {
        let cfg = real_config();
        let enc = Recorder::default();
        let reference = Reference {
            text: "<|speaker:3|>Tagged.".into(),
            codes: vec![vec![1]; 10],
        };
        prompt_rows(&enc, &cfg, "Hi.", Some(&reference)).unwrap();
        assert_eq!(enc.0.borrow()[2], "<|speaker:3|>Tagged.");
        assert!(prompt_rows(&enc, &cfg, " \n ", None).is_err());
        let silent = Reference {
            text: "  ".into(),
            codes: vec![vec![1]; 10],
        };
        assert!(prompt_rows(&enc, &cfg, "Hi.", Some(&silent)).is_err());
    }

    #[test]
    fn generation_defaults_come_from_the_file() {
        let v = serde_json::json!({
            "temperature": 0.7, "top_p": 0.9, "top_k": 50, "max_new_tokens": 512
        });
        let o = SynthOptions::from_value(&v).unwrap();
        assert_eq!(o.params.top_k, 50);
        assert_eq!(o.max_new_tokens, 512);
        assert!((o.params.top_p - 0.9).abs() < 1e-6);
    }

    // ── a tiny checkpoint with Audio8's names and switches ──────────────

    const DIM: usize = 32;
    const FFN: usize = 48;
    const HEAD: usize = 8;
    const HEADS: usize = 4;
    const KV: usize = 2;
    const CODES: usize = 3;
    const BOOK: usize = 16;
    const VOCAB: usize = 64;
    const SEMANTIC: usize = 40;
    const EOS: usize = 2;

    /// Deterministic values in `[-scale, scale)`, different per tensor.
    fn noise(seed: &str, n: usize, scale: f32) -> Vec<f32> {
        let mut state = seed.bytes().fold(0x9E37_79B9u64, |h, b| {
            h.wrapping_mul(31).wrapping_add(b as u64)
        });
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn tiny_tensors(bias_scale: f32) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        let mut t = BTreeMap::new();
        let mut put = |key: String, shape: Vec<usize>, scale: f32| {
            let n = shape.iter().product();
            let data = noise(&key, n, scale);
            t.insert(key, (data, shape));
        };
        let qkv = (HEADS + 2 * KV) * HEAD;
        put("embeddings.weight".into(), vec![VOCAB, DIM], 0.5);
        put(
            "codebook_embeddings.weight".into(),
            vec![BOOK * CODES, DIM],
            0.5,
        );
        put("fast_embeddings.weight".into(), vec![BOOK, DIM], 0.5);
        put("fast_output.weight".into(), vec![BOOK, DIM], 0.2);
        for (prefix, layers, bias) in [("", 2, true), ("fast_", 1, false)] {
            for l in 0..layers {
                let p = format!("{prefix}layers.{l}.");
                put(format!("{p}attention.wqkv.weight"), vec![qkv, DIM], 0.2);
                if bias {
                    put(format!("{p}attention.wqkv.bias"), vec![qkv], bias_scale);
                }
                put(
                    format!("{p}attention.wo.weight"),
                    vec![DIM, HEADS * HEAD],
                    0.2,
                );
                put(format!("{p}feed_forward.w1.weight"), vec![FFN, DIM], 0.2);
                put(format!("{p}feed_forward.w3.weight"), vec![FFN, DIM], 0.2);
                put(format!("{p}feed_forward.w2.weight"), vec![DIM, FFN], 0.2);
                put(format!("{p}attention_norm.weight"), vec![DIM], 0.1);
                put(format!("{p}ffn_norm.weight"), vec![DIM], 0.1);
            }
            put(format!("{prefix}norm.weight"), vec![DIM], 0.1);
        }
        // Norm gains sit around 1.
        for (key, (data, _)) in t.iter_mut() {
            if key.contains("norm.weight") {
                data.iter_mut().for_each(|v| *v += 1.0);
            }
        }
        t
    }

    fn write_checkpoint(dir: &Path, tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) {
        let mut config: Value = serde_json::from_str(CONFIG).unwrap();
        for (key, value) in [
            ("vocab_size", VOCAB),
            ("dim", DIM),
            ("fast_dim", DIM),
            ("n_layer", 2),
            ("n_fast_layer", 1),
            ("n_head", HEADS),
            ("fast_n_head", HEADS),
            ("n_local_heads", KV),
            ("fast_n_local_heads", KV),
            ("head_dim", HEAD),
            ("fast_head_dim", HEAD),
            ("intermediate_size", FFN),
            ("fast_intermediate_size", FFN),
            ("codebook_size", BOOK),
            ("num_codebooks", CODES),
            ("semantic_begin_id", SEMANTIC),
            ("semantic_end_id", SEMANTIC + BOOK - 1),
            ("eos_token_id", EOS),
            ("max_seq_len", 512),
        ] {
            config[key] = value.into();
        }
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let bytes: BTreeMap<&String, Vec<u8>> = tensors
            .iter()
            .map(|(k, (d, _))| (k, d.iter().flat_map(|v| v.to_le_bytes()).collect()))
            .collect();
        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .map(|(k, (_, shape))| {
                let view = TensorView::new(Dtype::F32, shape.clone(), &bytes[k]).unwrap();
                (k.clone(), view)
            })
            .collect();
        safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    }

    fn tiny_model(bias_scale: f32) -> (tempfile::TempDir, S2Model) {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &tiny_tensors(bias_scale));
        let model = S2Model::open(dir.path(), Device::Cpu, S2Options::default()).unwrap();
        (dir, model)
    }

    /// Text rows, then audio rows.
    fn tiny_rows() -> Vec<Row> {
        let mut rows: Vec<Row> = (3..9).map(|t| vec![t, 0, 0, 0]).collect();
        for t in 0..5u32 {
            let c = [t, (t * 3 + 1) % 16, (t * 5 + 2) % 16];
            rows.push(vec![SEMANTIC as u32 + c[0], c[0], c[1], c[2]]);
        }
        rows
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn every_real_tensor_is_read_and_every_read_tensor_is_real() {
        let manifest: BTreeMap<String, Value> = serde_json::from_str(MANIFEST).unwrap();
        let shipped: BTreeSet<String> = manifest.keys().map(|k| pattern(k)).collect();

        let (_dir, model) = tiny_model(0.3);
        let mut read: BTreeSet<String> = model
            .host_tensor_keys()
            .iter()
            .map(|k| pattern(k))
            .collect();
        let graphs = [
            model.prefill_builder(64).0,
            model.decode_builder(256).0,
            model.fast_builder().0,
        ];
        for b in &graphs {
            for p in &b.params {
                if let crate::nn::Init::Tensor { key, .. } = &p.init {
                    read.insert(pattern(key));
                }
            }
        }
        assert_eq!(read, shipped);
    }

    #[test]
    fn real_shapes_agree_with_the_config() {
        let cfg = real_config();
        let manifest: BTreeMap<String, Value> = serde_json::from_str(MANIFEST).unwrap();
        let shape = |key: &str| -> Vec<usize> {
            manifest[key]["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        let (s, f) = (&cfg.slow, &cfg.fast);
        let wqkv = (s.n_head + 2 * s.n_kv) * s.head_dim;
        assert_eq!(shape("layers.23.attention.wqkv.weight"), [wqkv, s.dim]);
        assert_eq!(shape("layers.23.attention.wqkv.bias"), [wqkv]);
        assert_eq!(shape("layers.0.feed_forward.w2.weight"), [s.dim, s.ffn]);
        assert_eq!(shape("embeddings.weight"), [cfg.vocab_size, s.dim]);
        assert_eq!(
            shape("codebook_embeddings.weight"),
            [cfg.codebook_size * cfg.num_codebooks, s.dim]
        );
        assert_eq!(shape("fast_embeddings.weight"), [cfg.codebook_size, f.dim]);
        assert_eq!(shape("fast_output.weight"), [cfg.codebook_size, f.dim]);
        assert_eq!(
            shape("fast_layers.3.feed_forward.w1.weight"),
            [f.ffn, f.dim]
        );
        assert!(!manifest.contains_key("layers.24.ffn_norm.weight"));
        assert!(!manifest.contains_key("fast_layers.4.ffn_norm.weight"));
        assert!(!manifest.contains_key("fast_layers.0.attention.wqkv.bias"));
    }

    #[test]
    fn an_audio_row_embeds_as_the_plain_sum() {
        let (_dir, model) = tiny_model(0.3);
        let t = tiny_tensors(0.3);
        let (table, book) = (
            &t["embeddings.weight"].0,
            &t["codebook_embeddings.weight"].0,
        );
        let (main, codes) = (SEMANTIC + 4, [4usize, 7, 9]);
        let row: Row = std::iter::once(main as u32)
            .chain(codes.iter().map(|&c| c as u32))
            .collect();
        let mut want = table[main * DIM..(main + 1) * DIM].to_vec();
        for (i, c) in codes.iter().enumerate() {
            let at = (i * BOOK + c) * DIM;
            for (w, v) in want.iter_mut().zip(&book[at..at + DIM]) {
                *w += v;
            }
        }
        let got = model.embed_rows(&[row]).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-6);
        // A text row carries no codebook terms.
        let text = model.embed_rows(&[vec![5, 1, 2, 3]]).unwrap();
        assert_eq!(text, table[5 * DIM..6 * DIM]);
    }

    #[test]
    fn decode_steps_match_one_prefill() {
        let (_dir, mut model) = tiny_model(0.3);
        let rows = tiny_rows();
        let split = 7;
        let whole = model.prefill(&rows, 64).unwrap();
        let mut stepped = model.prefill(&rows[..split], 64).unwrap();
        for row in &rows[split..] {
            let embed = model.embed_rows(std::slice::from_ref(row)).unwrap();
            model.step(&mut stepped, &embed).unwrap();
        }
        assert_eq!(stepped.len, rows.len());
        let d = max_abs_diff(&stepped.hidden, &whole.hidden);
        assert!(d < 1e-4, "hidden differs by {d}");
        let d = max_abs_diff(&stepped.logits, &whole.logits);
        assert!(d < 1e-4, "logits differ by {d}");
    }

    #[test]
    fn the_slow_qkv_bias_reaches_the_graph() {
        let rows = tiny_rows();
        let (_a, mut with) = tiny_model(0.3);
        let (_b, mut without) = tiny_model(0.0);
        let h1 = with.prefill(&rows, 64).unwrap().hidden;
        let h0 = without.prefill(&rows, 64).unwrap().hidden;
        assert!(max_abs_diff(&h1, &h0) > 1e-2);
    }

    #[cfg(feature = "export")]
    #[test]
    fn the_s2_exporter_refuses_audio8() {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &tiny_tensors(0.3));
        let out = dir.path().join("audio8.rlxp");
        let opts = rlx_core::export::ExportOptions::default();
        let err = crate::export::export_rlxp(dir.path(), &out, &opts).unwrap_err();
        assert!(err.to_string().contains("S2-Pro only"), "{err}");
        assert!(!out.exists());
    }

    /// Two slow stages and a frame maker, driven the way the browser's
    /// workers drive them, sample the frames `S2Model::generate` does.
    #[test]
    fn a_pipeline_of_stages_samples_what_generate_does() {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &tiny_tensors(0.3));
        let open = |stage: Option<crate::s2::Stage>| {
            S2Model::open(
                dir.path(),
                Device::Cpu,
                S2Options {
                    stage,
                    ..S2Options::default()
                },
            )
            .unwrap()
        };
        let stage = |layers, last| Some(crate::s2::Stage { layers, last });
        let params = SampleParams {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 8,
        };
        let (rows, max_new, seed) = (tiny_rows(), 40, 11);
        let want = open(None)
            .generate(&rows, max_new, params, &mut Rng::new(seed))
            .unwrap();

        let mut first = SlowStage::new(open(stage(0..1, false)));
        let mut last = SlowStage::new(open(stage(1..2, true)));
        let mut maker = FrameMaker::new(open(stage(2..2, false)), params, seed);
        let cap = cache_capacity(&first.model.cfg, rows.len(), max_new);
        let h = first.prefill(StageInput::Rows(&rows), cap).unwrap();
        last.prefill(StageInput::Hidden(&h), cap).unwrap();
        let mut got = Vec::new();
        for step in 0..max_new {
            let Some(row) = maker.next(last.hidden(), last.logits()).unwrap() else {
                break;
            };
            got.push(row.clone());
            if step + 1 < max_new {
                let h = first.step(StageInput::Rows(&[row])).unwrap();
                last.step(StageInput::Hidden(&h)).unwrap();
            }
        }
        let want: Vec<Row> = want
            .iter()
            .map(|f| {
                std::iter::once(f.main)
                    .chain(f.codes.iter().copied())
                    .collect()
            })
            .collect();
        assert!(want.len() > 3, "only {} frames", want.len());
        assert_eq!(got, want);
    }

    #[test]
    fn a_pipeline_of_stages_batches_what_generate_batch_does() {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &tiny_tensors(0.3));
        let open = |stage: Option<crate::s2::Stage>| {
            let opts = S2Options {
                stage,
                ..S2Options::default()
            };
            S2Model::open(dir.path(), Device::Cpu, opts).unwrap()
        };
        let stage = |layers, last| Some(crate::s2::Stage { layers, last });
        let params = SampleParams {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 8,
        };
        let rows = tiny_rows();
        let prompts: Vec<Vec<Row>> = vec![rows.clone(), rows[3..].to_vec(), rows[..10].to_vec()];
        let (max_new, seeds) = (30, [4u64, 9, 2]);
        let want = open(None)
            .generate_batch(&prompts, max_new, params, &seeds)
            .unwrap();

        let mut first = SlowStage::new(open(stage(0..1, false)));
        let mut last = SlowStage::new(open(stage(1..2, true)));
        let mut maker = FrameMaker::new(open(stage(2..2, false)), params, 0);
        maker.restart_batch(params, &seeds);
        let top = prompts.iter().map(Vec::len).max().unwrap();
        let cap = cache_capacity(&first.model.cfg, top, max_new);
        for p in &prompts {
            let h = first.queue(StageInput::Rows(p), cap).unwrap();
            last.queue(StageInput::Hidden(&h), cap).unwrap();
        }
        first.join().unwrap();
        last.join().unwrap();
        let mut got: Vec<Vec<Row>> = vec![Vec::new(); prompts.len()];
        for step in 0..max_new {
            let Some(rows) = maker
                .next_batch(last.batch_hidden(), last.batch_logits())
                .unwrap()
            else {
                break;
            };
            for (i, row) in rows.iter().enumerate().filter(|(i, _)| !maker.ended(*i)) {
                got[i].push(row.clone());
            }
            if step + 1 < max_new {
                let h = first.step_batch(StageInput::Rows(&rows)).unwrap();
                last.step_batch(StageInput::Hidden(&h)).unwrap();
            }
        }
        let want: Vec<Vec<Row>> = want
            .iter()
            .map(|fs| {
                fs.iter()
                    .map(|f| {
                        std::iter::once(f.main)
                            .chain(f.codes.iter().copied())
                            .collect()
                    })
                    .collect()
            })
            .collect();
        assert!(want.iter().all(|w| w.len() > 3));
        assert_eq!(got, want);
    }

    /// The fast decoder as one graph samples the codes the host loop does,
    /// from the same RNG: greedy, and at two sampling settings.
    #[test]
    fn the_fused_frame_samples_what_the_host_loop_does() {
        let (_dir, mut m) = tiny_model(0.3);
        let settings = [
            SampleParams::greedy(),
            SampleParams {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 8,
            },
            SampleParams {
                temperature: 1.3,
                top_p: 0.5,
                top_k: 16,
            },
        ];
        for params in settings {
            for seed in 0..6u64 {
                let hidden = noise(&format!("hidden{seed}"), DIM, 2.0);
                let main = SEMANTIC as u32 + (seed as u32 * 5) % BOOK as u32;
                let host = m
                    .fast_frame(&hidden, main, params, &mut Rng::new(seed), |_, _| {})
                    .unwrap();
                let fused = m
                    .fast_frame_fused(&hidden, main, params, &mut Rng::new(seed))
                    .unwrap();
                assert_eq!(fused, host, "{params:?} seed {seed}");
            }
        }
    }

    #[test]
    fn sentences_split_where_speech_pauses() {
        let got = split_sentences("It is 3.5 metres long. Is it? Yes! 好的。再见。 tail", 1);
        let want = [
            "It is 3.5 metres long.",
            "Is it?",
            "Yes!",
            "好的。",
            "再见。",
            "tail",
        ];
        assert_eq!(got, want);
        // A short piece joins the next one; a short tail joins the last.
        let got = split_sentences("Hi. This one is long enough. Ok.", 10);
        assert_eq!(got, ["Hi. This one is long enough. Ok."]);
        let got = split_sentences("This one is long enough. So is this one.", 10);
        assert_eq!(got, ["This one is long enough.", "So is this one."]);
    }

    #[test]
    fn a_batched_text_is_its_sentences_in_order() {
        let (_dir, mut m) = tiny_model(0.3);
        struct Ids;
        impl TextEncoder for Ids {
            fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
                Ok(text.bytes().map(|b| 3 + b as u32 % 30).collect())
            }
            fn special_id(&self, _token: &str) -> Option<u32> {
                None
            }
        }
        let opts = SynthOptions {
            params: SampleParams {
                temperature: 0.8,
                top_p: 0.9,
                top_k: 8,
            },
            seed: 5,
            max_new_tokens: 6,
        };
        let text =
            "The first sentence is here. A second one follows it. And then a third one to end it.";
        let sentences = split_sentences(text, SENTENCE_MIN_CHARS);
        assert_eq!(sentences.len(), 3);
        let together = synthesize_batch(&mut m, &Ids, text, None, &opts, 2).unwrap();
        // One by one: the first alone, the others with it as the reference.
        let first = synthesize_codes(&mut m, &Ids, &sentences[0], None, &opts).unwrap();
        let anchor = Reference {
            text: sentences[0].clone(),
            codes: first.codes.clone(),
        };
        let mut want = first.codes;
        for (i, t) in sentences[1..].iter().enumerate() {
            let o = SynthOptions {
                seed: opts.seed + 1 + i as u64,
                ..opts.clone()
            };
            let out = synthesize_codes(&mut m, &Ids, t, Some(&anchor), &o).unwrap();
            for (w, c) in want.iter_mut().zip(out.codes) {
                w.extend(c);
            }
        }
        assert_eq!(together.codes, want);
    }

    #[test]
    fn a_batch_samples_what_each_prompt_samples_alone() {
        let (_dir, mut m) = tiny_model(0.3);
        let rows = tiny_rows();
        // Different lengths, so the caches sit at different offsets.
        let prompts: Vec<Vec<Row>> = vec![rows.clone(), rows[2..].to_vec(), rows[..9].to_vec()];
        let settings = [
            SampleParams::greedy(),
            SampleParams {
                temperature: 0.8,
                top_p: 0.9,
                top_k: 8,
            },
        ];
        for params in settings {
            let seeds = [3u64, 11, 7];
            let batch = m.generate_batch(&prompts, 12, params, &seeds).unwrap();
            for (i, p) in prompts.iter().enumerate() {
                let alone = m.generate(p, 12, params, &mut Rng::new(seeds[i])).unwrap();
                assert_eq!(batch[i], alone, "{params:?} prompt {i}");
            }
        }
    }

    #[test]
    fn a_stage_refuses_to_generate_alone() {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path(), &tiny_tensors(0.3));
        let opts = S2Options {
            stage: Some(crate::s2::Stage {
                layers: 0..1,
                last: false,
            }),
            ..S2Options::default()
        };
        let mut m = S2Model::open(dir.path(), Device::Cpu, opts).unwrap();
        let err = m
            .generate(&tiny_rows(), 4, SampleParams::greedy(), &mut Rng::new(0))
            .unwrap_err();
        assert!(err.to_string().contains("pipeline stage"), "{err}");
    }

    #[test]
    fn a_frame_budget_of_the_remaining_context_is_reported_unfinished() {
        let (_dir, mut model) = tiny_model(0.3);
        struct Ids;
        impl TextEncoder for Ids {
            fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
                Ok(text.bytes().map(|b| 3 + b as u32 % 30).collect())
            }
            fn special_id(&self, _token: &str) -> Option<u32> {
                None
            }
        }
        let opts = SynthOptions {
            params: SampleParams::greedy(),
            seed: 0,
            max_new_tokens: 1 << 20,
        };
        let rows = prompt_rows(&Ids, &model.cfg, "Hi.", None).unwrap();
        let left = model.cfg.max_seq_len - rows.len();
        // Greedy on noise weights: whichever way it ends, the count is bounded
        // by the context and `finished` tells the two apart.
        let out = synthesize_codes(&mut model, &Ids, "Hi.", None, &opts).unwrap();
        assert_eq!(out.codes.len(), CODES);
        assert!(out.codes[0].len() <= left);
        assert_eq!(out.finished, out.codes[0].len() < left);

        let long = "x ".repeat(model.cfg.max_seq_len);
        assert!(synthesize_codes(&mut model, &Ids, &long, None, &opts).is_err());
    }
}
