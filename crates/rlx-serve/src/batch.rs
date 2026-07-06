// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Continuous batching: many concurrent requests' decode steps are folded into
//! one batched forward, so server throughput scales with the batch instead of
//! serializing requests.
//!
//! The scheduling itself reuses `rlx_runtime::paged_kv::BatchConstructor`
//! (decode-first, then prefill chunks, token-budgeted). This module adds the
//! per-sequence state machine — sampling, `logit_bias`, log-probs, EOS /
//! multi-token stop, incremental detok — and the [`crate::engine::Engine`]
//! wiring. The batched forward itself is behind [`BatchRunner`]; a real model
//! batches into one graph step, a mock drives the tests.

use crate::engine::{
    ChatTurn, Engine, FinishReason, GenRequest, ModelCard, StreamItem, top_logprobs,
};
use rlx_qwen3::Qwen3Generator;
use rlx_qwen3::sampling::{apply_logit_bias, sample_token_with_history};
use rlx_runtime::device_ext::supports_ragged_rope;
use rlx_runtime::kv_cache::LayerKvCache;
use rlx_runtime::lm::{LmRunner, SessionSnapshot};
use rlx_runtime::paged_kv::{BatchConstructor, BatchEntry, BatchKind};
use rlx_runtime::quantized_kv::{KvQuant, QuantizedKvCache};
use rlx_text::{ChatTemplate, TokenizerHandle, incremental_emit};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

/// Sequence identifier within the batcher.
pub type SeqId = u64;

/// The batched-forward seam. `forward_batch` runs **one** batched graph step
/// over the mixed prefill/decode `batch` and returns the last-position logits
/// per sequence (prefill → first-token logits; decode → next-token logits),
/// advancing each sequence's KV cache by its `input_tokens`. The production impl
/// batches into a single graph step; tests use a mock.
pub trait BatchRunner: Send {
    fn forward_batch(&mut self, batch: &[BatchEntry]) -> anyhow::Result<Vec<(SeqId, Vec<f32>)>>;
    /// Free a finished sequence's KV pages.
    fn release(&mut self, seq: SeqId);
    fn vocab_size(&self) -> usize;
}

/// A [`BatchRunner`] over a single [`LmRunner`] that time-slices the sequences
/// in a batch through one set of weights, swapping each sequence's KV cache in
/// and out via [`LmRunner::export_session`] / [`LmRunner::restore_session`].
///
/// This is the production-ready backend that works with any model family whose
/// runner supports session export/restore (e.g. `Qwen3Runner`): the scheduler
/// gets continuous-batching semantics — fair interleaving, no head-of-line
/// blocking, prompt-cache-style suffix prefill — today, without a fused
/// multi-sequence kernel. A future `BatchRunner` that folds the whole batch
/// into one graph step (shared matmuls) is a drop-in throughput upgrade behind
/// the same trait.
///
/// The forward cost is one per-sequence pass per entry (KV swap is a host-side
/// move), so throughput equals serialized decode plus scheduling fairness — the
/// latency win, not yet the batched-matmul throughput win.
pub struct RunnerBatchRunner {
    runner: Box<dyn LmRunner>,
    sessions: HashMap<SeqId, SessionSnapshot>,
    vocab: usize,
}

impl RunnerBatchRunner {
    /// Wrap a runner. Errors if it doesn't support session export (required to
    /// hold more than one in-flight sequence).
    pub fn new(runner: Box<dyn LmRunner>) -> anyhow::Result<Self> {
        if runner.export_session().is_none() {
            anyhow::bail!(
                "{}: runner does not support session export/restore, required for batching",
                runner.family()
            );
        }
        let vocab = runner.vocab_size();
        Ok(Self {
            runner,
            sessions: HashMap::new(),
            vocab,
        })
    }
}

impl BatchRunner for RunnerBatchRunner {
    fn forward_batch(&mut self, batch: &[BatchEntry]) -> anyhow::Result<Vec<(SeqId, Vec<f32>)>> {
        let mut out = Vec::with_capacity(batch.len());
        for e in batch {
            let logits = match e.kind {
                // Fresh prefill (first / whole chunk): a full prefill clears and
                // re-seeds the runner's cache from these tokens.
                BatchKind::Prefill if e.cached_len == 0 => {
                    self.runner.prefill_logits(&e.input_tokens)?
                }
                // Continuation chunk of a long prompt: prefill only the new
                // suffix on top of this sequence's snapshot.
                BatchKind::Prefill => {
                    let snap = self.sessions.get(&e.seq_id).ok_or_else(|| {
                        anyhow::anyhow!("no session for chunked prefill seq {}", e.seq_id)
                    })?;
                    let mut full = snap.tokens.clone();
                    full.extend_from_slice(&e.input_tokens);
                    let reuse = snap.tokens.len();
                    self.runner.prefill_logits_reusing(&full, snap, reuse)?
                }
                // Decode: swap this sequence's KV in, advance one token.
                BatchKind::Decode => {
                    let snap = self
                        .sessions
                        .get(&e.seq_id)
                        .ok_or_else(|| anyhow::anyhow!("no session for decode seq {}", e.seq_id))?;
                    self.runner.restore_session(snap);
                    self.runner.decode_logits(e.input_tokens[0])?
                }
            };
            // Snapshot the advanced session back for this sequence.
            if let Some(snap) = self.runner.export_session() {
                self.sessions.insert(e.seq_id, snap);
            }
            out.push((e.seq_id, logits));
        }
        Ok(out)
    }

    fn release(&mut self, seq: SeqId) {
        self.sessions.remove(&seq);
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

/// A [`BatchRunner`] that **fuses** decode steps into one batched forward for
/// real throughput: every in-flight sequence's decode runs through a single
/// [`Qwen3Generator::decode_batched_ragged`] call, so each weight matmul runs
/// once over the whole batch instead of once per sequence.
///
/// Ragged fusion means sequences at **different** cache lengths still share the
/// forward — each gets its own RoPE position and causal-mask row (the model
/// must be non-sliding, so absolute position equals cache length). This is the
/// general case: arbitrary concurrent requests with varied prompt lengths and
/// staggered arrivals all fuse, not just lockstep cohorts. Prefills run
/// individually (latency-bound, and each shapes its own graph).
///
/// Compared to [`RunnerBatchRunner`] (which time-slices one runner with no
/// shared matmuls), this multiplies decode throughput by the batch size.
/// Requires whole-prompt prefill — keep `batch_tokens >= max prompt`.
///
/// **KV-cache quantization.** With [`FusedBatchRunner::with_kv_quant`], each
/// sequence's KV history is stored block-quantized (q8_0 / q4_0 / q5_0 / f16)
/// instead of f32, cutting the server's persistent memory ~2–4× so many more
/// concurrent contexts fit. The decode graph still runs in f32 — the cache is
/// dequantized on read and only the **new** row is (re)quantized each step, so
/// old rows are never recompressed (no error compounding). Requires
/// `kv_dim % 32 == 0` for the q-formats (f16 works for any `kv_dim`).
pub struct FusedBatchRunner {
    generator: Qwen3Generator,
    sessions: HashMap<SeqId, SessionKv>,
    /// `None` ⇒ store KV in f32. `Some(scheme)` ⇒ store block-quantized.
    kv_quant: Option<KvQuant>,
    /// Whether the device's RoPE kernel supports per-token positions. When
    /// `false`, decodes are grouped by cache length and run through the
    /// uniform path (shared RoPE row — correct on every backend) instead of one
    /// ragged forward, so output stays correct at some loss of fusion.
    ragged: bool,
    kv_dim: usize,
    vocab: usize,
}

/// Per-sequence KV storage — plain f32 or block-quantized.
enum SessionKv {
    F32(LayerKvCache),
    Quant(QuantizedKvCache),
}

impl SessionKv {
    /// Materialize the full f32 cache for feeding the decode graph.
    fn to_layer_kv(&self) -> anyhow::Result<LayerKvCache> {
        match self {
            SessionKv::F32(c) => Ok(c.clone()),
            SessionKv::Quant(q) => {
                let mut layers_k = Vec::with_capacity(q.layers.len());
                let mut layers_v = Vec::with_capacity(q.layers.len());
                for l in &q.layers {
                    let (k, v) = l.read_all()?;
                    layers_k.push(k);
                    layers_v.push(v);
                }
                let layers_kv_base = vec![0; layers_k.len()];
                Ok(LayerKvCache {
                    past_len: q.past_len(),
                    layers_k,
                    layers_v,
                    layers_kv_base,
                })
            }
        }
    }
}

impl FusedBatchRunner {
    pub fn new(generator: Qwen3Generator) -> Self {
        Self::with_kv_quant(generator, None)
    }

    /// Build with optional KV-cache quantization (`None` keeps f32).
    pub fn with_kv_quant(generator: Qwen3Generator, kv_quant: Option<KvQuant>) -> Self {
        let cfg = generator.config();
        let vocab = cfg.vocab_size;
        let kv_dim = cfg.kv_proj_dim();
        let ragged = supports_ragged_rope(generator.device());
        Self {
            generator,
            sessions: HashMap::new(),
            kv_quant,
            ragged,
            kv_dim,
            vocab,
        }
    }

    /// Store a freshly-prefilled f32 cache, quantizing it whole if enabled.
    fn store_initial(&mut self, seq: SeqId, kv: LayerKvCache) -> anyhow::Result<()> {
        let session = match self.kv_quant {
            None => SessionKv::F32(kv),
            Some(scheme) => {
                let mut q = QuantizedKvCache::new(kv.layers_k.len(), self.kv_dim, scheme)?;
                for (l, layer) in q.layers.iter_mut().enumerate() {
                    layer.append_rows(&kv.layers_k[l], &kv.layers_v[l])?;
                }
                SessionKv::Quant(q)
            }
        };
        self.sessions.insert(seq, session);
        Ok(())
    }

    /// Advance a session with the cache returned by the decode graph. For
    /// quantized sessions, only the new (last) row is appended — the prior
    /// rows keep their original quantization.
    fn advance_session(&mut self, seq: SeqId, new_full: LayerKvCache) -> anyhow::Result<()> {
        let kv_dim = self.kv_dim;
        match self.sessions.get_mut(&seq) {
            Some(SessionKv::F32(c)) => *c = new_full,
            Some(SessionKv::Quant(q)) => {
                let start = new_full.past_len.saturating_sub(1) * kv_dim;
                for (l, layer) in q.layers.iter_mut().enumerate() {
                    let nk = &new_full.layers_k[l][start..start + kv_dim];
                    let nv = &new_full.layers_v[l][start..start + kv_dim];
                    layer.append_rows(nk, nv)?;
                }
            }
            None => anyhow::bail!("advance_session: missing seq {seq}"),
        }
        Ok(())
    }
}

impl BatchRunner for FusedBatchRunner {
    fn forward_batch(&mut self, batch: &[BatchEntry]) -> anyhow::Result<Vec<(SeqId, Vec<f32>)>> {
        let mut out = Vec::with_capacity(batch.len());

        // Prefills: whole-prompt only. Each seeds this sequence's KV.
        for e in batch.iter().filter(|e| e.kind == BatchKind::Prefill) {
            if e.cached_len != 0 {
                anyhow::bail!(
                    "FusedBatchRunner: chunked prefill unsupported (seq {}); raise batch_tokens",
                    e.seq_id
                );
            }
            let logits = self.generator.prefill_get_last_logits(&e.input_tokens)?;
            let (kv, _) = self
                .generator
                .export_cache()
                .ok_or_else(|| anyhow::anyhow!("prefill produced no cache"))?;
            self.store_initial(e.seq_id, kv)?;
            out.push((e.seq_id, logits));
        }

        // Decodes: fuse ALL of them — any cache lengths — into one ragged
        // forward. Each sequence's position/mask is derived from its cache.
        let decodes: Vec<&BatchEntry> = batch
            .iter()
            .filter(|e| e.kind == BatchKind::Decode)
            .collect();
        if !decodes.is_empty() {
            // Dequantize each session to an owned f32 cache for the graph.
            let mut f32_caches: Vec<LayerKvCache> = Vec::with_capacity(decodes.len());
            for e in &decodes {
                let s = self
                    .sessions
                    .get(&e.seq_id)
                    .ok_or_else(|| anyhow::anyhow!("no session for decode seq {}", e.seq_id))?;
                f32_caches.push(s.to_layer_kv()?);
            }

            // Per-decode (logits, advanced_kv), in `decodes` order.
            let results: Vec<(Vec<f32>, LayerKvCache)> = if self.ragged {
                // One fused forward for the whole batch — different cache
                // lengths share it via per-token RoPE + per-row mask.
                let items: Vec<(u32, &LayerKvCache)> = decodes
                    .iter()
                    .zip(&f32_caches)
                    .map(|(e, c)| (e.input_tokens[0], c))
                    .collect();
                self.generator.decode_batched_ragged(&items)?
            } else {
                // Backend lacks per-token RoPE: group by cache length and run
                // each group through the uniform path (shared RoPE row), then
                // reassemble in original order.
                let mut by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for (i, c) in f32_caches.iter().enumerate() {
                    by_len.entry(c.past_len).or_default().push(i);
                }
                let mut slots: Vec<Option<(Vec<f32>, LayerKvCache)>> =
                    (0..decodes.len()).map(|_| None).collect();
                for (past, idxs) in by_len {
                    let items: Vec<(u32, &LayerKvCache)> = idxs
                        .iter()
                        .map(|&i| (decodes[i].input_tokens[0], &f32_caches[i]))
                        .collect();
                    let group = self.generator.decode_batched_uniform(&items, past, past)?;
                    for (&i, r) in idxs.iter().zip(group) {
                        slots[i] = Some(r);
                    }
                }
                slots
                    .into_iter()
                    .map(|o| o.expect("every slot filled"))
                    .collect()
            };
            drop(f32_caches);

            for (e, (logits, new_kv)) in decodes.iter().zip(results) {
                self.advance_session(e.seq_id, new_kv)?;
                out.push((e.seq_id, logits));
            }
        }
        Ok(out)
    }

    fn release(&mut self, seq: SeqId) {
        self.sessions.remove(&seq);
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

struct SeqWork {
    req: GenRequest,
    prompt_len: usize,
    prefilled: usize,
    /// prompt + generated tokens (for the sampler history + detok).
    history: Vec<u32>,
    /// Byte offset into `decode(history)` already emitted (starts past the prompt).
    emitted: usize,
    /// Generated text so far (for multi-token stop matching).
    gen_text: String,
    completion: usize,
    tx: Sender<StreamItem>,
}

/// The continuous-batching scheduler: admit requests, fold their decode steps
/// into batched forwards, sample/stop/emit per sequence.
pub struct ContinuousBatcher {
    runner: Box<dyn BatchRunner>,
    tokenizer: Arc<TokenizerHandle>,
    constructor: BatchConstructor,
    decode_q: VecDeque<BatchEntry>,
    prefill_q: VecDeque<BatchEntry>,
    seqs: HashMap<SeqId, SeqWork>,
    next_id: SeqId,
}

impl ContinuousBatcher {
    pub fn new(
        runner: Box<dyn BatchRunner>,
        tokenizer: Arc<TokenizerHandle>,
        max_tokens_per_batch: usize,
        max_entries: usize,
    ) -> Self {
        Self {
            runner,
            tokenizer,
            constructor: BatchConstructor::new(max_tokens_per_batch, max_entries),
            decode_q: VecDeque::new(),
            prefill_q: VecDeque::new(),
            seqs: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn is_idle(&self) -> bool {
        self.seqs.is_empty() && self.decode_q.is_empty() && self.prefill_q.is_empty()
    }

    /// Admit a new request: enqueue its prompt for prefill.
    pub fn admit(&mut self, req: GenRequest, tx: Sender<StreamItem>) {
        if req.prompt_ids.is_empty() {
            let _ = tx.send(StreamItem::Error("empty prompt".into()));
            return;
        }
        let id = self.next_id;
        self.next_id += 1;
        let emitted = self
            .tokenizer
            .decode(&req.prompt_ids, true)
            .map(|s| s.len())
            .unwrap_or(0);
        self.prefill_q.push_back(BatchEntry {
            seq_id: id,
            kind: BatchKind::Prefill,
            input_tokens: req.prompt_ids.clone(),
            cached_len: 0,
        });
        self.seqs.insert(
            id,
            SeqWork {
                prompt_len: req.prompt_ids.len(),
                prefilled: 0,
                history: req.prompt_ids.clone(),
                emitted,
                gen_text: String::new(),
                completion: 0,
                req,
                tx,
            },
        );
    }

    /// Run one batched step. Returns `false` when there is no work.
    pub fn pump(&mut self) -> bool {
        let batch = self
            .constructor
            .build(&mut self.decode_q, &mut self.prefill_q);
        if batch.is_empty() {
            return false;
        }
        let logits = match self.runner.forward_batch(&batch) {
            Ok(l) => l.into_iter().collect::<HashMap<SeqId, Vec<f32>>>(),
            Err(e) => {
                for entry in &batch {
                    self.fail_seq(entry.seq_id, e.to_string());
                }
                return true;
            }
        };

        for entry in &batch {
            let sid = entry.seq_id;
            // A prefill chunk only yields a sampleable token once the whole
            // prompt is cached; decode entries always sample.
            if entry.kind == BatchKind::Prefill {
                if let Some(s) = self.seqs.get_mut(&sid) {
                    s.prefilled += entry.input_tokens.len();
                    if s.prefilled < s.prompt_len {
                        continue; // more prefill chunks pending
                    }
                }
            }
            let Some(mut row) = logits.get(&sid).cloned() else {
                continue;
            };
            self.advance_seq(sid, &mut row);
        }
        true
    }

    /// Sample the next token for sequence `sid` from `row`, emit it (or finish),
    /// and enqueue the next decode.
    fn advance_seq(&mut self, sid: SeqId, row: &mut [f32]) {
        let (tok, finish, item, more) = {
            let Some(s) = self.seqs.get_mut(&sid) else {
                return;
            };
            apply_logit_bias(row, &s.req.bias);
            let tok = sample_token_with_history(row, &s.req.opts, &s.history, s.completion as u64);

            if s.req.stop_ids.contains(&tok) {
                (tok, Some(FinishReason::Stop), None, false)
            } else {
                let logprob = s.req.want_logprobs.map(|k| top_logprobs(row, tok, k));
                s.history.push(tok);
                s.completion += 1;
                let (delta, new_emitted) =
                    incremental_emit(&self.tokenizer, &s.history, s.emitted, true)
                        .unwrap_or((String::new(), s.emitted));
                s.emitted = new_emitted;

                // Multi-token stop strings, matched on generated text.
                let candidate = format!("{}{delta}", s.gen_text);
                if let Some(cut) = crate::stop::first_stop(&candidate, &s.req.stop_strings) {
                    let visible = candidate[s.gen_text.len().min(cut)..cut].to_string();
                    let item = (!visible.is_empty()).then_some(StreamItem::Token {
                        id: tok,
                        text: visible,
                        logprob,
                    });
                    (tok, Some(FinishReason::Stop), item, false)
                } else {
                    s.gen_text = candidate;
                    let reached_len = s.completion >= s.req.max_tokens;
                    let item = Some(StreamItem::Token {
                        id: tok,
                        text: delta,
                        logprob,
                    });
                    let finish = reached_len.then_some(FinishReason::Length);
                    (tok, finish, item, !reached_len)
                }
            }
        };

        if let Some(it) = item {
            self.emit(sid, it);
        }
        match finish {
            Some(reason) => self.finish_seq(sid, reason),
            None if more => {
                // Enqueue the next decode step.
                if let Some(s) = self.seqs.get(&sid) {
                    self.decode_q.push_back(BatchEntry {
                        seq_id: sid,
                        kind: BatchKind::Decode,
                        input_tokens: vec![tok],
                        cached_len: (s.history.len() - 1) as u32,
                    });
                }
            }
            None => {}
        }
    }

    fn emit(&self, sid: SeqId, item: StreamItem) {
        if let Some(s) = self.seqs.get(&sid) {
            let _ = s.tx.send(item);
        }
    }

    fn finish_seq(&mut self, sid: SeqId, reason: FinishReason) {
        if let Some(s) = self.seqs.remove(&sid) {
            let _ = s.tx.send(StreamItem::Done {
                finish_reason: reason,
                prompt_tokens: s.prompt_len,
                completion_tokens: s.completion,
            });
        }
        self.runner.release(sid);
    }

    fn fail_seq(&mut self, sid: SeqId, msg: String) {
        if let Some(s) = self.seqs.remove(&sid) {
            let _ = s.tx.send(StreamItem::Error(msg));
        }
        self.runner.release(sid);
    }
}

/// An [`Engine`] backed by [`ContinuousBatcher`] on a background thread.
/// `run` enqueues the request and streams its per-sequence output back, while
/// the scheduler folds all in-flight requests into batched forwards.
pub struct BatchedEngine {
    req_tx: Sender<(GenRequest, Sender<StreamItem>)>,
    tokenizer: Arc<TokenizerHandle>,
    chat_template: Option<Arc<ChatTemplate>>,
    eos_ids: Vec<u32>,
    model_id: String,
}

impl BatchedEngine {
    pub fn new(
        runner: Box<dyn BatchRunner>,
        tokenizer: TokenizerHandle,
        chat_template: Option<ChatTemplate>,
        eos_ids: Vec<u32>,
        model_id: String,
        max_tokens_per_batch: usize,
        max_entries: usize,
    ) -> Self {
        let tokenizer = Arc::new(tokenizer);
        let (req_tx, req_rx) = channel::<(GenRequest, Sender<StreamItem>)>();
        let tok = tokenizer.clone();
        std::thread::spawn(move || {
            scheduler_loop(
                ContinuousBatcher::new(runner, tok, max_tokens_per_batch, max_entries),
                req_rx,
            );
        });
        Self {
            req_tx,
            tokenizer,
            chat_template: chat_template.map(Arc::new),
            eos_ids,
            model_id,
        }
    }
}

/// The background loop: admit pending requests, then pump until idle.
fn scheduler_loop(
    mut batcher: ContinuousBatcher,
    req_rx: Receiver<(GenRequest, Sender<StreamItem>)>,
) {
    loop {
        // Drain any waiting requests.
        while let Ok((req, tx)) = req_rx.try_recv() {
            batcher.admit(req, tx);
        }
        if batcher.is_idle() {
            // Block for the next request (or exit when all senders drop).
            match req_rx.recv() {
                Ok((req, tx)) => batcher.admit(req, tx),
                Err(_) => return,
            }
        }
        batcher.pump();
    }
}

impl Engine for BatchedEngine {
    fn model_cards(&self) -> Vec<ModelCard> {
        vec![ModelCard {
            id: self.model_id.clone(),
        }]
    }
    fn encode_chat(&self, turns: &[ChatTurn]) -> anyhow::Result<Vec<u32>> {
        let text = match &self.chat_template {
            Some(t) => {
                let msgs: Vec<rlx_text::ChatMessage> = turns
                    .iter()
                    .map(|m| rlx_text::ChatMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                    })
                    .collect();
                t.render(&msgs, true)?
            }
            None => turns
                .iter()
                .map(|m| format!("{}: {}\n", m.role, m.content))
                .collect(),
        };
        self.encode_text(&text)
    }
    fn encode_text(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        self.tokenizer.encode(text, true)
    }
    fn eos_ids(&self) -> Vec<u32> {
        self.eos_ids.clone()
    }
    fn decode_token(&self, id: u32) -> String {
        self.tokenizer.decode(&[id], false).unwrap_or_default()
    }
    fn run(&self, req: &GenRequest, emit: &mut dyn FnMut(StreamItem) -> bool) {
        let (tx, rx) = channel::<StreamItem>();
        if self.req_tx.send((req.clone(), tx)).is_err() {
            emit(StreamItem::Error("batcher stopped".into()));
            return;
        }
        for item in rx {
            let terminal = matches!(item, StreamItem::Done { .. } | StreamItem::Error(_));
            let keep = emit(item);
            if terminal || !keep {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_qwen3::SampleOpts;

    /// Drives each sequence through a fixed token list via one-hot logits,
    /// ending at a sentinel EOS. Records the max concurrent batch size seen.
    struct MockRunner {
        vocab: usize,
        seq_for: HashMap<SeqId, Vec<u32>>,
        pos: HashMap<SeqId, usize>,
        pub max_batch: usize,
    }
    impl MockRunner {
        fn onehot(&self, tok: u32) -> Vec<f32> {
            let mut v = vec![0.0; self.vocab];
            v[tok as usize] = 10.0;
            v
        }
    }
    impl BatchRunner for MockRunner {
        fn forward_batch(
            &mut self,
            batch: &[BatchEntry],
        ) -> anyhow::Result<Vec<(SeqId, Vec<f32>)>> {
            self.max_batch = self.max_batch.max(batch.len());
            let mut out = Vec::new();
            for e in batch {
                let p = self.pos.entry(e.seq_id).or_insert(0);
                let seq = &self.seq_for[&e.seq_id];
                let tok = seq[(*p).min(seq.len() - 1)];
                *p += 1;
                out.push((e.seq_id, self.onehot(tok)));
            }
            Ok(out)
        }
        fn release(&mut self, _seq: SeqId) {}
        fn vocab_size(&self) -> usize {
            self.vocab
        }
    }

    fn req(prompt: Vec<u32>, max: usize, eos: u32) -> GenRequest {
        GenRequest {
            prompt_ids: prompt,
            opts: SampleOpts::greedy(),
            bias: vec![],
            max_tokens: max,
            stop_ids: vec![eos],
            stop_strings: vec![],
            want_logprobs: None,
        }
    }

    #[test]
    fn batches_multiple_sequences_and_streams_each() {
        // Two requests; the mock makes seq A emit [5,6,EOS] and seq B [7,8,EOS].
        let vocab = 16;
        let eos = 9u32;
        let mut seq_for = HashMap::new();
        seq_for.insert(0u64, vec![5, 6, eos]);
        seq_for.insert(1u64, vec![7, 8, eos]);
        let runner = MockRunner {
            vocab,
            seq_for,
            pos: HashMap::new(),
            max_batch: 0,
        };

        // No real tokenizer in the test → decode just yields empty text, which
        // is fine: we assert on the token ids the scheduler samples.
        let tok = TokenizerHandle::from_raw(tokenizers::Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::default(),
        ));
        let mut batcher = ContinuousBatcher::new(Box::new(runner), Arc::new(tok), 256, 8);

        let (txa, rxa) = channel();
        let (txb, rxb) = channel();
        batcher.admit(req(vec![1, 2], 10, eos), txa);
        batcher.admit(req(vec![3, 4], 10, eos), txb);

        // Pump until idle.
        let mut steps = 0;
        while !batcher.is_idle() && steps < 100 {
            batcher.pump();
            steps += 1;
        }

        let collect = |rx: Receiver<StreamItem>| {
            let mut ids = Vec::new();
            let mut reason = None;
            while let Ok(it) = rx.try_recv() {
                match it {
                    StreamItem::Token { id, .. } => ids.push(id),
                    StreamItem::Done { finish_reason, .. } => reason = Some(finish_reason),
                    StreamItem::Error(e) => panic!("{e}"),
                }
            }
            (ids, reason)
        };
        let (ids_a, reason_a) = collect(rxa);
        let (ids_b, reason_b) = collect(rxb);

        assert_eq!(ids_a, vec![5, 6], "seq A tokens (EOS not emitted)");
        assert_eq!(ids_b, vec![7, 8], "seq B tokens");
        assert_eq!(reason_a, Some(FinishReason::Stop));
        assert_eq!(reason_b, Some(FinishReason::Stop));
    }

    /// A real `LmRunner` whose next-token is `last + 1`, with a single KV
    /// "cache" that is just the token history — exported/restored so that
    /// [`RunnerBatchRunner`] time-slicing must restore the right sequence's
    /// state before each step (else the two sequences would bleed together).
    struct SeqMockLm {
        vocab: usize,
        tokens: Vec<u32>,
    }
    impl SeqMockLm {
        fn onehot(&self, tok: u32) -> Vec<f32> {
            let mut v = vec![0.0; self.vocab];
            v[tok as usize] = 10.0;
            v
        }
        fn next_tok(&self) -> u32 {
            self.tokens.last().copied().unwrap_or(0) + 1
        }
    }
    impl LmRunner for SeqMockLm {
        fn family(&self) -> &'static str {
            "seqmock"
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn predict_logits(&mut self, _p: &[u32]) -> anyhow::Result<Vec<f32>> {
            Ok(vec![0.0; self.vocab])
        }
        fn prefill_logits(&mut self, p: &[u32]) -> anyhow::Result<Vec<f32>> {
            self.tokens = p.to_vec();
            Ok(self.onehot(self.next_tok()))
        }
        fn decode_logits(&mut self, t: u32) -> anyhow::Result<Vec<f32>> {
            self.tokens.push(t);
            Ok(self.onehot(self.next_tok()))
        }
        fn export_session(&self) -> Option<SessionSnapshot> {
            Some(SessionSnapshot {
                kv: rlx_runtime::kv_cache::LayerKvCache {
                    past_len: self.tokens.len(),
                    layers_k: vec![],
                    layers_v: vec![],
                    layers_kv_base: vec![],
                },
                tokens: self.tokens.clone(),
            })
        }
        fn restore_session(&mut self, snap: &SessionSnapshot) -> bool {
            self.tokens = snap.tokens.clone();
            true
        }
    }

    #[test]
    fn runner_batch_runner_time_slices_sequences_via_kv_swap() {
        let vocab = 32;
        let lm = SeqMockLm {
            vocab,
            tokens: vec![],
        };
        let runner = RunnerBatchRunner::new(Box::new(lm)).expect("supports sessions");

        let tok = TokenizerHandle::from_raw(tokenizers::Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::default(),
        ));
        let mut batcher = ContinuousBatcher::new(Box::new(runner), Arc::new(tok), 256, 8);

        // Seq A: prompt [1,2] → emits 3,4, then 5 == EOS. Seq B: prompt
        // [10,11] → emits 12, then 13 == EOS. The two share the `last+1` rule,
        // so correct output requires each step to resume the *right* history.
        let (txa, rxa) = channel();
        let (txb, rxb) = channel();
        batcher.admit(req(vec![1, 2], 10, 5), txa);
        batcher.admit(req(vec![10, 11], 10, 13), txb);

        let mut steps = 0;
        while !batcher.is_idle() && steps < 100 {
            batcher.pump();
            steps += 1;
        }

        let collect = |rx: Receiver<StreamItem>| {
            let mut ids = Vec::new();
            let mut reason = None;
            while let Ok(it) = rx.try_recv() {
                match it {
                    StreamItem::Token { id, .. } => ids.push(id),
                    StreamItem::Done { finish_reason, .. } => reason = Some(finish_reason),
                    StreamItem::Error(e) => panic!("{e}"),
                }
            }
            (ids, reason)
        };
        let (ids_a, reason_a) = collect(rxa);
        let (ids_b, reason_b) = collect(rxb);

        assert_eq!(ids_a, vec![3, 4], "seq A continues 1,2 → 3,4 (5 is EOS)");
        assert_eq!(ids_b, vec![12], "seq B continues 10,11 → 12 (13 is EOS)");
        assert_eq!(reason_a, Some(FinishReason::Stop));
        assert_eq!(reason_b, Some(FinishReason::Stop));
    }

    #[test]
    fn runner_batch_runner_rejects_session_less_runner() {
        // engine::tests-style runner without export/restore must be refused.
        struct NoSession;
        impl LmRunner for NoSession {
            fn family(&self) -> &'static str {
                "nosession"
            }
            fn vocab_size(&self) -> usize {
                4
            }
            fn predict_logits(&mut self, _p: &[u32]) -> anyhow::Result<Vec<f32>> {
                Ok(vec![0.0; 4])
            }
        }
        assert!(RunnerBatchRunner::new(Box::new(NoSession)).is_err());
    }

    // ---- FusedBatchRunner: real tiny Qwen3, fused == single-sequence ----

    fn tiny_cfg() -> rlx_qwen3::Qwen3Config {
        rlx_qwen3::Qwen3Config {
            vocab_size: 16,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            qk_norm: true,
            sliding_window: None,
            max_window_layers: usize::MAX,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    fn tiny_generator() -> Qwen3Generator {
        use rlx_core::weight_map::WeightMap;
        let cfg = tiny_cfg();
        let h = cfg.hidden_size;
        let q = cfg.q_proj_dim();
        let kv = cfg.kv_proj_dim();
        let int = cfg.intermediate_size;
        let dh = cfg.head_dim;
        let pat = |n: usize, salt: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
                    (x as f32 / (1u32 << 24) as f32) - 0.5
                })
                .collect()
        };
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        t.insert(
            "model.embed_tokens.weight".into(),
            (pat(cfg.vocab_size * h, 1), vec![cfg.vocab_size, h]),
        );
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            let s = i as u32;
            t.insert(
                format!("{lp}.input_layernorm.weight"),
                (pat(h, 100 + s), vec![h]),
            );
            t.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                (pat(h, 200 + s), vec![h]),
            );
            t.insert(
                format!("{lp}.self_attn.q_proj.weight"),
                (pat(q * h, 300 + s), vec![q, h]),
            );
            t.insert(
                format!("{lp}.self_attn.k_proj.weight"),
                (pat(kv * h, 400 + s), vec![kv, h]),
            );
            t.insert(
                format!("{lp}.self_attn.v_proj.weight"),
                (pat(kv * h, 500 + s), vec![kv, h]),
            );
            t.insert(
                format!("{lp}.self_attn.o_proj.weight"),
                (pat(h * q, 600 + s), vec![h, q]),
            );
            t.insert(
                format!("{lp}.self_attn.q_norm.weight"),
                (pat(dh, 700 + s), vec![dh]),
            );
            t.insert(
                format!("{lp}.self_attn.k_norm.weight"),
                (pat(dh, 800 + s), vec![dh]),
            );
            t.insert(
                format!("{lp}.mlp.gate_proj.weight"),
                (pat(int * h, 900 + s), vec![int, h]),
            );
            t.insert(
                format!("{lp}.mlp.up_proj.weight"),
                (pat(int * h, 1000 + s), vec![int, h]),
            );
            t.insert(
                format!("{lp}.mlp.down_proj.weight"),
                (pat(h * int, 1100 + s), vec![h, int]),
            );
        }
        t.insert("model.norm.weight".into(), (pat(h, 2000), vec![h]));
        t.insert(
            "lm_head.weight".into(),
            (pat(cfg.vocab_size * h, 3000), vec![cfg.vocab_size, h]),
        );
        let mut wm = WeightMap::from_tensors(t);
        Qwen3Generator::from_loader(cfg, &mut wm, rlx_runtime::Device::Cpu).unwrap()
    }

    /// `FusedBatchRunner` must produce the same logits as independent
    /// single-sequence prefill+decode, while fusing the decodes into one ragged
    /// forward. Uses **different-length** prompts so the two sequences decode at
    /// different cache positions — the real continuous-batching case.
    #[test]
    fn fused_batch_runner_matches_single_sequence() {
        let prompt_a = vec![1u32, 2, 3];
        let prompt_b = vec![6u32, 7, 8, 9, 10];
        let past_a = prompt_a.len() as u32;
        let past_b = prompt_b.len() as u32;

        // Reference: two independent single-sequence decodes.
        let mut g = tiny_generator();
        let pa = g.prefill_get_last_logits(&prompt_a).unwrap();
        let ta = argmax(&pa);
        let exp_a = g.decode_get_logits(ta).unwrap();
        let pb = g.prefill_get_last_logits(&prompt_b).unwrap();
        let tb = argmax(&pb);
        let exp_b = g.decode_get_logits(tb).unwrap();

        // Fused runner: prefill both, then ONE ragged decode forward.
        let mut runner = FusedBatchRunner::new(tiny_generator());
        let prefill = runner
            .forward_batch(&[
                BatchEntry {
                    seq_id: 0,
                    kind: BatchKind::Prefill,
                    input_tokens: prompt_a,
                    cached_len: 0,
                },
                BatchEntry {
                    seq_id: 1,
                    kind: BatchKind::Prefill,
                    input_tokens: prompt_b,
                    cached_len: 0,
                },
            ])
            .unwrap();
        let pf: HashMap<SeqId, Vec<f32>> = prefill.into_iter().collect();
        assert_eq!(argmax(&pf[&0]), ta, "prefill A first token");
        assert_eq!(argmax(&pf[&1]), tb, "prefill B first token");

        let decode = runner
            .forward_batch(&[
                BatchEntry {
                    seq_id: 0,
                    kind: BatchKind::Decode,
                    input_tokens: vec![ta],
                    cached_len: past_a,
                },
                BatchEntry {
                    seq_id: 1,
                    kind: BatchKind::Decode,
                    input_tokens: vec![tb],
                    cached_len: past_b,
                },
            ])
            .unwrap();
        let dec: HashMap<SeqId, Vec<f32>> = decode.into_iter().collect();

        let close = |a: &[f32], b: &[f32], who: &str| {
            assert_eq!(a.len(), b.len(), "{who} len");
            for (j, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-3 + 1e-3 * y.abs(),
                    "{who}[{j}]: {x} vs {y}"
                );
            }
        };
        close(&dec[&0], &exp_a, "fused decode A");
        close(&dec[&1], &exp_b, "fused decode B");
    }

    /// Quantized KV storage must track the f32 path closely across multiple
    /// decode steps (exercises whole-cache quantize on prefill + incremental
    /// per-row append on decode). F16 works for tiny_cfg's kv_dim (q-formats
    /// need kv_dim % 32 == 0, which real models satisfy).
    #[test]
    fn fused_batch_runner_quantized_kv_matches_f32() {
        let prompt_a = vec![1u32, 2, 3];
        let prompt_b = vec![6u32, 7, 8, 9, 10];

        // Run two decode steps through a runner with the given KV storage,
        // returning the final per-sequence logits.
        let run = |kv_quant: Option<KvQuant>| -> HashMap<SeqId, Vec<f32>> {
            let mut r = FusedBatchRunner::with_kv_quant(tiny_generator(), kv_quant);
            let pf: HashMap<SeqId, Vec<f32>> = r
                .forward_batch(&[
                    BatchEntry {
                        seq_id: 0,
                        kind: BatchKind::Prefill,
                        input_tokens: prompt_a.clone(),
                        cached_len: 0,
                    },
                    BatchEntry {
                        seq_id: 1,
                        kind: BatchKind::Prefill,
                        input_tokens: prompt_b.clone(),
                        cached_len: 0,
                    },
                ])
                .unwrap()
                .into_iter()
                .collect();
            let (mut ta, mut tb) = (argmax(&pf[&0]), argmax(&pf[&1]));
            let (mut pa, mut pb) = (prompt_a.len() as u32, prompt_b.len() as u32);
            let mut last = HashMap::new();
            for _ in 0..2 {
                last = r
                    .forward_batch(&[
                        BatchEntry {
                            seq_id: 0,
                            kind: BatchKind::Decode,
                            input_tokens: vec![ta],
                            cached_len: pa,
                        },
                        BatchEntry {
                            seq_id: 1,
                            kind: BatchKind::Decode,
                            input_tokens: vec![tb],
                            cached_len: pb,
                        },
                    ])
                    .unwrap()
                    .into_iter()
                    .collect();
                ta = argmax(&last[&0]);
                tb = argmax(&last[&1]);
                pa += 1;
                pb += 1;
            }
            last
        };

        let f32_out = run(None);
        let q_out = run(Some(KvQuant::F16));

        let close = |a: &[f32], b: &[f32], who: &str| {
            for (j, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= 0.05 + 0.02 * y.abs(),
                    "{who}[{j}]: quant {x} vs f32 {y}"
                );
            }
        };
        close(&q_out[&0], &f32_out[&0], "quant vs f32 seq A");
        close(&q_out[&1], &f32_out[&1], "quant vs f32 seq B");
    }

    /// The uniform-grouping fallback (used when a device lacks per-token RoPE)
    /// must produce the same logits as the ragged path for different-length
    /// sequences — it's the correctness safety net for GPU backends.
    #[test]
    fn fused_runner_uniform_fallback_matches_ragged() {
        let prompt_a = vec![1u32, 2, 3];
        let prompt_b = vec![6u32, 7, 8, 9, 10];

        let run = |force_uniform: bool| -> HashMap<SeqId, Vec<f32>> {
            let mut r = FusedBatchRunner::new(tiny_generator());
            if force_uniform {
                r.ragged = false; // simulate a backend without per-token RoPE
            }
            let pf: HashMap<SeqId, Vec<f32>> = r
                .forward_batch(&[
                    BatchEntry {
                        seq_id: 0,
                        kind: BatchKind::Prefill,
                        input_tokens: prompt_a.clone(),
                        cached_len: 0,
                    },
                    BatchEntry {
                        seq_id: 1,
                        kind: BatchKind::Prefill,
                        input_tokens: prompt_b.clone(),
                        cached_len: 0,
                    },
                ])
                .unwrap()
                .into_iter()
                .collect();
            r.forward_batch(&[
                BatchEntry {
                    seq_id: 0,
                    kind: BatchKind::Decode,
                    input_tokens: vec![argmax(&pf[&0])],
                    cached_len: prompt_a.len() as u32,
                },
                BatchEntry {
                    seq_id: 1,
                    kind: BatchKind::Decode,
                    input_tokens: vec![argmax(&pf[&1])],
                    cached_len: prompt_b.len() as u32,
                },
            ])
            .unwrap()
            .into_iter()
            .collect()
        };

        let ragged = run(false);
        let uniform = run(true);
        let close = |a: &[f32], b: &[f32], who: &str| {
            for (j, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-3 + 1e-3 * y.abs(),
                    "{who}[{j}]: {x} vs {y}"
                );
            }
        };
        close(&uniform[&0], &ragged[&0], "fallback seq A");
        close(&uniform[&1], &ragged[&1], "fallback seq B");
    }

    fn argmax(v: &[f32]) -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap()
    }
}
