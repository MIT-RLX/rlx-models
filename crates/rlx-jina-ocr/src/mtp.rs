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

//! FastMTP speculative draft head (`deepseek_ocr_mtp.py`).
//!
//! One dense draft block, applied **recursively** for `K` steps: each step
//! feeds its own post-norm output back in as the next step's
//! `previous_hidden_states`, so a single set of weights produces `K` draft
//! tokens. Embedding, final norm and LM head are shared with the target model
//! (`mtp_share_embedding_weights` / `mtp_share_norm` / `mtp_share_lm_head` —
//! and indeed the checkpoint ships no `mtp_module.heads.0.shared_head.*`).
//!
//! Step `k`, starting from target position `p` (the last position the target
//! model has a hidden state for) and the token it sampled at `p + 1`:
//!
//! ```text
//! pos  = p + 1 + k
//! e    = Emb(tok);  e = 0 if pos == 0
//! x    = eh_proj([ enorm(e) ; hnorm(h) ])
//! x    = block(x @ pos)                      // pre-norm attn + SwiGLU MLP
//! h    = model.norm(x)                       // post-norm: what step k+1 reads
//! tok  = argmax(lm_head @ h)                 // draft token for pos + 1
//! ```
//!
//! ## Status
//!
//! The head below is complete and tested, but the runner does **not** decode
//! speculatively with it. Accepting `K` drafts in one target pass needs a
//! multi-token-with-past decode graph (`seq = K + 1`, `MaskKind::Bias`), which
//! [`rlx_unlimited_ocr`] does not build — and verifying the drafts one token at
//! a time through the existing single-token graph costs exactly as much as
//! decoding them normally, so it would buy nothing. The reference is in the
//! same position: `DeepseekOCRForCausalLM` lists `mtp_module.*` in
//! `_keys_to_ignore_on_load_unexpected` and warns that transformers
//! `generate()` has no speculative path; FastMTP is a vLLM-only feature there.

use crate::config::JinaOcrConfig;
use anyhow::{Context, Result, ensure};
use rlx_unlimited_ocr::expert_pack::PackedLmWeights;
use rlx_unlimited_ocr::nn;
use rlx_unlimited_ocr::speculative::Drafter;
use rlx_unlimited_ocr::weights::{UnlimitedOcrWeightPrefix, UnlimitedOcrWeightStore};
use std::sync::Arc;

/// Tensor-name prefix of draft head `i`.
pub fn head_prefix(i: usize) -> String {
    format!("mtp_module.heads.{i}.")
}

/// Whether a checkpoint carries FastMTP draft weights at all.
pub fn checkpoint_has_mtp(store: &UnlimitedOcrWeightStore) -> bool {
    store.count_keys_with_prefix("mtp_module.") > 0
}

/// One draft block's parameters (all `[out, in]`, PyTorch `nn.Linear` layout).
#[derive(Debug, Clone, Default)]
pub struct MtpWeights {
    /// RMSNorm over the *embedding* input.
    pub enorm: Vec<f32>,
    /// RMSNorm over the *previous hidden state* input.
    pub hnorm: Vec<f32>,
    /// `[hidden, 2 * hidden]` — projects the concatenated pair back to `hidden`.
    pub eh_proj: Vec<f32>,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

impl MtpWeights {
    /// Load draft head `head_idx` from the checkpoint.
    pub fn load(store: &UnlimitedOcrWeightStore, head_idx: usize) -> Result<Self> {
        let p = head_prefix(head_idx);
        let names = [
            "enorm.weight",
            "hnorm.weight",
            "eh_proj.weight",
            "mtp_block.input_layernorm.weight",
            "mtp_block.post_attention_layernorm.weight",
            "mtp_block.self_attn.q_proj.weight",
            "mtp_block.self_attn.k_proj.weight",
            "mtp_block.self_attn.v_proj.weight",
            "mtp_block.self_attn.o_proj.weight",
            "mtp_block.mlp.gate_proj.weight",
            "mtp_block.mlp.up_proj.weight",
            "mtp_block.mlp.down_proj.weight",
        ];
        let keys: Vec<String> = names.iter().map(|n| format!("{p}{n}")).collect();
        let map = store.load_owned_keys(keys.clone())?;
        let take = |suffix: &str| -> Result<Vec<f32>> {
            let key = format!("{p}{suffix}");
            let (data, _) = map
                .get(&key)
                .with_context(|| format!("FastMTP tensor {key} missing"))?;
            Ok(data.to_vec())
        };
        Ok(Self {
            enorm: take("enorm.weight")?,
            hnorm: take("hnorm.weight")?,
            eh_proj: take("eh_proj.weight")?,
            input_layernorm: take("mtp_block.input_layernorm.weight")?,
            post_attention_layernorm: take("mtp_block.post_attention_layernorm.weight")?,
            q_proj: take("mtp_block.self_attn.q_proj.weight")?,
            k_proj: take("mtp_block.self_attn.k_proj.weight")?,
            v_proj: take("mtp_block.self_attn.v_proj.weight")?,
            o_proj: take("mtp_block.self_attn.o_proj.weight")?,
            gate_proj: take("mtp_block.mlp.gate_proj.weight")?,
            up_proj: take("mtp_block.mlp.up_proj.weight")?,
            down_proj: take("mtp_block.mlp.down_proj.weight")?,
        })
    }

    fn validate(&self, hidden: usize, intermediate: usize) -> Result<()> {
        let want = |name: &str, got: usize, exp: usize| -> Result<()> {
            ensure!(
                got == exp,
                "FastMTP {name}: expected {exp} elements, got {got}"
            );
            Ok(())
        };
        want("enorm", self.enorm.len(), hidden)?;
        want("hnorm", self.hnorm.len(), hidden)?;
        want("eh_proj", self.eh_proj.len(), hidden * 2 * hidden)?;
        want("input_layernorm", self.input_layernorm.len(), hidden)?;
        want(
            "post_attention_layernorm",
            self.post_attention_layernorm.len(),
            hidden,
        )?;
        for (name, w) in [
            ("q_proj", &self.q_proj),
            ("k_proj", &self.k_proj),
            ("v_proj", &self.v_proj),
            ("o_proj", &self.o_proj),
        ] {
            want(name, w.len(), hidden * hidden)?;
        }
        want("gate_proj", self.gate_proj.len(), intermediate * hidden)?;
        want("up_proj", self.up_proj.len(), intermediate * hidden)?;
        want("down_proj", self.down_proj.len(), hidden * intermediate)?;
        Ok(())
    }
}

/// Per-position K/V the draft block accumulates while proposing.
#[derive(Debug, Clone, Default)]
pub struct MtpKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Absolute position of each cached row.
    positions: Vec<usize>,
}

impl MtpKvCache {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.positions.clear();
    }

    /// Drop every cached row at position `>= from`, after a draft is rejected.
    pub fn truncate_from(&mut self, from: usize, hidden: usize) {
        let keep = self.positions.iter().take_while(|&&p| p < from).count();
        self.positions.truncate(keep);
        self.k.truncate(keep * hidden);
        self.v.truncate(keep * hidden);
    }
}

/// Weights shared with the target model.
pub struct SharedTargetWeights<'a> {
    /// `model.norm.weight` — `mtp_share_norm`.
    pub final_norm: &'a [f32],
    /// Row lookup into `model.embed_tokens.weight` — `mtp_share_embedding_weights`.
    pub embed: &'a dyn Fn(u32) -> Result<Vec<f32>>,
    /// `lm_head.weight` applied to a post-norm hidden — `mtp_share_lm_head`.
    pub lm_head: &'a dyn Fn(&[f32]) -> Result<Vec<f32>>,
}

/// The one wiring choice the shipped checkpoint does not pin down.
///
/// `jina-ocr-v1` ships FastMTP weights but not the module that runs them — the
/// card points at a vLLM plugin — so the concatenation order into `eh_proj` had
/// to be recovered by experiment rather than read off a reference. It is silent
/// if wrong: the head still emits in-vocabulary tokens, just uninformed ones,
/// so the only symptom is a low acceptance rate.
///
/// **Settled by measurement**: `embed_first` scores 30.4% top-1 agreement with
/// the target against 0.0% reversed (`examples/mtp_probe.rs`, real page). That
/// matches DeepSeek-V3's *code*, which concatenates `enorm(embed)` first — and
/// not the V3 paper, which writes `M[RMSNorm(h_i); RMSNorm(Emb(t_{i+1}))]` with
/// the operands the other way round. The knob stays so the sweep is
/// reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpWiring {
    /// `eh_proj([enorm(embed) ; hnorm(hidden)])` when true, the reverse
    /// otherwise.
    pub embed_first: bool,
}

impl Default for MtpWiring {
    fn default() -> Self {
        Self { embed_first: true }
    }
}

/// Host-side FastMTP draft head.
#[derive(Debug)]
pub struct MtpHead {
    weights: MtpWeights,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate: usize,
    eps: f32,
    steps: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    wiring: MtpWiring,
}

impl MtpHead {
    /// Build from a loaded checkpoint. Returns `Ok(None)` when the config
    /// disables MTP or the checkpoint carries no draft weights.
    pub fn load(cfg: &JinaOcrConfig, store: &UnlimitedOcrWeightStore) -> Result<Option<Self>> {
        if !cfg.mtp.is_enabled() || !checkpoint_has_mtp(store) {
            return Ok(None);
        }
        ensure!(
            !cfg.mtp.moe,
            "FastMTP draft block is dense; mtp_moe=true is unsupported"
        );
        ensure!(
            cfg.mtp.share_embedding_weights && cfg.mtp.share_lm_head && cfg.mtp.share_norm,
            "FastMTP head with unshared embed/norm/lm_head is unsupported \
             (checkpoint ships no shared_head.* tensors)"
        );
        let weights = MtpWeights::load(store, 0)?;
        Ok(Some(Self::new(cfg, weights)?))
    }

    /// Build from explicit weights (tests, or a caller with its own loader).
    pub fn new(cfg: &JinaOcrConfig, weights: MtpWeights) -> Result<Self> {
        let hidden = cfg.lm.hidden_size;
        let intermediate = cfg.lm.intermediate_size;
        weights.validate(hidden, intermediate)?;
        let head_dim = cfg.lm.head_dim();
        let (cos, sin) =
            nn::rope_tables(cfg.lm.max_position_embeddings, head_dim, cfg.lm.rope_theta);
        Ok(Self {
            weights,
            hidden,
            num_heads: cfg.lm.num_attention_heads,
            head_dim,
            intermediate,
            eps: cfg.lm.rms_norm_eps as f32,
            steps: cfg.mtp.num_speculative_steps,
            cos,
            sin,
            wiring: MtpWiring::default(),
        })
    }

    /// Override the wiring recovered by `examples/mtp_probe.rs`.
    pub fn set_wiring(&mut self, wiring: MtpWiring) {
        self.wiring = wiring;
    }

    pub fn wiring(&self) -> MtpWiring {
        self.wiring
    }

    /// Draft depth `K` (`mtp_num_speculative_steps`).
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Override the draft depth.
    ///
    /// The config's `K = 3` is not obviously the best operating point: each
    /// extra step costs a full host draft forward, while acceptance decays
    /// with depth because steps past the first chain the draft's *own* hidden
    /// state rather than the target's. A shallower draft can buy nearly the
    /// same tokens-per-round for a fraction of the host cost.
    pub fn set_steps(&mut self, steps: usize) -> Result<()> {
        ensure!(steps >= 1, "draft depth must be at least 1");
        self.steps = steps;
        Ok(())
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// One recursive draft step.
    ///
    /// `token` is the token at absolute `position`; `prev_hidden` is the
    /// **pre**-final-norm hidden state of `position - 1` — what
    /// `DeepseekV2Model` returns as `pre_norm_hidden_states`, since the draft
    /// block applies its own `hnorm` to it. Returns the pre-norm hidden for
    /// `position`; run it through [`Self::norm_for_lm_head`] to get the draft
    /// token for `position + 1`.
    pub fn step(
        &self,
        token: u32,
        position: usize,
        prev_hidden: &[f32],
        kv: &mut MtpKvCache,
        shared: &SharedTargetWeights<'_>,
    ) -> Result<Vec<f32>> {
        ensure!(
            prev_hidden.len() == self.hidden,
            "FastMTP prev_hidden len {} != {}",
            prev_hidden.len(),
            self.hidden
        );
        let w = &self.weights;

        // `torch.where(positions == 0, 0, inputs_embeds)` — there is no token
        // before position 0 to embed.
        let embed = if position == 0 {
            vec![0f32; self.hidden]
        } else {
            let e = (shared.embed)(token)?;
            ensure!(e.len() == self.hidden, "embedding row width");
            e
        };

        let e_normed = nn::rms_norm(&embed, 1, self.hidden, &w.enorm, self.eps);
        let h_normed = nn::rms_norm(prev_hidden, 1, self.hidden, &w.hnorm, self.eps);
        let mut pair = Vec::with_capacity(2 * self.hidden);
        if self.wiring.embed_first {
            pair.extend_from_slice(&e_normed);
            pair.extend_from_slice(&h_normed);
        } else {
            pair.extend_from_slice(&h_normed);
            pair.extend_from_slice(&e_normed);
        }
        let x = nn::linear_wt(&pair, 1, 2 * self.hidden, &w.eh_proj, self.hidden, None)?;

        self.block(&x, position, kv)
    }

    /// Apply the shared final norm — the step from a draft hidden state to the
    /// tensor `lm_head` consumes.
    ///
    /// Split out because the two consumers want different tensors: the next
    /// recursive step chains the **pre**-norm output (it applies its own
    /// `hnorm`), while the vocabulary projection needs the normed one. Chaining
    /// the normed tensor instead double-norms the recursion and quietly costs
    /// draft quality.
    pub fn norm_for_lm_head(&self, hidden: &[f32], shared: &SharedTargetWeights<'_>) -> Vec<f32> {
        nn::rms_norm(hidden, 1, self.hidden, shared.final_norm, self.eps)
    }

    /// Pre-norm decoder block: `x + attn(ln1(x))`, then `+ swiglu(ln2(·))`.
    fn block(&self, x: &[f32], position: usize, kv: &mut MtpKvCache) -> Result<Vec<f32>> {
        let w = &self.weights;
        let h = self.hidden;

        let normed = nn::rms_norm(x, 1, h, &w.input_layernorm, self.eps);
        let mut q = nn::linear_wt(&normed, 1, h, &w.q_proj, h, None)?;
        let mut k = nn::linear_wt(&normed, 1, h, &w.k_proj, h, None)?;
        let v = nn::linear_wt(&normed, 1, h, &w.v_proj, h, None)?;

        let pos = [position];
        nn::apply_rope(
            &mut q,
            1,
            self.num_heads,
            self.head_dim,
            &pos,
            &self.cos,
            &self.sin,
        );
        nn::apply_rope(
            &mut k,
            1,
            self.num_heads,
            self.head_dim,
            &pos,
            &self.cos,
            &self.sin,
        );

        kv.k.extend_from_slice(&k);
        kv.v.extend_from_slice(&v);
        kv.positions.push(position);

        let attn = self.attend(&q, kv);
        let attn_out = nn::linear_wt(&attn, 1, h, &w.o_proj, h, None)?;
        let mut residual = x.to_vec();
        nn::add_inplace(&mut residual, &attn_out);

        let normed2 = nn::rms_norm(&residual, 1, h, &w.post_attention_layernorm, self.eps);
        let mut gate = nn::linear_wt(&normed2, 1, h, &w.gate_proj, self.intermediate, None)?;
        let up = nn::linear_wt(&normed2, 1, h, &w.up_proj, self.intermediate, None)?;
        nn::silu(&mut gate);
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g *= *u;
        }
        let ffn = nn::linear_wt(&gate, 1, self.intermediate, &w.down_proj, h, None)?;
        nn::add_inplace(&mut residual, &ffn);
        Ok(residual)
    }

    /// Single-query attention over the whole draft cache (causal by
    /// construction: the cache only ever holds positions `<= position`).
    fn attend(&self, q: &[f32], kv: &MtpKvCache) -> Vec<f32> {
        let n_k = kv.positions.len();
        let h = self.hidden;
        let dh = self.head_dim;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut out = vec![0f32; h];
        let mut scores = vec![0f32; n_k];
        for head in 0..self.num_heads {
            let qh = &q[head * dh..(head + 1) * dh];
            for (t, score) in scores.iter_mut().enumerate() {
                let kh = &kv.k[t * h + head * dh..t * h + (head + 1) * dh];
                *score = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            nn::softmax_rows(&mut scores, 1, n_k);
            let dst = &mut out[head * dh..(head + 1) * dh];
            for (t, &p) in scores.iter().enumerate() {
                let vh = &kv.v[t * h + head * dh..t * h + (head + 1) * dh];
                for (o, val) in dst.iter_mut().zip(vh) {
                    *o += p * val;
                }
            }
        }
        out
    }

    /// Populate the draft block's KV cache from the prompt.
    ///
    /// The draft block is a transformer layer, so it has history of its own.
    /// Skipping this does not break anything visible — the drafts are still
    /// valid tokens and the verify pass still guarantees the output — it just
    /// makes them guesses drawn from two or three rows of context instead of
    /// the whole page, which shows up only as a low acceptance rate.
    ///
    /// Row `p` pairs the pre-norm hidden of `p - 1` with the embedding of the
    /// token *at* `p`, the same shift [`Self::step`] uses, so priming and
    /// drafting produce one continuous run of rows. Position 0 has no
    /// predecessor hidden state and is left out.
    pub fn prime(
        &self,
        tokens: &[u32],
        hidden_all: &[f32],
        kv: &mut MtpKvCache,
        shared: &SharedTargetWeights<'_>,
    ) -> Result<()> {
        let n = tokens.len();
        ensure!(
            hidden_all.len() == n * self.hidden,
            "FastMTP prime: hidden is {} elements, want {n}*{}",
            hidden_all.len(),
            self.hidden
        );
        kv.clear();
        for p in 1..n {
            let prev = &hidden_all[(p - 1) * self.hidden..p * self.hidden];
            // The returned hidden is discarded: priming is only here for the
            // K/V rows it leaves behind.
            self.step(tokens[p], p, prev, kv, shared)?;
        }
        Ok(())
    }

    /// Propose up to [`Self::steps`] draft tokens.
    ///
    /// `last_position` is the position the target model last produced a hidden
    /// state for, `target_hidden` that (pre-final-norm) state, and `next_token` the
    /// token the target sampled — which sits at `last_position + 1`. The `i`-th
    /// returned token is a guess at position `last_position + 2 + i`.
    pub fn propose(
        &self,
        next_token: u32,
        last_position: usize,
        target_hidden: &[f32],
        kv: &mut MtpKvCache,
        shared: &SharedTargetWeights<'_>,
    ) -> Result<Vec<u32>> {
        let mut drafts = Vec::with_capacity(self.steps);
        let mut hidden = target_hidden.to_vec();
        let mut token = next_token;
        for k in 0..self.steps {
            let position = last_position + 1 + k;
            hidden = self.step(token, position, &hidden, kv, shared)?;
            let logits = (shared.lm_head)(&self.norm_for_lm_head(&hidden, shared))?;
            token = argmax(&logits);
            drafts.push(token);
        }
        Ok(drafts)
    }
}

/// Adapts [`MtpHead`] to the generic speculative-decode loop.
///
/// The two APIs number positions differently and the mapping is the only place
/// this can go wrong. [`Drafter::draft`] is handed the token the target just
/// sampled together with the hidden state of the position *before* it, which is
/// exactly [`MtpHead::propose`]'s `(next_token, last_position, target_hidden)`
/// triple — so the arguments pass straight through.
///
/// Rollback is the off-by-one to watch. Draft row `k` sits at position
/// `next_pos + k` and was computed from draft token `k - 1`, so it is valid iff
/// every draft below it was accepted: rows up to and including
/// `next_pos + accepted` survive. [`Drafter::rollback`] names the last position
/// to keep, hence `truncate_from(pos + 1)`. Dropping one row too many would
/// only cost draft quality — dropping one too few feeds the next round a hidden
/// state derived from a token that was rejected.
pub struct MtpDrafter<'a> {
    head: &'a MtpHead,
    shared: SharedTargetWeights<'a>,
    kv: MtpKvCache,
}

impl<'a> MtpDrafter<'a> {
    pub fn new(head: &'a MtpHead, shared: SharedTargetWeights<'a>) -> Self {
        Self {
            head,
            shared,
            kv: MtpKvCache::default(),
        }
    }

    /// Rows currently cached by the draft block.
    pub fn cached_rows(&self) -> usize {
        self.kv.len()
    }

    /// Forget all draft state — call between documents.
    pub fn reset(&mut self) {
        self.kv.clear();
    }
}

impl Drafter for MtpDrafter<'_> {
    fn depth(&self) -> usize {
        self.head.steps()
    }

    fn draft(&mut self, last_token: u32, last_pos: usize, hidden: &[f32]) -> Result<Vec<u32>> {
        self.head
            .propose(last_token, last_pos, hidden, &mut self.kv, &self.shared)
    }

    fn rollback(&mut self, pos: usize) {
        self.kv.truncate_from(pos + 1, self.head.hidden_size());
    }

    fn prime(&mut self, tokens: &[u32], hidden_all: &[f32]) -> Result<()> {
        self.head
            .prime(tokens, hidden_all, &mut self.kv, &self.shared)
    }
}

/// A loaded FastMTP head together with the target weights it shares.
///
/// The draft block is tiny, but `mtp_share_lm_head` means every draft step ends
/// in a full `[vocab, hidden]` projection — 129280x1280 on jina-ocr-v1. That
/// weight has to be resident as host f32 (662 MB here), because the draft
/// hidden state never enters the compiled graph and so cannot use the device
/// copy. [`Self::host_bytes`] reports the cost; it is why this is constructed
/// explicitly rather than loaded with the model.
///
/// The embedding table is *not* duplicated — draft steps look rows up through
/// the already-resident packed weights.
pub struct JinaMtp {
    head: MtpHead,
    final_norm: Vec<f32>,
    lm_head: Vec<f32>,
    pack: Arc<PackedLmWeights>,
    vocab: usize,
    hidden: usize,
}

impl JinaMtp {
    /// Load the draft head and the shared target tensors.
    ///
    /// `Ok(None)` when the config disables MTP or the checkpoint carries no
    /// draft weights — callers should fall back to plain decode.
    pub fn load(
        cfg: &JinaOcrConfig,
        store: &UnlimitedOcrWeightStore,
        pack: Arc<PackedLmWeights>,
    ) -> Result<Option<Self>> {
        let Some(head) = MtpHead::load(cfg, store)? else {
            return Ok(None);
        };
        let hidden = cfg.lm.hidden_size;
        let vocab = cfg.lm.vocab_size;
        let norm_key = UnlimitedOcrWeightPrefix::lm_norm().to_string();
        let head_key = UnlimitedOcrWeightPrefix::lm_head().to_string();
        let map = store.load_owned_keys([norm_key.clone(), head_key.clone()])?;
        let take = |key: &str| -> Result<Vec<f32>> {
            let (data, _) = map
                .get(key)
                .with_context(|| format!("FastMTP shares {key}, which is missing"))?;
            Ok(data.to_vec())
        };
        let final_norm = take(&norm_key)?;
        let lm_head = take(&head_key)?;
        ensure!(
            final_norm.len() == hidden,
            "shared final norm is {} wide, want {hidden}",
            final_norm.len()
        );
        ensure!(
            lm_head.len() == vocab * hidden,
            "shared lm_head is {} elements, want {vocab}*{hidden}",
            lm_head.len()
        );
        Ok(Some(Self {
            head,
            final_norm,
            lm_head,
            pack,
            vocab,
            hidden,
        }))
    }

    /// Host memory this holds beyond the model itself.
    pub fn host_bytes(&self) -> usize {
        (self.final_norm.len() + self.lm_head.len()) * std::mem::size_of::<f32>()
    }

    pub fn steps(&self) -> usize {
        self.head.steps()
    }

    /// Override the head's wiring — see [`MtpWiring`].
    pub fn set_wiring(&mut self, wiring: MtpWiring) {
        self.head.set_wiring(wiring);
    }

    /// Override the draft depth — see [`MtpHead::set_steps`].
    pub fn set_steps(&mut self, steps: usize) -> Result<()> {
        self.head.set_steps(steps)
    }

    /// Run `f` with the raw head and its shared target weights.
    ///
    /// For probing the head directly, outside the drafting loop.
    pub fn with_head<R>(&self, f: impl FnOnce(&MtpHead, &SharedTargetWeights<'_>) -> R) -> R {
        let embed = |id: u32| self.pack.embed_tokens_lookup(&[id]);
        let lm_head = |h: &[f32]| nn::linear_wt(h, 1, self.hidden, &self.lm_head, self.vocab, None);
        let shared = SharedTargetWeights {
            final_norm: &self.final_norm,
            embed: &embed,
            lm_head: &lm_head,
        };
        f(&self.head, &shared)
    }

    /// Run `f` with a [`Drafter`] over this head.
    ///
    /// Scoped rather than returned because [`SharedTargetWeights`] borrows the
    /// two closures, which need a place to live for the drafter's lifetime.
    pub fn with_drafter<R>(&self, f: impl FnOnce(&mut MtpDrafter<'_>) -> R) -> R {
        let embed = |id: u32| self.pack.embed_tokens_lookup(&[id]);
        let lm_head = |h: &[f32]| nn::linear_wt(h, 1, self.hidden, &self.lm_head, self.vocab, None);
        let shared = SharedTargetWeights {
            final_norm: &self.final_norm,
            embed: &embed,
            lm_head: &lm_head,
        };
        let mut drafter = MtpDrafter::new(&self.head, shared);
        f(&mut drafter)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    rlx_runtime::search::argmax_u32(logits)
}

/// Greedy verification: how many leading draft tokens the target agrees with.
///
/// `target` holds the target model's greedy pick for each drafted position.
/// The accepted run stops at the first disagreement — the target's own token at
/// that position is then the one that gets emitted, so a fully rejected draft
/// still advances by one.
pub fn accept_prefix(drafts: &[u32], target: &[u32]) -> usize {
    drafts
        .iter()
        .zip(target)
        .take_while(|(d, t)| d == t)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::JinaOcrConfig;

    /// Tiny config with the real topology's shape relationships.
    fn tiny_cfg() -> JinaOcrConfig {
        let mut cfg = JinaOcrConfig::from_json_str(
            r#"{
                "model_type": "deepseek_vl_v2",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "first_k_dense_replace": 1,
                "vocab_size": 6,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "use_mla": false,
                "projector_config": {"input_dim": 16, "n_embed": 8, "projector_type": "linear"},
                "mtp_num_heads": 1,
                "mtp_num_speculative_steps": 3,
                "mtp_recursive": true
            }"#,
        )
        .expect("cfg");
        cfg.lm.projector.n_embed = cfg.lm.hidden_size;
        cfg
    }

    fn tiny_weights(cfg: &JinaOcrConfig) -> MtpWeights {
        let h = cfg.lm.hidden_size;
        let i = cfg.lm.intermediate_size;
        // Deterministic, non-degenerate values.
        let fill = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|k| ((k as f32 * 0.37 + seed).sin()) * 0.5)
                .collect()
        };
        MtpWeights {
            enorm: vec![1.0; h],
            hnorm: vec![1.0; h],
            eh_proj: fill(h * 2 * h, 0.1),
            input_layernorm: vec![1.0; h],
            post_attention_layernorm: vec![1.0; h],
            q_proj: fill(h * h, 0.2),
            k_proj: fill(h * h, 0.3),
            v_proj: fill(h * h, 0.4),
            o_proj: fill(h * h, 0.5),
            gate_proj: fill(i * h, 0.6),
            up_proj: fill(i * h, 0.7),
            down_proj: fill(h * i, 0.8),
        }
    }

    struct Shared {
        hidden: usize,
        vocab: usize,
    }

    impl Shared {
        fn embed(&self, id: u32) -> Result<Vec<f32>> {
            Ok((0..self.hidden)
                .map(|k| ((id as f32 + 1.0) * 0.11 + k as f32 * 0.23).sin())
                .collect())
        }

        fn lm_head(&self, hidden: &[f32]) -> Result<Vec<f32>> {
            Ok((0..self.vocab)
                .map(|v| {
                    hidden
                        .iter()
                        .enumerate()
                        .map(|(k, x)| x * ((v as f32 * 0.31 + k as f32 * 0.17).cos()))
                        .sum()
                })
                .collect())
        }
    }

    fn head_and_shared() -> (JinaOcrConfig, MtpHead, Shared, Vec<f32>) {
        let cfg = tiny_cfg();
        let head = MtpHead::new(&cfg, tiny_weights(&cfg)).expect("head");
        let shared = Shared {
            hidden: cfg.lm.hidden_size,
            vocab: cfg.lm.vocab_size,
        };
        let final_norm = vec![1.0f32; cfg.lm.hidden_size];
        (cfg, head, shared, final_norm)
    }

    fn shared_refs<'a>(
        s: &'a Shared,
        final_norm: &'a [f32],
        embed: &'a dyn Fn(u32) -> Result<Vec<f32>>,
        lm_head: &'a dyn Fn(&[f32]) -> Result<Vec<f32>>,
    ) -> SharedTargetWeights<'a> {
        let _ = s;
        SharedTargetWeights {
            final_norm,
            embed,
            lm_head,
        }
    }

    /// After a round keeps `accepted` draft tokens, the draft cache must hold
    /// exactly the rows those tokens' hidden states were derived from — rows
    /// `next_pos ..= next_pos + accepted`.
    ///
    /// Row `k` sits at `next_pos + k` and was computed from draft token `k-1`,
    /// so it is trustworthy only if every draft below it was accepted. Keeping
    /// one row too many is the dangerous direction: the next round would extend
    /// from a hidden state conditioned on a token the target rejected.
    #[test]
    fn drafter_rollback_keeps_exactly_the_accepted_prefix() {
        let (cfg, head, s, fnorm) = head_and_shared();
        let embed = |id: u32| s.embed(id);
        let lm = |h: &[f32]| s.lm_head(h);
        let steps = head.steps();
        let next_pos = 5usize;
        let hidden = vec![0.05f32; cfg.lm.hidden_size];

        for accepted in 0..steps {
            let shared = shared_refs(&s, &fnorm, &embed, &lm);
            let mut d = MtpDrafter::new(&head, shared);
            assert_eq!(d.depth(), steps);

            let draft = d.draft(3, next_pos - 1, &hidden).expect("draft");
            assert_eq!(draft.len(), steps);
            assert_eq!(
                d.cached_rows(),
                steps,
                "propose should cache one row per step"
            );
            assert_eq!(
                d.kv.positions,
                (next_pos..next_pos + steps).collect::<Vec<_>>(),
                "draft rows should start at the drafted token's own position"
            );

            d.rollback(next_pos + accepted);
            assert_eq!(
                d.kv.positions,
                (next_pos..=next_pos + accepted).collect::<Vec<_>>(),
                "accepted={accepted}"
            );
        }
    }

    /// Consecutive rounds must leave the cache one unbroken run of positions.
    ///
    /// A gap means the draft block attends over a hole; an overlap means a
    /// position is represented twice and gets double the attention weight.
    #[test]
    fn draft_cache_positions_stay_contiguous_across_rounds() {
        let (cfg, head, s, fnorm) = head_and_shared();
        let embed = |id: u32| s.embed(id);
        let lm = |h: &[f32]| s.lm_head(h);
        let shared = shared_refs(&s, &fnorm, &embed, &lm);
        let mut d = MtpDrafter::new(&head, shared);
        let hidden = vec![0.05f32; cfg.lm.hidden_size];
        let steps = head.steps();

        // Round 1 at position 5, one draft token accepted.
        let mut next_pos = 5usize;
        let accepted = 1usize;
        d.draft(3, next_pos - 1, &hidden).expect("round 1");
        d.rollback(next_pos + accepted);

        // The loop then emits the target token at `next_pos + accepted + 1`.
        next_pos += accepted + 1;
        d.draft(2, next_pos - 1, &hidden).expect("round 2");

        let want: Vec<usize> = (5..next_pos + steps).collect();
        assert_eq!(d.kv.positions, want, "cache should be one unbroken run");
        assert!(
            d.kv.positions.windows(2).all(|w| w[1] == w[0] + 1),
            "positions must be strictly consecutive"
        );

        d.reset();
        assert_eq!(d.cached_rows(), 0, "reset should clear the draft cache");
    }

    #[test]
    fn propose_emits_exactly_k_finite_draft_tokens() {
        let (cfg, head, s, fnorm) = head_and_shared();
        let embed = |id: u32| s.embed(id);
        let lm = |h: &[f32]| s.lm_head(h);
        let shared = shared_refs(&s, &fnorm, &embed, &lm);

        let mut kv = MtpKvCache::default();
        let target_hidden = vec![0.25f32; cfg.lm.hidden_size];
        let drafts = head
            .propose(3, 10, &target_hidden, &mut kv, &shared)
            .expect("propose");

        assert_eq!(drafts.len(), cfg.mtp.num_speculative_steps);
        assert_eq!(kv.len(), cfg.mtp.num_speculative_steps);
        assert!(drafts.iter().all(|&t| (t as usize) < cfg.lm.vocab_size));
    }

    /// Each step must read the *previous step's* output, not the target's —
    /// that is what "recursive" means, and reusing the target hidden would make
    /// every step identical.
    #[test]
    fn steps_are_recursive_not_repeated() {
        let (cfg, head, s, fnorm) = head_and_shared();
        let embed = |id: u32| s.embed(id);
        let lm = |h: &[f32]| s.lm_head(h);
        let shared = shared_refs(&s, &fnorm, &embed, &lm);

        let mut kv = MtpKvCache::default();
        let target_hidden = vec![0.25f32; cfg.lm.hidden_size];
        let h1 = head
            .step(3, 11, &target_hidden, &mut kv, &shared)
            .expect("step 1");
        let h2 = head.step(3, 12, &h1, &mut kv, &shared).expect("step 2");
        let diff: f32 = h1.iter().zip(&h2).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "recursive steps collapsed to the same state");
    }

    /// `torch.where(positions == 0, 0, inputs_embeds)`.
    #[test]
    fn position_zero_zeroes_the_token_embedding() {
        let (cfg, head, s, fnorm) = head_and_shared();
        let lm = |h: &[f32]| s.lm_head(h);
        let prev = vec![0.3f32; cfg.lm.hidden_size];

        let mut out = Vec::new();
        for tok in [1u32, 5u32] {
            let embed = |id: u32| s.embed(id);
            let shared = shared_refs(&s, &fnorm, &embed, &lm);
            let mut kv = MtpKvCache::default();
            out.push(head.step(tok, 0, &prev, &mut kv, &shared).expect("step"));
        }
        assert_eq!(out[0], out[1], "position 0 must ignore the token id");

        let mut out_nonzero = Vec::new();
        for tok in [1u32, 5u32] {
            let embed = |id: u32| s.embed(id);
            let shared = shared_refs(&s, &fnorm, &embed, &lm);
            let mut kv = MtpKvCache::default();
            out_nonzero.push(head.step(tok, 1, &prev, &mut kv, &shared).expect("step"));
        }
        assert_ne!(out_nonzero[0], out_nonzero[1]);
    }

    #[test]
    fn kv_cache_truncates_back_to_a_position() {
        let (cfg, head, s, fnorm) = head_and_shared();
        let embed = |id: u32| s.embed(id);
        let lm = |h: &[f32]| s.lm_head(h);
        let shared = shared_refs(&s, &fnorm, &embed, &lm);
        let mut kv = MtpKvCache::default();
        head.propose(1, 20, &vec![0.1; cfg.lm.hidden_size], &mut kv, &shared)
            .expect("propose");
        assert_eq!(kv.len(), 3);
        kv.truncate_from(22, cfg.lm.hidden_size);
        assert_eq!(kv.len(), 1, "positions 22, 23 dropped; 21 kept");
        kv.clear();
        assert!(kv.is_empty());
    }

    #[test]
    fn accept_prefix_stops_at_first_mismatch() {
        assert_eq!(accept_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(accept_prefix(&[1, 2, 3], &[1, 9, 3]), 1);
        assert_eq!(accept_prefix(&[1, 2, 3], &[9, 2, 3]), 0);
        assert_eq!(accept_prefix(&[1, 2, 3], &[1, 2]), 2);
        assert_eq!(accept_prefix(&[], &[1]), 0);
    }

    #[test]
    fn weight_shape_mismatch_is_rejected() {
        let cfg = tiny_cfg();
        let mut w = tiny_weights(&cfg);
        w.eh_proj.truncate(4);
        let err = MtpHead::new(&cfg, w).expect_err("should reject");
        assert!(err.to_string().contains("eh_proj"), "got {err}");
    }
}
