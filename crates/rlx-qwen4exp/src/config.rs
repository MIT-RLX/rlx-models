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

//! `qwen4exp` configuration — read from GGUF metadata or a HF `config.json`.
//!
//! The GGUF converter flattens HF's `text_config` into `qwen4exp.*` keys:
//!
//! ```text
//!   HF text_config                      GGUF metadata
//!   ──────────────────────────────────  ───────────────────────────────────────
//!   num_hidden_layers                   block_count − nextn_predict_layers
//!   hidden_size                         embedding_length
//!   head_dim                            attention.key_length / .value_length
//!   layer_types[i] == full_attention     attention.compress_ratios[i] != 0
//!   linear_num_key_heads                ssm.group_count
//!   linear_num_value_heads              ssm.time_step_rank
//!   linear_key_head_dim                 ssm.state_size
//!   hc_count / hc_lowrank               hyper_connection.count / .low_rank
//!   indexer_n_heads / _head_dim         attention.indexer.head_count / .key_length
//!   indexer_budget                      attention.indexer.top_k
//!   indexer_compress_ratio              attention.compress_ratios[i]
//!   ple_layer_ids (1-based)             ple.layers (0-based)
//!   ple_embed_dim / ngram heads         embedding_length_per_layer_input
//!   moe_intermediate_size               expert_feed_forward_length
//!   shared_expert_intermediate_size     expert_shared_feed_forward_length
//! ```
//!
//! Two conversions are **not** reversible and are why this crate targets the
//! GGUF spelling:
//!
//! * **The norm gammas are offset.** Every `Qwen4ExpTextRMSNorm` in the
//!   reference computes `x̂ · (1 + w)`; the converter folds the `+1` into the
//!   stored tensor. [`Qwen4ExpConfig::rms_norm_offset`] says whether the
//!   builder still has to add it (HF `true`, GGUF `false`). The *gated* norm
//!   (`ssm_norm`) is a plain `w` in both and is never offset.
//! * **The linear-attention value heads are re-ordered.** HF stores the `48`
//!   v-heads grouped by their `16` k-heads; the converter permutes them to
//!   tiled order so the GQA expansion is a whole-block repeat. See
//!   [`crate::gdn`].
//!
//! The 45-billion-parameter n-gram table is addressed by a hash whose
//! constants the checkpoint *also* carries, so [`PleHash`] derives them from
//! `(seed, vocab_size, layer)` and the GGUF values cross-check the derivation
//! rather than being trusted blindly.

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

/// GGUF `general.architecture` values this crate claims.
pub const ACCEPTED_ARCHES: &[&str] = &["qwen4exp"];

/// HF `model_type` of the multimodal wrapper.
pub const HF_MODEL_TYPE: &str = "qwen4_exp";
/// HF `model_type` of `config.json["text_config"]`.
pub const HF_TEXT_MODEL_TYPE: &str = "qwen4_exp_text";

/// Which token mixer a decoder layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Qwen3.5 GatedDeltaNet linear attention.
    Linear,
    /// Gated GQA self-attention behind the QSA block indexer. The checkpoint
    /// spells this `full_attention`; the reference renames it
    /// `indexed_attention` on load because the indexer is never absent.
    Indexed,
}

/// Qwen3.8-Flash-Next text-decoder configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4ExpConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Decoder layers **excluding** any MTP block.
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Per-head width of the gated attention layers (`key_length` ==
    /// `value_length` == 256).
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    /// HF keeps zero-centred norm gammas (`x̂ · (1 + w)`); the GGUF converter
    /// folds the `+1` in. `true` means the builder must still add it.
    pub rms_norm_offset: bool,

    // ── position encoding ───────────────────────────────────
    pub rope_theta: f32,
    /// `head_dim * partial_rotary_factor` — only the first 64 of 256 dims
    /// rotate.
    pub rope_dim_count: usize,
    /// Interleaved M-RoPE sections, `[11, 11, 10, 0]`.
    pub rope_dim_sections: Vec<usize>,
    pub mrope_interleaved: bool,

    // ── per-layer schedule ──────────────────────────────────
    /// One entry per [`Self::num_hidden_layers`].
    pub layer_types: Vec<LayerKind>,
    /// QSA block size per layer; `0` on a linear-attention layer.
    pub compress_ratios: Vec<usize>,

    // ── GatedDeltaNet linear attention ──────────────────────
    pub ssm_conv_kernel: usize,
    /// Per-head width of q/k/v inside the scan (`linear_*_head_dim`).
    pub ssm_state_size: usize,
    /// `linear_num_key_heads`.
    pub ssm_group_count: usize,
    /// `linear_num_value_heads`.
    pub ssm_time_step_rank: usize,
    /// `linear_num_value_heads * linear_value_head_dim`.
    pub ssm_inner_size: usize,
    /// `output_gate_type == "sigmoid"`; Qwen3.5's GDN gates with SiLU instead.
    pub output_gate_sigmoid: bool,

    // ── hyper-connections ───────────────────────────────────
    /// Parallel residual streams (`hc_count`).
    pub hc_count: usize,
    /// Rank of the learned input mixer (`hc_lowrank`).
    pub hc_lowrank: usize,

    // ── QSA indexer ─────────────────────────────────────────
    pub indexer_n_heads: usize,
    pub indexer_head_dim: usize,
    /// Tokens selected out of complete blocks (`indexer_budget`).
    pub indexer_budget: usize,
    /// Keys averaged into one index block (`indexer_compress_ratio`).
    pub indexer_compress_ratio: usize,

    // ── PLE n-gram embeddings ───────────────────────────────
    /// Zero-based layer indices carrying a PLE module.
    pub ple_layers: Vec<usize>,
    pub ple_ngram_size: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_conv_kernel: usize,
    /// Per-head n-gram embedding width (`ple_embed_dim / ple_n_heads`).
    pub ple_head_dim: usize,
    /// The token the hash resets its window on.
    pub ple_eos_token_id: u32,
    /// Stands in for an image placeholder's id when hashing; `0` = unset.
    pub ple_image_token_id: u32,
    pub ngram_vocab_size_base: u64,
    pub make_ngram_vocab_size_divisible_by: u64,
    /// Seed the per-layer hash multipliers are derived from.
    pub seed: u64,
    /// Hash constants read from the checkpoint, when it carries them.
    pub ple_constants: Option<PleConstants>,

    // ── MoE ─────────────────────────────────────────────────
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub expert_ffn_size: usize,
    pub shared_expert_ffn_size: usize,
    pub norm_topk_prob: bool,
    pub expert_weights_scale: f32,

    // ── MTP ─────────────────────────────────────────────────
    pub num_nextn_predict_layers: usize,
    /// Build the MTP block into the flow (off by default).
    pub with_mtp: bool,
}

/// The PLE hash constants as the checkpoint spells them.
///
/// Redundant with [`PleHash`], deliberately: the reference derives these from
/// `seed` and `vocab_size`, the converter copies them out of the checkpoint's
/// int64 buffers, and [`Qwen4ExpConfig::validate`] checks the two agree. A
/// mismatch means the hash would address the wrong rows of a 320-million-row
/// table — which produces plausible-looking embeddings and no error at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleConstants {
    /// One per n-gram position, `ngram_size` of them.
    pub layer_multipliers: Vec<u64>,
    /// First row of each hash head's slice of the shared table.
    pub head_offsets: Vec<u64>,
    /// Prime modulus of each hash head.
    pub head_vocab_sizes: Vec<u64>,
}

// SplitMix64, as `transformers` spells it in `_build_layer_multipliers`.
const MASK64: u64 = u64::MAX;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
/// `_PRIME_1` — strides the seed across PLE layers.
const PRIME_1: u64 = 10007;

fn splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(SPLITMIX_GAMMA) & MASK64;
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1) & MASK64;
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2) & MASK64;
    (v ^ (v >> 31)) & MASK64
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3u64;
    while d.saturating_mul(d) <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// `_find_nth_prime_after` — the `count`-th prime strictly greater than
/// `start`.
fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

/// The hashing half of one PLE module, derived rather than loaded.
///
/// `Qwen4ExpTextNGramEmbedding` turns a token window into one table row per
/// hash head:
///
/// ```text
///   mixed_n = (t[0]·m[0]) ^ (t[1]·m[1]) ^ … ^ (t[n-1]·m[n-1])     wrapping u64
///   row[h]  = mixed_{n(h)} mod vocab[h] + offset[h]
/// ```
///
/// with `n(h) = 2 + h / heads_per_ngram`, so the first `heads_per_ngram` heads
/// hash bigrams and the next block trigrams. `t[j]` is the token `j` positions
/// back, forced to EOS once the walk crosses an EOS or the start of the
/// sequence — the reference's `_shift_right_ignore_eos`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleHash {
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    /// `(ngram_size - 1) * heads_per_ngram`.
    pub n_heads: usize,
    pub layer_multipliers: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    /// `ceil(Σ head_vocab_sizes / divisor) · divisor` — the table's row count.
    pub padded_rows: u64,
    pub eos_token_id: u32,
}

impl PleHash {
    /// Derive the constants for the `ple_layer_index`-th PLE module.
    ///
    /// `ple_layer_index` is the module's position in `ple_layer_ids`, **not**
    /// the decoder layer it sits on: it strides both the seed
    /// (`seed + 10007 · index`) and the global hash-head numbering
    /// (`index · n_heads + h`), so two PLE layers never share a table slice.
    pub fn derive(cfg: &Qwen4ExpConfig, ple_layer_index: usize) -> Self {
        let n_heads = cfg.ple_n_heads();
        let max_long = (1u64 << 63) - 1;
        let multiplier_max = max_long / (cfg.vocab_size as u64).max(1);
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = cfg.seed.wrapping_add(PRIME_1 * ple_layer_index as u64);
        let layer_multipliers: Vec<u64> = (0..cfg.ple_ngram_size)
            .map(|i| {
                let v = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(i as u64 + 1)) & MASK64;
                2 * (splitmix64(v) % half_bound) + 1
            })
            .collect();

        let mut head_vocab_sizes = Vec::with_capacity(n_heads);
        let mut head_offsets = Vec::with_capacity(n_heads);
        let prior_heads = ple_layer_index * n_heads;
        let mut total = (0..prior_heads)
            .map(|global| nth_prime_after(cfg.ngram_vocab_size_base - 1, global + 1))
            .sum::<u64>();
        for h in 0..n_heads {
            let global = ple_layer_index * n_heads + h;
            let size = nth_prime_after(cfg.ngram_vocab_size_base - 1, global + 1);
            head_vocab_sizes.push(size);
            head_offsets.push(total);
            total += size;
        }
        let div = cfg.make_ngram_vocab_size_divisible_by.max(1);
        let padded_rows = total.div_ceil(div) * div;

        Self {
            ngram_size: cfg.ple_ngram_size,
            heads_per_ngram: cfg.ple_heads_per_ngram,
            n_heads,
            layer_multipliers,
            head_offsets,
            head_vocab_sizes,
            padded_rows,
            eos_token_id: cfg.ple_eos_token_id,
        }
    }

    /// Hash one token window into `n_heads` table rows.
    ///
    /// `window[j]` is the token `j` positions before the current one
    /// (`window[0]` being the token itself), already EOS-truncated — see
    /// [`crate::ple::ngram_rows`], which does the truncation and calls this.
    pub fn rows(&self, window: &[u64]) -> Vec<u32> {
        debug_assert_eq!(window.len(), self.ngram_size);
        let mut out = vec![0u32; self.n_heads];
        for n in 2..=self.ngram_size {
            // The reference always starts the fold from position 0's product
            // and XORs positions `1..n` in, so a bigram and a trigram share
            // their first term.
            let mut mixed = window[0].wrapping_mul(self.layer_multipliers[0]);
            for j in 1..n {
                mixed ^= window[j].wrapping_mul(self.layer_multipliers[j]);
            }
            let base = (n - 2) * self.heads_per_ngram;
            for g in 0..self.heads_per_ngram {
                let h = base + g;
                out[h] = (mixed % self.head_vocab_sizes[h] + self.head_offsets[h]) as u32;
            }
        }
        out
    }
}

impl Qwen4ExpConfig {
    /// `ssm_group_count` — the GatedDeltaNet's key/query heads.
    pub fn n_k_heads(&self) -> usize {
        self.ssm_group_count
    }

    /// `ssm_time_step_rank` — the GatedDeltaNet's value heads.
    pub fn n_v_heads(&self) -> usize {
        self.ssm_time_step_rank
    }

    /// Width of the GDN q (and k) half of `attn_qkv`.
    pub fn gdn_key_dim(&self) -> usize {
        self.ssm_state_size * self.n_k_heads()
    }

    /// Width of the GDN v half of `attn_qkv`, and of `attn_gate`.
    pub fn gdn_value_dim(&self) -> usize {
        self.ssm_state_size * self.n_v_heads()
    }

    /// `attn_qkv` output width — `2·key_dim + value_dim`, which is also the
    /// depthwise convolution's channel count.
    pub fn gdn_conv_dim(&self) -> usize {
        2 * self.gdn_key_dim() + self.gdn_value_dim()
    }

    /// Width of the hyper-connection residual, `hc_count · hidden_size`.
    pub fn hc_dim(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    /// `(ngram_size − 1) · heads_per_ngram` — hash heads per PLE module.
    pub fn ple_n_heads(&self) -> usize {
        (self.ple_ngram_size.saturating_sub(1)) * self.ple_heads_per_ngram
    }

    /// Total n-gram embedding width a PLE module consumes.
    pub fn ple_embed_dim(&self) -> usize {
        self.ple_n_heads() * self.ple_head_dim
    }

    /// Positions of history the dilated PLE convolution reaches back over.
    pub fn ple_conv_state_len(&self) -> usize {
        (self.ple_conv_kernel.saturating_sub(1)) * self.ple_ngram_size
    }

    /// Index of `layer` within `ple_layers`, if it has a PLE module.
    pub fn ple_layer_index(&self, layer: usize) -> Option<usize> {
        self.ple_layers.iter().position(|&l| l == layer)
    }

    pub fn layer_kind(&self, layer: usize) -> LayerKind {
        self.layer_types[layer]
    }

    /// Index blocks the QSA budget buys — `indexer_budget / compress_ratio`.
    pub fn qsa_block_topk(&self) -> usize {
        self.indexer_budget / self.indexer_compress_ratio.max(1)
    }

    /// Whether QSA selection is the identity at this prefill length, making
    /// the sparse mask exactly the causal mask.
    ///
    /// A query at position `q` sees `q + 1` tokens, which form
    /// `(q + 1) / ratio` complete blocks plus a tail; the selection keeps
    /// `min(block_topk, complete)` blocks. So whenever
    /// `seq / ratio <= block_topk` no query ever has more complete blocks than
    /// the budget, every one is picked, and blocks ∪ tail is exactly `0..=q`.
    /// This is an algebraic identity, not an approximation, so the builder can
    /// drop the whole indexer and use the fused causal kernel.
    pub fn qsa_is_dense(&self, seq: usize) -> bool {
        seq / self.indexer_compress_ratio.max(1) <= self.qsa_block_topk()
    }

    /// Blocks **this checkpoint** carries.
    ///
    /// `num_hidden_layers`, plus the MTP block only when [`Self::with_mtp`] is
    /// set. The MTP is gated rather than always counted because this
    /// architecture ships it as a *separate file*: `unsloth`'s and `ggml-org`'s
    /// conversions both run the converter with `--no-mtp` and publish
    /// `mtp-*.gguf` alongside, so a trunk GGUF has exactly
    /// `num_hidden_layers` blocks even though the HF config declares
    /// `mtp_num_hidden_layers = 1`.
    pub fn block_count(&self) -> usize {
        self.num_hidden_layers
            + if self.with_mtp {
                self.num_nextn_predict_layers
            } else {
                0
            }
    }

    /// Every tensor a checkpoint must carry for `block`, as GGUF names without
    /// the `blk.{i}.` prefix.
    ///
    /// The checkpoint contract stated once, in config terms, so it can be
    /// checked against a published file without downloading one — see
    /// `tests/tensor_manifest.rs`.
    pub fn block_tensor_names(&self, block: usize) -> Vec<String> {
        let is_mtp = block >= self.num_hidden_layers;
        let mut v: Vec<String> = Vec::new();

        // Hyper-connections stand in for every layer norm, so there is no
        // `attn_norm` / `ffn_norm` anywhere in this architecture.
        for site in ["hc_attn", "hc_ffn"] {
            for suffix in ["norm", "down", "up", "inject"] {
                v.push(format!("{site}_{suffix}.weight"));
            }
        }

        // The MTP block is a full-attention QSA layer whatever the trunk's
        // schedule says, plus its own embed/hidden projections and mixer.
        let kind = if is_mtp {
            LayerKind::Indexed
        } else {
            self.layer_kind(block)
        };
        match kind {
            LayerKind::Linear => v.extend(
                [
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_beta.weight",
                    "ssm_alpha.weight",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ]
                .map(String::from),
            ),
            LayerKind::Indexed => v.extend(
                [
                    "attn_q.weight",
                    "attn_q_norm.weight",
                    "attn_k.weight",
                    "attn_k_norm.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                    "indexer.q_proj.weight",
                    "indexer.k_proj.weight",
                    "indexer.q_norm.weight",
                    "indexer.k_norm.weight",
                ]
                .map(String::from),
            ),
        }

        if !is_mtp && self.ple_layer_index(block).is_some() {
            v.extend(
                [
                    "ple_key.weight",
                    "ple_value.weight",
                    "ple_norm_key.weight",
                    "ple_norm_query.weight",
                    "ple_norm_conv.weight",
                    "ple_conv1d.weight",
                ]
                .map(String::from),
            );
        }

        // Every block is MoE — there is no leading dense stretch.
        v.extend(
            [
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "ffn_gate_inp_shexp.weight",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
            ]
            .map(String::from),
        );

        if is_mtp {
            v.extend(
                [
                    "nextn.eh_proj.weight",
                    "nextn.enorm.weight",
                    "nextn.hnorm.weight",
                    "nextn.hc_head_norm.weight",
                    "nextn.hc_head_down.weight",
                    "nextn.hc_head_up.weight",
                ]
                .map(String::from),
            );
        }

        v.sort();
        v
    }

    /// The whole checkpoint's tensor names, `blk.{i}.`-prefixed, plus the
    /// global ones. See [`Self::block_tensor_names`].
    ///
    /// Note what is *absent*: `output_norm.weight`. The final
    /// hyper-connection mixer (`output_hc_*`) is the output norm — there is no
    /// separate one.
    pub fn tensor_manifest(&self) -> Vec<String> {
        let mut v = vec![
            "token_embd.weight".to_string(),
            "output_hc_norm.weight".to_string(),
            "output_hc_down.weight".to_string(),
            "output_hc_up.weight".to_string(),
        ];
        if !self.tie_word_embeddings {
            v.push("output.weight".to_string());
        }
        if !self.ple_layers.is_empty() {
            v.push("per_layer_token_embd.weight".to_string());
        }
        for i in 0..self.block_count() {
            v.extend(
                self.block_tensor_names(i)
                    .into_iter()
                    .map(|s| format!("blk.{i}.{s}")),
            );
        }
        v.sort();
        v
    }

    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("rope_dim_count", self.rope_dim_count),
            ("ssm_conv_kernel", self.ssm_conv_kernel),
            ("ssm_state_size", self.ssm_state_size),
            ("ssm_group_count", self.ssm_group_count),
            ("ssm_time_step_rank", self.ssm_time_step_rank),
            ("ssm_inner_size", self.ssm_inner_size),
            ("indexer_n_heads", self.indexer_n_heads),
            ("indexer_head_dim", self.indexer_head_dim),
            ("indexer_budget", self.indexer_budget),
            ("num_experts", self.num_experts),
            ("expert_ffn_size", self.expert_ffn_size),
            ("shared_expert_ffn_size", self.shared_expert_ffn_size),
        ] {
            if value == 0 {
                bail!("qwen4exp: {name} must be > 0");
            }
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            bail!("qwen4exp: rms_norm_eps must be finite and > 0");
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            bail!("qwen4exp: rope_theta must be finite and > 0");
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            bail!(
                "qwen4exp: num_attention_heads ({}) must be divisible by \
                 num_key_value_heads ({})",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }
        if self.layer_types.len() != self.num_hidden_layers {
            bail!(
                "qwen4exp: layer_types has {} entries, expected num_hidden_layers = {}",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        if self.compress_ratios.len() < self.num_hidden_layers {
            bail!(
                "qwen4exp: compress_ratios has {} entries, need at least {}",
                self.compress_ratios.len(),
                self.num_hidden_layers
            );
        }
        // A count of 1 has nothing to mix; the reference config rejects it.
        if self.hc_count <= 1 {
            bail!(
                "qwen4exp: hc_count must be > 1, got {} (hyper-connections are not optional here)",
                self.hc_count
            );
        }
        if self.hc_lowrank == 0 {
            bail!("qwen4exp: hc_lowrank must be > 0");
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            bail!(
                "qwen4exp: num_experts_per_tok ({}) must be in 1..={}",
                self.num_experts_per_tok,
                self.num_experts
            );
        }
        if !self.norm_topk_prob {
            // The gate kernel always renormalizes a `top_k > 1` selection, so
            // an unnormalized router would be silently wrong.
            bail!("qwen4exp: norm_topk_prob = false is not implemented");
        }
        if self.indexer_compress_ratio <= 1 {
            bail!(
                "qwen4exp: QSA needs a compress ratio above 1, got {}",
                self.indexer_compress_ratio
            );
        }
        if !self
            .indexer_budget
            .is_multiple_of(self.indexer_compress_ratio)
        {
            bail!(
                "qwen4exp: indexer_budget ({}) must be divisible by the compress ratio ({})",
                self.indexer_budget,
                self.indexer_compress_ratio
            );
        }
        if self.rope_dim_count > self.indexer_head_dim {
            // The indexer rotates its own query with the attention RoPE, so
            // the rotary half has to fit inside the index head.
            bail!(
                "qwen4exp: rope_dim_count ({}) exceeds indexer_head_dim ({})",
                self.rope_dim_count,
                self.indexer_head_dim
            );
        }
        if self.rope_dim_count > self.head_dim || !self.rope_dim_count.is_multiple_of(2) {
            bail!(
                "qwen4exp: rope_dim_count ({}) must be even and no larger than head_dim ({})",
                self.rope_dim_count,
                self.head_dim
            );
        }
        if self.ssm_state_size * self.n_v_heads() != self.ssm_inner_size {
            bail!(
                "qwen4exp: ssm_inner_size ({}) != state_size ({}) * v_heads ({})",
                self.ssm_inner_size,
                self.ssm_state_size,
                self.n_v_heads()
            );
        }
        if !self.n_v_heads().is_multiple_of(self.n_k_heads().max(1)) {
            bail!(
                "qwen4exp: v_heads ({}) must be a multiple of k_heads ({})",
                self.n_v_heads(),
                self.n_k_heads()
            );
        }
        // Layer schedule and QSA ratios have to say the same thing: a layer
        // with a ratio but no attention weights (or the reverse) would load.
        for (i, &kind) in self.layer_types.iter().enumerate() {
            let ratio = self.compress_ratios[i];
            match kind {
                LayerKind::Indexed if ratio != self.indexer_compress_ratio => bail!(
                    "qwen4exp: layer {i} is an attention layer but its compress ratio is \
                     {ratio}, not {}",
                    self.indexer_compress_ratio
                ),
                LayerKind::Linear if ratio != 0 => bail!(
                    "qwen4exp: layer {i} is a linear-attention layer but carries compress \
                     ratio {ratio}"
                ),
                _ => {}
            }
        }
        for &l in &self.ple_layers {
            if l >= self.num_hidden_layers {
                bail!(
                    "qwen4exp: PLE layer {l} is out of range (num_hidden_layers = {})",
                    self.num_hidden_layers
                );
            }
            // The PLE convolution history is a recurrent-cache row, which only
            // linear-attention layers have.
            if self.layer_types[l] != LayerKind::Linear {
                bail!("qwen4exp: PLE layer {l} is not a linear-attention layer");
            }
        }
        if !self.ple_layers.is_empty() {
            if self.ple_n_heads() == 0 {
                bail!("qwen4exp: PLE is enabled but ngram_size/heads_per_ngram give no heads");
            }
            if self.ple_head_dim == 0 || self.ple_conv_kernel == 0 {
                bail!("qwen4exp: PLE head width and convolution kernel must be > 0");
            }
        }

        // Cross-check the derived hash against what the checkpoint carries.
        // Getting this wrong addresses the wrong rows of a 320-million-row
        // table, which looks exactly like a working model.
        if let Some(c) = &self.ple_constants {
            for (idx, _) in self.ple_layers.iter().enumerate() {
                let h = PleHash::derive(self, idx);
                if idx == 0 {
                    if c.layer_multipliers.len() < h.layer_multipliers.len()
                        || c.layer_multipliers[..h.layer_multipliers.len()]
                            != h.layer_multipliers[..]
                    {
                        bail!(
                            "qwen4exp: derived PLE multipliers {:?} do not match the \
                             checkpoint's {:?}",
                            h.layer_multipliers,
                            c.layer_multipliers
                        );
                    }
                    if c.head_offsets.len() < h.head_offsets.len()
                        || c.head_offsets[..h.head_offsets.len()] != h.head_offsets[..]
                    {
                        bail!("qwen4exp: derived PLE head offsets do not match the checkpoint");
                    }
                    if c.head_vocab_sizes.len() < h.head_vocab_sizes.len()
                        || c.head_vocab_sizes[..h.head_vocab_sizes.len()] != h.head_vocab_sizes[..]
                    {
                        bail!("qwen4exp: derived PLE head vocab sizes do not match the checkpoint");
                    }
                }
            }
        }
        Ok(())
    }
}

impl Qwen4ExpConfig {
    /// The published **Qwen3.8-Flash-Next** text configuration, spelled out.
    ///
    /// Every value here is read off `Qwen/Qwen3.8-Flash-Next`'s `config.json`
    /// and the `qwen4exp.*` metadata of its GGUF conversions. It exists so
    /// that the parts of this crate derived from config alone — the PLE hash
    /// constants, the tensor manifest, the QSA budget algebra — can be checked
    /// against the real model without downloading 131 safetensors shards, and
    /// so a caller can build a correctly-shaped synthetic model with one call.
    pub fn qwen38_flash_next() -> Self {
        let layer_types: Vec<LayerKind> = (0usize..48)
            .map(|i| {
                if (i + 1).is_multiple_of(4) {
                    LayerKind::Indexed
                } else {
                    LayerKind::Linear
                }
            })
            .collect();
        let compress_ratios = layer_types
            .iter()
            .map(|k| match k {
                LayerKind::Indexed => 4,
                LayerKind::Linear => 0,
            })
            .collect();
        Self {
            vocab_size: 248320,
            hidden_size: 2560,
            num_hidden_layers: 48,
            num_attention_heads: 24,
            num_key_value_heads: 2,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            tie_word_embeddings: false,
            rms_norm_offset: false,
            rope_theta: 1e7,
            rope_dim_count: 64,
            rope_dim_sections: vec![11, 11, 10, 0],
            mrope_interleaved: true,
            layer_types,
            compress_ratios,
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 48,
            ssm_inner_size: 6144,
            output_gate_sigmoid: true,
            hc_count: 4,
            hc_lowrank: 320,
            indexer_n_heads: 4,
            indexer_head_dim: 128,
            indexer_budget: 2048,
            indexer_compress_ratio: 4,
            ple_layers: vec![1],
            ple_ngram_size: 3,
            ple_heads_per_ngram: 8,
            ple_conv_kernel: 4,
            ple_head_dim: 160,
            ple_eos_token_id: 248044,
            ple_image_token_id: 248056,
            ngram_vocab_size_base: 20_000_000,
            make_ngram_vocab_size_divisible_by: 128,
            seed: 1234,
            ple_constants: None,
            num_experts: 512,
            num_experts_per_tok: 10,
            expert_ffn_size: 640,
            shared_expert_ffn_size: 640,
            norm_topk_prob: true,
            expert_weights_scale: 1.0,
            // The architecture has one MTP block. It ships as its own
            // `mtp-*.gguf`, so `with_mtp` (and therefore `block_count`) stays
            // false until a caller asks for it.
            num_nextn_predict_layers: 1,
            with_mtp: false,
        }
    }
}

// ─────────────────────────── GGUF ───────────────────────────

fn meta<'a>(raw: &'a GgufFile, key: &str) -> Option<&'a MetaValue> {
    raw.metadata.get(key)
}

fn meta_u32(raw: &GgufFile, key: &str) -> Result<u32> {
    meta(raw, key)
        .and_then(MetaValue::as_u32)
        .with_context(|| format!("qwen4exp: missing or non-integer GGUF key `{key}`"))
}

fn meta_u32_or(raw: &GgufFile, key: &str, default: u32) -> u32 {
    meta(raw, key)
        .and_then(MetaValue::as_u32)
        .unwrap_or(default)
}

fn as_f32(v: &MetaValue) -> Option<f32> {
    match v {
        MetaValue::F32(x) => Some(*x),
        MetaValue::F64(x) => Some(*x as f32),
        _ => None,
    }
}

fn meta_f32_or(raw: &GgufFile, key: &str, default: f32) -> f32 {
    meta(raw, key).and_then(as_f32).unwrap_or(default)
}

fn meta_bool_or(raw: &GgufFile, key: &str, default: bool) -> bool {
    match meta(raw, key) {
        Some(MetaValue::Bool(b)) => *b,
        Some(other) => other.as_u32().map(|v| v != 0).unwrap_or(default),
        None => default,
    }
}

fn meta_array<'a>(raw: &'a GgufFile, key: &str) -> Option<&'a [MetaValue]> {
    match meta(raw, key) {
        Some(MetaValue::Array(v)) => Some(v.as_slice()),
        _ => None,
    }
}

fn meta_u32_vec(raw: &GgufFile, key: &str) -> Option<Vec<usize>> {
    meta_array(raw, key).map(|a| a.iter().map(|v| v.as_u32().unwrap_or(0) as usize).collect())
}

/// The hash constants are up to 45 bits wide, so they are read at u64 — an
/// `as_u32` read would silently truncate every multiplier.
fn meta_u64_vec(raw: &GgufFile, key: &str) -> Option<Vec<u64>> {
    meta_array(raw, key).map(|a| a.iter().filter_map(MetaValue::as_u64).collect())
}

impl Qwen4ExpConfig {
    /// Parse a `qwen4exp` GGUF's metadata.
    ///
    /// Metadata only, so this works on the metadata-carrying first shard of a
    /// split (unsloth's `…-00001-of-000NN.gguf` holds all 67 KV pairs and zero
    /// tensors).
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = meta(raw, "general.architecture")
            .and_then(MetaValue::as_str)
            .unwrap_or_default();
        if !ACCEPTED_ARCHES.contains(&arch) {
            bail!(
                "Qwen4ExpConfig::from_gguf: expected general.architecture ∈ \
                 {ACCEPTED_ARCHES:?}, got {arch:?}"
            );
        }

        let block_count = meta_u32(raw, "qwen4exp.block_count")? as usize;
        let num_nextn_predict_layers =
            meta_u32_or(raw, "qwen4exp.nextn_predict_layers", 0) as usize;
        if num_nextn_predict_layers >= block_count {
            bail!(
                "qwen4exp: nextn_predict_layers ({num_nextn_predict_layers}) >= \
                 block_count ({block_count})"
            );
        }
        let num_hidden_layers = block_count - num_nextn_predict_layers;

        let head_dim = meta_u32(raw, "qwen4exp.attention.key_length")? as usize;
        let value_length = meta_u32_or(raw, "qwen4exp.attention.value_length", head_dim as u32);
        if value_length as usize != head_dim {
            bail!(
                "qwen4exp: asymmetric attention head ({head_dim} key vs {value_length} value) \
                 is not part of this architecture"
            );
        }

        // `compress_ratios` is the layer schedule: a non-zero entry is a QSA
        // (full-attention) layer, zero a linear-attention one. It is sized for
        // every block, so the MTP tail is trimmed off here.
        let compress_ratios =
            meta_u32_vec(raw, "qwen4exp.attention.compress_ratios").unwrap_or_default();
        let indexer_compress_ratio = {
            let nz: Vec<usize> = compress_ratios
                .iter()
                .copied()
                .filter(|&r| r != 0)
                .collect();
            match nz.first() {
                // One block size for the whole model; llama.cpp rejects a mix
                // and so does this.
                Some(&r) => {
                    if nz.iter().any(|&x| x != r) {
                        bail!("qwen4exp: QSA layers must share one compress ratio, got {nz:?}");
                    }
                    r
                }
                None => meta_u32_or(raw, "qwen4exp.attention.indexer.compress_ratio", 4) as usize,
            }
        };
        let layer_types: Vec<LayerKind> = if compress_ratios.len() >= num_hidden_layers {
            compress_ratios[..num_hidden_layers]
                .iter()
                .map(|&r| {
                    if r == 0 {
                        LayerKind::Linear
                    } else {
                        LayerKind::Indexed
                    }
                })
                .collect()
        } else {
            let interval = meta_u32_or(raw, "qwen4exp.full_attention_interval", 4).max(1) as usize;
            (0..num_hidden_layers)
                .map(|i| {
                    if (i + 1).is_multiple_of(interval) {
                        LayerKind::Indexed
                    } else {
                        LayerKind::Linear
                    }
                })
                .collect()
        };
        // Reconstruct the ratios when the key was absent, so the two always
        // agree and `validate` can check them against each other.
        let compress_ratios: Vec<usize> = if compress_ratios.len() >= num_hidden_layers {
            compress_ratios
        } else {
            layer_types
                .iter()
                .map(|k| match k {
                    LayerKind::Indexed => indexer_compress_ratio,
                    LayerKind::Linear => 0,
                })
                .collect()
        };

        let ple_layers: Vec<usize> = {
            let mut v = meta_u32_vec(raw, "qwen4exp.ple.layers").unwrap_or_default();
            v.sort_unstable();
            v.dedup();
            v
        };
        let ple_constants = (!ple_layers.is_empty())
            .then(|| {
                Some(PleConstants {
                    layer_multipliers: meta_u64_vec(raw, "qwen4exp.ple.layer_multipliers")?,
                    head_offsets: meta_u64_vec(raw, "qwen4exp.ple.head_offsets")?,
                    head_vocab_sizes: meta_u64_vec(raw, "qwen4exp.ple.head_vocab_sizes")?,
                })
            })
            .flatten();

        let rope_dim_sections =
            meta_u32_vec(raw, "qwen4exp.rope.dimension_sections").unwrap_or_default();

        let cfg = Self {
            vocab_size: meta_u32(raw, "qwen4exp.vocab_size").or_else(|_| {
                // The converter writes the vocab through the tokenizer
                // rather than an arch key on some files.
                meta_array(raw, "tokenizer.ggml.tokens")
                    .map(|t| t.len() as u32)
                    .context("qwen4exp: no vocab_size and no tokenizer token list")
            })? as usize,
            hidden_size: meta_u32(raw, "qwen4exp.embedding_length")? as usize,
            num_hidden_layers,
            num_attention_heads: meta_u32(raw, "qwen4exp.attention.head_count")? as usize,
            num_key_value_heads: meta_u32(raw, "qwen4exp.attention.head_count_kv")? as usize,
            head_dim,
            rms_norm_eps: meta_f32_or(raw, "qwen4exp.attention.layer_norm_rms_epsilon", 1e-6),
            max_position_embeddings: meta_u32_or(raw, "qwen4exp.context_length", 1 << 18) as usize,
            // qwen4exp ships an untied `output.weight`; the loader falls back
            // to the embedding only when the tensor is missing.
            tie_word_embeddings: false,
            // The converter already folded `1 +` into every norm gamma.
            rms_norm_offset: false,

            rope_theta: meta_f32_or(raw, "qwen4exp.rope.freq_base", 10_000_000.0),
            rope_dim_count: meta_u32_or(raw, "qwen4exp.rope.dimension_count", 0) as usize,
            rope_dim_sections,
            // Qwen3.5 and its descendants always interleave the M-RoPE
            // sections (`Qwen3_5RotaryEmbedding`); there is no GGUF key for it.
            mrope_interleaved: true,

            layer_types,
            compress_ratios,

            ssm_conv_kernel: meta_u32(raw, "qwen4exp.ssm.conv_kernel")? as usize,
            ssm_state_size: meta_u32(raw, "qwen4exp.ssm.state_size")? as usize,
            ssm_group_count: meta_u32(raw, "qwen4exp.ssm.group_count")? as usize,
            ssm_time_step_rank: meta_u32(raw, "qwen4exp.ssm.time_step_rank")? as usize,
            ssm_inner_size: meta_u32(raw, "qwen4exp.ssm.inner_size")? as usize,
            // `output_gate_type = sigmoid` is the one numerical difference from
            // Qwen3.5's GDN. It has no GGUF key: llama.cpp hardcodes the
            // sigmoid in `qwen4exp.cpp build_norm_gated`.
            output_gate_sigmoid: true,

            hc_count: meta_u32(raw, "qwen4exp.hyper_connection.count")? as usize,
            hc_lowrank: meta_u32(raw, "qwen4exp.hyper_connection.low_rank")? as usize,

            indexer_n_heads: meta_u32(raw, "qwen4exp.attention.indexer.head_count")? as usize,
            indexer_head_dim: meta_u32(raw, "qwen4exp.attention.indexer.key_length")? as usize,
            indexer_budget: meta_u32(raw, "qwen4exp.attention.indexer.top_k")? as usize,
            indexer_compress_ratio,

            ple_ngram_size: meta_u32_or(raw, "qwen4exp.ple.ngram_size", 3) as usize,
            ple_heads_per_ngram: meta_u32_or(raw, "qwen4exp.ple.heads_per_ngram", 8) as usize,
            ple_conv_kernel: meta_u32_or(raw, "qwen4exp.ple.conv_kernel", 4) as usize,
            ple_head_dim: meta_u32_or(raw, "qwen4exp.embedding_length_per_layer_input", 0) as usize,
            ple_eos_token_id: meta_u32_or(raw, "qwen4exp.ple.eos_token_id", 0),
            ple_image_token_id: meta_u32_or(raw, "qwen4exp.ple.image_token_id", 0),
            ngram_vocab_size_base: 20_000_000,
            make_ngram_vocab_size_divisible_by: 128,
            seed: 1234,
            ple_layers,
            ple_constants,

            num_experts: meta_u32(raw, "qwen4exp.expert_count")? as usize,
            num_experts_per_tok: meta_u32(raw, "qwen4exp.expert_used_count")? as usize,
            expert_ffn_size: meta_u32(raw, "qwen4exp.expert_feed_forward_length").or_else(|_| {
                meta_array(raw, "qwen4exp.expert_feed_forward_length")
                    .and_then(|a| a.first())
                    .and_then(MetaValue::as_u32)
                    .context("qwen4exp: expert_feed_forward_length missing")
            })? as usize,
            shared_expert_ffn_size: meta_u32(raw, "qwen4exp.expert_shared_feed_forward_length")?
                as usize,
            norm_topk_prob: meta_bool_or(raw, "qwen4exp.expert_weights_norm", true),
            expert_weights_scale: meta_f32_or(raw, "qwen4exp.expert_weights_scale", 1.0),

            num_nextn_predict_layers,
            with_mtp: false,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a `qwen4exp` GGUF at `path` (metadata only — no tensor data is
    /// mapped).
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = GgufFile::from_path(path)
            .with_context(|| format!("qwen4exp: reading GGUF metadata from {path:?}"))?;
        Self::from_gguf(&raw)
    }

    /// Parse a split GGUF from all of its shard paths.
    pub fn from_gguf_paths(paths: &[impl AsRef<Path>]) -> Result<Self> {
        let raw =
            GgufFile::from_split_paths(paths).context("qwen4exp: reading split GGUF metadata")?;
        Self::from_gguf(&raw)
    }
}

// ─────────────────────────── HF config.json ───────────────────────────

#[derive(Debug, Deserialize)]
struct HfRopeParameters {
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
    #[serde(default)]
    mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    mrope_interleaved: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct HfMtp {
    #[serde(default)]
    num_hidden_layers: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct HfTextConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    head_dim: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,

    #[serde(default)]
    rope_parameters: Option<HfRopeParameters>,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,

    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    full_attention_interval: Option<usize>,

    #[serde(default = "default_conv")]
    linear_conv_kernel_dim: usize,
    #[serde(default = "default_lin_head")]
    linear_key_head_dim: usize,
    #[serde(default = "default_lin_head")]
    linear_value_head_dim: usize,
    #[serde(default = "default_k_heads")]
    linear_num_key_heads: usize,
    #[serde(default = "default_v_heads")]
    linear_num_value_heads: usize,
    #[serde(default)]
    output_gate_type: Option<String>,
    #[serde(default = "default_act")]
    hidden_act: String,

    #[serde(default = "default_hc_count")]
    hc_count: usize,
    #[serde(default = "default_hc_lowrank")]
    hc_lowrank: usize,

    #[serde(default)]
    indexer_n_heads: Option<usize>,
    #[serde(default)]
    indexer_head_dim: Option<usize>,
    #[serde(default)]
    indexer_budget: Option<usize>,
    #[serde(default)]
    indexer_compress_ratio: Option<usize>,

    #[serde(default)]
    ple_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    ple_embed_dim: Option<usize>,
    #[serde(default = "default_conv")]
    ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram")]
    ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    heads_per_ngram: usize,
    #[serde(default = "default_ngram_base")]
    ngram_vocab_size_base: u64,
    #[serde(default = "default_ngram_div")]
    make_ngram_vocab_size_divisible_by: u64,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,

    #[serde(default = "default_experts")]
    num_experts: usize,
    #[serde(default = "default_experts_per_tok")]
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,

    #[serde(default)]
    mtp_num_hidden_layers: Option<usize>,
    #[serde(default)]
    mtp: Option<HfMtp>,
}

fn default_head_dim() -> usize {
    256
}
fn default_eps() -> f32 {
    1e-6
}
fn default_conv() -> usize {
    4
}
fn default_lin_head() -> usize {
    128
}
fn default_k_heads() -> usize {
    16
}
fn default_v_heads() -> usize {
    32
}
fn default_act() -> String {
    "silu".to_string()
}
fn default_hc_count() -> usize {
    4
}
fn default_hc_lowrank() -> usize {
    320
}
fn default_ngram() -> usize {
    3
}
fn default_heads_per_ngram() -> usize {
    8
}
fn default_ngram_base() -> u64 {
    20_000_000
}
fn default_ngram_div() -> u64 {
    128
}
fn default_seed() -> u64 {
    1234
}
fn default_experts() -> usize {
    512
}
fn default_experts_per_tok() -> usize {
    10
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct HfConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default)]
    text_config: Option<HfTextConfig>,
    #[serde(flatten)]
    flat: Option<HfTextConfig>,
}

fn first_eos(v: &serde_json::Value) -> Option<u32> {
    match v {
        serde_json::Value::Number(n) => n.as_u64().map(|x| x as u32),
        // The converter takes the LAST entry of a list: the PLE hash resets on
        // the primary EOS, which is the one `generation_config` puts last.
        serde_json::Value::Array(a) => a.last().and_then(first_eos),
        _ => None,
    }
}

impl Qwen4ExpConfig {
    /// Parse a HF `config.json` (either the multimodal wrapper, which nests
    /// `text_config`, or a bare text config).
    pub fn from_hf_json(json: &str) -> Result<Self> {
        let raw: HfConfig =
            serde_json::from_str(json).context("qwen4exp: parsing HF config.json")?;
        let t = raw
            .text_config
            .or(raw.flat)
            .context("qwen4exp: config.json has neither a text_config nor text fields")?;

        let rope = t.rope_parameters.unwrap_or(HfRopeParameters {
            rope_theta: None,
            partial_rotary_factor: None,
            mrope_section: None,
            mrope_interleaved: None,
        });
        let partial = rope
            .partial_rotary_factor
            .or(t.partial_rotary_factor)
            .unwrap_or(1.0);
        let rope_dim_count = (t.head_dim as f32 * partial) as usize;
        // llama.cpp always passes four sections to `ggml_rope_multi`; HF lists
        // three and leaves the (vision) remainder implicit.
        let mut sections = rope
            .mrope_section
            .unwrap_or_else(|| vec![rope_dim_count / 2]);
        while sections.len() < 4 {
            sections.push(0);
        }

        let layer_types: Vec<LayerKind> = match t.layer_types {
            Some(ref v) if v.len() >= t.num_hidden_layers => v[..t.num_hidden_layers]
                .iter()
                .map(|s| match s.as_str() {
                    "linear_attention" => LayerKind::Linear,
                    _ => LayerKind::Indexed,
                })
                .collect(),
            _ => {
                let interval = t.full_attention_interval.unwrap_or(4).max(1);
                (0..t.num_hidden_layers)
                    .map(|i| {
                        if (i + 1).is_multiple_of(interval) {
                            LayerKind::Indexed
                        } else {
                            LayerKind::Linear
                        }
                    })
                    .collect()
            }
        };
        let indexer_compress_ratio = t.indexer_compress_ratio.unwrap_or(4);
        let compress_ratios: Vec<usize> = layer_types
            .iter()
            .map(|k| match k {
                LayerKind::Indexed => indexer_compress_ratio,
                LayerKind::Linear => 0,
            })
            .collect();

        if t.linear_key_head_dim != t.linear_value_head_dim {
            bail!(
                "qwen4exp: linear_key_head_dim ({}) != linear_value_head_dim ({}); the GGUF \
                 contract has one `ssm.state_size` for both",
                t.linear_key_head_dim,
                t.linear_value_head_dim
            );
        }

        // `ple_layer_ids` is ONE-based in the HF config and zero-based in the
        // GGUF. Normalizing here is what lets one struct serve both readers.
        let ple_layers: Vec<usize> = {
            let mut v: Vec<usize> = t
                .ple_layer_ids
                .unwrap_or_default()
                .into_iter()
                .filter(|&l| l >= 1)
                .map(|l| l - 1)
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let ple_n_heads = (t.ngram_size.saturating_sub(1)) * t.heads_per_ngram;
        let ple_embed_dim = t.ple_embed_dim.unwrap_or(t.hidden_size);
        let ple_head_dim = if ple_n_heads == 0 {
            0
        } else {
            if !ple_embed_dim.is_multiple_of(ple_n_heads) {
                bail!(
                    "qwen4exp: ple_embed_dim ({ple_embed_dim}) is not divisible by the \
                     {ple_n_heads} n-gram heads"
                );
            }
            ple_embed_dim / ple_n_heads
        };

        let gate = t.output_gate_type.unwrap_or(t.hidden_act);
        if gate != "sigmoid" && gate != "silu" {
            bail!("qwen4exp: unsupported output_gate_type {gate:?}");
        }

        let nextn = t
            .mtp_num_hidden_layers
            .or_else(|| t.mtp.as_ref().and_then(|m| m.num_hidden_layers))
            .unwrap_or(0);

        let cfg = Self {
            vocab_size: t.vocab_size,
            hidden_size: t.hidden_size,
            num_hidden_layers: t.num_hidden_layers,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            rms_norm_eps: t.rms_norm_eps,
            max_position_embeddings: t.max_position_embeddings.unwrap_or(1 << 18),
            tie_word_embeddings: t.tie_word_embeddings.unwrap_or(false),
            // HF gammas are zero-centred: the builder has to add the 1.
            rms_norm_offset: true,

            rope_theta: rope.rope_theta.unwrap_or(10_000_000.0),
            rope_dim_count,
            rope_dim_sections: sections,
            mrope_interleaved: rope.mrope_interleaved.unwrap_or(true),

            layer_types,
            compress_ratios,

            ssm_conv_kernel: t.linear_conv_kernel_dim,
            ssm_state_size: t.linear_key_head_dim,
            ssm_group_count: t.linear_num_key_heads,
            ssm_time_step_rank: t.linear_num_value_heads,
            ssm_inner_size: t.linear_num_value_heads * t.linear_value_head_dim,
            output_gate_sigmoid: gate == "sigmoid",

            hc_count: t.hc_count,
            hc_lowrank: t.hc_lowrank,

            indexer_n_heads: t.indexer_n_heads.unwrap_or(4),
            indexer_head_dim: t.indexer_head_dim.unwrap_or(128),
            indexer_budget: t.indexer_budget.unwrap_or(2048),
            indexer_compress_ratio,

            ple_ngram_size: t.ngram_size,
            ple_heads_per_ngram: t.heads_per_ngram,
            ple_conv_kernel: t.ple_conv_kernel_size,
            ple_head_dim,
            ple_eos_token_id: t
                .eos_token_id
                .as_ref()
                .and_then(first_eos)
                .unwrap_or_default(),
            ple_image_token_id: raw.image_token_id.unwrap_or(0),
            ngram_vocab_size_base: t.ngram_vocab_size_base,
            make_ngram_vocab_size_divisible_by: t.make_ngram_vocab_size_divisible_by,
            seed: t.seed,
            ple_layers,
            // HF carries the constants as checkpoint buffers, not config, so
            // the derivation stands alone here.
            ple_constants: None,

            num_experts: t.num_experts,
            num_experts_per_tok: t.num_experts_per_tok,
            expert_ffn_size: t.moe_intermediate_size,
            shared_expert_ffn_size: t.shared_expert_intermediate_size,
            norm_topk_prob: t.norm_topk_prob,
            expert_weights_scale: 1.0,

            num_nextn_predict_layers: nextn,
            with_mtp: false,
        };
        let _ = raw.model_type;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a HF `config.json` at `path`.
    pub fn from_hf_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("qwen4exp: reading {path:?}"))?;
        Self::from_hf_json(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flash_next() -> Qwen4ExpConfig {
        Qwen4ExpConfig::qwen38_flash_next()
    }

    #[test]
    fn published_config_validates() {
        flash_next().validate().unwrap();
    }

    /// The PLE hash constants are derived from `(seed, vocab_size)`, and the
    /// published GGUF also writes them out. They must agree exactly: the
    /// multipliers are 45-bit, the moduli are distinct primes near 20 M, and
    /// being off by one prime shifts every row of a 320-million-row table.
    #[test]
    fn derived_ple_hash_matches_the_published_metadata() {
        let cfg = flash_next();
        let h = PleHash::derive(&cfg, 0);

        assert_eq!(
            h.layer_multipliers,
            vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
            "qwen4exp.ple.layer_multipliers"
        );
        assert_eq!(
            h.head_vocab_sizes[..12],
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147
            ],
            "qwen4exp.ple.head_vocab_sizes"
        );
        assert_eq!(
            h.head_offsets[..12],
            [
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655
            ],
            "qwen4exp.ple.head_offsets"
        );
        assert_eq!(h.n_heads, 16);
        // `per_layer_token_embd.weight` is [160, 320001536] in the checkpoint.
        assert_eq!(h.padded_rows, 320_001_536);
    }

    /// A PLE module on a later layer must land on a disjoint slice of the
    /// shared table and hash with different multipliers, or two layers would
    /// read the same embeddings.
    #[test]
    fn a_second_ple_module_gets_its_own_table_slice() {
        let cfg = flash_next();
        let a = PleHash::derive(&cfg, 0);
        let b = PleHash::derive(&cfg, 1);
        assert_ne!(a.layer_multipliers, b.layer_multipliers);
        let a_end = a.head_offsets[15] + a.head_vocab_sizes[15];
        assert_eq!(b.head_offsets[0], a_end, "slices must be contiguous");
    }

    #[test]
    fn qsa_selection_is_the_identity_up_to_the_budget() {
        let cfg = flash_next();
        assert_eq!(cfg.qsa_block_topk(), 512);
        assert!(cfg.qsa_is_dense(2048));
        assert!(cfg.qsa_is_dense(2051), "the tail is not a block");
        assert!(!cfg.qsa_is_dense(2052));
    }

    #[test]
    fn linear_layer_widths_follow_the_head_counts() {
        let cfg = flash_next();
        assert_eq!(cfg.gdn_key_dim(), 2048);
        assert_eq!(cfg.gdn_value_dim(), 6144);
        assert_eq!(cfg.gdn_conv_dim(), 10240, "attn_qkv output width");
        assert_eq!(cfg.hc_dim(), 10240, "hyper-connection residual width");
        assert_eq!(cfg.ple_embed_dim(), 2560);
        assert_eq!(cfg.ple_conv_state_len(), 9, "(4 - 1) * 3");
    }

    #[test]
    fn hc_count_of_one_is_rejected() {
        let mut cfg = flash_next();
        cfg.hc_count = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn grouped_query_attention_dimensions_must_divide_evenly() {
        let mut cfg = flash_next();
        cfg.num_attention_heads = 25;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("must be divisible"), "{err}");
    }

    #[test]
    fn rotary_width_must_fit_the_attention_head() {
        let mut cfg = flash_next();
        cfg.rope_dim_count = cfg.head_dim + 2;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("rope_dim_count"), "{err}");
    }

    #[test]
    fn a_ple_layer_must_be_a_linear_layer() {
        let mut cfg = flash_next();
        cfg.ple_layers = vec![3];
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("not a linear-attention layer"), "{err}");
    }

    #[test]
    fn mismatched_hash_constants_are_rejected() {
        let mut cfg = flash_next();
        cfg.ple_constants = Some(PleConstants {
            layer_multipliers: vec![1, 2, 3],
            head_offsets: vec![0; 16],
            head_vocab_sizes: vec![20_000_003; 16],
        });
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("PLE multipliers"), "{err}");
    }

    #[test]
    fn hf_wrapper_parses_the_published_shape_and_one_based_ple_layer() {
        let json = r#"{
          "model_type": "qwen4_exp",
          "image_token_id": 248056,
          "text_config": {
            "vocab_size": 248320,
            "hidden_size": 2560,
            "num_hidden_layers": 4,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "output_gate_type": "sigmoid",
            "hc_count": 4,
            "hc_lowrank": 320,
            "indexer_n_heads": 4,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "eos_token_id": [248043, 248044],
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "rope_parameters": {
              "rope_theta": 10000000.0,
              "partial_rotary_factor": 0.25,
              "mrope_section": [11, 11, 10],
              "mrope_interleaved": true
            }
          }
        }"#;
        let cfg = Qwen4ExpConfig::from_hf_json(json).unwrap();
        assert_eq!(cfg.ple_layers, vec![1]);
        assert_eq!(cfg.rope_dim_count, 64);
        assert_eq!(cfg.ple_eos_token_id, 248044);
        assert_eq!(cfg.gdn_conv_dim(), 10240);
        assert_eq!(cfg.layer_kind(3), LayerKind::Indexed);
        assert!(cfg.rms_norm_offset);
    }
}
