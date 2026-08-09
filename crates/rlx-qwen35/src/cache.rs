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

//! Decode-time recurrent state for the qwen35 hybrid trunk.

use crate::config::Qwen35Config;

/// Per-trunk-layer recurrent payload carried across decode steps.
#[derive(Debug, Clone)]
pub enum Qwen35LayerState {
    /// Gated-DeltaNet + depthwise conv block.
    Linear {
        /// `[batch, k-1, conv_channels]` causal conv ring.
        conv_state: Vec<f32>,
        /// `[batch, n_v_heads, n_state, n_state]` SSM matrix per head.
        ssm_state: Vec<f32>,
    },
    /// Standard attention block — pre-GQA K/V cache.
    /// Only allocated for full-attention trunk layers (`full_attention_interval`).
    FullAttn {
        /// `[batch, past_seq, n_kv * head_dim]`, post-RoPE K.
        /// Empty when [`Self::FullAttn::packed`] is set (dequant on feed).
        past_k: Vec<f32>,
        /// `[batch, past_seq, n_kv * head_dim]`, pre-GQA V.
        past_v: Vec<f32>,
        /// Optional host-side quantized KV (`RLX_QWEN35_KV_QUANT=f16|q4_0|q8_0`).
        /// Cuts long-context RAM; graph feeds still receive f32.
        packed: Option<Box<rlx_runtime::quantized_kv::QuantizedKvLayer>>,
    },
}

/// Parse `RLX_QWEN35_KV_QUANT` → scheme (full-attn layers only).
fn kv_quant_from_env() -> Option<rlx_runtime::quantized_kv::KvQuant> {
    use rlx_runtime::quantized_kv::KvQuant;
    match std::env::var("RLX_QWEN35_KV_QUANT")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("f16") | Some("fp16") => Some(KvQuant::F16),
        Some("q4_0") | Some("q4") => Some(KvQuant::Q4_0),
        Some("q8_0") | Some("q8") => Some(KvQuant::Q8_0),
        Some("q5_0") | Some("q5") => Some(KvQuant::Q5_0),
        _ => None,
    }
}

pub(crate) fn maybe_pack_full_attn_kv(
    past_k: Vec<f32>,
    past_v: Vec<f32>,
    kv_cols: usize,
) -> anyhow::Result<Qwen35LayerState> {
    use anyhow::{Context, bail};
    use rlx_runtime::quantized_kv::QuantizedKvLayer;

    let Some(scheme) = kv_quant_from_env() else {
        return Ok(Qwen35LayerState::FullAttn {
            past_k,
            past_v,
            packed: None,
        });
    };
    if past_k.len() != past_v.len() {
        bail!("pack kv: k/v len mismatch");
    }
    if !past_k.len().is_multiple_of(kv_cols) {
        bail!(
            "pack kv: len {} not multiple of kv_cols {kv_cols}",
            past_k.len()
        );
    }
    let mut layer = QuantizedKvLayer::new(kv_cols, scheme).context("QuantizedKvLayer::new")?;
    layer.append_rows(&past_k, &past_v).context("append_rows")?;
    Ok(Qwen35LayerState::FullAttn {
        past_k: Vec::new(),
        past_v: Vec::new(),
        packed: Some(Box::new(layer)),
    })
}

pub(crate) fn full_attn_kv_f32(layer: &Qwen35LayerState) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    use anyhow::{Context, bail};
    match layer {
        Qwen35LayerState::FullAttn {
            past_k,
            past_v,
            packed: None,
        } => Ok((past_k.clone(), past_v.clone())),
        Qwen35LayerState::FullAttn {
            packed: Some(q), ..
        } => q.read_all().context("quantized kv read_all"),
        _ => bail!("full_attn_kv_f32: not a FullAttn layer"),
    }
}

/// Host-side decode cache seeded from a prefill-with-states forward.
#[derive(Debug, Clone)]
pub struct Qwen35DecodeCache {
    pub batch: usize,
    /// Physical resident length of the full-attn KV (rows actually stored). With
    /// sink+window eviction this can be < `abs_pos`.
    pub past_seq: usize,
    /// True absolute position of the next token (total tokens seen, never
    /// shrunk). Equals `past_seq` unless KV rows have been evicted. Decode RoPE
    /// uses THIS so a retained key's original-position RoPE still aligns with the
    /// query after the middle of the cache is dropped.
    pub abs_pos: usize,
    /// Actual prompt length per batch row (before generation).
    pub prompt_lens: Vec<usize>,
    pub layers: Vec<Qwen35LayerState>,
}

impl Qwen35DecodeCache {
    pub fn n_trunk(&self) -> usize {
        self.layers.len()
    }

    /// An empty cache (`past_seq=0`, zero recurrent state, no KV) — the seed for
    /// processing a prompt through the DECODE path (`prefill_chunk`) so the prompt
    /// representation is bit-identical to sequential decode.
    pub fn empty(cfg: &Qwen35Config, batch: usize) -> Self {
        let n_state = cfg.ssm_state_size;
        let n_v_heads = cfg.ssm_time_step_rank;
        let conv_channels = linear_conv_channels(cfg);
        let k_conv = cfg.ssm_conv_kernel;
        let layers = trunk_layer_kinds(cfg)
            .into_iter()
            .map(|is_full| {
                if is_full {
                    Qwen35LayerState::FullAttn {
                        past_k: Vec::new(),
                        past_v: Vec::new(),
                        packed: None,
                    }
                } else {
                    Qwen35LayerState::Linear {
                        conv_state: vec![0f32; batch * (k_conv - 1) * conv_channels],
                        ssm_state: vec![0f32; batch * n_v_heads * n_state * n_state],
                    }
                }
            })
            .collect();
        Qwen35DecodeCache {
            batch,
            past_seq: 0,
            abs_pos: 0,
            prompt_lens: vec![0; batch],
            layers,
        }
    }
}

/// Trunk layer kinds in declaration order (excludes MTP).
pub fn trunk_layer_kinds(cfg: &Qwen35Config) -> Vec<bool> {
    let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
    let interval = cfg.full_attention_interval.max(1);
    (0..n_main).map(|il| ((il + 1) % interval) == 0).collect()
}

/// Number of extra graph outputs after logits (and optional MTP).
pub fn recurrent_output_count(cfg: &Qwen35Config) -> usize {
    trunk_layer_kinds(cfg).len() * 2
}

/// Logit outputs before recurrent state exports: `[trunk, (optional mtp)]`.
pub fn logit_output_count(with_mtp: bool) -> usize {
    1 + usize::from(with_mtp)
}

fn truncate_logits_row(_cfg: &Qwen35Config, logits: Vec<f32>, _batch: usize) -> Vec<f32> {
    // Graph LM head width comes from the embedding table (`lm_vocab_size`);
    // do not clip to `cfg.vocab_size` (metadata can under-report vs GGUF).
    logits
}

fn parse_mtp_logits(cfg: &Qwen35Config, batch: usize, mtp: Vec<f32>) -> anyhow::Result<Vec<f32>> {
    use anyhow::bail;
    let lm_vocab = mtp.len() / batch.max(1);
    let expected = batch * lm_vocab;
    if mtp.len() != expected {
        bail!(
            "mtp logits: len={} expected batch*lm_vocab={expected}",
            mtp.len()
        );
    }
    Ok(truncate_logits_row(cfg, mtp, batch))
}

/// Zero-initialized recurrent inputs for a prefill-cache seed graph.
pub fn zero_recurrent_inputs(cfg: &Qwen35Config, batch: usize) -> Vec<(String, Vec<f32>)> {
    let n_state = cfg.ssm_state_size;
    let n_v_heads = cfg.ssm_time_step_rank;
    let conv_channels = linear_conv_channels(cfg);
    let k_conv = cfg.ssm_conv_kernel;
    let head_dim = cfg.key_length;
    let kv_cols = cfg.num_key_value_heads * head_dim;

    let mut out = Vec::new();
    for (il, is_full) in trunk_layer_kinds(cfg).into_iter().enumerate() {
        if is_full {
            let _ = kv_cols;
            let _ = head_dim;
            // Full-attn layers have no recurrent *inputs* on prefill seed.
        } else {
            out.push((
                format!("conv_state_l{il}"),
                vec![0f32; batch * (k_conv - 1) * conv_channels],
            ));
            out.push((
                format!("ssm_state_l{il}"),
                vec![0f32; batch * n_v_heads * n_state * n_state],
            ));
        }
    }
    out
}

fn linear_conv_channels(cfg: &Qwen35Config) -> usize {
    let n_state = cfg.ssm_state_size;
    let n_k_heads = cfg.ssm_group_count;
    let n_v_heads = cfg.ssm_time_step_rank;
    let key_dim = n_state * n_k_heads;
    let value_dim = n_state * n_v_heads;
    key_dim * 2 + value_dim
}

/// Build `[batch, bucket_upper + 1]` attention mask for bucketed decode.
///
/// Bucketed decode concatenates padded `past_k` `[B, upper, kv]` with the new
/// token → `[B, upper + 1, kv]`; the new K/V row lives at index `upper`, not
/// `past_seq`. Mask ones at `i < past_seq` (valid prefix only) and at
/// `i == bucket_upper` (current query slot). Matches Metal bucketed-decode
/// parity tests (`metal_bucketed_decode_attn_parity.rs`).
pub fn build_decode_attention_mask(
    batch: usize,
    past_seq: usize,
    bucket_upper: usize,
    prompt_lens: &[usize],
    generated_per_row: &[usize],
) -> Vec<f32> {
    let mask_len = bucket_upper + 1;
    let mut mask = vec![0f32; batch * mask_len];
    for b in 0..batch {
        let valid = prompt_lens.get(b).copied().unwrap_or(past_seq)
            + generated_per_row.get(b).copied().unwrap_or(0);
        let base = b * mask_len;
        for i in 0..mask_len {
            if i == bucket_upper || (i < past_seq && i < valid) {
                mask[base + i] = 1.0;
            }
        }
    }
    mask
}

/// Pad `[batch, actual, kv_cols]` K/V to `[batch, bucket_upper, kv_cols]`.
///
/// Under in-place KV (`RLX_QWEN35_INPLACE_KV`) the graph declares `past_k/v` one
/// row larger so `Op::KvAppend` can write the new token at index `bucket_upper`;
/// pad to `bucket_upper + 1` rows (the trailing row starts zero, gets overwritten
/// on-GPU by KvAppend). The padded stride must match the graph's declared shape.
pub fn pad_kv_to_bucket(
    src: &[f32],
    batch: usize,
    actual_past: usize,
    bucket_upper: usize,
    kv_cols: usize,
) -> Vec<f32> {
    let rows = bucket_upper + rlx_ir::env::flag("RLX_QWEN35_INPLACE_KV") as usize;
    let mut out = vec![0f32; batch * rows * kv_cols];
    for b in 0..batch {
        let src_base = b * actual_past * kv_cols;
        let dst_base = b * rows * kv_cols;
        let copy_len = actual_past * kv_cols;
        out[dst_base..dst_base + copy_len].copy_from_slice(&src[src_base..src_base + copy_len]);
    }
    out
}

/// Slice bucketed K/V outputs back to dense `[batch, actual_past, kv_cols]`.
///
/// **Decode layout** (`concat(past_k_padded[upper], new_k[1], dim=1)`):
/// buffer is `[batch, upper + 1, kv]` with the new token at index `upper`, not
/// at `past_seq`. Copying a contiguous prefix of length `past_seq + 1` would
/// keep the zero pad at `past_seq` and drop the real new K/V — every cached
/// step then absorbs padding and sequential AR drifts.
///
/// **Prefill / dense pad**: when `src` is only `[batch, src_seq, kv]` (no
/// trailing new-token row), take the contiguous valid prefix.
pub fn slice_kv_from_bucket(
    src: &[f32],
    batch: usize,
    actual_past: usize,
    bucket_upper: usize,
    kv_cols: usize,
) -> anyhow::Result<Vec<f32>> {
    use anyhow::bail;
    if actual_past == 0 {
        return Ok(Vec::new());
    }
    let decode_seq = bucket_upper.saturating_add(1);
    let decode_len = batch * decode_seq * kv_cols;
    if src.len() >= decode_len {
        let mut out = vec![0f32; batch * actual_past * kv_cols];
        let old_past = actual_past - 1;
        for b in 0..batch {
            let src_base = b * decode_seq * kv_cols;
            let dst_base = b * actual_past * kv_cols;
            let prefix = old_past * kv_cols;
            if prefix > 0 {
                out[dst_base..dst_base + prefix].copy_from_slice(&src[src_base..src_base + prefix]);
            }
            let src_new = src_base + bucket_upper * kv_cols;
            let dst_new = dst_base + old_past * kv_cols;
            out[dst_new..dst_new + kv_cols].copy_from_slice(&src[src_new..src_new + kv_cols]);
        }
        return Ok(out);
    }

    let src_seq = if batch == 0 || kv_cols == 0 {
        0
    } else {
        src.len() / (batch * kv_cols)
    };
    if src_seq < actual_past {
        bail!(
            "slice_kv_from_bucket: need at least {actual_past} rows, got {src_seq} \
             (batch={batch}, bucket_upper={bucket_upper}, src.len={})",
            src.len()
        );
    }
    let mut out = vec![0f32; batch * actual_past * kv_cols];
    for b in 0..batch {
        let src_base = b * src_seq * kv_cols;
        let dst_base = b * actual_past * kv_cols;
        let copy_len = actual_past * kv_cols;
        out[dst_base..dst_base + copy_len].copy_from_slice(&src[src_base..src_base + copy_len]);
    }
    Ok(out)
}

/// Multi-token bucketed slice: KV buffer is `concat(past_padded[upper], new[n_new])`
/// → `[batch, upper + n_new, kv]` with the `n_new` new tokens at rows
/// `[upper, upper+n_new)` (the `[past_seq, upper)` slots are zero pad). Reassemble
/// dense `[batch, new_past, kv]` = real prefix `[0, past_seq)` ++ new `[upper, ...)`.
/// `new_past = past_seq + n_new`. For `n_new == 1` this matches
/// [`slice_kv_from_bucket`]'s decode branch.
pub fn slice_kv_from_bucket_n(
    src: &[f32],
    batch: usize,
    new_past: usize,
    n_new: usize,
    bucket_upper: usize,
    kv_cols: usize,
) -> anyhow::Result<Vec<f32>> {
    use anyhow::bail;
    if new_past == 0 {
        return Ok(Vec::new());
    }
    let old_past = new_past.saturating_sub(n_new);
    let buf_seq = bucket_upper + n_new;
    let buf_len = batch * buf_seq * kv_cols;
    if src.len() < buf_len {
        bail!(
            "slice_kv_from_bucket_n: need {buf_len} elems ({buf_seq} rows), got {} \
             (batch={batch}, upper={bucket_upper}, n_new={n_new})",
            src.len()
        );
    }
    let mut out = vec![0f32; batch * new_past * kv_cols];
    for b in 0..batch {
        let src_base = b * buf_seq * kv_cols;
        let dst_base = b * new_past * kv_cols;
        let prefix = old_past * kv_cols;
        if prefix > 0 {
            out[dst_base..dst_base + prefix].copy_from_slice(&src[src_base..src_base + prefix]);
        }
        // The n_new new rows live contiguously at bucket_upper.
        let src_new = src_base + bucket_upper * kv_cols;
        let dst_new = dst_base + prefix;
        let new_len = n_new * kv_cols;
        out[dst_new..dst_new + new_len].copy_from_slice(&src[src_new..src_new + new_len]);
    }
    Ok(out)
}

/// Additive per-query attention bias `[batch, n_head, n_new, upper + n_new]` for
/// bucketed multi-token verify (`MaskKind::Bias`: `s += M`).
///
/// KV layout is `concat(past_padded[upper], new[n_new])`: real past keys occupy
/// `[0, past_seq)`, pad slots `[past_seq, upper)`, and the `n_new` new tokens
/// `[upper, upper+n_new)`. Query `qi` (absolute position `past_seq + qi`) attends
/// the real past plus new keys `[upper, upper+qi]` (causal among the new tokens);
/// every other slot (padding + future new) gets `-1e9`. batch==1 → `valid` is the
/// full `past_seq` (no intra-cache padding on a single stream).
pub fn build_verify_bias_mask(
    batch: usize,
    n_head: usize,
    past_seq: usize,
    n_new: usize,
    bucket_upper: usize,
) -> Vec<f32> {
    let seq_k = bucket_upper + n_new;
    let mut mask = vec![-1e9f32; batch * n_head * n_new * seq_k];
    let valid = past_seq.min(bucket_upper);
    for b in 0..batch {
        for h in 0..n_head {
            for qi in 0..n_new {
                let row = (((b * n_head + h) * n_new) + qi) * seq_k;
                for ki in 0..valid {
                    mask[row + ki] = 0.0;
                }
                for ki in bucket_upper..=(bucket_upper + qi) {
                    mask[row + ki] = 0.0;
                }
            }
        }
    }
    mask
}

/// Zero padded prompt positions in full-attention KV (variable-length batch).
pub fn zero_prompt_padding_kv(
    cfg: &Qwen35Config,
    cache: &mut Qwen35DecodeCache,
    padded_seq: usize,
) {
    let head_dim = cfg.key_length;
    let kv_cols = cfg.num_key_value_heads * head_dim;
    let kinds = trunk_layer_kinds(cfg);
    for (il, layer) in cache.layers.iter_mut().enumerate() {
        if !kinds[il] {
            continue;
        }
        if let Qwen35LayerState::FullAttn {
            past_k,
            past_v,
            packed: None,
        } = layer
        {
            for b in 0..cache.batch {
                let prompt_len = cache.prompt_lens.get(b).copied().unwrap_or(padded_seq);
                if prompt_len >= padded_seq {
                    continue;
                }
                for t in prompt_len..padded_seq {
                    let start = b * padded_seq * kv_cols + t * kv_cols;
                    past_k[start..start + kv_cols].fill(0.0);
                    past_v[start..start + kv_cols].fill(0.0);
                }
            }
        } else if let Qwen35LayerState::FullAttn {
            packed: Some(_), ..
        } = layer
        {
            // Quantized host KV: padding zeros are applied on the f32 feed path
            // via mask; packed rows stay dense for the actual past_seq.
            let _ = kv_cols;
        }
    }
}

/// Build host feeds for a single decode step from `cache`.
///
/// `tokens` must have length `cache.batch` — one next-token id per row.
/// When `bucket_upper` is `Some`, pads K/V and supplies a custom mask.
pub fn decode_step_feeds(
    cfg: &Qwen35Config,
    cache: &Qwen35DecodeCache,
    tokens: &[u32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    bucket_upper: Option<usize>,
    generated_per_row: &[usize],
    // `Some(token_embd)` under RLX_QWEN35_HOST_EMBED: gather each new token's
    // embedding row host-side and feed `inputs_embeds`, so the decode graph
    // never registers the full `[vocab, hidden]` F32 table (Bonsai-27B 4.7 GiB).
    host_embed_table: Option<&[f32]>,
    // When set, feed flat `[batch * hidden]` as `inputs_embeds` (Gepard audio frames).
    custom_embed: Option<&[f32]>,
) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
    use anyhow::bail;

    if tokens.len() != cache.batch {
        bail!(
            "decode_step_feeds: expected {} tokens, got {}",
            cache.batch,
            tokens.len()
        );
    }
    let mut feeds = vec![
        (
            "input_ids".into(),
            tokens.iter().map(|&t| t as f32).collect(),
        ),
        ("rope_cos".into(), rope_cos.to_vec()),
        ("rope_sin".into(), rope_sin.to_vec()),
    ];
    if let Some(embed) = custom_embed {
        feeds.push(("inputs_embeds".into(), embed.to_vec()));
    } else if let Some(tbl) = host_embed_table {
        let n_embd = cfg.hidden_size;
        let mut e = vec![0f32; tokens.len() * n_embd];
        for (b, &t) in tokens.iter().enumerate() {
            let src = (t as usize) * n_embd;
            if src + n_embd <= tbl.len() {
                e[b * n_embd..b * n_embd + n_embd].copy_from_slice(&tbl[src..src + n_embd]);
            }
        }
        feeds.push(("inputs_embeds".into(), e));
    }
    if let Some(upper) = bucket_upper {
        let mask = build_decode_attention_mask(
            cache.batch,
            cache.past_seq,
            upper,
            &cache.prompt_lens,
            generated_per_row,
        );
        feeds.push(("mask".into(), mask));
    }
    let head_dim = cfg.key_length;
    let kv_cols = cfg.num_key_value_heads * head_dim;
    let kinds = trunk_layer_kinds(cfg);
    for (il, layer) in cache.layers.iter().enumerate() {
        let is_full = kinds[il];
        match (layer, is_full) {
            (
                Qwen35LayerState::Linear {
                    conv_state,
                    ssm_state,
                },
                false,
            ) => {
                feeds.push((format!("conv_state_l{il}"), conv_state.clone()));
                feeds.push((format!("ssm_state_l{il}"), ssm_state.clone()));
            }
            (Qwen35LayerState::FullAttn { .. }, true) => {
                // Resident-KV (`RLX_QWEN35_GPU_KV`): past_k/v are bound as resident
                // arena handles and grown in-place by Op::KvAppend — never fed from
                // host. Skip building the O(ctx) feed (the host mirror is also stale
                // here since the resident path doesn't append per-token).
                if rlx_ir::env::flag("RLX_QWEN35_GPU_KV") {
                    continue;
                }
                let (past_k, past_v) = full_attn_kv_f32(layer)?;
                if let Some(upper) = bucket_upper {
                    feeds.push((
                        format!("past_k_l{il}"),
                        pad_kv_to_bucket(&past_k, cache.batch, cache.past_seq, upper, kv_cols),
                    ));
                    feeds.push((
                        format!("past_v_l{il}"),
                        pad_kv_to_bucket(&past_v, cache.batch, cache.past_seq, upper, kv_cols),
                    ));
                } else {
                    feeds.push((format!("past_k_l{il}"), past_k));
                    feeds.push((format!("past_v_l{il}"), past_v));
                }
            }
            _ => {}
        }
    }
    Ok(feeds)
}

/// Parse prefill-cache graph outputs into logits/hidden + [`Qwen35DecodeCache`].
/// When `trunk_is_hidden`, the first output is `[batch × hidden_size]` not logits.
pub fn seed_cache_from_outputs(
    cfg: &Qwen35Config,
    batch: usize,
    seq: usize,
    prompt_lens: &[usize],
    outputs: Vec<Vec<f32>>,
    with_mtp: bool,
    trunk_is_hidden: bool,
) -> anyhow::Result<(Vec<f32>, Qwen35DecodeCache, Option<Vec<f32>>)> {
    use anyhow::{Context, bail};
    let n_head = logit_output_count(with_mtp);
    let n_extra = recurrent_output_count(cfg);
    if outputs.len() != n_head + n_extra {
        bail!(
            "prefill-cache: expected {} outputs, got {}",
            n_head + n_extra,
            outputs.len()
        );
    }
    let mut iter = outputs.into_iter();
    let trunk = iter.next().context("trunk head output missing")?;
    let head_dim = cfg.key_length;
    let kv_cols = cfg.num_key_value_heads * head_dim;
    let logits = if trunk_is_hidden {
        let n = cfg.hidden_size;
        let expected_last = batch * n;
        let expected_full = batch * seq * n;
        if trunk.len() == expected_last {
            trunk
        } else if trunk.len() == expected_full
            || (trunk.len().is_multiple_of(n)
                && trunk.len() >= batch.max(1) * n
                && trunk.len() % (batch.max(1) * n) == 0)
        {
            let row_stride = trunk.len() / batch.max(1);
            let seq_dim = row_stride / n;
            if batch > 1 && !prompt_lens.is_empty() {
                let mut out = Vec::with_capacity(batch * n);
                for b in 0..batch {
                    let pl = prompt_lens.get(b).copied().unwrap_or(seq).min(seq_dim);
                    let idx = pl.saturating_sub(1);
                    let off = b * row_stride + idx * n;
                    out.extend_from_slice(&trunk[off..off + n]);
                }
                out
            } else if !prompt_lens.is_empty() {
                let last_pl = *prompt_lens.iter().max().unwrap_or(&seq);
                let idx = last_pl.saturating_sub(1).min(seq_dim.saturating_sub(1));
                let off = idx * n;
                trunk[off..off + n].to_vec()
            } else {
                trunk[expected_full.saturating_sub(n)..].to_vec()
            }
        } else {
            bail!(
                "prefill-cache hidden: len={} expected batch*hidden={expected_last} \
                 or batch*seq*hidden={expected_full} (or padded max_seq layout)",
                trunk.len()
            );
        }
    } else {
        let lm_vocab = trunk.len() / batch.max(1);
        let expected_logits = batch * lm_vocab;
        if trunk.len() != expected_logits {
            bail!(
                "prefill-cache logits: len={} expected batch*lm_vocab={expected_logits} \
                 (batch={batch}, lm_vocab={lm_vocab})",
                trunk.len()
            );
        }
        truncate_logits_row(cfg, trunk, batch)
    };
    let mtp_logits = if with_mtp {
        Some(parse_mtp_logits(
            cfg,
            batch,
            iter.next().context("mtp logits missing")?,
        )?)
    } else {
        None
    };

    let mut layers = Vec::with_capacity(trunk_layer_kinds(cfg).len());
    for (il, is_full) in trunk_layer_kinds(cfg).into_iter().enumerate() {
        if is_full {
            let k = iter.next().context("past_k missing")?;
            let v = iter.next().context("past_v missing")?;
            let expected = batch * seq * kv_cols;
            let (past_k, past_v) = if k.len() == expected && v.len() == expected {
                (k, v)
            } else if k.len() % kv_cols == 0 && v.len() % kv_cols == 0 {
                let k_bucket = k.len() / (batch.max(1) * kv_cols);
                let v_bucket = v.len() / (batch.max(1) * kv_cols);
                if k_bucket >= seq && v_bucket >= seq {
                    (
                        slice_kv_from_bucket(&k, batch, seq, k_bucket, kv_cols)?,
                        slice_kv_from_bucket(&v, batch, seq, v_bucket, kv_cols)?,
                    )
                } else {
                    bail!(
                        "layer {il} kv: k.len={} v.len={} expected {expected} \
                         (k_bucket={k_bucket} v_bucket={v_bucket} < seq={seq})",
                        k.len(),
                        v.len()
                    );
                }
            } else {
                bail!(
                    "layer {il} kv: k.len={} v.len={} expected {expected}",
                    k.len(),
                    v.len()
                );
            };
            layers.push(maybe_pack_full_attn_kv(past_k, past_v, kv_cols)?);
        } else {
            let conv = iter.next().context("conv_state missing")?;
            let ssm = iter.next().context("ssm_state missing")?;
            let conv_ring =
                batch * (cfg.ssm_conv_kernel.saturating_sub(1)) * linear_conv_channels(cfg);
            let conv_state = if conv.len() == conv_ring {
                conv
            } else {
                bail!(
                    "layer {il} conv_state: len={} expected {conv_ring}",
                    conv.len()
                );
            };
            layers.push(Qwen35LayerState::Linear {
                conv_state,
                ssm_state: ssm,
            });
        }
    }
    Ok((
        logits,
        Qwen35DecodeCache {
            batch,
            past_seq: seq,
            abs_pos: seq,
            prompt_lens: prompt_lens.to_vec(),
            layers,
        },
        mtp_logits,
    ))
}

/// StreamingLLM-style sink+window KV eviction (single-stream). Keeps the first
/// `sinks` rows (attention sinks) and the most recent `window` rows of every
/// full-attn layer's host K/V mirror, dropping the middle. Retained keys keep
/// their ORIGINAL-position RoPE, and decode RoPE uses `cache.abs_pos` (the true
/// position), so the query still aligns with the retained keys after the cut —
/// this is why RoPE had to be decoupled from `past_seq`.
///
/// `past_seq` (physical resident rows) drops to `sinks + window`; `abs_pos` is
/// untouched. Triggered lazily: only compacts once `past_seq` exceeds
/// `sinks + window + slack`, then trims back to `sinks + window`, so the O(rows)
/// memcpy is amortized over ~`slack` steps. No-op when the cache still fits, on
/// batch > 1, or when packed KV is in use (would need dequant/repack first).
pub fn evict_kv_sink_window(
    cfg: &Qwen35Config,
    cache: &mut Qwen35DecodeCache,
    sinks: usize,
    window: usize,
    slack: usize,
) -> bool {
    if cache.batch != 1 || window == 0 || kv_quant_from_env().is_some() {
        return false;
    }
    let keep = sinks + window;
    if cache.past_seq <= keep + slack {
        return false;
    }
    let kv_cols = cfg.num_key_value_heads * cfg.key_length;
    let old = cache.past_seq;
    // Keep rows [0, sinks) ++ [old-window, old); drop the middle.
    let win_start = old - window;
    for layer in cache.layers.iter_mut() {
        if let Qwen35LayerState::FullAttn { past_k, past_v, .. } = layer {
            compact_sink_window(past_k, kv_cols, sinks, win_start, old);
            compact_sink_window(past_v, kv_cols, sinks, win_start, old);
        }
    }
    cache.past_seq = keep;
    // Keep prompt_lens ≤ past_seq so `dense_generated_per_row` (past_seq −
    // prompt_len) stays non-negative; the sinks are (part of) the prompt.
    for pl in cache.prompt_lens.iter_mut() {
        *pl = (*pl).min(keep);
    }
    true
}

/// In-place retain rows `[0, sinks) ++ [win_start, old)` of a row-major
/// `[old, kv_cols]` f32 buffer, shifting the window block up to abut the sinks.
fn compact_sink_window(
    buf: &mut Vec<f32>,
    kv_cols: usize,
    sinks: usize,
    win_start: usize,
    old: usize,
) {
    if buf.len() != old * kv_cols {
        return; // unexpected shape (e.g. empty packed mirror); leave untouched
    }
    let src = win_start * kv_cols;
    let dst = sinks * kv_cols;
    let len = (old - win_start) * kv_cols;
    buf.copy_within(src..src + len, dst);
    buf.truncate(dst + len);
}

/// Advance `cache` from decode-graph outputs (logits or normed hidden + states).
/// When `trunk_is_hidden`, the first output is `[batch × hidden_size]` not logits.
pub fn advance_cache_from_decode_outputs(
    cfg: &Qwen35Config,
    cache: &mut Qwen35DecodeCache,
    outputs: Vec<Vec<f32>>,
    bucket_upper: Option<usize>,
    mtp_logits_path: bool,
    want_mtp: bool,
    trunk_is_hidden: bool,
) -> anyhow::Result<(Vec<f32>, Option<Vec<f32>>)> {
    advance_cache_from_decode_outputs_n(
        cfg,
        cache,
        1,
        outputs,
        bucket_upper,
        mtp_logits_path,
        want_mtp,
        trunk_is_hidden,
    )
}

/// Like [`advance_cache_from_decode_outputs`] but for a continued `n_new`-token
/// forward (chunked prefill): the full-attn KV grew by `n_new` rows and
/// `past_seq`/`abs_pos` advance by `n_new`. The graph's state outputs are the
/// FINAL state after all `n_new` tokens (GDN carry scanned them).
#[allow(clippy::too_many_arguments)]
pub fn advance_cache_from_decode_outputs_n(
    cfg: &Qwen35Config,
    cache: &mut Qwen35DecodeCache,
    n_new: usize,
    outputs: Vec<Vec<f32>>,
    bucket_upper: Option<usize>,
    mtp_logits_path: bool,
    want_mtp: bool,
    trunk_is_hidden: bool,
) -> anyhow::Result<(Vec<f32>, Option<Vec<f32>>)> {
    use anyhow::{Context, bail};
    let n_head = logit_output_count(mtp_logits_path);
    let n_extra = recurrent_output_count(cfg);
    if outputs.len() != n_head + n_extra {
        bail!(
            "decode: expected {} outputs, got {}",
            n_head + n_extra,
            outputs.len()
        );
    }
    let mut iter = outputs.into_iter();
    let trunk = iter.next().context("trunk head output missing")?;
    let new_past = cache.past_seq + n_new;
    let head_dim = cfg.key_length;
    let kv_cols = cfg.num_key_value_heads * head_dim;
    let batch = cache.batch;

    let trunk_out = if trunk_is_hidden {
        let expected = batch * cfg.hidden_size;
        if trunk.len() != expected {
            bail!(
                "decode hidden: len={} expected batch*hidden={expected}",
                trunk.len()
            );
        }
        trunk
    } else {
        let lm_vocab = trunk.len() / batch.max(1);
        let expected_logits = batch * lm_vocab;
        if trunk.len() != expected_logits {
            bail!(
                "decode logits: len={} expected batch*lm_vocab={expected_logits}",
                trunk.len()
            );
        }
        truncate_logits_row(cfg, trunk, batch)
    };
    let mtp_logits = if mtp_logits_path {
        let raw = iter.next().context("mtp logits missing")?;
        if want_mtp {
            Some(parse_mtp_logits(cfg, batch, raw)?)
        } else {
            None
        }
    } else {
        None
    };

    let mut new_layers = Vec::with_capacity(cache.layers.len());
    let kinds = trunk_layer_kinds(cfg);
    for (il, layer) in cache.layers.iter().enumerate() {
        let is_full = kinds[il];
        if is_full {
            let k = iter.next().context("new_k missing")?;
            let v = iter.next().context("new_v missing")?;
            let (k, v) = if let Some(upper) = bucket_upper {
                // Bucketed verify: KV is `concat(past_padded[upper], new[n_new])`;
                // slice back to the dense `new_past` rows (n_new-aware).
                (
                    slice_kv_from_bucket_n(&k, batch, new_past, n_new, upper, kv_cols)?,
                    slice_kv_from_bucket_n(&v, batch, new_past, n_new, upper, kv_cols)?,
                )
            } else {
                (k, v)
            };
            let expected = batch * new_past * kv_cols;
            if k.len() != expected || v.len() != expected {
                bail!(
                    "layer {il} kv: k.len={} v.len={} expected {expected}",
                    k.len(),
                    v.len()
                );
            }
            new_layers.push(maybe_pack_full_attn_kv(k, v, kv_cols)?);
            let _ = layer;
        } else {
            let conv = iter.next().context("conv_state missing")?;
            let ssm = iter.next().context("ssm_state missing")?;
            new_layers.push(Qwen35LayerState::Linear {
                conv_state: conv,
                ssm_state: ssm,
            });
        }
    }
    cache.past_seq = new_past;
    cache.abs_pos += n_new;
    cache.layers = new_layers;
    Ok((trunk_out, mtp_logits))
}

/// Advance the decode cache for the **GPU-resident KV** path (batch==1).
///
/// Unlike [`advance_cache_from_decode_outputs`], the full-attention K/V never
/// leave the device: `outputs` contains ONLY `[logits, mtp?, then GDN
/// conv/ssm in layer order]` (full-attn K/V outputs stayed in the arena and were
/// folded into the resident handles by `feed_kv_row`). The new K/V rows arrive
/// via `new_kv_rows[il] = Some((k_row, v_row))` (one `[kv_cols]` row each, read
/// O(1) from the resident output) and are APPENDED to the host mirror — no
/// `slice_kv_from_bucket` O(ctx) copy. The mirror is kept only so a bucket
/// change can rebind the handles.
/// Advance the decode cache for the **GPU-resident inplace-KV** path (batch==1).
///
/// The full-attention K/V live entirely in the resident arena handles and are
/// grown in-place by `Op::KvAppend` — they never touch the host, so this only
/// advances `past_seq` and refreshes the O(1) GDN conv/ssm state (host-fed).
/// `outputs` = `[logits, mtp?, then GDN conv/ssm in layer order]` (no K/V).
/// The full-attn host mirror is left stale (the arena is the source of truth);
/// it is re-synced only on a bucket change (read_gpu_handle → mirror → rebind).
pub fn advance_cache_resident_inplace(
    cfg: &Qwen35Config,
    cache: &mut Qwen35DecodeCache,
    outputs: Vec<Vec<f32>>,
    mtp_logits_path: bool,
    want_mtp: bool,
) -> anyhow::Result<(Vec<f32>, Option<Vec<f32>>)> {
    use anyhow::{Context, bail};
    let batch = cache.batch;
    if batch != 1 {
        bail!("advance_cache_resident_inplace: resident KV path is batch==1 only");
    }
    let mut iter = outputs.into_iter();
    let trunk = iter
        .next()
        .context("resident decode: logits output missing")?;
    let logits = truncate_logits_row(cfg, trunk, batch);
    let mtp = if mtp_logits_path {
        let raw = iter.next().context("resident decode: mtp logits missing")?;
        if want_mtp {
            Some(parse_mtp_logits(cfg, batch, raw)?)
        } else {
            None
        }
    } else {
        None
    };

    let kinds = trunk_layer_kinds(cfg);
    for (il, layer) in cache.layers.iter_mut().enumerate() {
        if kinds[il] {
            continue; // full-attn K/V resident in arena; mirror untouched
        }
        let conv = iter.next().context("resident decode: conv_state missing")?;
        let ssm = iter.next().context("resident decode: ssm_state missing")?;
        *layer = Qwen35LayerState::Linear {
            conv_state: conv,
            ssm_state: ssm,
        };
    }
    cache.past_seq += 1;
    cache.abs_pos += 1;
    Ok((logits, mtp))
}

/// Describe per-layer buffer sizes for a config (trunk only).
#[allow(dead_code)]
pub fn trunk_layer_state_sizes(cfg: &Qwen35Config) -> Vec<(bool, usize, usize)> {
    let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
    let interval = cfg.full_attention_interval.max(1);
    let n_state = cfg.ssm_state_size;
    let n_v_heads = cfg.ssm_time_step_rank;
    let conv_channels = linear_conv_channels(cfg);
    let k_conv = cfg.ssm_conv_kernel;

    let mut out = Vec::with_capacity(n_main);
    for il in 0..n_main {
        let is_full_attn = ((il + 1) % interval) == 0;
        if is_full_attn {
            out.push((true, 0, 0));
        } else {
            out.push((
                false,
                (k_conv - 1) * conv_channels,
                n_v_heads * n_state * n_state,
            ));
        }
    }
    out
}

/// Pack per-row prompts into `[batch, max_seq]` row-major F32 ids (zero-pad).
pub fn pack_input_ids(batch_prompts: &[Vec<u32>], max_seq: usize) -> anyhow::Result<Vec<f32>> {
    use anyhow::bail;
    if batch_prompts.is_empty() {
        bail!("pack_input_ids: batch must be non-empty");
    }
    let batch = batch_prompts.len();
    let mut out = vec![0f32; batch * max_seq];
    for (b, prompt) in batch_prompts.iter().enumerate() {
        if prompt.len() > max_seq {
            bail!(
                "pack_input_ids: row {b} length {} exceeds max_seq={max_seq}",
                prompt.len()
            );
        }
        let base = b * max_seq;
        for (i, &t) in prompt.iter().enumerate() {
            out[base + i] = t as f32;
        }
    }
    Ok(out)
}

/// Per-row index of the last real prompt token (0-based).
pub fn last_token_indices(prompt_lens: &[usize]) -> Vec<f32> {
    prompt_lens
        .iter()
        .map(|&l| l.saturating_sub(1) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_full_attn_cfg() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 16,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            nextn_predict_layers: 0,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            key_length: 2,
            value_length: 2,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_dim_count: 2,
            rope_dim_sections: vec![],
            mrope_interleaved: false,
            rms_norm_offset: false,
            full_attention_interval: 1,
            ssm_conv_kernel: 4,
            ssm_group_count: 2,
            ssm_inner_size: 8,
            ssm_state_size: 4,
            ssm_time_step_rank: 2,
            tie_word_embeddings: true,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    #[test]
    fn advance_decode_consumes_mtp_before_kv_states() {
        let cfg = one_full_attn_cfg();
        let batch = 1;
        let past_seq = 1;
        let kv_cols = cfg.num_key_value_heads * cfg.key_length;
        let new_past = past_seq + 1;
        let kv_len = batch * new_past * kv_cols;

        let mut cache = Qwen35DecodeCache {
            batch,
            past_seq,
            abs_pos: past_seq,
            prompt_lens: vec![past_seq],
            layers: vec![Qwen35LayerState::FullAttn {
                past_k: vec![0.0; batch * past_seq * kv_cols],
                past_v: vec![0.0; batch * past_seq * kv_cols],
                packed: None,
            }],
        };

        let trunk_logits = vec![1.0; batch * cfg.vocab_size];
        let mtp_logits = vec![2.0; batch * cfg.vocab_size];
        assert_ne!(
            mtp_logits.len(),
            kv_len,
            "test needs distinct mtp vs kv lengths"
        );
        let new_k = vec![3.0; kv_len];
        let new_v = vec![4.0; kv_len];

        let outputs = vec![
            trunk_logits.clone(),
            mtp_logits.clone(),
            new_k.clone(),
            new_v.clone(),
        ];
        let (trunk_out, mtp) =
            advance_cache_from_decode_outputs(&cfg, &mut cache, outputs, None, true, true, false)
                .unwrap();
        assert_eq!(trunk_out, trunk_logits);
        assert_eq!(mtp.unwrap(), mtp_logits);
        assert_eq!(cache.past_seq, new_past);
        match &cache.layers[0] {
            Qwen35LayerState::FullAttn {
                past_k,
                past_v,
                packed: None,
            } => {
                assert_eq!(past_k, &new_k);
                assert_eq!(past_v, &new_v);
            }
            _ => panic!("expected full-attn layer"),
        }

        let mut cache2 = cache.clone();
        cache2.past_seq = past_seq;
        let bad = vec![trunk_logits, new_k, new_v, mtp_logits];
        assert!(
            advance_cache_from_decode_outputs(&cfg, &mut cache2, bad, None, true, true, false)
                .is_err()
        );
    }

    #[test]
    fn advance_decode_discards_mtp_when_not_wanted() {
        let cfg = one_full_attn_cfg();
        let batch = 1;
        let kv_cols = cfg.num_key_value_heads * cfg.key_length;
        let kv_len = batch * 2 * kv_cols;

        let mut cache = Qwen35DecodeCache {
            batch,
            past_seq: 1,
            abs_pos: 1,
            prompt_lens: vec![1],
            layers: vec![Qwen35LayerState::FullAttn {
                past_k: vec![0.0; batch * kv_cols],
                past_v: vec![0.0; batch * kv_cols],
                packed: None,
            }],
        };

        let outputs = vec![
            vec![0.0; batch * cfg.vocab_size],
            vec![1.0; batch * cfg.vocab_size],
            vec![2.0; kv_len],
            vec![3.0; kv_len],
        ];
        let (_, mtp) =
            advance_cache_from_decode_outputs(&cfg, &mut cache, outputs, None, true, false, false)
                .unwrap();
        assert!(mtp.is_none());
    }

    #[test]
    fn decode_attention_mask_enables_upper_slot() {
        let past_seq = 17usize;
        let upper = 32usize;
        let mask = build_decode_attention_mask(1, past_seq, upper, &[35], &[1]);
        assert_eq!(mask.len(), upper + 1);
        for (i, &v) in mask.iter().enumerate() {
            let want = if i < past_seq || i == upper { 1.0 } else { 0.0 };
            assert_eq!(v, want, "mask[{i}]");
        }
    }

    #[test]
    fn slice_kv_from_bucket_keeps_new_row_at_upper() {
        // past_seq=2, upper=4 → output rows [p0, p1, pad, pad, new]
        let batch = 1;
        let past_seq = 2usize;
        let upper = 4usize;
        let kv_cols = 3usize;
        let new_past = past_seq + 1;
        let mut src = vec![0f32; batch * (upper + 1) * kv_cols];
        // past rows
        src[0..3].copy_from_slice(&[1.0, 1.0, 1.0]);
        src[3..6].copy_from_slice(&[2.0, 2.0, 2.0]);
        // new token at upper
        src[upper * kv_cols..(upper + 1) * kv_cols].copy_from_slice(&[9.0, 9.0, 9.0]);
        let out = slice_kv_from_bucket(&src, batch, new_past, upper, kv_cols).unwrap();
        assert_eq!(out, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn slice_kv_from_bucket_prefix_when_no_decode_pad() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0];
        // batch=1, src_seq=4, kv=2, take first 3 rows
        let out = slice_kv_from_bucket(&src, 1, 3, 4, 2).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
