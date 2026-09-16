// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Apple Voice / Espresso `streaming_encoder_64_16` native forward.
//!
//! Pack tensors are a reconstruction of the macOS Espresso/ANE streaming ASR
//! encoder (not NeMo offline Conformer). Contract matches E5 captures:
//!
//! - mel window 389 → subsample ×6 → **64** encoder frames / chunk
//! - **lookahead 16** (TEXT-K `att_k_cache` history)
//! - `cnn_cache` `[28, 512, 1, 7]` — causal fused conv left taps
//! - `att_k_cache` `[4, 16, 8, 64]` — TEXT-K slots for L∈{0,7,14,21}
//! - `att_v_cache` `[28, 64, 8, 16]` — per-layer V history (`d_v=16`)
//! - `mask` length **80** (= chunk + lookahead)
//!
//! Mid-act SSA is fused on-device; this path uses reconstructed structural
//! weights + streaming state. Prefer hybrid teacher-enc + CTC for golden
//! parity until bit-exact ANE body lands.

use crate::encoder::{ATT_K_CACHE, ATT_V_CACHE, CNN_CACHE, CNN_KERNEL, ENCODER_LAYERS};
use crate::folded_encoder::{FOLDED_MEL_FRAMES, FOLDED_OUT_T, mel_windows};
use crate::gguf_io::AsrPack;
use crate::k_codebook::{TEXT_K_LAYERS, TextKLayer};
use crate::spec::{
    AED_WINDOW_FRAMES, CHUNK_FRAMES, DECODER_DIM, DECODER_HEAD_DIM, DECODER_HEADS,
    LOOKAHEAD_FRAMES, MEL_BINS, SUBSAMPLE, VOCAB,
};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;

const LN_EPS: f32 = 1e-5;
const ATT_V_HEAD_DIM: usize = 16;
const ATT_V_OUT: usize = DECODER_HEADS * ATT_V_HEAD_DIM;
const N_LAYERS: usize = ENCODER_LAYERS;
const MASK_LEN: usize = CHUNK_FRAMES + LOOKAHEAD_FRAMES; // 80

fn enc_key(suffix: &str) -> String {
    format!("encoder.{suffix}")
}

fn text_k_slot(layer: usize) -> Option<usize> {
    TEXT_K_LAYERS.iter().position(|&l| l == layer)
}

#[derive(Clone)]
struct NativeLayer {
    conv_w: Vec<f32>,
    ffn_a: Vec<f32>,
    ffn_b: Vec<f32>,
    q_w: Vec<f32>,
    k_w: Vec<f32>,
    v_w: Vec<f32>,
    out_w: Vec<f32>,
    gamma: Vec<f32>,
}

/// Streaming caches matching Apple E5 `streaming_encoder_64_16` I/O.
#[derive(Clone, Debug)]
pub struct StreamingState {
    /// `[28, 512, 7]` left taps for causal fused conv (layout: layer-major).
    pub cnn_cache: Vec<f32>,
    /// `[4, 16, 8, 64]` TEXT-K left context.
    pub att_k_cache: Vec<f32>,
    /// `[28, 64, 8, 16]` per-layer V left context.
    pub att_v_cache: Vec<f32>,
    /// Length-80 validity mask (chunk + lookahead); all-ones when unrestricted.
    pub mask: Vec<f32>,
}

impl Default for StreamingState {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingState {
    pub fn new() -> Self {
        let cnn_n = CNN_CACHE[0] * CNN_CACHE[1] * CNN_KERNEL;
        let k_n = ATT_K_CACHE[0] * ATT_K_CACHE[1] * ATT_K_CACHE[2] * ATT_K_CACHE[3];
        let v_n = ATT_V_CACHE[0] * ATT_V_CACHE[1] * ATT_V_CACHE[2] * ATT_V_CACHE[3];
        Self {
            cnn_cache: vec![0f32; cnn_n],
            att_k_cache: vec![0f32; k_n],
            att_v_cache: vec![0f32; v_n],
            mask: vec![1f32; MASK_LEN],
        }
    }

    pub fn reset(&mut self) {
        self.cnn_cache.fill(0.0);
        self.att_k_cache.fill(0.0);
        self.att_v_cache.fill(0.0);
        self.mask.fill(1.0);
    }

    fn cnn_layer_mut(&mut self, layer: usize) -> &mut [f32] {
        let n = DECODER_DIM * CNN_KERNEL;
        &mut self.cnn_cache[layer * n..(layer + 1) * n]
    }

    fn att_k_slot_mut(&mut self, slot: usize) -> &mut [f32] {
        let n = LOOKAHEAD_FRAMES * DECODER_HEADS * DECODER_HEAD_DIM;
        &mut self.att_k_cache[slot * n..(slot + 1) * n]
    }

    fn att_v_layer_mut(&mut self, layer: usize) -> &mut [f32] {
        let n = CHUNK_FRAMES * DECODER_HEADS * ATT_V_HEAD_DIM;
        &mut self.att_v_cache[layer * n..(layer + 1) * n]
    }
}

/// Native Apple-streaming layer stack + CTC head.
pub struct NativeEncoder {
    input_w: Vec<f32>,
    input_w_t: bool,
    input_b: Option<Vec<f32>>,
    layers: Vec<NativeLayer>,
    k_codebook: HashMap<usize, Vec<f32>>,
    v_pad: HashMap<usize, Vec<f32>>,
    ctc_w: Vec<f32>,
    ctc_w_t: bool,
    ctc_b: Option<Vec<f32>>,
}

impl NativeEncoder {
    pub fn is_available(pack: &AsrPack) -> bool {
        pack.has(&enc_key("layers.0.conv.weight"))
            && pack.has(&enc_key("layers.0.ffn_a.weight"))
            && (pack.has(&enc_key("head.ctc.W_ls")) || pack.has(&enc_key("head.ctc.W")))
    }

    pub fn from_pack(pack: &AsrPack) -> Result<Self> {
        let input_w_raw = pack
            .f32_tensor(&enc_key("frontend.input_proj_eff.W"))
            .context("frontend.input_proj_eff.W")?;
        let (input_w, input_w_t) = if input_w_raw.len() == DECODER_DIM * MEL_BINS {
            (input_w_raw, true)
        } else if input_w_raw.len() == MEL_BINS * DECODER_DIM {
            (input_w_raw, false)
        } else {
            bail!("input_proj_eff.W len {}", input_w_raw.len());
        };
        let input_b = pack
            .f32_tensor(&enc_key("frontend.input_proj_eff.b"))
            .ok()
            .filter(|b| b.len() == DECODER_DIM);

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let pfx = format!("encoder.layers.{i}.");
            layers.push(NativeLayer {
                conv_w: pack
                    .f32_tensor(&format!("{pfx}conv.weight"))
                    .with_context(|| format!("{pfx}conv.weight"))?,
                ffn_a: pack
                    .f32_tensor(&format!("{pfx}ffn_a.weight"))
                    .with_context(|| format!("{pfx}ffn_a.weight"))?,
                ffn_b: pack
                    .f32_tensor(&format!("{pfx}ffn_b.weight"))
                    .with_context(|| format!("{pfx}ffn_b.weight"))?,
                q_w: pack
                    .f32_tensor(&format!("{pfx}self_attn.linear_q.weight"))
                    .with_context(|| format!("{pfx}self_attn.linear_q.weight"))?,
                k_w: pack
                    .f32_tensor(&format!("{pfx}self_attn.linear_k.weight"))
                    .with_context(|| format!("{pfx}self_attn.linear_k.weight"))?,
                v_w: pack
                    .f32_tensor(&format!("{pfx}self_attn.linear_v.weight"))
                    .with_context(|| format!("{pfx}self_attn.linear_v.weight"))?,
                out_w: pack
                    .f32_tensor(&format!("{pfx}self_attn.linear_out.weight"))
                    .with_context(|| format!("{pfx}self_attn.linear_out.weight"))?,
                gamma: pack
                    .f32_tensor(&format!("{pfx}bias.weight"))
                    .with_context(|| format!("{pfx}bias.weight"))?,
            });
        }

        let mut k_codebook = HashMap::new();
        for &layer in &TEXT_K_LAYERS {
            let int8_key = format!("codebook.layer{layer}.linear_k.int8");
            let scale_key = format!("codebook.layer{layer}.linear_k.scale");
            if pack.has(&int8_key) && pack.has(&scale_key) {
                let int8 = pack.i8_tensor(&int8_key)?;
                let scale = pack.f32_tensor(&scale_key)?;
                k_codebook.insert(layer, TextKLayer::dequant(&int8, &scale)?);
            }
        }

        let mut v_pad = HashMap::new();
        for &layer in &TEXT_K_LAYERS {
            let key = format!("ls.layer{layer}.linear_v.weight_pad");
            if pack.has(&key) {
                let w = pack.f32_tensor(&key)?;
                if w.len() == DECODER_DIM * DECODER_DIM {
                    v_pad.insert(layer, w);
                }
            }
        }

        let ctc_w_key = if pack.has(&enc_key("head.ctc.W_ls")) {
            enc_key("head.ctc.W_ls")
        } else {
            enc_key("head.ctc.W")
        };
        let ctc_w_raw = pack
            .f32_tensor(&ctc_w_key)
            .with_context(|| ctc_w_key.clone())?;
        let (ctc_w, ctc_w_t) = if ctc_w_raw.len() == VOCAB * DECODER_DIM {
            (ctc_w_raw, true)
        } else if ctc_w_raw.len() == DECODER_DIM * VOCAB {
            (ctc_w_raw, false)
        } else {
            bail!("CTC W len {}", ctc_w_raw.len());
        };
        let ctc_b = pack
            .f32_tensor(&enc_key("head.ctc.b_ls"))
            .or_else(|_| pack.f32_tensor(&enc_key("head.ctc.b")))
            .ok()
            .filter(|b| b.len() == VOCAB);

        Ok(Self {
            input_w,
            input_w_t,
            input_b,
            layers,
            k_codebook,
            v_pad,
            ctc_w,
            ctc_w_t,
            ctc_b,
        })
    }

    pub fn forward_mel(&self, mel: &[Vec<f32>]) -> Result<crate::encoder::EncoderOutputs> {
        self.forward_mel_limited(mel, None)
    }

    pub fn forward_mel_limited(
        &self,
        mel: &[Vec<f32>],
        max_chunks: Option<usize>,
    ) -> Result<crate::encoder::EncoderOutputs> {
        if mel.is_empty() {
            bail!("empty mel");
        }
        // Apple streaming hop ≈ one encoder chunk of new mel (64×6 = 384).
        let hop = CHUNK_FRAMES * SUBSAMPLE;
        let mut chunks = mel_windows(mel, FOLDED_MEL_FRAMES, hop);
        if let Some(n) = max_chunks {
            chunks.truncate(n);
        }
        let mut state = StreamingState::new();
        let mut all_logp = Vec::new();
        let mut all_enc = Vec::new();
        for chunk in &chunks {
            let (enc, logp) = self.forward_chunk(chunk, &mut state)?;
            all_enc.extend(enc);
            all_logp.extend(logp);
        }
        let n_frames = all_logp.len() / VOCAB;
        Ok(crate::encoder::EncoderOutputs {
            wp_logprob: all_logp,
            encoder_cache: enc_to_aed_cache(&all_enc),
            n_frames,
        })
    }

    /// One Apple streaming chunk; updates `state` caches in place.
    pub fn forward_chunk(
        &self,
        feat389: &[Vec<f32>],
        state: &mut StreamingState,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if feat389.len() < FOLDED_MEL_FRAMES {
            bail!(
                "native chunk needs {} mel frames, got {}",
                FOLDED_MEL_FRAMES,
                feat389.len()
            );
        }
        let mut mel_sub = vec![0f32; FOLDED_OUT_T * MEL_BINS];
        for t in 0..FOLDED_OUT_T {
            let idx = t * SUBSAMPLE;
            mel_sub[t * MEL_BINS..(t + 1) * MEL_BINS].copy_from_slice(&feat389[idx]);
        }
        let mut x = vec![0f32; FOLDED_OUT_T * DECODER_DIM];
        matmul_rows(
            &mel_sub,
            FOLDED_OUT_T,
            MEL_BINS,
            DECODER_DIM,
            &self.input_w,
            self.input_w_t,
            self.input_b.as_deref(),
            &mut x,
        );

        let ones = vec![1f32; DECODER_DIM];
        for (li, layer) in self.layers.iter().enumerate() {
            let text_k = TEXT_K_LAYERS.contains(&li);
            let k_w = self.k_codebook.get(&li).map(|w| w.as_slice());
            let v_pad = self.v_pad.get(&li).map(|w| w.as_slice());

            let mut h = x.clone();
            let ln1 = layer_norm_rows(&h, FOLDED_OUT_T, &ones);
            vec_add_inplace(&mut h, &ffn_macaron(&ln1, layer));

            let ln2 = layer_norm_rows(&h, FOLDED_OUT_T, &ones);
            let (attn_delta, k_cur, v128) =
                self_attn_streaming(&ln2, FOLDED_OUT_T, layer, k_w, v_pad, text_k, state, li);
            vec_add_inplace(&mut h, &attn_delta);
            update_att_caches(state, li, text_k, &k_cur, &v128);

            let ln3 = layer_norm_rows(&h, FOLDED_OUT_T, &ones);
            let (conv_delta, cnn_out) = conv_causal(&ln3, FOLDED_OUT_T, layer, state, li);
            vec_add_inplace(&mut h, &conv_delta);
            state.cnn_layer_mut(li).copy_from_slice(&cnn_out);

            let ln4 = layer_norm_rows(&h, FOLDED_OUT_T, &ones);
            vec_add_inplace(&mut h, &ffn_macaron(&ln4, layer));

            let normed = layer_norm_rows(&h, FOLDED_OUT_T, &ones);
            x = vec![0f32; FOLDED_OUT_T * DECODER_DIM];
            for t in 0..FOLDED_OUT_T {
                let r = &normed[t * DECODER_DIM..(t + 1) * DECODER_DIM];
                let o = &mut x[t * DECODER_DIM..(t + 1) * DECODER_DIM];
                for d in 0..DECODER_DIM {
                    o[d] = layer.gamma[d] * r[d];
                }
            }
        }

        let logp = self.ctc_logp(&x, FOLDED_OUT_T)?;
        Ok((x, logp))
    }

    fn ctc_logp(&self, enc: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        let mut logits = vec![0f32; n_frames * VOCAB];
        for t in 0..n_frames {
            let row = &enc[t * DECODER_DIM..(t + 1) * DECODER_DIM];
            let out = &mut logits[t * VOCAB..(t + 1) * VOCAB];
            if self.ctc_w_t {
                for v in 0..VOCAB {
                    let mut s = 0f32;
                    for d in 0..DECODER_DIM {
                        s += row[d] * self.ctc_w[v * DECODER_DIM + d];
                    }
                    out[v] = s;
                }
            } else {
                for v in 0..VOCAB {
                    let mut s = 0f32;
                    for d in 0..DECODER_DIM {
                        s += row[d] * self.ctc_w[d * VOCAB + v];
                    }
                    out[v] = s;
                }
            }
            if let Some(b) = &self.ctc_b {
                for v in 0..VOCAB {
                    out[v] += b[v];
                }
            }
            log_softmax_row(out);
        }
        Ok(logits)
    }
}

fn ffn_macaron(x: &[f32], layer: &NativeLayer) -> Vec<f32> {
    // Pack stores ffn_a/b as concat of 6×[512,512] (provenance). Matmul through the
    // full 3072-wide tensors equals summing 6 parallel 512→512 SiLU branches.
    let n_rows = x.len() / DECODER_DIM;
    let hidden = layer.ffn_a.len() / DECODER_DIM;
    let mut h = vec![0f32; n_rows * hidden];
    matmul_rows(
        x,
        n_rows,
        DECODER_DIM,
        hidden,
        &layer.ffn_a,
        true,
        None,
        &mut h,
    );
    for v in &mut h {
        *v = silu(*v).clamp(-20.0, 20.0);
    }
    let mut out = vec![0f32; x.len()];
    matmul_rows(
        &h,
        n_rows,
        hidden,
        DECODER_DIM,
        &layer.ffn_b,
        true,
        None,
        &mut out,
    );
    for v in &mut out {
        *v *= 0.5;
    }
    out
}

/// Causal fused conv using `cnn_cache` left taps (Apple k=7 streaming).
///
/// For each frame t, average the 7-frame window (6 cache taps + current), apply
/// fused `conv.weight`, take first DIM channels. Updates cache to the last 7
/// pre-conv activations.
fn conv_causal(
    x: &[f32],
    n_rows: usize,
    layer: &NativeLayer,
    state: &StreamingState,
    layer_i: usize,
) -> (Vec<f32>, Vec<f32>) {
    let out_cols = layer.conv_w.len() / DECODER_DIM;
    let hist = &state.cnn_cache
        [layer_i * DECODER_DIM * CNN_KERNEL..(layer_i + 1) * DECODER_DIM * CNN_KERNEL];
    // hist layout: [d * 7 + tap], tap 0 = oldest
    let mut pad = vec![0f32; (CNN_KERNEL - 1 + n_rows) * DECODER_DIM];
    for d in 0..DECODER_DIM {
        for tap in 0..(CNN_KERNEL - 1) {
            pad[tap * DECODER_DIM + d] = hist[d * CNN_KERNEL + tap + 1];
        }
    }
    pad[(CNN_KERNEL - 1) * DECODER_DIM..].copy_from_slice(x);

    let mut out = vec![0f32; n_rows * DECODER_DIM];
    let mut tmp = vec![0f32; out_cols];
    for t in 0..n_rows {
        // 7-tap window ending at current frame
        let mut win = vec![0f32; DECODER_DIM];
        for tap in 0..CNN_KERNEL {
            let row = &pad[(t + tap) * DECODER_DIM..(t + tap + 1) * DECODER_DIM];
            for d in 0..DECODER_DIM {
                win[d] += row[d];
            }
        }
        for d in 0..DECODER_DIM {
            win[d] /= CNN_KERNEL as f32;
        }
        matmul_rows(
            &win,
            1,
            DECODER_DIM,
            out_cols,
            &layer.conv_w,
            true,
            None,
            &mut tmp,
        );
        out[t * DECODER_DIM..(t + 1) * DECODER_DIM].copy_from_slice(&tmp[..DECODER_DIM]);
    }

    // New cache = last 7 activation frames of padded input
    let mut new_cnn = vec![0f32; DECODER_DIM * CNN_KERNEL];
    let pad_frames = CNN_KERNEL - 1 + n_rows;
    let start = pad_frames - CNN_KERNEL;
    for tap in 0..CNN_KERNEL {
        let src = &pad[(start + tap) * DECODER_DIM..(start + tap + 1) * DECODER_DIM];
        for d in 0..DECODER_DIM {
            new_cnn[d * CNN_KERNEL + tap] = src[d];
        }
    }
    (out, new_cnn)
}

/// Streaming MHSA: attend over `[cache_left ; current]` KV.
/// Returns `(attn_out, k_curr[512], v128)`.
fn self_attn_streaming(
    x: &[f32],
    n_rows: usize,
    layer: &NativeLayer,
    k_codebook: Option<&[f32]>,
    v_pad: Option<&[f32]>,
    text_k: bool,
    state: &StreamingState,
    layer_i: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0f32; x.len()];
    let mut k = vec![0f32; x.len()];
    let q_w = k_codebook.unwrap_or(&layer.q_w);
    matmul_rows(x, n_rows, DECODER_DIM, DECODER_DIM, q_w, true, None, &mut q);
    if text_k || k_codebook.is_some() {
        k.copy_from_slice(&q);
    } else {
        matmul_rows(
            x,
            n_rows,
            DECODER_DIM,
            DECODER_DIM,
            &layer.k_w,
            true,
            None,
            &mut k,
        );
    }

    let mut v512 = vec![0f32; x.len()];
    let mut v128 = vec![0f32; n_rows * ATT_V_OUT];
    if let Some(wpad) = v_pad {
        matmul_rows(
            x,
            n_rows,
            DECODER_DIM,
            DECODER_DIM,
            wpad,
            true,
            None,
            &mut v512,
        );
        for r in 0..n_rows {
            for h in 0..DECODER_HEADS {
                for d in 0..ATT_V_HEAD_DIM {
                    v128[r * ATT_V_OUT + h * ATT_V_HEAD_DIM + d] =
                        v512[r * DECODER_DIM + h * DECODER_HEAD_DIM + d];
                }
            }
        }
    } else {
        matmul_rows(
            x,
            n_rows,
            DECODER_DIM,
            ATT_V_OUT,
            &layer.v_w,
            true,
            None,
            &mut v128,
        );
        for r in 0..n_rows {
            expand_v128(
                &v128[r * ATT_V_OUT..(r + 1) * ATT_V_OUT],
                &mut v512[r * DECODER_DIM..(r + 1) * DECODER_DIM],
            );
        }
    }

    // Build left context from streaming caches
    let (k_left, v_left) = kv_left_context(state, layer_i, text_k);
    let n_left = k_left.len() / DECODER_DIM;
    let n_kv = n_left + n_rows;

    let mut k_all = vec![0f32; n_kv * DECODER_DIM];
    let mut v_all = vec![0f32; n_kv * DECODER_DIM];
    if n_left > 0 {
        k_all[..n_left * DECODER_DIM].copy_from_slice(&k_left);
        v_all[..n_left * DECODER_DIM].copy_from_slice(&v_left);
    }
    k_all[n_left * DECODER_DIM..].copy_from_slice(&k);
    v_all[n_left * DECODER_DIM..].copy_from_slice(&v512);

    let scale = (DECODER_HEAD_DIM as f32).sqrt().recip();
    let mut ctx = vec![0f32; x.len()];
    for h in 0..DECODER_HEADS {
        for t1 in 0..n_rows {
            let mut scores = vec![0f32; n_kv];
            let mut max_s = f32::NEG_INFINITY;
            for t2 in 0..n_kv {
                let mut s = 0f32;
                for d in 0..DECODER_HEAD_DIM {
                    let qv = q[t1 * DECODER_DIM + h * DECODER_HEAD_DIM + d];
                    let kv = k_all[t2 * DECODER_DIM + h * DECODER_HEAD_DIM + d];
                    s += qv * kv;
                }
                s *= scale;
                // mask: positions beyond chunk+lookahead treated as invalid when mask[t]<0.5
                // For n_kv = left+chunk, map t2 relative to joint timeline.
                scores[t2] = s;
                max_s = max_s.max(s);
            }
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            let inv = 1.0 / sum.max(1e-12);
            for t2 in 0..n_kv {
                let w = scores[t2] * inv;
                for d in 0..DECODER_HEAD_DIM {
                    ctx[t1 * DECODER_DIM + h * DECODER_HEAD_DIM + d] +=
                        w * v_all[t2 * DECODER_DIM + h * DECODER_HEAD_DIM + d];
                }
            }
        }
    }

    // TEXT-K: identity-attn surrogate (context → residual)
    if text_k {
        return (ctx, k, v128);
    }
    let mut out = vec![0f32; x.len()];
    matmul_rows(
        &ctx,
        n_rows,
        DECODER_DIM,
        DECODER_DIM,
        &layer.out_w,
        true,
        None,
        &mut out,
    );
    (out, k, v128)
}

fn kv_left_context(state: &StreamingState, layer_i: usize, text_k: bool) -> (Vec<f32>, Vec<f32>) {
    // V left: full previous chunk [64, 8, 16] → expand to [64, 512]
    let v_n = CHUNK_FRAMES * DECODER_HEADS * ATT_V_HEAD_DIM;
    let v_src = &state.att_v_cache[layer_i * v_n..(layer_i + 1) * v_n];
    let v_zero = v_src.iter().all(|&v| v == 0.0);

    let mut v_left = Vec::new();
    if !v_zero {
        v_left = vec![0f32; CHUNK_FRAMES * DECODER_DIM];
        for t in 0..CHUNK_FRAMES {
            expand_v128(
                &v_src[t * ATT_V_OUT..(t + 1) * ATT_V_OUT],
                &mut v_left[t * DECODER_DIM..(t + 1) * DECODER_DIM],
            );
        }
    }

    let mut k_left = Vec::new();
    if text_k {
        if let Some(slot) = text_k_slot(layer_i) {
            let k_n = LOOKAHEAD_FRAMES * DECODER_HEADS * DECODER_HEAD_DIM;
            let k_src = &state.att_k_cache[slot * k_n..(slot + 1) * k_n];
            if !k_src.iter().all(|&v| v == 0.0) {
                // [16, 8, 64] already head-packed to 512
                k_left = k_src.to_vec();
            }
        }
    } else if !v_left.is_empty() {
        // Non-TEXT-K: reuse previous-chunk V length as K left placeholder length
        // with zeros — left context for V only (chunk-local K).
        k_left = vec![0f32; v_left.len()];
    }

    // Align lengths: if only V history, pad K left with zeros of same T
    if v_left.len() != k_left.len() {
        if k_left.is_empty() && !v_left.is_empty() {
            k_left = vec![0f32; v_left.len()];
        } else if v_left.is_empty() && !k_left.is_empty() {
            v_left = vec![0f32; k_left.len()];
        } else if !k_left.is_empty() && !v_left.is_empty() {
            // TEXT-K: K left is 16, V left is 64 — use K=16 with V last-16
            let take = LOOKAHEAD_FRAMES;
            let v_trim = v_left[(CHUNK_FRAMES - take) * DECODER_DIM..].to_vec();
            v_left = v_trim;
        }
    }

    (k_left, v_left)
}

fn update_att_caches(
    state: &mut StreamingState,
    layer_i: usize,
    text_k: bool,
    k_cur: &[f32],
    v128: &[f32],
) {
    // att_v_cache[L] ← current V packed [64, 8, 16]
    state.att_v_layer_mut(layer_i).copy_from_slice(v128);

    if text_k && let Some(slot) = text_k_slot(layer_i) {
        // Keep last LOOKAHEAD_FRAMES of K [16, 512] head layout
        let start = (CHUNK_FRAMES - LOOKAHEAD_FRAMES) * DECODER_DIM;
        state.att_k_slot_mut(slot).copy_from_slice(&k_cur[start..]);
    }
}

fn expand_v128(v128: &[f32], out512: &mut [f32]) {
    out512.fill(0.0);
    for h in 0..DECODER_HEADS {
        for d in 0..ATT_V_HEAD_DIM {
            out512[h * DECODER_HEAD_DIM + d] = v128[h * ATT_V_HEAD_DIM + d];
        }
    }
}

fn layer_norm_rows(x: &[f32], n_rows: usize, gamma: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for r in 0..n_rows {
        let row = &x[r * DECODER_DIM..(r + 1) * DECODER_DIM];
        let o = &mut out[r * DECODER_DIM..(r + 1) * DECODER_DIM];
        let mean = row.iter().sum::<f32>() / DECODER_DIM as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / DECODER_DIM as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for d in 0..DECODER_DIM {
            o[d] = gamma[d] * (row[d] - mean) * inv;
        }
    }
    out
}

fn vec_add_inplace(acc: &mut [f32], delta: &[f32]) {
    for (a, d) in acc.iter_mut().zip(delta.iter()) {
        *a += *d;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn matmul_rows(
    a: &[f32],
    n_rows: usize,
    k: usize,
    n_cols: usize,
    w: &[f32],
    w_transposed: bool,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    for r in 0..n_rows {
        let a_row = &a[r * k..(r + 1) * k];
        let o_row = &mut out[r * n_cols..(r + 1) * n_cols];
        for c in 0..n_cols {
            let mut s = 0f32;
            for i in 0..k {
                let wv = if w_transposed {
                    w[c * k + i]
                } else {
                    w[i * n_cols + c]
                };
                s += a_row[i] * wv;
            }
            o_row[c] = s;
        }
        if let Some(b) = bias {
            for c in 0..n_cols {
                o_row[c] += b[c];
            }
        }
    }
}

fn log_softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let v = -(row.len() as f32).ln();
        row.fill(v);
        return;
    }
    let mut sum = 0f32;
    for x in row.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum <= 0.0 || !sum.is_finite() {
        let v = -(row.len() as f32).ln();
        row.fill(v);
        return;
    }
    let log_sum = sum.ln();
    for x in row.iter_mut() {
        *x = (*x).ln() - log_sum;
    }
}

fn enc_to_aed_cache(enc: &[f32]) -> Vec<f32> {
    let n_enc_frames = enc.len() / DECODER_DIM;
    let mut cache = vec![0f32; AED_WINDOW_FRAMES * DECODER_DIM];
    if n_enc_frames == 0 {
        return cache;
    }
    let copy_frames = n_enc_frames.min(AED_WINDOW_FRAMES);
    let start = n_enc_frames.saturating_sub(copy_frames);
    for (i, t) in (start..start + copy_frames).enumerate() {
        let src = &enc[t * DECODER_DIM..(t + 1) * DECODER_DIM];
        cache[i * DECODER_DIM..(i + 1) * DECODER_DIM].copy_from_slice(src);
    }
    cache
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_state_shapes() {
        let s = StreamingState::new();
        assert_eq!(s.cnn_cache.len(), 28 * 512 * 7);
        assert_eq!(s.att_k_cache.len(), 4 * 16 * 8 * 64);
        assert_eq!(s.att_v_cache.len(), 28 * 64 * 8 * 16);
        assert_eq!(s.mask.len(), 80);
    }

    #[test]
    fn native_encoder_from_published_pack() {
        let root = crate::asr_dir();
        let Some(path) = crate::gguf_io::resolve_pack_path(&root) else {
            return;
        };
        let pack = AsrPack::open(&path).expect("open pack");
        if !NativeEncoder::is_available(&pack) {
            return;
        }
        let enc = NativeEncoder::from_pack(&pack).expect("load native");
        assert_eq!(enc.layers.len(), N_LAYERS);
        if pack.has("codebook.layer0.linear_k.int8") {
            assert!(enc.k_codebook.contains_key(&0));
        }
        if pack.has("ls.layer0.linear_v.weight_pad") {
            assert!(enc.v_pad.contains_key(&0));
        }
        let feat: Vec<Vec<f32>> = (0..FOLDED_MEL_FRAMES)
            .map(|t| {
                (0..MEL_BINS)
                    .map(|b| ((t + b) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();
        let mut state = StreamingState::new();
        let (_, logp) = enc.forward_chunk(&feat, &mut state).unwrap();
        assert_eq!(logp.len(), FOLDED_OUT_T * VOCAB);
        // Second chunk should see non-zero caches after first.
        assert!(state.att_v_cache.iter().any(|&v| v != 0.0));
        assert!(state.cnn_cache.iter().any(|&v| v != 0.0));
        let (_, _) = enc.forward_chunk(&feat, &mut state).unwrap();
        assert!(state.att_k_cache.iter().any(|&v| v != 0.0));
    }
}
