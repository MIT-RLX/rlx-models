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

//! Qwen3-Omni audio encoder HIR: chunked Conv2d stem → sinusoidal positions →
//! windowed transformer → LayerNorm + 2-layer adapter to the LM hidden size.

use crate::audio::AudioGeometry;
use crate::config::AudioEncoderConfig;
use crate::weights::AsrWeightPrefix;
use anyhow::Result;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::WeightSource;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Op, Shape};
use std::collections::HashMap;

const LN_EPS: f32 = 1e-5;
const MAX_TIMESCALE: f64 = 10_000.0;

struct EncoderBuilder<'a> {
    hir: &'a mut HirModule,
    params: &'a mut HashMap<String, Vec<f32>>,
    weights: &'a mut dyn WeightSource,
    f: DType,
}

impl EncoderBuilder<'_> {
    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    fn load_param(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        let id = self.hir.param(key, Shape::new(&shape, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    fn register_param(&mut self, key: &str, data: Vec<f32>, dims: &[usize]) -> HirNodeId {
        let id = self.hir.param(key, Shape::new(dims, self.f));
        self.params.insert(key.to_string(), data);
        id
    }

    /// `mm(x, Wᵀ) + b`, with `b` broadcast over the leading dims.
    fn linear(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: Option<&str>,
        out: usize,
    ) -> Result<HirNodeId> {
        let w = self.load_param(w_key, true)?;
        let mut y = self.g().mm(x, w);
        if let Some(bk) = b_key {
            let b = self.load_param(bk, false)?;
            let b3 = self.g().reshape_(b, vec![1, 1, out as i64]);
            y = self.g().add(y, b3);
        }
        Ok(y)
    }

    fn layer_norm(&mut self, x: HirNodeId, w_key: &str, b_key: &str) -> Result<HirNodeId> {
        let g = self.load_param(w_key, false)?;
        let b = self.load_param(b_key, false)?;
        Ok(self.g().ln(x, g, b, LN_EPS))
    }

    /// One Conv2d (stride 2, pad 1, k 3) + bias + GELU.
    fn conv(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: &str,
        num_chunks: usize,
        in_c: usize,
        out_c: usize,
        out_hw: [usize; 2],
    ) -> Result<HirNodeId> {
        let _ = in_c;
        let w = self.load_param(w_key, false)?;
        let out_shape = Shape::new(&[num_chunks, out_c, out_hw[0], out_hw[1]], self.f);
        let conv = self.g().conv2d(x, w, [3, 3], [2, 2], [1, 1], 1, out_shape);
        let b = self.load_param(b_key, false)?;
        let b4 = self.g().reshape_(b, vec![1, out_c as i64, 1, 1]);
        let biased = self.g().add(conv, b4);
        Ok(self.g().gelu(biased))
    }

    /// Block-diagonal windowed self-attention over the full contiguous
    /// `[1, seq, d]`. With `win_bias` (multi-window) one attention runs with an
    /// additive `[1, nh, seq, seq]` bias (0 within window, −∞ across) —
    /// equivalent to per-window full attention but without the `narrow`/`concat`
    /// path whose strided views crash the attention kernel.
    fn windowed_attention(
        &mut self,
        x: HirNodeId,
        layer: usize,
        cfg: &AudioEncoderConfig,
        seq: usize,
        win_bias: Option<HirNodeId>,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let nh = cfg.encoder_attention_heads;
        let hd = cfg.head_dim();
        let f = self.f;
        let p = |s: &str| AsrWeightPrefix::audio_layer(layer, s);

        let q = self.linear(
            x,
            &p("self_attn.q_proj.weight"),
            Some(&p("self_attn.q_proj.bias")),
            d,
        )?;
        let k = self.linear(
            x,
            &p("self_attn.k_proj.weight"),
            Some(&p("self_attn.k_proj.bias")),
            d,
        )?;
        let v = self.linear(
            x,
            &p("self_attn.v_proj.weight"),
            Some(&p("self_attn.v_proj.bias")),
            d,
        )?;

        let out_shape = Shape::new(&[1, seq, d], f);
        let attn = match win_bias {
            Some(bias) => self.g().add_node(
                Op::Attention {
                    num_heads: nh,
                    head_dim: hd,
                    mask_kind: MaskKind::Bias,
                    score_scale: None,
                    attn_logit_softcap: None,
                },
                vec![q, k, v, bias],
                out_shape,
            ),
            None => self
                .g()
                .attention_kind(q, k, v, nh, hd, MaskKind::None, out_shape),
        };

        self.linear(
            attn,
            &p("self_attn.out_proj.weight"),
            Some(&p("self_attn.out_proj.bias")),
            d,
        )
    }

    fn encoder_layer(
        &mut self,
        x: HirNodeId,
        layer: usize,
        cfg: &AudioEncoderConfig,
        seq: usize,
        win_bias: Option<HirNodeId>,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let ffn = cfg.encoder_ffn_dim;
        let p = |s: &str| AsrWeightPrefix::audio_layer(layer, s);

        let normed = self.layer_norm(
            x,
            &p("self_attn_layer_norm.weight"),
            &p("self_attn_layer_norm.bias"),
        )?;
        let attn = self.windowed_attention(normed, layer, cfg, seq, win_bias)?;
        let x = self.g().add(x, attn);

        let normed = self.layer_norm(
            x,
            &p("final_layer_norm.weight"),
            &p("final_layer_norm.bias"),
        )?;
        let h = self.linear(normed, &p("fc1.weight"), Some(&p("fc1.bias")), ffn)?;
        let h = self.g().gelu(h);
        let h = self.linear(h, &p("fc2.weight"), Some(&p("fc2.bias")), d)?;
        Ok(self.g().add(x, h))
    }
}

/// Sinusoidal position table `[positions, channels]` (Whisper-style).
fn sinusoid_table(positions: usize, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let log_inc = MAX_TIMESCALE.ln() / (half as f64 - 1.0);
    let mut out = vec![0f32; positions * channels];
    for pos in 0..positions {
        for i in 0..half {
            let scaled = pos as f64 * (-log_inc * i as f64).exp();
            out[pos * channels + i] = scaled.sin() as f32;
            out[pos * channels + half + i] = scaled.cos() as f32;
        }
    }
    out
}

/// Additive `[nh, seq, seq]` block-diagonal bias: 0 within a window, −∞ across.
fn window_bias_data(windows: &[usize], nh: usize) -> Vec<f32> {
    let t: usize = windows.iter().sum();
    let mut winof = vec![0usize; t];
    let mut pos = 0;
    for (wi, &w) in windows.iter().enumerate() {
        for _ in 0..w {
            if pos < t {
                winof[pos] = wi;
                pos += 1;
            }
        }
    }
    let neg = -1e9f32;
    let mut out = vec![0f32; nh * t * t];
    for i in 0..t {
        for j in 0..t {
            if winof[i] != winof[j] {
                for h in 0..nh {
                    out[h * t * t + i * t + j] = neg;
                }
            }
        }
    }
    out
}

/// Build the audio encoder graph.
///
/// Input  `"mel"`  : `[1, num_mel_bins, padded_frames]` where
///                   `padded_frames = num_chunks * max_chunk_len`.
/// Output `"audio_embeds"` : `[1, num_audio_tokens, output_dim]`.
pub fn build_encoder_built(
    cfg: &AudioEncoderConfig,
    weights: &mut WeightMap,
    geom: &AudioGeometry,
) -> Result<rlx_flow::BuiltModel> {
    let f = DType::F32;
    let d = cfg.d_model;
    let ds = cfg.downsample_hidden_size;
    let mels = cfg.num_mel_bins;
    let nc = geom.num_chunks;
    let mcl = geom.max_chunk_len;
    let t_pc = geom.t_pc;
    let fan = ds * geom.freq_pc;
    let padded = nc * mcl;

    let mut hir = HirModule::new("qwen3_asr_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let mel = hir.input("mel", Shape::new(&[1, mels, padded], f));

    let mut b = EncoderBuilder {
        hir: &mut hir,
        params: &mut params,
        weights: &mut WeightMapSource(weights),
        f,
    };

    // [1, mels, P] -> [mels, nc, mcl] -> [nc, mels, mcl] -> [nc, 1, mels, mcl]
    let x = b
        .g()
        .reshape_(mel, vec![mels as i64, nc as i64, mcl as i64]);
    let x = b.g().transpose_(x, vec![1, 0, 2]);
    let x = b
        .g()
        .reshape_(x, vec![nc as i64, 1, mels as i64, mcl as i64]);

    // Conv2d stem (each chunk independently downsampled in freq & time).
    let x = b.conv(
        x,
        AsrWeightPrefix::CONV2D1_W,
        AsrWeightPrefix::CONV2D1_B,
        nc,
        1,
        ds,
        [geom.conv_freq[0], geom.conv_time[0]],
    )?;
    let x = b.conv(
        x,
        AsrWeightPrefix::CONV2D2_W,
        AsrWeightPrefix::CONV2D2_B,
        nc,
        ds,
        ds,
        [geom.conv_freq[1], geom.conv_time[1]],
    )?;
    let x = b.conv(
        x,
        AsrWeightPrefix::CONV2D3_W,
        AsrWeightPrefix::CONV2D3_B,
        nc,
        ds,
        ds,
        [geom.conv_freq[2], geom.conv_time[2]],
    )?;

    // [nc, ds, freq_pc, t_pc] -> [nc, t_pc, ds, freq_pc] -> [nc, t_pc, ds*freq_pc]
    let x = b.g().transpose_(x, vec![0, 3, 1, 2]);
    let x = b.g().reshape_(x, vec![nc as i64, t_pc as i64, fan as i64]);
    // conv_out: Linear(fan -> d), no bias.
    let conv_out_w = b.load_param(AsrWeightPrefix::CONV_OUT_W, true)?;
    let x = b.g().mm(x, conv_out_w);

    // + sinusoidal positions (first t_pc rows), broadcast over chunks.
    let pos = b.register_param(
        "qwen3_asr.audio.sinusoid",
        sinusoid_table(t_pc, d),
        &[t_pc, d],
    );
    let pos = b.g().reshape_(pos, vec![1, t_pc as i64, d as i64]);
    let x = b.g().add(x, pos);

    // Flatten chunks, drop CNN padding -> [1, num_audio_tokens, d].
    let x = b.g().reshape_(x, vec![(nc * t_pc) as i64, d as i64]);
    let x = b.g().narrow_(x, 0, 0, geom.num_audio_tokens);
    let mut x = b
        .g()
        .reshape_(x, vec![1, geom.num_audio_tokens as i64, d as i64]);

    // Block-diagonal attention bias (only when >1 window); single-window keeps
    // the bias-free fast path.
    let seq = geom.num_audio_tokens;
    let nh = cfg.encoder_attention_heads;
    let win_bias = if geom.windows.len() > 1 {
        let data = window_bias_data(&geom.windows, nh);
        Some(b.register_param("qwen3_asr.audio.winmask", data, &[1, nh, seq, seq]))
    } else {
        None
    };

    for layer in 0..cfg.num_hidden_layers {
        x = b.encoder_layer(x, layer, cfg, seq, win_bias)?;
    }

    let x = b.layer_norm(x, AsrWeightPrefix::LN_POST_W, AsrWeightPrefix::LN_POST_B)?;
    let x = b.linear(
        x,
        AsrWeightPrefix::PROJ1_W,
        Some(AsrWeightPrefix::PROJ1_B),
        d,
    )?;
    let x = b.g().gelu(x);
    let x = b.linear(
        x,
        AsrWeightPrefix::PROJ2_W,
        Some(AsrWeightPrefix::PROJ2_B),
        cfg.output_dim,
    )?;

    hir.outputs = vec![x];
    rlx_core::flow_util::built_from_hir(hir, params)
}
