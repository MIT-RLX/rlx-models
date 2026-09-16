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

//! FireRedAudio understanding encoder HIR: Conv1d stem (via Conv2d) →
//! sinusoidal positions → windowed transformer → stride-2 adapter to the
//! backbone hidden size.

use crate::audio::{AudioGeometry, conv_len};
use crate::config::AudioEncoderConfig;
use crate::weights::AudioWeightPrefix;
use anyhow::{Result, ensure};
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

    /// Load Conv1d weight `[O, I, K]` as Conv2d `[O, I, 1, K]` (same layout).
    fn load_conv1d_as_conv2d(&mut self, key: &str) -> Result<(HirNodeId, usize, usize)> {
        let (data, shape) = self.weights.take(key, false)?;
        ensure!(
            shape.len() == 3 && shape[2] == 3,
            "expected Conv1d weight [O,I,3] for {key}, got {shape:?}"
        );
        let (o, i) = (shape[0], shape[1]);
        let id = self.hir.param(key, Shape::new(&[o, i, 1, 3], self.f));
        self.params.insert(key.to_string(), data);
        Ok((id, o, i))
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

    /// Conv1d via Conv2d: input `[N, C, 1, T]`, weight `[O, I, 1, 3]`,
    /// kernel `[1, 3]`, stride `[1, S]`, pad `[0, P]`, then bias + optional GELU.
    fn conv1d(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: &str,
        batch: usize,
        out_c: usize,
        out_t: usize,
        stride: usize,
        pad: usize,
        gelu: bool,
    ) -> Result<HirNodeId> {
        let (w, _, _) = self.load_conv1d_as_conv2d(w_key)?;
        let out_shape = Shape::new(&[batch, out_c, 1, out_t], self.f);
        let conv = self
            .g()
            .conv2d(x, w, [1, 3], [1, stride], [0, pad], 1, out_shape);
        let b = self.load_param(b_key, false)?;
        let b4 = self.g().reshape_(b, vec![1, out_c as i64, 1, 1]);
        let biased = self.g().add(conv, b4);
        if gelu {
            Ok(self.g().gelu(biased))
        } else {
            Ok(biased)
        }
    }

    /// Block-diagonal windowed self-attention over `[1, seq, d]`.
    /// `k_proj` has no bias (Qwen2.5-Omni audio attention).
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
        let p = |s: &str| AudioWeightPrefix::audio_layer(layer, s);

        let q = self.linear(
            x,
            &p("self_attn.q_proj.weight"),
            Some(&p("self_attn.q_proj.bias")),
            d,
        )?;
        let k = self.linear(x, &p("self_attn.k_proj.weight"), None, d)?;
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
                    v_head_dim: None,
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
        let p = |s: &str| AudioWeightPrefix::audio_layer(layer, s);

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

    /// Adapter: Conv1d stride-2 ×2 → LayerNorm → Linear → GELU → Linear.
    fn adapter(
        &mut self,
        x: HirNodeId,
        cfg: &AudioEncoderConfig,
        t_in: usize,
        t_out: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let od = cfg.output_dim;
        let t3 = conv_len(t_in);
        let t4 = conv_len(t3);
        ensure!(
            t4 == t_out,
            "adapter out length {t4} != expected {t_out} (t_in={t_in})"
        );

        // [1, T, D] → [1, D, 1, T]
        let x = self.g().reshape_(x, vec![1, t_in as i64, d as i64]);
        let x = self.g().transpose_(x, vec![0, 2, 1]);
        let x = self.g().reshape_(x, vec![1, d as i64, 1, t_in as i64]);

        let x = self.conv1d(
            x,
            AudioWeightPrefix::ADAPTER_CONV3_W,
            AudioWeightPrefix::ADAPTER_CONV3_B,
            1,
            d,
            t3,
            2,
            1,
            false,
        )?;
        let x = self.conv1d(
            x,
            AudioWeightPrefix::ADAPTER_CONV4_W,
            AudioWeightPrefix::ADAPTER_CONV4_B,
            1,
            d,
            t4,
            2,
            1,
            false,
        )?;

        // [1, D, 1, T'] → [1, T', D]
        let x = self.g().reshape_(x, vec![1, d as i64, t4 as i64]);
        let x = self.g().transpose_(x, vec![0, 2, 1]);

        let x = self.layer_norm(
            x,
            AudioWeightPrefix::ADAPTER_LN_W,
            AudioWeightPrefix::ADAPTER_LN_B,
        )?;
        let x = self.linear(
            x,
            AudioWeightPrefix::ADAPTER_LINEAR1_W,
            Some(AudioWeightPrefix::ADAPTER_LINEAR1_B),
            od,
        )?;
        let x = self.g().gelu(x);
        self.linear(
            x,
            AudioWeightPrefix::ADAPTER_LINEAR2_W,
            Some(AudioWeightPrefix::ADAPTER_LINEAR2_B),
            od,
        )
    }
}

/// Sinusoidal position table `[positions, channels]` (Whisper / Qwen2.5-Omni).
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

/// Additive `[1, nh, seq, seq]` block-diagonal bias: 0 within a window, −∞ across.
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
    let mels = cfg.num_mel_bins;
    let nc = geom.num_chunks;
    let mcl = geom.max_chunk_len;
    let t_pc = geom.t_pc;
    let padded = nc * mcl;

    ensure!(
        cfg.d_model.is_multiple_of(cfg.encoder_attention_heads),
        "d_model {} not divisible by heads {}",
        cfg.d_model,
        cfg.encoder_attention_heads
    );
    ensure!(
        t_pc <= cfg.max_source_positions,
        "post-conv2 length {t_pc} exceeds max_source_positions {}",
        cfg.max_source_positions
    );

    let mut hir = HirModule::new("fireredaudio_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let mel = hir.input("mel", Shape::new(&[1, mels, padded], f));

    let mut b = EncoderBuilder {
        hir: &mut hir,
        params: &mut params,
        weights: &mut WeightMapSource(weights),
        f,
    };

    // [1, mels, P] → [mels, nc, mcl] → [nc, mels, mcl] → [nc, mels, 1, mcl]
    let x = b
        .g()
        .reshape_(mel, vec![mels as i64, nc as i64, mcl as i64]);
    let x = b.g().transpose_(x, vec![1, 0, 2]);
    let x = b
        .g()
        .reshape_(x, vec![nc as i64, mels as i64, 1, mcl as i64]);

    // conv1: mel→d, stride 1; conv2: d→d, stride 2
    let x = b.conv1d(
        x,
        AudioWeightPrefix::CONV1_W,
        AudioWeightPrefix::CONV1_B,
        nc,
        d,
        mcl,
        1,
        1,
        true,
    )?;
    let x = b.conv1d(
        x,
        AudioWeightPrefix::CONV2_W,
        AudioWeightPrefix::CONV2_B,
        nc,
        d,
        t_pc,
        2,
        1,
        true,
    )?;

    // [nc, d, 1, t_pc] → [nc, t_pc, d]
    let x = b.g().reshape_(x, vec![nc as i64, d as i64, t_pc as i64]);
    let x = b.g().transpose_(x, vec![0, 2, 1]);

    // + sinusoidal positions (first t_pc rows of the max_source_positions table)
    let pos = b.register_param(
        "fireredaudio.audio.sinusoid",
        sinusoid_table(t_pc, d),
        &[t_pc, d],
    );
    let pos = b.g().reshape_(pos, vec![1, t_pc as i64, d as i64]);
    let x = b.g().add(x, pos);

    // Flatten chunks, drop CNN padding → [1, num_after_cnn, d]
    let x = b.g().reshape_(x, vec![(nc * t_pc) as i64, d as i64]);
    let x = b.g().narrow_(x, 0, 0, geom.num_after_cnn);
    let mut x = b
        .g()
        .reshape_(x, vec![1, geom.num_after_cnn as i64, d as i64]);

    let seq = geom.num_after_cnn;
    let nh = cfg.encoder_attention_heads;
    let win_bias = if geom.windows.len() > 1 {
        let data = window_bias_data(&geom.windows, nh);
        Some(b.register_param("fireredaudio.audio.winmask", data, &[1, nh, seq, seq]))
    } else {
        None
    };

    for layer in 0..cfg.encoder_layers {
        x = b.encoder_layer(x, layer, cfg, seq, win_bias)?;
    }

    let x = b.adapter(x, cfg, geom.num_after_cnn, geom.num_audio_tokens)?;

    hir.outputs = vec![x];
    rlx_core::flow_util::built_from_hir(hir, params)
}
