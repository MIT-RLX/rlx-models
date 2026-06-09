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

//! Mimi encoder transformer (`encoder.encoder_transformer.*`).
//!
//! 8-layer Mimi-style transformer over the SEANet conv output:
//!   LayerNorm → MHA(RoPE, sliding-window causal) → LayerScale → +residual
//!   LayerNorm → fc1 → GELU → fc2 → LayerScale → +residual
//!
//! Differs from `decode.rs::PreTransformer`:
//!   - LayerNorm (with bias) instead of RMSNorm
//!   - GELU MLP (no gate, fc1 → GELU → fc2) instead of SwiGLU
//!   - num_heads == num_kv_heads (full MHA, not GQA)
//!   - Standard Llama-style rotate-half RoPE (full head-dim rotated)
//!
//! Causal mask with sliding window `W` = 250; for short inputs (T < W) this is a
//! pure lower-triangular causal mask.

use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::collections::HashMap;

const TF_PREFIX: &str = "encoder.encoder_transformer.";

#[derive(Debug, Clone)]
pub struct EncoderTransformerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub sliding_window: usize,
    pub norm_eps: f32,
    pub rope_theta: f64,
    pub layer_scale_initial_scale: f32,
}

impl EncoderTransformerConfig {
    pub fn from_speech_tokenizer_dir(dir: &std::path::Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let text = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let enc = v
            .get("encoder_config")
            .context("missing encoder_config in speech_tokenizer config.json")?;
        let usize_f = |k: &str| -> Result<usize> {
            enc.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("encoder_config.{k}"))
        };
        let f32_f = |k: &str, default: f32| -> f32 {
            enc.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(default)
        };
        let f64_f = |k: &str, default: f64| -> f64 {
            enc.get(k).and_then(|x| x.as_f64()).unwrap_or(default)
        };
        Ok(Self {
            hidden_size: usize_f("hidden_size")?,
            num_hidden_layers: usize_f("num_hidden_layers")?,
            num_attention_heads: usize_f("num_attention_heads")?,
            num_key_value_heads: usize_f("num_key_value_heads")?,
            head_dim: usize_f("head_dim")?,
            intermediate_size: usize_f("intermediate_size")?,
            sliding_window: usize_f("sliding_window")?,
            norm_eps: f32_f("norm_eps", 1e-5),
            rope_theta: f64_f("rope_theta", 10_000.0),
            layer_scale_initial_scale: f32_f("layer_scale_initial_scale", 0.01),
        })
    }
}

// -----------------------------------------------------------------------------
// Ops.

/// Standard LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias.
/// Input is `[T, C]`; weight/bias are `[C]`.
fn layer_norm(x: &Array2<f32>, w: &Array1<f32>, b: &Array1<f32>, eps: f32) -> Array2<f32> {
    let (t, c) = x.dim();
    let mut out = Array2::<f32>::zeros((t, c));
    let inv_c = 1.0 / c as f32;
    for ti in 0..t {
        let row = x.row(ti);
        let mut sum = 0f32;
        for ci in 0..c {
            sum += row[ci];
        }
        let mean = sum * inv_c;
        let mut var = 0f32;
        for ci in 0..c {
            let d = row[ci] - mean;
            var += d * d;
        }
        var *= inv_c;
        let scale = 1.0 / (var + eps).sqrt();
        for ci in 0..c {
            out[[ti, ci]] = ((row[ci] - mean) * scale) * w[ci] + b[ci];
        }
    }
    out
}

/// Linear y = x @ W^T (W stored as `[out_features, in_features]`).
fn linear(x: ArrayView2<f32>, w: &Array2<f32>) -> Array2<f32> {
    // x: [T, in], w: [out, in]. Compute x.dot(&w.t()) → [T, out].
    x.dot(&w.t())
}

/// Abramowitz & Stegun 7.1.26 — max abs error ~1.5e-7.
#[inline]
fn erf_approx(x: f32) -> f32 {
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_72_f32;
    let a3 = 1.421_413_8_f32;
    let a4 = -1.453_152_1_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.3275911_f32;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[inline]
fn gelu_approx(x: f32) -> f32 {
    // PyTorch `nn.GELU` default = exact GELU (erf-based).
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

fn gelu_inplace(x: &mut Array2<f32>) {
    for v in x.iter_mut() {
        *v = gelu_approx(*v);
    }
}

// -----------------------------------------------------------------------------
// Layers.

#[derive(Debug, Clone)]
struct Attention {
    q_w: Array2<f32>,
    k_w: Array2<f32>,
    v_w: Array2<f32>,
    o_w: Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    scaling: f32,
}

#[derive(Debug, Clone)]
struct Mlp {
    fc1_w: Array2<f32>,
    fc2_w: Array2<f32>,
}

#[derive(Debug, Clone)]
struct LayerScale {
    scale: Array1<f32>,
}

impl LayerScale {
    fn apply(&self, x: &mut Array2<f32>) {
        let (t, c) = x.dim();
        debug_assert_eq!(c, self.scale.len());
        for ti in 0..t {
            for ci in 0..c {
                x[[ti, ci]] *= self.scale[ci];
            }
        }
    }
}

#[derive(Debug, Clone)]
struct TransformerLayer {
    input_norm_w: Array1<f32>,
    input_norm_b: Array1<f32>,
    post_norm_w: Array1<f32>,
    post_norm_b: Array1<f32>,
    attn: Attention,
    attn_scale: LayerScale,
    mlp: Mlp,
    mlp_scale: LayerScale,
}

#[derive(Debug, Clone)]
pub struct MimiEncoderTransformer {
    pub cfg: EncoderTransformerConfig,
    layers: Vec<TransformerLayer>,
    /// Pre-computed inv_freq for RoPE, shape `[head_dim / 2]`.
    inv_freq: Array1<f32>,
}

impl MimiEncoderTransformer {
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Forward. Input is `[T, hidden]`, output is `[T, hidden]`.
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let mut h = x.to_owned();
        let t = h.dim().0;
        let (cos, sin) = self.rope_cos_sin(t);
        let mask = build_sliding_causal_mask(t, self.cfg.sliding_window);
        for layer in &self.layers {
            h = forward_layer(layer, &h, &cos, &sin, &mask, self.cfg.norm_eps);
        }
        h
    }

    /// Forward with per-layer intermediates for parity testing.
    pub fn forward_with_intermediates(
        &self,
        x: ArrayView2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let mut h = x.to_owned();
        let t = h.dim().0;
        let (cos, sin) = self.rope_cos_sin(t);
        let mask = build_sliding_causal_mask(t, self.cfg.sliding_window);
        let mut outs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            h = forward_layer(layer, &h, &cos, &sin, &mask, self.cfg.norm_eps);
            outs.push(h.clone());
        }
        (h, outs)
    }

    /// Cos/sin tables for RoPE, shape `[T, head_dim]` each. Per Mimi RoPE
    /// (Llama-style): `emb = cat(freqs, freqs, -1)`.
    fn rope_cos_sin(&self, t: usize) -> (Array2<f32>, Array2<f32>) {
        let half = self.cfg.head_dim / 2;
        let mut cos = Array2::<f32>::zeros((t, self.cfg.head_dim));
        let mut sin = Array2::<f32>::zeros((t, self.cfg.head_dim));
        for ti in 0..t {
            for hi in 0..half {
                let f = ti as f32 * self.inv_freq[hi];
                let c = f.cos();
                let s = f.sin();
                cos[[ti, hi]] = c;
                cos[[ti, hi + half]] = c;
                sin[[ti, hi]] = s;
                sin[[ti, hi + half]] = s;
            }
        }
        (cos, sin)
    }
}

fn build_sliding_causal_mask(t: usize, window: usize) -> Array2<f32> {
    // mask[i, j] = 0 if j ≤ i AND (i - j) < window, else -inf.
    let mut m = Array2::<f32>::from_elem((t, t), f32::NEG_INFINITY);
    for i in 0..t {
        let lo = i.saturating_sub(window - 1);
        for j in lo..=i {
            m[[i, j]] = 0.0;
        }
    }
    m
}

fn forward_layer(
    layer: &TransformerLayer,
    x: &Array2<f32>,
    cos: &Array2<f32>,
    sin: &Array2<f32>,
    mask: &Array2<f32>,
    eps: f32,
) -> Array2<f32> {
    // ---- attention block ----
    let n = layer_norm(x, &layer.input_norm_w, &layer.input_norm_b, eps);
    let mut attn_out = attention(&layer.attn, &n, cos, sin, mask);
    layer.attn_scale.apply(&mut attn_out);
    let mut h = x.clone();
    h += &attn_out;
    // ---- MLP block ----
    let n2 = layer_norm(&h, &layer.post_norm_w, &layer.post_norm_b, eps);
    let mut mlp_out = mlp_forward(&layer.mlp, &n2);
    layer.mlp_scale.apply(&mut mlp_out);
    h += &mlp_out;
    h
}

fn mlp_forward(mlp: &Mlp, x: &Array2<f32>) -> Array2<f32> {
    let mut h = linear(x.view(), &mlp.fc1_w);
    gelu_inplace(&mut h);
    linear(h.view(), &mlp.fc2_w)
}

/// Multi-head attention. `x` is `[T, hidden]`; cos/sin are `[T, head_dim]`;
/// `mask` is `[T, T]` with 0 / -inf entries.
fn attention(
    a: &Attention,
    x: &Array2<f32>,
    cos: &Array2<f32>,
    sin: &Array2<f32>,
    mask: &Array2<f32>,
) -> Array2<f32> {
    let (t, _hidden) = x.dim();
    let h = a.num_heads;
    let d = a.head_dim;

    // Projections: [T, h*d].
    let q = linear(x.view(), &a.q_w);
    let k = linear(x.view(), &a.k_w);
    let v = linear(x.view(), &a.v_w);

    // Reshape into [H, T, D] and apply RoPE to Q and K.
    let mut q3 = Array3::<f32>::zeros((h, t, d));
    let mut k3 = Array3::<f32>::zeros((h, t, d));
    let mut v3 = Array3::<f32>::zeros((h, t, d));
    for ti in 0..t {
        for hi in 0..h {
            for di in 0..d {
                q3[[hi, ti, di]] = q[[ti, hi * d + di]];
                k3[[hi, ti, di]] = k[[ti, hi * d + di]];
                v3[[hi, ti, di]] = v[[ti, hi * d + di]];
            }
        }
    }
    apply_rope_inplace(&mut q3, cos, sin);
    apply_rope_inplace(&mut k3, cos, sin);

    // Per-head attention.
    let mut out = Array3::<f32>::zeros((h, t, d));
    for hi in 0..h {
        // attn = (Q @ K^T) * scale + mask. Shape [T, T].
        let mut attn = Array2::<f32>::zeros((t, t));
        for i in 0..t {
            for j in 0..t {
                let mut s = 0f32;
                for di in 0..d {
                    s += q3[[hi, i, di]] * k3[[hi, j, di]];
                }
                attn[[i, j]] = s * a.scaling + mask[[i, j]];
            }
        }
        // Softmax over j per row.
        for i in 0..t {
            let mut maxv = f32::NEG_INFINITY;
            for j in 0..t {
                if attn[[i, j]] > maxv {
                    maxv = attn[[i, j]];
                }
            }
            let mut sum = 0f32;
            for j in 0..t {
                let v = (attn[[i, j]] - maxv).exp();
                attn[[i, j]] = v;
                sum += v;
            }
            let inv = 1.0 / sum.max(1e-12);
            for j in 0..t {
                attn[[i, j]] *= inv;
            }
        }
        // out[h, i, :] = sum_j attn[i, j] * V[h, j, :]
        for i in 0..t {
            for di in 0..d {
                let mut acc = 0f32;
                for j in 0..t {
                    acc += attn[[i, j]] * v3[[hi, j, di]];
                }
                out[[hi, i, di]] = acc;
            }
        }
    }

    // Reshape [H, T, D] → [T, H*D], apply o_proj.
    let mut o = Array2::<f32>::zeros((t, h * d));
    for ti in 0..t {
        for hi in 0..h {
            for di in 0..d {
                o[[ti, hi * d + di]] = out[[hi, ti, di]];
            }
        }
    }
    linear(o.view(), &a.o_w)
}

fn apply_rope_inplace(x: &mut Array3<f32>, cos: &Array2<f32>, sin: &Array2<f32>) {
    let (h, t, d) = x.dim();
    let half = d / 2;
    for hi in 0..h {
        for ti in 0..t {
            for di in 0..half {
                let x1 = x[[hi, ti, di]];
                let x2 = x[[hi, ti, di + half]];
                let c1 = cos[[ti, di]];
                let s1 = sin[[ti, di]];
                let c2 = cos[[ti, di + half]];
                let s2 = sin[[ti, di + half]];
                x[[hi, ti, di]] = x1 * c1 + (-x2) * s1;
                x[[hi, ti, di + half]] = x2 * c2 + x1 * s2;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Weight loader.

fn take_tensor(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    let (data, shape) = raw
        .remove(name)
        .with_context(|| format!("missing tensor {name}"))?;
    ensure!(
        shape == expected_shape,
        "{name} shape {:?} != {:?}",
        shape,
        expected_shape
    );
    Ok(data)
}

fn arr1(data: Vec<f32>) -> Array1<f32> {
    Array1::from_vec(data)
}

fn arr2(data: Vec<f32>, rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_vec((rows, cols), data).expect("arr2 reshape")
}

pub fn build_encoder_transformer(
    cfg: &EncoderTransformerConfig,
    raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<MimiEncoderTransformer> {
    let mut local: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        if let Some(rest) = k.strip_prefix(TF_PREFIX) {
            local.insert(rest.to_string(), v);
        }
    }
    let h = cfg.num_attention_heads;
    let d = cfg.head_dim;
    ensure!(
        h * d == cfg.hidden_size,
        "num_heads * head_dim ({}*{}) != hidden_size ({})",
        h,
        d,
        cfg.hidden_size
    );
    ensure!(
        cfg.num_key_value_heads == h,
        "encoder transformer assumes MHA (num_kv_heads == num_heads), got {} vs {}",
        cfg.num_key_value_heads,
        h
    );
    let scaling = 1.0f32 / (d as f32).sqrt();

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let pfx = format!("layers.{i}");
        let q_w = arr2(
            take_tensor(
                &mut local,
                &format!("{pfx}.self_attn.q_proj.weight"),
                &[cfg.hidden_size, cfg.hidden_size],
            )?,
            cfg.hidden_size,
            cfg.hidden_size,
        );
        let k_w = arr2(
            take_tensor(
                &mut local,
                &format!("{pfx}.self_attn.k_proj.weight"),
                &[cfg.hidden_size, cfg.hidden_size],
            )?,
            cfg.hidden_size,
            cfg.hidden_size,
        );
        let v_w = arr2(
            take_tensor(
                &mut local,
                &format!("{pfx}.self_attn.v_proj.weight"),
                &[cfg.hidden_size, cfg.hidden_size],
            )?,
            cfg.hidden_size,
            cfg.hidden_size,
        );
        let o_w = arr2(
            take_tensor(
                &mut local,
                &format!("{pfx}.self_attn.o_proj.weight"),
                &[cfg.hidden_size, cfg.hidden_size],
            )?,
            cfg.hidden_size,
            cfg.hidden_size,
        );
        let in_w = arr1(take_tensor(
            &mut local,
            &format!("{pfx}.input_layernorm.weight"),
            &[cfg.hidden_size],
        )?);
        let in_b = arr1(take_tensor(
            &mut local,
            &format!("{pfx}.input_layernorm.bias"),
            &[cfg.hidden_size],
        )?);
        let post_w = arr1(take_tensor(
            &mut local,
            &format!("{pfx}.post_attention_layernorm.weight"),
            &[cfg.hidden_size],
        )?);
        let post_b = arr1(take_tensor(
            &mut local,
            &format!("{pfx}.post_attention_layernorm.bias"),
            &[cfg.hidden_size],
        )?);
        let attn_scale = arr1(take_tensor(
            &mut local,
            &format!("{pfx}.self_attn_layer_scale.scale"),
            &[cfg.hidden_size],
        )?);
        let mlp_scale = arr1(take_tensor(
            &mut local,
            &format!("{pfx}.mlp_layer_scale.scale"),
            &[cfg.hidden_size],
        )?);
        let fc1_w = arr2(
            take_tensor(
                &mut local,
                &format!("{pfx}.mlp.fc1.weight"),
                &[cfg.intermediate_size, cfg.hidden_size],
            )?,
            cfg.intermediate_size,
            cfg.hidden_size,
        );
        let fc2_w = arr2(
            take_tensor(
                &mut local,
                &format!("{pfx}.mlp.fc2.weight"),
                &[cfg.hidden_size, cfg.intermediate_size],
            )?,
            cfg.hidden_size,
            cfg.intermediate_size,
        );
        layers.push(TransformerLayer {
            input_norm_w: in_w,
            input_norm_b: in_b,
            post_norm_w: post_w,
            post_norm_b: post_b,
            attn: Attention {
                q_w,
                k_w,
                v_w,
                o_w,
                num_heads: h,
                head_dim: d,
                scaling,
            },
            attn_scale: LayerScale { scale: attn_scale },
            mlp: Mlp { fc1_w, fc2_w },
            mlp_scale: LayerScale { scale: mlp_scale },
        });
    }

    if !local.is_empty() {
        let leftover: Vec<&String> = local.keys().take(5).collect();
        bail!(
            "{} unused encoder transformer tensors (first: {:?})",
            local.len(),
            leftover
        );
    }

    // RoPE inv_freq.
    let half = d / 2;
    let mut inv_freq = Array1::<f32>::zeros(half);
    for hi in 0..half {
        let exp = (2 * hi) as f64 / d as f64;
        inv_freq[hi] = (1.0 / cfg.rope_theta.powf(exp)) as f32;
    }

    Ok(MimiEncoderTransformer {
        cfg: cfg.clone(),
        layers,
        inv_freq,
    })
}

/// Open from a `Qwen3-TTS-Base/speech_tokenizer/` directory.
pub fn open_encoder_transformer(tok_dir: &std::path::Path) -> Result<MimiEncoderTransformer> {
    let cfg = EncoderTransformerConfig::from_speech_tokenizer_dir(tok_dir)?;
    let ckpt = rlx_core::safetensors_checkpoint::SafetensorsCheckpoint::open(tok_dir)?;
    let want: std::collections::HashSet<String> = ckpt
        .keys()
        .filter(|k| k.starts_with(TF_PREFIX))
        .map(str::to_string)
        .collect();
    ensure!(!want.is_empty(), "no encoder_transformer.* tensors found");
    let mut wm = ckpt.load_selected(&want)?;
    let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::with_capacity(want.len());
    for k in want.iter() {
        let (data, shape) = wm.take(k)?;
        raw.insert(k.clone(), (data, shape));
    }
    build_encoder_transformer(&cfg, raw)
}
