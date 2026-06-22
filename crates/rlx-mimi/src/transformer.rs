use crate::config::MimiConfig;
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct Attention {
    pub(crate) q_w: Array2<f32>,
    pub(crate) k_w: Array2<f32>,
    pub(crate) v_w: Array2<f32>,
    pub(crate) o_w: Array2<f32>,
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) scaling: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    pub(crate) fc1_w: Array2<f32>,
    pub(crate) fc2_w: Array2<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct LayerScale {
    pub(crate) scale: Array1<f32>,
}

impl LayerScale {
    fn apply(&self, x: &mut Array2<f32>) {
        let (t, c) = x.dim();
        for ti in 0..t {
            for ci in 0..c {
                x[[ti, ci]] *= self.scale[ci];
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TransformerLayer {
    pub(crate) input_norm_w: Array1<f32>,
    pub(crate) input_norm_b: Array1<f32>,
    pub(crate) post_norm_w: Array1<f32>,
    pub(crate) post_norm_b: Array1<f32>,
    pub(crate) attn: Attention,
    pub(crate) attn_scale: LayerScale,
    pub(crate) mlp: Mlp,
    pub(crate) mlp_scale: LayerScale,
}

pub struct MimiTransformer {
    pub(crate) layers: Vec<TransformerLayer>,
    pub(crate) inv_freq: Array1<f32>,
    pub(crate) sliding_window: usize,
    pub(crate) norm_eps: f32,
}

impl MimiTransformer {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let mut h = x.to_owned();
        let t = h.dim().0;
        let (cos, sin) = self.rope_cos_sin(t);
        let mask = build_sliding_causal_mask(t, self.sliding_window);
        for layer in &self.layers {
            h = forward_layer(layer, &h, &cos, &sin, &mask, self.norm_eps);
        }
        h
    }

    fn rope_cos_sin(&self, t: usize) -> (Array2<f32>, Array2<f32>) {
        let half = self.inv_freq.len();
        let head_dim = half * 2;
        let mut cos = Array2::<f32>::zeros((t, head_dim));
        let mut sin = Array2::<f32>::zeros((t, head_dim));
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
    let mut m = Array2::<f32>::from_elem((t, t), f32::NEG_INFINITY);
    for i in 0..t {
        let lo = i.saturating_sub(window - 1);
        for j in lo..=i {
            m[[i, j]] = 0.0;
        }
    }
    m
}

fn layer_norm(x: &Array2<f32>, w: &Array1<f32>, b: &Array1<f32>, eps: f32) -> Array2<f32> {
    let (t, c) = x.dim();
    let mut out = Array2::<f32>::zeros((t, c));
    let inv_c = 1.0 / c as f32;
    for ti in 0..t {
        let row = x.row(ti);
        let mean = row.iter().sum::<f32>() * inv_c;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() * inv_c;
        let scale = 1.0 / (var + eps).sqrt();
        for ci in 0..c {
            out[[ti, ci]] = (row[ci] - mean) * scale * w[ci] + b[ci];
        }
    }
    out
}

fn linear(x: ArrayView2<f32>, w: &Array2<f32>) -> Array2<f32> {
    x.dot(&w.t())
}

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

fn gelu_approx(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

fn gelu_inplace(x: &mut Array2<f32>) {
    for v in x.iter_mut() {
        *v = gelu_approx(*v);
    }
}

fn forward_layer(
    layer: &TransformerLayer,
    x: &Array2<f32>,
    cos: &Array2<f32>,
    sin: &Array2<f32>,
    mask: &Array2<f32>,
    eps: f32,
) -> Array2<f32> {
    let n = layer_norm(x, &layer.input_norm_w, &layer.input_norm_b, eps);
    let mut attn_out = attention(&layer.attn, &n, cos, sin, mask);
    layer.attn_scale.apply(&mut attn_out);
    let mut h = x.clone();
    h += &attn_out;
    let n2 = layer_norm(&h, &layer.post_norm_w, &layer.post_norm_b, eps);
    let mut mlp_out = linear(n2.view(), &layer.mlp.fc1_w);
    gelu_inplace(&mut mlp_out);
    let mlp_out = linear(mlp_out.view(), &layer.mlp.fc2_w);
    let mut mlp_out = mlp_out;
    layer.mlp_scale.apply(&mut mlp_out);
    h += &mlp_out;
    h
}

fn attention(
    a: &Attention,
    x: &Array2<f32>,
    cos: &Array2<f32>,
    sin: &Array2<f32>,
    mask: &Array2<f32>,
) -> Array2<f32> {
    let (t, _) = x.dim();
    let h = a.num_heads;
    let d = a.head_dim;
    let q = linear(x.view(), &a.q_w);
    let k = linear(x.view(), &a.k_w);
    let v = linear(x.view(), &a.v_w);
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
    let mut out = Array3::<f32>::zeros((h, t, d));
    for hi in 0..h {
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
        for i in 0..t {
            let maxv = attn
                .row(i)
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
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

pub fn build_transformer(
    cfg: &MimiConfig,
    prefix: &str,
    raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<MimiTransformer> {
    let pfx = format!("{prefix}.");
    let mut local: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (k, v) in raw {
        if let Some(rest) = k.strip_prefix(&pfx) {
            local.insert(rest.to_string(), v);
        }
    }
    let h = cfg.num_attention_heads;
    let d = cfg.head_dim;
    ensure!(h * d == cfg.hidden_size, "heads*head_dim != hidden_size");
    let scaling = 1.0f32 / (d as f32).sqrt();
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("layers.{i}");
        layers.push(TransformerLayer {
            input_norm_w: Array1::from_vec(take_tensor(
                &mut local,
                &format!("{lp}.input_layernorm.weight"),
                &[cfg.hidden_size],
            )?),
            input_norm_b: Array1::from_vec(take_tensor(
                &mut local,
                &format!("{lp}.input_layernorm.bias"),
                &[cfg.hidden_size],
            )?),
            post_norm_w: Array1::from_vec(take_tensor(
                &mut local,
                &format!("{lp}.post_attention_layernorm.weight"),
                &[cfg.hidden_size],
            )?),
            post_norm_b: Array1::from_vec(take_tensor(
                &mut local,
                &format!("{lp}.post_attention_layernorm.bias"),
                &[cfg.hidden_size],
            )?),
            attn: Attention {
                q_w: Array2::from_shape_vec(
                    (cfg.hidden_size, cfg.hidden_size),
                    take_tensor(
                        &mut local,
                        &format!("{lp}.self_attn.q_proj.weight"),
                        &[cfg.hidden_size, cfg.hidden_size],
                    )?,
                )?,
                k_w: Array2::from_shape_vec(
                    (cfg.hidden_size, cfg.hidden_size),
                    take_tensor(
                        &mut local,
                        &format!("{lp}.self_attn.k_proj.weight"),
                        &[cfg.hidden_size, cfg.hidden_size],
                    )?,
                )?,
                v_w: Array2::from_shape_vec(
                    (cfg.hidden_size, cfg.hidden_size),
                    take_tensor(
                        &mut local,
                        &format!("{lp}.self_attn.v_proj.weight"),
                        &[cfg.hidden_size, cfg.hidden_size],
                    )?,
                )?,
                o_w: Array2::from_shape_vec(
                    (cfg.hidden_size, cfg.hidden_size),
                    take_tensor(
                        &mut local,
                        &format!("{lp}.self_attn.o_proj.weight"),
                        &[cfg.hidden_size, cfg.hidden_size],
                    )?,
                )?,
                num_heads: h,
                head_dim: d,
                scaling,
            },
            attn_scale: LayerScale {
                scale: Array1::from_vec(take_tensor(
                    &mut local,
                    &format!("{lp}.self_attn_layer_scale.scale"),
                    &[cfg.hidden_size],
                )?),
            },
            mlp: Mlp {
                fc1_w: Array2::from_shape_vec(
                    (cfg.intermediate_size, cfg.hidden_size),
                    take_tensor(
                        &mut local,
                        &format!("{lp}.mlp.fc1.weight"),
                        &[cfg.intermediate_size, cfg.hidden_size],
                    )?,
                )?,
                fc2_w: Array2::from_shape_vec(
                    (cfg.hidden_size, cfg.intermediate_size),
                    take_tensor(
                        &mut local,
                        &format!("{lp}.mlp.fc2.weight"),
                        &[cfg.hidden_size, cfg.intermediate_size],
                    )?,
                )?,
            },
            mlp_scale: LayerScale {
                scale: Array1::from_vec(take_tensor(
                    &mut local,
                    &format!("{lp}.mlp_layer_scale.scale"),
                    &[cfg.hidden_size],
                )?),
            },
        });
    }
    let half = d / 2;
    let mut inv_freq = Array1::<f32>::zeros(half);
    for i in 0..half {
        inv_freq[i] = 1.0 / (cfg.rope_theta.powf(i as f64 / half as f64) as f32);
    }
    Ok(MimiTransformer {
        layers,
        inv_freq,
        sliding_window: cfg.sliding_window,
        norm_eps: cfg.norm_eps,
    })
}
