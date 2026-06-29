// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Native AIF probe — Eq. 2 prefill dynamics from LM weights + prefill hidden,
// without Python or HF `output_attentions`.

use crate::aif::{AifProbe, VisionKeySpan};
use crate::config::Qwen25VlLmConfig;
use crate::mrope::mrope_prefill_feeds;
use anyhow::{Context, Result, ensure};
use rlx_qwen3::Qwen3Config;
use std::collections::HashMap;

/// Inputs for native prefill probe (Eq. 2).
pub struct NativePrefillProbeInputs<'a> {
    pub cfg: &'a Qwen25VlLmConfig,
    pub weights: &'a HashMap<String, (Vec<f32>, Vec<usize>)>,
    /// Row-major `[seq, hidden_size]` prefill hidden (before LM stack).
    pub hidden: &'a [f32],
    pub mrope_sections: &'a [[usize; 4]],
    pub vision: VisionKeySpan,
    pub seq: usize,
}

/// Eq. 2 — image-to-text dynamics `[vision_idx][layer]`.
pub fn compute_dynamics_eq2_prefill(inp: &NativePrefillProbeInputs<'_>) -> Result<Vec<Vec<f32>>> {
    let lm = &inp.cfg.lm;
    let h = lm.hidden_size;
    ensure!(
        inp.hidden.len() == inp.seq * h,
        "hidden len {} != seq {} * hidden {}",
        inp.hidden.len(),
        inp.seq,
        h
    );
    ensure!(inp.vision.len() > 0, "native probe requires vision tokens");
    ensure!(
        inp.vision.end <= inp.seq,
        "vision span {}..{} exceeds seq {}",
        inp.vision.start,
        inp.vision.end,
        inp.seq
    );

    let head_half = inp.cfg.head_half();
    let (rope_cos, rope_sin) =
        mrope_prefill_feeds(inp.cfg, inp.seq, Some(inp.mrope_sections), head_half);

    let n_vis = inp.vision.len();
    let n_layers = lm.num_hidden_layers;
    let mut dynamics = vec![vec![0f32; n_layers]; n_vis];

    let mut hidden = inp.hidden.to_vec();
    for layer in 0..n_layers {
        let layer_d = forward_layer_probe_qk(
            lm,
            inp.weights,
            layer,
            &mut hidden,
            inp.seq,
            &rope_cos,
            &rope_sin,
            head_half,
        )?;
        extract_eq2_row(&layer_d, inp.vision, lm, &mut dynamics, layer);
    }
    Ok(dynamics)
}

/// Build [`AifProbe`] from native prefill dynamics.
pub fn native_prefill_probe(inp: &NativePrefillProbeInputs<'_>) -> Result<AifProbe> {
    Ok(AifProbe::build(compute_dynamics_eq2_prefill(inp)?))
}

struct LayerQk {
    q: Vec<f32>,
    k: Vec<f32>,
}

fn forward_layer_probe_qk(
    lm: &Qwen3Config,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    layer: usize,
    hidden: &mut [f32],
    seq: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
    head_half: usize,
) -> Result<LayerQk> {
    let lp = format!("model.layers.{layer}");
    let h = lm.hidden_size;
    let nh = lm.num_attention_heads;
    let nkv = lm.num_key_value_heads;
    let dh = lm.head_dim;
    let q_dim = lm.q_proj_dim();
    let kv_dim = lm.kv_proj_dim();
    let _group = nh / nkv;
    let scale = (dh as f32).sqrt().recip();

    let in_ln = tensor_1d(weights, &format!("{lp}.input_layernorm.weight"))?;
    let q_w = tensor(weights, &format!("{lp}.self_attn.q_proj.weight"))?;
    let k_w = tensor(weights, &format!("{lp}.self_attn.k_proj.weight"))?;
    let v_w = tensor(weights, &format!("{lp}.self_attn.v_proj.weight"))?;
    let o_w = tensor(weights, &format!("{lp}.self_attn.o_proj.weight"))?;
    let post_ln = tensor_1d(weights, &format!("{lp}.post_attention_layernorm.weight"))?;
    let gate_w = tensor(weights, &format!("{lp}.mlp.gate_proj.weight"))?;
    let up_w = tensor(weights, &format!("{lp}.mlp.up_proj.weight"))?;
    let down_w = tensor(weights, &format!("{lp}.mlp.down_proj.weight"))?;

    let q_bias = lm
        .attention_bias
        .then(|| tensor_1d(weights, &format!("{lp}.self_attn.q_proj.bias")))
        .transpose()
        .ok()
        .flatten();
    let k_bias = lm
        .attention_bias
        .then(|| tensor_1d(weights, &format!("{lp}.self_attn.k_proj.bias")))
        .transpose()
        .ok()
        .flatten();
    let v_bias = lm
        .attention_bias
        .then(|| tensor_1d(weights, &format!("{lp}.self_attn.v_proj.bias")))
        .transpose()
        .ok()
        .flatten();

    let q_norm = lm
        .qk_norm
        .then(|| tensor_1d(weights, &format!("{lp}.self_attn.q_norm.weight")))
        .transpose()
        .ok()
        .flatten();
    let k_norm = lm
        .qk_norm
        .then(|| tensor_1d(weights, &format!("{lp}.self_attn.k_norm.weight")))
        .transpose()
        .ok()
        .flatten();

    let mut q = vec![0f32; seq * q_dim];
    let mut k = vec![0f32; seq * kv_dim];
    let mut v = vec![0f32; seq * kv_dim];

    for t in 0..seq {
        let row = &hidden[t * h..(t + 1) * h];
        let normed = rms_norm_row(row, in_ln, lm.rms_norm_eps as f32);
        let q_row = linear_row(&normed, q_w.0, q_w.1, q_bias.as_deref())?;
        let k_row = linear_row(&normed, k_w.0, k_w.1, k_bias.as_deref())?;
        let v_row = linear_row(&normed, v_w.0, v_w.1, v_bias.as_deref())?;
        q[t * q_dim..(t + 1) * q_dim].copy_from_slice(&q_row);
        k[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&k_row);
        v[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&v_row);
    }

    if let (Some(qng), Some(kng)) = (q_norm, k_norm) {
        per_head_rms_rows(&mut q, seq, nh, dh, qng, lm.rms_norm_eps as f32);
        per_head_rms_rows(&mut k, seq, nkv, dh, kng, lm.rms_norm_eps as f32);
    }

    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    for t in 0..seq {
        let c = &rope_cos[t * head_half..(t + 1) * head_half];
        let s = &rope_sin[t * head_half..(t + 1) * head_half];
        apply_rope_neox(&mut q_rope[t * q_dim..(t + 1) * q_dim], nh, dh, c, s);
        apply_rope_neox(&mut k_rope[t * kv_dim..(t + 1) * kv_dim], nkv, dh, c, s);
    }

    let k_rep = repeat_kv_heads(&k_rope, seq, nkv, nh, dh);
    let v_rep = repeat_kv_heads(&v, seq, nkv, nh, dh);

    let mut attn_out = vec![0f32; seq * q_dim];
    for t in 0..seq {
        causal_attention_row(
            &q_rope[t * q_dim..(t + 1) * q_dim],
            &k_rep,
            &v_rep,
            nh,
            dh,
            t,
            scale,
            &mut attn_out[t * q_dim..(t + 1) * q_dim],
        );
    }

    for t in 0..seq {
        let base = t * h;
        let skip = hidden[base..base + h].to_vec();
        let proj = linear_row(&attn_out[t * q_dim..(t + 1) * q_dim], o_w.0, o_w.1, None)?;
        for i in 0..h {
            hidden[base + i] = skip[i] + proj[i];
        }
    }

    for t in 0..seq {
        let base = t * h;
        let row = hidden[base..base + h].to_vec();
        let normed = rms_norm_row(&row, post_ln, lm.rms_norm_eps as f32);
        let gate = linear_row(&normed, gate_w.0, gate_w.1, None)?;
        let up = linear_row(&normed, up_w.0, up_w.1, None)?;
        let mut swiglu = vec![0f32; gate.len()];
        for i in 0..gate.len() {
            swiglu[i] = silu(gate[i]) * up[i];
        }
        let ffn = linear_row(&swiglu, down_w.0, down_w.1, None)?;
        for i in 0..h {
            hidden[base + i] = row[i] + ffn[i];
        }
    }

    Ok(LayerQk {
        q: q_rope,
        k: k_rep,
    })
}

fn extract_eq2_row(
    qk: &LayerQk,
    vision: VisionKeySpan,
    lm: &Qwen3Config,
    dynamics: &mut [Vec<f32>],
    layer: usize,
) {
    let nh = lm.num_attention_heads;
    let dh = lm.head_dim;
    let q_dim = lm.q_proj_dim();
    let scale = (dh as f32).sqrt().recip();
    let vision_range: std::collections::HashSet<usize> = (vision.start..vision.end).collect();

    for (vi, qi) in (vision.start..vision.end).enumerate() {
        let mut best = 0f32;
        for head in 0..nh {
            let q_off = qi * q_dim + head * dh;
            let q_head = &qk.q[q_off..q_off + dh];
            let mut scores = vec![0f32; qi + 1];
            for j in 0..=qi {
                let k_off = j * q_dim + head * dh;
                let k_head = &qk.k[k_off..k_off + dh];
                scores[j] = dot(q_head, k_head) * scale;
            }
            apply_causal_mask(&mut scores);
            softmax_row(&mut scores);
            for (j, &prob) in scores.iter().enumerate().take(qi + 1) {
                if !vision_range.contains(&j) {
                    best = best.max(prob);
                }
            }
        }
        dynamics[vi][layer] = best;
    }
}

fn tensor_1d<'a>(
    weights: &'a HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
) -> Result<&'a [f32]> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    ensure!(
        shape.len() == 1,
        "expected rank-1 weight {key}, got rank {}",
        shape.len()
    );
    Ok(data.as_slice())
}

fn tensor<'a>(
    weights: &'a HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
) -> Result<(&'a [f32], [usize; 2])> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    ensure!(
        shape.len() == 2,
        "expected rank-2 weight {key}, got rank {}",
        shape.len()
    );
    Ok((data.as_slice(), [shape[0], shape[1]]))
}

fn rms_norm_row(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps;
    let inv = ss.sqrt().recip();
    x.iter()
        .zip(gamma.iter())
        .map(|(v, g)| v * inv * g)
        .collect()
}

fn per_head_rms_rows(x: &mut [f32], seq: usize, heads: usize, dh: usize, gamma: &[f32], eps: f32) {
    let row = heads * dh;
    for t in 0..seq {
        for h in 0..heads {
            let off = t * row + h * dh;
            let normed = rms_norm_row(&x[off..off + dh], gamma, eps);
            x[off..off + dh].copy_from_slice(&normed);
        }
    }
}

fn linear_row(x: &[f32], w: &[f32], shape: [usize; 2], bias: Option<&[f32]>) -> Result<Vec<f32>> {
    let (out, inp) = (shape[0], shape[1]);
    ensure!(x.len() == inp, "linear input {inp} != x.len {}", x.len());
    ensure!(
        w.len() == out * inp,
        "weight len {} != out*in {out}*{inp}",
        w.len()
    );
    let mut y = vec![0f32; out];
    for o in 0..out {
        let mut acc = 0f32;
        let row = o * inp;
        for i in 0..inp {
            acc += w[row + i] * x[i];
        }
        y[o] = acc;
    }
    if let Some(b) = bias {
        ensure!(b.len() == out, "bias len {} != out {out}", b.len());
        for o in 0..out {
            y[o] += b[o];
        }
    }
    Ok(y)
}

fn apply_rope_neox(x: &mut [f32], heads: usize, dh: usize, cos: &[f32], sin: &[f32]) {
    let half = dh / 2;
    for h in 0..heads {
        let base = h * dh;
        for i in 0..half.min(cos.len()) {
            let x0 = x[base + i];
            let x1 = x[base + half + i];
            x[base + i] = x0 * cos[i] - x1 * sin[i];
            x[base + half + i] = x0 * sin[i] + x1 * cos[i];
        }
    }
}

fn repeat_kv_heads(k: &[f32], seq: usize, nkv: usize, nh: usize, dh: usize) -> Vec<f32> {
    let kv_row = nkv * dh;
    let q_row = nh * dh;
    let group = nh / nkv;
    let mut out = vec![0f32; seq * q_row];
    for t in 0..seq {
        for h in 0..nh {
            let src = t * kv_row + (h / group) * dh;
            let dst = t * q_row + h * dh;
            out[dst..dst + dh].copy_from_slice(&k[src..src + dh]);
        }
    }
    out
}

fn causal_attention_row(
    q_row: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    nh: usize,
    dh: usize,
    qi: usize,
    scale: f32,
    out: &mut [f32],
) {
    let q_dim = nh * dh;
    for head in 0..nh {
        let q_off = head * dh;
        let q_head = &q_row[q_off..q_off + dh];
        let mut scores = vec![0f32; qi + 1];
        for j in 0..=qi {
            let k_off = j * q_dim + head * dh;
            scores[j] = dot(q_head, &k_all[k_off..k_off + dh]) * scale;
        }
        apply_causal_mask(&mut scores);
        softmax_row(&mut scores);
        let o_off = head * dh;
        for d in 0..dh {
            out[o_off + d] = 0.0;
        }
        for j in 0..=qi {
            let v_off = j * q_dim + head * dh;
            let prob = scores[j];
            for d in 0..dh {
                out[o_off + d] += prob * v_all[v_off + d];
            }
        }
    }
}

fn apply_causal_mask(scores: &mut [f32]) {
    // Already restricted to j <= qi; upper triangle not present.
    let _ = scores;
}

fn softmax_row(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Dynamics from exported prefill Q/K side outputs (one layer pair per slice).
pub fn dynamics_from_graph_qk_layers(
    q_layers: &[&[f32]],
    k_layers: &[&[f32]],
    vision: VisionKeySpan,
    _seq: usize,
    lm: &Qwen3Config,
) -> Result<Vec<Vec<f32>>> {
    ensure!(q_layers.len() == k_layers.len(), "q/k layer count mismatch");
    let n_layers = q_layers.len();
    let n_vis = vision.len();
    let mut dynamics = vec![vec![0f32; n_layers]; n_vis];
    for (layer, (q, k)) in q_layers.iter().zip(k_layers.iter()).enumerate() {
        extract_eq2_row(
            &LayerQk {
                q: q.to_vec(),
                k: k.to_vec(),
            },
            vision,
            lm,
            &mut dynamics,
            layer,
        );
    }
    Ok(dynamics)
}

/// Fig. 6 decode-step dynamics from exported decode Q/K (text query → visual keys).
pub fn dynamics_from_graph_qk_decode_step(
    q_layers: &[&[f32]],
    k_layers: &[&[f32]],
    vision: VisionKeySpan,
    lm: &Qwen3Config,
) -> Result<Vec<Vec<f32>>> {
    ensure!(
        q_layers.len() == k_layers.len(),
        "decode q/k layer count mismatch"
    );
    let n_layers = q_layers.len();
    let n_vis = vision.len();
    let mut dynamics = vec![vec![0f32; n_layers]; n_vis];
    for (layer, (q, k)) in q_layers.iter().zip(k_layers.iter()).enumerate() {
        extract_decode_step_row(
            &LayerQk {
                q: q.to_vec(),
                k: k.to_vec(),
            },
            vision,
            lm,
            &mut dynamics,
            layer,
        );
    }
    Ok(dynamics)
}

fn extract_decode_step_row(
    qk: &LayerQk,
    vision: VisionKeySpan,
    lm: &Qwen3Config,
    dynamics: &mut [Vec<f32>],
    layer: usize,
) {
    let nh = lm.num_attention_heads;
    let dh = lm.head_dim;
    let q_dim = lm.q_proj_dim();
    let k_len = qk.k.len() / q_dim.max(1);
    let scale = (dh as f32).sqrt().recip();

    for vi in vision.start..vision.end {
        let ki = vi;
        if ki >= k_len {
            continue;
        }
        let mut best = 0f32;
        for head in 0..nh {
            let q_off = head * dh;
            let q_head = &qk.q[q_off..q_off + dh];
            let mut scores = vec![0f32; k_len];
            for j in 0..k_len {
                let k_off = j * q_dim + head * dh;
                scores[j] = dot(q_head, &qk.k[k_off..k_off + dh]) * scale;
            }
            softmax_row(&mut scores);
            best = best.max(scores[ki]);
        }
        dynamics[vi - vision.start][layer] = best;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    #[test]
    fn native_probe_shapes_on_tiny_lm() {
        let cfg = synth::tiny_lm_cfg();
        let weights = synth::synth_lm_weight_map(&cfg);
        let seq = 6usize;
        let h = cfg.lm.hidden_size;
        let hidden: Vec<f32> = (0..seq * h).map(|i| 0.001 * (i as f32)).collect();
        let vision = VisionKeySpan { start: 2, end: 5 };
        let mrope: Vec<[usize; 4]> = (0..seq).map(|p| [p, p, p, 0]).collect();
        let inp = NativePrefillProbeInputs {
            cfg: &cfg,
            weights: &weights,
            hidden: &hidden,
            mrope_sections: &mrope,
            vision,
            seq,
        };
        let probe = native_prefill_probe(&inp).expect("probe");
        assert_eq!(probe.dynamics.len(), vision.len());
        assert_eq!(probe.dynamics[0].len(), cfg.lm.num_hidden_layers);
        assert!(probe.mu.iter().all(|v| v.is_finite()));
    }
}
