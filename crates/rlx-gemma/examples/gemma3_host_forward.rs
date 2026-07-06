// Host-side Gemma 3 forward (llama.cpp gemma3.cpp semantics) for parity bisection.
//
//   RLX_PARITY_IDS='2,105,...' cargo run --release -p rlx-gemma \
//     --features parity-llama --example gemma3_host_forward -- /path/to/model.gguf

use anyhow::{Context, Result, bail};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_gemma::config::{GemmaArch, GemmaConfig, gemma_cfg_from_gguf};
use std::path::PathBuf;

fn parse_ids() -> Result<Vec<u32>> {
    if let Ok(raw) = std::env::var("RLX_PARITY_IDS") {
        return raw
            .split(',')
            .map(|s| s.trim().parse().context("RLX_PARITY_IDS"))
            .collect();
    }
    bail!("set RLX_PARITY_IDS");
}

fn gemma_rms_rows(x: &mut [f32], gamma: &[f32], eps: f32, cols: usize) {
    let rows = x.len() / cols;
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let sumsq: f32 = row.iter().map(|v| v * v).sum();
        let inv = (sumsq / cols as f32 + eps).sqrt().recip();
        for i in 0..cols {
            row[i] = row[i] * inv * (1.0 + gamma[i]);
        }
    }
}

fn gemma_rms_per_head(x: &mut [f32], gamma: &[f32], eps: f32, heads: usize, dh: usize) {
    for h in 0..heads {
        let slice = &mut x[h * dh..(h + 1) * dh];
        let sumsq: f32 = slice.iter().map(|v| v * v).sum();
        let inv = (sumsq / dh as f32 + eps).sqrt().recip();
        for i in 0..dh {
            let g = if gamma.is_empty() {
                1.0
            } else {
                1.0 + gamma[i]
            };
            slice[i] = slice[i] * inv * g;
        }
    }
}

fn matmul_seq(x: &[f32], seq: usize, in_dim: usize, w: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; seq * out_dim];
    for t in 0..seq {
        let xrow = &x[t * in_dim..(t + 1) * in_dim];
        for o in 0..out_dim {
            let wrow = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0f32;
            for i in 0..in_dim {
                acc += wrow[i] * xrow[i];
            }
            out[t * out_dim + o] = acc;
        }
    }
    out
}

fn gelu_approx(x: f32) -> f32 {
    // tanh approximation used by Gemma / llama.cpp
    let x3 = x * x * x;
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x3)).tanh())
}

fn rope_neox(q: &mut [f32], cos: &[f32], sin: &[f32], pos: usize, nh: usize, dh: usize) {
    let half = dh / 2;
    for h in 0..nh {
        let base = h * dh;
        for i in 0..half {
            let c = cos[pos * half + i];
            let s = sin[pos * half + i];
            let x0 = q[base + i];
            let x1 = q[base + half + i];
            q[base + i] = x0 * c - x1 * s;
            q[base + half + i] = x0 * s + x1 * c;
        }
    }
}

fn build_rope_tables(theta: f64, dh: usize, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dh / 2;
    let mut inv = Vec::with_capacity(half);
    for i in 0..half {
        inv.push(1.0 / theta.powf(2.0 * i as f64 / dh as f64));
    }
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for p in 0..max_pos {
        for i in 0..half {
            let angle = p as f64 * inv[i];
            cos[p * half + i] = angle.cos() as f32;
            sin[p * half + i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    nh: usize,
    n_kv: usize,
    dh: usize,
    sliding: Option<usize>,
    q_scale: f32,
) -> Vec<f32> {
    let group = nh / n_kv;
    let mut out = vec![0f32; seq * nh * dh];
    for tq in 0..seq {
        for h in 0..nh {
            let kv_h = h / group;
            let q_off = (tq * nh + h) * dh;
            let mut scores = vec![0f32; seq];
            let mut max_s = f32::NEG_INFINITY;
            for tk in 0..seq {
                if tk > tq {
                    scores[tk] = f32::NEG_INFINITY;
                    continue;
                }
                if let Some(w) = sliding {
                    if tq.saturating_sub(tk) >= w {
                        scores[tk] = f32::NEG_INFINITY;
                        continue;
                    }
                }
                let k_off = (tk * n_kv + kv_h) * dh;
                let mut dot = 0f32;
                for d in 0..dh {
                    dot += q[q_off + d] * q_scale * k[k_off + d];
                }
                scores[tk] = dot;
                if dot > max_s {
                    max_s = dot;
                }
            }
            let mut sum = 0f32;
            for tk in 0..seq {
                if scores[tk].is_finite() {
                    scores[tk] = (scores[tk] - max_s).exp();
                    sum += scores[tk];
                } else {
                    scores[tk] = 0.0;
                }
            }
            if sum > 0.0 {
                for tk in 0..seq {
                    scores[tk] /= sum;
                }
            }
            let o_off = (tq * nh + h) * dh;
            for d in 0..dh {
                let mut acc = 0f32;
                for tk in 0..seq {
                    let v_off = (tk * n_kv + kv_h) * dh;
                    acc += scores[tk] * v[v_off + d];
                }
                out[o_off + d] = acc;
            }
        }
    }
    out
}

fn take_w(loader: &mut GgufLoader, key: &str) -> Result<Vec<f32>> {
    let (data, _) = loader.take(key).with_context(|| format!("missing {key}"))?;
    Ok(data)
}

fn take_proj(loader: &mut GgufLoader, key: &str) -> Result<Vec<f32>> {
    let (data, _) = loader
        .take_transposed(key)
        .with_context(|| format!("missing {key}"))?;
    Ok(data)
}

fn forward_with_embed(
    cfg: &GemmaConfig,
    loader: &mut GgufLoader,
    ids: &[u32],
    embed: &[f32],
    max_layers: Option<usize>,
    use_v_norm: bool,
) -> Result<Vec<f32>> {
    if cfg.arch != GemmaArch::Gemma3 {
        bail!("host forward is Gemma 3 only");
    }
    let h = cfg.hidden_size;
    let seq = ids.len();
    let nh = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let dh = cfg.head_dim();
    let eps = cfg.rms_norm_eps as f32;
    let int_dim = cfg.intermediate_size;
    let q_scale = cfg.attn_score_scale().unwrap_or(1.0 / (dh as f32).sqrt());
    let sliding_w = cfg.sliding_window;
    let stride = cfg.arch.sliding_window_stride();

    let mut hidden = vec![0f32; seq * h];
    let scale = (h as f32).sqrt();
    for (t, &id) in ids.iter().enumerate() {
        let row = &embed[id as usize * h..(id as usize + 1) * h];
        for i in 0..h {
            hidden[t * h + i] = row[i] * scale;
        }
    }

    let (swa_cos, swa_sin) = build_rope_tables(10_000.0, dh, seq);
    let (full_cos, full_sin) = build_rope_tables(1_000_000.0, dh, seq);

    let n_layers = max_layers
        .unwrap_or(cfg.num_hidden_layers)
        .min(cfg.num_hidden_layers);

    for layer in 0..n_layers {
        let lp = format!("model.layers.{layer}");
        let is_full = stride > 1 && (layer + 1).is_multiple_of(stride);
        let (cos, sin) = if is_full {
            (&full_cos, &full_sin)
        } else {
            (&swa_cos, &swa_sin)
        };
        let mask = if is_full { None } else { sliding_w };

        let attn_norm = take_w(loader, &format!("{lp}.input_layernorm.weight"))?;
        let mut normed = hidden.clone();
        gemma_rms_rows(&mut normed, &attn_norm, eps, h);

        let q_w = take_proj(loader, &format!("{lp}.self_attn.q_proj.weight"))?;
        let k_w = take_proj(loader, &format!("{lp}.self_attn.k_proj.weight"))?;
        let v_w = take_proj(loader, &format!("{lp}.self_attn.v_proj.weight"))?;
        let o_w = take_proj(loader, &format!("{lp}.self_attn.o_proj.weight"))?;
        let q_norm_w = take_w(loader, &format!("{lp}.self_attn.q_norm.weight"))?;
        let k_norm_w = take_w(loader, &format!("{lp}.self_attn.k_norm.weight"))?;

        let mut q = matmul_seq(&normed, seq, h, &q_w, nh * dh);
        let mut k = matmul_seq(&normed, seq, h, &k_w, n_kv * dh);
        let v = matmul_seq(&normed, seq, h, &v_w, n_kv * dh);

        for t in 0..seq {
            let qrow = &mut q[t * nh * dh..(t + 1) * nh * dh];
            gemma_rms_per_head(qrow, &q_norm_w, eps, nh, dh);
            let krow = &mut k[t * n_kv * dh..(t + 1) * n_kv * dh];
            gemma_rms_per_head(krow, &k_norm_w, eps, n_kv, dh);
            rope_neox(qrow, cos, sin, t, nh, dh);
            rope_neox(krow, cos, sin, t, n_kv, dh);
        }
        // Gemma 3 / llama.cpp: no V RMS-norm. RLX applies unit v_norm (Gemma 4 path).
        let mut v = v;
        if use_v_norm {
            for t in 0..seq {
                let vrow = &mut v[t * n_kv * dh..(t + 1) * n_kv * dh];
                gemma_rms_per_head(vrow, &[], eps, n_kv, dh);
            }
        }

        let attn = attention(&q, &k, &v, seq, nh, n_kv, dh, mask, q_scale);
        let mut attn_out = matmul_seq(&attn, seq, nh * dh, &o_w, h);

        let post_attn_w = take_w(loader, &format!("{lp}.post_attention_layernorm.weight"))?;
        gemma_rms_rows(&mut attn_out, &post_attn_w, eps, h);
        for i in 0..hidden.len() {
            hidden[i] += attn_out[i];
        }

        let pre_ffn_w = take_w(loader, &format!("{lp}.pre_feedforward_layernorm.weight"))?;
        let mut ffn_in = hidden.clone();
        gemma_rms_rows(&mut ffn_in, &pre_ffn_w, eps, h);

        let gate_w = take_proj(loader, &format!("{lp}.mlp.gate_proj.weight"))?;
        let up_w = take_proj(loader, &format!("{lp}.mlp.up_proj.weight"))?;
        let down_w = take_proj(loader, &format!("{lp}.mlp.down_proj.weight"))?;
        let gate = matmul_seq(&ffn_in, seq, h, &gate_w, int_dim);
        let up = matmul_seq(&ffn_in, seq, h, &up_w, int_dim);
        let mut inner = vec![0f32; seq * int_dim];
        for i in 0..inner.len() {
            inner[i] = gelu_approx(gate[i]) * up[i];
        }
        let mut ffn_out = matmul_seq(&inner, seq, int_dim, &down_w, h);
        let post_ffn_w = take_w(loader, &format!("{lp}.post_feedforward_layernorm.weight"))?;
        gemma_rms_rows(&mut ffn_out, &post_ffn_w, eps, h);
        for i in 0..hidden.len() {
            hidden[i] += ffn_out[i];
        }
    }

    if n_layers == cfg.num_hidden_layers {
        let norm_w = take_w(loader, "model.norm.weight")?;
        gemma_rms_rows(&mut hidden, &norm_w, eps, h);
    }
    Ok(hidden)
}

fn lm_head_tied(hidden_last: &[f32], embed: &[f32], vocab: usize, h: usize) -> Vec<f32> {
    let mut logits = vec![0f32; vocab];
    for v in 0..vocab {
        let row = &embed[v * h..(v + 1) * h];
        logits[v] = hidden_last.iter().zip(row).map(|(a, b)| a * b).sum();
    }
    logits
}

fn argmax(v: &[f32]) -> (usize, f32) {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &x)| (i, x))
        .unwrap_or((0, 0.0))
}

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .context("usage: gemma3_host_forward <gguf>")?
        .into();
    let ids = parse_ids()?;
    let raw = rlx_gguf::GgufFile::from_path(&path)?;
    let cfg = gemma_cfg_from_gguf(&raw)?;
    let mut loader = GgufLoader::from_file(path.to_string_lossy().as_ref())?;
    let embed = take_w(&mut loader, "model.embed_tokens.weight")?;
    let use_v_norm = std::env::var("RLX_HOST_V_NORM").ok().as_deref() != Some("0");
    let max_layers = std::env::var("RLX_HOST_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok());
    let hidden = forward_with_embed(&cfg, &mut loader, &ids, &embed, max_layers, use_v_norm)?;
    let h = cfg.hidden_size;
    let last = &hidden[(ids.len() - 1) * h..ids.len() * h];
    let rms = (last.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / h as f64).sqrt();
    println!(
        "host last_hidden rms={rms:.4} first4={:?}",
        &last[..4.min(h)]
    );

    if max_layers.is_some() && max_layers != Some(cfg.num_hidden_layers) {
        return Ok(());
    }

    let normed = hidden;
    let logits = lm_head_tied(
        &normed[(ids.len() - 1) * h..ids.len() * h],
        &embed,
        cfg.vocab_size,
        h,
    );
    let (top, val) = argmax(&logits);
    println!(
        "host top1={top} ({val:.4}) logit[11634]={:.4}",
        logits[11634]
    );

    #[cfg(feature = "parity-llama")]
    {
        let llama = rlx_gemma::llama_reference::last_token_logits(&path, &ids)?;
        let (lt, lv) = argmax(&llama);
        println!("llama top1={lt} ({lv:.4}) logit[11634]={:.4}", llama[11634]);
        if top == lt {
            println!("HOST MATCHES LLAMA ✓");
        } else {
            println!("HOST MISMATCH vs llama");
        }
    }
    Ok(())
}
