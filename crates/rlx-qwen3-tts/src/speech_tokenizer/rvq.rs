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

//! Split Residual Vector Quantizer — encoder side.
//!
//! Mirrors `MimiSplitResidualVectorQuantizer.encode`:
//!   semantic_rvq.encode(emb) ‖ acoustic_rvq.encode(emb, num_q - 1)  →  codes [K, T]
//!
//! Each RVQ branch:
//!   x = input_proj(emb)                  ([hidden] → [codebook_dim])
//!   residual = x
//!   for layer in layers[:K]:
//!     indices = nearest_codebook_entry(residual)
//!     residual = residual − codebook[indices]
//!     append(indices)
//!
//! Codebook entries: `embed = embed_sum / max(cluster_usage, eps)`.

use anyhow::{Context, Result, bail, ensure};
use ndarray::Array2;
use std::collections::HashMap;

const EPSILON: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct RvqConfig {
    pub hidden_size: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub num_semantic_quantizers: usize,
    pub num_total_quantizers: usize,
    pub encoder_valid_num_quantizers: usize,
}

impl RvqConfig {
    pub fn from_speech_tokenizer_dir(dir: &std::path::Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let text = std::fs::read_to_string(&cfg_path)?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let enc = v.get("encoder_config").context("missing encoder_config")?;
        let u = |k: &str| -> Result<usize> {
            enc.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("encoder_config.{k}"))
        };
        let valid_q = v
            .get("encoder_valid_num_quantizers")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| 0);
        Ok(Self {
            hidden_size: u("hidden_size")?,
            codebook_dim: u("vector_quantization_hidden_dimension")?,
            codebook_size: u("codebook_size")?,
            num_semantic_quantizers: u("num_semantic_quantizers")?,
            num_total_quantizers: u("num_quantizers")?,
            encoder_valid_num_quantizers: if valid_q > 0 {
                valid_q
            } else {
                u("num_quantizers")?
            },
        })
    }
}

/// One codebook: stores `embed = embed_sum / clamp(cluster_usage, eps)`,
/// shape `[codebook_size, codebook_dim]`.
#[derive(Debug, Clone)]
struct Codebook {
    embed: Array2<f32>,
}

impl Codebook {
    /// Nearest-neighbor lookup. `x` is `[T, D]`; returns indices `[T]`.
    fn nearest(&self, x: &Array2<f32>) -> Vec<u32> {
        let (t, d) = x.dim();
        let (n, dd) = self.embed.dim();
        debug_assert_eq!(d, dd);
        // dist^2 = |x|^2 + |e|^2 − 2 x·e. argmin over e → argmax of (2 x·e − |e|^2).
        let mut e_sq = vec![0f32; n];
        for i in 0..n {
            let mut s = 0f32;
            for j in 0..d {
                s += self.embed[[i, j]] * self.embed[[i, j]];
            }
            e_sq[i] = s;
        }
        let mut out = vec![0u32; t];
        for ti in 0..t {
            let mut best_idx = 0usize;
            let mut best = f32::NEG_INFINITY;
            for i in 0..n {
                let mut dot = 0f32;
                for j in 0..d {
                    dot += x[[ti, j]] * self.embed[[i, j]];
                }
                let score = 2.0 * dot - e_sq[i];
                if score > best {
                    best = score;
                    best_idx = i;
                }
            }
            out[ti] = best_idx as u32;
        }
        out
    }

    fn lookup(&self, indices: &[u32]) -> Array2<f32> {
        let d = self.embed.dim().1;
        let mut out = Array2::<f32>::zeros((indices.len(), d));
        for (ti, &idx) in indices.iter().enumerate() {
            for j in 0..d {
                out[[ti, j]] = self.embed[[idx as usize, j]];
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
struct Rvq {
    input_proj_w: Array2<f32>, // [codebook_dim, hidden]
    layers: Vec<Codebook>,
}

impl Rvq {
    /// Encode `embeddings` `[hidden, T]` with `num_q` codebooks, returning
    /// indices `[num_q, T]`.
    fn encode(&self, embeddings: &Array2<f32>, num_q: usize) -> Vec<Vec<u32>> {
        // Apply input_proj (1x1 conv = linear over channels).
        let (h, t) = embeddings.dim();
        let d = self.input_proj_w.dim().0;
        debug_assert_eq!(h, self.input_proj_w.dim().1);
        // x[oc, t] = sum_ic w[oc, ic] * emb[ic, t]
        let mut x = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            for oc in 0..d {
                let mut s = 0f32;
                for ic in 0..h {
                    s += self.input_proj_w[[oc, ic]] * embeddings[[ic, ti]];
                }
                x[[ti, oc]] = s;
            }
        }

        let mut residual = x;
        let mut all_indices = Vec::with_capacity(num_q);
        for layer in self.layers.iter().take(num_q) {
            let indices = layer.nearest(&residual);
            let q = layer.lookup(&indices);
            for ti in 0..residual.dim().0 {
                for j in 0..d {
                    residual[[ti, j]] -= q[[ti, j]];
                }
            }
            all_indices.push(indices);
        }
        all_indices
    }
}

#[derive(Debug, Clone)]
pub struct SplitRvq {
    pub cfg: RvqConfig,
    semantic: Rvq,
    acoustic: Rvq,
}

impl SplitRvq {
    /// Encode `embeddings` `[hidden, T]`. Returns codes as `[T][num_q]` —
    /// one Vec<u32> per frame, length `num_q`.
    ///
    /// `num_q` defaults to `cfg.num_total_quantizers`; pass `Some(n)` to truncate.
    pub fn encode_frames(&self, embeddings: &Array2<f32>, num_q: Option<usize>) -> Vec<Vec<u32>> {
        let total = num_q.unwrap_or(self.cfg.num_total_quantizers);
        let n_sem = self.cfg.num_semantic_quantizers;
        let sem_codes = self.semantic.encode(embeddings, n_sem);
        let n_acoustic = total.saturating_sub(n_sem);
        let acc_codes = if n_acoustic > 0 {
            self.acoustic.encode(embeddings, n_acoustic)
        } else {
            Vec::new()
        };
        let t = embeddings.dim().1;
        let mut frames = Vec::with_capacity(t);
        for ti in 0..t {
            let mut row = Vec::with_capacity(total);
            for k in 0..n_sem {
                row.push(sem_codes[k][ti]);
            }
            for k in 0..n_acoustic {
                row.push(acc_codes[k][ti]);
            }
            frames.push(row);
        }
        frames
    }
}

// -----------------------------------------------------------------------------
// Weight loader.

fn build_codebook(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    codebook_size: usize,
    codebook_dim: usize,
) -> Result<Codebook> {
    let usage_key = format!("{prefix}.cluster_usage");
    let sum_key = format!("{prefix}.embed_sum");
    let (usage, usage_shape) = raw
        .remove(&usage_key)
        .with_context(|| format!("missing {usage_key}"))?;
    let (sum, sum_shape) = raw
        .remove(&sum_key)
        .with_context(|| format!("missing {sum_key}"))?;
    ensure!(
        usage_shape == vec![codebook_size],
        "{usage_key} shape {:?} != [{codebook_size}]",
        usage_shape
    );
    ensure!(
        sum_shape == vec![codebook_size, codebook_dim],
        "{sum_key} shape {:?} != [{codebook_size}, {codebook_dim}]",
        sum_shape
    );
    // Drop `initialized` (we don't need it for encode).
    let init_key = format!("{prefix}.initialized");
    raw.remove(&init_key);

    let mut embed = Array2::<f32>::zeros((codebook_size, codebook_dim));
    for i in 0..codebook_size {
        let scale = 1.0 / usage[i].max(EPSILON);
        for j in 0..codebook_dim {
            embed[[i, j]] = sum[i * codebook_dim + j] * scale;
        }
    }
    Ok(Codebook { embed })
}

fn build_branch(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    cfg: &RvqConfig,
    num_layers: usize,
) -> Result<Rvq> {
    // input_proj weight shape: [codebook_dim, hidden_size, 1]
    let in_proj_key = format!("{prefix}.input_proj.weight");
    let (in_proj, in_shape) = raw
        .remove(&in_proj_key)
        .with_context(|| format!("missing {in_proj_key}"))?;
    ensure!(
        in_shape == vec![cfg.codebook_dim, cfg.hidden_size, 1],
        "{in_proj_key} shape {:?} != [{}, {}, 1]",
        in_shape,
        cfg.codebook_dim,
        cfg.hidden_size
    );
    let input_proj_w = Array2::from_shape_vec((cfg.codebook_dim, cfg.hidden_size), in_proj)?;

    // output_proj is present in the weights but unused for encode.
    let out_proj_key = format!("{prefix}.output_proj.weight");
    raw.remove(&out_proj_key);

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let pfx = format!("{prefix}.layers.{i}.codebook");
        layers.push(build_codebook(
            raw,
            &pfx,
            cfg.codebook_size,
            cfg.codebook_dim,
        )?);
    }
    Ok(Rvq {
        input_proj_w,
        layers,
    })
}

pub fn build_split_rvq(
    cfg: &RvqConfig,
    raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<SplitRvq> {
    let mut local: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        if let Some(rest) = k.strip_prefix("encoder.quantizer.") {
            local.insert(rest.to_string(), v);
        }
    }
    let n_acoustic = cfg.num_total_quantizers - cfg.num_semantic_quantizers;
    let semantic = build_branch(
        &mut local,
        "semantic_residual_vector_quantizer",
        cfg,
        cfg.num_semantic_quantizers,
    )?;
    let acoustic = build_branch(
        &mut local,
        "acoustic_residual_vector_quantizer",
        cfg,
        n_acoustic,
    )?;

    if !local.is_empty() {
        let leftover: Vec<&String> = local.keys().take(5).collect();
        bail!(
            "{} unused quantizer tensors (first: {:?})",
            local.len(),
            leftover
        );
    }
    Ok(SplitRvq {
        cfg: cfg.clone(),
        semantic,
        acoustic,
    })
}

pub fn open_split_rvq(tok_dir: &std::path::Path) -> Result<SplitRvq> {
    let cfg = RvqConfig::from_speech_tokenizer_dir(tok_dir)?;
    let ckpt = rlx_core::safetensors_checkpoint::SafetensorsCheckpoint::open(tok_dir)?;
    let want: std::collections::HashSet<String> = ckpt
        .keys()
        .filter(|k| k.starts_with("encoder.quantizer."))
        .map(str::to_string)
        .collect();
    ensure!(!want.is_empty(), "no encoder.quantizer.* tensors found");
    let mut wm = ckpt.load_selected(&want)?;
    let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::with_capacity(want.len());
    for k in want.iter() {
        let (data, shape) = wm.take(k)?;
        raw.insert(k.clone(), (data, shape));
    }
    build_split_rvq(&cfg, raw)
}
