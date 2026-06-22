use crate::config::MimiConfig;
use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

const EPSILON: f32 = 1e-5;

#[derive(Debug, Clone)]
struct Codebook {
    embed: Array2<f32>,
}

impl Codebook {
    fn nearest(&self, x: ArrayView2<f32>) -> Vec<u32> {
        let (t, d) = x.dim();
        let (n, dd) = self.embed.dim();
        debug_assert_eq!(d, dd);
        // score[t, i] = 2 * x[t]·e[i] - |e[i]|²  →  argmax == nearest neighbor
        let mut e_sq = vec![0f32; n];
        for i in 0..n {
            e_sq[i] = self.embed.row(i).dot(&self.embed.row(i));
        }
        let scores = x.dot(&self.embed.t());
        let mut out = vec![0u32; t];
        for ti in 0..t {
            let mut best_idx = 0usize;
            let mut best = f32::NEG_INFINITY;
            for i in 0..n {
                let score = 2.0 * scores[[ti, i]] - e_sq[i];
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
struct RvqBranch {
    input_proj_w: Array2<f32>,
    output_proj_w: Array2<f32>,
    layers: Vec<Codebook>,
}

impl RvqBranch {
    fn encode(&self, embeddings: &Array2<f32>, num_q: usize) -> Vec<Vec<u32>> {
        // input_proj: [D, H] @ [H, T] → [D, T] → transpose → [T, D]
        let mut residual = self.input_proj_w.dot(embeddings).t().into_owned();
        let d = residual.dim().1;
        let t = residual.dim().0;
        let mut all = Vec::with_capacity(num_q);
        for layer in self.layers.iter().take(num_q) {
            let indices = layer.nearest(residual.view());
            let q = layer.lookup(&indices);
            for ti in 0..t {
                for j in 0..d {
                    residual[[ti, j]] -= q[[ti, j]];
                }
            }
            all.push(indices);
        }
        all
    }

    fn decode(&self, codes: &[Vec<u32>]) -> Array2<f32> {
        let t = codes.iter().map(|c| c.len()).max().unwrap_or(0);
        let d = self.layers.first().map(|l| l.embed.dim().1).unwrap_or(0);
        let mut sum = Array2::<f32>::zeros((t, d));
        for (layer, frame) in self.layers.iter().zip(codes.iter()) {
            if frame.is_empty() {
                continue;
            }
            sum += &layer.lookup(frame);
        }
        self.output_proj_w.dot(&sum.t()).into_owned()
    }
}

pub struct SplitRvq {
    cfg: MimiConfig,
    semantic: RvqBranch,
    acoustic: RvqBranch,
}

impl SplitRvq {
    pub fn encode_frames(&self, embeddings: &Array2<f32>, num_q: Option<usize>) -> Vec<Vec<u32>> {
        let total = num_q.unwrap_or(self.cfg.num_quantizers);
        let n_sem = self.cfg.num_semantic_quantizers;
        let sem = self.semantic.encode(embeddings, n_sem);
        let n_acc = total.saturating_sub(n_sem);
        let acc = if n_acc > 0 {
            self.acoustic.encode(embeddings, n_acc)
        } else {
            Vec::new()
        };
        let t = embeddings.dim().1;
        let mut frames = Vec::with_capacity(t);
        for ti in 0..t {
            let mut row = Vec::with_capacity(total);
            for k in 0..n_sem {
                row.push(sem[k][ti]);
            }
            for k in 0..n_acc {
                row.push(acc[k][ti]);
            }
            frames.push(row);
        }
        frames
    }

    pub fn decode_frames(&self, frames: &[Vec<u32>]) -> Array2<f32> {
        let t = frames.len();
        if t == 0 {
            return Array2::<f32>::zeros((self.cfg.hidden_size, 0));
        }
        let num_q = frames[0].len().min(self.cfg.num_quantizers);
        let n_sem = self.cfg.num_semantic_quantizers.min(num_q);
        let n_acc = num_q.saturating_sub(n_sem);
        let mut sem_codes: Vec<Vec<u32>> = (0..n_sem).map(|_| Vec::with_capacity(t)).collect();
        let mut acc_codes: Vec<Vec<u32>> = (0..n_acc).map(|_| Vec::with_capacity(t)).collect();
        for row in frames {
            for (k, &code) in row.iter().take(num_q).enumerate() {
                if k < n_sem {
                    sem_codes[k].push(code);
                } else {
                    acc_codes[k - n_sem].push(code);
                }
            }
        }
        let sem = self.semantic.decode(&sem_codes);
        let acc = if n_acc > 0 {
            self.acoustic.decode(&acc_codes)
        } else {
            Array2::<f32>::zeros((self.cfg.hidden_size, t))
        };
        sem + acc
    }
}

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
    ensure!(usage_shape == vec![codebook_size]);
    ensure!(sum_shape == vec![codebook_size, codebook_dim]);
    raw.remove(&format!("{prefix}.initialized"));
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
    cfg: &MimiConfig,
    num_layers: usize,
) -> Result<RvqBranch> {
    let in_key = format!("{prefix}.input_proj.weight");
    let (in_proj, in_shape) = raw
        .remove(&in_key)
        .with_context(|| format!("missing {in_key}"))?;
    ensure!(
        in_shape == vec![cfg.vector_quantization_hidden_dimension, cfg.hidden_size, 1],
        "{in_key} shape {in_shape:?}"
    );
    let input_proj_w = Array2::from_shape_vec(
        (cfg.vector_quantization_hidden_dimension, cfg.hidden_size),
        in_proj,
    )?;
    let out_key = format!("{prefix}.output_proj.weight");
    let (out_proj, out_shape) = raw
        .remove(&out_key)
        .with_context(|| format!("missing {out_key}"))?;
    ensure!(
        out_shape == vec![cfg.hidden_size, cfg.vector_quantization_hidden_dimension, 1],
        "{out_key} shape {out_shape:?}"
    );
    let output_proj_w = Array2::from_shape_vec(
        (cfg.hidden_size, cfg.vector_quantization_hidden_dimension),
        out_proj,
    )?;
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        layers.push(build_codebook(
            raw,
            &format!("{prefix}.layers.{i}.codebook"),
            cfg.codebook_size,
            cfg.vector_quantization_hidden_dimension,
        )?);
    }
    Ok(RvqBranch {
        input_proj_w,
        output_proj_w,
        layers,
    })
}

pub fn build_split_rvq(
    cfg: &MimiConfig,
    raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<SplitRvq> {
    let mut local: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (k, v) in raw {
        if let Some(rest) = k.strip_prefix("quantizer.") {
            local.insert(rest.to_string(), v);
        }
    }
    let n_acc = cfg.num_quantizers - cfg.num_semantic_quantizers;
    let semantic = build_branch(
        &mut local,
        "semantic_residual_vector_quantizer",
        cfg,
        cfg.num_semantic_quantizers,
    )?;
    let acoustic = build_branch(&mut local, "acoustic_residual_vector_quantizer", cfg, n_acc)?;
    if !local.is_empty() {
        bail!(
            "unused quantizer tensors: {:?}",
            local.keys().take(4).collect::<Vec<_>>()
        );
    }
    Ok(SplitRvq {
        cfg: cfg.clone(),
        semantic,
        acoustic,
    })
}
