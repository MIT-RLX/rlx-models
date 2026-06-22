// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Host euclidean residual VQ (codebook_dim == dimension, no projections).

/// `latent`: `[dim, T]`. `codebooks[q]`: `[codebook_size, dim]`. Codes `[n_q][T]`.
pub fn rvq_encode(
    codebooks: &[Vec<f32>],
    latent: &[f32],
    dim: usize,
    t: usize,
    n_q: usize,
) -> Vec<Vec<u32>> {
    let nq = n_q.min(codebooks.len());
    let mut residual = latent.to_vec();
    let mut codes = Vec::with_capacity(nq);
    for cb in codebooks.iter().take(nq) {
        let cbsize = cb.len() / dim;
        let e_sq: Vec<f32> = (0..cbsize)
            .map(|i| {
                (0..dim)
                    .map(|d| cb[i * dim + d] * cb[i * dim + d])
                    .sum::<f32>()
            })
            .collect();
        let mut row = vec![0u32; t];
        for ti in 0..t {
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for i in 0..cbsize {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += residual[d * t + ti] * cb[i * dim + d];
                }
                let score = 2.0 * dot - e_sq[i];
                if score > best_score {
                    best_score = score;
                    best = i;
                }
            }
            row[ti] = best as u32;
            for d in 0..dim {
                residual[d * t + ti] -= cb[best * dim + d];
            }
        }
        codes.push(row);
    }
    codes
}

/// Codes `[n_q][T]` → quantized latent `[dim, T]` (sum of codebook rows).
pub fn rvq_decode(codebooks: &[Vec<f32>], codes: &[Vec<u32>], dim: usize) -> Vec<f32> {
    let t = codes.first().map(|c| c.len()).unwrap_or(0);
    let mut z = vec![0f32; dim * t];
    for (cb, row) in codebooks.iter().zip(codes) {
        for (ti, &idx) in row.iter().enumerate() {
            let base = idx as usize * dim;
            for d in 0..dim {
                z[d * t + ti] += cb[base + d];
            }
        }
    }
    z
}
