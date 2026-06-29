//! Shared weight-map helpers for Kyutai TTS checkpoint loading.

use crate::weights::WeightMap;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};

pub fn take_mat2(weights: &WeightMap, key: &str) -> Result<Array2<f32>> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    ensure_len(key, data, shape.iter().product())?;
    Ok(Array2::from_shape_vec((shape[0], shape[1]), data.clone())?)
}

pub fn take_vec1(weights: &WeightMap, key: &str) -> Result<Array1<f32>> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    let n: usize = shape.iter().product();
    ensure_len(key, data, n)?;
    Ok(Array1::from_vec(data.clone()))
}

/// Kyutai checkpoints store RMSNorm scales as `[1, 1, d]` — squeeze to `[d]`.
pub fn take_rms_alpha(weights: &WeightMap, key: &str) -> Result<Array1<f32>> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    let n: usize = shape.iter().product();
    ensure_len(key, data, n)?;
    if shape.len() == 3 && shape[0] == 1 && shape[1] == 1 {
        let d = shape[2];
        return Ok(Array1::from_iter(data.iter().take(d).copied()));
    }
    Ok(Array1::from_vec(data.clone()))
}

/// Split fused QKV `[3·d, d]` into three `[d, d]` matrices (row-major).
pub fn split_qkv(
    in_proj: &Array2<f32>,
    d: usize,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    anyhow::ensure!(
        in_proj.nrows() == 3 * d && in_proj.ncols() == d,
        "expected in_proj [{}, {d}], got {:?}",
        3 * d,
        in_proj.dim()
    );
    Ok((
        in_proj.slice(ndarray::s![0..d, ..]).to_owned(),
        in_proj.slice(ndarray::s![d..2 * d, ..]).to_owned(),
        in_proj.slice(ndarray::s![2 * d..3 * d, ..]).to_owned(),
    ))
}

fn ensure_len(key: &str, data: &[f32], expected: usize) -> Result<()> {
    anyhow::ensure!(
        data.len() == expected,
        "tensor {key}: shape product {expected} != len {}",
        data.len()
    );
    Ok(())
}
