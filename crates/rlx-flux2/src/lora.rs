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

//! FLUX.2 LoRA adapter merge (PEFT / diffusers safetensors → base weight map).

use anyhow::{Context, Result, bail, ensure};
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;
use std::path::Path;

/// LoRA side tensor in a matched pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoraSide {
    A,
    B,
}

/// Normalize HF / diffusers key prefixes so LoRA keys align with denoiser weights.
fn normalize_lora_key(key: &str) -> String {
    crate::adapt::normalize_flux2_key(key)
}

fn parse_lora_side(key: &str) -> Option<(&str, LoraSide)> {
    for (suffix, side) in [
        (".lora_A.weight", LoraSide::A),
        (".lora_B.weight", LoraSide::B),
        (".lora_down.weight", LoraSide::A),
        (".lora_up.weight", LoraSide::B),
        (".lora_a.weight", LoraSide::A),
        (".lora_b.weight", LoraSide::B),
    ] {
        if let Some(base) = key.strip_suffix(suffix) {
            return Some((base, side));
        }
    }
    None
}

fn lora_delta(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    scale: f32,
) -> Result<Vec<f32>> {
    ensure!(
        a_shape.len() == 2 && b_shape.len() == 2,
        "LoRA A/B must be rank-2"
    );
    let (rank_a, in_dim) = (a_shape[0], a_shape[1]);
    let (out_dim, rank_b) = (b_shape[0], b_shape[1]);
    ensure!(
        rank_a == rank_b,
        "LoRA rank mismatch: A {a_shape:?} vs B {b_shape:?}"
    );
    ensure!(
        a.len() == rank_a * in_dim && b.len() == out_dim * rank_b,
        "LoRA tensor size mismatch"
    );
    let mut delta = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            let mut acc = 0.0f32;
            for r in 0..rank_a {
                acc += b[o * rank_b + r] * a[r * in_dim + i];
            }
            delta[o * in_dim + i] = scale * acc;
        }
    }
    Ok(delta)
}

/// Merge LoRA safetensors into `base` in-place. Returns the number of merged layers.
pub fn apply_flux2_lora(base: &mut WeightMap, lora: &WeightMap, scale: f32) -> Result<usize> {
    if scale == 0.0 {
        return Ok(0);
    }

    #[allow(clippy::type_complexity)]
    let mut pairs: HashMap<
        String,
        (
            Option<(Vec<f32>, Vec<usize>)>,
            Option<(Vec<f32>, Vec<usize>)>,
        ),
    > = HashMap::new();

    for key in lora.keys() {
        let norm = normalize_lora_key(key);
        let Some((base_prefix, side)) = parse_lora_side(&norm) else {
            continue;
        };
        let Some((data, shape)) = lora.get(key) else {
            continue;
        };
        let entry = pairs.entry(base_prefix.to_string()).or_default();
        match side {
            LoraSide::A => entry.0 = Some((data.to_vec(), shape.to_vec())),
            LoraSide::B => entry.1 = Some((data.to_vec(), shape.to_vec())),
        }
    }

    let mut merged = 0usize;
    for (prefix, (a, b)) in pairs {
        let (a, b) = match (a, b) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let weight_key = format!("{prefix}.weight");
        if !base.has(&weight_key) {
            continue;
        }
        let delta = lora_delta(&a.0, &a.1, &b.0, &b.1, scale)?;
        base.merge_add_weight(&weight_key, &delta)?;
        merged += 1;
    }
    Ok(merged)
}

/// Load LoRA from safetensors and merge into `base`.
pub fn load_and_apply_flux2_lora(
    base: &mut WeightMap,
    lora_path: &Path,
    scale: f32,
) -> Result<usize> {
    let path = lora_path
        .to_str()
        .with_context(|| format!("non-utf8 LoRA path {lora_path:?}"))?;
    let lora = WeightMap::from_file(path)?;
    apply_flux2_lora(base, &lora, scale)
}

/// Load LoRA from a directory of safetensors shards.
pub fn load_and_apply_flux2_lora_dir(
    base: &mut WeightMap,
    lora_dir: &Path,
    scale: f32,
) -> Result<usize> {
    let lora = WeightMap::from_safetensors_dir(lora_dir)?;
    apply_flux2_lora(base, &lora, scale)
}

/// Parse `--lora-scale` style input; rejects NaN/inf.
pub fn parse_lora_scale(s: &str) -> Result<f32> {
    let v: f32 = s.parse().context("lora scale: f32")?;
    if !v.is_finite() {
        bail!("lora scale must be finite");
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn lora_delta_rank1_matches_manual() {
        // W is 2x2 zeros; delta = scale * B @ A with rank 1
        let a = vec![1.0f32, 2.0]; // [1, 2]
        let b = vec![3.0f32, 4.0]; // [2, 1]
        let delta = lora_delta(&a, &[1, 2], &b, &[2, 1], 1.0).unwrap();
        // B @ A = [[3],[4]] @ [[1,2]] = [[3,6],[4,8]]
        assert_eq!(delta, vec![3.0, 6.0, 4.0, 8.0]);
    }

    #[test]
    fn apply_lora_merges_into_base_weight() {
        let mut base = WeightMap::from_tensors(HashMap::from([(
            "proj.weight".to_string(),
            (vec![10.0, 20.0], vec![2, 1]),
        )]));
        let lora = WeightMap::from_tensors(HashMap::from([
            ("proj.lora_A.weight".to_string(), (vec![2.0], vec![1, 1])),
            (
                "proj.lora_B.weight".to_string(),
                (vec![3.0, 4.0], vec![2, 1]),
            ),
        ]));
        apply_flux2_lora(&mut base, &lora, 1.0).unwrap();
        let (w, _) = base.get("proj.weight").unwrap();
        assert!((w[0] - 16.0).abs() < 1e-5);
        assert!((w[1] - 28.0).abs() < 1e-5);
    }
}
