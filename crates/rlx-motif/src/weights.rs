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

//! Checkpoint adaptation — two rewrites, both idempotent.
//!
//! **1. PolyNorm coefficients.** Upstream stores `act_fn.weight` (`[3]` dense /
//! `[E, 3]` per expert) and `act_fn.bias` (`[1]` / `[E, 1]`), and applies
//! `σ(weight)` — plus, for the routed experts only, `clamp(bias, ±bias_clamp)` —
//! on every forward. Both are pure functions of parameters, so they are folded
//! here into one `act_fn.coeff` row `[w₀, w₁, w₂, b]`. That keeps the graph free
//! of parameter-only math and, more importantly, turns the per-expert
//! coefficients into a table the MoE can `Gather` by routed expert id (upstream
//! has to fall back to an eager Python loop over experts for exactly this
//! reason).
//!
//! **2. Expert bank layout.** `MotifExperts` stores `gate_up_proj` as
//! `[E, 2·inter, hidden]` and `down_proj` as `[E, hidden, inter]` (both `[out,
//! in]`, matching `x @ W.T`). [`rlx_ir::op::Op::GroupedMatMul`] wants `[E, K, N]`,
//! so both are transposed here rather than in-graph: a constant-folded in-graph
//! transpose keeps *both* copies of the bank resident, and this bank is the
//! whole model.
//!
//! **Memory**: Motif-3 is ~314 B parameters (629 GB of bf16 across 155 shards)
//! and [`WeightMap`] is f32, so materializing it this way is not something a
//! single host does. Real-weight inference needs the paged/packed expert path
//! (see `rlx_kimi_k3::moe`), which is not wired here; the builder is exercised
//! on scaled-down configs.

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;

use crate::config::MotifConfig;

/// Rewrite a stock Motif checkpoint in `wm` into the layout the graph reads.
///
/// Safe to call twice — layers already carrying the rewritten tensors are
/// skipped, so prefill and decode graphs can share one map.
pub fn prepare_checkpoint(cfg: &MotifConfig, wm: &mut WeightMap) -> Result<()> {
    for layer in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer}");
        if cfg.is_moe_layer(layer) {
            let moe = format!("{lp}.moe");
            let experts = format!("{moe}.experts");
            // One marker gates BOTH rewrites. Deciding "already transposed" from
            // the bank shape alone is ambiguous — `[E, 2·inter, hidden]` and
            // `[E, hidden, 2·inter]` are the same shape whenever
            // `2·inter == hidden`, and skipping on that false positive leaves the
            // banks in `[E, N, K]` order, where `GroupedMatMul` reads `n` from
            // the wrong axis and silently writes only part of its output.
            if !wm.has(&format!("{experts}.act_fn.coeff")) {
                fold_expert_coeffs(cfg, &experts, wm)?;
                transpose_expert_banks(cfg, &experts, wm)?;
            }
            if cfg.num_shared_experts > 0 {
                fold_mlp_coeffs(&format!("{moe}.shared_experts"), wm)?;
            }
        } else {
            fold_mlp_coeffs(&format!("{lp}.mlp"), wm)?;
        }
    }
    Ok(())
}

/// Drop the `model.mtp_layers.*` block. `modeling_motif.py` never instantiates
/// it (`num_nextn_predict_layers` is speculative-decoding metadata), so those
/// tensors are dead weight in a `WeightMap` that is already the binding
/// constraint. Returns how many tensors were dropped.
pub fn drop_mtp_layers(wm: &mut WeightMap) -> usize {
    let dead: Vec<String> = wm
        .keys()
        .filter(|k| k.starts_with("model.mtp_layers."))
        .map(str::to_string)
        .collect();
    let n = dead.len();
    for k in dead {
        let _ = wm.take(&k);
    }
    n
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `[E, 3]` + `[E, 1]` → `[E, 4]` = `[σ(w₀), σ(w₁), σ(w₂), clamp(b, ±bias_clamp)]`.
fn fold_expert_coeffs(cfg: &MotifConfig, experts: &str, wm: &mut WeightMap) -> Result<()> {
    let out = format!("{experts}.act_fn.coeff");
    if wm.has(&out) {
        return Ok(());
    }
    let e = cfg.num_experts;
    let w = take_checked(wm, &format!("{experts}.act_fn.weight"), &[e, 3])?;
    let b = take_checked(wm, &format!("{experts}.act_fn.bias"), &[e, 1])?;
    let mut coeff = vec![0f32; e * 4];
    for ei in 0..e {
        for j in 0..3 {
            coeff[ei * 4 + j] = if cfg.polynorm_sigmoid_weight {
                sigmoid(w[ei * 3 + j])
            } else {
                w[ei * 3 + j]
            };
        }
        coeff[ei * 4 + 3] = match cfg.polynorm_bias_clamp {
            Some(c) => b[ei].clamp(-c, c),
            None => b[ei],
        };
    }
    wm.insert(out, coeff, vec![e, 4]);
    Ok(())
}

/// `[3]` + `[1]` → `[1, 4]`. `PolyNormTorch` does **not** clamp its bias, so this
/// is not the expert fold with `E = 1`.
fn fold_mlp_coeffs(mlp: &str, wm: &mut WeightMap) -> Result<()> {
    let out = format!("{mlp}.act_fn.coeff");
    if wm.has(&out) {
        return Ok(());
    }
    let w = take_checked(wm, &format!("{mlp}.act_fn.weight"), &[3])?;
    let b = take_checked(wm, &format!("{mlp}.act_fn.bias"), &[1])?;
    let coeff = vec![sigmoid(w[0]), sigmoid(w[1]), sigmoid(w[2]), b[0]];
    wm.insert(out, coeff, vec![1, 4]);
    Ok(())
}

/// `[E, N, K] → [E, K, N]` for both expert banks. Callers gate this on the
/// `act_fn.coeff` marker — see [`prepare_checkpoint`] — because the shapes alone
/// cannot say whether the rewrite already ran.
fn transpose_expert_banks(cfg: &MotifConfig, experts: &str, wm: &mut WeightMap) -> Result<()> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size();
    let e = cfg.num_experts;

    let gu_key = format!("{experts}.gate_up_proj");
    let gu = take_checked(wm, &gu_key, &[e, 2 * inter, hidden])?;
    let gu = transpose_stack(&gu, e, 2 * inter, hidden);
    wm.insert(gu_key, gu, vec![e, hidden, 2 * inter]);

    let dn_key = format!("{experts}.down_proj");
    let dn = take_checked(wm, &dn_key, &[e, hidden, inter])?;
    let dn = transpose_stack(&dn, e, hidden, inter);
    wm.insert(dn_key, dn, vec![e, inter, hidden]);
    Ok(())
}

/// Transpose each `[n, k]` slab of an `[e, n, k]` stack into `[k, n]`.
fn transpose_stack(src: &[f32], e: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; e * n * k];
    for ei in 0..e {
        let s = &src[ei * n * k..(ei + 1) * n * k];
        let d = &mut out[ei * n * k..(ei + 1) * n * k];
        for r in 0..n {
            for c in 0..k {
                d[c * n + r] = s[r * k + c];
            }
        }
    }
    out
}

fn take_checked(wm: &mut WeightMap, key: &str, want: &[usize]) -> Result<Vec<f32>> {
    let (data, shape) = wm
        .take(key)
        .with_context(|| format!("Motif checkpoint is missing {key}"))?;
    if shape != want {
        anyhow::bail!("{key}: expected shape {want:?}, checkpoint has {shape:?}");
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg() -> MotifConfig {
        MotifConfig::from_json_str(
            r#"{"hidden_size":4,"moe_intermediate_size":3,"num_experts":2,
                "num_hidden_layers":2,"n_dense_first_layers":1,
                "interleave_moe_layer_step":1,"num_shared_experts":1,
                "experts_top_k":1,"polynorm_bias_clamp":0.5}"#,
        )
        .unwrap()
    }

    fn seeded(n: usize, base: f32) -> Vec<f32> {
        (0..n).map(|i| base + i as f32).collect()
    }

    #[test]
    fn folds_coefficients_and_transposes_banks() {
        let c = cfg();
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let ex = "model.layers.1.moe.experts";
        // [E=2, 2I=6, H=4] and [E=2, H=4, I=3], both [out, in].
        t.insert(
            format!("{ex}.gate_up_proj"),
            (seeded(48, 0.0), vec![2, 6, 4]),
        );
        t.insert(
            format!("{ex}.down_proj"),
            (seeded(24, 100.0), vec![2, 4, 3]),
        );
        t.insert(
            format!("{ex}.act_fn.weight"),
            (vec![0.0, 0.0, 0.0, 10.0, -10.0, 0.0], vec![2, 3]),
        );
        t.insert(format!("{ex}.act_fn.bias"), (vec![0.25, 9.0], vec![2, 1]));
        for m in ["model.layers.1.moe.shared_experts", "model.layers.0.mlp"] {
            t.insert(format!("{m}.act_fn.weight"), (vec![0.0; 3], vec![3]));
            t.insert(format!("{m}.act_fn.bias"), (vec![9.0], vec![1]));
        }
        let mut wm = WeightMap::from_tensors(t);
        prepare_checkpoint(&c, &mut wm).expect("prepare");

        // σ(0) = 0.5; the expert bias is clamped to ±0.5, the MLP bias is not.
        let (coeff, shape) = wm.get(&format!("{ex}.act_fn.coeff")).unwrap();
        assert_eq!(shape, &[2, 4]);
        assert!((coeff[0] - 0.5).abs() < 1e-6);
        assert!((coeff[3] - 0.25).abs() < 1e-6, "under the clamp, untouched");
        assert!(coeff[4] > 0.999, "σ(10)");
        assert!(coeff[5] < 0.001, "σ(-10)");
        assert!((coeff[7] - 0.5).abs() < 1e-6, "9.0 clamped to +0.5");
        let (mlp, mshape) = wm.get("model.layers.0.mlp.act_fn.coeff").unwrap();
        assert_eq!(mshape, &[1, 4]);
        assert_eq!(mlp[3], 9.0, "PolyNormTorch does not clamp its bias");

        // [E,2I,H] → [E,H,2I]: element (row r, col c) lands at (c, r).
        let (gu, gshape) = wm.get(&format!("{ex}.gate_up_proj")).unwrap();
        assert_eq!(gshape, &[2, 4, 6]);
        assert_eq!(gu[0], 0.0);
        assert_eq!(gu[1], 4.0, "next output row, same input col");
        assert_eq!(gu[6], 1.0, "next input col");
        assert_eq!(gu[24], 24.0, "expert 1 starts at the second slab");
        let (dn, dshape) = wm.get(&format!("{ex}.down_proj")).unwrap();
        assert_eq!(dshape, &[2, 3, 4]);
        assert_eq!(dn[0], 100.0);
        assert_eq!(dn[1], 103.0);
    }

    #[test]
    fn is_idempotent() {
        let c = cfg();
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let ex = "model.layers.1.moe.experts";
        t.insert(format!("{ex}.gate_up_proj"), (vec![0.0; 48], vec![2, 6, 4]));
        t.insert(format!("{ex}.down_proj"), (vec![0.0; 24], vec![2, 4, 3]));
        t.insert(format!("{ex}.act_fn.weight"), (vec![0.0; 6], vec![2, 3]));
        t.insert(format!("{ex}.act_fn.bias"), (vec![0.0; 2], vec![2, 1]));
        for m in ["model.layers.1.moe.shared_experts", "model.layers.0.mlp"] {
            t.insert(format!("{m}.act_fn.weight"), (vec![0.0; 3], vec![3]));
            t.insert(format!("{m}.act_fn.bias"), (vec![0.0], vec![1]));
        }
        let mut wm = WeightMap::from_tensors(t);
        prepare_checkpoint(&c, &mut wm).unwrap();
        prepare_checkpoint(&c, &mut wm).expect("second call must be a no-op");
    }

    /// `2·inter == hidden` makes `[E, 2·inter, hidden]` and `[E, hidden, 2·inter]`
    /// the same shape, so a shape-based "already transposed?" check silently skips
    /// the rewrite — and `GroupedMatMul`, which reads `n` off the bank's last axis,
    /// then writes only `inter/hidden` of each output row and leaves the rest as
    /// whatever the arena held. The `act_fn.coeff` marker is what makes this safe.
    #[test]
    fn transposes_even_when_the_bank_shape_is_ambiguous() {
        let c = MotifConfig::from_json_str(
            r#"{"hidden_size":4,"moe_intermediate_size":2,"num_experts":2,
                "num_hidden_layers":1,"n_dense_first_layers":0,
                "interleave_moe_layer_step":1,"num_shared_experts":0,
                "experts_top_k":1}"#,
        )
        .unwrap();
        let ex = "model.layers.0.moe.experts";
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        // [E=2, 2*inter=4, hidden=4] — square, hence ambiguous.
        t.insert(
            format!("{ex}.gate_up_proj"),
            (seeded(32, 0.0), vec![2, 4, 4]),
        );
        t.insert(
            format!("{ex}.down_proj"),
            (seeded(16, 100.0), vec![2, 4, 2]),
        );
        t.insert(format!("{ex}.act_fn.weight"), (vec![0.0; 6], vec![2, 3]));
        t.insert(format!("{ex}.act_fn.bias"), (vec![0.0; 2], vec![2, 1]));
        let mut wm = WeightMap::from_tensors(t);
        prepare_checkpoint(&c, &mut wm).expect("prepare");

        let (gu, _) = wm.get(&format!("{ex}.gate_up_proj")).unwrap();
        assert_eq!(gu[1], 4.0, "square bank must still be transposed");
        let (_, dshape) = wm.get(&format!("{ex}.down_proj")).unwrap();
        assert_eq!(dshape, &[2, 2, 4]);
        // …and running again must not transpose it back.
        prepare_checkpoint(&c, &mut wm).expect("second call");
        let (gu, gshape) = wm.get(&format!("{ex}.gate_up_proj")).unwrap();
        assert_eq!(gshape, &[2, 4, 4]);
        assert_eq!(
            gu[1], 4.0,
            "second call must be a no-op, not a re-transpose"
        );
    }

    #[test]
    fn reports_the_missing_tensor_by_name() {
        let c = cfg();
        let mut wm = WeightMap::from_tensors(HashMap::new());
        let err = prepare_checkpoint(&c, &mut wm).unwrap_err();
        assert!(
            format!("{err:#}").contains("model.layers.0.mlp.act_fn.weight"),
            "unhelpful error: {err:#}"
        );
    }

    #[test]
    fn drops_the_unused_mtp_block() {
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        t.insert(
            "model.mtp_layers.0.input_proj.weight".into(),
            (vec![0.0], vec![1]),
        );
        t.insert("model.norm.weight".into(), (vec![1.0], vec![1]));
        let mut wm = WeightMap::from_tensors(t);
        assert_eq!(drop_mtp_layers(&mut wm), 1);
        assert!(!wm.has("model.mtp_layers.0.input_proj.weight"));
        assert!(wm.has("model.norm.weight"));
    }
}
