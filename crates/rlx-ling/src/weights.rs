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

//! Checkpoint adaptation: Ling stores MoE experts one tensor per expert, the
//! shared MoE builder wants them stacked.
//!
//! Upstream (`model.layers.{i}.mlp.experts.{e}.…`):
//! ```text
//!   gate_proj.weight  [moe_inter, hidden]
//!   up_proj.weight    [moe_inter, hidden]
//!   down_proj.weight  [hidden, moe_inter]
//! ```
//! [`rlx_deepseek::moe::emit_deepseek_moe`] with
//! [`DeepseekMoeDims::experts_pretransposed`] wants them stacked *and* already in
//! `GroupedMatMul`'s `[E, K, N]` layout:
//! ```text
//!   experts.gate_up_proj  [E, hidden, 2*moe_inter]   (gate cols then up cols)
//!   experts.down_proj     [E, moe_inter, hidden]
//! ```
//! plus the router bias under DeepSeek's name. This module does all three
//! rewrites in place, so a stock HF Ling checkpoint feeds the shared builder
//! unchanged.
//!
//! Transposing here rather than in-graph is what makes real weights fit: a
//! constant-folded in-graph transpose keeps both copies of the expert bank
//! resident, 27.8 GB → 55.6 GB, which overruns Metal's 41.75 GB max buffer.
//!
//! **Memory**: stacking is still a copy, and `WeightMap` is f32, so the routed
//! experts of Ling-3.0-tiny are `23 · 128 · 3 · 512 · 1536 · 4 B ≈ 27.8 GB`
//! resident. [`pack_layer_experts`] is the way out — it MXFP4-packs one expert
//! at a time straight from the checkpoint, needs no transpose and no stacked
//! bank, and holds ~3 MB of f32 at a time instead of 1.2 GB per layer.

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;

use crate::config::LingConfig;

/// Router-bias key used by [`rlx_deepseek::moe::emit_deepseek_moe`].
const DS_ROUTER_BIAS: &str = "e_score_correction_bias";
/// Router-bias key used by Bailing checkpoints.
const LING_ROUTER_BIAS: &str = "expert_bias";

/// `(gate_up, down)` shapes of one layer's stacked expert banks, in
/// `GroupedMatMul`'s `[E, K, N]` layout.
pub(crate) fn stacked_bank_shapes(cfg: &LingConfig) -> (Vec<usize>, Vec<usize>) {
    let (h, i, e) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts);
    (vec![e, h, 2 * i], vec![e, i, h])
}

/// Param names of one layer's stacked expert banks.
pub(crate) fn expert_bank_keys(mlp: &str) -> (String, String) {
    (
        format!("{mlp}.experts.gate_up_proj"),
        format!("{mlp}.experts.down_proj"),
    )
}

/// Rewrite a Ling checkpoint in `wm` into the layout the graph builder reads.
///
/// Idempotent: layers already carrying the stacked tensors are skipped, so it is
/// safe to call twice (e.g. prefill then decode graph builds off one map).
pub fn prepare_checkpoint(cfg: &LingConfig, wm: &mut WeightMap) -> Result<()> {
    for layer in 0..cfg.num_hidden_layers {
        if !cfg.is_moe_layer(layer) {
            continue;
        }
        let mlp = format!("model.layers.{layer}.mlp");
        rename_router_bias(&mlp, wm)?;
        if wm.has(&format!("{mlp}.experts.gate_up_proj")) {
            continue;
        }
        stack_experts(cfg, &mlp, wm)?;
    }
    Ok(())
}

/// Move Ling's `expert_bias` to the router-bias name the shared MoE builder
/// reads. Public because the MXFP4 path skips [`prepare_checkpoint`] entirely
/// but still needs this one rename.
pub fn rename_router_bias_pub(mlp: &str, wm: &mut WeightMap) -> Result<()> {
    rename_router_bias(mlp, wm)
}

fn rename_router_bias(mlp: &str, wm: &mut WeightMap) -> Result<()> {
    let from = format!("{mlp}.gate.{LING_ROUTER_BIAS}");
    let to = format!("{mlp}.gate.{DS_ROUTER_BIAS}");
    if wm.has(&to) || !wm.has(&from) {
        return Ok(());
    }
    let (data, shape) = wm.take(&from)?;
    wm.insert(to, data, shape);
    Ok(())
}

/// Stack + transpose one layer's experts from any per-tensor source.
///
/// Returns `(gate_up [E, hidden, 2*inter], down [E, inter, hidden])`. Transposing
/// here rather than in-graph is deliberate: a constant-folded in-graph transpose
/// keeps BOTH copies of the expert bank resident (27.8 GB → 55.6 GB for
/// Ling-3.0-tiny, which overruns Metal's max buffer). Paired with
/// `DeepseekMoeDims::experts_pretransposed`.
pub(crate) fn stack_layer_from<F>(
    cfg: &LingConfig,
    mlp: &str,
    fetch: F,
) -> Result<(Vec<f32>, Vec<f32>)>
where
    F: FnMut(&str, &[usize]) -> Result<Vec<f32>>,
{
    let (h, i, e) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts);
    let mut gate_up = vec![0f32; e * h * 2 * i];
    let mut down = vec![0f32; e * i * h];
    stack_layer_into(cfg, mlp, fetch, &mut gate_up, &mut down)?;
    Ok((gate_up, down))
}

/// As [`stack_layer_from`] but writes into caller-owned buffers.
///
/// Streaming 23 layers with fresh 1.2 GB allocations each made RSS climb ~0.55 GB
/// per layer — the allocator does not return freed blocks that size promptly.
/// Reusing two buffers keeps the streaming peak flat.
pub(crate) fn stack_layer_into<F>(
    cfg: &LingConfig,
    mlp: &str,
    mut fetch: F,
    gate_up: &mut [f32],
    down: &mut [f32],
) -> Result<()>
where
    F: FnMut(&str, &[usize]) -> Result<Vec<f32>>,
{
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let e = cfg.num_experts;
    debug_assert_eq!(gate_up.len(), e * hidden * 2 * inter);
    debug_assert_eq!(down.len(), e * inter * hidden);
    for ei in 0..e {
        let base = format!("{mlp}.experts.{ei}");
        let g = fetch(&format!("{base}.gate_proj.weight"), &[inter, hidden])?;
        let u = fetch(&format!("{base}.up_proj.weight"), &[inter, hidden])?;
        let d = fetch(&format!("{base}.down_proj.weight"), &[hidden, inter])?;
        // gate rows then up rows, transposed: [2*inter, hidden] → [hidden, 2*inter]
        let gu = &mut gate_up[ei * hidden * 2 * inter..(ei + 1) * hidden * 2 * inter];
        for row in 0..inter {
            for col in 0..hidden {
                gu[col * 2 * inter + row] = g[row * hidden + col];
                gu[col * 2 * inter + inter + row] = u[row * hidden + col];
            }
        }
        // down: [hidden, inter] → [inter, hidden]
        let dn = &mut down[ei * inter * hidden..(ei + 1) * inter * hidden];
        for row in 0..hidden {
            for col in 0..inter {
                dn[col * hidden + row] = d[row * inter + col];
            }
        }
    }
    Ok(())
}

/// One MoE layer's expert banks, MXFP4-packed and ready for `set_param_typed`.
///
/// Shapes follow the packed op's `[E, N, K]` (no transpose anywhere):
/// `gate_up` is `[E, 2*inter, hidden]`, `down` is `[E, hidden, inter]`.
pub struct PackedBanks {
    /// `[E, 2*inter, hidden]` E2M1 nibbles, two per byte.
    pub gate_up_codes: Vec<u8>,
    /// `[E, 2*inter, hidden/group]` bf16 scales (the grouped op's convention).
    pub gate_up_scales: Vec<u8>,
    /// `[E, hidden, inter]` E2M1 nibbles.
    pub down_codes: Vec<u8>,
    /// `[E, hidden, inter/group]` bf16 scales.
    pub down_scales: Vec<u8>,
}

impl PackedBanks {
    /// Total packed bytes for this layer (codes + scales).
    pub fn bytes(&self) -> usize {
        self.gate_up_codes.len()
            + self.gate_up_scales.len()
            + self.down_codes.len()
            + self.down_scales.len()
    }
}

/// Read + MXFP4-pack one layer's experts, **one expert at a time**.
///
/// The f32 path has to stage the whole `[E, …]` bank (1.2 GB/layer) because
/// stacking interleaves a transpose across experts. MXFP4 needs no transpose, so
/// expert `e`'s rows are exactly its own `gate_proj`/`up_proj`/`down_proj`
/// tensors — each can be quantized and appended immediately, and the f32 high
/// water mark is one expert (~3 MB) instead of 1.2 GB. Output is ~169 MB/layer.
pub fn pack_layer_experts<F>(
    cfg: &LingConfig,
    mlp: &str,
    mut fetch: F,
    group_size: usize,
) -> Result<PackedBanks>
where
    F: FnMut(&str, &[usize]) -> Result<Vec<f32>>,
{
    use rlx_core::mxfp4_pack::quantize_rows;
    let (h, i, e) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts);
    anyhow::ensure!(
        h.is_multiple_of(group_size) && i.is_multiple_of(group_size),
        "MXFP4 group size {group_size} does not divide hidden={h} / inter={i}"
    );
    let mut out = PackedBanks {
        gate_up_codes: Vec::with_capacity(e * 2 * i * (h / 2)),
        gate_up_scales: Vec::with_capacity(e * 2 * i * (h / group_size) * 2),
        down_codes: Vec::with_capacity(e * h * (i / 2)),
        down_scales: Vec::with_capacity(e * h * (i / group_size) * 2),
    };
    for ei in 0..e {
        let base = format!("{mlp}.experts.{ei}");
        // `gate_up` rows are gate-then-up, matching the `narrow_` split the
        // emitter does on the op's output.
        for name in ["gate_proj", "up_proj"] {
            let w = fetch(&format!("{base}.{name}.weight"), &[i, h])?;
            let q = quantize_rows(&w, i, h, group_size);
            drop(w);
            out.gate_up_codes.extend_from_slice(&q.codes);
            out.gate_up_scales.extend_from_slice(&q.scales_bf16());
        }
        let d = fetch(&format!("{base}.down_proj.weight"), &[h, i])?;
        let q = quantize_rows(&d, h, i, group_size);
        drop(d);
        out.down_codes.extend_from_slice(&q.codes);
        out.down_scales.extend_from_slice(&q.scales_bf16());
    }
    Ok(out)
}

/// Upload one layer's packed banks into a compiled graph.
///
/// `biases` is the MXFP4 zero-point slot — always zero (the scheme is
/// symmetric), but the op still reads five operands, so the param must exist.
pub fn upload_packed_banks(
    compiled: &mut rlx_runtime::CompiledGraph,
    mlp: &str,
    banks: &PackedBanks,
) {
    use rlx_deepseek::moe::packed_bank_keys;
    use rlx_ir::DType;
    for (stem, codes, scales) in [
        (
            format!("{mlp}.experts.gate_up"),
            &banks.gate_up_codes,
            &banks.gate_up_scales,
        ),
        (
            format!("{mlp}.experts.down"),
            &banks.down_codes,
            &banks.down_scales,
        ),
    ] {
        let (c_key, s_key, b_key) = packed_bank_keys(&stem);
        compiled.set_param_typed(&c_key, codes, DType::U8);
        compiled.set_param_typed(&s_key, scales, DType::BF16);
        compiled.set_param_typed(&b_key, &vec![0u8; scales.len()], DType::BF16);
    }
}

fn stack_experts(cfg: &LingConfig, mlp: &str, wm: &mut WeightMap) -> Result<()> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let e = cfg.num_experts;
    let (gate_up, down) = stack_layer_from(cfg, mlp, |key, want| take_checked(wm, key, want))?;
    wm.insert(
        format!("{mlp}.experts.gate_up_proj"),
        gate_up,
        vec![e, hidden, 2 * inter],
    );
    wm.insert(
        format!("{mlp}.experts.down_proj"),
        down,
        vec![e, inter, hidden],
    );
    Ok(())
}

fn take_checked(wm: &mut WeightMap, key: &str, want: &[usize]) -> Result<Vec<f32>> {
    let (data, shape) = wm
        .take(key)
        .with_context(|| format!("Ling checkpoint is missing {key}"))?;
    if shape != want {
        anyhow::bail!("{key}: expected shape {want:?}, checkpoint has {shape:?}");
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg() -> LingConfig {
        LingConfig::from_json_str(
            r#"{"hidden_size":4,"moe_intermediate_size":3,"num_experts":2,
                "num_hidden_layers":2,"first_k_dense_replace":1,"n_group":2,
                "topk_group":1,"num_experts_per_tok":1}"#,
        )
        .unwrap()
    }

    /// Per-expert tensors are stacked expert-major AND transposed into
    /// `GroupedMatMul`'s `[E, K, N]` layout, with gate columns before up columns.
    #[test]
    fn stacks_experts_in_expert_major_order() {
        let c = cfg();
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for ei in 0..2 {
            let b = format!("model.layers.1.mlp.experts.{ei}");
            let tag = (ei as f32) * 100.0;
            t.insert(
                format!("{b}.gate_proj.weight"),
                ((0..12).map(|i| tag + i as f32).collect(), vec![3, 4]),
            );
            t.insert(
                format!("{b}.up_proj.weight"),
                ((0..12).map(|i| tag + 20.0 + i as f32).collect(), vec![3, 4]),
            );
            t.insert(
                format!("{b}.down_proj.weight"),
                ((0..12).map(|i| tag + 40.0 + i as f32).collect(), vec![4, 3]),
            );
        }
        t.insert(
            "model.layers.1.mlp.gate.expert_bias".into(),
            (vec![0.5, -0.5], vec![2]),
        );
        let mut wm = WeightMap::from_tensors(t);
        prepare_checkpoint(&c, &mut wm).expect("prepare");

        // [E=2, hidden=4, 2*inter=6]; source gate/up are [inter=3, hidden=4].
        let (gu, shape) = wm.get("model.layers.1.mlp.experts.gate_up_proj").unwrap();
        assert_eq!(shape, &[2, 4, 6]);
        assert_eq!(gu[0], 0.0, "expert 0, hidden col 0, gate row 0");
        assert_eq!(
            gu[3], 20.0,
            "up columns follow the gate columns within a row"
        );
        assert_eq!(gu[6], 1.0, "hidden col 1 → next row of the transposed bank");
        assert_eq!(gu[24], 100.0, "expert 1 starts at the second slab (4*6)");
        // [E=2, inter=3, hidden=4]; source down is [hidden=4, inter=3].
        let (dn, dshape) = wm.get("model.layers.1.mlp.experts.down_proj").unwrap();
        assert_eq!(dshape, &[2, 3, 4]);
        assert_eq!(dn[0], 40.0);
        assert_eq!(dn[1], 43.0, "transposed: next hidden row, same inter col");

        // Router bias moved to the name the shared builder reads.
        assert!(wm.has("model.layers.1.mlp.gate.e_score_correction_bias"));
        assert!(!wm.has("model.layers.1.mlp.gate.expert_bias"));
        // Dense layer 0 untouched.
        assert!(!wm.has("model.layers.0.mlp.experts.gate_up_proj"));
    }

    #[test]
    fn is_idempotent() {
        let c = cfg();
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for ei in 0..2 {
            let b = format!("model.layers.1.mlp.experts.{ei}");
            t.insert(format!("{b}.gate_proj.weight"), (vec![0.0; 12], vec![3, 4]));
            t.insert(format!("{b}.up_proj.weight"), (vec![0.0; 12], vec![3, 4]));
            t.insert(format!("{b}.down_proj.weight"), (vec![0.0; 12], vec![4, 3]));
        }
        t.insert(
            "model.layers.1.mlp.gate.expert_bias".into(),
            (vec![0.0; 2], vec![2]),
        );
        let mut wm = WeightMap::from_tensors(t);
        prepare_checkpoint(&c, &mut wm).unwrap();
        prepare_checkpoint(&c, &mut wm).expect("second call must be a no-op");
    }

    #[test]
    fn reports_the_missing_tensor_by_name() {
        let c = cfg();
        let mut wm = WeightMap::from_tensors(HashMap::new());
        let err = prepare_checkpoint(&c, &mut wm).unwrap_err();
        assert!(
            format!("{err:#}").contains("experts.0.gate_proj.weight"),
            "unhelpful error: {err:#}"
        );
    }
}
