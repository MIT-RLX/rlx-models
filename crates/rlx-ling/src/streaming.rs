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

//! Deferred expert loading: keep the 27.8 GB of routed experts out of the build
//! and upload them one layer at a time after the arena exists.
//!
//! How it works:
//!
//! 1. Load the checkpoint *without* per-expert tensors (~3.8 GB instead of 32 GB).
//! 2. Give the builder zero-filled placeholders for the two banks per MoE layer.
//!    `vec![0f32; n]` goes through `alloc_zeroed`, so those pages stay virtual.
//! 3. Drop the placeholders from the param map before compiling — they were never
//!    written, so this costs nothing and stops `attach_built_params` from
//!    faulting in 27.8 GB of zeros.
//! 4. Stream the banks in per layer: read that layer's experts, stack + transpose
//!    into two reused buffers (~1.2 GB), `set_param`, next.
//!
//! ## What this does and does not buy you
//!
//! Measured on Ling-3.0-tiny (M4 Pro, seq 64, `RSS current (peak)`):
//!
//! ```text
//!            eager                          deferred
//!   load     21.9 GB (peak 32.4) in 9.9s    5.7 GB (peak 5.8) in 0.9s
//!   build    31.7 GB, 31.6 GB of params     3.8 GB of params reach the compiler
//!   compile  arena on top of live params    arena filled by streaming
//!   peak     ~48.6 GB                       ~49.3 GB
//! ```
//!
//! **The peak is unchanged.** It is not the params/arena overlap — it is the
//! arena itself: on CPU it starts at 6.3 GB and climbs linearly to ~48 GB as the
//! f32 experts fault in. Deferring changes *when* those pages become resident,
//! not how many there are.
//!
//! What it does buy: an 11× faster load, a 5.7 GB rather than 32 GB load-phase
//! footprint, and a deterministic 31.6 → 3.8 GB cut in what the compiler holds.
//! On a machine where the *load* spike is what kills you, that is the difference;
//! on one where the steady footprint is, it is not.
//!
//! Genuinely fixing the peak needs the weights to stop being f32 — that is what
//! [`load_and_compile_mxfp4`] does. It is not a variant of this path but a
//! simpler one: the packed op reads the stock `[E, N, K]` orientation, so there
//! is no transpose, no stacked f32 bank, and no placeholder-then-drop dance, and
//! packing runs one expert at a time so the f32 high-water mark is ~3 MB rather
//! than the 1.2 GB/layer staged here. Arena 29.5 → ~4.0 GiB.

use anyhow::{Context, Result};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashSet;
use std::path::Path;

use crate::config::LingConfig;
use crate::flow::{build_ling_text_flow, build_ling_text_flow_plan};
use crate::quant::{Quant, QuantPlan};
use crate::weights::{
    expert_bank_keys, pack_layer_experts, stack_layer_into, stacked_bank_shapes,
    upload_packed_banks,
};

/// True for the per-expert tensors this path defers (`…mlp.experts.<n>.…`).
fn is_per_expert_tensor(name: &str) -> bool {
    let Some(rest) = name.split_once(".mlp.experts.").map(|(_, r)| r) else {
        return false;
    };
    rest.split('.')
        .next()
        .is_some_and(|seg| !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()))
}

/// Load every tensor **except** the per-expert MoE weights.
pub fn load_without_experts(dir: &Path) -> Result<(SafetensorsCheckpoint, WeightMap)> {
    let ckpt = SafetensorsCheckpoint::open(dir)
        .with_context(|| format!("open safetensors checkpoint at {dir:?}"))?;
    let want: HashSet<String> = ckpt
        .keys()
        .filter(|k| !is_per_expert_tensor(k))
        .map(|k| k.to_string())
        .collect();
    let wm = ckpt.load_selected(&want)?;
    Ok((ckpt, wm))
}

/// Insert zero-filled placeholders for each MoE layer's two expert banks, and
/// move the router bias to the name the shared builder reads.
///
/// The placeholders exist only so the graph gets correctly-shaped param nodes;
/// [`compile_deferred`] drops them before they are ever read.
pub fn install_expert_placeholders(cfg: &LingConfig, wm: &mut WeightMap) -> Result<()> {
    for layer in 0..cfg.num_hidden_layers {
        if !cfg.is_moe_layer(layer) {
            continue;
        }
        let mlp = format!("model.layers.{layer}.mlp");
        crate::weights::rename_router_bias_pub(&mlp, wm)?;
        let (gu_shape, dn_shape) = stacked_bank_shapes(cfg);
        let (gu_key, dn_key) = expert_bank_keys(&mlp);
        wm.insert(gu_key, vec![0f32; gu_shape.iter().product()], gu_shape);
        wm.insert(dn_key, vec![0f32; dn_shape.iter().product()], dn_shape);
    }
    Ok(())
}

/// Build + compile with the expert banks deferred, then stream them in.
///
/// `wm` must have come from [`load_without_experts`] and been through
/// [`install_expert_placeholders`].
pub fn compile_deferred(
    cfg: &LingConfig,
    ckpt: &SafetensorsCheckpoint,
    wm: &mut WeightMap,
    seq: usize,
    device: Device,
    with_lm_head: bool,
    mut on_progress: impl FnMut(&str),
) -> Result<CompiledGraph> {
    let built = build_ling_text_flow(cfg, wm, seq, with_lm_head)?;
    let profile = built.profile().clone();
    let typed = built.typed_params.clone();
    let (graph, mut params) = built.into_graph_parts()?;

    // Drop the placeholders *before* the arena is allocated. They were never
    // written to, so these pages were never resident — this just stops
    // `attach_built_params` from touching (and thus faulting in) 27.8 GB of zeros.
    let deferred: Vec<String> = params
        .keys()
        .filter(|k| k.ends_with(".experts.gate_up_proj") || k.ends_with(".experts.down_proj"))
        .cloned()
        .collect();
    for k in &deferred {
        params.remove(k);
    }
    on_progress(&format!(
        "deferred {} expert banks; {:.1} GB of params go into the arena eagerly",
        deferred.len(),
        params.values().map(|v| v.len()).sum::<usize>() as f64 * 4.0 / 1e9
    ));

    let options = rlx_core::flow_bridge::compile_options_for_profile(&profile, device);
    let mut compiled = Session::new(device).compile_with(graph, &options);
    rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
    on_progress("arena compiled, non-expert params uploaded");

    stream_expert_banks(cfg, ckpt, &mut compiled, &mut on_progress)?;
    Ok(compiled)
}

/// Read, stack and upload the expert banks one layer at a time.
pub fn stream_expert_banks(
    cfg: &LingConfig,
    ckpt: &SafetensorsCheckpoint,
    compiled: &mut CompiledGraph,
    mut on_progress: impl FnMut(&str),
) -> Result<()> {
    let moe_layers: Vec<usize> = (0..cfg.num_hidden_layers)
        .filter(|&i| cfg.is_moe_layer(i))
        .collect();
    // Two reused buffers rather than a fresh 1.2 GB pair per layer. Note this
    // does NOT flatten the RSS climb during streaming — measured, it is
    // unchanged. The climb is the arena's own pages faulting in as the experts
    // land in it, which is real memory, not allocator churn. Reuse is kept
    // because 46 large short-lived allocations are pointless either way.
    let (gu_shape, dn_shape) = stacked_bank_shapes(cfg);
    let mut gate_up = vec![0f32; gu_shape.iter().product()];
    let mut down = vec![0f32; dn_shape.iter().product()];
    for (n, &layer) in moe_layers.iter().enumerate() {
        let mlp = format!("model.layers.{layer}.mlp");
        stack_layer_into(
            cfg,
            &mlp,
            |key, want| {
                let (data, shape) = ckpt
                    .load_tensor_f32(key)
                    .with_context(|| format!("Ling checkpoint is missing {key}"))?;
                if shape != want {
                    anyhow::bail!("{key}: expected shape {want:?}, checkpoint has {shape:?}");
                }
                Ok(data)
            },
            &mut gate_up,
            &mut down,
        )?;
        let (gu_key, dn_key) = expert_bank_keys(&mlp);
        compiled.set_param(&gu_key, &gate_up);
        compiled.set_param(&dn_key, &down);
        if n == 0 || (n + 1) % 8 == 0 || n + 1 == moe_layers.len() {
            on_progress(&format!(
                "streamed expert banks {}/{}",
                n + 1,
                moe_layers.len()
            ));
        }
    }
    Ok(())
}

/// One-call convenience: open `dir`, build for `seq`, compile on `device` with
/// the low-peak path.
pub fn load_and_compile(
    cfg: &LingConfig,
    dir: &Path,
    seq: usize,
    device: Device,
    with_lm_head: bool,
    on_progress: impl FnMut(&str),
) -> Result<CompiledGraph> {
    let (ckpt, mut wm) = load_without_experts(dir)?;
    install_expert_placeholders(cfg, &mut wm)?;
    compile_deferred(cfg, &ckpt, &mut wm, seq, device, with_lm_head, on_progress)
}

/// MXFP4 sibling of [`load_and_compile`] — the whole model packed to 4 bits.
///
/// Simpler than the f32 path, not just smaller: the packed op reads the stock
/// `[E, N, K]` orientation, so there is no transpose, no stacked f32 bank, and
/// no placeholder-then-drop dance. The banks are declared by name in the graph
/// and streamed in per layer, one expert at a time
/// ([`crate::weights::pack_layer_experts`]), so the f32 high water mark is a
/// single expert.
///
/// Arena: ~4.0 GiB instead of 29.5 GiB, which is what puts Ling-3.0-tiny on a
/// 16 GB card.
pub fn load_and_compile_mxfp4(
    cfg: &LingConfig,
    dir: &Path,
    seq: usize,
    device: Device,
    with_lm_head: bool,
    on_progress: impl FnMut(&str),
) -> Result<CompiledGraph> {
    load_and_compile_plan(
        cfg,
        dir,
        seq,
        device,
        with_lm_head,
        QuantPlan::mxfp4_all(),
        on_progress,
    )
}

/// [`load_and_compile_mxfp4`] with the LM head's precision chosen separately.
pub fn load_and_compile_plan(
    cfg: &LingConfig,
    dir: &Path,
    seq: usize,
    device: Device,
    with_lm_head: bool,
    plan: QuantPlan,
    mut on_progress: impl FnMut(&str),
) -> Result<CompiledGraph> {
    let (ckpt, mut wm) = load_without_experts(dir)?;
    for layer in 0..cfg.num_hidden_layers {
        if cfg.is_moe_layer(layer) {
            crate::weights::rename_router_bias_pub(&format!("model.layers.{layer}.mlp"), &mut wm)?;
        }
    }
    let built = build_ling_text_flow_plan(cfg, &mut wm, seq, with_lm_head, plan)?;
    let profile = built.profile().clone();
    let typed = built.typed_params.clone();
    let (graph, params) = built.into_graph_parts()?;
    on_progress(&format!(
        "MXFP4 build: {:.2} GB dense params + {:.2} GB packed (non-expert)",
        params.values().map(|v| v.len()).sum::<usize>() as f64 * 4.0 / 1e9,
        typed.iter().map(|(_, b, _)| b.len()).sum::<usize>() as f64 / 1e9,
    ));

    let options = rlx_core::flow_bridge::compile_options_for_profile(&profile, device);
    let mut compiled = Session::new(device).compile_with(graph, &options);
    rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
    on_progress("arena compiled, non-expert params uploaded");

    stream_expert_banks_mxfp4(cfg, &ckpt, &mut compiled, on_progress)?;
    Ok(compiled)
}

/// Pack and upload the expert banks one layer at a time (MXFP4).
pub fn stream_expert_banks_mxfp4(
    cfg: &LingConfig,
    ckpt: &SafetensorsCheckpoint,
    compiled: &mut CompiledGraph,
    mut on_progress: impl FnMut(&str),
) -> Result<()> {
    let moe_layers: Vec<usize> = (0..cfg.num_hidden_layers)
        .filter(|&i| cfg.is_moe_layer(i))
        .collect();
    let gs = Quant::MXFP4.group_size().expect("MXFP4 has a group size");
    let mut total = 0usize;
    for (n, &layer) in moe_layers.iter().enumerate() {
        let mlp = format!("model.layers.{layer}.mlp");
        let banks = pack_layer_experts(
            cfg,
            &mlp,
            |key, want| {
                let (data, shape) = ckpt
                    .load_tensor_f32(key)
                    .with_context(|| format!("Ling checkpoint is missing {key}"))?;
                if shape != want {
                    anyhow::bail!("{key}: expected shape {want:?}, checkpoint has {shape:?}");
                }
                Ok(data)
            },
            gs,
        )?;
        total += banks.bytes();
        upload_packed_banks(compiled, &mlp, &banks);
        if n == 0 || (n + 1) % 8 == 0 || n + 1 == moe_layers.len() {
            on_progress(&format!(
                "packed expert banks {}/{} ({:.2} GB MXFP4 so far)",
                n + 1,
                moe_layers.len(),
                total as f64 / 1e9
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_per_expert_tensor_names() {
        assert!(is_per_expert_tensor(
            "model.layers.3.mlp.experts.17.gate_proj.weight"
        ));
        assert!(is_per_expert_tensor(
            "model.layers.0.mlp.experts.0.up_proj.weight"
        ));
        // The stacked banks and the shared expert must NOT be deferred by name.
        assert!(!is_per_expert_tensor(
            "model.layers.3.mlp.experts.gate_up_proj"
        ));
        assert!(!is_per_expert_tensor(
            "model.layers.3.mlp.shared_experts.up_proj.weight"
        ));
        assert!(!is_per_expert_tensor(
            "model.layers.3.attention.q_proj.weight"
        ));
        assert!(!is_per_expert_tensor("lm_head.weight"));
    }
}
