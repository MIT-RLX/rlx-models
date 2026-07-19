// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Metal compile / run helpers for Zonos.
//!
//! F16 Linear weights keep the activation arena under the 4 GiB MPSGraph cliff.
//! Schedule-split hybrid is **off by default** for Zonos (measured slower than
//! all-thunk F16 on fox); set `RLX_ZONOS_MPSGRAPH_HYBRID=1` to opt in.
//! `RLX_ZONOS_DISABLE_MPSGRAPH=1` forces all-thunk Metal (also disables hybrid).

use rlx_runtime::Device;
use std::cell::Cell;

thread_local! {
    static METAL_GUARD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

fn mpsgraph_force_off() -> bool {
    matches!(
        std::env::var("RLX_ZONOS_DISABLE_MPSGRAPH").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn hybrid_opt_in() -> bool {
    matches!(
        std::env::var("RLX_ZONOS_MPSGRAPH_HYBRID").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Optionally disable MPSGraph / hybrid for the duration of `f`.
///
/// Defaults: MPSGraph hybrid **off** (all-thunk F16 is faster for Zonos decode).
/// Prefer MPSMatrix for remaining thunk matmuls (`m=2` Linears) —
/// override with `RLX_METAL_SGEMM_MPS=0`.
pub fn metal_compile_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device != Device::Metal {
        return f();
    }
    METAL_GUARD_DEPTH.with(|depth| {
        let enter = depth.get() == 0;
        if enter {
            if mpsgraph_force_off() {
                rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
            } else if !hybrid_opt_in() {
                // Hybrid is enabled upstream when whole-graph MPS fails; for
                // Zonos it regresses vs all-thunk F16 (~114s vs ~76s fox).
                rlx_ir::env::set("RLX_DISABLE_MPSGRAPH_HYBRID", "1");
            }
            if rlx_ir::env::var("RLX_METAL_SGEMM_MPS").is_none() {
                rlx_ir::env::set("RLX_METAL_SGEMM_MPS", "1");
            }
            if rlx_ir::env::var("RLX_MPS_THRESHOLD_FLOP").is_none() {
                // CFG decode m=2,k=2048,n=2048 → ~8M MACs; default 16M skipped MPS.
                rlx_ir::env::set("RLX_MPS_THRESHOLD_FLOP", "1000000");
            }
        }
        depth.set(depth.get() + 1);
        let out = f();
        let next = depth.get().saturating_sub(1);
        depth.set(next);
        if next == 0 {
            if mpsgraph_force_off() {
                rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
            } else if !hybrid_opt_in() {
                rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH_HYBRID");
            }
        }
        out
    })
}

/// No-op run wrapper (compile-time guard is the control surface).
pub fn metal_run_guard<R, F>(_device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}
