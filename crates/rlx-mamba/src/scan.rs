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

//! Mamba1 SSM execution via [`rlx_ssm`] flows compiled with [`rlx_runtime`].
//!
//! * Prefill: [`selective_scan_flow`] / [`selective_scan_on_device`] — [`rlx_ssm::MambaScanStage`].
//! * Decode: [`selective_scan_step_flow`] / [`selective_scan_step_on_device`] — [`rlx_ssm::Mamba1StepStage`].
//!
//! Graphs are cached per `(device, shape, weight fingerprint)`; CUDA/wgpu use native
//! `Op::SelectiveScan` when the corresponding `rlx-runtime` feature is enabled.

use anyhow::{Context, Result, ensure};
use rlx_flow::BuiltModel;
use rlx_flow::MapWeights;
use rlx_flow::prelude::ModelFlow;
use rlx_ir::{DType, Shape};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use rlx_ssm::{Mamba1StepStage, MambaScanStage, MambaScanWeightKeys};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, Once, OnceLock};

const SCAN_WEIGHT_PREFIX: &str = "mamba.scan";

/// Register SSM custom ops once (no-op after first call).
pub fn ensure_ssm_ops_registered() {
    static ONCE: Once = Once::new();
    ONCE.call_once(rlx_ssm::register_ir_ops);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ScanShapeKey {
    batch: usize,
    seq: usize,
    hidden: usize,
    state: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StepShapeKey {
    batch: usize,
    hidden: usize,
    state: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PrefillCacheKey {
    device: Device,
    shape: ScanShapeKey,
    weights: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StepCacheKey {
    device: Device,
    shape: StepShapeKey,
    weights: u64,
}

fn weight_tag(a_log: &[f32], d: &[f32]) -> u64 {
    let mut h = DefaultHasher::new();
    a_log.len().hash(&mut h);
    d.len().hash(&mut h);
    for v in a_log.iter().take(64) {
        v.to_bits().hash(&mut h);
    }
    for v in d {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

fn prefill_cache() -> &'static Mutex<HashMap<PrefillCacheKey, CompiledGraph>> {
    static CACHE: OnceLock<Mutex<HashMap<PrefillCacheKey, CompiledGraph>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn step_cache() -> &'static Mutex<HashMap<StepCacheKey, CompiledGraph>> {
    static CACHE: OnceLock<Mutex<HashMap<StepCacheKey, CompiledGraph>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Pick a runtime device that can execute `Op::SelectiveScan` / `mamba1_step`.
pub fn effective_scan_device(preferred: Device) -> Device {
    match preferred {
        #[cfg(feature = "cuda")]
        Device::Cuda if rlx_cuda::is_available() => Device::Cuda,
        #[cfg(feature = "wgpu")]
        Device::Gpu if rlx_wgpu::is_available() => Device::Gpu,
        #[cfg(feature = "rocm")]
        Device::Rocm if rlx_rocm::is_available() => Device::Rocm,
        // Metal/MLX run the SSM reference path on CPU until native scan ships there.
        _ => Device::Cpu,
    }
}

/// Prefill scan on CPU (`MambaScanStage` → `SelectiveScan`).
pub fn selective_scan_flow(
    batch: usize,
    seq: usize,
    hidden: usize,
    state: usize,
    x: &[f32],
    dt_raw: &[f32],
    b: &[f32],
    c: &[f32],
    a_log: &[f32],
    d: &[f32],
) -> Result<Vec<f32>> {
    selective_scan_on_device(
        Device::Cpu,
        batch,
        seq,
        hidden,
        state,
        x,
        dt_raw,
        b,
        c,
        a_log,
        d,
    )
}

/// Prefill scan on the given RLX device (CUDA/wgpu when available, else CPU).
pub fn selective_scan_on_device(
    device: Device,
    batch: usize,
    seq: usize,
    hidden: usize,
    state: usize,
    x: &[f32],
    dt_raw: &[f32],
    b: &[f32],
    c: &[f32],
    a_log: &[f32],
    d: &[f32],
) -> Result<Vec<f32>> {
    let bs = batch * seq;
    ensure!(x.len() == bs * hidden, "x length");
    ensure!(dt_raw.len() == bs * hidden, "dt_raw length");
    ensure!(b.len() == bs * state, "b length");
    ensure!(c.len() == bs * state, "c length");
    ensure!(a_log.len() == hidden * state, "a_log length");
    ensure!(d.len() == hidden, "d length");

    ensure_ssm_ops_registered();
    let device = effective_scan_device(device);
    let key = PrefillCacheKey {
        device,
        shape: ScanShapeKey {
            batch,
            seq,
            hidden,
            state,
        },
        weights: weight_tag(a_log, d),
    };

    let mut cache = prefill_cache().lock().expect("prefill cache");
    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(key) {
        let compiled = compile_scan_graph(device, batch, seq, hidden, state, a_log, d)?;
        e.insert(compiled);
    }
    let compiled = cache.get_mut(&key).expect("prefill cache entry");
    let outs = compiled.run(&[("x", x), ("dt_raw", dt_raw), ("b", b), ("c", c)]);
    let y = outs
        .into_iter()
        .next()
        .context("MambaScanStage flow must produce one output")?;
    ensure!(y.len() == bs * hidden, "scan output length");
    Ok(y)
}

fn compile_scan_graph(
    device: Device,
    batch: usize,
    seq: usize,
    hidden: usize,
    state: usize,
    a_log: &[f32],
    d: &[f32],
) -> Result<CompiledGraph> {
    let x_shape = Shape::new(&[batch, seq, hidden], DType::F32);
    let n_shape = Shape::new(&[batch, seq, state], DType::F32);

    let mut weights = MapWeights::default();
    weights.insert(
        format!("{SCAN_WEIGHT_PREFIX}.A_log"),
        a_log.to_vec(),
        vec![hidden, state],
    );
    weights.insert(format!("{SCAN_WEIGHT_PREFIX}.D"), d.to_vec(), vec![hidden]);

    let scan_plugin = MambaScanStage::new(
        MambaScanWeightKeys::hf(SCAN_WEIGHT_PREFIX),
        state,
        x_shape.clone(),
    )
    .plugin();

    let bind = move |emit: &mut rlx_flow::escape::Emit<'_>,
                     input: Option<rlx_flow::FlowValue>|
          -> Result<Option<rlx_flow::FlowValue>> {
        let dt_id = emit
            .state
            .inputs
            .get("dt_raw")
            .map(|(id, _)| *id)
            .context("missing graph input dt_raw")?;
        let b_id = emit
            .state
            .inputs
            .get("b")
            .map(|(id, _)| *id)
            .context("missing graph input b")?;
        let c_id = emit
            .state
            .inputs
            .get("c")
            .map(|(id, _)| *id)
            .context("missing graph input c")?;
        emit.state.named.insert("ssm.dt_raw".into(), dt_id);
        emit.state.named.insert("ssm.b".into(), b_id);
        emit.state.named.insert("ssm.c".into(), c_id);
        Ok(input)
    };

    let flow = ModelFlow::new("mamba1_ssm_scan")
        .input("x", x_shape)
        .input("dt_raw", Shape::new(&[batch, seq, hidden], DType::F32))
        .input("b", n_shape.clone())
        .input("c", n_shape)
        .plugin_named("bind_ssm_inputs", bind)
        .plugin_named("mamba_scan", scan_plugin);

    let built = flow.build(&mut weights)?;
    compile_built(device, built)
}

/// Single-token decode SSM step on CPU.
pub fn selective_scan_step_flow(
    batch: usize,
    hidden: usize,
    state: usize,
    x: &[f32],
    dt_raw: &[f32],
    b: &[f32],
    c: &[f32],
    state_in_out: &mut [f32],
    a_log: &[f32],
    d: &[f32],
) -> Result<Vec<f32>> {
    selective_scan_step_on_device(
        Device::Cpu,
        batch,
        hidden,
        state,
        x,
        dt_raw,
        b,
        c,
        state_in_out,
        a_log,
        d,
    )
}

/// Single-token decode SSM step (`mamba1_step` custom op).
pub fn selective_scan_step_on_device(
    device: Device,
    batch: usize,
    hidden: usize,
    state: usize,
    x: &[f32],
    dt_raw: &[f32],
    b: &[f32],
    c: &[f32],
    state_in_out: &mut [f32],
    a_log: &[f32],
    d: &[f32],
) -> Result<Vec<f32>> {
    ensure!(x.len() == batch * hidden, "x length");
    ensure!(dt_raw.len() == batch * hidden, "dt_raw length");
    ensure!(b.len() == batch * state, "b length");
    ensure!(c.len() == batch * state, "c length");
    ensure!(state_in_out.len() == batch * hidden * state, "state length");
    ensure!(a_log.len() == hidden * state, "a_log length");
    ensure!(d.len() == hidden, "d length");

    ensure_ssm_ops_registered();
    let device = effective_scan_device(device);
    let key = StepCacheKey {
        device,
        shape: StepShapeKey {
            batch,
            hidden,
            state,
        },
        weights: weight_tag(a_log, d),
    };

    let mut cache = step_cache().lock().expect("step cache");
    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(key) {
        let compiled = compile_step_graph(device, batch, hidden, state, a_log, d)?;
        e.insert(compiled);
    }
    let compiled = cache.get_mut(&key).expect("step cache entry");
    let outs = compiled.run(&[
        ("x", x),
        ("dt_raw", dt_raw),
        ("b", b),
        ("c", c),
        ("state_in", state_in_out),
    ]);
    let packed = outs
        .into_iter()
        .next()
        .context("Mamba1StepStage flow must produce one output")?;
    let tail = hidden + hidden * state;
    ensure!(packed.len() == batch * tail, "step packed output length");

    let mut y = vec![0.0; batch * hidden];
    for bi in 0..batch {
        let base = bi * tail;
        y[bi * hidden..(bi + 1) * hidden].copy_from_slice(&packed[base..base + hidden]);
        state_in_out[bi * hidden * state..(bi + 1) * hidden * state]
            .copy_from_slice(&packed[base + hidden..base + tail]);
    }
    Ok(y)
}

fn compile_step_graph(
    device: Device,
    batch: usize,
    hidden: usize,
    state: usize,
    a_log: &[f32],
    d: &[f32],
) -> Result<CompiledGraph> {
    let x_shape = Shape::new(&[batch, hidden], DType::F32);
    let n_shape = Shape::new(&[batch, state], DType::F32);
    let state_shape = Shape::new(&[batch, hidden, state], DType::F32);

    let mut weights = MapWeights::default();
    weights.insert(
        format!("{SCAN_WEIGHT_PREFIX}.A_log"),
        a_log.to_vec(),
        vec![hidden, state],
    );
    weights.insert(format!("{SCAN_WEIGHT_PREFIX}.D"), d.to_vec(), vec![hidden]);

    let step_plugin = Mamba1StepStage::new(batch, hidden, state).plugin();

    let bind = move |emit: &mut rlx_flow::escape::Emit<'_>,
                     input: Option<rlx_flow::FlowValue>|
          -> Result<Option<rlx_flow::FlowValue>> {
        let x_id = emit
            .state
            .inputs
            .get("x")
            .map(|(id, _)| *id)
            .context("missing graph input x")?;
        let dt_id = emit
            .state
            .inputs
            .get("dt_raw")
            .map(|(id, _)| *id)
            .context("missing graph input dt_raw")?;
        let b_id = emit
            .state
            .inputs
            .get("b")
            .map(|(id, _)| *id)
            .context("missing graph input b")?;
        let c_id = emit
            .state
            .inputs
            .get("c")
            .map(|(id, _)| *id)
            .context("missing graph input c")?;
        let state_id = emit
            .state
            .inputs
            .get("state_in")
            .map(|(id, _)| *id)
            .context("missing graph input state_in")?;

        let a_log_id = emit.load_param(&format!("{SCAN_WEIGHT_PREFIX}.A_log"), false)?;
        let d_id = emit.load_param(&format!("{SCAN_WEIGHT_PREFIX}.D"), false)?;

        emit.state.named.insert("mamba1.x".into(), x_id);
        emit.state.named.insert("mamba1.dt_raw".into(), dt_id);
        emit.state.named.insert("mamba1.b".into(), b_id);
        emit.state.named.insert("mamba1.c".into(), c_id);
        emit.state.named.insert("mamba1.a_log".into(), a_log_id);
        emit.state.named.insert("mamba1.d_skip".into(), d_id);
        emit.state.named.insert("mamba1.state_in".into(), state_id);
        Ok(input)
    };

    let flow = ModelFlow::new("mamba1_ssm_step")
        .input("x", x_shape)
        .input("dt_raw", Shape::new(&[batch, hidden], DType::F32))
        .input("b", n_shape.clone())
        .input("c", n_shape)
        .input("state_in", state_shape)
        .plugin_named("bind_mamba1_step", bind)
        .plugin_named("mamba1_step", step_plugin);

    let built = flow.build(&mut weights)?;
    compile_built(device, built)
}

fn compile_built(device: Device, built: BuiltModel) -> Result<CompiledGraph> {
    let (graph, params) = built.into_graph_parts()?;
    let mut compiled = Session::new(device).compile_with(graph, &CompileOptions::new());
    for (name, data) in params {
        compiled.set_param(&name, data.as_slice());
    }
    Ok(compiled)
}

/// Reference selective scan (pre-`rlx-ssm` prefill loop).
#[allow(dead_code)]
pub(crate) fn selective_scan_eager(
    batch: usize,
    seq: usize,
    hidden: usize,
    state: usize,
    x: &[f32],
    delta: &[f32],
    a_log: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
) -> Vec<f32> {
    let bs = batch * seq;
    let mut a_neg = vec![0.0; hidden * state];
    for i in 0..hidden * state {
        a_neg[i] = -a_log[i].exp();
    }

    let mut state_buf = vec![0.0; batch * hidden * state];
    let mut y = vec![0.0; bs * hidden];
    for t in 0..seq {
        for b_idx in 0..batch {
            let bt = b_idx * seq + t;
            let delta_row = &delta[bt * hidden..(bt + 1) * hidden];
            let b_row = &b[bt * state..(bt + 1) * state];
            let c_row = &c[bt * state..(bt + 1) * state];
            let u_row = &x[bt * hidden..(bt + 1) * hidden];
            let s_base = b_idx * hidden * state;
            for hi in 0..hidden {
                let d_t = delta_row[hi];
                let u_t = u_row[hi];
                let mut acc = 0.0f32;
                for ni in 0..state {
                    let da = (d_t * a_neg[hi * state + ni]).exp();
                    let dbu = d_t * b_row[ni] * u_t;
                    let s = &mut state_buf[s_base + hi * state + ni];
                    *s = *s * da + dbu;
                    acc += *s * c_row[ni];
                }
                y[bt * hidden + hi] = acc;
            }
        }
    }
    for r in 0..bs {
        for c_idx in 0..hidden {
            y[r * hidden + c_idx] += d[c_idx] * x[r * hidden + c_idx];
        }
    }
    y
}

/// Reference single-step SSM update.
#[cfg(test)]
pub(crate) fn selective_scan_step_eager(
    batch: usize,
    hidden: usize,
    state: usize,
    x: &[f32],
    dt_raw: &[f32],
    a_log: &[f32],
    b: &[f32],
    c: &[f32],
    state_in_out: &mut [f32],
    d: &[f32],
) -> Vec<f32> {
    let mut a_neg = vec![0.0; hidden * state];
    for i in 0..hidden * state {
        a_neg[i] = -a_log[i].exp();
    }
    let mut delta = vec![0.0; batch * hidden];
    for (o, &v) in delta.iter_mut().zip(dt_raw.iter()) {
        let ax = v.abs();
        *o = v.max(0.0) + (1.0 + (-ax).exp()).ln();
    }

    let mut y = vec![0.0; batch * hidden];
    for b_idx in 0..batch {
        let b_row = &b[b_idx * state..(b_idx + 1) * state];
        let c_row = &c[b_idx * state..(b_idx + 1) * state];
        let s_base = b_idx * hidden * state;
        for hi in 0..hidden {
            let d_t = delta[b_idx * hidden + hi];
            let u_t = x[b_idx * hidden + hi];
            let mut acc = 0.0f32;
            for ni in 0..state {
                let da = (d_t * a_neg[hi * state + ni]).exp();
                let dbu = d_t * b_row[ni] * u_t;
                let s = &mut state_in_out[s_base + hi * state + ni];
                *s = *s * da + dbu;
                acc += *s * c_row[ni];
            }
            y[b_idx * hidden + hi] = acc + d[hi] * u_t;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn softplus(x: f32) -> f32 {
        let ax = x.abs();
        x.max(0.0) + (1.0 + (-ax).exp()).ln()
    }

    #[test]
    fn flow_matches_eager_reference() {
        let batch = 2usize;
        let seq = 5usize;
        let hidden = 8usize;
        let state = 4usize;
        let bs = batch * seq;

        let mut rng = 99u64;
        let mut next = || {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((rng >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.2
        };

        let x: Vec<f32> = (0..bs * hidden).map(|_| next()).collect();
        let dt_pre: Vec<f32> = (0..bs * hidden).map(|_| next()).collect();
        let b_mat: Vec<f32> = (0..bs * state).map(|_| next()).collect();
        let c_mat: Vec<f32> = (0..bs * state).map(|_| next()).collect();
        let a_log: Vec<f32> = (0..hidden * state)
            .map(|i| ((i % state + 1) as f32).ln())
            .collect();
        let d: Vec<f32> = vec![1.0; hidden];

        let delta: Vec<f32> = dt_pre.iter().map(|&v| softplus(v)).collect();

        let eager = selective_scan_eager(
            batch, seq, hidden, state, &x, &delta, &a_log, &b_mat, &c_mat, &d,
        );
        let flow = selective_scan_flow(
            batch, seq, hidden, state, &x, &dt_pre, &b_mat, &c_mat, &a_log, &d,
        )
        .expect("scan flow");

        let mut max_abs = 0.0f32;
        for (a, b) in eager.iter().zip(flow.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 1e-4,
            "flow vs eager selective scan max_abs = {max_abs}"
        );
    }

    #[test]
    fn step_flow_matches_eager_reference() {
        let batch = 2usize;
        let hidden = 8usize;
        let state = 4usize;

        let mut rng = 7u64;
        let mut next = || {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((rng >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.2
        };

        let x: Vec<f32> = (0..batch * hidden).map(|_| next()).collect();
        let dt_pre: Vec<f32> = (0..batch * hidden).map(|_| next()).collect();
        let b_mat: Vec<f32> = (0..batch * state).map(|_| next()).collect();
        let c_mat: Vec<f32> = (0..batch * state).map(|_| next()).collect();
        let a_log: Vec<f32> = (0..hidden * state)
            .map(|i| ((i % state + 1) as f32).ln())
            .collect();
        let d: Vec<f32> = vec![1.0; hidden];

        let mut state_eager = vec![0.0; batch * hidden * state];
        let mut state_flow = state_eager.clone();
        let eager = selective_scan_step_eager(
            batch,
            hidden,
            state,
            &x,
            &dt_pre,
            &a_log,
            &b_mat,
            &c_mat,
            &mut state_eager,
            &d,
        );
        let flow = selective_scan_step_flow(
            batch,
            hidden,
            state,
            &x,
            &dt_pre,
            &b_mat,
            &c_mat,
            &mut state_flow,
            &a_log,
            &d,
        )
        .expect("step flow");

        let mut max_y = 0.0f32;
        for (a, b) in eager.iter().zip(flow.iter()) {
            max_y = max_y.max((a - b).abs());
        }
        let mut max_state = 0.0f32;
        for (a, b) in state_eager.iter().zip(state_flow.iter()) {
            max_state = max_state.max((a - b).abs());
        }
        assert!(max_y < 1e-4, "step y max_abs = {max_y}");
        assert!(max_state < 1e-4, "step state max_abs = {max_state}");
    }
}
