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

//! Host vs compiled-core parity helpers (requires `dev` feature).

use anyhow::{Context, Result};
use ndarray::Array2;
use rlx_core::flow_util::{built_from_hir, compile_cache_ensure_built};
use rlx_runtime::Device;
use rlx_runtime::compile_cache::CompileCache;

use crate::config::TimesFM3Config;
use crate::flow::{CoreGraphDims, build_resblock_hir};
use crate::host::model::{CorePrepared, TimesFM3Model};
use crate::session::TimesFM3Session;
use crate::weights::TimesFM3Weights;

/// Max absolute elementwise difference between two same-shaped tensors.
pub fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_resblock_compiled(
    cfg: &TimesFM3Config,
    weights: &TimesFM3Weights,
    prep: &CorePrepared,
    device: Device,
) -> Result<Array2<f32>> {
    let dims = CoreGraphDims {
        b: prep.b,
        v: prep.v,
        n: prep.n,
    };
    let (hir, params) = build_resblock_hir(cfg, weights, dims)?;
    let built = built_from_hir(hir, params.clone())?;
    let mut cache = CompileCache::new(device, 1);
    compile_cache_ensure_built(&mut cache, dims.key(), built)?;
    let cg = cache.get_or_compile(dims.key(), || panic!("resblock cache missing"));
    for (name, data) in &params {
        cg.set_param(name, data);
    }
    let out = cg
        .run(&[("res_in", prep.res_in.as_slice().unwrap())])
        .into_iter()
        .next()
        .context("resblock produced no output")?;
    let rows = prep.b * prep.v * prep.n;
    let cols = cfg.model_dims();
    Ok(Array2::from_shape_vec((rows, cols), out)?)
}

/// Host resblock output for [`CorePrepared::res_in`].
pub fn host_resblock_out(model: &TimesFM3Model, prep: &CorePrepared) -> Array2<f32> {
    let w = &model.weights.resblock;
    let flat = prep.res_in.clone();
    let mut h = flat.clone();
    if let Some(ref ln) = w.pre_norm {
        h = crate::host::math::rms_norm(h.view(), ln.view());
    }
    h = crate::host::math::linear2(h.view(), w.hidden.view(), None);
    crate::host::math::relu_array(&mut h);
    let out = crate::host::math::linear2(h.view(), w.output.view(), None);
    let res = crate::host::math::linear2(flat.view(), w.residual.view(), None);
    out + res
}

/// Compare host vs compiled resblock on the same prepared inputs.
pub fn compare_resblock(
    model: &TimesFM3Model,
    prep: &CorePrepared,
    device: Device,
) -> Result<(Array2<f32>, Array2<f32>, f32)> {
    let host = host_resblock_out(model, prep);
    let compiled = run_resblock_compiled(&model.cfg, &model.weights, prep, device)?;
    let diff = max_abs_diff(&host, &compiled);
    Ok((host, compiled, diff))
}

/// Run host and compiled cores on the same [`CorePrepared`] inputs.
pub fn compare_core(
    model: &TimesFM3Model,
    prep: &CorePrepared,
    device: Device,
) -> Result<(Array2<f32>, Array2<f32>, f32)> {
    let host = model.core_forward_host(prep);
    let mut session = TimesFM3Session::from_model(
        TimesFM3Model::from_weights(model.cfg.clone(), model.weights.clone()),
        device,
    );
    let compiled = session.run_core(prep)?;
    let diff = max_abs_diff(&host, &compiled);
    Ok((host, compiled, diff))
}
