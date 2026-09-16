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

//! Device-backed TimesFM-3 session with a compiled core graph.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use ndarray::{Array2, Array4, ArrayView1, ArrayView3};
use rlx_core::flow_util::{built_from_hir, compile_cache_ensure_built};
use rlx_runtime::Device;
use rlx_runtime::compile_cache::CompileCache;

use crate::config::TimesFM3Config;
use crate::flow::{CoreGraphDims, build_core_hir};
use crate::host::model::{CorePrepared, TimesFM3Model};
use crate::host::preprocess::build_decode_inputs;
use crate::rope::{build_attn_bias, build_rope_tables, repeat_heads, seq_positions};

const CORE_CACHE_CAPACITY: usize = 8;

/// TimesFM-3 on an RLX backend: host preprocess/decode + compiled core.
pub struct TimesFM3Session {
    model: TimesFM3Model,
    device: Device,
    core_cache: CompileCache,
    core_params: HashMap<u64, Arc<HashMap<String, Vec<f32>>>>,
}

impl TimesFM3Session {
    pub fn open(weights_path: &Path, cfg: TimesFM3Config, device: Device) -> Result<Self> {
        let model = TimesFM3Model::load(weights_path, cfg)?;
        Ok(Self {
            model,
            device,
            core_cache: CompileCache::new(device, CORE_CACHE_CAPACITY),
            core_params: HashMap::new(),
        })
    }

    pub fn from_model(model: TimesFM3Model, device: Device) -> Self {
        Self {
            model,
            device,
            core_cache: CompileCache::new(device, CORE_CACHE_CAPACITY),
            core_params: HashMap::new(),
        }
    }

    pub fn config(&self) -> &TimesFM3Config {
        &self.model.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn model(&self) -> &TimesFM3Model {
        &self.model
    }

    pub fn warm(&mut self, b: usize, v: usize, n: usize) -> Result<()> {
        self.ensure_core(CoreGraphDims { b, v, n })
    }

    fn ensure_core(&mut self, dims: CoreGraphDims) -> Result<()> {
        let key = dims.key();
        if self.core_cache.contains(key) {
            return Ok(());
        }
        let (hir, params) = build_core_hir(&self.model.cfg, &self.model.weights, dims)?;
        self.core_params.insert(key, Arc::new(params.clone()));
        let built = built_from_hir(hir, params)?;
        compile_cache_ensure_built(&mut self.core_cache, key, built)?;
        Ok(())
    }

    /// Run the compiled core graph (used by parity tests).
    #[cfg(any(feature = "dev", test))]
    pub fn run_core(&mut self, prep: &CorePrepared) -> Result<Array2<f32>> {
        self.run_core_inner(prep)
    }

    fn run_core_inner(&mut self, prep: &CorePrepared) -> Result<Array2<f32>> {
        let dims = CoreGraphDims {
            b: prep.b,
            v: prep.v,
            n: prep.n,
        };
        self.ensure_core(dims)?;
        let key = dims.key();
        let params = Arc::clone(
            self.core_params
                .get(&key)
                .context("core params missing after ensure")?,
        );
        let cg = self
            .core_cache
            .get_or_compile(key, || panic!("core cache missing after ensure"));
        for (name, data) in params.iter() {
            cg.set_param(name, data);
        }

        let hd = self.model.cfg.head_dim();
        let nh = self.model.cfg.num_heads();
        let seq_mask = flatten_bv_mask(prep.effective_mask.view());
        let seq_pos = seq_positions(&seq_mask);
        let (cos_seq, sin_seq) = build_rope_tables(&seq_pos, hd);
        let (cos_seq, sin_seq) =
            repeat_heads(&cos_seq, &sin_seq, prep.b * prep.v, prep.n, hd / 2, nh);
        let attn_bias_seq = build_attn_bias(&seq_mask, prep.n, true);

        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("res_in", prep.res_in.as_slice().unwrap()),
            ("rope_cos_seq", &cos_seq),
            ("rope_sin_seq", &sin_seq),
            ("attn_bias_seq", &attn_bias_seq),
        ];

        let var_mask = flatten_var_mask(prep.effective_mask.view());
        let var_pos = seq_positions(&var_mask);
        let (cos_var, sin_var) = build_rope_tables(&var_pos, hd);
        let (cos_var, sin_var) =
            repeat_heads(&cos_var, &sin_var, prep.b * prep.n, prep.v, hd / 2, nh);
        let attn_bias_var = build_attn_bias(&var_mask, prep.v, false);
        let cos_var_buf = cos_var;
        let sin_var_buf = sin_var;
        let attn_var_buf = attn_bias_var;
        if self.model.cfg.use_variate_attention {
            inputs.push(("rope_cos_var", &cos_var_buf));
            inputs.push(("rope_sin_var", &sin_var_buf));
            inputs.push(("attn_bias_var", &attn_var_buf));
        }

        let out = cg
            .run(&inputs)
            .into_iter()
            .next()
            .context("core produced no output")?;
        let rows = prep.b * prep.v * prep.n;
        let cols = self.model.cfg.output_patch_len * self.model.cfg.num_quantiles();
        ensure!(
            out.len() == rows * cols,
            "core output {} != {rows}x{cols}",
            out.len()
        );
        Ok(Array2::from_shape_vec((rows, cols), out)?)
    }

    pub fn decode(
        &mut self,
        target: ArrayView3<f32>,
        horizon: usize,
        past_only: Option<ArrayView3<f32>>,
        past_future: Option<ArrayView3<f32>>,
        mask: Option<ArrayView1<bool>>,
    ) -> Result<Array4<f32>> {
        let b = target.shape()[0];
        let (values, masks, pit, cpm, ctx_patches, _padded_hor) = build_decode_inputs(
            &self.model.cfg,
            target,
            horizon,
            mask,
            past_only,
            past_future,
        );
        let prep =
            self.model
                .prepare_core(values.view(), masks.view(), pit.view(), Some(cpm.row(0)))?;
        let raw = if self.device == Device::Cpu && !use_compiled_core_on_cpu() {
            self.model.core_forward_host(&prep)
        } else {
            self.run_core_inner(&prep)?
        };
        let logits = self.model.finalize_core(raw.view(), &prep)?;
        Ok(extract_forecast(
            &self.model.cfg,
            logits.view(),
            b,
            horizon,
            ctx_patches,
        ))
    }
}

/// When set, run the compiled core on CPU instead of the host reference (parity debugging).
fn use_compiled_core_on_cpu() -> bool {
    std::env::var("RLX_TIMESFM3_COMPILED_CORE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn flatten_bv_mask(mask: ndarray::ArrayView3<bool>) -> Array2<bool> {
    let (b, v, n) = mask.dim();
    let mut out = Array2::from_elem((b * v, n), false);
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                out[[bi * v + vi, ni]] = mask[[bi, vi, ni]];
            }
        }
    }
    out
}

fn flatten_var_mask(mask: ndarray::ArrayView3<bool>) -> Array2<bool> {
    let (b, v, n) = mask.dim();
    let mut out = Array2::from_elem((b * n, v), false);
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                out[[bi * n + ni, vi]] = mask[[bi, vi, ni]];
            }
        }
    }
    out
}

fn extract_forecast(
    cfg: &TimesFM3Config,
    logits: ndarray::ArrayView5<f32>,
    b: usize,
    horizon: usize,
    ctx_patches: usize,
) -> Array4<f32> {
    use ndarray::s;

    let extract_len = if cfg.use_stitching {
        (2 * cfg.input_patch_len).min(cfg.output_patch_len)
    } else {
        cfg.output_patch_len
    };
    let overlap = extract_len - cfg.input_patch_len;
    let num_forecast_patches = ((horizon as f32 - overlap as f32) / cfg.input_patch_len as f32)
        .ceil()
        .max(1.0) as usize;

    if cfg.use_stitching {
        let mut patch_preds = ndarray::Array5::<f32>::zeros((
            b,
            logits.shape()[1],
            num_forecast_patches,
            extract_len,
            cfg.num_quantiles(),
        ));
        for pi in 0..num_forecast_patches {
            let idx = ctx_patches - 1 + pi;
            for t in 0..extract_len {
                for q in 0..cfg.num_quantiles() {
                    patch_preds
                        .slice_mut(s![.., .., pi, t, q])
                        .assign(&logits.slice(s![.., .., idx, t, q]));
                }
            }
        }
        let stitched = crate::host::preprocess::stitch_patches_quantile(
            patch_preds.view(),
            cfg.input_patch_len,
        );
        stitched.slice(s![.., .., ..horizon, ..]).to_owned()
    } else {
        let idx = ctx_patches - 1;
        logits
            .slice(s![.., .., idx, ..horizon, ..])
            .to_owned()
            .into_shape_with_order((b, logits.shape()[1], horizon, cfg.num_quantiles()))
            .unwrap()
    }
}
