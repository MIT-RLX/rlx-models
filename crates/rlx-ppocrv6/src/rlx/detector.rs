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

//! DB text detector: native HIR + safetensors, host box post-process.

use crate::capabilities::validate_device;
use crate::config::{DetectionParams, Tier};
use crate::detection::{DetBox, db_boxes_from_prob};
use crate::model::build_detection;
use crate::preprocess::{RgbPage, det_nchw, det_resize};
use anyhow::{Result, anyhow};
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::attach_built_params;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct RlxDetector {
    weights_dir: PathBuf,
    tier: Tier,
    params: DetectionParams,
    device: Device,
    cache: Mutex<HashMap<(usize, usize), CompiledGraph>>,
}

impl RlxDetector {
    pub fn from_safetensors(
        weights_dir: impl AsRef<Path>,
        tier: Tier,
        params: DetectionParams,
        device: Device,
    ) -> Result<Self> {
        validate_device(device)?;
        Ok(Self {
            weights_dir: weights_dir.as_ref().to_path_buf(),
            tier,
            params,
            device,
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn ensure_compiled(&self, h: usize, w: usize) -> Result<()> {
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        if cache.contains_key(&(h, w)) {
            return Ok(());
        }
        let built = build_detection(self.tier, &self.weights_dir, h, w)?;
        let compile_opts = compile_options_for_profile(&CompileProfile::encoder(), self.device);
        let mut compiled = Session::new(self.device).compile_with(built.graph, &compile_opts);
        attach_built_params(&mut compiled, built.params, &[]);
        cache.insert((h, w), compiled);
        Ok(())
    }

    pub fn detect(&self, page: &RgbPage) -> Result<Vec<DetBox>> {
        let (resized, sx, sy, h, w) = det_resize(page, self.params.limit_side_len);
        self.ensure_compiled(h, w)?;
        let input = det_nchw(&resized);
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        let compiled = cache
            .get_mut(&(h, w))
            .ok_or_else(|| anyhow!("det graph missing after compile"))?;
        let flat = compiled
            .run(&[("x", input.as_slice())])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("det produced no outputs"))?;
        let expected = h * w;
        let map = if flat.len() >= expected {
            flat[flat.len() - expected..].to_vec()
        } else {
            return Err(anyhow!(
                "det output length {} < expected H*W={}",
                flat.len(),
                expected
            ));
        };
        let tensor = NdTensor::from_data([h, w], map);
        Ok(db_boxes_from_prob(tensor.view(), &self.params, sx, sy))
    }
}
