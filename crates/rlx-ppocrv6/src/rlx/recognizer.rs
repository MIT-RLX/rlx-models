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

//! Line recognizer: native HIR + safetensors, host CTC decode.

use crate::capabilities::validate_device;
use crate::config::{RecognitionParams, Tier};
use crate::model::build_recognition;
use crate::preprocess::rec_resize_pad;
use crate::recognition::CharDict;
use anyhow::{Result, anyhow};
use image::RgbImage;
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::attach_built_params;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct RlxRecognizer {
    weights_dir: PathBuf,
    tier: Tier,
    params: RecognitionParams,
    dict: CharDict,
    device: Device,
    cache: Mutex<HashMap<(usize, usize), CompiledGraph>>,
}

impl RlxRecognizer {
    pub fn from_safetensors(
        weights_dir: impl AsRef<Path>,
        tier: Tier,
        dict: CharDict,
        params: RecognitionParams,
        device: Device,
    ) -> Result<Self> {
        validate_device(device)?;
        Ok(Self {
            weights_dir: weights_dir.as_ref().to_path_buf(),
            tier,
            params,
            dict,
            device,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn dict(&self) -> &CharDict {
        &self.dict
    }

    fn ensure_compiled(&self, height: usize, width: usize) -> Result<()> {
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        if cache.contains_key(&(height, width)) {
            return Ok(());
        }
        let built = build_recognition(self.tier, &self.weights_dir, height, width)?;
        let compile_opts = compile_options_for_profile(&CompileProfile::encoder(), self.device);
        let mut compiled = Session::new(self.device).compile_with(built.graph, &compile_opts);
        attach_built_params(&mut compiled, built.params, &[]);
        cache.insert((height, width), compiled);
        Ok(())
    }

    pub fn recognize_crop(&self, crop: &RgbImage) -> Result<String> {
        let (nchw, h, w) = rec_resize_pad(crop, self.params.image_height, self.params.max_width)?;
        self.ensure_compiled(h, w)?;
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        let compiled = cache
            .get_mut(&(h, w))
            .ok_or_else(|| anyhow!("rec graph missing"))?;
        let flat = compiled
            .run(&[("x", nchw.as_slice())])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("rec produced no outputs"))?;
        let num_classes = self.dict.num_classes();
        if flat.len() % num_classes != 0 {
            anyhow::bail!(
                "rec output len {} not divisible by num_classes {}",
                flat.len(),
                num_classes
            );
        }
        let seq = flat.len() / num_classes;
        Ok(self.dict.decode_greedy(&flat, seq, num_classes))
    }

    pub fn recognize_crops(&self, crops: &[RgbImage]) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(crops.len());
        for c in crops {
            out.push(self.recognize_crop(c)?);
        }
        Ok(out)
    }
}
