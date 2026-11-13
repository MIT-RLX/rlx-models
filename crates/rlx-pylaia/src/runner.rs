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

//! Text-line image → transcription.

use crate::config::PylaiaConfig;
use crate::graph::{self, INPUT};
use crate::preprocess;
use crate::symbols::SymbolTable;
use crate::weights::{PylaiaWeights, read_checkpoint};
use anyhow::{Context, Result, anyhow, ensure};
use image::GrayImage;
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::{attach_built_params, graph_from_hir};
use rlx_core::validate_standard_device;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::path::Path;

/// A recognized line.
#[derive(Debug, Clone)]
pub struct Recognition {
    pub text: String,
    /// Decoded class indices (blank-free, repeats merged).
    pub labels: Vec<usize>,
    /// Output frames the CTC ran over.
    pub frames: usize,
}

/// A loaded PyLaia model; compiles one graph per input size on demand.
pub struct PylaiaRunner {
    cfg: PylaiaConfig,
    weights: PylaiaWeights,
    symbols: SymbolTable,
    device: Device,
    cache: HashMap<(usize, usize), CompiledGraph>,
}

impl PylaiaRunner {
    /// Open a PyLaia model directory: `model` (architecture), `weights.ckpt`
    /// and `syms.txt`, as Teklia publishes them on the Hugging Face Hub.
    pub fn open(dir: &Path, device: Device) -> Result<Self> {
        validate_standard_device("pylaia", device)?;
        let cfg = PylaiaConfig::from_model_file(&dir.join("model"))?;
        let table = read_checkpoint(&dir.join("weights.ckpt"))?;
        let weights = PylaiaWeights::from_table(&cfg, table)
            .with_context(|| format!("loading {}", dir.join("weights.ckpt").display()))?;
        let symbols = SymbolTable::from_file(&dir.join("syms.txt"))?;
        ensure!(
            symbols.len() == cfg.num_labels,
            "syms.txt has {} symbols, the model outputs {}",
            symbols.len(),
            cfg.num_labels
        );
        Ok(Self::from_parts(cfg, weights, symbols, device))
    }

    /// Assemble from already-loaded parts (synthetic tests, custom loaders).
    pub fn from_parts(
        cfg: PylaiaConfig,
        weights: PylaiaWeights,
        symbols: SymbolTable,
        device: Device,
    ) -> Self {
        Self {
            cfg,
            weights,
            symbols,
            device,
            cache: HashMap::new(),
        }
    }

    pub fn config(&self) -> &PylaiaConfig {
        &self.cfg
    }

    pub fn symbols(&self) -> &SymbolTable {
        &self.symbols
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Per-frame logits `[frames, labels]` for a preprocessed `[C, H, W]`
    /// input (the exact tensor PyLaia feeds its network).
    pub fn logits(&mut self, input: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        ensure!(
            input.len() == self.cfg.in_channels * height * width,
            "input has {} values, expected {}x{height}x{width}",
            input.len(),
            self.cfg.in_channels
        );
        if !self.cache.contains_key(&(height, width)) {
            let g = graph::build(&self.cfg, &self.weights, height, width)?;
            let (graph, params) = graph_from_hir(g.hir, g.params)?;
            let opts = compile_options_for_profile(&CompileProfile::encoder(), self.device);
            let mut compiled = Session::new(self.device).compile_with(graph, &opts);
            attach_built_params(&mut compiled, params, &[]);
            self.cache.insert((height, width), compiled);
        }
        let compiled = self.cache.get_mut(&(height, width)).expect("just inserted");
        compiled
            .run(&[(INPUT, input)])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("pylaia graph returned no output"))
    }

    /// Recognize a grayscale line image already at the model's height.
    pub fn recognize_gray(&mut self, img: &GrayImage) -> Result<Recognition> {
        ensure!(
            self.cfg.in_channels == 1,
            "this model takes {}-channel images",
            self.cfg.in_channels
        );
        let min_w = self.cfg.min_width(img.height() as usize);
        let (x, h, w) = preprocess::to_input(img, min_w);
        let logits = self.logits(&x, h, w)?;
        let classes = self.cfg.num_labels;
        let labels = rlx_core::ctc::greedy_labels(&logits, classes, 0);
        Ok(Recognition {
            text: self.symbols.decode(&labels),
            frames: logits.len() / classes,
            labels,
        })
    }

    /// Open an image file and recognize it, optionally rescaling it to
    /// `height` first (PyLaia itself expects lines at the training height).
    pub fn recognize_path(&mut self, path: &Path, height: Option<u32>) -> Result<Recognition> {
        let img = image::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut gray = preprocess::to_gray(&img);
        if let Some(h) = height {
            gray = preprocess::resize_to_height(&gray, h);
        }
        self.recognize_gray(&gray)
    }
}
