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

use anyhow::{Context, Result, bail};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

pub struct OcrGraphBuilder {
    pub hir: rlx_ir::hir::HirModule,
    pub params: HashMap<String, Vec<f32>>,
    zero_bias: HashMap<usize, HirNodeId>,
}

impl OcrGraphBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            hir: rlx_ir::hir::HirModule::new(name),
            params: HashMap::new(),
            zero_bias: HashMap::new(),
        }
    }

    pub fn m(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    pub fn zero_bias(&mut self, channels: usize) -> Result<HirNodeId> {
        if let Some(&id) = self.zero_bias.get(&channels) {
            return Ok(id);
        }
        let key = format!("ocr.zero_bias.{channels}");
        let data = vec![0f32; channels];
        let id = self.m().param(&key, Shape::new(&[channels], DType::F32));
        self.params.insert(key, data);
        self.zero_bias.insert(channels, id);
        Ok(id)
    }

    pub fn load_param(&mut self, wm: &mut WeightMap, key: &str) -> Result<HirNodeId> {
        let (data, shape) = wm
            .take(key)
            .with_context(|| format!("missing weight {key}"))?;
        let id = self.m().param(key, Shape::new(&shape, DType::F32));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    pub fn load_param_optional(
        &mut self,
        wm: &mut WeightMap,
        key: &str,
    ) -> Result<Option<HirNodeId>> {
        if !wm.has(key) {
            return Ok(None);
        }
        Ok(Some(self.load_param(wm, key)?))
    }

    pub fn finish(self) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
        rlx_core::flow_util::graph_from_hir(self.hir, self.params)
    }
}

/// 26 fused pointwise+BN blocks in traversal order (13 `DoubleConv` × 2).
pub const DET_ONNX_PW: [(&str, &str); 26] = [
    ("onnx::Conv_470", "onnx::Conv_471"),
    ("onnx::Conv_473", "onnx::Conv_474"),
    ("onnx::Conv_476", "onnx::Conv_477"),
    ("onnx::Conv_479", "onnx::Conv_480"),
    ("onnx::Conv_482", "onnx::Conv_483"),
    ("onnx::Conv_485", "onnx::Conv_486"),
    ("onnx::Conv_488", "onnx::Conv_489"),
    ("onnx::Conv_491", "onnx::Conv_492"),
    ("onnx::Conv_494", "onnx::Conv_495"),
    ("onnx::Conv_497", "onnx::Conv_498"),
    ("onnx::Conv_500", "onnx::Conv_501"),
    ("onnx::Conv_503", "onnx::Conv_504"),
    ("onnx::Conv_506", "onnx::Conv_507"),
    ("onnx::Conv_509", "onnx::Conv_510"),
    ("onnx::Conv_512", "onnx::Conv_513"),
    ("onnx::Conv_515", "onnx::Conv_516"),
    ("onnx::Conv_518", "onnx::Conv_519"),
    ("onnx::Conv_521", "onnx::Conv_522"),
    ("onnx::Conv_524", "onnx::Conv_525"),
    ("onnx::Conv_527", "onnx::Conv_528"),
    ("onnx::Conv_530", "onnx::Conv_531"),
    ("onnx::Conv_533", "onnx::Conv_534"),
    ("onnx::Conv_536", "onnx::Conv_537"),
    ("onnx::Conv_539", "onnx::Conv_540"),
    ("onnx::Conv_542", "onnx::Conv_543"),
    ("onnx::Conv_545", "onnx::Conv_546"),
];

pub const DET_DW_KEYS: [&str; 26] = [
    "in_conv.seq.0.seq.0.weight",
    "in_conv.seq.1.seq.0.weight",
    "down.0.seq.0.seq.0.seq.0.weight",
    "down.0.seq.0.seq.1.seq.0.weight",
    "down.1.seq.0.seq.0.seq.0.weight",
    "down.1.seq.0.seq.1.seq.0.weight",
    "down.2.seq.0.seq.0.seq.0.weight",
    "down.2.seq.0.seq.1.seq.0.weight",
    "down.3.seq.0.seq.0.seq.0.weight",
    "down.3.seq.0.seq.1.seq.0.weight",
    "down.4.seq.0.seq.0.seq.0.weight",
    "down.4.seq.0.seq.1.seq.0.weight",
    "down.5.seq.0.seq.0.seq.0.weight",
    "down.5.seq.0.seq.1.seq.0.weight",
    "up.5.contract.seq.0.seq.0.weight",
    "up.5.contract.seq.1.seq.0.weight",
    "up.4.contract.seq.0.seq.0.weight",
    "up.4.contract.seq.1.seq.0.weight",
    "up.3.contract.seq.0.seq.0.weight",
    "up.3.contract.seq.1.seq.0.weight",
    "up.2.contract.seq.0.seq.0.weight",
    "up.2.contract.seq.1.seq.0.weight",
    "up.1.contract.seq.0.seq.0.weight",
    "up.1.contract.seq.1.seq.0.weight",
    "up.0.contract.seq.0.seq.0.weight",
    "up.0.contract.seq.1.seq.0.weight",
];

pub fn detection_input_hw() -> (usize, usize) {
    if let Ok(s) = std::env::var("OCR_DETECTION_HW") {
        if let Some(hw) = parse_hw(&s) {
            return hw;
        }
    }
    (800, 600)
}

pub fn parse_hw(s: &str) -> Option<(usize, usize)> {
    let (h, w) = s.split_once(',')?;
    Some((h.trim().parse().ok()?, w.trim().parse().ok()?))
}

pub fn assert_weights_drained(wm: &WeightMap, context: &str) -> Result<()> {
    let leftover: Vec<_> = wm
        .keys()
        .filter(|k| !k.starts_with('/') && !k.contains("Constant") && !k.contains("Unsqueeze"))
        .collect();
    if leftover.is_empty() {
        return Ok(());
    }
    let mut keys = leftover;
    keys.sort();
    bail!("{context}: unmapped weights: {keys:?}");
}
