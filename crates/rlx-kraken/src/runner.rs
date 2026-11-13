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

//! Text-line image → transcription with a Kraken recognition model.

use crate::codec::Codec;
use crate::graph::{self, INPUT};
use crate::mlmodel::MlModel;
use crate::preprocess;
use crate::vgsl::{Block, OutputKind, Vgsl};
use anyhow::{Context, Result, anyhow, bail, ensure};
use image::DynamicImage;
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::{attach_built_params, graph_from_hir};
use rlx_core::ordered_json::Json;
use rlx_core::validate_standard_device;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::path::Path;

/// Kraken's default `padding` (`RecognitionInferenceConfig`).
pub const DEFAULT_PADDING: usize = 16;

/// A recognized line.
#[derive(Debug, Clone)]
pub struct Recognition {
    pub text: String,
    /// Decoded labels with their frame spans.
    pub labels: Vec<rlx_core::ctc::CtcLabel>,
    pub frames: usize,
}

/// A loaded Kraken recognition model; one compiled graph per line width.
pub struct KrakenRunner {
    spec: Vgsl,
    model: MlModel,
    codec: Codec,
    padding: usize,
    device: Device,
    cache: HashMap<(usize, usize), (CompiledGraph, usize)>,
}

impl KrakenRunner {
    /// Open a Kraken recognition model: a Core ML `.mlmodel` or a
    /// Kraken ≥ 6 `.safetensors`.
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        validate_standard_device("kraken", device)?;
        let st = path.extension().and_then(|e| e.to_str()) == Some("safetensors");
        let (model, meta) = if st {
            let (m, meta) = MlModel::open_safetensors(path)?;
            (m, Some(meta))
        } else {
            let m = MlModel::open(path)?;
            let meta = m
                .metadata
                .get("kraken_meta")
                .map(|s| Json::parse(s).map_err(|e| anyhow!("kraken_meta: {e:?}")))
                .transpose()?;
            (m, meta)
        };
        let field = |k: &str| meta.as_ref().and_then(|m| m.get(k)).cloned();
        let vgsl = model
            .metadata
            .get("vgsl")
            .cloned()
            .or_else(|| field("vgsl")?.as_str().map(String::from))
            .ok_or_else(|| anyhow!("{}: no VGSL spec in the metadata", path.display()))?;
        let spec = Vgsl::parse(&vgsl)?;
        let kind = match field("model_type") {
            Some(Json::Str(s)) => Some(s),
            Some(Json::Array(a)) => a.first().and_then(Json::as_str).map(String::from),
            _ => None,
        };
        if let Some(t) = kind {
            ensure!(
                t == "recognition",
                "{} is a {t} model, not a recognition model",
                path.display()
            );
        }
        let codec = match (model.metadata.get("codec"), field("codec")) {
            (Some(text), _) => Codec::from_json(text)?,
            (None, Some(Json::Str(text))) => Codec::from_json(&text)?,
            (None, Some(obj)) => Codec::from_value(&obj)?,
            (None, None) => bail!("{}: no codec in the metadata", path.display()),
        };
        let padding = match field("hyper_params") {
            Some(Json::Str(repr)) => hyper_int(&repr, "pad"),
            Some(h) => match h.get("pad") {
                Some(Json::Int(p)) => usize::try_from(*p).ok(),
                Some(Json::UInt(p)) => usize::try_from(*p).ok(),
                _ => None,
            },
            None => None,
        }
        .unwrap_or(DEFAULT_PADDING);
        Self::from_parts(spec, model, codec, padding, device)
    }

    /// Assemble from already-parsed parts.
    pub fn from_parts(
        spec: Vgsl,
        model: MlModel,
        codec: Codec,
        padding: usize,
        device: Device,
    ) -> Result<Self> {
        match spec.blocks.last() {
            Some(Block::Output {
                kind: OutputKind::Ctc,
                out,
                ..
            }) => ensure!(
                codec.max_label() < *out,
                "codec uses label {} but the network has {out} classes",
                codec.max_label()
            ),
            _ => bail!("not a CTC recognition model (last block is not `O1c…`)"),
        }
        Ok(Self {
            spec,
            model,
            codec,
            padding,
            device,
            cache: HashMap::new(),
        })
    }

    pub fn spec(&self) -> &Vgsl {
        &self.spec
    }

    /// Columns of white padding added on each side of a line.
    pub fn padding(&self) -> usize {
        self.padding
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Per-frame logits `[frames, classes]` for a preprocessed `[C, H, W]`
    /// line (the tensor Kraken feeds its network).
    pub fn logits(&mut self, input: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        let c = self.spec.input.channels;
        ensure!(
            input.len() == c * height * width,
            "input has {} values, expected {c}x{height}x{width}",
            input.len()
        );
        if !self.cache.contains_key(&(height, width)) {
            let g = graph::build(&self.spec, &self.model, height, width)?;
            ensure!(
                g.rows == 1,
                "the network leaves {} rows; a CTC line model must reduce the height to 1",
                g.rows
            );
            let classes = g.classes;
            let (graph, params) = graph_from_hir(g.hir, g.params)?;
            let opts = compile_options_for_profile(&CompileProfile::encoder(), self.device);
            let mut compiled = Session::new(self.device).compile_with(graph, &opts);
            attach_built_params(&mut compiled, params, &[]);
            self.cache.insert((height, width), (compiled, classes));
        }
        let (compiled, _) = self.cache.get_mut(&(height, width)).expect("just inserted");
        compiled
            .run(&[(INPUT, input)])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("kraken graph returned no output"))
    }

    /// Recognize one text-line image.
    pub fn recognize(&mut self, img: &DynamicImage) -> Result<Recognition> {
        let inp = self.spec.input;
        let (x, h, w) = preprocess::line_tensor(img, inp.channels, inp.height, self.padding);
        let logits = self.logits(&x, h, w)?;
        let classes = match self.spec.blocks.last() {
            Some(Block::Output { out, .. }) => *out,
            _ => unreachable!("checked in from_parts"),
        };
        let labels = rlx_core::ctc::greedy_decode(&logits, classes, 0);
        let ids: Vec<usize> = labels.iter().map(|l| l.label).collect();
        Ok(Recognition {
            text: self.codec.decode(&ids),
            frames: logits.len() / classes,
            labels,
        })
    }

    pub fn recognize_path(&mut self, path: &Path) -> Result<Recognition> {
        let img = image::open(path).with_context(|| format!("opening {}", path.display()))?;
        self.recognize(&img)
    }
}

/// An integer from Kraken's `hyper_params`, a Python dict `repr`
/// (`"{'pad': 16, 'freq': 1.0, …}"`).
fn hyper_int(repr: &str, key: &str) -> Option<usize> {
    let pat = format!("'{key}':");
    let rest = &repr[repr.find(&pat)? + pat.len()..];
    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyper_params_repr() {
        let r = "{'pad': 16, 'freq': 1.0, 'batch_size': 64}";
        assert_eq!(hyper_int(r, "pad"), Some(16));
        assert_eq!(hyper_int(r, "batch_size"), Some(64));
        assert_eq!(hyper_int(r, "lag"), None);
    }
}
