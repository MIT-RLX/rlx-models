// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Compile + run the maskable ViT forward on a chosen backend.

use anyhow::{Result, anyhow};
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};

use super::config::VitConfig;
use super::forward::{VitGraph, build_vit_graph};
use super::preprocess::{PreprocessWeights, assemble_hidden, rgb_u8_to_imagenet_nchw};
use super::weights::LoadedVit;

/// A compiled ViT encoder runner. Holds the current head/FFN masks (default
/// all-ones = no pruning); [`Self::set_masks`] swaps them with no recompilation.
pub struct VitRunner {
    compiled: CompiledGraph,
    cfg: VitConfig,
    preprocess: PreprocessWeights,
    device: Device,
    batch: usize,
    head_mask: Vec<f32>,
    ffn_mask: Vec<f32>,
}

impl VitRunner {
    /// Compile `cfg` for `device` and upload `loaded`'s params.
    pub fn from_loaded(
        cfg: VitConfig,
        loaded: LoadedVit,
        device: Device,
        batch: usize,
    ) -> Result<Self> {
        rlx_core::validate_standard_device("vit-elastic", device)?;

        // Same wgpu arena-reuse workaround `rlx-uni2` uses for this encoder shape.
        if matches!(device, Device::Gpu | Device::WebGpu)
            && !rlx_ir::env::flag("RLX_ARENA_NO_REUSE")
        {
            rlx_ir::env::set("RLX_ARENA_NO_REUSE", "1");
        }

        let vg = build_vit_graph(&cfg, batch);
        let head_mask = vg.ones_head_mask();
        let ffn_mask = vg.ones_ffn_mask();
        let param_names = vg.param_names();
        let VitGraph { graph, .. } = vg;

        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        for name in &param_names {
            let data = loaded
                .params
                .get(name)
                .ok_or_else(|| anyhow!("missing prepared param for graph node {name}"))?;
            compiled.set_param(name, data);
        }

        Ok(Self {
            compiled,
            cfg,
            preprocess: loaded.preprocess,
            device,
            batch,
            head_mask,
            ffn_mask,
        })
    }

    pub fn config(&self) -> &VitConfig {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn batch(&self) -> usize {
        self.batch
    }
    pub fn preprocess(&self) -> &PreprocessWeights {
        &self.preprocess
    }

    /// Set the per-channel head/FFN masks (`[L·H]` / `[L·inner]`). All-ones =
    /// the unpruned model. No recompilation.
    pub fn set_masks(&mut self, head_mask: Vec<f32>, ffn_mask: Vec<f32>) -> Result<()> {
        let hl = self.cfg.num_hidden_layers * self.cfg.hidden_size;
        let fl = self.cfg.num_hidden_layers * self.cfg.ffn_inner();
        if head_mask.len() != hl || ffn_mask.len() != fl {
            return Err(anyhow!(
                "mask length mismatch: head {} (want {hl}), ffn {} (want {fl})",
                head_mask.len(),
                ffn_mask.len()
            ));
        }
        self.head_mask = head_mask;
        self.ffn_mask = ffn_mask;
        Ok(())
    }

    /// Reset masks to all-ones (unpruned).
    pub fn reset_masks(&mut self) {
        self.head_mask.iter_mut().for_each(|m| *m = 1.0);
        self.ffn_mask.iter_mut().for_each(|m| *m = 1.0);
    }

    /// Forward from an assembled `"hidden"` tensor `[B·seq·H]`; returns the
    /// post-norm token sequence `[B·seq·H]`.
    pub fn forward_hidden(&mut self, hidden: &[f32]) -> Result<Vec<f32>> {
        let outs = self.compiled.run(&[
            ("hidden", hidden),
            ("head_mask", self.head_mask.as_slice()),
            ("ffn_mask", self.ffn_mask.as_slice()),
        ]);
        outs.into_iter()
            .next()
            .ok_or_else(|| anyhow!("vit forward returned no output"))
    }

    /// Pooled `[CLS]` embeddings (`[B]` vectors of length `H`) from a `"hidden"`.
    pub fn embed_hidden(&mut self, hidden: &[f32]) -> Result<Vec<Vec<f32>>> {
        let flat = self.forward_hidden(hidden)?;
        let seq = self.cfg.seq_len();
        let h = self.cfg.hidden_size;
        let per = seq * h;
        Ok((0..self.batch)
            .map(|b| flat[b * per..b * per + h].to_vec())
            .collect())
    }

    /// End-to-end forward on a single HWC u8 image (resized + normalized to
    /// `cfg.img_size`), broadcast over the batch. Returns the pooled `[CLS]`
    /// embeddings and the flat token sequence.
    pub fn predict_image(
        &mut self,
        rgb: &[u8],
        h_in: usize,
        w_in: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let img = self.cfg.img_size;
        let one = rgb_u8_to_imagenet_nchw(rgb, h_in, w_in, img);
        let mut nchw = Vec::with_capacity(one.len() * self.batch);
        for _ in 0..self.batch {
            nchw.extend_from_slice(&one);
        }
        let hidden = assemble_hidden(&self.preprocess, &nchw, self.batch)?;
        let tokens = self.forward_hidden(&hidden)?;
        let seq = self.cfg.seq_len();
        let h = self.cfg.hidden_size;
        let per = seq * h;
        let emb = (0..self.batch)
            .map(|b| tokens[b * per..b * per + h].to_vec())
            .collect();
        Ok((emb, tokens))
    }
}
