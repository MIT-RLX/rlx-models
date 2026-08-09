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

//! Qwen3.5 VLM vision encoder — CPU encode + multimodal prompt helpers.

use super::config::MmProjConfig;
use super::flow::build_qwen35_vision_built;
use super::preprocess::{build_vision_position_hw, preprocess_rgb, vision_rope_feeds};
use super::weights::MmProjWeights;
use anyhow::{Context, Result};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::GgufLoader;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Vision tower forward output.
#[derive(Debug, Clone)]
pub struct VisionEncodeOutput {
    pub embeddings: Vec<f32>,
    pub grid_x: usize,
    pub grid_y: usize,
    pub n_tokens: usize,
}

/// Vision encoder wrapping compiled mmproj graphs (one per image size, cached).
/// Runs on the caller's backend (Metal / MLX / CoreML / CPU); falls back to CPU if
/// the requested backend can't compile the tower, so VLM keeps working everywhere.
pub struct Qwen35VisionEncoder {
    cfg: MmProjConfig,
    weights: MmProjWeights,
    /// Compiled graph per `(img_w, img_h)` — smart-resize yields a few distinct
    /// sizes across a session; caching avoids the multi-second recompile (and the
    /// cold first-run param upload) on every image.
    cache: std::collections::HashMap<(usize, usize), rlx_runtime::CompiledGraph>,
    /// Backend actually in use (may be CPU after a fallback).
    device: Device,
}

/// Build the vision graph and compile it on `device`, falling back to CPU when the
/// backend can't compile the tower (e.g. an op it doesn't cover). Params are baked
/// into the compiled graph, so the caller keeps only the graph + the device used.
fn compile_vision(
    cfg: &MmProjConfig,
    weights: &MmProjWeights,
    img_w: usize,
    img_h: usize,
    device: Device,
) -> Result<(rlx_runtime::CompiledGraph, Device)> {
    // Decompose SDPA into matmul→softmax→matmul on GPU backends (the fused ViT
    // attention kernel is ~10× slower there; batched GEMM is fast), but keep the
    // fused op on CPU where its BLAS attention already beats the decomposition.
    // Only sets the default — an explicit RLX_QWEN35_VISION_SDPA_DECOMP wins.
    if std::env::var("RLX_QWEN35_VISION_SDPA_DECOMP").is_err() {
        // SAFETY: single-threaded runner build; env read at vision HIR build below.
        unsafe {
            std::env::set_var(
                "RLX_QWEN35_VISION_SDPA_DECOMP",
                if device == Device::Cpu { "0" } else { "1" },
            );
        }
    }
    let built = build_qwen35_vision_built(cfg, weights, img_w, img_h)?;
    if device == Device::Cpu {
        return Ok((compile_built(built, Device::Cpu)?, Device::Cpu));
    }
    let t = std::time::Instant::now();
    match compile_built(built, device) {
        Ok(c) => {
            eprintln!(
                "[qwen35] vision encoder compiled on {device:?} ({img_w}x{img_h}) in {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
            Ok((c, device))
        }
        Err(e) => {
            eprintln!("[qwen35] vision encoder: {device:?} compile failed ({e}); using CPU");
            let built = build_qwen35_vision_built(cfg, weights, img_w, img_h)?;
            Ok((compile_built(built, Device::Cpu)?, Device::Cpu))
        }
    }
}

impl Qwen35VisionEncoder {
    /// Load mmproj GGUF from disk and compile for the given image size on `device`.
    pub fn from_mmproj(
        path: impl AsRef<Path>,
        img_w: usize,
        img_h: usize,
        device: Device,
    ) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.to_str().context("mmproj path utf8")?;
        let t_io = std::time::Instant::now();
        let mut loader = GgufLoader::from_file(path_str)?;
        let cfg = MmProjConfig::from_gguf(loader.file())?;
        let weights = MmProjWeights::from_loader(&cfg, &mut loader)?;
        eprintln!(
            "[qwen35] mmproj weights loaded (F32) in {:.1}ms",
            t_io.elapsed().as_secs_f64() * 1e3
        );
        Self::from_parts(cfg, weights, img_w, img_h, device)
    }

    /// Build from already-loaded config + weights on `device` (tests / inline mmproj).
    /// Compiles `(img_w, img_h)` up front — this also probes the backend so the
    /// CPU fallback (if any) is resolved at load, not on the first image.
    pub fn from_parts(
        cfg: MmProjConfig,
        weights: MmProjWeights,
        img_w: usize,
        img_h: usize,
        device: Device,
    ) -> Result<Self> {
        let (compiled, device) = compile_vision(&cfg, &weights, img_w, img_h, device)?;
        let mut cache = std::collections::HashMap::new();
        cache.insert((img_w, img_h), compiled);
        Ok(Self {
            cfg,
            weights,
            cache,
            device,
        })
    }

    pub fn config(&self) -> &MmProjConfig {
        &self.cfg
    }

    /// Encode an RGB u8 buffer. Compiles (and caches) a graph per smart-resized
    /// size the first time it's seen; repeat sizes reuse the warm cached graph.
    pub fn encode_rgb(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<VisionEncodeOutput> {
        let trace = std::env::var("RLX_QWEN35_VISION_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        let (nchw, tw, th) = preprocess_rgb(rgb, w, h, &self.cfg);
        let t_pre = t0.elapsed();

        // Compile-on-miss, then reuse: the second+ image at a given size skips the
        // multi-second compile AND the cold first-run (params already resident).
        let tc = std::time::Instant::now();
        if !self.cache.contains_key(&(tw, th)) {
            let (compiled, device) = compile_vision(&self.cfg, &self.weights, tw, th, self.device)?;
            self.device = device;
            self.cache.insert((tw, th), compiled);
        }
        let t_compile = tc.elapsed();

        let (gx, gy) = self.cfg.output_grid(tw, th);
        let n_tokens = gx * gy;
        let proj = self.cfg.llm_hidden_size;
        let position_hw = build_vision_position_hw(tw, th, &self.cfg);
        let head_dim = self.cfg.n_embd / self.cfg.n_head;
        let (rope_cos, rope_sin) = vision_rope_feeds(&position_hw, head_dim);

        let compiled = self.cache.get_mut(&(tw, th)).expect("graph cached above");
        let tr = std::time::Instant::now();
        let outs = compiled.run(&[
            ("image", &nchw),
            ("vision_rope_cos", &rope_cos),
            ("vision_rope_sin", &rope_sin),
        ]);
        let t_run = tr.elapsed();
        if trace {
            eprintln!(
                "[qwen35] vision encode breakdown: preprocess={:.1}ms compile(miss)={:.1}ms run={:.1}ms",
                t_pre.as_secs_f64() * 1e3,
                t_compile.as_secs_f64() * 1e3,
                t_run.as_secs_f64() * 1e3
            );
        }
        let emb = outs
            .into_iter()
            .next()
            .context("vision graph produced no outputs")?;

        anyhow::ensure!(
            emb.len() == n_tokens * proj,
            "vision output len {} != n_tokens*proj {}*{}",
            emb.len(),
            n_tokens,
            proj
        );

        Ok(VisionEncodeOutput {
            embeddings: emb,
            grid_x: gx,
            grid_y: gy,
            n_tokens,
        })
    }
}

/// Convenience: load encoder from path string on `device`.
pub fn load_vision_encoder(
    mmproj_path: &str,
    img_w: usize,
    img_h: usize,
    device: Device,
) -> Result<Qwen35VisionEncoder> {
    Qwen35VisionEncoder::from_mmproj(PathBuf::from(mmproj_path), img_w, img_h, device)
}

#[cfg(feature = "qwen35-vlm")]
pub fn encode_image_file(
    encoder: &mut Qwen35VisionEncoder,
    path: &str,
) -> Result<VisionEncodeOutput> {
    let (rgb, w, h) = super::preprocess::load_rgb_image(path)?;
    encoder.encode_rgb(&rgb, w, h)
}
