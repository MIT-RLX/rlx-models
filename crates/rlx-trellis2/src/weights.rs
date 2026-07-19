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

//! Checkpoint path resolution + weight loading for TRELLIS.2-4B.

use crate::config::{
    DitConfig, PipelineConfig, PipelineType, SparseStructureVaeArgs, SparseVaeConfig,
};
use crate::dit_flow::CompiledDit;
use anyhow::{Context, Result, bail, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One loaded DiT (config + weights + optional compiled sessions).
pub struct LoadedDit {
    pub cfg: DitConfig,
    pub weights: WeightMap,
    pub path: PathBuf,
    /// Compiled graphs keyed by `(n_pos, n_cond)`.
    compiled: HashMap<(usize, usize), CompiledDit>,
}

impl LoadedDit {
    /// Compile (lazily) a fixed-shape DiT session for `device`.
    pub fn ensure_compiled(
        &mut self,
        device: Device,
        n_pos: usize,
        n_cond: usize,
    ) -> Result<&mut CompiledDit> {
        let key = (n_pos, n_cond);
        if !self.compiled.contains_key(&key) {
            eprintln!(
                "rlx-trellis2: compiling DiT {} (n_pos={n_pos}, n_cond={n_cond}, device={device:?})…",
                self.path.display()
            );
            let t0 = Instant::now();
            let session =
                crate::dit_flow::compile_dit(&self.cfg, &self.weights, device, n_pos, n_cond)?;
            eprintln!(
                "rlx-trellis2: DiT compile done in {:.1}s",
                t0.elapsed().as_secs_f64()
            );
            self.compiled.insert(key, session);
        }
        Ok(self.compiled.get_mut(&key).expect("just inserted"))
    }

    /// Compiled forward (exact `n_pos`). Host builds `t_mod` + NeoX RoPE tables.
    pub fn forward_compiled(
        &mut self,
        device: Device,
        tokens: &[f32],
        coords: &[f32],
        n_pos: usize,
        cond: &[f32],
        n_cond: usize,
        t: f32,
    ) -> Result<Vec<f32>> {
        use crate::dit_host::shared_modulation;
        use crate::rope::RopeTables;
        let t_mod = shared_modulation(&self.cfg, &self.weights, t)?;
        let tables = RopeTables::neox(
            coords,
            n_pos,
            self.cfg.head_dim(),
            3,
            self.cfg.args.rope_freq,
        );
        let compiled = self.ensure_compiled(device, n_pos, n_cond)?;
        compiled.forward(tokens, &t_mod, cond, &tables.cos, &tables.sin)
    }

    /// Pad to `n_bucket`, run compiled forward, slice back to `n_real` tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_compiled_padded(
        &mut self,
        device: Device,
        tokens: &[f32],
        coords: &[f32],
        n_real: usize,
        n_bucket: usize,
        cond: &[f32],
        n_cond: usize,
        t: f32,
    ) -> Result<Vec<f32>> {
        ensure!(
            n_real > 0 && n_real <= n_bucket,
            "n_real={n_real} bucket={n_bucket}"
        );
        let in_ch = self.cfg.args.in_channels;
        let out_ch = self.cfg.args.out_channels;
        let mut tok = vec![0.0f32; n_bucket * in_ch];
        tok[..n_real * in_ch].copy_from_slice(tokens);
        let mut coords_p = vec![0.0f32; n_bucket * 3];
        coords_p[..n_real * 3].copy_from_slice(coords);
        for i in n_real..n_bucket {
            coords_p[i * 3] = 1.0e4;
            coords_p[i * 3 + 1] = 1.0e4;
            coords_p[i * 3 + 2] = 1.0e4;
        }
        let out = self.forward_compiled(device, &tok, &coords_p, n_bucket, cond, n_cond, t)?;
        Ok(out[..n_real * out_ch].to_vec())
    }
}

/// One loaded sparse VAE decoder.
pub struct LoadedSparseVae {
    pub cfg: SparseVaeConfig,
    pub weights: WeightMap,
    pub path: PathBuf,
}

/// Dense sparse-structure decoder.
pub struct LoadedSsDecoder {
    pub cfg: SparseStructureVaeArgs,
    pub weights: WeightMap,
    pub path: PathBuf,
}

/// Resolved on-disk layout for a TRELLIS.2 checkpoint directory.
#[derive(Debug, Clone)]
pub struct CheckpointPaths {
    pub root: PathBuf,
    pub pipeline_json: PathBuf,
    pub sparse_structure_decoder: PathBuf,
    pub sparse_structure_flow: PathBuf,
    pub shape_slat_decoder: PathBuf,
    pub shape_slat_flow_512: PathBuf,
    pub shape_slat_flow_1024: Option<PathBuf>,
    pub tex_slat_decoder: Option<PathBuf>,
    pub tex_slat_flow_512: Option<PathBuf>,
    pub tex_slat_flow_1024: Option<PathBuf>,
}

impl CheckpointPaths {
    /// Resolve stems from `pipeline.json` under `model_dir`, with an optional
    /// override for the external `microsoft/TRELLIS-image-large` structure
    /// decoder directory (containing `ss_dec_conv3d_16l8_fp16.{json,safetensors}`).
    ///
    /// Required local stems must exist; optional cascade/texture stems may be
    /// absent (`None`).
    pub fn resolve(
        model_dir: impl AsRef<Path>,
        ss_decoder_dir: Option<&Path>,
    ) -> Result<(PipelineConfig, Self)> {
        Self::resolve_inner(model_dir, ss_decoder_dir, false)
    }

    /// Like [`Self::resolve`], but missing required stems become empty paths
    /// (for `--dry` inventory) instead of hard errors.
    pub fn resolve_lenient(
        model_dir: impl AsRef<Path>,
        ss_decoder_dir: Option<&Path>,
    ) -> Result<(PipelineConfig, Self)> {
        Self::resolve_inner(model_dir, ss_decoder_dir, true)
    }

    fn resolve_inner(
        model_dir: impl AsRef<Path>,
        ss_decoder_dir: Option<&Path>,
        lenient: bool,
    ) -> Result<(PipelineConfig, Self)> {
        let root = model_dir.as_ref().to_path_buf();
        let pipeline_json = root.join("pipeline.json");
        let pipe = PipelineConfig::from_file(&pipeline_json)?;
        let m = &pipe.args.models;

        let ss_flow = resolve_local_stem(&root, &m.sparse_structure_flow_model, lenient)?;
        let shape_dec = resolve_local_stem(&root, &m.shape_slat_decoder, lenient)?;
        let shape_512 = resolve_local_stem(&root, &m.shape_slat_flow_model_512, lenient)?;
        let shape_1024 = resolve_local_stem_opt(&root, &m.shape_slat_flow_model_1024);
        let tex_dec = resolve_local_stem_opt(&root, &m.tex_slat_decoder);
        let tex_512 = resolve_local_stem_opt(&root, &m.tex_slat_flow_model_512);
        let tex_1024 = resolve_local_stem_opt(&root, &m.tex_slat_flow_model_1024);

        let ss_dec = match resolve_ss_decoder(&root, &m.sparse_structure_decoder, ss_decoder_dir) {
            Ok(p) => p,
            Err(e) if lenient => {
                eprintln!("rlx-trellis2: ss decoder not resolved ({e:#})");
                PathBuf::new()
            }
            Err(e) => return Err(e),
        };

        let paths = Self {
            root,
            pipeline_json,
            sparse_structure_decoder: ss_dec,
            sparse_structure_flow: ss_flow,
            shape_slat_decoder: shape_dec,
            shape_slat_flow_512: shape_512,
            shape_slat_flow_1024: shape_1024,
            tex_slat_decoder: tex_dec,
            tex_slat_flow_512: tex_512,
            tex_slat_flow_1024: tex_1024,
        };
        Ok((pipe, paths))
    }

    /// Files required for a given pipeline type (shape path; texture optional).
    pub fn missing_for(&self, pipeline: PipelineType, need_texture: bool) -> Vec<String> {
        let mut miss = Vec::new();
        for (label, p) in [
            ("sparse_structure_decoder", &self.sparse_structure_decoder),
            ("sparse_structure_flow", &self.sparse_structure_flow),
            ("shape_slat_decoder", &self.shape_slat_decoder),
            ("shape_slat_flow_512", &self.shape_slat_flow_512),
        ] {
            push_if_missing(&mut miss, label, p);
        }
        match pipeline {
            PipelineType::Res512 => {
                if need_texture {
                    push_opt(&mut miss, "tex_slat_decoder", &self.tex_slat_decoder);
                    push_opt(&mut miss, "tex_slat_flow_512", &self.tex_slat_flow_512);
                }
            }
            PipelineType::Res1024 => {
                push_opt(
                    &mut miss,
                    "shape_slat_flow_1024",
                    &self.shape_slat_flow_1024,
                );
                if need_texture {
                    push_opt(&mut miss, "tex_slat_decoder", &self.tex_slat_decoder);
                    push_opt(&mut miss, "tex_slat_flow_1024", &self.tex_slat_flow_1024);
                }
            }
            PipelineType::Cascade1024 | PipelineType::Cascade1536 => {
                push_opt(
                    &mut miss,
                    "shape_slat_flow_1024",
                    &self.shape_slat_flow_1024,
                );
                if need_texture {
                    push_opt(&mut miss, "tex_slat_decoder", &self.tex_slat_decoder);
                    push_opt(&mut miss, "tex_slat_flow_1024", &self.tex_slat_flow_1024);
                }
            }
        }
        miss
    }
}

fn push_if_missing(miss: &mut Vec<String>, label: &str, stem: &Path) {
    if stem.as_os_str().is_empty() {
        miss.push(format!("{label}: (not resolved)"));
        return;
    }
    let st = stem.with_extension("safetensors");
    let js = stem.with_extension("json");
    if !st.is_file() {
        miss.push(format!("{label}: {}", st.display()));
    }
    if !js.is_file() {
        miss.push(format!("{label} config: {}", js.display()));
    }
}

fn push_opt(miss: &mut Vec<String>, label: &str, stem: &Option<PathBuf>) {
    match stem {
        Some(p) => push_if_missing(miss, label, p),
        None => miss.push(format!("{label}: (not found)")),
    }
}

fn resolve_local_stem(root: &Path, rel: &str, lenient: bool) -> Result<PathBuf> {
    let p = root.join(rel);
    let st = p.with_extension("safetensors");
    let js = p.with_extension("json");
    if st.is_file() && js.is_file() {
        return Ok(p);
    }
    if lenient {
        return Ok(p);
    }
    bail!(
        "missing local checkpoint stem {} (need {}.safetensors + {}.json)",
        p.display(),
        p.display(),
        p.display()
    );
}

fn resolve_local_stem_opt(root: &Path, rel: &str) -> Option<PathBuf> {
    let p = root.join(rel);
    let st = p.with_extension("safetensors");
    let js = p.with_extension("json");
    if st.is_file() && js.is_file() {
        Some(p)
    } else {
        None
    }
}

/// Resolve `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16`.
fn resolve_ss_decoder(
    model_root: &Path,
    spec: &str,
    override_dir: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(dir) = override_dir {
        let stem = dir.join("ss_dec_conv3d_16l8_fp16");
        if stem.with_extension("safetensors").is_file() {
            return Ok(stem);
        }
        // allow pointing at the stem itself
        if dir.with_extension("safetensors").is_file() {
            return Ok(dir.to_path_buf());
        }
        bail!(
            "ss decoder override {} does not contain ss_dec_conv3d_16l8_fp16.safetensors",
            dir.display()
        );
    }

    // Local relative path (rare but valid).
    if !spec.contains('/') || spec.starts_with("ckpts/") {
        if let Ok(p) = resolve_local_stem(model_root, spec, false) {
            return Ok(p);
        }
    }

    // HuggingFace hub cache: models--microsoft--TRELLIS-image-large
    if let Some(p) = find_hf_ss_decoder() {
        return Ok(p);
    }

    // Sibling checkout next to the 4B dir.
    if let Some(parent) = model_root.parent() {
        let cand = parent
            .join("TRELLIS-image-large")
            .join("ckpts")
            .join("ss_dec_conv3d_16l8_fp16");
        if cand.with_extension("safetensors").is_file() {
            return Ok(cand);
        }
    }

    bail!(
        "could not resolve sparse_structure_decoder {spec:?}; pass --ss-decoder-dir pointing at \
         microsoft/TRELLIS-image-large/ckpts (or a directory with ss_dec_conv3d_16l8_fp16.*)"
    )
}

fn find_hf_ss_decoder() -> Option<PathBuf> {
    let home = dirs_home()?;
    let hub = home.join(".cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots");
    if !hub.is_dir() {
        return None;
    }
    for snap in std::fs::read_dir(&hub).ok()?.flatten() {
        let stem = snap.path().join("ckpts").join("ss_dec_conv3d_16l8_fp16");
        if stem.with_extension("safetensors").is_file() && stem.with_extension("json").is_file() {
            return Some(stem);
        }
    }
    None
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub fn load_dit(stem: &Path) -> Result<LoadedDit> {
    let cfg = DitConfig::from_file(stem.with_extension("json"))
        .with_context(|| format!("dit config {}", stem.display()))?;
    let weights = rlx_core::load_weight_map(stem.with_extension("safetensors"), &[])
        .with_context(|| format!("dit weights {}", stem.display()))?;
    Ok(LoadedDit {
        cfg,
        weights,
        path: stem.to_path_buf(),
        compiled: HashMap::new(),
    })
}

pub fn load_sparse_vae(stem: &Path) -> Result<LoadedSparseVae> {
    let cfg = SparseVaeConfig::from_file(stem.with_extension("json"))
        .with_context(|| format!("vae config {}", stem.display()))?;
    let weights = rlx_core::load_weight_map(stem.with_extension("safetensors"), &[])
        .with_context(|| format!("vae weights {}", stem.display()))?;
    Ok(LoadedSparseVae {
        cfg,
        weights,
        path: stem.to_path_buf(),
    })
}

pub fn load_ss_decoder(stem: &Path) -> Result<LoadedSsDecoder> {
    let cfg = SparseStructureVaeArgs::from_file(stem.with_extension("json"))
        .with_context(|| format!("ss decoder config {}", stem.display()))?;
    let weights = rlx_core::load_weight_map(stem.with_extension("safetensors"), &[])
        .with_context(|| format!("ss decoder weights {}", stem.display()))?;
    Ok(LoadedSsDecoder {
        cfg,
        weights,
        path: stem.to_path_buf(),
    })
}
