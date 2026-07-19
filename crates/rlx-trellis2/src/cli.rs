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

//! `rlx-trellis2` CLI — image → textured 3D mesh (OBJ / colored PLY).

use crate::config::PipelineType;
use crate::pipeline::{Trellis2Input, Trellis2Runner};
use crate::preprocess::PreprocessOptions;
use crate::weights::CheckpointPaths;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::{Path, PathBuf};

/// Parse args and run image→mesh.
pub fn run(args: &[String]) -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut ss_decoder_dir: Option<PathBuf> = None;
    let mut dinov3_weights: Option<PathBuf> = None;
    let mut dinov3_config: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut pipeline_type = "512".to_string();
    let mut seed = 42u64;
    let mut max_num_tokens = 49_152usize;
    let mut steps_override: Option<usize> = None;
    let mut shape_only = false;
    let mut no_rembg = false;
    let mut no_dino = false;
    let mut eager_dit = false;
    let mut dry = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => model_dir = Some(req(args, &mut i)?.into()),
            "--ss-decoder-dir" => ss_decoder_dir = Some(req(args, &mut i)?.into()),
            "--dinov3-weights" => dinov3_weights = Some(req(args, &mut i)?.into()),
            "--dinov3-config" => dinov3_config = Some(req(args, &mut i)?.into()),
            "--image" => image = Some(req(args, &mut i)?.into()),
            "--output" | "-o" => output = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--pipeline-type" => pipeline_type = req(args, &mut i)?,
            "--seed" => {
                seed = req(args, &mut i)?.parse().context("--seed: u64")?;
            }
            "--max-num-tokens" => {
                max_num_tokens = req(args, &mut i)?
                    .parse()
                    .context("--max-num-tokens: usize")?;
            }
            "--steps" => {
                steps_override = Some(req(args, &mut i)?.parse().context("--steps: usize")?);
            }
            "--shape-only" => {
                shape_only = true;
                i += 1;
            }
            "--no-rembg" => {
                no_rembg = true;
                i += 1;
            }
            "--no-dino" => {
                no_dino = true;
                i += 1;
            }
            "--eager-dit" => {
                eager_dit = true;
                i += 1;
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let model_dir = model_dir.ok_or_else(|| anyhow!("--model-dir is required"))?;
    let pipe_ty = PipelineType::parse(&pipeline_type)?;
    let device = parse_standard_device("trellis2", &device)?;

    if dry {
        let (pipe, paths) =
            CheckpointPaths::resolve_lenient(&model_dir, ss_decoder_dir.as_deref())?;
        let missing = paths.missing_for(pipe_ty, !shape_only);
        println!("pipeline: {}", pipe.args.default_pipeline_type);
        println!("requested: {pipeline_type} -> {pipe_ty:?}");
        println!("model_dir: {}", paths.root.display());
        println!("ss_decoder: {}", paths.sparse_structure_decoder.display());
        println!("ss_flow: {}", paths.sparse_structure_flow.display());
        println!("shape_dec: {}", paths.shape_slat_decoder.display());
        println!("shape_512: {}", paths.shape_slat_flow_512.display());
        if let Some(p) = &paths.shape_slat_flow_1024 {
            println!("shape_1024: {}", p.display());
        }
        if let Some(p) = &paths.tex_slat_decoder {
            println!("tex_dec: {}", p.display());
        }
        if let Some(p) = &paths.tex_slat_flow_512 {
            println!("tex_512: {}", p.display());
        }
        if missing.is_empty() {
            println!("checkpoints: ok for {pipe_ty:?} (shape_only={shape_only})");
        } else {
            println!("missing:");
            for m in missing {
                println!("  - {m}");
            }
        }
        return Ok(());
    }

    let image = image.ok_or_else(|| anyhow!("--image is required (or use --dry)"))?;
    if !no_dino && dinov3_weights.is_none() {
        bail!("--dinov3-weights is required (or pass --no-dino)");
    }
    let output = output.unwrap_or_else(|| {
        if shape_only {
            PathBuf::from("trellis2_out.obj")
        } else {
            PathBuf::from("trellis2_out.glb")
        }
    });

    let mut builder = Trellis2Runner::builder()
        .model_dir(model_dir)
        .device(device)
        .pipeline_type(pipe_ty)
        .shape_only(shape_only)
        .max_num_tokens(max_num_tokens)
        .no_dino(no_dino)
        .eager_dit(eager_dit);
    if let Some(w) = dinov3_weights {
        builder = builder.dinov3_weights(w);
    }
    if let Some(n) = steps_override {
        builder = builder.steps_override(n);
    }
    if let Some(p) = ss_decoder_dir {
        builder = builder.ss_decoder_dir(p);
    }
    if let Some(p) = dinov3_config {
        builder = builder.dinov3_config(p);
    }

    let mut runner = builder.build()?;
    let img = image::open(&image).with_context(|| format!("opening {}", image.display()))?;
    let out = runner.generate(Trellis2Input {
        image: &img,
        seed,
        preprocess: PreprocessOptions {
            allow_rgb_fallback: no_rembg,
            ..PreprocessOptions::default()
        },
    })?;

    let bytes_or_text = export_mesh(&output, &out.mesh, out.tex_slat.is_some())?;
    match bytes_or_text {
        Export::Text(s) => {
            std::fs::write(&output, s).with_context(|| format!("writing {}", output.display()))?;
        }
        Export::Binary(b) => {
            std::fs::write(&output, b).with_context(|| format!("writing {}", output.display()))?;
        }
    }
    eprintln!(
        "rlx-trellis2: wrote {} (res={}, verts={}, faces={}, tex={})",
        output.display(),
        out.resolution,
        out.mesh.mesh.vertices.len(),
        out.mesh.mesh.faces.len(),
        out.tex_slat.is_some(),
    );
    Ok(())
}

enum Export {
    Text(String),
    Binary(Vec<u8>),
}

fn export_mesh(path: &Path, mesh: &crate::mesh::MeshWithPbr, has_tex: bool) -> Result<Export> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    Ok(match ext.as_str() {
        "ply" => Export::Text(mesh.to_ply()),
        "obj" => Export::Text(mesh.to_obj()),
        "glb" => Export::Binary(mesh.to_glb()),
        _ if has_tex => {
            eprintln!(
                "rlx-trellis2: unknown extension {:?}; writing colored PLY (use .ply, .glb, or .obj)",
                path.extension()
            );
            Export::Text(mesh.to_ply())
        }
        _ => Export::Text(mesh.to_obj()),
    })
}

fn print_help() {
    eprintln!(
        "rlx-trellis2 — Microsoft TRELLIS.2-4B image→3D\n\
         \n\
         Required:\n\
           --model-dir DIR          TRELLIS.2-4B root (pipeline.json + ckpts/)\n\
           --image PATH             input image (RGBA cutout preferred)\n\
           --dinov3-weights PATH    DINOv3 ViT-L/16 safetensors (unless --no-dino)\n\
         \n\
         Optional:\n\
           --output / -o PATH       .glb (UV+PBR textures), .ply (vertex colors),\n\
                                    or .obj (geom); default trellis2_out.glb\n\
                                    (or .obj with --shape-only)\n\
           --ss-decoder-dir DIR     microsoft/TRELLIS-image-large/ckpts\n\
           --dinov3-config PATH     HF config.json for ViT-L/16\n\
           --pipeline-type T        512 | 1024 | 1024_cascade | 1536_cascade\n\
                                    (default CLI: 512)\n\
           --seed N                 RNG seed (default 42)\n\
           --steps N                override Euler steps for all samplers\n\
           --max-num-tokens N       cascade token cap (default 49152)\n\
           --device cpu|metal|…     DINOv3 + compiled DiT device\n\
                                    (Metal/MLX/CUDA accelerate DiT via AdaLN/SDPA;\n\
                                    CPU uses the host DiT reference)\n\
           --eager-dit              force host DiT even on GPU devices\n\
           --shape-only             skip texture DiT/VAE\n\
           --no-dino                zero image cond (no DINOv3 weights)\n\
           --no-rembg               allow RGB without alpha (no BiRefNet)\n\
           --dry                    resolve checkpoints and exit\n\
         \n\
         Texture needs ckpts/tex_dec_* + ckpts/slat_flow_imgshape2tex_*_512.\n\
         Atlas GLB UVs are per-vertex (not official o_voxel remesh bake)."
    );
}
