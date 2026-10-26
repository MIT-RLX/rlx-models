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

//! LFM2-VL's image path: the processor (`Lfm2VlImageProcessor` +
//! `Lfm2VlProcessor`'s token layout), the SigLIP2 NaFlex tower and the
//! pixel-unshuffle projector.
//!
//! An image becomes either one tile at a "smart" size (both sides multiples
//! of 32 px, 64..256 output tokens), or — when it is too large for that — a
//! grid of 512 px tiles plus a thumbnail at the smart size. Each tile is cut
//! into 16 px patches, encoded by the tower (one compiled graph at the
//! processor's 1024-patch maximum, padding masked out), pixel-unshuffled 2x2
//! (4 patches -> one 4608-wide token) and projected into the LM width. The
//! prompt's `<image>` expands to the matching run of image tokens with the
//! `<|image_start|>` / `<|img_row_R_col_C|>` / `<|img_thumbnail|>` /
//! `<|image_end|>` markers.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_core::weight_storage::WeightStorage;
use rlx_ir::DType;
use rlx_runtime::{CompiledGraph, Device};
use rlx_siglip2::flow::{NaflexEncoderSpec, build_naflex_encoder};
use rlx_siglip2::naflex::resize_position_grid;

pub use crate::processor::*;

/// LFM2-VL's vision config (`vision_config` + projector keys of `config.json`).
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    /// Side of the learned position grid (`sqrt(num_patches)`).
    pub pos_side: usize,
    pub layer_norm_eps: f32,
    pub projector_hidden: usize,
    pub text_hidden: usize,
}

impl VisionConfig {
    pub fn from_config(cfg: &serde_json::Value) -> Result<Self> {
        let v = cfg
            .get("vision_config")
            .ok_or_else(|| anyhow!("config.json has no vision_config"))?;
        let u = |o: &serde_json::Value, k: &str| -> Result<usize> {
            o.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("config.json: missing {k}"))
        };
        let num_patches = u(v, "num_patches")?;
        let pos_side = (num_patches as f64).sqrt() as usize;
        ensure!(
            pos_side * pos_side == num_patches,
            "vision num_patches {num_patches} is not a square"
        );
        let act = cfg
            .get("projector_hidden_act")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("gelu");
        ensure!(
            act == "gelu",
            "projector_hidden_act `{act}` (only exact GELU is ported)"
        );
        ensure!(
            !cfg.get("projector_use_layernorm")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            "projector_use_layernorm = true is not ported"
        );
        if let Some(a) = v.get("hidden_act").and_then(serde_json::Value::as_str) {
            ensure!(
                a == "gelu_pytorch_tanh",
                "vision hidden_act `{a}` is not gelu_pytorch_tanh"
            );
        }
        Ok(Self {
            hidden_size: u(v, "hidden_size")?,
            num_layers: u(v, "num_hidden_layers")?,
            num_heads: u(v, "num_attention_heads")?,
            patch_size: u(v, "patch_size")?,
            pos_side,
            layer_norm_eps: v
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-6) as f32,
            projector_hidden: u(cfg, "projector_hidden_size")?,
            text_hidden: u(
                cfg.get("text_config")
                    .ok_or_else(|| anyhow!("config.json has no text_config"))?,
                "hidden_size",
            )?,
        })
    }
}

impl VisionConfig {
    /// The same settings from an llama.cpp mmproj's `clip.*` metadata; the
    /// position grid and projector widths come from the tensor shapes.
    pub fn from_mmproj(gguf: &rlx_gguf::GgufFile) -> Result<Self> {
        let meta = |k: &str| {
            gguf.metadata
                .get(k)
                .ok_or_else(|| anyhow!("mmproj has no `{k}`"))
        };
        let u = |k: &str| -> Result<usize> {
            meta(k)?
                .as_u32()
                .map(|v| v as usize)
                .ok_or_else(|| anyhow!("mmproj `{k}` is not an integer"))
        };
        let proj_type = match meta("clip.projector_type")? {
            rlx_gguf::MetaValue::String(s) => s.clone(),
            other => bail!("clip.projector_type is {other:?}"),
        };
        ensure!(
            proj_type == "lfm2",
            "mmproj projector type `{proj_type}` is not lfm2 (LFM2-VL)"
        );
        let eps = match meta("clip.vision.attention.layer_norm_epsilon")? {
            rlx_gguf::MetaValue::F32(x) => *x,
            rlx_gguf::MetaValue::F64(x) => *x as f32,
            other => bail!("layer_norm_epsilon is {other:?}"),
        };
        let shape = |name: &str| -> Result<Vec<usize>> {
            let t = gguf
                .tensors
                .get(name)
                .ok_or_else(|| anyhow!("mmproj has no tensor {name}"))?;
            Ok(t.shape.iter().rev().copied().collect())
        };
        let hidden = u("clip.vision.embedding_length")?;
        let pos = shape("v.position_embd.weight")?;
        ensure!(
            pos.len() == 2 && pos[1] == hidden,
            "v.position_embd {pos:?}"
        );
        let pos_side = (pos[0] as f64).sqrt() as usize;
        ensure!(
            pos_side * pos_side == pos[0],
            "position grid {} is not a square",
            pos[0]
        );
        let mm1 = shape("mm.1.weight")?;
        Ok(Self {
            hidden_size: hidden,
            num_layers: u("clip.vision.block_count")?,
            num_heads: u("clip.vision.attention.head_count")?,
            patch_size: u("clip.vision.patch_size")?,
            pos_side,
            layer_norm_eps: eps,
            projector_hidden: mm1[0],
            text_hidden: u("clip.vision.projection_dim")?,
        })
    }
}

/// HF name for an llama.cpp clip tensor of an LFM2-VL mmproj (the reverse of
/// llama.cpp's `conversion/lfm2.py` + its tensor map).
fn mmproj_hf_name(gguf: &str) -> Option<String> {
    let fixed = match gguf {
        "v.patch_embd.weight" => Some("embeddings.patch_embedding.weight"),
        "v.patch_embd.bias" => Some("embeddings.patch_embedding.bias"),
        "v.position_embd.weight" => Some("embeddings.position_embedding.weight"),
        "v.post_ln.weight" => Some("post_layernorm.weight"),
        "v.post_ln.bias" => Some("post_layernorm.bias"),
        _ => None,
    };
    if let Some(f) = fixed {
        return Some(format!("{TOWER}.{f}"));
    }
    if let Some(rest) = gguf.strip_prefix("mm.") {
        let (i, tail) = rest.split_once('.')?;
        return Some(format!("{PROJECTOR}.linear_{i}.{tail}"));
    }
    let rest = gguf.strip_prefix("v.blk.")?;
    let (i, tail) = rest.split_once('.')?;
    let (part, kind) = tail.rsplit_once('.')?;
    let hf = match part {
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_out" => "self_attn.out_proj",
        "ffn_up" => "mlp.fc1",
        "ffn_down" => "mlp.fc2",
        "ln1" => "layer_norm1",
        "ln2" => "layer_norm2",
        _ => return None,
    };
    Some(format!("{TOWER}.encoder.layers.{i}.{hf}.{kind}"))
}

/// Every mmproj tensor dequantized and under its HF name. The patch
/// embedding is stored as a conv kernel `[hidden, 3, ps, ps]` (channel
/// first); it goes back to HF's linear `[hidden, ps·ps·3]`, whose columns run
/// `(row, col, channel)` — the order the processor writes a patch in.
fn mmproj_weight_map(
    loader: &mut dyn rlx_core::weight_loader::WeightLoader,
    cfg: &VisionConfig,
) -> Result<WeightMap> {
    let mut tensors = std::collections::HashMap::new();
    for name in loader.remaining_keys() {
        let hf =
            mmproj_hf_name(&name).ok_or_else(|| anyhow!("unexpected mmproj tensor `{name}`"))?;
        let (mut data, mut shape) = loader.take(&name)?;
        if name == "v.patch_embd.weight" {
            let (h, ps) = (cfg.hidden_size, cfg.patch_size);
            ensure!(shape == [h, 3, ps, ps], "v.patch_embd.weight {shape:?}");
            let mut lin = vec![0f32; data.len()];
            for o in 0..h {
                for c in 0..3 {
                    for ry in 0..ps {
                        for rx in 0..ps {
                            lin[o * 3 * ps * ps + (ry * ps + rx) * 3 + c] =
                                data[((o * 3 + c) * ps + ry) * ps + rx];
                        }
                    }
                }
            }
            data = lin;
            shape = vec![h, 3 * ps * ps];
        }
        tensors.insert(hf, (data, shape));
    }
    Ok(WeightMap::from_tensors(tensors))
}

pub(crate) const TOWER: &str = "model.vision_tower.vision_model";
pub(crate) const PROJECTOR: &str = "model.multi_modal_projector";

/// The vision tower and projector, compiled.
pub struct VisionEncoder {
    cfg: VisionConfig,
    proc: ProcessorConfig,
    /// Learned position grid `[side · side, hidden]`.
    pos_grid: Vec<f32>,
    tower: CompiledGraph,
    projector: CompiledGraph,
    /// Rows the projector graph is compiled for (the most tokens a tile makes).
    proj_rows: usize,
}

impl VisionEncoder {
    /// Load the vision half of an LFM2-VL checkpoint directory, its linear
    /// weights stored as `storage` says ([`WeightStorage::Auto`] keeps the
    /// checkpoint's bf16).
    pub fn load(dir: &Path, device: Device, storage: WeightStorage) -> Result<Self> {
        let raw =
            std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
        let config: serde_json::Value =
            serde_json::from_str(&raw).context("parsing config.json")?;
        let cfg = VisionConfig::from_config(&config)?;
        let proc = ProcessorConfig::from_dir(dir)?;

        let mut want: HashSet<String> = HashSet::new();
        let names = rlx_core::SafetensorsSource::File(dir.to_path_buf())
            .loader()?
            .remaining_keys();
        for n in names {
            if n.starts_with(TOWER) || n.starts_with(PROJECTOR) {
                want.insert(n);
            }
        }
        ensure!(
            !want.is_empty(),
            "no {TOWER}.* tensors in {}",
            dir.display()
        );
        let stored = rlx_core::weight_registry::open_weight_loader(dir)?
            .stored_dtype(&format!("{TOWER}.encoder.layers.0.self_attn.q_proj.weight"));
        let weights = WeightMap::from_safetensors_dir_selected(dir, &want)?;
        Self::from_weights(cfg, proc, weights, storage.resolve(stored), device)
    }

    /// Load the vision half from an llama.cpp `mmproj` GGUF
    /// (`clip.projector_type = lfm2`, as in `LiquidAI/d1-3B-GGUF`).
    ///
    /// Quantized tensors (Q8_0, F16) are dequantized to f32 — the tower is
    /// 0.4B parameters — and renamed to the HF names the builders use. The
    /// processor settings come from `processor_config.json` in `config_dir`
    /// when it is there, else the LFM2-VL defaults.
    pub fn load_mmproj(
        mmproj: &Path,
        config_dir: Option<&Path>,
        device: Device,
        storage: WeightStorage,
    ) -> Result<Self> {
        let mut loader = rlx_core::weight_registry::open_weight_loader(mmproj)
            .with_context(|| format!("opening {}", mmproj.display()))?;
        let gguf = loader
            .gguf_file()
            .ok_or_else(|| anyhow!("{} is not a GGUF file", mmproj.display()))?;
        let cfg = VisionConfig::from_mmproj(gguf)?;
        let proc = match config_dir {
            Some(d) => ProcessorConfig::from_dir(d)?,
            None => ProcessorConfig::default(),
        };
        // `Auto`: an F16 mmproj stays f16; a Q8_0 one is dequantized, and
        // rounding it to a half type would not be lossless, so it stays f32.
        let stored = loader.stored_dtype("v.blk.0.attn_q.weight");
        let mut dtype = storage.resolve(stored);
        // On Metal an F16 tower is the worse choice under `Auto`: MPSGraph
        // bakes only f32 params into constants and casts f16 ones on every run
        // — measured 0.5 GB MORE peak and 20% slower than f32 for d1's F16
        // mmproj (bf16 is fine: it unpacks into a reused scratch). An explicit
        // `WeightStorage::F16` is still honoured.
        if storage == WeightStorage::Auto && dtype == DType::F16 && device == Device::Metal {
            dtype = DType::F32;
        }
        let weights = mmproj_weight_map(loader.as_mut(), &cfg)?;
        Self::from_weights(cfg, proc, weights, dtype, device)
    }

    fn from_weights(
        cfg: VisionConfig,
        proc: ProcessorConfig,
        mut weights: WeightMap,
        weight_dtype: DType,
        device: Device,
    ) -> Result<Self> {
        ensure!(
            proc.encoder_patch_size == cfg.patch_size,
            "processor patch size {} != tower patch size {}",
            proc.encoder_patch_size,
            cfg.patch_size
        );
        let (pos_grid, pshape) =
            weights.take(&format!("{TOWER}.embeddings.position_embedding.weight"))?;
        ensure!(
            pshape == [cfg.pos_side * cfg.pos_side, cfg.hidden_size],
            "position_embedding {pshape:?}"
        );

        let seq = proc.max_num_patches();
        let spec = NaflexEncoderSpec {
            prefix: TOWER.into(),
            width: cfg.hidden_size,
            heads: cfg.num_heads,
            layers: cfg.num_layers,
            patch_dim: 3 * cfg.patch_size * cfg.patch_size,
            seq,
            eps: cfg.layer_norm_eps,
            weight_dtype,
        };
        let tower =
            rlx_core::flow_util::compile_built(build_naflex_encoder(&spec, &mut weights)?, device)?;

        let ds = proc.downsample_factor;
        let proj_rows = seq / (ds * ds);
        let projector = rlx_core::flow_util::compile_built(
            build_projector(&cfg, ds, proj_rows, weight_dtype, &mut weights)?,
            device,
        )?;
        Ok(Self {
            cfg,
            proc,
            pos_grid,
            tower,
            projector,
            proj_rows,
        })
    }

    pub fn processor(&self) -> &ProcessorConfig {
        &self.proc
    }

    /// Projected features `[n_tokens, text_hidden]` for one processed image,
    /// in the order its placeholder lists image tokens.
    pub fn encode(&mut self, img: &ProcessedImage) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for tile in &img.tiles {
            out.extend(self.encode_tile(tile)?);
        }
        Ok(out)
    }

    /// Projected features of one tile (`[ph·pw/4, text_hidden]`).
    pub fn encode_tile(&mut self, tile: &Tile) -> Result<Vec<f32>> {
        let hidden = self.cfg.hidden_size;
        let pd = 3 * self.cfg.patch_size * self.cfg.patch_size;
        let seq = self.proc.max_num_patches();
        let n = tile.ph * tile.pw;
        let ds = self.proc.downsample_factor;
        ensure!(
            n <= seq,
            "tile of {n} patches exceeds the {seq}-patch tower"
        );
        ensure!(
            tile.ph.is_multiple_of(ds) && tile.pw.is_multiple_of(ds),
            "tile grid {}x{} is not divisible by {ds}",
            tile.ph,
            tile.pw
        );
        let mut pv = vec![0f32; seq * pd];
        pv[..n * pd].copy_from_slice(&tile.pixel_values);
        // SigLIP2 pads past the real patches with the resized grid's first row.
        let pos = resize_position_grid(&self.pos_grid, self.cfg.pos_side, hidden, tile.ph, tile.pw);
        let mut pos_embed = vec![0f32; seq * hidden];
        for p in 0..seq {
            let src = if p < n { p } else { 0 };
            pos_embed[p * hidden..(p + 1) * hidden]
                .copy_from_slice(&pos[src * hidden..(src + 1) * hidden]);
        }
        let mut mask = vec![0f32; seq];
        mask[..n].fill(1.0);
        let feats = self
            .tower
            .run(&[
                ("pixel_values", pv.as_slice()),
                ("pos_embed", pos_embed.as_slice()),
                ("key_mask", mask.as_slice()),
            ])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("vision tower returned nothing"))?;
        let unshuffled = pixel_unshuffle(&feats[..n * hidden], tile.ph, tile.pw, hidden, ds);
        let rows = n / (ds * ds);
        let width = hidden * ds * ds;
        let mut padded = vec![0f32; self.proj_rows * width];
        padded[..rows * width].copy_from_slice(&unshuffled);
        let proj = self
            .projector
            .run(&[("image_features", padded.as_slice())])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("projector returned nothing"))?;
        Ok(proj[..rows * self.cfg.text_hidden].to_vec())
    }
}

/// `linear_2(gelu(linear_1(x)))` over `[1, rows, hidden · ds²]`.
pub(crate) fn build_projector(
    cfg: &VisionConfig,
    ds: usize,
    rows: usize,
    weight_dtype: DType,
    weights: &mut WeightMap,
) -> Result<rlx_flow::BuiltModel> {
    use rlx_flow::{CompileProfile, ModelFlow};
    use rlx_ir::hir::HirMut;
    use rlx_ir::{HirGraphExt, Shape};
    let width = cfg.hidden_size * ds * ds;
    let out = Shape::new(&[1, rows, cfg.text_hidden], DType::F32);
    for (name, want) in [
        (
            format!("{PROJECTOR}.linear_1.weight"),
            vec![cfg.projector_hidden, width],
        ),
        (
            format!("{PROJECTOR}.linear_2.weight"),
            vec![cfg.text_hidden, cfg.projector_hidden],
        ),
    ] {
        let got = weights
            .get(&name)
            .map(|(_, s)| s.to_vec())
            .ok_or_else(|| anyhow!("missing {name}"))?;
        if got != want {
            bail!("{name} is {got:?}, expected {want:?}");
        }
    }
    ModelFlow::new("lfm2_vl_projector")
        .with_profile(CompileProfile::encoder())
        .input("image_features", Shape::new(&[1, rows, width], DType::F32))
        .plugin_named("projector", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("projector needs input"))?
                .hir_id();
            let w1 =
                emit.load_param_typed(&format!("{PROJECTOR}.linear_1.weight"), true, weight_dtype)?;
            let b1 = emit.load_param(&format!("{PROJECTOR}.linear_1.bias"), false)?;
            let w2 =
                emit.load_param_typed(&format!("{PROJECTOR}.linear_2.weight"), true, weight_dtype)?;
            let b2 = emit.load_param(&format!("{PROJECTOR}.linear_2.bias"), false)?;
            let mut gb = HirMut::new(emit.hir());
            let m1 = gb.mm(x, w1);
            let a1 = gb.add(m1, b1);
            let g = gb.gelu(a1);
            let m2 = gb.mm(g, w2);
            let y = gb.add(m2, b2);
            Ok(Some(emit.wrap(y, out.clone())))
        })
        .output("image_embeds")
        .build_with(&mut rlx_core::flow_util::WeightMapSource(weights), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshuffle_matches_reference_layout() {
        // x[h, w, c] = 100h + 10w + c over a 4x4x2 map.
        let (ph, pw, c) = (4, 4, 2);
        let x: Vec<f32> = (0..ph * pw * c)
            .map(|i| {
                let (h, w, k) = (i / (pw * c), (i / c) % pw, i % c);
                (100 * h + 10 * w + k) as f32
            })
            .collect();
        let y = pixel_unshuffle(&x, ph, pw, c, 2);
        // Row (i=0, j=1): x[0,2], x[0,3], x[1,2], x[1,3].
        assert_eq!(&y[8..16], &[20., 21., 30., 31., 120., 121., 130., 131.]);
    }

    #[test]
    fn smart_resize_and_tiling_follow_the_processor() {
        let cfg = ProcessorConfig::default();
        // COCO 640x480: not too large; smart size 416x576 (13x18 tokens).
        let img = RgbImage::new(640, 480, vec![0; 640 * 480 * 3]).unwrap();
        let p = process_image(&img, &cfg).unwrap();
        assert_eq!((p.rows, p.cols, p.image_size), (1, 1, (416, 576)));
        assert_eq!((p.tiles[0].ph, p.tiles[0].pw), (26, 36));
        assert_eq!(p.placeholder(&cfg).matches(IMAGE_TOKEN).count(), 13 * 18);
        // 1248x810 (a capped 1400x900): too large -> grid + thumbnail.
        let img = RgbImage::new(1248, 802, vec![0; 1248 * 802 * 3]).unwrap();
        let p = process_image(&img, &cfg).unwrap();
        assert!(p.rows * p.cols > 1);
        assert_eq!(p.tiles.len(), p.rows * p.cols + 1);
    }
}
