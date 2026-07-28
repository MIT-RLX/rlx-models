// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Pixtral vision encoder — host preprocess + compiled ViT/projector graph.

use crate::config::PixtralVisionConfig;
use crate::flow::build_pixtral_vision;
use crate::preprocess::{apply_patch_embed, image_to_patch_rows};
use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

pub struct PixtralWeights {
    pub patch_embd: Vec<f32>,
    pub pre_ln: Vec<f32>,
    pub layers: Vec<PixtralLayerWeights>,
    pub mm_input_norm: Option<Vec<f32>>,
    pub mm_patch_merger: Option<Vec<f32>>,
    pub mm1_w: Vec<f32>,
    pub mm1_b: Option<Vec<f32>>,
    pub mm2_w: Vec<f32>,
    pub mm2_b: Option<Vec<f32>>,
    pub img_break: Option<Vec<f32>>,
}

pub struct PixtralLayerWeights {
    pub ln1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub o: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

impl PixtralWeights {
    pub fn from_mmproj(path: impl AsRef<Path>, cfg: &PixtralVisionConfig) -> Result<Self> {
        let path = path.as_ref();
        // llama.cpp mmproj GGUFs tag `general.architecture` as `clip` (some
        // Pixtral conversions use `pixtral`); the real projector identity is
        // already checked via `clip.projector_type` in `PixtralVisionConfig`.
        let mut wm = rlx_core::load_weight_map(path, &["clip", "pixtral"])
            .with_context(|| format!("rlx-mistral-vl: load {path:?}"))?;
        Self::from_weight_map(&mut wm, cfg)
    }

    pub fn from_weight_map(wm: &mut WeightMap, cfg: &PixtralVisionConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let ff = cfg.intermediate_size;
        let take = |wm: &mut WeightMap, key: &str| -> Result<Vec<f32>> {
            let (data, _) = wm
                .take(key)
                .with_context(|| format!("missing mmproj tensor `{key}`"))?;
            Ok(data)
        };
        let try_take = |wm: &mut WeightMap, key: &str| -> Option<Vec<f32>> {
            wm.take(key).ok().map(|(d, _)| d)
        };

        let patch_embd = take(wm, "v.patch_embd.weight")?;
        let pre_ln = take(wm, "v.pre_ln.weight")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(PixtralLayerWeights {
                ln1: take(wm, &format!("v.blk.{i}.ln1.weight"))?,
                ln2: take(wm, &format!("v.blk.{i}.ln2.weight"))?,
                q: take(wm, &format!("v.blk.{i}.attn_q.weight"))?,
                k: take(wm, &format!("v.blk.{i}.attn_k.weight"))?,
                v: take(wm, &format!("v.blk.{i}.attn_v.weight"))?,
                o: take(wm, &format!("v.blk.{i}.attn_out.weight"))?,
                gate: take(wm, &format!("v.blk.{i}.ffn_gate.weight"))?,
                up: take(wm, &format!("v.blk.{i}.ffn_up.weight"))?,
                down: take(wm, &format!("v.blk.{i}.ffn_down.weight"))?,
            });
            ensure!(layers[i].q.len() == h * h, "attn_q size layer {i}");
            ensure!(layers[i].gate.len() == h * ff, "ffn_gate size layer {i}");
            ensure!(layers[i].down.len() == ff * h, "ffn_down size layer {i}");
        }

        let mm_input_norm = try_take(wm, "mm.input_norm.weight");
        let mm_patch_merger = try_take(wm, "mm.patch_merger.weight");
        let mm1_w = take(wm, "mm.1.weight")?;
        let mm1_b = try_take(wm, "mm.1.bias");
        let mm2_w = take(wm, "mm.2.weight")?;
        let mm2_b = try_take(wm, "mm.2.bias");
        let img_break = try_take(wm, "v.token_embd.img_break");

        Ok(Self {
            patch_embd,
            pre_ln,
            layers,
            mm_input_norm,
            mm_patch_merger,
            mm1_w,
            mm1_b,
            mm2_w,
            mm2_b,
            img_break,
        })
    }
}

/// Compiled Pixtral vision tower (Metal / CUDA / CPU via [`Session`]).
pub struct PixtralVisionEncoder {
    cfg: PixtralVisionConfig,
    weights: PixtralWeights,
    path: PathBuf,
    device: Device,
    graph_key: Option<(usize, usize)>,
    compiled: Option<rlx_runtime::CompiledGraph>,
}

impl PixtralVisionEncoder {
    pub fn from_mmproj(path: impl Into<PathBuf>) -> Result<Self> {
        Self::from_mmproj_on_device(path, Device::Cpu)
    }

    pub fn from_mmproj_on_device(path: impl Into<PathBuf>, device: Device) -> Result<Self> {
        let path = path.into();
        let cfg = PixtralVisionConfig::from_mmproj_gguf(&path)?;
        let weights = PixtralWeights::from_mmproj(&path, &cfg)?;
        Ok(Self {
            cfg,
            weights,
            path,
            device,
            graph_key: None,
            compiled: None,
        })
    }

    pub fn with_device(mut self, device: Device) -> Self {
        if self.device != device {
            self.device = device;
            self.graph_key = None;
            self.compiled = None;
        }
        self
    }

    pub fn config(&self) -> &PixtralVisionConfig {
        &self.cfg
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Encode RGB → LM soft tokens `[n_tokens * projector_output_dim]`.
    ///
    /// Host: resize / patch-embed / pre-RMS / 2D RoPE feeds / `img_break`.
    /// Device: ViT + patch merger + MLP projector.
    ///
    /// **Memory:** the ViT runs full (unmasked) self-attention over every patch,
    /// so device memory scales O(n_patches²). `n_patches` is bounded by
    /// `image_size` — `image_to_patch_rows` resizes the long edge down to it —
    /// but a large `image_size` (or feeding a near-square image at the default
    /// 1540) still costs several GB on a materialized-attention backend. Errors
    /// (rather than OOMs) if the grid exceeds a defensive cap.
    pub fn encode_rgb(&mut self, rgb: &[u8], img_w: usize, img_h: usize) -> Result<Vec<f32>> {
        let patch_dim = self.cfg.num_channels * self.cfg.patch_size * self.cfg.patch_size;
        let hidden_size = self.cfg.hidden_size;
        let eps = self.cfg.layer_norm_eps;
        let head_dim = self.cfg.head_dim();
        let theta = self.cfg.rope_theta;
        let n_merge = self.cfg.spatial_merge_size.max(1);
        let proj_dim = self.cfg.projector_output_dim;

        let (patches, grid_x, grid_y) = image_to_patch_rows(rgb, img_w, img_h, &self.cfg)?;
        let n_patches = grid_x * grid_y;
        // Guard a pathological / misconfigured `image_size` from producing a
        // runaway O(n_patches²) attention alloc — a clear error beats an OOM.
        // The cap sits well above Pixtral's ~12.1k max (image_size 1540/patch 14).
        const MAX_PATCHES: usize = 1 << 15; // 32768
        ensure!(
            n_patches <= MAX_PATCHES,
            "pixtral vision: {n_patches} patches ({grid_x}×{grid_y}) exceeds the \
             {MAX_PATCHES} cap — image_size={} is too large (ViT attention is \
             O(n_patches²)); use a smaller image or clip.vision.image_size",
            self.cfg.image_size
        );
        let mut h = apply_patch_embed(
            &patches,
            n_patches,
            patch_dim,
            hidden_size,
            &self.weights.patch_embd,
        )?;
        h = rms_norm_rows(&h, n_patches, hidden_size, &self.weights.pre_ln, eps);

        self.ensure_compiled(grid_x, grid_y)?;
        let (rope_cos, rope_sin) = pixtral_vision_rope_feeds(grid_x, grid_y, head_dim, theta);

        let outs = self
            .compiled
            .as_mut()
            .context("pixtral vision graph not compiled")?
            .run(&[
                ("hidden", h.as_slice()),
                ("vision_rope_cos", &rope_cos),
                ("vision_rope_sin", &rope_sin),
            ]);
        let mut out = outs
            .into_iter()
            .next()
            .context("pixtral vision graph produced no outputs")?;

        let p_x = grid_x / n_merge;
        let p_y = grid_y / n_merge;
        let n_merged = p_x * p_y;
        ensure!(
            out.len() == n_merged * proj_dim,
            "projector out len {} != {}×{}",
            out.len(),
            n_merged,
            proj_dim
        );

        if let Some(brk) = &self.weights.img_break {
            ensure!(brk.len() == proj_dim, "img_break dim {}", brk.len());
            out = insert_img_break(&out, p_x, p_y, proj_dim, brk);
        }
        Ok(out)
    }

    fn ensure_compiled(&mut self, grid_x: usize, grid_y: usize) -> Result<()> {
        if self.graph_key == Some((grid_x, grid_y)) && self.compiled.is_some() {
            return Ok(());
        }
        let built = build_pixtral_vision(&self.cfg, &self.weights, grid_x, grid_y)?;
        self.compiled = Some(compile_built(built.model, self.device)?);
        self.graph_key = Some((grid_x, grid_y));
        Ok(())
    }
}

/// GPT-J pair cos/sin for Pixtral 2D RoPE. Numerically equivalent to
/// llama.cpp `tools/mtmd/clip.cpp::build_rope_2d` (invoked by the pixtral graph
/// with `pos_h=row`, `pos_w=col`, `interleave_freq=true`): the head splits into
/// a first half rotated by height positions and a second half by width, each an
/// `ggml_rope_ext` mode-0 (adjacent-pair / GPT-J) rotation. Since GPT-J pairs
/// `(2k,2k+1)` with `cos/sin[k]`, one full-head `RopeStyle::GptJ` over the
/// concatenated `[height | width]` pair-feeds reproduces both half-rotations.
///
/// Verified against the reference in
/// `tests`::`rope_feeds_match_llama_cpp_build_rope_2d`.
///
/// Shape `[n_patches, head_dim/2]`: first `head_dim/4` pairs use height
/// positions, the next `head_dim/4` use width positions (scaled by
/// `theta^(-2/head_dim)`, matching ggml's `freq_scale_odd`).
pub fn pixtral_vision_rope_feeds(
    grid_x: usize,
    grid_y: usize,
    head_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = grid_x * grid_y;
    let half = head_dim / 2;
    let pairs = half; // head_dim/2 GPT-J pairs
    let pairs_h = half / 2;
    let freq_scale_w = theta.powf(-2.0 / head_dim as f32);
    let mut cos = vec![0f32; n * pairs];
    let mut sin = vec![0f32; n * pairs];
    for gy in 0..grid_y {
        for gx in 0..grid_x {
            let t = gy * grid_x + gx;
            let base = t * pairs;
            for i in 0..pairs_h {
                let freq = theta.powf(-2.0 * i as f32 / half as f32);
                let angle = gy as f32 * freq;
                cos[base + i] = angle.cos();
                sin[base + i] = angle.sin();
            }
            for i in 0..pairs_h {
                let freq = freq_scale_w * theta.powf(-2.0 * i as f32 / half as f32);
                let angle = gx as f32 * freq;
                let j = pairs_h + i;
                cos[base + j] = angle.cos();
                sin[base + j] = angle.sin();
            }
        }
    }
    (cos, sin)
}

fn insert_img_break(tokens: &[f32], p_x: usize, p_y: usize, dim: usize, brk: &[f32]) -> Vec<f32> {
    // one break per row except last → n_tokens = p_x*p_y + p_y - 1
    let n_out = p_x * p_y + p_y.saturating_sub(1);
    let mut out = vec![0f32; n_out * dim];
    let mut dst = 0usize;
    for y in 0..p_y {
        for x in 0..p_x {
            let src = (y * p_x + x) * dim;
            out[dst * dim..(dst + 1) * dim].copy_from_slice(&tokens[src..src + dim]);
            dst += 1;
        }
        if y + 1 < p_y {
            out[dst * dim..(dst + 1) * dim].copy_from_slice(brk);
            dst += 1;
        }
    }
    out
}

fn rms_norm_rows(x: &[f32], n: usize, h: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; n * h];
    for t in 0..n {
        let row = &x[t * h..(t + 1) * h];
        let mut ss = 0f32;
        for &v in row {
            ss += v * v;
        }
        let scale = (ss / h as f32 + eps).sqrt().recip();
        for i in 0..h {
            out[t * h + i] = row[i] * scale * w[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{PixtralLayerWeights, PixtralWeights, insert_img_break, pixtral_vision_rope_feeds};

    #[test]
    fn img_break_inserted_between_rows_only() {
        // 2×2 merged grid, dim=1, break token = 9 → one break after row 0, none
        // after the final row. Output length = p_x*p_y + (p_y - 1).
        let toks = [0.0, 1.0, 2.0, 3.0];
        let out = insert_img_break(&toks, 2, 2, 1, &[9.0]);
        assert_eq!(out, vec![0.0, 1.0, 9.0, 2.0, 3.0]);
    }

    #[test]
    fn rope_feeds_shape_and_origin() {
        let (cos, sin) = pixtral_vision_rope_feeds(2, 2, 4, 10000.0);
        let n = 2 * 2;
        let half = 4 / 2;
        assert_eq!(cos.len(), n * half);
        assert_eq!(sin.len(), n * half);
        // Patch (gy=0,gx=0) is at zero position on both axes → cos=1, sin=0.
        assert!((cos[0] - 1.0).abs() < 1e-6 && sin[0].abs() < 1e-6);
        assert!((cos[1] - 1.0).abs() < 1e-6 && sin[1].abs() < 1e-6);
        // Patch t=2 is (gy=1,gx=0): the height pair (i=0) has angle = 1 rad.
        let base = 2 * half; // pairs == head_dim/2 == half
        assert!((cos[base] - 1.0f32.cos()).abs() < 1e-6);
        assert!((sin[base] - 1.0f32.sin()).abs() < 1e-6);
    }

    /// Pins `pixtral_vision_rope_feeds` to llama.cpp's authoritative
    /// `clip.cpp::build_rope_2d` (called by the pixtral graph in
    /// `tools/mtmd/models/pixtral.cpp` as `build_rope_2d(cur, pos_h, pos_w,
    /// rope_theta, /*interleave*/true)`, with `pos_h[i]=i/grid_x` and
    /// `pos_w[i]=i%grid_x`). The reference below is reconstructed from ggml's
    /// mode-0 rope semantics independently of our production formula:
    ///   • head dim d split; first half `[0,d/2)` rotates with pos_h(=gy),
    ///     second half `[d/2,d)` with pos_w(=gx); both `ggml_rope_ext` mode 0,
    ///     n_dims=d/2, so pair m gets `θ_m = freq_scale · θ^(-2m/(d/2))`;
    ///   • `freq_scale = 1` (height) and `θ^(-2/d)` (width, `interleave_freq`).
    #[test]
    fn rope_feeds_match_llama_cpp_build_rope_2d() {
        let (grid_x, grid_y, d, theta) = (3usize, 2usize, 8usize, 10000f32);
        let half = d / 2; // pairs total (= n_dims/2 of the *full* head)
        let quarter = d / 4; // pairs per axis (= (d/2)/2)
        let freq_scale_odd = theta.powf(-2.0 / d as f32);

        let n = grid_x * grid_y;
        let mut ref_cos = vec![0f32; n * half];
        let mut ref_sin = vec![0f32; n * half];
        for gy in 0..grid_y {
            for gx in 0..grid_x {
                let t = gy * grid_x + gx; // pos_h=t/grid_x=gy, pos_w=t%grid_x=gx
                let base = t * half;
                for k in 0..half {
                    // First half → height; second half → width (with odd scale).
                    let (pos, freq) = if k < quarter {
                        (gy as f32, theta.powf(-2.0 * k as f32 / half as f32))
                    } else {
                        let m = k - quarter;
                        (
                            gx as f32,
                            freq_scale_odd * theta.powf(-2.0 * m as f32 / half as f32),
                        )
                    };
                    let angle = pos * freq;
                    ref_cos[base + k] = angle.cos();
                    ref_sin[base + k] = angle.sin();
                }
            }
        }

        let (cos, sin) = pixtral_vision_rope_feeds(grid_x, grid_y, d, theta);
        assert_eq!(cos.len(), ref_cos.len());
        for i in 0..cos.len() {
            assert!(
                (cos[i] - ref_cos[i]).abs() < 1e-6 && (sin[i] - ref_sin[i]).abs() < 1e-6,
                "rope feed[{i}] diverges from llama.cpp build_rope_2d: cos {} vs {}, sin {} vs {}",
                cos[i],
                ref_cos[i],
                sin[i],
                ref_sin[i]
            );
        }
    }

    /// End-to-end: build the Pixtral vision HIR with tiny synthetic weights,
    /// compile it on CPU, and run it. Exercises every reshape / matmul /
    /// attention / gather / patch-merger / projector dim in `flow.rs` without
    /// needing a real GGUF — catches shape mismatches that only surface at run.
    #[test]
    fn vision_graph_compiles_and_runs_on_cpu() {
        use crate::config::PixtralVisionConfig;
        use crate::flow::build_pixtral_vision;
        use rlx_core::flow_util::compile_built;
        use rlx_runtime::Device;

        let cfg = PixtralVisionConfig {
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2, // head_dim = 4
            intermediate_size: 16,
            patch_size: 2,
            spatial_merge_size: 2,
            projector_output_dim: 6, // deliberately != hidden_size
            num_channels: 3,
            ..PixtralVisionConfig::default()
        };

        let h = cfg.hidden_size;
        let ff = cfg.intermediate_size;
        let merge_sq = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let proj = cfg.projector_output_dim;
        let mk = |n: usize, v: f32| vec![v; n];
        let layer = || PixtralLayerWeights {
            ln1: mk(h, 1.0),
            ln2: mk(h, 1.0),
            q: mk(h * h, 0.05),
            k: mk(h * h, 0.05),
            v: mk(h * h, 0.05),
            o: mk(h * h, 0.05),
            gate: mk(h * ff, 0.05),
            up: mk(h * ff, 0.05),
            down: mk(ff * h, 0.05),
        };
        // patch_embd / pre_ln / img_break are host-side only — the graph never
        // reads them, so empty/None is fine here.
        let weights = PixtralWeights {
            patch_embd: vec![],
            pre_ln: vec![],
            layers: vec![layer(), layer()],
            mm_input_norm: Some(mk(h, 1.0)),
            mm_patch_merger: Some(mk(h * h * merge_sq, 0.02)),
            mm1_w: mk(proj * h, 0.05),
            mm1_b: None,
            mm2_w: mk(proj * proj, 0.05),
            mm2_b: None,
            img_break: None,
        };

        let (grid_x, grid_y) = (2, 2);
        let n_pos = grid_x * grid_y;
        let built = build_pixtral_vision(&cfg, &weights, grid_x, grid_y).unwrap();
        assert_eq!(built.n_merged, 1); // (2/2) * (2/2)
        let mut compiled = compile_built(built.model, Device::Cpu).unwrap();

        let hidden = vec![0.05f32; n_pos * h];
        let (cos, sin) = pixtral_vision_rope_feeds(grid_x, grid_y, cfg.head_dim(), cfg.rope_theta);
        let outs = compiled.run(&[
            ("hidden", hidden.as_slice()),
            ("vision_rope_cos", cos.as_slice()),
            ("vision_rope_sin", sin.as_slice()),
        ]);
        let out = outs.into_iter().next().expect("vision graph output");
        assert_eq!(out.len(), built.n_merged * proj);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite projector output"
        );
    }
}
