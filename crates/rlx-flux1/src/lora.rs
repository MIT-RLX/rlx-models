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

//! LoRA adapters for the FLUX.1 denoiser, merged exactly into the base
//! weights as they stream in (`W += scale · B·A`).
//!
//! Three key dialects are accepted:
//! - **diffusers / PEFT** (`transformer.transformer_blocks.3.attn.to_q.lora_A.weight`),
//!   what SimpleTuner and `diffusers` write — the open-genmoji LoRAs use it;
//! - **BFL** module names (`double_blocks.3.img_attn.qkv.lora_A.weight`, optionally
//!   under `diffusion_model.`);
//! - **kohya** (`lora_unet_double_blocks_3_img_attn_qkv.lora_down.weight` + `.alpha`).
//!
//! diffusers splits projections BFL fuses (`to_q`/`to_k`/`to_v` → `img_attn.qkv`,
//! single-block `to_q`/`to_k`/`to_v`/`proj_mlp` → `linear1`), so each diffusers
//! adapter becomes a *row band* of the fused BFL matrix. `norm_out.linear` is
//! the one place the two layouts order their halves differently
//! (`scale, shift` vs `shift, scale`); its rows are swapped on the way in.

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::config::Flux1Config;

/// One low-rank update landing on rows `[row_offset, row_offset + out)` of a
/// BFL weight matrix.
#[derive(Debug, Clone)]
pub struct LoraPatch {
    pub row_offset: usize,
    pub out_dim: usize,
    pub in_dim: usize,
    pub rank: usize,
    /// `[rank, in_dim]`.
    pub a: Vec<f32>,
    /// `[out_dim, rank]`.
    pub b: Vec<f32>,
    /// `user_scale · alpha / rank` (alpha defaults to rank → 1).
    pub scale: f32,
}

/// Every patch, keyed by the BFL weight name (`….weight`) it modifies.
#[derive(Debug, Clone, Default)]
pub struct LoraSet {
    patches: HashMap<String, Vec<LoraPatch>>,
}

/// Where a LoRA module lands in a BFL checkpoint.
#[derive(Debug, Clone, PartialEq)]
struct Target {
    module: String,
    row_offset: usize,
}

impl LoraSet {
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    /// Number of BFL tensors touched.
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// BFL weight names this set modifies.
    pub fn targets(&self) -> impl Iterator<Item = &str> {
        self.patches.keys().map(String::as_str)
    }

    /// Load one adapter file at `scale` and add its patches to the set.
    /// `base_modules` are the BFL linear module names (weight key without
    /// `.weight`), used to resolve kohya's underscore-joined names.
    pub fn add_file(
        &mut self,
        path: &Path,
        scale: f32,
        cfg: &Flux1Config,
        base_modules: &[String],
    ) -> Result<usize> {
        let ckpt = SafetensorsCheckpoint::open_file(path)
            .with_context(|| format!("opening LoRA {}", path.display()))?;
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        for k in ckpt.keys() {
            tensors.insert(k.to_string(), ckpt.load_tensor_f32(k)?);
        }
        self.add_tensors(tensors, scale, cfg, base_modules)
            .with_context(|| format!("LoRA {}", path.display()))
    }

    /// [`Self::add_file`] over in-memory tensors (`name → (data, shape)`).
    pub fn add_tensors(
        &mut self,
        mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
        scale: f32,
        cfg: &Flux1Config,
        base_modules: &[String],
    ) -> Result<usize> {
        let kohya: HashMap<String, &String> = base_modules
            .iter()
            .map(|m| (m.replace('.', "_"), m))
            .collect();
        let mut pairs: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
        let mut skipped_te = 0usize;
        for key in tensors.keys() {
            let Some((module, is_a)) = split_lora_key(key) else {
                continue;
            };
            let e = pairs.entry(module.to_string()).or_default();
            if is_a {
                e.0 = Some(key.clone());
            } else {
                e.1 = Some(key.clone());
            }
        }
        let mut added = 0usize;
        for (module, (a_key, b_key)) in pairs {
            let (Some(a_key), Some(b_key)) = (a_key, b_key) else {
                bail!("LoRA module `{module}` has only one of its A/B halves");
            };
            if module.starts_with("lora_te") || module.contains("text_encoder") {
                skipped_te += 1;
                continue;
            }
            let targets = resolve_module(&module, cfg, &kohya)
                .with_context(|| format!("unrecognized LoRA module `{module}`"))?;
            let (a, a_shape) = tensors.remove(&a_key).expect("listed key");
            let (b, b_shape) = tensors.remove(&b_key).expect("listed key");
            ensure!(
                a_shape.len() == 2 && b_shape.len() == 2 && a_shape[0] == b_shape[1],
                "LoRA `{module}`: A {a_shape:?} and B {b_shape:?} do not compose"
            );
            let (rank, in_dim, out_dim) = (a_shape[0], a_shape[1], b_shape[0]);
            let alpha = tensors
                .get(&format!("{module}.alpha"))
                .and_then(|(v, _)| v.first().copied());
            let module_scale = scale * alpha.map_or(1.0, |al| al / rank as f32);
            if targets.len() == 1 {
                let t = &targets[0];
                self.push(t, out_dim, in_dim, rank, a, b, module_scale);
            } else {
                // Row-split adapter (norm_out swap): each target takes an equal band of B.
                let band = out_dim / targets.len();
                ensure!(
                    band * targets.len() == out_dim,
                    "LoRA `{module}`: B rows {out_dim} do not split"
                );
                for (n, t) in targets.iter().enumerate() {
                    let rows = b[n * band * rank..(n + 1) * band * rank].to_vec();
                    self.push(t, band, in_dim, rank, a.clone(), rows, module_scale);
                }
            }
            added += 1;
        }
        if skipped_te > 0 {
            eprintln!(
                "[rlx-flux1] LoRA: skipped {skipped_te} text-encoder modules (not supported)"
            );
        }
        Ok(added)
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        t: &Target,
        out_dim: usize,
        in_dim: usize,
        rank: usize,
        a: Vec<f32>,
        b: Vec<f32>,
        scale: f32,
    ) {
        self.patches
            .entry(format!("{}.weight", t.module))
            .or_default()
            .push(LoraPatch {
                row_offset: t.row_offset,
                out_dim,
                in_dim,
                rank,
                a,
                b,
                scale,
            });
    }

    /// Every patch on `key` as one low-rank pair, for applying the adapter as
    /// a graph branch `y += (x Aᵀ) Bᵀ` instead of merging it: `A` stacks the
    /// patches' `A`s (`[R, in]`, `R = Σ rank`), `B` is block-sparse
    /// `[rows, R]` with each patch's `scale · B` on its row band.
    pub fn stacked(&self, key: &str, rows: usize) -> Option<(Vec<f32>, Vec<f32>, usize)> {
        let patches = self.patches.get(key)?;
        let total: usize = patches.iter().map(|p| p.rank).sum();
        let in_dim = patches[0].in_dim;
        let mut a = Vec::with_capacity(total * in_dim);
        let mut b = vec![0.0f32; rows * total];
        let mut r0 = 0;
        for p in patches {
            a.extend_from_slice(&p.a);
            for o in 0..p.out_dim {
                for r in 0..p.rank {
                    b[(p.row_offset + o) * total + r0 + r] = p.scale * p.b[o * p.rank + r];
                }
            }
            r0 += p.rank;
        }
        Some((a, b, total))
    }

    /// Add every patch for `key` into `w` (`shape = [rows, cols]`, row-major).
    pub fn apply(&self, key: &str, w: &mut [f32], shape: &[usize]) -> Result<bool> {
        let Some(patches) = self.patches.get(key) else {
            return Ok(false);
        };
        ensure!(
            shape.len() == 2,
            "LoRA target `{key}` is not a matrix: {shape:?}"
        );
        let (rows, cols) = (shape[0], shape[1]);
        for p in patches {
            ensure!(
                p.in_dim == cols && p.row_offset + p.out_dim <= rows,
                "LoRA patch [{}+{}, {}] does not fit `{key}` {shape:?}",
                p.row_offset,
                p.out_dim,
                p.in_dim
            );
            let band = &mut w[p.row_offset * cols..(p.row_offset + p.out_dim) * cols];
            band.par_chunks_mut(cols).enumerate().for_each(|(o, row)| {
                let b_row = &p.b[o * p.rank..(o + 1) * p.rank];
                for (r, &bv) in b_row.iter().enumerate() {
                    let s = p.scale * bv;
                    if s == 0.0 {
                        continue;
                    }
                    let a_row = &p.a[r * cols..(r + 1) * cols];
                    for (wv, &av) in row.iter_mut().zip(a_row) {
                        *wv += s * av;
                    }
                }
            });
        }
        Ok(true)
    }
}

/// `(module, is_A)` for a LoRA weight key, or `None` for anything else
/// (alphas, metadata tensors).
fn split_lora_key(key: &str) -> Option<(&str, bool)> {
    for (suffix, is_a) in [
        (".lora_A.weight", true),
        (".lora_B.weight", false),
        (".lora_down.weight", true),
        (".lora_up.weight", false),
        (".lora_A.default.weight", true),
        (".lora_B.default.weight", false),
    ] {
        if let Some(m) = key.strip_suffix(suffix) {
            return Some((m, is_a));
        }
    }
    None
}

fn strip_prefixes(mut m: &str) -> &str {
    for p in [
        "base_model.model.",
        "transformer.",
        "diffusion_model.",
        "model.diffusion_model.",
        "unet.",
    ] {
        if let Some(rest) = m.strip_prefix(p) {
            m = rest;
        }
    }
    m
}

fn resolve_module(
    module: &str,
    cfg: &Flux1Config,
    kohya: &HashMap<String, &String>,
) -> Option<Vec<Target>> {
    let t = |module: String, row_offset: usize| Target { module, row_offset };
    if let Some(rest) = module.strip_prefix("lora_unet_") {
        return kohya.get(rest).map(|m| vec![t((*m).clone(), 0)]);
    }
    let m = strip_prefixes(module);
    if m.starts_with("double_blocks.")
        || m.starts_with("single_blocks.")
        || m.starts_with("final_layer.")
        || ["img_in", "txt_in"].contains(&m)
        || m.starts_with("time_in.")
        || m.starts_with("vector_in.")
        || m.starts_with("guidance_in.")
    {
        return Some(vec![t(m.to_string(), 0)]);
    }
    let d = cfg.hidden;
    let (head, rest) = m.split_once('.').unwrap_or((m, ""));
    let block = |rest: &str| -> Option<(usize, String)> {
        let (i, tail) = rest.split_once('.')?;
        Some((i.parse().ok()?, tail.to_string()))
    };
    match head {
        "transformer_blocks" => {
            let (i, tail) = block(rest)?;
            let p = format!("double_blocks.{i}");
            let (name, off) = match tail.as_str() {
                "attn.to_q" => ("img_attn.qkv", 0),
                "attn.to_k" => ("img_attn.qkv", d),
                "attn.to_v" => ("img_attn.qkv", 2 * d),
                "attn.add_q_proj" => ("txt_attn.qkv", 0),
                "attn.add_k_proj" => ("txt_attn.qkv", d),
                "attn.add_v_proj" => ("txt_attn.qkv", 2 * d),
                "attn.to_out.0" => ("img_attn.proj", 0),
                "attn.to_add_out" => ("txt_attn.proj", 0),
                "ff.net.0.proj" => ("img_mlp.0", 0),
                "ff.net.2" => ("img_mlp.2", 0),
                "ff_context.net.0.proj" => ("txt_mlp.0", 0),
                "ff_context.net.2" => ("txt_mlp.2", 0),
                "norm1.linear" => ("img_mod.lin", 0),
                "norm1_context.linear" => ("txt_mod.lin", 0),
                _ => return None,
            };
            Some(vec![t(format!("{p}.{name}"), off)])
        }
        "single_transformer_blocks" => {
            let (i, tail) = block(rest)?;
            let p = format!("single_blocks.{i}");
            let (name, off) = match tail.as_str() {
                "attn.to_q" => ("linear1", 0),
                "attn.to_k" => ("linear1", d),
                "attn.to_v" => ("linear1", 2 * d),
                "proj_mlp" => ("linear1", 3 * d),
                "proj_out" => ("linear2", 0),
                "norm.linear" => ("modulation.lin", 0),
                _ => return None,
            };
            Some(vec![t(format!("{p}.{name}"), off)])
        }
        "x_embedder" => Some(vec![t("img_in".into(), 0)]),
        "context_embedder" => Some(vec![t("txt_in".into(), 0)]),
        "proj_out" => Some(vec![t("final_layer.linear".into(), 0)]),
        // diffusers' AdaLayerNormContinuous emits (scale, shift); BFL (shift, scale).
        "norm_out" if rest == "linear" => Some(vec![
            t("final_layer.adaLN_modulation.1".into(), d),
            t("final_layer.adaLN_modulation.1".into(), 0),
        ]),
        "time_text_embed" => {
            let (emb, lin) = rest.split_once('.')?;
            let bfl = match emb {
                "timestep_embedder" => "time_in",
                "guidance_embedder" => "guidance_in",
                "text_embedder" => "vector_in",
                _ => return None,
            };
            let layer = match lin {
                "linear_1" => "in_layer",
                "linear_2" => "out_layer",
                _ => return None,
            };
            Some(vec![t(format!("{bfl}.{layer}"), 0)])
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modules(cfg: &Flux1Config) -> Vec<String> {
        cfg.expected_tensors()
            .into_iter()
            .filter_map(|k| k.strip_suffix(".weight").map(str::to_string))
            .collect()
    }

    #[test]
    fn diffusers_keys_land_on_fused_bands() {
        let cfg = Flux1Config::dev();
        let k = HashMap::new();
        let r = |m: &str| resolve_module(m, &cfg, &k).unwrap();
        assert_eq!(
            r("transformer.transformer_blocks.4.attn.add_k_proj"),
            vec![Target {
                module: "double_blocks.4.txt_attn.qkv".into(),
                row_offset: 3072
            }]
        );
        assert_eq!(
            r("transformer.single_transformer_blocks.37.proj_mlp"),
            vec![Target {
                module: "single_blocks.37.linear1".into(),
                row_offset: 9216
            }]
        );
        assert_eq!(r("transformer.proj_out")[0].module, "final_layer.linear");
        assert_eq!(r("transformer.norm_out.linear")[0].row_offset, 3072);
    }

    #[test]
    fn kohya_names_resolve_through_the_checkpoint() {
        let cfg = Flux1Config::dev();
        let mods = modules(&cfg);
        let kohya: HashMap<String, &String> =
            mods.iter().map(|m| (m.replace('.', "_"), m)).collect();
        let t = resolve_module("lora_unet_double_blocks_3_img_attn_qkv", &cfg, &kohya).unwrap();
        assert_eq!(t[0].module, "double_blocks.3.img_attn.qkv");
    }

    #[test]
    fn apply_matches_dense_update() {
        let cfg = Flux1Config::tiny();
        let d = cfg.hidden;
        let rank = 3;
        let a: Vec<f32> = (0..rank * d)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let b: Vec<f32> = (0..d * rank)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1)
            .collect();
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "transformer.transformer_blocks.1.attn.to_k.lora_A.weight".to_string(),
            (a.clone(), vec![rank, d]),
        );
        tensors.insert(
            "transformer.transformer_blocks.1.attn.to_k.lora_B.weight".to_string(),
            (b.clone(), vec![d, rank]),
        );
        let mut set = LoraSet::default();
        set.add_tensors(tensors, 0.5, &cfg, &modules(&cfg)).unwrap();
        let mut w = vec![0.0f32; 3 * d * d];
        assert!(
            set.apply("double_blocks.1.img_attn.qkv.weight", &mut w, &[3 * d, d])
                .unwrap()
        );
        for o in 0..3 * d {
            for i in 0..d {
                let want = if (d..2 * d).contains(&o) {
                    0.5 * (0..rank)
                        .map(|r| b[(o - d) * rank + r] * a[r * d + i])
                        .sum::<f32>()
                } else {
                    0.0
                };
                assert!((w[o * d + i] - want).abs() < 1e-5, "row {o} col {i}");
            }
        }
    }
}
