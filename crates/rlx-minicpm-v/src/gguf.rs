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

//! The official GGUF release: an LM GGUF (`general.architecture = qwen35`)
//! plus a vision `mmproj` GGUF (`clip.projector_type = minicpmv4_6 | 4_7`),
//! as llama.cpp and OpenBMB's llama.cpp-omni load them.
//!
//! The mmproj holds the same tensors as the HF checkpoint under llama.cpp
//! names; [`load_mmproj_weights`] renames them and pins the HF shapes (the
//! data is already row-major `[out, in]`). The repo ships no `config.json` or
//! `tokenizer.json`, so [`config_from_gguf`] reads the `clip.*` keys and finds
//! the marker ids in the LM's embedded vocabulary.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_gguf::{GgufFile, MetaValue};

use crate::config::{
    DownsampleMode, MarkerIds, MiniCpmVConfig, MropeMode, SliceConfig, VisionConfig,
};
use crate::vision::{MERGER, PATCH_LINEAR, TOWER};

/// An LM GGUF and its vision mmproj.
#[derive(Debug, Clone)]
pub struct GgufPair {
    pub lm: PathBuf,
    pub mmproj: PathBuf,
}

impl GgufPair {
    /// Pick the pair out of a downloaded repo directory. `prefer` selects the
    /// LM quantization by substring (`"Q4_K_M"`, `"F16"`); without it the
    /// directory must hold exactly one LM GGUF. The 16x mmproj
    /// (`mmproj-model-f16.gguf`) wins over the `-4x` variant.
    pub fn from_dir(dir: &Path, prefer: Option<&str>) -> Result<Self> {
        let mut lms = Vec::new();
        let mut projs = Vec::new();
        for e in std::fs::read_dir(dir).with_context(|| format!("minicpm-v: read {dir:?}"))? {
            let p = e?.path();
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            if !name.to_ascii_lowercase().ends_with(".gguf") {
                continue;
            }
            if name.starts_with("mmproj") {
                projs.push(p);
            } else if !name.to_ascii_lowercase().contains("dspark") {
                lms.push(p);
            }
        }
        projs.sort_by_key(|p| p.to_string_lossy().contains("-4x"));
        let mmproj = projs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("minicpm-v: no mmproj*.gguf in {dir:?}"))?;
        let lm = match prefer {
            Some(q) => lms
                .into_iter()
                .find(|p| p.to_string_lossy().contains(q))
                .ok_or_else(|| anyhow!("minicpm-v: no LM GGUF matching {q:?} in {dir:?}"))?,
            None if lms.len() == 1 => lms.remove(0),
            None => bail!(
                "minicpm-v: {} LM GGUFs in {dir:?}; pick one with --prefer (e.g. Q4_K_M, F16)",
                lms.len()
            ),
        };
        Ok(Self { lm, mmproj })
    }

    /// `true` when `dir` holds an mmproj GGUF and no `config.json`.
    pub fn is_gguf_dir(dir: &Path) -> bool {
        dir.is_dir()
            && !dir.join("config.json").is_file()
            && std::fs::read_dir(dir).is_ok_and(|rd| {
                rd.flatten().any(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    n.starts_with("mmproj") && n.ends_with(".gguf")
                })
            })
    }
}

fn meta<'a>(f: &'a GgufFile, k: &str) -> Option<&'a MetaValue> {
    f.metadata.get(k)
}

fn meta_u(f: &GgufFile, k: &str) -> Result<usize> {
    meta(f, k)
        .and_then(MetaValue::as_u32)
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("minicpm-v: mmproj metadata {k} missing"))
}

fn meta_f32s(f: &GgufFile, k: &str) -> Option<Vec<f32>> {
    meta(f, k)?
        .as_array()?
        .iter()
        .map(|v| match v {
            MetaValue::F32(x) => Some(*x),
            MetaValue::F64(x) => Some(*x as f32),
            _ => None,
        })
        .collect()
}

/// Token id of `piece` in the LM GGUF's `tokenizer.ggml.tokens`.
fn token_id(tokens: &[MetaValue], piece: &str) -> Result<u32> {
    tokens
        .iter()
        .position(|t| t.as_str() == Some(piece))
        .map(|i| i as u32)
        .ok_or_else(|| anyhow!("minicpm-v: token {piece:?} not in the GGUF vocabulary"))
}

/// Build the config from the mmproj's `clip.*` keys and the LM's vocabulary.
pub fn config_from_gguf(pair: &GgufPair) -> Result<MiniCpmVConfig> {
    let proj = GgufFile::from_path_mmap(&pair.mmproj)
        .with_context(|| format!("minicpm-v: open {:?}", pair.mmproj))?;
    let lm = GgufFile::from_path_mmap(&pair.lm)
        .with_context(|| format!("minicpm-v: open {:?}", pair.lm))?;

    let ptype = meta(&proj, "clip.projector_type")
        .and_then(MetaValue::as_str)
        .unwrap_or_default();
    let (model_type, mrope) = match ptype {
        "minicpmv4_7" => ("minicpmv4_7", MropeMode::Canvas),
        "minicpmv4_6" => ("minicpmv4_6", MropeMode::Linear),
        other => {
            bail!("minicpm-v: mmproj projector_type {other:?} (expected minicpmv4_6 / minicpmv4_7)")
        }
    };
    let pos = proj
        .tensors
        .get("v.position_embd.weight")
        .ok_or_else(|| anyhow!("minicpm-v: mmproj has no v.position_embd.weight"))?;
    let n_pos: usize = pos.shape.iter().copied().max().unwrap_or(0);
    let side = (n_pos as f64).sqrt().round() as usize;
    ensure!(
        side * side == n_pos,
        "minicpm-v: {n_pos} position rows is not a square grid"
    );

    // The window merger is the 16x mmproj; the `-4x` file ships without it.
    let x16 = proj.tensors.contains_key("v.vit_merger.ds_ffn_up.weight");
    let insert_layer_id = meta(&proj, "clip.vision.wa_layer_indexes")
        .and_then(MetaValue::as_array)
        .and_then(|a| a.first())
        .and_then(MetaValue::as_u32)
        .map(|v| v as usize)
        .unwrap_or(6);
    let vision = VisionConfig {
        hidden_size: meta_u(&proj, "clip.vision.embedding_length")?,
        intermediate_size: meta_u(&proj, "clip.vision.feed_forward_length")?,
        num_attention_heads: meta_u(&proj, "clip.vision.attention.head_count")?,
        num_hidden_layers: meta_u(&proj, "clip.vision.block_count")?,
        patch_size: meta_u(&proj, "clip.vision.patch_size")?,
        num_patches_per_side: side,
        layer_norm_eps: match meta(&proj, "clip.vision.attention.layer_norm_epsilon") {
            Some(MetaValue::F32(x)) => *x,
            _ => 1e-6,
        },
        insert_layer_id,
        window: 2,
        flat_positions: false,
    };
    let mut slice = SliceConfig::default();
    if let Ok(n) = meta_u(&proj, "clip.vision.max_slice_nums") {
        slice.max_slice_nums = n;
    }
    // `clip.vision.image_size` is the slicing resolution here (448), not the
    // position grid (980 in the HF config); that comes from the table above.
    if let Ok(n) = meta_u(&proj, "clip.vision.image_size") {
        slice.scale_resolution = n;
    }
    if let Some(m) = meta_f32s(&proj, "clip.vision.image_mean").filter(|v| v.len() == 3) {
        slice.image_mean = [m[0], m[1], m[2]];
    }
    if let Some(s) = meta_f32s(&proj, "clip.vision.image_std").filter(|v| v.len() == 3) {
        slice.image_std = [s[0], s[1], s[2]];
    }

    let tokens = meta(&lm, "tokenizer.ggml.tokens")
        .and_then(MetaValue::as_array)
        .ok_or_else(|| anyhow!("minicpm-v: LM GGUF has no tokenizer.ggml.tokens"))?;
    let image = token_id(tokens, "<|image_pad|>")?;
    let markers = MarkerIds {
        image,
        image_start: token_id(tokens, "<image>")?,
        image_end: token_id(tokens, "</image>")?,
        slice_start: token_id(tokens, "<slice>")?,
        slice_end: token_id(tokens, "</slice>")?,
        // Byte-level BPE spells "\n" as `Ċ`.
        newline: token_id(tokens, "\u{010A}")?,
    };
    Ok(MiniCpmVConfig {
        model_type: model_type.to_string(),
        vision,
        slice,
        downsample: if x16 {
            DownsampleMode::X16
        } else {
            DownsampleMode::X4
        },
        mrope,
        merger_times: 1,
        merge_kernel: 2,
        text_hidden_size: meta_u(&proj, "clip.vision.projection_dim")?,
        markers: Some(markers),
        image_token_id: image,
        video_token_id: token_id(tokens, "<|video_pad|>").ok(),
        projector: crate::config::Projector::Merger,
    })
}

/// End-of-turn ids: `<|im_end|>` and `<|endoftext|>` from the LM vocabulary
/// (HF's `generation_config.json` lists the same pair).
pub fn eos_from_gguf(lm: &Path) -> Result<Vec<u32>> {
    let f = GgufFile::from_path_mmap(lm).with_context(|| format!("minicpm-v: open {lm:?}"))?;
    let tokens = meta(&f, "tokenizer.ggml.tokens")
        .and_then(MetaValue::as_array)
        .ok_or_else(|| anyhow!("minicpm-v: LM GGUF has no tokenizer.ggml.tokens"))?;
    let mut out = Vec::new();
    for t in ["<|endoftext|>", "<|im_end|>"] {
        if let Ok(id) = token_id(tokens, t) {
            out.push(id);
        }
    }
    ensure!(
        !out.is_empty(),
        "minicpm-v: no end-of-turn token in the GGUF vocabulary"
    );
    Ok(out)
}

/// The mmproj tensors under the HF names and shapes [`crate::vision`] reads.
pub fn load_mmproj_weights(mmproj: &Path, cfg: &MiniCpmVConfig) -> Result<WeightMap> {
    let f =
        GgufFile::from_path_mmap(mmproj).with_context(|| format!("minicpm-v: open {mmproj:?}"))?;
    let v = &cfg.vision;
    let (c, i) = (v.hidden_size, v.intermediate_size);
    let w = v.window * v.window;
    let m = cfg.merge_kernel * cfg.merge_kernel;
    let p = v.patch_size;
    let side2 = v.num_patches_per_side * v.num_patches_per_side;
    let mut wm = WeightMap::from_tensors(Default::default());
    let mut put = |gguf: &str, hf: String, shape: Vec<usize>| -> Result<()> {
        let (data, _) = f
            .dequant_f32(gguf)
            .with_context(|| format!("minicpm-v: mmproj tensor {gguf}"))?;
        ensure!(
            data.len() == shape.iter().product::<usize>(),
            "minicpm-v: {gguf} has {} values, {hf} needs {shape:?}",
            data.len()
        );
        wm.insert(hf, data, shape);
        Ok(())
    };
    let linear = |put: &mut dyn FnMut(&str, String, Vec<usize>) -> Result<()>,
                  g: &str,
                  h: &str,
                  out: usize,
                  inp: usize|
     -> Result<()> {
        put(
            &format!("{g}.weight"),
            format!("{h}.weight"),
            vec![out, inp],
        )?;
        put(&format!("{g}.bias"), format!("{h}.bias"), vec![out])
    };
    let norm = |put: &mut dyn FnMut(&str, String, Vec<usize>) -> Result<()>,
                g: &str,
                h: &str,
                n: usize|
     -> Result<()> {
        put(&format!("{g}.weight"), format!("{h}.weight"), vec![n])?;
        put(&format!("{g}.bias"), format!("{h}.bias"), vec![n])
    };

    put(
        "v.patch_embd.weight",
        PATCH_LINEAR.to_string(),
        vec![c, 3 * p * p],
    )?;
    put(
        "v.patch_embd.bias",
        format!("{TOWER}.embeddings.patch_embedding.bias"),
        vec![c],
    )?;
    put(
        "v.position_embd.weight",
        format!("{TOWER}.embeddings.position_embedding.weight"),
        vec![side2, c],
    )?;
    for l in 0..v.num_hidden_layers {
        let (g, h) = (format!("v.blk.{l}"), format!("{TOWER}.encoder.layers.{l}"));
        norm(
            &mut put,
            &format!("{g}.ln1"),
            &format!("{h}.layer_norm1"),
            c,
        )?;
        norm(
            &mut put,
            &format!("{g}.ln2"),
            &format!("{h}.layer_norm2"),
            c,
        )?;
        for (gn, hn) in [
            ("attn_q", "q_proj"),
            ("attn_k", "k_proj"),
            ("attn_v", "v_proj"),
            ("attn_out", "out_proj"),
        ] {
            linear(
                &mut put,
                &format!("{g}.{gn}"),
                &format!("{h}.self_attn.{hn}"),
                c,
                c,
            )?;
        }
        linear(
            &mut put,
            &format!("{g}.ffn_up"),
            &format!("{h}.mlp.fc1"),
            i,
            c,
        )?;
        linear(
            &mut put,
            &format!("{g}.ffn_down"),
            &format!("{h}.mlp.fc2"),
            c,
            i,
        )?;
    }
    norm(&mut put, "v.post_ln", &format!("{TOWER}.post_layernorm"), c)?;
    if cfg.downsample.uses_window_merger() {
        let h = format!("{TOWER}.vit_merger");
        norm(&mut put, "v.vit_merger.ln1", &format!("{h}.layer_norm1"), c)?;
        for (gn, hn) in [
            ("attn_q", "q_proj"),
            ("attn_k", "k_proj"),
            ("attn_v", "v_proj"),
            ("attn_out", "out_proj"),
        ] {
            linear(
                &mut put,
                &format!("v.vit_merger.{gn}"),
                &format!("{h}.self_attn.{hn}"),
                c,
                c,
            )?;
        }
        norm(
            &mut put,
            "v.vit_merger.ds_ln",
            &format!("{h}.pre_norm"),
            w * c,
        )?;
        linear(
            &mut put,
            "v.vit_merger.ds_ffn_up",
            &format!("{h}.linear_1"),
            w * i,
            w * c,
        )?;
        linear(
            &mut put,
            "v.vit_merger.ds_ffn_down",
            &format!("{h}.linear_2"),
            c,
            w * i,
        )?;
    }
    norm(
        &mut put,
        "mm.input_norm",
        &format!("{MERGER}.pre_norm"),
        m * c,
    )?;
    linear(
        &mut put,
        "mm.up",
        &format!("{MERGER}.linear_1"),
        m * c,
        m * c,
    )?;
    linear(
        &mut put,
        "mm.down",
        &format!("{MERGER}.linear_2"),
        cfg.text_hidden_size,
        m * c,
    )?;
    Ok(wm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_16x_mmproj_and_preferred_lm() {
        let d = std::env::temp_dir().join(format!("minicpmv-gguf-pick-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for n in [
            "MiniCPM-V-4_7-1B-F16.gguf",
            "MiniCPM-V-4_7-1B-Q4_K_M.gguf",
            "dspark-MiniCPM-V-4_7-1B-F16.gguf",
            "mmproj-model-f16-4x.gguf",
            "mmproj-model-f16.gguf",
        ] {
            std::fs::write(d.join(n), b"").unwrap();
        }
        assert!(GgufPair::is_gguf_dir(&d));
        let p = GgufPair::from_dir(&d, Some("Q4_K_M")).unwrap();
        assert!(p.lm.ends_with("MiniCPM-V-4_7-1B-Q4_K_M.gguf"));
        assert!(p.mmproj.ends_with("mmproj-model-f16.gguf"));
        assert!(
            GgufPair::from_dir(&d, None).is_err(),
            "two LMs need --prefer"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
