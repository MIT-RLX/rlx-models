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

//! Named parameter tensors for compiled training graphs.

use anyhow::{Context, Result, bail, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_voxtral_tts::VoxtralTtsWeightStore;
use rlx_voxtral_tts::config::CodecArgs;
use rlx_voxtral_tts::load::PREFIX_CODEC;
use std::collections::HashMap;
use std::path::Path;

use crate::codec_graph::ParamSlot;

#[derive(Debug, Clone, Default)]
pub struct WeightStore(pub HashMap<String, Vec<f32>>);

impl WeightStore {
    pub fn apply(&self, exec: &mut rlx_runtime::CompiledGraph) {
        for (name, data) in &self.0 {
            exec.set_param(name, data);
        }
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.0.get(name).map(|v| v.as_slice())
    }

    pub fn subset(&self, prefix: &str) -> Self {
        Self(
            self.0
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }

    pub fn merge(&mut self, other: &Self) {
        for (k, v) in &other.0 {
            self.0.insert(k.clone(), v.clone());
        }
    }
}

pub fn graph_param_to_hf_key(param: &str) -> String {
    format!("{PREFIX_CODEC}{param}")
}

pub fn hf_key_to_graph_param(key: &str) -> Option<String> {
    key.strip_prefix(PREFIX_CODEC).map(str::to_string)
}

/// `lora.0.wq_a` → `layers.0.lora.wq_a`
pub fn lora_param_to_hf_key(param: &str) -> String {
    let rest = param.strip_prefix("lora.").unwrap_or(param);
    if let Some((li, tail)) = rest.split_once('.') {
        format!("layers.{li}.lora.{tail}")
    } else {
        param.to_string()
    }
}

/// Infer safetensors shape `[rank, dim]` for a LoRA param from flat length.
pub fn lora_param_shape(name: &str, len: usize) -> Result<Vec<usize>> {
    let (_, side) = name
        .rsplit_once('.')
        .ok_or_else(|| anyhow::anyhow!("bad lora param name {name}"))?;
    let rank = infer_rank(len, side)?;
    let dim = len / rank;
    Ok(vec![rank, dim])
}

fn infer_rank(len: usize, hint: &str) -> Result<usize> {
    for rank in [4usize, 8, 16, 32, 64, 128] {
        if len.is_multiple_of(rank) {
            return Ok(rank);
        }
    }
    bail!("cannot infer lora rank for len={len} ({hint})")
}

pub fn init_lora_param(
    name: &str,
    rank: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
) -> Vec<f32> {
    let n = if name.ends_with("w2_a") || name.ends_with("w1_b") || name.ends_with("w3_b") {
        rank * ffn_dim
    } else if name.ends_with("wq_b") || name.ends_with("wo_a") {
        rank * q_dim
    } else if name.ends_with("_a") {
        rank * hidden
    } else if name.ends_with("wk_b") || name.ends_with("wv_b") {
        rank * kv_dim
    } else {
        rank * hidden
    };
    vec![0.001; n.max(1)]
}

fn backbone_linear_needs_transpose(name: &str) -> bool {
    name.ends_with(".attention.wq.weight")
        || name.ends_with(".attention.wk.weight")
        || name.ends_with(".attention.wv.weight")
        || name.ends_with(".attention.wo.weight")
        || name.ends_with(".feed_forward.w1.weight")
        || name.ends_with(".feed_forward.w2.weight")
        || name.ends_with(".feed_forward.w3.weight")
}

fn transpose_linear_weight(data: Vec<f32>, out_rows: usize, in_cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..out_rows {
        for c in 0..in_cols {
            out[c * out_rows + r] = data[r * in_cols + c];
        }
    }
    out
}

/// `layers.0.lora.wq_a` → `lora.0.wq_a`
pub fn hf_lora_key_to_graph_param(key: &str) -> Option<String> {
    let rest = key.strip_prefix("layers.")?;
    let (li, tail) = rest.split_once('.')?;
    let tail = tail.strip_prefix("lora.")?;
    Some(format!("lora.{li}.{tail}"))
}

/// Overlay trained codec encoder tensors from a safetensors export onto a codec snapshot.
pub fn merge_codec_encoder_overlay(
    codec: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    encoder_path: &Path,
) -> Result<()> {
    use safetensors::SafeTensors;
    let bytes = std::fs::read(encoder_path)
        .with_context(|| format!("read encoder overlay {}", encoder_path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    for key in st.names() {
        let hf_key = if key.starts_with(PREFIX_CODEC) {
            key.to_string()
        } else {
            graph_param_to_hf_key(key)
        };
        if !(hf_key.starts_with(PREFIX_CODEC)
            && (hf_key.contains("encoder_blocks") || hf_key.contains("input_proj")))
        {
            continue;
        }
        let view = st.tensor(key)?;
        let shape: Vec<usize> = view.shape().to_vec();
        let data = tensor_view_to_f32(view.data(), view.dtype())?;
        codec.insert(hf_key, (data, shape));
    }
    Ok(())
}

pub fn codec_has_encoder(codec: &HashMap<String, (Vec<f32>, Vec<usize>)>) -> bool {
    codec.keys().any(|k| {
        k.starts_with(PREFIX_CODEC) && (k.contains("input_proj") || k.contains("encoder_blocks"))
    })
}

pub(crate) fn tensor_view_to_f32(raw: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f32>> {
    match dtype {
        safetensors::Dtype::F32 => {
            ensure!(raw.len().is_multiple_of(4));
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        safetensors::Dtype::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect()),
        other => bail!("unsupported overlay dtype {other:?}"),
    }
}

pub fn load_lora_backbone_for_graph(
    model_dir: &Path,
    param_names: &[String],
) -> Result<WeightStore> {
    let store = VoxtralTtsWeightStore::open(model_dir)?;
    let snap = store.tensor_snapshot_for_backbone()?;
    let mut out = WeightStore::default();
    for name in param_names {
        if name.starts_with("lora.") {
            continue;
        }
        if name == "__zero" {
            continue;
        }
        let (data, shape) = snap
            .get(name)
            .cloned()
            .with_context(|| format!("missing backbone param {name}"))?;
        let data = if shape.len() == 2 && backbone_linear_needs_transpose(name) {
            transpose_linear_weight(data, shape[0], shape[1])
        } else {
            data
        };
        out.0.insert(name.clone(), data);
    }
    Ok(out)
}

pub fn load_codec_weights(
    model_dir: &Path,
    include_encoder: bool,
    codec: &CodecArgs,
) -> Result<(WeightStore, WeightStore)> {
    let store = VoxtralTtsWeightStore::open(model_dir)?;
    let mut map = store.load_prefix(PREFIX_CODEC)?;
    let mut encoder = WeightStore::default();
    let mut decoder = WeightStore::default();

    for key in map.keys().map(str::to_string).collect::<Vec<_>>() {
        let (data, _shape) = map.take(&key).with_context(|| format!("take {key}"))?;
        let Some(param) = hf_key_to_graph_param(&key) else {
            continue;
        };
        if is_encoder_param(&param) {
            if include_encoder {
                encoder.0.insert(param, data);
            }
        } else if is_decoder_param(&param) {
            decoder.0.insert(param, data);
        }
    }

    if decoder.0.is_empty() {
        bail!("no codec decoder tensors under {PREFIX_CODEC}");
    }
    if include_encoder && encoder.0.is_empty() {
        let mut snap = store.tensor_snapshot(PREFIX_CODEC)?;
        rlx_voxtral_tts::codec::encoder_seed::seed_encoder_from_decoder(&mut snap, codec)?;
        for (hf_key, (data, _shape)) in snap {
            let Some(param) = hf_key_to_graph_param(&hf_key) else {
                continue;
            };
            if is_encoder_param(&param) {
                encoder.0.insert(param, data);
            }
        }
        ensure!(
            !encoder.0.is_empty(),
            "failed to seed encoder weights from decoder snapshot"
        );
    }
    Ok((encoder, decoder))
}

/// Resize or seed host buffers so every graph param slot has the expected flat length.
pub fn fit_params_to_graph(weights: &mut WeightStore, slots: &[ParamSlot]) -> Result<()> {
    for slot in slots {
        let expected = slot.num_elems;
        ensure!(expected > 0, "param {} has zero elems", slot.name);
        match weights.0.get_mut(&slot.name) {
            Some(w) if w.len() == expected => {}
            Some(w) => {
                resize_param_buffer(w, expected);
            }
            None => {
                weights
                    .0
                    .insert(slot.name.clone(), default_graph_param(&slot.name, expected));
            }
        }
    }
    Ok(())
}

fn init_param_buffer(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.001 * (((i as f32 * 0.013) % 1.0) - 0.5))
        .collect()
}

fn resize_param_buffer(w: &mut Vec<f32>, expected: usize) {
    if w.len() > expected {
        w.truncate(expected);
        return;
    }
    if w.len() < expected {
        let old = w.len();
        w.resize(expected, 0.0);
        for (i, x) in w.iter_mut().enumerate().skip(old) {
            *x = 0.001 * (((i as f32 * 0.013) % 1.0) - 0.5);
        }
    }
}

/// Default host value for graph params not present in checkpoints (e.g. RMS `__beta`).
pub fn default_graph_param(name: &str, n: usize) -> Vec<f32> {
    if name.ends_with(".__beta") {
        vec![0.0; n]
    } else {
        init_param_buffer(n)
    }
}

#[cfg(test)]
fn input_proj_conv_elems(codec: &CodecArgs) -> usize {
    codec.dim * codec.pretransform_patch_size * codec.patch_proj_kernel_size
}

pub fn load_lora_base_weights(model_dir: &Path) -> Result<WeightMap> {
    let store = VoxtralTtsWeightStore::open(model_dir)?;
    store.load_prefix(rlx_voxtral_tts::load::PREFIX_BACKBONE)
}

pub fn is_encoder_param(name: &str) -> bool {
    name.starts_with("input_proj") || name.starts_with("encoder_blocks")
}

/// Safetensors shape for a flat codec graph param buffer.
pub fn graph_param_shape(name: &str, len: usize, codec: &CodecArgs) -> Vec<usize> {
    let dim = codec.dim;
    let hidden = codec.hidden_dim;
    if name.ends_with("feed_forward.w1.weight") || name.ends_with("feed_forward.w3.weight") {
        return vec![hidden, dim];
    }
    if name.ends_with("feed_forward.w2.weight") {
        return vec![dim, hidden];
    }
    if name.ends_with(".attention.wq.weight")
        || name.ends_with(".attention.wk.weight")
        || name.ends_with(".attention.wv.weight")
    {
        return vec![dim, dim];
    }
    if name.ends_with(".attention.wo.weight") {
        return vec![dim, dim];
    }
    if name.ends_with(".attention.q_norm.weight") || name.ends_with(".attention.k_norm.weight") {
        return vec![dim];
    }
    if name.ends_with(".conv.weight") {
        if name == "input_proj.conv.weight" {
            return vec![
                dim,
                codec.pretransform_patch_size,
                codec.patch_proj_kernel_size,
            ];
        }
        if name == "output_proj.conv.weight" {
            return vec![
                codec.pretransform_patch_size,
                dim,
                codec.patch_proj_kernel_size,
            ];
        }
        if name.contains("encoder_blocks") && len == dim * codec.latent_dim() * 3 {
            return vec![codec.latent_dim(), dim, 3];
        }
        for &k in codec
            .encoder_convs_kernels()
            .iter()
            .chain([3_usize, 4].iter())
        {
            if len == dim * dim * k {
                return vec![dim, dim, k];
            }
        }
    }
    if name.ends_with(".conv.parametrizations.weight.original0") {
        return vec![dim, 1, 1];
    }
    if name.ends_with(".conv.parametrizations.weight.original1") {
        if len == codec.pretransform_patch_size * dim * codec.patch_proj_kernel_size {
            return vec![
                codec.pretransform_patch_size,
                dim,
                codec.patch_proj_kernel_size,
            ];
        }
        if len == dim * 292 * 3 {
            return vec![dim, 292, 3];
        }
        if len == dim * dim * 4 {
            return vec![dim, dim, 4];
        }
    }
    if name.contains("_norm") || name.contains("_scale") || name.ends_with(".__beta") {
        return vec![len];
    }
    vec![len]
}

pub fn test_codec_with_dims(dim: usize, patch: usize, kernel: usize) -> CodecArgs {
    CodecArgs {
        channels: 1,
        sampling_rate: 24000,
        pretransform_patch_size: patch,
        patch_proj_kernel_size: kernel,
        semantic_codebook_size: 1024,
        semantic_dim: 256,
        acoustic_codebook_size: 1024,
        acoustic_dim: 36,
        dim,
        hidden_dim: dim * 2,
        head_dim: 128,
        n_heads: 8,
        n_kv_heads: 2,
        attn_sliding_window_size: 512,
        encoder_transformer_lengths_str: "1,1".into(),
        encoder_convs_kernels_str: "3,4".into(),
        encoder_convs_strides_str: "1,2".into(),
        decoder_transformer_lengths_str: "1,1".into(),
        decoder_convs_kernels_str: "3,4".into(),
        decoder_convs_strides_str: "1,2".into(),
    }
}

fn is_decoder_param(name: &str) -> bool {
    name.starts_with("decoder_blocks")
        || name.starts_with("output_proj")
        || name.starts_with("quantizer.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_voxtral_tts::config::CodecArgs;

    fn sample_codec() -> CodecArgs {
        CodecArgs {
            channels: 1,
            sampling_rate: 24000,
            pretransform_patch_size: 240,
            patch_proj_kernel_size: 7,
            semantic_codebook_size: 128,
            semantic_dim: 256,
            acoustic_codebook_size: 21,
            acoustic_dim: 36,
            dim: 1024,
            hidden_dim: 4096,
            head_dim: 128,
            n_heads: 8,
            n_kv_heads: 8,
            attn_sliding_window_size: 16,
            encoder_transformer_lengths_str: "1,1".into(),
            encoder_convs_kernels_str: "4,3".into(),
            encoder_convs_strides_str: "2,1".into(),
            decoder_transformer_lengths_str: "1,1".into(),
            decoder_convs_kernels_str: "3,4".into(),
            decoder_convs_strides_str: "1,2".into(),
        }
    }

    #[test]
    fn graph_param_shape_ffn_matches_codec() {
        let codec = sample_codec();
        assert_eq!(
            graph_param_shape(
                "encoder_blocks.0.layers.0.feed_forward.w1.weight",
                4096 * 1024,
                &codec
            ),
            vec![4096, 1024]
        );
        assert_eq!(
            graph_param_shape(
                "encoder_blocks.0.layers.0.feed_forward.w2.weight",
                1024 * 4096,
                &codec
            ),
            vec![1024, 4096]
        );
    }

    #[test]
    fn init_encoder_from_decoder_seeds_input_proj() {
        let dir = match std::env::var("RLX_VOXTRAL_TTS_DIR") {
            Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
            _ => {
                eprintln!(
                    "skip init_encoder_from_decoder_seeds_input_proj: set RLX_VOXTRAL_TTS_DIR"
                );
                return;
            }
        };
        let cfg = rlx_voxtral_tts::config::VoxtralTtsConfig::from_model_dir(&dir).unwrap();
        let codec = &cfg.audio_config.codec_args;
        let (enc, _dec) = load_codec_weights(&dir, true, codec).unwrap();
        let w = enc.get("input_proj.conv.weight").unwrap();
        assert_eq!(w.len(), input_proj_conv_elems(codec));
        assert!(
            enc.get("encoder_blocks.0.layers.0.attention.wq.weight")
                .is_some()
        );
        assert!(
            enc.get("encoder_blocks.7.conv.weight").is_some()
                || enc
                    .0
                    .keys()
                    .any(|k| k.starts_with("encoder_blocks.7.conv.parametrizations"))
        );
    }

    #[test]
    fn fit_params_to_graph_resizes_mismatch() {
        let mut weights = WeightStore::default();
        weights
            .0
            .insert("input_proj.conv.weight".into(), vec![0.0; 1720320]);
        let slots = vec![ParamSlot {
            name: "input_proj.conv.weight".into(),
            param: rlx_ir::NodeId(0),
            grad: None,
            trainable: true,
            num_elems: 1024 * 240 * 7,
        }];
        fit_params_to_graph(&mut weights, &slots).unwrap();
        assert_eq!(
            weights.get("input_proj.conv.weight").unwrap().len(),
            1024 * 240 * 7
        );
    }
}
