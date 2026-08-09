//! Native safetensors weight loader for `dsm_tts_*.safetensors`.
//!
//! Returns a `HashMap<key, (Vec<f32>, Vec<usize>)>` matching the same eager
//! layout used by [`crate::transformer::StreamingTransformer::new`] and the
//! other native modules. Supports F32, F16, and BF16 tensors — everything is
//! widened to f32 at load time so the rest of the runtime stays single-dtype.
//!
//! The actual *key map* for the Kyutai TTS layout is published by the
//! upstream `moshi` 0.6.4 `tts` module and is parameterised by
//! [`crate::config::KyutaiTtsConfig`]. The constants in this module document
//! the expected names — see [`expected_kyutai_tts_keys`].
//!
//! This loader is dependency-light: only `safetensors` + `half`, no candle.

use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// (`flat data`, `shape`) tuple — the canonical eager weight value.
pub type WeightEntry = (Vec<f32>, Vec<usize>);

/// Flat `name → (values, shape)` map used by every native module.
pub type WeightMap = HashMap<String, WeightEntry>;

/// Load a `.safetensors` file and widen every tensor to f32.
pub fn load_weight_map(path: &Path) -> Result<WeightMap> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse safetensors {}", path.display()))?;
    let mut out: WeightMap = HashMap::new();
    for (name, view) in st.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let data =
            decode_tensor(view.dtype(), view.data()).with_context(|| format!("tensor {name}"))?;
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            bail!(
                "tensor {name}: shape product {expected} != element count {}",
                data.len()
            );
        }
        out.insert(name.to_string(), (data, shape));
    }
    Ok(out)
}

/// Decode a raw tensor byte buffer into f32 values.
pub fn decode_tensor(dtype: Dtype, raw: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if !raw.len().is_multiple_of(4) {
                bail!("F32 buffer length {} not divisible by 4", raw.len());
            }
            let n = raw.len() / 4;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 4;
                out.push(f32::from_le_bytes([
                    raw[off],
                    raw[off + 1],
                    raw[off + 2],
                    raw[off + 3],
                ]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            if !raw.len().is_multiple_of(2) {
                bail!("F16 buffer length {} not divisible by 2", raw.len());
            }
            let n = raw.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 2;
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                out.push(f16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            if !raw.len().is_multiple_of(2) {
                bail!("BF16 buffer length {} not divisible by 2", raw.len());
            }
            let n = raw.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 2;
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                out.push(bf16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        other => Err(anyhow!("unsupported safetensors dtype {other:?}")),
    }
}

/// Storage dtype for safetensors round-trip tests (F32 / F16 / BF16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightStorageDtype {
    F32,
    F16,
    Bf16,
}

/// Widen f32 values through a narrower storage dtype (simulates checkpoint precision).
pub fn roundtrip_f32_via_storage(data: &[f32], storage: WeightStorageDtype) -> Vec<f32> {
    match storage {
        WeightStorageDtype::F32 => data.to_vec(),
        WeightStorageDtype::F16 => data.iter().map(|&v| f16::from_f32(v).to_f32()).collect(),
        WeightStorageDtype::Bf16 => data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect(),
    }
}

/// Clone a weight map with every tensor cast through `storage` and back to f32.
pub fn roundtrip_weight_map(weights: &WeightMap, storage: WeightStorageDtype) -> WeightMap {
    weights
        .iter()
        .map(|(k, (data, shape))| {
            (
                k.clone(),
                (roundtrip_f32_via_storage(data, storage), shape.clone()),
            )
        })
        .collect()
}

/// Strip a leading prefix from every weight key, returning a fresh map.
pub fn strip_prefix(weights: &WeightMap, prefix: &str) -> WeightMap {
    let mut out = WeightMap::new();
    for (k, v) in weights {
        if let Some(rest) = k.strip_prefix(prefix) {
            out.insert(rest.to_string(), v.clone());
        }
    }
    out
}

/// Verify every expected key exists; return the missing ones (empty = OK).
pub fn missing_keys(weights: &WeightMap, expected: &[String]) -> Vec<String> {
    expected
        .iter()
        .filter(|k| !weights.contains_key(*k))
        .cloned()
        .collect()
}

/// Expected key inventory for the Kyutai TTS 1.6B en/fr checkpoint.
///
/// Matches `kyutai/tts-1.6b-en_fr/dsm_tts_1e68beda@240.safetensors` as loaded by
/// [`crate::model::KyutaiTtsModel`], [`crate::depformer_stream::DepformerStream`],
/// and `crate::model::ConditionerBundle`.
pub fn expected_kyutai_tts_keys(cfg: &crate::config::KyutaiTtsConfig) -> Vec<String> {
    // Text embedding (demuxed second stream).
    let mut keys = vec![
        "text_emb.weight".into(),
        "text_emb.out1.weight".into(),
        "text_emb.out2.weight".into(),
        "text_linear.weight".into(),
    ];

    // Per-codebook backbone input embeddings (dense `[card, dim]`).
    for q in 0..cfg.n_q {
        keys.push(format!("emb.{q}.weight"));
    }

    // Backbone temporal transformer layers.
    for li in 0..cfg.num_layers {
        let p = format!("transformer.layers.{li}.");
        keys.push(format!("{p}norm1.alpha"));
        keys.push(format!("{p}norm2.alpha"));
        keys.push(format!("{p}self_attn.in_proj_weight"));
        keys.push(format!("{p}self_attn.out_proj.weight"));
        keys.push(format!("{p}gating.linear_in.weight"));
        keys.push(format!("{p}gating.linear_out.weight"));
        if cfg.cross_attention {
            keys.push(format!("{p}norm_cross.weight"));
            keys.push(format!("{p}norm_cross.bias"));
            keys.push(format!("{p}cross_attention.in_proj_weight"));
            keys.push(format!("{p}cross_attention.out_proj.weight"));
        }
    }

    // DepFormer depth decoder.
    let num_heads_unique = cfg
        .depformer
        .weights_per_step_schedule
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(11);
    for h in 0..num_heads_unique {
        keys.push(format!("depformer_in.{h}.weight"));
    }
    for cb in 0..cfg.dep_q {
        keys.push(format!("linears.{cb}.weight"));
    }
    keys.push("depformer_text_emb.weight".into());
    keys.push("depformer_text_emb.low_rank.weight".into());
    for cb in 0..cfg.dep_q.saturating_sub(1) {
        keys.push(format!("depformer_emb.{cb}.weight"));
        keys.push(format!("depformer_emb.{cb}.low_rank.weight"));
    }
    for li in 0..cfg.depformer.num_layers {
        let p = format!("depformer.layers.{li}.");
        keys.push(format!("{p}norm1.alpha"));
        keys.push(format!("{p}norm2.alpha"));
        keys.push(format!("{p}self_attn.in_proj_weight"));
        keys.push(format!("{p}self_attn.out_proj.weight"));
        for h in 0..num_heads_unique {
            keys.push(format!("{p}gating.{h}.linear_in.weight"));
            keys.push(format!("{p}gating.{h}.linear_out.weight"));
        }
    }

    // Conditioners (`condition_provider` prefix in the checkpoint).
    let pfx = "condition_provider.conditioners.";
    keys.push(format!("{pfx}cfg.embed.weight"));
    keys.push(format!("{pfx}cfg.output_proj.weight"));
    keys.push(format!("{pfx}control.embed.weight"));
    keys.push(format!("{pfx}control.output_proj.weight"));
    keys.push(format!("{pfx}speaker_wavs.output_proj.weight"));

    keys.push("out_norm.alpha".into());

    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_view_f32(vals: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    fn make_view_f16(vals: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            bytes.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
        }
        bytes
    }

    fn make_view_bf16(vals: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            bytes.extend_from_slice(&bf16::from_f32(*v).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn decode_f32_round_trips() {
        let want = vec![0.0f32, 1.0, -2.5, 3.125];
        let raw = make_view_f32(&want);
        let got = decode_tensor(Dtype::F32, &raw).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn decode_f16_widens_to_f32() {
        let want = vec![0.0f32, 1.0, -0.5, 2.0];
        let raw = make_view_f16(&want);
        let got = decode_tensor(Dtype::F16, &raw).unwrap();
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-3, "got {a} want {b}");
        }
    }

    #[test]
    fn decode_bf16_widens_to_f32() {
        let want = vec![1.0f32, -2.0, 0.5];
        let raw = make_view_bf16(&want);
        let got = decode_tensor(Dtype::BF16, &raw).unwrap();
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-2, "got {a} want {b}");
        }
    }

    #[test]
    fn decode_rejects_odd_f32_buffer() {
        // 7 bytes is not a valid F32 payload.
        let raw = vec![0u8; 7];
        assert!(decode_tensor(Dtype::F32, &raw).is_err());
    }

    #[test]
    fn strip_prefix_keeps_only_matching_keys() {
        let mut w = WeightMap::new();
        w.insert(
            "transformer.layers.0.norm1.alpha".into(),
            (vec![1.0], vec![1]),
        );
        w.insert("text_emb.weight".into(), (vec![2.0, 3.0], vec![1, 2]));
        let inner = strip_prefix(&w, "transformer.");
        assert!(inner.contains_key("layers.0.norm1.alpha"));
        assert!(!inner.contains_key("text_emb.weight"));
        assert_eq!(inner.len(), 1);
    }

    #[test]
    fn expected_keys_inventory_covers_published_config() {
        let cfg = crate::config::KyutaiTtsConfig::v1_6b_en_fr();
        let keys = expected_kyutai_tts_keys(&cfg);
        let emb_keys = keys.iter().filter(|k| k.starts_with("emb.")).count();
        assert_eq!(emb_keys, cfg.n_q, "one dense emb per codebook");
        let per_layer = 6 + if cfg.cross_attention { 4 } else { 0 };
        let backbone_keys = keys
            .iter()
            .filter(|k| k.starts_with("transformer.layers."))
            .count();
        assert_eq!(backbone_keys, cfg.num_layers * per_layer);
        let dep_layer_keys = keys
            .iter()
            .filter(|k| k.starts_with("depformer.layers."))
            .count();
        assert_eq!(dep_layer_keys, cfg.depformer.num_layers * (4 + 11 * 2));
        assert_eq!(
            keys.iter()
                .filter(|k| k.starts_with("depformer_in."))
                .count(),
            11
        );
        assert_eq!(
            keys.iter().filter(|k| k.starts_with("linears.")).count(),
            cfg.dep_q
        );
        for sentinel in [
            "text_emb.out1.weight",
            "text_linear.weight",
            "out_norm.alpha",
            "condition_provider.conditioners.cfg.embed.weight",
        ] {
            assert!(keys.contains(&sentinel.to_string()), "missing {sentinel}");
        }
    }

    /// When real weights are on disk, every expected key must be present.
    #[test]
    fn expected_keys_match_checkpoint_when_present() {
        let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| crate::download::default_kyutai_tts_dir());
        let path = dir.join(crate::download::TTS_WEIGHTS_FILE);
        if !path.is_file() {
            eprintln!(
                "skip: no weights at {} (fetch with --fetch)",
                path.display()
            );
            return;
        }
        let cfg = crate::config::KyutaiTtsConfig::v1_6b_en_fr();
        let weights = load_weight_map(&path).expect("load checkpoint");
        let expected = expected_kyutai_tts_keys(&cfg);
        let gaps = missing_keys(&weights, &expected);
        assert!(
            gaps.is_empty(),
            "checkpoint missing {} expected keys (first 5: {:?})",
            gaps.len(),
            &gaps[..gaps.len().min(5)]
        );
    }

    #[test]
    fn missing_keys_reports_gaps() {
        let cfg = crate::config::KyutaiTtsConfig::v1_6b_en_fr();
        let expected = expected_kyutai_tts_keys(&cfg);
        let mut have = WeightMap::new();
        have.insert("text_emb.weight".into(), (vec![], vec![]));
        let gaps = missing_keys(&have, &expected);
        assert!(!gaps.is_empty());
        assert!(!gaps.contains(&"text_emb.weight".to_string()));
        assert!(gaps.contains(&"text_linear.weight".to_string()));
    }
}
