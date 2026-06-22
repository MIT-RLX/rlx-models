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
/// This is the published key layout under
/// `kyutai/tts-1.6b-en_fr/dsm_tts_1e68beda@240.safetensors`. Keys are derived
/// directly from the architecture in [`crate::config::KyutaiTtsConfig`]:
///
/// - Text embedding + text out projection
/// - Per-codebook input embeddings (low-rank pair `A` / `B`)
/// - Backbone temporal layers (norm1, norm2, self_attn, gating, optional cross_attn + norm_cross)
/// - DepFormer per-head transformer slices (in_proj, out_proj, FFN, RMSNorm)
/// - Conditioner tables (LUT + tensor)
/// - Output RMSNorm
///
/// The map is deterministic — useful for parity tests against any safetensors
/// file you point at it.
pub fn expected_kyutai_tts_keys(cfg: &crate::config::KyutaiTtsConfig) -> Vec<String> {
    let mut keys = Vec::new();

    // Text embedding + output projection.
    keys.push("text_emb.weight".into());
    keys.push("text_linear.weight".into());

    // Per-codebook low-rank input embeddings (A: [card, rank], B: [rank, dim]).
    for q in 0..cfg.n_q {
        keys.push(format!("emb.{q}.low_rank.a"));
        keys.push(format!("emb.{q}.low_rank.b"));
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
            keys.push(format!("{p}norm_cross.alpha"));
            keys.push(format!("{p}cross_attention.in_proj_q.weight"));
            keys.push(format!("{p}cross_attention.in_proj_k.weight"));
            keys.push(format!("{p}cross_attention.in_proj_v.weight"));
            keys.push(format!("{p}cross_attention.out_proj.weight"));
        }
    }

    // DepFormer: distinct heads from the schedule (1.6B → 11 heads for 32 codebooks).
    let mut heads: Vec<usize> = cfg.depformer.weights_per_step_schedule.clone();
    heads.sort_unstable();
    heads.dedup();
    for h in heads {
        let p = format!("depformer.heads.{h}.");
        keys.push(format!("{p}in_proj.weight"));
        keys.push(format!("{p}out_proj.weight"));
        keys.push(format!("{p}gating.linear_in.weight"));
        keys.push(format!("{p}gating.linear_out.weight"));
        keys.push(format!("{p}norm.alpha"));
    }

    // Conditioner tables: one per named conditioner.
    for name in cfg.conditioners.keys() {
        // LUT and tensor conditioners both ship a `.weight` table for their
        // embedding/projection, plus tensor conditioners include an
        // optional `.output_proj.weight`.
        keys.push(format!("conditioners.{name}.weight"));
    }

    // Final RMSNorm on the temporal stream.
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
        // Spot-check: 33 codebooks would be wrong; n_q is 32 → 64 low-rank entries.
        let low_rank_keys = keys.iter().filter(|k| k.starts_with("emb.")).count();
        assert_eq!(low_rank_keys, 2 * cfg.n_q, "expected 2 keys per codebook");
        // 16 backbone layers × {2 norms + 2 self-attn + 2 gate + (5 cross when enabled)}.
        let per_layer = 6 + if cfg.cross_attention { 5 } else { 0 };
        let backbone_keys = keys
            .iter()
            .filter(|k| k.starts_with("transformer.layers."))
            .count();
        assert_eq!(backbone_keys, cfg.num_layers * per_layer);
        // DepFormer: 11 unique heads × 5 weights each = 55.
        let head_keys = keys
            .iter()
            .filter(|k| k.starts_with("depformer.heads."))
            .count();
        assert_eq!(head_keys, 11 * 5);
        // 3 conditioners.
        let cond_keys = keys
            .iter()
            .filter(|k| k.starts_with("conditioners."))
            .count();
        assert_eq!(cond_keys, cfg.conditioners.len());
        // text_emb + text_linear + out_norm.
        for sentinel in ["text_emb.weight", "text_linear.weight", "out_norm.alpha"] {
            assert!(keys.contains(&sentinel.to_string()), "missing {sentinel}");
        }
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
