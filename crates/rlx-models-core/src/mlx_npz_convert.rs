// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// GPLv3 — see the workspace license header.

//! One-shot converter for **legacy mlx-examples Llama** checkpoints
//! (`mlx-community/Llama-2-7b-chat-mlx` and friends): a flat mlx `config.json`
//! (`dim`/`n_heads`/`n_layers`/…, `vocab_size` possibly `-1`) plus a
//! `weights.npz` whose tensors use **Meta-original** names
//! (`tok_embeddings`, `layers.N.attention.wq/wk/wv/wo`,
//! `layers.N.feed_forward.w1/w2/w3`, `attention_norm`/`ffn_norm`, `norm`,
//! `output`). This rewrites them into a standard **HuggingFace** layout
//! (`config.json` with `model_type=llama` + HF field names, `model.safetensors`
//! with `model.layers.N.self_attn.q_proj.weight` …), so the fully-supported HF
//! safetensors path loads it unchanged. Deliberately isolated — it touches no
//! shared dispatch/loader code, so it cannot regress any other model.
//!
//! Crucially it applies the HF **q/k permutation** (the inverse of what
//! `transformers`' `convert_llama_weights_to_hf.py` does when reading Meta
//! weights): Meta stores un-permuted `wq`/`wk`, but HF's contiguous-half
//! (NeoX) rotary — which `rlx-llama32` uses — expects the permuted layout.

use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::path::Path;

/// Summary of a conversion, for CLI reporting.
#[derive(Debug, Clone)]
pub struct MlxNpzConvertReport {
    pub n_tensors: usize,
    pub n_layers: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub tied_embeddings: bool,
    pub out_dir: String,
}

/// HF q/k row permutation (Meta → HF). `w` is row-major `[dim1, dim2]`; the row
/// axis is regrouped `[n_heads, dim1/n_heads/2, 2] → [n_heads, 2, dim1/n_heads/2]`
/// (the `transpose(1,2)` in the reference convert script). Returns the reordered
/// rows; `dim2` (columns) is untouched.
fn permute_qk(w: &[f32], dim1: usize, dim2: usize, n_heads: usize) -> Vec<f32> {
    let per_head = dim1 / n_heads; // rows per head (== head_dim)
    let half = per_head / 2;
    let mut out = vec![0f32; w.len()];
    for h in 0..n_heads {
        for t in 0..2 {
            for a in 0..half {
                // dest row after transpose(1,2)+reshape, source row before.
                let dst = h * per_head + t * half + a;
                let src = h * per_head + a * 2 + t;
                out[dst * dim2..dst * dim2 + dim2]
                    .copy_from_slice(&w[src * dim2..src * dim2 + dim2]);
            }
        }
    }
    out
}

/// Map a Meta-original tensor name to its HF equivalent. Returns `None` for
/// tensors HF drops (e.g. cached `rope.freqs`). The second field is `true` when
/// the tensor is a q/k projection that must be `permute_qk`-ed.
pub fn meta_name_to_hf(name: &str) -> Option<(String, bool)> {
    if name == "tok_embeddings.weight" {
        return Some(("model.embed_tokens.weight".into(), false));
    }
    if name == "output.weight" {
        return Some(("lm_head.weight".into(), false));
    }
    if name == "norm.weight" {
        return Some(("model.norm.weight".into(), false));
    }
    if name.starts_with("rope.") || name == "freqs" {
        return None; // cached rotary freqs — regenerated at load
    }
    // layers.{N}.{sub}
    let rest = name.strip_prefix("layers.")?;
    let (n, tail) = rest.split_once('.')?;
    let hf_tail: (String, bool) = match tail {
        "attention.wq.weight" => ("self_attn.q_proj.weight".into(), true),
        "attention.wk.weight" => ("self_attn.k_proj.weight".into(), true),
        "attention.wv.weight" => ("self_attn.v_proj.weight".into(), false),
        "attention.wo.weight" => ("self_attn.o_proj.weight".into(), false),
        "feed_forward.w1.weight" => ("mlp.gate_proj.weight".into(), false),
        "feed_forward.w3.weight" => ("mlp.up_proj.weight".into(), false),
        "feed_forward.w2.weight" => ("mlp.down_proj.weight".into(), false),
        "attention_norm.weight" => ("input_layernorm.weight".into(), false),
        "ffn_norm.weight" => ("post_attention_layernorm.weight".into(), false),
        _ => return None,
    };
    Some((format!("model.layers.{n}.{}", hf_tail.0), hf_tail.1))
}

/// Convert a legacy mlx-examples Llama directory (`config.json`, `weights.npz`,
/// `tokenizer.model`) into an HF-layout directory the standard safetensors path
/// can load. Writes `dst_dir/{config.json, model.safetensors}` and copies
/// `tokenizer.model` when present.
pub fn convert_mlx_npz_to_hf(src_dir: &Path, dst_dir: &Path) -> Result<MlxNpzConvertReport> {
    // ── flat mlx config ──
    let cfg_path = src_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).with_context(|| format!("reading {cfg_path:?}"))?;
    let cfg: serde_json::Value = serde_json::from_slice(&cfg_bytes)
        .with_context(|| format!("parsing flat mlx {cfg_path:?}"))?;
    let getu = |k: &str| {
        cfg.get(k)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
    };
    let getf = |k: &str| cfg.get(k).and_then(serde_json::Value::as_f64);
    let dim = getu("dim").ok_or_else(|| anyhow!("flat config missing `dim`"))?;
    let n_layers = getu("n_layers").ok_or_else(|| anyhow!("flat config missing `n_layers`"))?;
    let n_heads = getu("n_heads").ok_or_else(|| anyhow!("flat config missing `n_heads`"))?;
    let n_kv_heads = getu("n_kv_heads").unwrap_or(n_heads);
    let norm_eps = getf("norm_eps").unwrap_or(1e-5);
    let rope_theta = getf("rope_theta").unwrap_or(10000.0);
    let head_dim = dim / n_heads;
    let _kv_dim = n_kv_heads * head_dim;

    // ── npz (Meta-named dense F32 tensors) ──
    let npz = src_dir.join("weights.npz");
    if !npz.is_file() {
        bail!("expected {npz:?} (legacy mlx-examples layout)");
    }
    let weights = rlx_mlx_io::load_path(&npz).with_context(|| format!("loading {npz:?}"))?;

    // Derive dims the flat config leaves as `-1` / omits, from tensor shapes.
    let embed = weights
        .tensors
        .get("tok_embeddings.weight")
        .ok_or_else(|| anyhow!("npz missing tok_embeddings.weight"))?;
    let vocab_size = embed.shape[0];
    let inter = weights
        .tensors
        .get("layers.0.feed_forward.w1.weight")
        .map(|t| t.shape[0])
        .ok_or_else(|| anyhow!("npz missing layers.0.feed_forward.w1.weight"))?;
    let tied = !weights.tensors.contains_key("output.weight");

    // ── remap + permute → HF tensors (F32) ──
    let mut hf: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::with_capacity(weights.tensors.len());
    let mut names: Vec<&String> = weights.tensors.keys().collect();
    names.sort();
    for name in names {
        let t = &weights.tensors[name];
        let Some((hf_name, permute)) = meta_name_to_hf(name) else {
            continue; // dropped (rope.freqs etc.)
        };
        let data = t.data_f32.clone().ok_or_else(|| {
            anyhow!("{name}: expected dense F32 in npz (quantized legacy npz unsupported)")
        })?;
        let out = if permute {
            let (d1, d2) = (t.shape[0], t.shape[1]);
            let nh = if hf_name.contains("k_proj") {
                n_kv_heads
            } else {
                n_heads
            };
            permute_qk(&data, d1, d2, nh)
        } else {
            data
        };
        hf.push((hf_name, t.shape.clone(), out));
    }
    let n_tensors = hf.len();

    // ── write HF config.json + model.safetensors ──
    std::fs::create_dir_all(dst_dir).with_context(|| format!("mkdir {dst_dir:?}"))?;
    let hf_cfg = serde_json::json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": dim,
        "intermediate_size": inter,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "rms_norm_eps": norm_eps,
        "rope_theta": rope_theta,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": tied,
        "torch_dtype": "float32",
    });
    std::fs::write(
        dst_dir.join("config.json"),
        serde_json::to_vec_pretty(&hf_cfg)?,
    )
    .with_context(|| "writing HF config.json")?;

    // safetensors serialize: F32 little-endian bytes per tensor.
    let mut views: Vec<(String, safetensors::tensor::TensorView)> = Vec::with_capacity(n_tensors);
    let byte_bufs: Vec<Vec<u8>> = hf
        .iter()
        .map(|(_, _, d)| {
            let mut b = Vec::with_capacity(d.len() * 4);
            for &v in d {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b
        })
        .collect();
    for ((name, shape, _), bytes) in hf.iter().zip(&byte_bufs) {
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .map_err(|e| anyhow!("{name}: safetensors view: {e}"))?;
        views.push((name.clone(), view));
    }
    let meta: Option<HashMap<String, String>> = None;
    safetensors::serialize_to_file(views, meta, &dst_dir.join("model.safetensors"))
        .map_err(|e| anyhow!("writing model.safetensors: {e}"))?;

    // Copy the tokenizer (SentencePiece `tokenizer.model`, or `tokenizer.json`).
    for tk in ["tokenizer.model", "tokenizer.json", "tokenizer_config.json"] {
        let s = src_dir.join(tk);
        if s.is_file() {
            let _ = std::fs::copy(&s, dst_dir.join(tk));
        }
    }

    Ok(MlxNpzConvertReport {
        n_tensors,
        n_layers,
        hidden_size: dim,
        n_heads,
        n_kv_heads,
        vocab_size,
        intermediate_size: inter,
        tied_embeddings: tied,
        out_dir: dst_dir.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_remap_covers_meta_llama() {
        assert_eq!(
            meta_name_to_hf("tok_embeddings.weight").unwrap().0,
            "model.embed_tokens.weight"
        );
        assert_eq!(
            meta_name_to_hf("output.weight").unwrap().0,
            "lm_head.weight"
        );
        assert_eq!(
            meta_name_to_hf("norm.weight").unwrap().0,
            "model.norm.weight"
        );
        let (q, pq) = meta_name_to_hf("layers.3.attention.wq.weight").unwrap();
        assert_eq!(q, "model.layers.3.self_attn.q_proj.weight");
        assert!(pq, "q_proj must be permuted");
        let (k, pk) = meta_name_to_hf("layers.3.attention.wk.weight").unwrap();
        assert_eq!(k, "model.layers.3.self_attn.k_proj.weight");
        assert!(pk, "k_proj must be permuted");
        assert!(!meta_name_to_hf("layers.3.attention.wv.weight").unwrap().1);
        assert_eq!(
            meta_name_to_hf("layers.0.feed_forward.w1.weight")
                .unwrap()
                .0,
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            meta_name_to_hf("layers.0.feed_forward.w3.weight")
                .unwrap()
                .0,
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            meta_name_to_hf("layers.0.feed_forward.w2.weight")
                .unwrap()
                .0,
            "model.layers.0.mlp.down_proj.weight"
        );
        assert_eq!(
            meta_name_to_hf("layers.0.attention_norm.weight").unwrap().0,
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            meta_name_to_hf("layers.0.ffn_norm.weight").unwrap().0,
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert!(meta_name_to_hf("rope.freqs").is_none());
    }

    #[test]
    fn permute_qk_is_a_row_permutation_matching_hf() {
        // 2 heads, head_dim 4, dim2 3. Row r = head*4 + inner. permute regroups
        // inner [0,1,2,3] (== [a0t0,a0t1,a1t0,a1t1]) → [a0t0,a1t0,a0t1,a1t1].
        let (n_heads, head_dim, dim2) = (2usize, 4usize, 3usize);
        let dim1 = n_heads * head_dim;
        let w: Vec<f32> = (0..dim1 * dim2).map(|i| i as f32).collect();
        let p = permute_qk(&w, dim1, dim2, n_heads);
        // Reference index math on the row axis.
        let per_head = head_dim;
        let half = per_head / 2;
        for h in 0..n_heads {
            for t in 0..2 {
                for a in 0..half {
                    let dst = h * per_head + t * half + a;
                    let src = h * per_head + a * 2 + t;
                    for c in 0..dim2 {
                        assert_eq!(p[dst * dim2 + c], w[src * dim2 + c], "h{h} t{t} a{a} c{c}");
                    }
                }
            }
        }
        // It is a permutation: same multiset of rows.
        let mut a = w.clone();
        let mut b = p.clone();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(a, b);
    }
}
