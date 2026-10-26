// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Graph params ↔ checkpoint tensors.
//!
//! Under [`crate::common::CanonicalParams`] every graph param that carries model
//! weights corresponds to exactly one HF checkpoint tensor; [`hf_name`] is that
//! correspondence (plus the layout change: the builders store `nn.Linear`
//! weights `[in, out]`, the checkpoint `[out, in]`). Params with no checkpoint
//! tensor behind them — zero RMSNorm betas, causal-conv padding, broadcast
//! helpers — map to `None` and are never trained or exported.
//!
//! On top of the mapping: writing trained weights back as a loadable checkpoint
//! ([`export_full`]: only the changed tensors are written, everything else is
//! linked from the source checkpoint, so it works for the 1.6 TB model), and
//! PEFT-format LoRA adapters ([`save_lora_adapter`] / [`load_lora_adapter`]).

use crate::source::layer_prefix;
use anyhow::{Context, Result, bail, ensure};
use safetensors::{Dtype, tensor::TensorView};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// How a graph param's data relates to its checkpoint tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Same elements in the same order (shape may differ by unit axes).
    Same,
    /// Graph `[in, out]`, checkpoint `[out, in]`.
    Transpose,
    /// Graph `[E, L, 2·mi]` fused gate/up bank; checkpoint per-expert MXFP4
    /// `experts.{e}.w1` / `w3` `[mi, L]`.
    ExpertGateUp,
    /// Graph `[E, mi, L]` down bank; checkpoint per-expert MXFP4 `experts.{e}.w2`
    /// `[L, mi]`.
    ExpertDown,
}

/// The checkpoint tensor behind canonical graph param `graph`, or `None` for a
/// param that is not a checkpoint tensor. For the expert banks the returned name
/// is the bank's prefix (`…block_sparse_moe.experts`).
pub fn hf_name(graph: &str) -> Option<(String, Layout)> {
    use Layout::*;
    let top = |s: &str| format!("language_model.model.{s}");
    match graph {
        "embed_tokens" => return Some((top("embed_tokens.weight"), Same)),
        "final_norm" => return Some((top("norm.weight"), Same)),
        "out_res.norm" => return Some((top("output_attn_res_norm.weight"), Same)),
        "out_res.proj" => return Some((top("output_attn_res_proj.weight"), Same)),
        "lm_head.weight" => return Some(("language_model.lm_head.weight".into(), Transpose)),
        _ => {}
    }
    let rest = graph.strip_prefix('l')?;
    let (idx, local) = rest.split_once('.')?;
    let i: usize = idx.parse().ok()?;
    let lp = layer_prefix(i);
    let t = |s: &str, l: Layout| Some((format!("{lp}.{s}"), l));
    match local {
        "input_ln" => return t("input_layernorm.weight", Same),
        "post_ln" => return t("post_attention_layernorm.weight", Same),
        "sa_res.norm" => return t("self_attention_res_norm.weight", Same),
        "sa_res.proj" => return t("self_attention_res_proj.weight", Same),
        "mlp_res.norm" => return t("mlp_res_norm.weight", Same),
        "mlp_res.proj" => return t("mlp_res_proj.weight", Same),
        _ => {}
    }
    if local.ends_with(".zero_beta")
        || local.ends_with(".causal_pad")
        || local.ends_with(".k_rot_ones")
        || local.ends_with(".moe_batch_zero")
        || local.contains(".tap")
    {
        return None;
    }
    if let Some(a) = local.strip_prefix("self_attn.") {
        return match a {
            "dt_bias" | "A_log" => t(&format!("self_attn.{a}"), Same),
            "o_norm" | "q_a_layernorm" | "kv_a_layernorm" | "q_conv1d" | "k_conv1d"
            | "v_conv1d" => t(&format!("self_attn.{a}.weight"), Same),
            p if p.ends_with("_proj") || p == "kv_a_proj_with_mqa" => {
                t(&format!("self_attn.{p}.weight"), Transpose)
            }
            _ => None,
        };
    }
    if let Some(m) = local.strip_prefix("mlp.") {
        return m
            .ends_with("_proj")
            .then(|| (format!("{lp}.mlp.{m}.weight"), Transpose));
    }
    if let Some(m) = local.strip_prefix("block_sparse_moe.") {
        let b = |s: &str, l: Layout| Some((format!("{lp}.block_sparse_moe.{s}"), l));
        return match m {
            "gate.weight" => b("gate.weight", Transpose),
            "gate.e_score_correction_bias" => b("gate.e_score_correction_bias", Same),
            "routed_expert_norm" => b("routed_expert_norm.weight", Same),
            "experts.gate_up" => b("experts", ExpertGateUp),
            "experts.down" => b("experts", ExpertDown),
            p if p.ends_with("_proj") => b(&format!("{p}.weight"), Transpose),
            _ => None,
        };
    }
    None
}

/// Is `graph` a weight a full fine-tune updates by default? Checkpoint tensors
/// only, minus the routed-expert banks (1.45 TB on the real model) and the
/// router's score-correction bias (a load-balancing statistic HF updates outside
/// the optimizer, not a learned weight).
pub fn trainable_by_default(graph: &str) -> bool {
    match hf_name(graph) {
        Some((_, Layout::ExpertGateUp | Layout::ExpertDown)) => false,
        Some((name, _)) => !name.ends_with("e_score_correction_bias"),
        None => false,
    }
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = v[r * cols + c];
        }
    }
    out
}

/// Encode row-major `[rows, cols]` as MXFP4 the way the checkpoint stores it:
/// `(codes [rows, cols/2], scales [rows, cols/32])`, E2M1 nibbles low-first and an
/// E8M0 exponent per 32-wide block chosen so the block max fits `±6`.
pub fn encode_mxfp4(v: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
    const LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut codes = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u8; rows * cols / 32];
    for r in 0..rows {
        for blk in 0..cols / 32 {
            let row = &v[r * cols + blk * 32..r * cols + blk * 32 + 32];
            let amax = row.iter().fold(0f32, |m, x| m.max(x.abs()));
            let e = if amax > 0.0 {
                ((amax / 6.0).log2().ceil() as i32 + 127).clamp(0, 254)
            } else {
                0
            };
            scales[r * cols / 32 + blk] = e as u8;
            let s = 2f32.powi(e - 127);
            for (o, &x) in row.iter().enumerate() {
                let mag = (x.abs() / s).min(6.0);
                let (mut best, mut err) = (0usize, f32::MAX);
                for (li, &lv) in LEVELS.iter().enumerate() {
                    let d = (lv - mag).abs();
                    if d < err {
                        best = li;
                        err = d;
                    }
                }
                let nib = best as u8 | if x < 0.0 && best != 0 { 8 } else { 0 };
                let c = blk * 32 + o;
                let byte = &mut codes[r * cols / 2 + c / 2];
                if c.is_multiple_of(2) {
                    *byte |= nib;
                } else {
                    *byte |= nib << 4;
                }
            }
        }
    }
    (codes, scales)
}

enum Out {
    F32(Vec<f32>),
    Bf16(Vec<u8>),
    U8(Vec<u8>),
}

/// Convert trained graph params into checkpoint tensors `name → (shape, data)`.
/// `shape_of` returns the checkpoint shape of a tensor (from the source index).
fn to_checkpoint_tensors(
    trained: &HashMap<String, Vec<f32>>,
    graph_shapes: &HashMap<String, Vec<usize>>,
    mut shape_of: impl FnMut(&str) -> Result<Vec<usize>>,
    bf16: bool,
) -> Result<BTreeMap<String, (Vec<usize>, Out)>> {
    let mut out = BTreeMap::new();
    let float = |v: Vec<f32>| {
        if bf16 {
            Out::Bf16(
                v.iter()
                    .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
                    .collect(),
            )
        } else {
            Out::F32(v)
        }
    };
    for (gname, values) in trained {
        let Some((hf, layout)) = hf_name(gname) else {
            bail!("{gname} is not a checkpoint tensor");
        };
        let gshape = graph_shapes
            .get(gname)
            .with_context(|| format!("no shape recorded for {gname}"))?;
        match layout {
            Layout::Same => {
                let shape = shape_of(&hf)?;
                ensure!(
                    shape.iter().product::<usize>() == values.len(),
                    "{gname}: {} values for checkpoint shape {shape:?}",
                    values.len()
                );
                // bf16 export rounds every float tensor, including the ones the
                // source keeps in f32 (conv, A_log, dt_bias); pass `bf16 = false`
                // to keep full precision.
                out.insert(hf, (shape, float(values.clone())));
            }
            Layout::Transpose => {
                ensure!(
                    gshape.len() == 2,
                    "{gname}: transpose of rank {}",
                    gshape.len()
                );
                let (rows, cols) = (gshape[0], gshape[1]);
                out.insert(hf, (vec![cols, rows], float(transpose(values, rows, cols))));
            }
            Layout::ExpertGateUp | Layout::ExpertDown => {
                ensure!(
                    gshape.len() == 3,
                    "{gname}: expert bank of rank {}",
                    gshape.len()
                );
                let (e, a, b) = (gshape[0], gshape[1], gshape[2]);
                for x in 0..e {
                    let bank = &values[x * a * b..(x + 1) * a * b];
                    let mut put = |w: &str, rows: usize, cols: usize, m: Vec<f32>| {
                        let (codes, scales) = encode_mxfp4(&m, rows, cols);
                        out.insert(
                            format!("{hf}.{x}.{w}.weight_packed"),
                            (vec![rows, cols / 2], Out::U8(codes)),
                        );
                        out.insert(
                            format!("{hf}.{x}.{w}.weight_scale"),
                            (vec![rows, cols / 32], Out::U8(scales)),
                        );
                    };
                    if layout == Layout::ExpertGateUp {
                        // [L, 2mi] → w1 = gateᵀ [mi, L], w3 = upᵀ [mi, L]
                        let (l, mi2) = (a, b);
                        let mi = mi2 / 2;
                        let mut w1 = vec![0f32; mi * l];
                        let mut w3 = vec![0f32; mi * l];
                        for r in 0..l {
                            for o in 0..mi {
                                w1[o * l + r] = bank[r * mi2 + o];
                                w3[o * l + r] = bank[r * mi2 + mi + o];
                            }
                        }
                        put("w1", mi, l, w1);
                        put("w3", mi, l, w3);
                    } else {
                        // [mi, L] → w2 = downᵀ [L, mi]
                        put("w2", b, a, transpose(bank, a, b));
                    }
                }
            }
        }
    }
    Ok(out)
}

fn write_safetensors(path: &Path, tensors: &BTreeMap<String, (Vec<usize>, Out)>) -> Result<()> {
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, (shape, data))| {
            let (dt, bytes): (Dtype, &[u8]) = match data {
                Out::F32(v) => (Dtype::F32, bytemuck_f32(v)),
                Out::Bf16(b) => (Dtype::BF16, b),
                Out::U8(b) => (Dtype::U8, b),
            };
            TensorView::new(dt, shape.clone(), bytes)
                .map(|v| (name.clone(), v))
                .map_err(|e| anyhow::anyhow!("{name}: {e:?}"))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(views, None, path)
        .map_err(|e| anyhow::anyhow!("writing {path:?}: {e:?}"))
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // f32 has no padding and every bit pattern is a valid u8 sequence.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Write `trained` (canonical graph name → values) as a checkpoint in `out_dir`
/// that loads like the source: the changed tensors go to one new shard
/// (`model-trained.safetensors`, bf16 when `bf16`, else f32), every other shard is
/// **symlinked** from `src_dir` (copied where symlinks are unavailable), and the
/// index points each tensor at its shard. `config.json` and tokenizer files are
/// copied. `graph_shapes` gives each trained param's graph shape.
pub fn export_full(
    src_dir: &Path,
    out_dir: &Path,
    trained: &HashMap<String, Vec<f32>>,
    graph_shapes: &HashMap<String, Vec<usize>>,
    bf16: bool,
) -> Result<()> {
    ensure!(
        src_dir.canonicalize()? != out_dir.canonicalize().unwrap_or_default(),
        "export target must differ from the source checkpoint"
    );
    std::fs::create_dir_all(out_dir)?;
    let mut ck = crate::loader::CheckpointLoader::open(src_dir)?;
    let tensors = to_checkpoint_tensors(trained, graph_shapes, |n| ck.tensor_shape(n), bf16)?;
    const SHARD: &str = "model-trained.safetensors";
    write_safetensors(&out_dir.join(SHARD), &tensors)?;

    let idx: serde_json::Value = serde_json::from_reader(std::fs::File::open(
        src_dir.join("model.safetensors.index.json"),
    )?)?;
    let mut map = idx
        .get("weight_map")
        .and_then(|v| v.as_object())
        .cloned()
        .context("source index has no weight_map")?;
    let mut linked = std::collections::BTreeSet::new();
    for (name, file) in map.iter_mut() {
        if tensors.contains_key(name) {
            *file = serde_json::Value::String(SHARD.into());
        } else if let Some(f) = file.as_str() {
            linked.insert(f.to_string());
        }
    }
    for name in tensors.keys() {
        map.entry(name.clone())
            .or_insert_with(|| serde_json::Value::String(SHARD.into()));
    }
    for f in &linked {
        let (src, dst) = (src_dir.join(f).canonicalize()?, out_dir.join(f));
        if dst.symlink_metadata().is_ok() {
            std::fs::remove_file(&dst)?;
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(&src, &dst)?;
        #[cfg(not(unix))]
        std::fs::copy(&src, &dst).map(|_| ())?;
    }
    std::fs::write(
        out_dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "metadata": idx.get("metadata").cloned().unwrap_or_default(),
            "weight_map": map,
        }))?,
    )?;
    for extra in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ] {
        let p = src_dir.join(extra);
        if p.exists() {
            std::fs::copy(&p, out_dir.join(extra))?;
        }
    }
    Ok(())
}

/// Read canonical graph weight `graph` from a checkpoint in the graph's layout
/// (linear weights transposed to `[in, out]`). Not for expert banks.
pub fn load_graph_weight(
    ck: &mut crate::loader::CheckpointLoader,
    graph: &str,
) -> Result<Vec<f32>> {
    let (hf, layout) =
        hf_name(graph).with_context(|| format!("{graph}: not a checkpoint tensor"))?;
    let (shape, v) = ck.tensor_f32_shaped(&hf)?;
    match layout {
        Layout::Same => Ok(v),
        Layout::Transpose => {
            ensure!(shape.len() == 2, "{hf}: rank {}", shape.len());
            Ok(transpose(&v, shape[0], shape[1]))
        }
        _ => bail!("{graph}: expert banks are not single tensors"),
    }
}

/// PEFT module path for a LoRA-targeted graph weight (`lora.<weight>.a/.b`).
fn peft_module(graph_weight: &str) -> Result<String> {
    let (hf, layout) = hf_name(graph_weight)
        .with_context(|| format!("{graph_weight}: not a checkpoint tensor"))?;
    ensure!(
        layout == Layout::Transpose,
        "{graph_weight}: LoRA is only defined on linear weights"
    );
    Ok(hf.trim_end_matches(".weight").to_string())
}

/// Save LoRA adapters (`lora.<weight>.a` `[in, r]` / `.b` `[r, out]`, as produced
/// by `rlx_tune::inject_lora`) in **PEFT** layout: `adapter_model.safetensors`
/// with `base_model.model.<module>.lora_A.weight` `[r, in]` / `lora_B.weight`
/// `[out, r]`, and an `adapter_config.json`.
pub fn save_lora_adapter(
    dir: &Path,
    adapters: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    rank: usize,
    alpha: f32,
    target_modules: &[String],
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut tensors = BTreeMap::new();
    for (name, v) in adapters {
        let base = name
            .strip_prefix("lora.")
            .with_context(|| format!("{name}: not a LoRA param"))?;
        let (weight, ab) = base
            .rsplit_once('.')
            .with_context(|| format!("{name}: malformed LoRA name"))?;
        let module = peft_module(weight)?;
        let shape = shapes
            .get(name)
            .with_context(|| format!("no shape for {name}"))?;
        ensure!(shape.len() == 2, "{name}: rank-{} adapter", shape.len());
        let (rows, cols) = (shape[0], shape[1]);
        let which = match ab {
            "a" => "lora_A",
            "b" => "lora_B",
            _ => bail!("{name}: expected .a or .b"),
        };
        tensors.insert(
            format!("base_model.model.{module}.{which}.weight"),
            (vec![cols, rows], Out::F32(transpose(v, rows, cols))),
        );
    }
    write_safetensors(&dir.join("adapter_model.safetensors"), &tensors)?;
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": target_modules,
            "base_model_name_or_path": "moonshotai/Kimi-K3",
        }))?,
    )?;
    Ok(())
}

/// A LoRA adapter read back from [`save_lora_adapter`]'s layout.
pub struct LoraAdapter {
    /// `lora.<graph weight>.a/.b` → values in the graph layout.
    pub params: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
}

/// Inverse of [`hf_name`] for linear weights: PEFT module → graph weight name.
fn graph_weight_for_module(module: &str) -> Option<String> {
    let hf = format!("{module}.weight");
    if hf == "language_model.lm_head.weight" {
        return Some("lm_head.weight".into());
    }
    let rest = hf.strip_prefix("language_model.model.layers.")?;
    let (i, local) = rest.split_once('.')?;
    let local = local.strip_suffix(".weight")?;
    let g = format!("l{i}.{local}");
    (hf_name(&g).map(|(n, _)| n) == Some(hf)).then_some(g)
}

/// Read an adapter saved by [`save_lora_adapter`] (or any PEFT LoRA adapter on
/// Kimi-K3 linear modules).
pub fn load_lora_adapter(dir: &Path) -> Result<LoraAdapter> {
    let cfg: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(dir.join("adapter_config.json"))?)?;
    let rank = cfg["r"].as_u64().context("adapter_config: r")? as usize;
    let alpha = cfg["lora_alpha"].as_f64().unwrap_or(rank as f64) as f32;
    let target_modules = cfg["target_modules"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let bytes = std::fs::read(dir.join("adapter_model.safetensors"))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| anyhow::anyhow!("adapter_model.safetensors: {e:?}"))?;
    let mut params = HashMap::new();
    let mut shapes = HashMap::new();
    for (name, view) in st.tensors() {
        let body = name
            .strip_prefix("base_model.model.")
            .with_context(|| format!("{name}: not a PEFT key"))?;
        let (module, which) = if let Some(m) = body.strip_suffix(".lora_A.weight") {
            (m, "a")
        } else if let Some(m) = body.strip_suffix(".lora_B.weight") {
            (m, "b")
        } else {
            continue;
        };
        let gw = graph_weight_for_module(module)
            .with_context(|| format!("{module}: not a Kimi-K3 linear module"))?;
        let shape = view.shape().to_vec();
        ensure!(shape.len() == 2, "{name}: rank-{}", shape.len());
        let data: Vec<f32> = match view.dtype() {
            Dtype::F32 => view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::BF16 => view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => bail!("{name}: unsupported dtype {other:?}"),
        };
        let (rows, cols) = (shape[0], shape[1]);
        let key = format!("lora.{gw}.{which}");
        params.insert(key.clone(), transpose(&data, rows, cols));
        shapes.insert(key, vec![cols, rows]);
    }
    Ok(LoraAdapter {
        params,
        shapes,
        rank,
        alpha,
        target_modules,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_map_to_checkpoint_tensors() {
        let cases = [
            (
                "l3.self_attn.q_proj",
                "language_model.model.layers.3.self_attn.q_proj.weight",
            ),
            (
                "l3.self_attn.q_conv1d",
                "language_model.model.layers.3.self_attn.q_conv1d.weight",
            ),
            (
                "l3.self_attn.A_log",
                "language_model.model.layers.3.self_attn.A_log",
            ),
            (
                "l0.mlp.down_proj",
                "language_model.model.layers.0.mlp.down_proj.weight",
            ),
            (
                "l5.sa_res.proj",
                "language_model.model.layers.5.self_attention_res_proj.weight",
            ),
            (
                "l2.block_sparse_moe.gate.weight",
                "language_model.model.layers.2.block_sparse_moe.gate.weight",
            ),
            (
                "l2.block_sparse_moe.shared_experts.up_proj",
                "language_model.model.layers.2.block_sparse_moe.shared_experts.up_proj.weight",
            ),
            ("lm_head.weight", "language_model.lm_head.weight"),
            (
                "out_res.proj",
                "language_model.model.output_attn_res_proj.weight",
            ),
        ];
        for (g, hf) in cases {
            assert_eq!(hf_name(g).map(|x| x.0).as_deref(), Some(hf), "{g}");
        }
        for g in [
            "l1.input_ln.zero_beta",
            "l1.self_attn.q_conv1d.causal_pad",
            "l2.self_attn.k_rot_ones",
            "l2.block_sparse_moe.moe_batch_zero",
        ] {
            assert!(hf_name(g).is_none(), "{g}");
        }
        assert!(!trainable_by_default("l2.block_sparse_moe.experts.gate_up"));
        assert!(!trainable_by_default(
            "l2.block_sparse_moe.gate.e_score_correction_bias"
        ));
        assert!(trainable_by_default("l2.block_sparse_moe.gate.weight"));
        assert_eq!(
            graph_weight_for_module("language_model.model.layers.4.self_attn.o_proj").as_deref(),
            Some("l4.self_attn.o_proj")
        );
    }

    #[test]
    fn mxfp4_round_trips_representable_values() {
        // Values on the e2m1 grid times a power of two survive exactly.
        let row: Vec<f32> = (0..32)
            .map(|i| [0.0, 0.5, -1.0, 1.5, 2.0, -3.0, 4.0, 6.0][i % 8] * 0.25)
            .collect();
        let (codes, scales) = encode_mxfp4(&row, 1, 32);
        const VAL: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let s = 2f32.powi(scales[0] as i32 - 127);
        for (c, want) in row.iter().enumerate() {
            let b = codes[c / 2];
            let nib = if c % 2 == 0 { b & 15 } else { b >> 4 };
            assert_eq!(VAL[nib as usize] * s, *want, "col {c}");
        }
    }
}
