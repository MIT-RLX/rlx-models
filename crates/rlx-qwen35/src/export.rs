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

//! Export a Qwen3.5 GGUF as an `.rlxp` package that rlx runs with no
//! rlx-models code: the hybrid decode step (gated delta-net layers carrying
//! recurrent state, attention layers carrying a KV cache), its K-quant
//! projections kept packed and read from the GGUF as stored, the RoPE
//! tables, and the tokenizer and chat template the GGUF's metadata holds.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::{Checkpoint, Package, Referencer};
use rlx_core::weight_loader::GgufLoader;
use rlx_ir::Op;
use serde_json::{Value, json};

use crate::{Qwen35Config, Qwen35Weights, build_qwen35_decode_graph_ext};

pub use rlx_core::export::{ExportOptions, ExportReport};

/// Drafts a verify step checks when the options ask for none
/// (`speculate`): it takes the last token and them.
const DEFAULT_DRAFTS: usize = 1;

/// Cache capacities when the options list none.
const DEFAULT_UPPERS: [usize; 4] = [64, 256, 1024, 4096];

/// The `train` variant's sequence length when the options list none.
const DEFAULT_TRAIN_SEQ: usize = 128;

fn decode_graph(
    cfg: &Qwen35Config,
    gguf: &str,
    upper: usize,
    mtp: bool,
    // `Some(n)`: the `n`-token verify step (MTP only), its `n - 1` drafts
    // chained.
    verify: Option<usize>,
) -> Result<(
    rlx_ir::Graph,
    HashMap<String, Vec<f32>>,
    crate::PackedParams,
)> {
    let mut loader = GgufLoader::from_file(gguf)?;
    // The weights loader also reads the MTP heads (`nextn_predict_layers`),
    // which the GGUF loader refuses unless asked; only the `mtp` variant's
    // step uses them.
    loader.include_mtp(true);
    let weights = Qwen35Weights::from_loader_packed(&mut loader, cfg)?;
    if !mtp {
        return build_qwen35_decode_graph_ext(cfg, weights, 1, upper, true, false, false, false);
    }
    // The MTP head's draft for the token after next, its attention with a
    // cache of its own (`mtp_kv`) that grows with the trunk's.
    let opts = crate::flow::Qwen35DecodeOpts {
        use_custom_mask: true,
        enable_mtp_head: true,
        mtp_kv: true,
        ..crate::flow::Qwen35DecodeOpts::step(1, upper)
    };
    let opts = match verify {
        // The token and its drafts at once: logits for every row, the
        // trunk's causal mask derived from the per-key one. Its trailing
        // rows may be padding (a pass of fewer tokens), which the delta-net
        // scans skip and the drafts start before.
        Some(tokens) => crate::flow::Qwen35DecodeOpts {
            seq: tokens,
            verify_all: true,
            keys_mask: true,
            multi_row: true,
            mtp_drafts: tokens - 1,
            ..opts
        },
        None => opts,
    };
    let (built, packed) = crate::flow::build_qwen35_decode_built(cfg, weights, &opts)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    Ok((graph, params, packed))
}

fn meta_str(raw: &rlx_gguf::GgufFile, key: &str) -> Option<String> {
    raw.metadata
        .get(key)
        .and_then(rlx_gguf::MetaValue::as_str)
        .map(String::from)
}

fn meta_u64(raw: &rlx_gguf::GgufFile, key: &str) -> Option<u64> {
    match raw.metadata.get(key)? {
        rlx_gguf::MetaValue::U32(v) => Some(u64::from(*v)),
        rlx_gguf::MetaValue::U64(v) => Some(*v),
        rlx_gguf::MetaValue::I32(v) => u64::try_from(*v).ok(),
        rlx_gguf::MetaValue::I64(v) => u64::try_from(*v).ok(),
        _ => None,
    }
}

/// Export the GGUF `gguf` as the package `out`.
pub fn export_rlxp(gguf: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let path = gguf.to_str().context("GGUF path is not UTF-8")?;
    let raw = rlx_gguf::GgufFile::from_path(gguf).with_context(|| format!("open {path}"))?;
    let cfg = Qwen35Config::from_gguf(&raw)?;
    if opts.variant.as_deref() == Some("train") {
        return export_train(gguf, &raw, &cfg, out, opts);
    }
    if opts.variant.as_deref() == Some("distill") {
        return export_distill(gguf, &raw, &cfg, out, opts);
    }
    let mut uppers = opts.uppers_or(&DEFAULT_UPPERS);
    uppers.sort();
    let Some(&longest) = uppers.last() else {
        bail!("no cache capacity to export");
    };
    // The `mtp` variant: each step also outputs the MTP head's draft logits
    // (output 1), the states after them.
    let mtp = match opts.variant.as_deref() {
        None => false,
        Some("mtp") if cfg.nextn_predict_layers > 0 => true,
        Some("mtp") => bail!("the GGUF has no MTP layers (nextn_predict_layers = 0)"),
        Some(other) => {
            bail!("qwen35 has no variant '{other}' (only 'mtp', 'train' and 'distill')")
        }
    };
    let first_state = 1 + usize::from(mtp);

    // One symbolic step from two builds no other dim has, else one per size.
    let mut symbolic_failure = None;
    let mut graphs = Vec::new();
    let mut params = HashMap::new();
    let mut packed = crate::PackedParams::new();
    if opts.symbolic {
        let (a, pa, ka) = decode_graph(&cfg, path, 36, mtp, None)?;
        let (b, _, _) = decode_graph(&cfg, path, 52, mtp, None)?;
        match rlx_ir::dynamic::generalize(&a, &b, &[(0, 37, 53), (1, 36, 52)]) {
            Ok(mut g) => {
                g.name = "decode".into();
                graphs.push(g);
                params = pa;
                packed = ka;
            }
            Err(e) => symbolic_failure = Some(e),
        }
    }
    if graphs.is_empty() {
        for &upper in &uppers {
            let (mut g, p, k) = decode_graph(&cfg, path, upper, mtp, None)?;
            g.name = format!("decode_{upper}");
            if params.is_empty() {
                params = p;
                packed = k;
            }
            graphs.push(g);
        }
    }

    // The MTP variant's verify step: the last token and the draft, at once.
    // Symbolic only; it shares the step's states and their order.
    let mut verify_function = None;
    let verify_tokens = opts.speculate.max(DEFAULT_DRAFTS) + 1;
    if mtp && symbolic_failure.is_none() && opts.symbolic {
        let at = |upper| decode_graph(&cfg, path, upper, mtp, Some(verify_tokens));
        let (a, pa, ka) = at(36)?;
        let (b, c) = (at(52)?.0, at(44)?.0);
        // Its own params (the mask's constants) besides the step's.
        for (k, v) in pa {
            params.entry(k).or_insert(v);
        }
        for (k, v) in ka {
            packed.entry(k).or_insert(v);
        }
        // Its keys run up to `upper + 2 * tokens - 1` (the rows, then each
        // draft's), past `generalize`'s default offset, so fitted through
        // three sizes (in `upper` alone: `cap` moves with it) and checked
        // against the third.
        let reach = 2 * verify_tokens as i64;
        let mut g =
            rlx_ir::dynamic::generalize_checked(&a, &b, &c, &[(1, 36, 52, 44)], Some(reach))
                .map_err(|e| anyhow::anyhow!("the verify step is not symbolic: {e}"))?;
        let mut at_c = rlx_ir::DimBinding::new();
        at_c.set(0, 45);
        at_c.set(1, 44);
        if format!("{}", rlx_ir::dynamic::specialize(&g, &at_c)) != format!("{c}") {
            bail!("the symbolic verify step differs from the one built at its third size");
        }
        let names = |g: &rlx_ir::Graph| -> Vec<String> {
            g.nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Input { name }
                        if name.starts_with("past_")
                            || name.contains("_state_")
                            || name == "mtp_hidden" =>
                    {
                        Some(name.clone())
                    }
                    _ => None,
                })
                .collect()
        };
        if names(&g) != names(&graphs[0]) || g.outputs.len() != graphs[0].outputs.len() {
            bail!("the verify step's states differ from the decode step's");
        }
        g.name = "verify".into();
        graphs.push(g);
        verify_function = Some("verify");
    }

    // The f32 params against the GGUF's tensors as the runtime dequantizes
    // them, so a reference reads back exactly what was built.
    let packed_keys: std::collections::HashSet<String> =
        packed.values().map(|(k, _, _)| k.clone()).collect();
    // A packed tensor can also feed an f32 param (the tied LM head reads the
    // embedding's blob packed while the token lookup reads it as f32): keep
    // it as a candidate where some param is its size, without dequantizing
    // every packed matrix.
    let sizes: std::collections::HashSet<usize> = params.values().map(Vec::len).collect();
    let mut refr = Referencer::open_gguf_only(gguf, opts.embed_weights, &|name, n| {
        !packed_keys.contains(name) || sizes.contains(&n)
    })?;
    refr.take(std::mem::take(&mut params))?;
    for (param, (key, _, _)) in &packed {
        refr.packed(param, key);
    }
    let ck = Checkpoint::open(gguf)?;
    let mut pkg = Package::for_checkpoint(&ck, opts, "qwen35", "text-generation", "rlx-qwen35");
    pkg.graphs = graphs;
    pkg.reference(refr, out)?;

    // RoPE rows for every position the package reaches.
    let half = cfg.rope_dim_count / 2;
    let rows = longest + 1;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for pos in 0..rows {
        let (c, s) = crate::mrope_row_for_sections(&cfg, crate::text_section_pos(pos), half);
        if c.len() != half {
            bail!("a RoPE row is {} wide, not {half}", c.len());
        }
        cos.extend(c);
        sin.extend(s);
    }
    pkg.hold_f32("rope_cos", vec![rows, half], &cos);
    pkg.hold_f32("rope_sin", vec![rows, half], &sin);

    // Inputs, and each state paired with the output that updates it: the
    // states are inputs and outputs in the same (layer) order.
    let g = &pkg.graphs[0];
    let mut inputs = serde_json::Map::new();
    let mut state_inputs = Vec::new();
    for n in g.nodes() {
        let Op::Input { name } = &n.op else { continue };
        match name.as_str() {
            "input_ids" => {
                inputs.insert(name.clone(), json!("token"));
            }
            "mask" => {
                inputs.insert(name.clone(), json!("decode_mask"));
            }
            "rope_cos" | "rope_sin" => {
                inputs.insert(name.clone(), json!({"table": name}));
            }
            n if n.starts_with("conv_state_")
                || n.starts_with("ssm_state_")
                || n == "mtp_hidden" =>
            {
                state_inputs.push((n.to_string(), "replace"));
            }
            n if n.starts_with("past_k_") || n.starts_with("past_v_") => {
                state_inputs.push((n.to_string(), "append"));
            }
            other => bail!("decode input '{other}' has no feed"),
        }
    }
    if state_inputs.len() + first_state != g.outputs.len() {
        bail!(
            "{} states but {} outputs after the logits",
            state_inputs.len(),
            g.outputs.len() - first_state
        );
    }
    // The logits' own width: the LM head is often wider than
    // `cfg.vocab_size`, and the runtime takes the last `vocab_size` logits.
    let vocab = match g.node(g.outputs[0]).shape.dims().last() {
        Some(rlx_ir::Dim::Static(v)) => *v,
        _ => cfg.vocab_size,
    };
    let state: Vec<Value> = state_inputs
        .iter()
        .enumerate()
        .map(|(i, (n, kind))| json!([n, i + first_state, kind]))
        .collect();

    // The tokenizer and chat template the GGUF carries.
    let tokenizer = rlx_core::tokenizer::build_gguf_tokenizer(gguf)?;
    let tokenizer_json = tokenizer
        .to_string(false)
        .map_err(|e| anyhow::anyhow!("serialize tokenizer: {e}"))?;
    // The steps by size; the verify step is named by its own key.
    let functions: Vec<String> = pkg
        .graphs
        .iter()
        .map(|g| g.name.clone())
        .filter(|n| Some(n.as_str()) != verify_function)
        .collect();
    let tokenizer_id = pkg.sidecar(
        "tokenizer.json",
        "application/json",
        tokenizer_json.into_bytes(),
    );
    let mut chat = pkg.chat(&ck);
    let mut stop = Vec::new();
    for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
        if let Some(id) = meta_u64(&raw, key)
            && !stop.contains(&id)
        {
            stop.push(id);
        }
    }
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok).map(u64::from)
            && !stop.contains(&id)
        {
            stop.push(id);
        }
    }
    if let Some(id) = meta_u64(&raw, "tokenizer.ggml.eos_token_id")
        && let Some(t) = tokenizer.id_to_token(id as u32)
    {
        chat["eos_token"] = json!(t);
    }

    let mut entry = json!({
            "functions": functions,
            "sizes": uppers.iter().map(|u| json!({"cap": u + 1, "upper": u})).collect::<Vec<_>>(),
            "tokenizer": tokenizer_id,
            "chat": chat,
            "stop_ids": stop,
            "vocab_size": vocab,
            "inputs": inputs,
            "state": state,
            "logits": 0,
    });
    if mtp {
        // The MTP head's logits for the token after the one `logits` picks.
        entry["draft_logits"] = json!(1);
    }
    if let Some(f) = verify_function {
        // A speculative step: up to `tokens` new tokens (the last one, then
        // the drafts), every row's logits, and the next `drafts` drafts;
        // same states. A pass of fewer tokens pads the rest, its `rows`
        // inputs saying which are real.
        let g = pkg
            .graphs
            .iter()
            .find(|g| g.name == f)
            .context("the verify step")?;
        let mut rows = serde_json::Map::new();
        for n in g.nodes() {
            let Op::Input { name } = &n.op else { continue };
            let kind = if name.starts_with("gdn_pad_l") {
                "pad_rows"
            } else if name.starts_with("gdn_conv_shift_l") {
                "pad_shift"
            } else if name == "mtp_last_row" {
                "last_row"
            } else {
                continue;
            };
            rows.insert(name.clone(), json!(kind));
        }
        entry["verify"] = json!({
            "function": f,
            "tokens": verify_tokens,
            "drafts": verify_tokens - 1,
            "logits": 0,
            "draft_logits": 1,
            "rows": rows,
        });
    }
    pkg.entry("generate", entry);
    pkg.metadata("context_length", json!(longest));
    pkg.metadata("source", json!(meta_str(&raw, "general.basename")));
    for g in &mut pkg.graphs {
        let first = g.outputs[0];
        g.node_mut(first).name = Some("logits".into());
        if mtp {
            let draft = g.outputs[1];
            g.node_mut(draft).name = Some("draft_logits".into());
        }
    }
    if symbolic_failure.is_none() && opts.symbolic {
        pkg.dims = vec![("cap".into(), 0), ("upper".into(), 1)];
    }
    let report = pkg.write(out, opts)?;
    Ok(ExportReport {
        symbolic_failure,
        ..report
    })
}

/// The `train` variant: one function `forward`, the whole sequence at once
/// (`input_ids` `[1, seq]` → `logits` `[1, seq, vocab]`, `seq` the first of
/// `opts.seqs`, else 128), its matmul weights dense params, so
/// `rlx_runtime::model::trainer` trains it. The model block's `train` holds
/// the defaults: next-token cross-entropy against `labels`, LoRA (rank 8,
/// alpha 16) on `*attn_q*` / `*attn_v*` — the full-attention layers' q and v
/// and the delta-net layers' fused `attn_qkv`.
fn export_train(
    gguf: &Path,
    raw: &rlx_gguf::GgufFile,
    cfg: &Qwen35Config,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    let path = gguf.to_str().context("GGUF path is not UTF-8")?;
    let seq = opts.seqs_or(&[DEFAULT_TRAIN_SEQ])[0];
    let mut loader = GgufLoader::from_file(path)?;
    loader.include_mtp(true);
    let weights = Qwen35Weights::from_loader(&mut loader, cfg)?;
    let (mut graph, params, packed) =
        crate::build_qwen35_graph_sized(cfg, weights, 1, seq, true, false, false)?;
    if !packed.is_empty() {
        bail!(
            "the train variant needs dense weights, {} are packed",
            packed.len()
        );
    }
    graph.name = "forward".into();
    let first = graph.outputs[0];
    graph.node_mut(first).name = Some("logits".into());
    graph.outputs.truncate(1);

    let sizes: std::collections::HashSet<usize> = params.values().map(Vec::len).collect();
    let mut refr =
        Referencer::open_gguf_only(gguf, opts.embed_weights, &|_, n| sizes.contains(&n))?;
    refr.take(params)?;
    let ck = Checkpoint::open(gguf)?;
    // Dense f32 weights unless asked otherwise: the trainer differentiates
    // through them.
    let opts = ExportOptions {
        weight_dtype: Some(opts.weight_dtype.unwrap_or(rlx_ir::DType::F32)),
        ..opts.clone()
    };
    let mut pkg = Package::for_checkpoint(&ck, &opts, "qwen35", "text-generation", "rlx-qwen35");
    pkg.graphs = vec![graph];
    pkg.reference(refr, out)?;
    let tokenizer = rlx_core::tokenizer::build_gguf_tokenizer(gguf)?;
    let tokenizer_json = tokenizer
        .to_string(false)
        .map_err(|e| anyhow::anyhow!("serialize tokenizer: {e}"))?;
    let tokenizer_id = pkg.sidecar(
        "tokenizer.json",
        "application/json",
        tokenizer_json.into_bytes(),
    );
    pkg.metadata("seq", json!(seq));
    pkg.metadata("tokenizer", json!(tokenizer_id));
    pkg.metadata("source", json!(meta_str(raw, "general.basename")));
    // Next-token prediction: `labels[i]` is the token after `input_ids[i]`.
    pkg.model["train"] = json!({
        "function": "forward",
        "loss": {"cross_entropy": {"logits": "logits", "labels": "labels"}},
        "train": {"lora": {"targets": ["*attn_q*", "*attn_v*"], "rank": 8, "alpha": 16}},
    });
    pkg.write(out, &opts)
}

/// The `distill` variant: one function `distill` that trains the MTP head
/// on the trunk it drafts for, the trunk frozen. Over a sequence
/// (`input_ids` `[1, seq]`), the head's row i (the hidden at i, token i+1)
/// predicts token i+2, which the trunk's row i+1 predicts too; the output
/// `kl` `[1, seq - 1]` is, per row, KL(trunk ‖ head) of the two
/// distributions. A draft is kept when it is the trunk's argmax, so the
/// trunk's whole distribution is a better target than the text's next
/// token. The model block's `train` trains the MTP layer's weights only (not
/// the embedding and LM head it shares with the trunk).
fn export_distill(
    gguf: &Path,
    raw: &rlx_gguf::GgufFile,
    cfg: &Qwen35Config,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    use rlx_ir::op::{CmpOp, ReduceOp};
    use rlx_ir::{DType, GraphExt, Shape};
    if cfg.nextn_predict_layers == 0 {
        bail!("the GGUF has no MTP layers (nextn_predict_layers = 0)");
    }
    let path = gguf.to_str().context("GGUF path is not UTF-8")?;
    let seq = opts.seqs_or(&[DEFAULT_TRAIN_SEQ])[0];
    let mut loader = GgufLoader::from_file(path)?;
    loader.include_mtp(true);
    let weights = Qwen35Weights::from_loader(&mut loader, cfg)?;
    let (built, packed) = crate::flow::Qwen35Flow::prefill(cfg, &weights, 1, seq)
        .lm_head()
        .mtp_head()
        .mtp_every_row()
        .build_with_weights(&weights)?;
    if !packed.is_empty() {
        bail!(
            "the distill variant needs dense weights, {} are packed",
            packed.len()
        );
    }
    let (mut g, params) = rlx_core::flow_util::graph_from_built(built)?;
    let [logits, draft] = g.outputs[..] else {
        bail!("the distill graph has {} outputs, not 2", g.outputs.len());
    };
    let vocab = g
        .node(logits)
        .shape
        .dims()
        .last()
        .map_or(0, |d| d.unwrap_static());
    let rows = seq - 1;
    // log softmax over the vocab, stable: x - max - log Σ exp(x - max), the
    // max a constant to the gradient.
    let log_softmax = |g: &mut rlx_ir::Graph, x| {
        let max = g.add_node(
            Op::Reduce {
                op: ReduceOp::Max,
                axes: vec![2],
                keep_dim: true,
            },
            vec![x],
            Shape::new(&[1, rows, 1], DType::F32),
        );
        let max = g.stop_gradient(max);
        let shifted = g.sub(x, max);
        let e = g.exp(shifted);
        let sum = g.sum(e, vec![2], true);
        let lse = g.log(sum);
        g.sub(shifted, lse)
    };
    let trunk_rows = g.narrow_(logits, 1, 0, rows);
    let trunk = g.narrow_(logits, 1, 1, rows);
    let trunk = g.stop_gradient(trunk);
    let head = g.narrow_(draft, 1, 0, rows);
    let log_q = log_softmax(&mut g, trunk);
    let log_p = log_softmax(&mut g, head);
    let q = g.exp(log_q);
    let gap = g.sub(log_q, log_p);
    let per = g.mul(q, gap);
    let kl = g.sum(per, vec![2], false);
    // Only the rows where the text's next token is the trunk's own pick: a
    // draft is made after a greedy token, never after one the trunk would
    // not have chosen, so those rows are the ones decoding meets.
    let ids = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "input_ids"))
        .map(|n| n.id)
        .context("the distill graph has no input_ids")?;
    let next = g.narrow_(ids, 1, 1, rows);
    let next = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![next],
        Shape::new(&[1, rows], DType::F32),
    );
    let pick = g.add_node(
        Op::ArgMax {
            axis: 2,
            keep_dim: false,
        },
        vec![trunk_rows],
        Shape::new(&[1, rows], DType::F32),
    );
    let same = g.add_node(
        Op::Compare(CmpOp::Eq),
        vec![pick, next],
        Shape::new(&[1, rows], DType::Bool),
    );
    let greedy = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![same],
        Shape::new(&[1, rows], DType::F32),
    );
    let greedy = g.stop_gradient(greedy);
    let kl = g.mul(kl, greedy);
    g.node_mut(kl).name = Some("kl".into());
    g.outputs = vec![kl];
    g.name = "distill".into();
    if vocab == 0 {
        bail!("the distill graph's logits have no vocab axis");
    }

    // The MTP layer's own weights: not the embedding copy, the LM head it
    // shares, or the zero norm betas the builder makes.
    let prefix = format!("blk.{}.", cfg.num_hidden_layers - cfg.nextn_predict_layers);
    let mut trained: Vec<String> = params
        .keys()
        .filter(|n| n.starts_with(&prefix) && n.ends_with(".weight"))
        .filter(|n| !n.contains("embed_tokens") && !n.contains("shared_head_head"))
        .cloned()
        .collect();
    trained.sort();
    if trained.is_empty() {
        bail!("no MTP weights under {prefix}");
    }

    let sizes: std::collections::HashSet<usize> = params.values().map(Vec::len).collect();
    let mut refr =
        Referencer::open_gguf_only(gguf, opts.embed_weights, &|_, n| sizes.contains(&n))?;
    refr.take(params)?;
    let ck = Checkpoint::open(gguf)?;
    let opts = ExportOptions {
        weight_dtype: Some(opts.weight_dtype.unwrap_or(DType::F32)),
        ..opts.clone()
    };
    let mut pkg = Package::for_checkpoint(&ck, &opts, "qwen35", "text-generation", "rlx-qwen35");
    pkg.graphs = vec![g];
    pkg.reference(refr, out)?;
    let tokenizer = rlx_core::tokenizer::build_gguf_tokenizer(gguf)?;
    let tokenizer_json = tokenizer
        .to_string(false)
        .map_err(|e| anyhow::anyhow!("serialize tokenizer: {e}"))?;
    let tokenizer_id = pkg.sidecar(
        "tokenizer.json",
        "application/json",
        tokenizer_json.into_bytes(),
    );
    pkg.metadata("seq", json!(seq));
    pkg.metadata("tokenizer", json!(tokenizer_id));
    pkg.metadata("source", json!(meta_str(raw, "general.basename")));
    pkg.model["train"] = json!({
        "function": "distill",
        "loss": {"output": "kl"},
        "train": {"params": trained},
    });
    pkg.write(out, &opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `qwen35` runner's name, for Qwen3.5 GGUFs.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("qwen35", "text-generation", export_rlxp).claims(&["qwen35", "qwen3_5"]),
    );
}
