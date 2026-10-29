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

/// Tokens a verify step takes: the last token and one MTP draft.
const VERIFY_TOKENS: usize = 2;

/// Cache capacities when the options list none.
const DEFAULT_UPPERS: [usize; 4] = [64, 256, 1024, 4096];

fn decode_graph(
    cfg: &Qwen35Config,
    gguf: &str,
    upper: usize,
    mtp: bool,
    // `Some(n)`: the `n`-token verify step (MTP only).
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
        // The token and its draft at once: logits for both rows, the trunk's
        // causal mask derived from the per-key one.
        Some(tokens) => crate::flow::Qwen35DecodeOpts {
            seq: tokens,
            verify_all: true,
            keys_mask: true,
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
        Some(other) => bail!("qwen35 has no variant '{other}' (only 'mtp')"),
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
    if mtp && symbolic_failure.is_none() && opts.symbolic {
        let at = |upper| decode_graph(&cfg, path, upper, mtp, Some(VERIFY_TOKENS));
        let (a, pa, ka) = at(36)?;
        let (b, c) = (at(52)?.0, at(44)?.0);
        // Its own params (the mask's constants) besides the step's.
        for (k, v) in pa {
            params.entry(k).or_insert(v);
        }
        for (k, v) in ka {
            packed.entry(k).or_insert(v);
        }
        // Its keys run up to `upper + 3`, past `generalize`'s default offset,
        // so fitted through three sizes (in `upper` alone: `cap` moves with
        // it) and checked against the third.
        let mut g = rlx_ir::dynamic::generalize_checked(&a, &b, &c, &[(1, 36, 52, 44)], Some(4))
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
    let tokenizer = crate::tokenizer::build_gguf_tokenizer(gguf)?;
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
        // A speculative step: `tokens` new tokens (the last one, then the
        // draft), every row's logits, and the next draft; same states.
        entry["verify"] = json!({
            "function": f,
            "tokens": VERIFY_TOKENS,
            "logits": 0,
            "draft_logits": 1,
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

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `qwen35` runner's name, for Qwen3.5 GGUFs.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("qwen35", "text-generation", export_rlxp).claims(&["qwen35", "qwen3_5"]),
    );
}
