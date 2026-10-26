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

//! A decoder-only language model as a package: [`Decoder`] says how to
//! build its prompt graph and its decode step from the checkpoint at a
//! size, and [`export_decoder`] does the rest — one step symbolic in its
//! cache (and batch) with its state inside, a prompt function, batched
//! prompt functions, a speculative check, the RoPE tables, the tokenizer,
//! the chat template and sampling defaults, the card, and the params
//! referenced from the checkpoint rather than copied. Qwen3 and the Llama
//! family (`rlx-qwen3`), Gemma, Qwen2.5-VL and Qwen3-VL are decoders.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail};
use rlx_ir::Op;
use serde_json::{Value, json};

use super::{Checkpoint, ExportOptions, ExportReport, Package};
use crate::weight_loader::WeightLoader;

/// The cache capacities a decoder is built at when the options list none.
pub const DEFAULT_UPPERS: [usize; 4] = [64, 256, 1024, 4096];

/// The prompt graph's length symbol (the step's cache is 0).
const SEQ: u32 = 1;
/// The check function's cache length (its rows, less the tokens it adds).
const CHECK_PAST: u32 = 3;
/// The check's keys: its cache rows, `check_past + tokens`.
const CHECK_KEYS: u32 = 4;

/// The prompt function: the prefill graph symbolic in its length (`seq`),
/// its last-row `Narrow` a `Gather` at the input `last`, so a prompt padded
/// past its end still reads its own last row. Checked at a third length:
/// two sizes fit any line.
fn prefill_function(
    build: impl Fn(usize) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)>,
    keep: &[String],
    take: &mut dyn FnMut(HashMap<String, Vec<f32>>) -> Result<()>,
) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>), String> {
    prompt_function(build, SEQ, keep, take)
}

/// The prompt graph `build` makes at a length, symbolic in it (symbol
/// `sym`), its last-row `Narrow` a `Gather` at the input `last`: what a
/// package's prompt functions are (a media prefill's too). Its params go to
/// `take` as soon as the first graph is built (before the next two), but
/// for those named in `keep`, which it returns for the caller to change.
pub fn prompt_function(
    build: impl Fn(usize) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)>,
    sym: u32,
    keep: &[String],
    take: &mut dyn FnMut(HashMap<String, Vec<f32>>) -> Result<()>,
) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>), String> {
    let at = |seq| build(seq).map_err(|e| format!("building the prompt graph at {seq}: {e}"));
    let (a, mut params) = at(37)?;
    let kept: HashMap<String, Vec<f32>> = keep
        .iter()
        .filter_map(|k| params.remove(k).map(|v| (k.clone(), v)))
        .collect();
    take(params).map_err(|e| e.to_string())?;
    let params = kept;
    let (b, _) = at(53)?;
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(sym, 37, 53)])?;
    let (c, _) = at(20)?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(sym, 20);
    if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{c}") {
        return Err(
            "the prompt graph generalized from 37 and 53 tokens is not the one built at 20".into(),
        );
    }
    let (mut last, mut reads) = (None, 0);
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, inputs| {
        let symbolic_start = g
            .sym_attrs()
            .iter()
            .any(|a| a.node == node.id && a.path == ["start"]);
        match node.op {
            Op::Narrow {
                axis: 1, len: 1, ..
            } if symbolic_start => {
                let at = *last.get_or_insert_with(|| {
                    out.input("last", rlx_ir::Shape::new(&[1], rlx_ir::DType::I32))
                });
                reads += 1;
                Ok(Some(out.add_node(
                    Op::Gather { axis: 1 },
                    vec![inputs[0], at],
                    node.shape.clone(),
                )))
            }
            _ => Ok(None),
        }
    })?;
    if reads != 1 {
        return Err(format!(
            "the prompt graph reads its last row {reads} times, not once"
        ));
    }
    g.name = "prefill".into();
    Ok((g, params))
}

/// The prompt graph's RoPE tables span every position the model knows
/// (40960 rows for Qwen3, 21 MB): cut them to the `rows` the package runs,
/// and where they are the step's tables, read those instead.
fn share_rope_tables(
    model: &dyn Decoder,
    g: &mut rlx_ir::Graph,
    params: &mut HashMap<String, Vec<f32>>,
    rows: usize,
) {
    let tables = model.rope_inputs();
    let step_rows: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
        (0..rows).map(|pos| model.rope_row(pos)).collect();
    for (t, (cos_param, sin_param)) in model.prompt_rope_params().into_iter().enumerate() {
        let Some((cos_input, sin_input, half)) = tables.get(t).cloned() else {
            continue;
        };
        for (name, table, cos) in [
            (cos_param, format!("{cos_input}.table"), true),
            (sin_param, format!("{sin_input}.table"), false),
        ] {
            let Some(id) = g
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, Op::Param { name: p } if *p == name))
                .map(|n| n.id)
            else {
                continue;
            };
            let Some(data) = params.remove(&name) else {
                continue;
            };
            if data.len() < rows * half {
                params.insert(name, data);
                continue;
            }
            let head = data[..rows * half].to_vec();
            let same = step_rows
                .iter()
                .flat_map(|r| if cos { r[t].0.iter() } else { r[t].1.iter() })
                .zip(&head)
                .all(|(a, b)| a == b);
            let node = g.node_mut(id);
            node.shape = rlx_ir::Shape::new(&[rows, half], node.shape.dtype());
            if same {
                node.op = Op::Param {
                    name: table.clone(),
                };
                node.name = Some(table);
            } else {
                params.insert(name, head);
            }
        }
    }
}

/// A graph and its params.
pub type Built = (rlx_ir::Graph, HashMap<String, Vec<f32>>);

/// A decoder-only family, as the exporter needs it: its graphs (built from
/// the checkpoint's weights), its sizes and its RoPE. [`export_decoder`]
/// does the rest; Qwen3 (and the Llama family) is one, `rlx-gemma`'s
/// export another.
pub trait Decoder {
    /// The model block's `family`.
    fn family(&self) -> String;
    /// The crate that exported, for the model block's metadata.
    fn exported_by(&self) -> &str;
    /// Positions the model knows.
    fn max_positions(&self) -> usize;
    fn layers(&self) -> usize;
    fn vocab_size(&self) -> usize;
    /// The RoPE tables the step reads a row of: cos input, sin input, and
    /// their width.
    fn rope_inputs(&self) -> Vec<(String, String, usize)>;
    /// Row `pos` of each table, in [`Self::rope_inputs`]' order.
    fn rope_row(&self, pos: usize) -> Vec<(Vec<f32>, Vec<f32>)>;
    /// The decode step: `batch` sequences, a cache of `upper + 1` rows, its
    /// mask an input (`mask`), the new row last.
    fn step(&self, weights: &mut dyn WeightLoader, batch: usize, upper: usize) -> Result<Built>;
    /// The prompt graph for `batch` prompts of `seq` tokens: the last row's
    /// logits, then each layer's K and V.
    fn prompt(&self, weights: &mut dyn WeightLoader, batch: usize, seq: usize) -> Result<Built>;
    /// `tokens` after a cache of `past` rows, for speculative decoding.
    fn check(
        &self,
        _weights: &mut dyn WeightLoader,
        _past: usize,
        _tokens: usize,
    ) -> Result<Built> {
        bail!("{} has no check function", self.family())
    }
    /// The prompt graph's full RoPE tables (cos and sin params), paired
    /// with [`Self::rope_inputs`], which the package cuts to its context.
    fn prompt_rope_params(&self) -> Vec<(String, String)> {
        Vec::new()
    }
    /// How the keys hold positions (the generate entry's `rope`), so a
    /// batch that moves a request's rows rotates them back; `None` when one
    /// table cannot.
    fn key_rope(&self) -> Option<Value> {
        None
    }
    /// Whether the step and check read their tables at `position +
    /// rope_offset` (multimodal RoPE: a prompt's media share positions, so
    /// the text after them is at fewer positions than rows).
    fn rope_offset(&self) -> bool {
        false
    }
    /// The step's further mask inputs and their windows (`(input, window)`):
    /// sliding-window layers', which the stateful step computes as it does
    /// `mask`.
    fn windows(&self) -> Vec<(String, usize)> {
        Vec::new()
    }
    /// The graphs a prompt with media runs (`generate.media`), built from
    /// `weights` (each call a fresh copy of the checkpoint); `rows` RoPE
    /// rows. Each graph's params go to `take` as soon as it is built (the
    /// exporter keeps only those no checkpoint tensor equals). `None`: text
    /// only.
    fn media(
        &self,
        _weights: &dyn Fn() -> Result<Box<dyn WeightLoader>>,
        _rows: usize,
        _take: &mut dyn FnMut(HashMap<String, Vec<f32>>) -> Result<()>,
    ) -> Result<Option<MediaParts>> {
        Ok(None)
    }
    /// `tokenizer.json` built from a GGUF's metadata, for a family whose
    /// GGUFs carry their tokenizer (one beside the GGUF wins).
    fn gguf_tokenizer(&self, gguf: &Path) -> Result<Vec<u8>> {
        bail!(
            "{}: no tokenizer.json beside it, and {} builds none from a GGUF",
            gguf.display(),
            self.family()
        )
    }
}

/// What [`Decoder::media`] adds to a package.
pub struct MediaParts {
    /// Functions, named as `entry` names them.
    pub graphs: Vec<rlx_ir::Graph>,
    /// Their symbols (`(name, id)`; ids past the step's, 10 and up).
    pub dims: Vec<(String, u32)>,
    /// `generate.media`.
    pub entry: Value,
}

/// Export `model`, whose checkpoint is `path` (a Hugging Face directory:
/// `model.safetensors`, `config.json`, `tokenizer.json`,
/// `tokenizer_config.json`, optional `generation_config.json` and
/// `README.md`; or a GGUF), as the package `out`.
pub fn export_decoder(
    model: &dyn Decoder,
    path: &Path,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    export_decoder_with(model, path, out, opts, &mut |_| Ok(()))
}

/// [`export_decoder`] for a model a language model is part of (a TTS whose
/// LM speaks codes): `extend` adds its functions, sidecars and entries —
/// a program that runs the `generate` entry (`entry.run`) — before the
/// package is written. Its symbols are past the decoder's (from 16).
pub fn export_decoder_with(
    model: &dyn Decoder,
    path: &Path,
    out: &Path,
    opts: &ExportOptions,
    extend: &mut dyn FnMut(&mut Package) -> Result<()>,
) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    // Each graph's params, decided as it is built: a reference to the
    // checkpoint (its values dropped), or held.
    let mut refr = ck.referencer(opts.embed_weights)?;
    let max_positions = model.max_positions();
    let mut uppers: Vec<usize> = opts
        .uppers_or(&DEFAULT_UPPERS)
        .into_iter()
        .filter(|&u| u < max_positions)
        .collect();
    uppers.sort();
    let Some(&longest) = uppers.last() else {
        bail!("no cache length fits {max_positions} positions");
    };
    // How far the tables reach: the context the package runs.
    let reach = opts
        .max_context
        .unwrap_or(longest)
        .clamp(longest, max_positions - 1);
    let tables = model.rope_inputs();
    // Each table's rows, as `StatefulDecode` takes them.
    let stateful_tables = |rows: usize| -> Vec<(String, String, usize)> {
        tables
            .iter()
            .flat_map(|(c, s, _)| {
                [
                    (c.clone(), format!("{c}.table"), rows),
                    (s.clone(), format!("{s}.table"), rows),
                ]
            })
            .collect()
    };

    // The checkpoint as the generator loads it, read for each graph: a
    // copy of a 3B model's f32 weights is 15 GB.
    // The checkpoint, each tensor read from its file when a builder takes
    // it: a build holds only the params it makes.
    let fresh = || ck.loader();

    let build_at = |batch: usize, upper: usize| {
        let mut wm = fresh()?;
        model.step(wm.as_mut(), batch, upper)
    };
    let build = |upper: usize| build_at(1, upper);
    // One symbolic decode graph, if it is symbolic in the cache capacity:
    // built at two capacities no other dim has, and generalized. Its cache
    // rows are `cap = upper + 1`; `upper` itself may appear too.
    let mut symbolic_failure = None;
    let mut stateful_failure = None;
    let mut batch_failure = None;
    let mut batch_symbolic = false;
    let mut symbolic = None;
    if opts.symbolic {
        // Batch too, at sizes no other dim has (5 and 7); else batch 1.
        // Each first build's params decided before the second is built.
        let mut batched = None;
        if opts.batch {
            // `KvAppend` at batch > 1: the stateful rewrite makes each a
            // `DynamicUpdateSlice`, which writes every sequence's row.
            let (a, pa) = rlx_flow::blocks::with_kv_append_any_batch(|| build_at(5, 36))?;
            refr.take(pa)?;
            let (b, _) = rlx_flow::blocks::with_kv_append_any_batch(|| build_at(7, 52))?;
            match rlx_ir::dynamic::generalize(&a, &b, &[(0, 37, 53), (1, 36, 52), (2, 5, 7)]) {
                Ok(g) => batched = Some(g),
                Err(e) => batch_failure = Some(e),
            }
        }
        batch_symbolic = batched.is_some();
        let generalized = match batched {
            Some(g) => Ok(g),
            None => {
                let (a, pa) = build(36)?;
                refr.take(pa)?;
                let (b, _) = build(52)?;
                rlx_ir::dynamic::generalize(&a, &b, &[(0, 37, 53), (1, 36, 52)])
            }
        };
        match generalized {
            Ok(mut g) => {
                g.name = "decode".into();
                // Its state inside: RoPE rows and the mask from `position`,
                // each new K/V row written there.
                if opts.stateful {
                    let rows = reach + 1;
                    let spec = rlx_ir::stateful::StatefulDecode {
                        position: "position".into(),
                        tables: stateful_tables(rows),
                        mask: Some(("mask".into(), "decode.iota".into(), rows)),
                        cache_symbol: 0,
                        tokens: 1,
                        // With a batch symbol: each sequence's own start, so
                        // requests join a running batch.
                        starts: batch_symbolic.then(|| "starts".to_string()),
                        rope_offset: model.rope_offset().then(|| "rope_offset".to_string()),
                        windows: model.windows(),
                    };
                    match rlx_ir::stateful::stateful_decode(&g, &spec) {
                        Ok(step) => {
                            g = step;
                            g.name = "step".into();
                        }
                        Err(e) => stateful_failure = Some(e),
                    }
                }
                symbolic = Some(g);
            }
            Err(e) => symbolic_failure = Some(e),
        }
    }
    let stateful = opts.stateful && symbolic.is_some() && stateful_failure.is_none();
    let prompt_sizes: Vec<usize> = opts
        .prefill
        .iter()
        .copied()
        .filter(|&s| s > 1 && s <= longest)
        .collect();
    let mut prefill_failure = None;
    let mut prefill = None;
    // The prompt graphs' RoPE tables, cut and shared before they are
    // decided.
    let rope_keep: Vec<String> = model
        .prompt_rope_params()
        .into_iter()
        .flat_map(|(c, s)| [c, s])
        .collect();
    if stateful && !prompt_sizes.is_empty() {
        let build = |seq: usize| {
            let mut wm = fresh()?;
            model.prompt(wm.as_mut(), 1, seq)
        };
        match prefill_function(build, &rope_keep, &mut |p| refr.take(p)) {
            Ok((mut g, mut p)) => {
                share_rope_tables(model, &mut g, &mut p, reach + 1);
                refr.take(p)?;
                prefill = Some(g);
            }
            Err(e) => prefill_failure = Some(e),
        }
    }
    // The same at each batch size: one graph per batch, symbolic in `seq`.
    let mut prefill_batches = Vec::new();
    if prefill.is_some() {
        for &batch in opts.prefill_batches.iter().filter(|&&b| b > 1) {
            let build = |seq: usize| {
                let mut wm = fresh()?;
                model.prompt(wm.as_mut(), batch, seq)
            };
            match prefill_function(build, &rope_keep, &mut |p| refr.take(p)) {
                Ok((mut g, mut p)) => {
                    g.name = format!("prefill_b{batch}");
                    // Its RoPE tables cut and shared as the prompt
                    // function's are; then its params are that function's
                    // but for any of its own: keep only those.
                    share_rope_tables(model, &mut g, &mut p, reach + 1);
                    refr.take(p)?;
                    prefill_batches.push(g);
                }
                Err(e) => {
                    prefill_failure = Some(format!("at batch {batch}: {e}"));
                    break;
                }
            }
        }
    }
    let mut graphs = match symbolic {
        Some(g) => vec![g],
        None => {
            let mut graphs = Vec::new();
            for &upper in &uppers {
                let (mut graph, p) = build(upper)?;
                graph.name = format!("decode_{upper}");
                refr.take(p)?;
                graphs.push(graph);
            }
            graphs
        }
    };
    if let Some(g) = prefill.take() {
        graphs.push(g);
        graphs.extend(prefill_batches);
    }
    let has_prefill = stateful && graphs.iter().any(|g| g.name == "prefill");
    // The check: `speculate + 1` tokens after a cache, symbolic in the cache
    // (`check_past`), its state inside like the step's.
    let check_tokens = opts.speculate + 1;
    // Each listed cache length, its keys no further than the tables reach.
    let mut check_sizes: Vec<usize> = uppers
        .iter()
        .map(|&u| u.min((reach + 1).saturating_sub(check_tokens)))
        .filter(|&u| u > 0)
        .collect();
    check_sizes.dedup();
    let mut check_failure = None;
    if stateful && opts.speculate > 0 && !check_sizes.is_empty() {
        let check_at = |past: usize| {
            let mut wm = fresh()?;
            model.check(wm.as_mut(), past, check_tokens)
        };
        let built = check_at(36).and_then(|(a, pa)| {
            refr.take(pa)?;
            Ok((a, check_at(52)?.0))
        });
        let (a, b) = match built {
            Ok(v) => v,
            Err(e) => {
                check_failure = Some(e.to_string());
                (rlx_ir::Graph::new("none"), rlx_ir::Graph::new("none"))
            }
        };
        let rows = reach + 1;
        // The keys (`past + tokens`) a symbol of their own: an offset larger
        // than its scale is not one `generalize` derives.
        let sizes = [
            (CHECK_PAST, 36, 52),
            (CHECK_KEYS, 36 + check_tokens, 52 + check_tokens),
        ];
        let made = if check_failure.is_some() {
            Err(check_failure.take().unwrap_or_default())
        } else {
            rlx_ir::dynamic::generalize(&a, &b, &sizes)
        }
        .and_then(|g| {
            rlx_ir::stateful::stateful_decode(
                &g,
                &rlx_ir::stateful::StatefulDecode {
                    position: "position".into(),
                    tables: stateful_tables(rows),
                    mask: Some(("mask".into(), "decode.iota".into(), rows)),
                    cache_symbol: CHECK_PAST,
                    tokens: check_tokens,
                    starts: None,
                    rope_offset: model.rope_offset().then(|| "rope_offset".to_string()),
                    windows: model.windows(),
                },
            )
        });
        match made {
            Ok(mut g) => {
                g.name = "check".into();
                graphs.push(g);
            }
            Err(e) => check_failure = Some(e),
        }
    }
    let has_check = graphs.iter().any(|g| g.name == "check");
    // A prompt's media: further graphs, their params with the rest.
    let media = model.media(&fresh, reach + 1, &mut |p| refr.take(p))?;
    let mut media_entry = Value::Null;
    let mut media_dims = Vec::new();
    if let Some(m) = media {
        graphs.extend(m.graphs);
        media_entry = m.entry;
        media_dims = m.dims;
    }
    let mut pkg = Package::for_checkpoint(
        &ck,
        opts,
        &model.family(),
        "text-generation",
        model.exported_by(),
    );
    pkg.graphs = graphs;
    // What the checkpoint's files hold, and the params they do not.
    pkg.reference(refr, out)?;

    // RoPE rows for every position the package reaches.
    let rows = reach + 1;
    let mut cos = vec![Vec::new(); tables.len()];
    let mut sin = vec![Vec::new(); tables.len()];
    for pos in 0..rows {
        for (t, (c, s)) in model.rope_row(pos).into_iter().enumerate() {
            cos[t].extend(c);
            sin[t].extend(s);
        }
    }
    for (t, (c, s, half)) in tables.iter().enumerate() {
        // A stateful step reads the tables by name; a host-protocol one is
        // fed their rows (`{table}`).
        let suffix = if stateful { ".table" } else { "" };
        pkg.hold_f32(format!("{c}{suffix}"), vec![rows, *half], &cos[t]);
        pkg.hold_f32(format!("{s}{suffix}"), vec![rows, *half], &sin[t]);
    }
    if stateful {
        let iota: Vec<f32> = (0..rows).map(|i| i as f32).collect();
        pkg.hold_f32("decode.iota", vec![rows], &iota);
    }

    // Inputs and the caches, by the decode graph's own names.
    let mut inputs = serde_json::Map::new();
    let mut state = Vec::new();
    for n in pkg.graphs[0].nodes() {
        let Op::Input { name } = &n.op else { continue };
        let feed = match name.as_str() {
            "input_ids" => json!("token"),
            "position" => json!("position"),
            "starts" => json!("starts"),
            "rope_offset" => json!("rope_offset"),
            "mask" => json!("decode_mask"),
            n if tables.iter().any(|(c, s, _)| n == c || n == s) => json!({"table": name}),
            other => {
                let layer = |p: &str| other.strip_prefix(p).and_then(|i| i.parse::<usize>().ok());
                if let Some(i) = layer("past_k_") {
                    state.push(json!([other, 1 + 2 * i]));
                } else if let Some(i) = layer("past_v_") {
                    state.push(json!([other, 2 + 2 * i]));
                } else {
                    bail!("decode input '{other}' has no feed");
                }
                continue;
            }
        };
        inputs.insert(name.clone(), feed);
    }

    let stop = ck.stop_ids();
    let generation = ck.generation();
    let tokenizer = pkg.tokenizer(&ck, Some(&|p: &Path| model.gguf_tokenizer(p)))?;
    let chat = pkg.chat(&ck);
    let graphs = &pkg.graphs;
    let mut entry = json!({
            "functions": graphs
                .iter()
                .filter(|g| g.name == "step" || g.name.starts_with("decode"))
                .map(|g| g.name.clone())
                .collect::<Vec<_>>(),
            // A stateful step's caches have `cap` rows; the host-protocol
            // graph's also `upper = cap - 1`, where its new row goes.
            "sizes": uppers
                .iter()
                .map(|u| {
                    let mut size = if stateful { json!({"cap": u + 1}) } else { json!({"cap": u + 1, "upper": u}) };
                    if batch_symbolic {
                        size["batch"] = json!(1);
                    }
                    size
                })
                .collect::<Vec<_>>(),
            // `speculate + 1` tokens at once: what a speculative decoder
            // checks its draft with.
            "check": if has_check {
                json!({
                    "function": "check",
                    "sizes": check_sizes
                        .iter()
                        .map(|u| json!({"check_past": u, "check_keys": u + check_tokens}))
                        .collect::<Vec<_>>(),
                    "tokens": check_tokens,
                    "inputs": if model.rope_offset() {
                        json!({"input_ids": "tokens", "position": "position", "rope_offset": "rope_offset"})
                    } else {
                        json!({"input_ids": "tokens", "position": "position"})
                    },
                    "logits": 0,
                })
            } else {
                Value::Null
            },
            // The step's batch symbol: `generate_batch` sizes it.
            "batch": if batch_symbolic { json!("batch") } else { Value::Null },
            "tokenizer": tokenizer,
            "chat": chat,
            "stop_ids": stop,
            "vocab_size": model.vocab_size(),
            "inputs": inputs,
            "state": state,
            "state_update": if stateful { "replace" } else { "append_last_row" },
            "logits": 0,
            "generation": generation,
            // How the keys hold positions: a batch moving a request's rows
            // rotates them back.
            "rope": if stateful { model.key_rope().unwrap_or(Value::Null) } else { Value::Null },
            // A symbolic step grows past `sizes` to the tables' reach.
            "max_context": reach,
    });
    // A stateful step's caches are its variables (`%k = var`): the entry
    // need not pair them.
    if stateful && let Some(e) = entry.as_object_mut() {
        e.remove("state");
    }
    // The prompt in one run: its logits, then each layer's K and V rows,
    // the step's caches in order.
    if has_prefill {
        entry["prefill"] = json!({
            "function": "prefill",
            "sizes": prompt_sizes.iter().map(|s| json!({"seq": s})).collect::<Vec<_>>(),
            "inputs": {"input_ids": "tokens", "last": "last"},
            "logits": 0,
            "state": (1..=2 * model.layers()).collect::<Vec<_>>(),
            // The same at fixed batch sizes, for several prompts at once.
            "batches": graphs
                .iter()
                .filter_map(|g| {
                    let b: usize = g.name.strip_prefix("prefill_b")?.parse().ok()?;
                    Some(json!({"function": g.name, "batch": b}))
                })
                .collect::<Vec<_>>(),
        });
    }
    if !media_entry.is_null() {
        entry["media"] = media_entry;
    }
    pkg.entry("generate", entry);
    pkg.metadata("context_length", json!(reach));
    // The step's first output is its logits: say so (once the builder's
    // shared names are gone).
    for g in &mut pkg.graphs {
        super::tidy(g);
        let first = g.outputs[0];
        if g.node(first).name.is_none() {
            g.node_mut(first).name = Some("logits".into());
        }
    }
    if symbolic_failure.is_none() && opts.symbolic {
        pkg.dims = if has_prefill {
            vec![("cap".into(), 0), ("seq".into(), SEQ)]
        } else if stateful {
            vec![("cap".into(), 0)]
        } else {
            vec![("cap".into(), 0), ("upper".into(), 1)]
        };
        if batch_symbolic {
            pkg.dims.push(("batch".into(), 2));
        }
        if has_check {
            pkg.dims.push(("check_past".into(), CHECK_PAST));
            pkg.dims.push(("check_keys".into(), CHECK_KEYS));
        }
        pkg.dims.extend(media_dims);
    }
    extend(&mut pkg)?;
    let report = pkg.write(out, opts)?;
    Ok(ExportReport {
        symbolic_failure,
        stateful_failure,
        prefill_failure,
        batch_failure,
        check_failure,
        ..report
    })
}

/// The prompt graph's length symbol in a [`Forward`] package.
const FORWARD_SEQ: u32 = 1;

/// A language model as one full-sequence graph — tokens in, each
/// position's logits out — for a family whose runner keeps no cache and
/// runs the whole context for each token. [`export_forward`] writes it as a
/// `generate` entry with only a prompt function, which rlx runs the same
/// way: over the whole context, each token.
pub trait Forward {
    /// The model block's `family`.
    fn family(&self) -> String;
    /// The crate that exported, for the model block's metadata.
    fn exported_by(&self) -> &str;
    /// The longest context the graph runs correctly.
    fn max_positions(&self) -> usize;
    fn vocab_size(&self) -> usize;
    /// The graph for one sequence of `seq` tokens: its first output the
    /// logits, `seq × vocab` values (the last row is read).
    fn forward(&self, weights: &mut dyn WeightLoader, seq: usize) -> Result<Built>;
    /// The token input's name.
    fn tokens_input(&self) -> &str {
        "input_ids"
    }
    /// Inputs (or params) that take one row per position (RoPE tables): each
    /// one's name and rows `0..rows`, `width` wide. The package holds them,
    /// and the graph reads its first `seq` rows.
    fn tables(&self, _rows: usize) -> Vec<(String, Vec<f32>, usize)> {
        Vec::new()
    }
}

/// The forward graph at `seq`: its tables read from held params (the first
/// `seq` rows), its logits the row at the input `last`.
fn forward_at(
    model: &dyn Forward,
    weights: &mut dyn WeightLoader,
    seq: usize,
    tables: &[(String, usize)],
) -> Result<Built> {
    let (g, mut params) = model.forward(weights, seq)?;
    for (name, _) in tables {
        params.remove(name);
    }
    let vocab = model.vocab_size();
    let f32s = |d: &[usize]| rlx_ir::Shape::new(d, rlx_ir::DType::F32);
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| {
        let (Op::Input { name } | Op::Param { name }) = &node.op else {
            return Ok(None);
        };
        let Some((_, width)) = tables.iter().find(|(t, _)| t == name) else {
            return Ok(None);
        };
        let held = out.param(format!("{name}.table"), f32s(&[0, *width]));
        let rows = out.add_node(
            Op::Narrow {
                axis: 0,
                start: 0,
                len: seq,
            },
            vec![held],
            node.shape.clone(),
        );
        Ok(Some(rows))
    })
    .map_err(anyhow::Error::msg)?;
    // The last row's logits, wherever the prompt ends in a padded run.
    let logits = g.outputs[0];
    let n = g.node(logits).shape.num_elements().unwrap_or(0);
    if n != seq * vocab {
        bail!("the forward graph's first output has {n} values, not {seq} × {vocab}");
    }
    let rows = g.reshape(logits, vec![seq as i64, vocab as i64], f32s(&[seq, vocab]));
    let last = g.input("last", rlx_ir::Shape::new(&[1], rlx_ir::DType::I32));
    let row = g.add_node(Op::Gather { axis: 0 }, vec![rows, last], f32s(&[1, vocab]));
    g.node_mut(row).name = Some("logits".into());
    g.outputs = vec![row];
    g.name = "forward".into();
    Ok((g, params))
}

/// Export `model`, whose checkpoint is `path`, as the package `out`: one
/// prompt function symbolic in its length, run over the whole context for
/// each token (`entries.generate` with a `prefill` and no step).
pub fn export_forward(
    model: &dyn Forward,
    path: &Path,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    let max_positions = model.max_positions();
    let mut sizes: Vec<usize> = opts
        .uppers_or(&DEFAULT_UPPERS)
        .into_iter()
        .filter(|&u| u <= max_positions)
        .collect();
    sizes.sort();
    let Some(&longest) = sizes.last() else {
        bail!("no length fits {max_positions} positions");
    };
    let tables = model.tables(longest);
    let widths: Vec<(String, usize)> = tables.iter().map(|(n, _, w)| (n.clone(), *w)).collect();
    let mut refr = ck.referencer(opts.embed_weights)?;
    let build = |seq: usize| {
        let mut w = ck.loader()?;
        forward_at(model, w.as_mut(), seq, &widths)
    };
    let (a, pa) = build(37)?;
    let (b, pb) = build(53)?;
    // One graph symbolic in its length, checked at a third (two sizes fit
    // any line), else one per size.
    let symbolic = rlx_ir::dynamic::generalize(&a, &b, &[(FORWARD_SEQ, 37, 53)]).and_then(|g| {
        let (c, _) = build(20).map_err(|e| e.to_string())?;
        let mut bind = rlx_ir::DimBinding::new();
        bind.set(FORWARD_SEQ, 20);
        if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{c}") {
            return Err(
                "the graph generalized from 37 and 53 tokens is not the one built at 20".into(),
            );
        }
        Ok(g)
    });
    let mut graphs = Vec::new();
    let mut symbolic_failure = None;
    match symbolic {
        Ok(g) => {
            refr.take(pa)?;
            graphs.push(g);
        }
        Err(e) => {
            symbolic_failure = Some(e);
            // A param whose values depend on the length is one per size
            // (`name@seq`); the rest are shared.
            drop((pa, pb));
            let mut builds = Vec::new();
            for &seq in &sizes {
                let (mut g, params) = build(seq)?;
                g.name = format!("forward_{seq}");
                builds.push((seq, g, params));
            }
            let (per, params) = super::graph::per_size(builds);
            refr.take(params)?;
            graphs = per;
        }
    }
    drop((a, b));
    // The tables' declared rows: the held tensors' (each graph reads the
    // first `seq` of them).
    for g in &mut graphs {
        for (name, _, width) in &tables {
            let table = format!("{name}.table");
            for node in g.nodes_mut() {
                if matches!(&node.op, Op::Param { name: p } if *p == table) {
                    node.shape = rlx_ir::Shape::new(&[longest, *width], node.shape.dtype());
                }
            }
        }
    }

    let mut pkg = Package::for_checkpoint(
        &ck,
        opts,
        &model.family(),
        "text-generation",
        model.exported_by(),
    );
    let prefill = if symbolic_failure.is_none() {
        pkg.dims = vec![("seq".into(), FORWARD_SEQ)];
        json!({
            "function": "forward",
            "sizes": sizes.iter().map(|s| json!({"seq": s})).collect::<Vec<_>>(),
        })
    } else {
        json!({"functions": graphs.iter().map(|g| g.name.clone()).collect::<Vec<_>>()})
    };
    pkg.graphs = graphs;
    pkg.reference(refr, out)?;
    for (name, rows, width) in tables {
        pkg.hold_f32(format!("{name}.table"), vec![longest, width], &rows);
    }
    let mut prefill = prefill;
    prefill["inputs"] = json!({model.tokens_input(): "tokens", "last": "last"});
    prefill["logits"] = json!(0);
    prefill["state"] = json!([]);
    let tokenizer = pkg.tokenizer(&ck, None)?;
    let chat = pkg.chat(&ck);
    pkg.entry(
        "generate",
        json!({
            "tokenizer": tokenizer,
            "chat": chat,
            "stop_ids": ck.stop_ids(),
            "vocab_size": model.vocab_size(),
            "inputs": {},
            // No step: the prompt function over the whole context, each token.
            "prefill": prefill,
            "generation": ck.generation(),
            "max_context": longest,
        }),
    );
    pkg.metadata("context_length", json!(longest));
    let report = pkg.write(out, opts)?;
    Ok(ExportReport {
        symbolic_failure,
        ..report
    })
}
