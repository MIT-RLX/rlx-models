// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Full-weight fine-tuning and quantization-aware training (QAT) of a language
//! model, model-agnostic.
//!
//! A model family supplies a [`TrainableLm`]: its full-sequence forward graph
//! (token ids → logits, causal, no KV cache), the values of every graph param,
//! and the name of the token-id input. This module appends a next-token
//! cross-entropy loss, trains with [`crate::Trainer`], measures held-out loss,
//! and writes the result back under the checkpoint's own tensor names.
//!
//! With a number format the model part runs in it — activations rounded
//! in-graph, weights rounded on each step's upload — while the loss, the
//! gradients and the f32 master weights stay full precision (the
//! straight-through estimator). The logits are a graph output of the model part,
//! so they stay f32 too, as they would be when serving.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, NumberFormat, Session};

use crate::{DpConfig, ParamSlot, Trainer};

/// A language model's training forward.
pub struct TrainableLm {
    /// Token ids `[batch, seq]` → logits `[batch, seq, vocab]` (or
    /// `[batch·seq, vocab]`), as `outputs[0]`.
    pub graph: Graph,
    /// Every graph param's value. Params that are not checkpoint tensors (RoPE
    /// tables, zero biases) are fed but never trained or exported.
    pub params: HashMap<String, Vec<f32>>,
    /// Name of the token-id input.
    pub token_input: String,
    pub batch: usize,
    pub seq: usize,
    /// Other graph inputs, fed unchanged with every batch (RoPE tables a
    /// family takes as inputs rather than params).
    pub extra_inputs: Vec<(String, Vec<f32>)>,
}

#[derive(Debug, Clone)]
pub struct LmTrainOptions {
    /// Train the model part in this format (QAT); `None` fine-tunes in f32.
    pub format: Option<NumberFormat>,
    pub steps: usize,
    pub lr: f32,
    pub device: Device,
    pub seed: u64,
}

impl Default for LmTrainOptions {
    fn default() -> Self {
        Self {
            format: None,
            steps: 100,
            lr: 1e-5,
            device: Device::Cpu,
            seed: 0,
        }
    }
}

pub struct TrainedLm {
    /// Trained values of the checkpoint params, by graph param name.
    pub params: HashMap<String, Vec<f32>>,
    pub losses: Vec<f32>,
}

/// Next-token loss graph over `lm` (model part rewritten for `format` first).
/// Extra inputs: `labels` and `label_mask`, both `[batch·seq]`.
fn loss_graph(lm: &TrainableLm, format: Option<NumberFormat>) -> Result<Graph> {
    let mut g = lm.graph.clone();
    let (batch, seq) = (lm.batch, lm.seq);
    if let Some(NumberFormat::Float { format, scaling }) = format {
        g = rlx_runtime::emulate_format::emulate_float_format(
            g,
            format,
            scaling,
            &Default::default(),
        );
    }
    let logits = *g.outputs.first().context("model graph has no output")?;
    let shape = g.shape(logits).clone();
    let vocab = shape
        .dims()
        .last()
        .context("logits have no shape")?
        .unwrap_static();
    let rows = batch * seq;
    ensure!(
        shape.num_elements() == Some(rows * vocab),
        "logits {shape:?} are not [batch·seq = {rows}, vocab]"
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![rows as i64, vocab as i64],
        },
        vec![logits],
        Shape::new(&[rows, vocab], DType::F32),
    );
    let labels = g.input("labels", Shape::new(&[rows], DType::F32));
    let mask = g.input("label_mask", Shape::new(&[rows], DType::F32));
    let loss = crate::loss::cross_entropy_masked(&mut g, flat, labels, mask);
    g.set_outputs(vec![loss]);
    Ok(g)
}

/// Replace every alias param in `g` with its primary (transposed if needed),
/// so a tied weight is one trainable value.
fn tie_aliases(g: Graph, map: &CheckpointMap) -> Result<Graph> {
    if map.aliases.is_empty() {
        return Ok(g);
    }
    let param_id = |name: &str| {
        g.nodes().iter().find_map(|n| match &n.op {
            Op::Param { name: p } if p == name => Some(n.id),
            _ => None,
        })
    };
    let mut redirect: HashMap<rlx_ir::NodeId, (rlx_ir::NodeId, bool)> = HashMap::new();
    for (alias, (primary, transposed)) in &map.aliases {
        let (Some(a), Some(p)) = (param_id(alias), param_id(primary)) else {
            continue;
        };
        redirect.insert(a, (p, *transposed));
    }
    let mut out = Graph::new(&g.name);
    let mut ids: HashMap<rlx_ir::NodeId, rlx_ir::NodeId> = HashMap::new();
    // Params are leaves: emit them all first so a primary exists before any alias.
    let params: Vec<&rlx_ir::Node> = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Param { .. }) && !redirect.contains_key(&n.id))
        .collect();
    for n in &params {
        ids.insert(n.id, out.add_node(n.op.clone(), vec![], n.shape.clone()));
    }
    for n in g.nodes() {
        if ids.contains_key(&n.id) {
            continue;
        }
        let new = if let Some(&(p, transposed)) = redirect.get(&n.id) {
            if transposed {
                out.add_node(
                    Op::Transpose { perm: vec![1, 0] },
                    vec![ids[&p]],
                    n.shape.clone(),
                )
            } else {
                ids[&p]
            }
        } else {
            let inputs = n.inputs.iter().map(|i| ids[i]).collect();
            out.add_node(n.op.clone(), inputs, n.shape.clone())
        };
        ids.insert(n.id, new);
    }
    out.set_outputs(g.outputs.iter().map(|o| ids[o]).collect());
    Ok(out)
}

/// The params of `g` that `map` ties to checkpoint tensors.
fn trainable_slots(g: &Graph, map: &CheckpointMap) -> Vec<ParamSlot> {
    g.nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } if map.entries.contains_key(name) => Some(ParamSlot {
                name: name.clone(),
                node: n.id,
            }),
            _ => None,
        })
        .collect()
}

/// One batch: random `seq + 1`-token windows from `sequences`; the model reads
/// `seq` tokens and predicts each next one.
fn sample(lm: &TrainableLm, sequences: &[&Vec<u32>], rng: &mut u64) -> Vec<(String, Vec<f32>)> {
    let (b, t) = (lm.batch, lm.seq);
    let mut ids = Vec::with_capacity(b * t);
    let mut labels = Vec::with_capacity(b * t);
    for _ in 0..b {
        let s = sequences[next(rng, sequences.len())];
        let start = next(rng, s.len() - t);
        ids.extend(s[start..start + t].iter().map(|&x| x as f32));
        labels.extend(s[start + 1..start + t + 1].iter().map(|&x| x as f32));
    }
    let mut inputs = vec![
        (lm.token_input.clone(), ids),
        ("labels".into(), labels),
        ("label_mask".into(), vec![1.0; b * t]),
    ];
    inputs.extend(lm.extra_inputs.iter().cloned());
    inputs
}

fn next(state: &mut u64, n: usize) -> usize {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as usize) % n.max(1)
}

/// Train `lm` on token `sequences` (each longer than `seq`). Exactly the params
/// `map` ties to checkpoint tensors are trained (see [`map_checkpoint`]).
pub fn train_lm(
    lm: &TrainableLm,
    map: &CheckpointMap,
    sequences: &[Vec<u32>],
    opts: &LmTrainOptions,
) -> Result<TrainedLm> {
    let usable: Vec<&Vec<u32>> = sequences.iter().filter(|s| s.len() > lm.seq).collect();
    ensure!(
        !usable.is_empty(),
        "no token sequence is longer than {} tokens",
        lm.seq
    );
    if let Some(f @ (NumberFormat::F64 | NumberFormat::F64x2)) = opts.format {
        bail!("training in {f} is not available yet");
    }
    let g = tie_aliases(loss_graph(lm, opts.format)?, map)?;
    let slots = trainable_slots(&g, map);
    ensure!(
        !slots.is_empty(),
        "no graph param maps to a checkpoint tensor"
    );
    let mut dp = DpConfig::new(opts.lr).device(opts.device);
    if let Some(f) = opts.format.filter(|f| *f != NumberFormat::F32) {
        dp.qat = Some(f);
        dp.qat_forward_rewritten = true;
    }
    let mut trainer = Trainer::new(g, &slots, &lm.params, opts.steps, None, &dp)?;
    let mut rng = opts.seed.wrapping_add(0x9e37_79b9);
    let mut batch = |_: usize, _: usize| sample(lm, &usable, &mut rng);
    let mut losses = Vec::with_capacity(opts.steps);
    for _ in 0..opts.steps {
        losses.push(trainer.step(&mut batch)?.loss);
    }
    let params = trainer
        .params()
        .into_iter()
        .filter(|(k, _)| map.entries.contains_key(k))
        .collect();
    Ok(TrainedLm { params, losses })
}

/// Mean next-token loss on `batches` windows of `sequences`, with the model in
/// `format` (as it would be served: weights rounded on upload, activations
/// in-graph, logits f32). `params` overrides `lm.params` (trained values).
pub fn eval_lm_loss(
    lm: &TrainableLm,
    map: &CheckpointMap,
    params: &HashMap<String, Vec<f32>>,
    sequences: &[Vec<u32>],
    format: Option<NumberFormat>,
    batches: usize,
    device: Device,
    seed: u64,
) -> Result<f32> {
    let usable: Vec<&Vec<u32>> = sequences.iter().filter(|s| s.len() > lm.seq).collect();
    ensure!(
        !usable.is_empty(),
        "no held-out sequence is longer than {} tokens",
        lm.seq
    );
    // The format comes from the compile options here, as when serving.
    let loss = tie_aliases(loss_graph(lm, None)?, map)?;
    let mut opts = CompileOptions::new();
    opts.number_format = format;
    let mut cg = Session::new(device).compile_with(loss, &opts);
    for (k, v) in &lm.params {
        cg.set_param(k, params.get(k).unwrap_or(v));
    }
    let mut rng = seed.wrapping_add(0x51ed_270b);
    let mut total = 0f64;
    for _ in 0..batches.max(1) {
        let batch = sample(lm, &usable, &mut rng);
        let inputs: Vec<(&str, &[f32])> = batch
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        total += cg.run(&inputs).remove(0)[0] as f64;
    }
    Ok((total / batches.max(1) as f64) as f32)
}

/// Tensor names and shapes from a safetensors file's header (no data read).
pub fn safetensors_shapes(path: &Path) -> Result<HashMap<String, Vec<usize>>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let json: serde_json::Value = serde_json::from_slice(&header).context("safetensors header")?;
    let mut out = HashMap::new();
    for (name, v) in json
        .as_object()
        .context("safetensors header is not an object")?
    {
        if name == "__metadata__" {
            continue;
        }
        let shape = v["shape"]
            .as_array()
            .with_context(|| format!("{name}: no shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).context("bad dim"))
            .collect::<Result<Vec<_>>>()?;
        out.insert(name.clone(), shape);
    }
    Ok(out)
}

/// How a graph param's value derives from a checkpoint tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Layout {
    Same,
    /// The graph holds the 2-D tensor transposed (`[in, out]` for `[out, in]`).
    Transposed,
    /// The graph holds the tensor plus a constant (Gemma-style `1 + w` norms).
    Offset(f32),
}

/// Which checkpoint tensor each graph param holds, and how.
pub struct CheckpointMap {
    file: std::path::PathBuf,
    /// Graph param → (checkpoint tensor, layout, graph shape).
    entries: HashMap<String, (String, Layout, Vec<usize>)>,
    /// Tied weights: a second graph param holding the same checkpoint tensor
    /// as a mapped one (a tied LM head materialized as the transposed
    /// embedding). Alias → (primary param, transposed?). Training computes the
    /// alias from its primary so both uses update one weight.
    aliases: HashMap<String, (String, bool)>,
    /// Params that hold no checkpoint tensor (fed, not trained), with sizes.
    unmapped: Vec<(String, usize)>,
    mapped_elems: usize,
}

impl CheckpointMap {
    /// Graph params tied to checkpoint tensors (the trainable ones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Fraction of graph-param elements that are trained (aliases count as
    /// trained), and the untrained params of at least 0.1% of the total —
    /// RoPE tables are expected here; anything large means a family stores a
    /// weight in a form the checkpoint does not (e.g. stacked experts).
    pub fn coverage(&self) -> (f64, Vec<(String, usize)>) {
        let unmapped: usize = self.unmapped.iter().map(|u| u.1).sum();
        let total = (self.mapped_elems + unmapped).max(1);
        let notable = self
            .unmapped
            .iter()
            .filter(|u| u.1 * 1000 >= total)
            .cloned()
            .collect();
        (self.mapped_elems as f64 / total as f64, notable)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Derive, for every param of `lm`, the checkpoint tensor its initial value
/// reproduces exactly and how (same, transposed, or offset by a constant).
///
/// Families load weights differently — transposed, `1 + w` norms, or under
/// GGUF-style names through a translating loader — so the mapping is not
/// assumed. A same-named tensor is tried first (any layout); otherwise the
/// unique unclaimed tensor with identical content, as-is or transposed, and
/// never a constant-valued one (that proves nothing). Params that reproduce no
/// tensor (RoPE tables, zero biases) are constants: fed, never trained.
pub fn map_checkpoint(lm: &TrainableLm, checkpoint_file: &Path) -> Result<CheckpointMap> {
    let bytes = std::fs::read(checkpoint_file)
        .with_context(|| format!("read {}", checkpoint_file.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).context("parse checkpoint")?;
    let ckpt = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
        let t = st.tensor(name)?;
        Ok((t.shape().to_vec(), to_f32(t.dtype(), t.data())?))
    };
    let mut by_numel: HashMap<usize, Vec<String>> = HashMap::new();
    for (name, view) in st.tensors() {
        by_numel
            .entry(view.shape().iter().product())
            .or_default()
            .push(name);
    }
    let mut graph_params: Vec<(String, Vec<usize>)> = lm
        .graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some((
                name.clone(),
                n.shape.dims().iter().map(|d| d.unwrap_static()).collect(),
            )),
            _ => None,
        })
        .collect();
    graph_params.sort();
    graph_params.dedup_by(|a, b| a.0 == b.0);
    let names: Vec<&str> = st.names().into_iter().collect();
    // Params named like a checkpoint tensor claim it before any content
    // match, so a tied copy (`lm_head.tied_t`) becomes an alias of the real
    // tensor whatever the two names sort as — matched the other way round,
    // both would claim the embedding, train apart, and the export would keep
    // whichever was written last.
    graph_params.sort_by_key(|(n, _)| !names.contains(&n.as_str()));

    let mut entries: HashMap<String, (String, Layout, Vec<usize>)> = HashMap::new();
    let mut aliases = HashMap::new();
    let mut unmapped = Vec::new();
    let mut mapped_elems = 0usize;
    let mut claimed: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (name, gshape) in graph_params {
        let Some(init) = lm.params.get(&name) else {
            continue;
        };
        let mut found = None;
        if names.contains(&name.as_str()) {
            let (cs, c) = ckpt(&name)?;
            found = derive_layout(init, &gshape, &c, &cs, true).map(|l| (name.clone(), l));
        }
        if found.is_none() && !is_constant(init) {
            let mut hits = Vec::new();
            for cand in by_numel.get(&init.len()).into_iter().flatten() {
                if claimed.contains(cand) {
                    // Same tensor as a mapped param: a tied weight, if exact.
                    let (cs, c) = ckpt(cand)?;
                    if let Some(l @ (Layout::Same | Layout::Transposed)) =
                        derive_layout(init, &gshape, &c, &cs, false)
                    {
                        let primary = entries
                            .iter()
                            .find(|(_, e): &(&String, &(String, Layout, Vec<usize>))| e.0 == *cand)
                            .map(|(p, e)| (p.clone(), e.1));
                        if let Some((p, pl)) = primary {
                            // Alias = primary transposed iff exactly one side is.
                            let transposed =
                                (l == Layout::Transposed) != (pl == Layout::Transposed);
                            aliases.insert(name.clone(), (p, transposed));
                        }
                    }
                    continue;
                }
                let (cs, c) = ckpt(cand)?;
                if let Some(l) = derive_layout(init, &gshape, &c, &cs, false) {
                    hits.push((cand.clone(), l));
                }
            }
            ensure!(
                hits.len() <= 1,
                "{name}: initial value matches {} checkpoint tensors ({}), cannot tell which",
                hits.len(),
                hits.iter()
                    .map(|h| h.0.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            found = hits.pop();
        }
        if let Some((ck, layout)) = found {
            claimed.insert(ck.clone());
            mapped_elems += init.len();
            entries.insert(name, (ck, layout, gshape));
        } else if aliases.contains_key(&name) {
            mapped_elems += init.len();
        } else {
            unmapped.push((name, init.len()));
        }
    }
    Ok(CheckpointMap {
        file: checkpoint_file.to_path_buf(),
        entries,
        aliases,
        unmapped,
        mapped_elems,
    })
}

fn is_constant(v: &[f32]) -> bool {
    v.iter().all(|x| *x == v[0])
}

/// Write trained params as `out_dir/model.safetensors` under the checkpoint's
/// names and shapes (mapped back through `map`), copying every other file of
/// the checkpoint directory (config, tokenizer) and every tensor that was not
/// trained.
pub fn export_lm(
    trained: &HashMap<String, Vec<f32>>,
    map: &CheckpointMap,
    out_dir: &Path,
) -> Result<()> {
    let src_dir = map.file.parent().context("checkpoint has no directory")?;
    std::fs::create_dir_all(out_dir)?;
    for entry in std::fs::read_dir(src_dir)? {
        let p = entry?.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // The export is one `model.safetensors`; a shard index would point
        // loaders at files that are not there.
        let weights_file = name.ends_with(".safetensors")
            || name.ends_with(".safetensors.index.json")
            || name.ends_with(".gguf");
        if p.is_file() && !weights_file {
            std::fs::copy(&p, out_dir.join(name))?;
        }
    }
    let bytes = std::fs::read(&map.file)?;
    let st = safetensors::SafeTensors::deserialize(&bytes).context("parse checkpoint")?;
    let mut replaced: HashMap<&str, Vec<f32>> = HashMap::new();
    for (param, v) in trained {
        let Some((ck, layout, gshape)) = map.entries.get(param) else {
            continue;
        };
        let back = match layout {
            Layout::Same => v.clone(),
            Layout::Offset(c) => v.iter().map(|x| x - c).collect(),
            Layout::Transposed => transpose(v, gshape[0], gshape[1]),
        };
        replaced.insert(ck.as_str(), back);
    }
    let mut tensors = Vec::new();
    for name in st.names() {
        let t = st.tensor(name)?;
        let data = match replaced.remove(name) {
            Some(d) => d,
            None => to_f32(t.dtype(), t.data())?,
        };
        tensors.push((name.to_string(), t.shape().to_vec(), data));
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    save_f32_safetensors(&out_dir.join("model.safetensors"), &tensors)
}

/// The layout that turns checkpoint tensor `c` (shape `cs`) into the graph's
/// initial value `init` (shape `gs`), if any reproduces it exactly.
fn derive_layout(
    init: &[f32],
    gs: &[usize],
    c: &[f32],
    cs: &[usize],
    allow_offset: bool,
) -> Option<Layout> {
    if init.len() != c.len() {
        return None;
    }
    let close = |a: f32, b: f32| (a - b).abs() <= 1e-6 * (1.0 + b.abs());
    if init.iter().zip(c).all(|(a, b)| close(*a, *b)) {
        return Some(Layout::Same);
    }
    if gs.len() == 2 && cs.len() == 2 && gs[0] == cs[1] && gs[1] == cs[0] {
        let t = transpose(c, cs[0], cs[1]);
        if init.iter().zip(&t).all(|(a, b)| close(*a, *b)) {
            return Some(Layout::Transposed);
        }
    }
    let off = init[0] - c[0];
    if allow_offset && off != 0.0 && init.iter().zip(c).all(|(a, b)| close(*a, *b + off)) {
        return Some(Layout::Offset(off));
    }
    None
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; v.len()];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = v[i * cols + j];
        }
    }
    t
}

fn to_f32(dtype: safetensors::Dtype, data: &[u8]) -> Result<Vec<f32>> {
    use safetensors::Dtype;
    Ok(match dtype {
        Dtype::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::BF16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect(),
        Dtype::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        other => bail!("checkpoint dtype {other:?} is not a float type"),
    })
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: renormalize.
            let shift = m.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((m << shift) & 0x3ff) << 13
        }
        (31, 0) => sign | 0x7f80_0000,
        (31, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Write f32 tensors `(name, shape, data)` to a safetensors file.
pub fn save_f32_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, d)| d.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let views = tensors
        .iter()
        .zip(&bytes)
        .map(|((n, s, _), b)| {
            TensorView::new(Dtype::F32, s.clone(), b)
                .map(|v| (n.clone(), v))
                .map_err(|e| anyhow::anyhow!("tensor view for `{n}`: {e}"))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize_to_file(views, None, path)
        .with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tied LM head whose name sorts before the embedding (Gemma's
    /// `lm_head.tied_t` < `model.embed_tokens.weight`) must still map as an
    /// alias of the embedding, not as a second owner of the same tensor.
    #[test]
    fn tied_head_sorting_first_is_an_alias() {
        let (v, h) = (6usize, 4usize);
        let embed: Vec<f32> = (0..v * h).map(|i| i as f32 * 0.25 - 2.0).collect();
        let head: Vec<f32> = (0..h * v).map(|i| embed[(i % v) * h + i / v]).collect();
        let mut g = Graph::new("tied");
        let e = g.add_node(
            Op::Param {
                name: "model.embed_tokens.weight".into(),
            },
            vec![],
            Shape::new(&[v, h], DType::F32),
        );
        let t = g.add_node(
            Op::Param {
                name: "lm_head.tied_t".into(),
            },
            vec![],
            Shape::new(&[h, v], DType::F32),
        );
        let out = g.add_node(Op::MatMul, vec![e, t], Shape::new(&[v, v], DType::F32));
        g.set_outputs(vec![out]);
        let lm = TrainableLm {
            graph: g,
            params: HashMap::from([
                ("model.embed_tokens.weight".into(), embed.clone()),
                ("lm_head.tied_t".into(), head),
            ]),
            token_input: "input_ids".into(),
            batch: 1,
            seq: 1,
            extra_inputs: Vec::new(),
        };
        let dir = std::env::temp_dir().join(format!("rlx-tune-tied-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        save_f32_safetensors(
            &file,
            &[("model.embed_tokens.weight".into(), vec![v, h], embed)],
        )
        .unwrap();
        let map = map_checkpoint(&lm, &file).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(map.entries.contains_key("model.embed_tokens.weight"));
        assert!(!map.entries.contains_key("lm_head.tied_t"));
        assert_eq!(
            map.aliases.get("lm_head.tied_t"),
            Some(&("model.embed_tokens.weight".to_string(), true))
        );
    }
}
