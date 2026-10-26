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

//! Export Fish Audio S2-Pro as an `.rlxp` package rlx runs alone: text in,
//! 44.1 kHz speech out, through the one exporter ([`rlx_core::export`]).
//!
//! Its graphs are [`crate::s2::S2Model`]'s and [`crate::codec::FishCodec`]'s:
//! `embed` (rows: the token embedding, plus an audio row's codebook rows,
//! scaled by `1/√11`); `prefill` (the slow stack over the prompt, symbolic in
//! its rows and the caches' rows: the last row's hidden state and band
//! logits, each layer's keys and values); `step` (one row after the caches,
//! its key and value written in at `pos`); `fast` (one fast-decoder position
//! against its 10-row caches) and `fast_embed`; `dequant` (each code's
//! decoded codebook row, summed) and `codec` (the post transformer,
//! upsampling and the DAC decoder, symbolic in the frames). The stacks'
//! weights are the checkpoint's tensors, referenced; the codec's, from
//! `codec.pth`, held. The program (`program/`) builds the conversation (the
//! crate's `prompt.rs`), samples (its `sampling.rs`: top-p, top-k,
//! repetition-aware), fills each frame's codes and decodes.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_core::weight_map::WeightMap;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::codec::{FRAME_SAMPLES, FishCodec, NUM_CODEBOOKS, SAMPLE_RATE, build_decoder};
use crate::nn::{Builder, Init, TensorOp};
use crate::s2::{S2Model, S2Options};

/// The symbols: the codec's frames, the prompt's rows, the caches' rows.
const FRAMES: u32 = 0;
const PROMPT: u32 = 2;
const CAP: u32 = 3;
/// RoPE rows the tables hold.
const MAX_ROWS: usize = 8192;
/// The codec's post transformer attends to this many keys (itself
/// included).
const CODEC_WINDOW: usize = 128;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// A builder's graph as a package's: each checkpoint tensor a param of its
/// own name (a transposed one read through a `Transpose`, a `+1` one
/// through an `Add`), each computed tensor a param `held` takes. The
/// checkpoint names it reads go to `refs`.
pub(crate) fn package_graph(
    mut b: Builder,
    outs: Vec<NodeId>,
    held: &mut HashMap<String, Vec<f32>>,
    refs: &mut BTreeSet<String>,
) -> Result<Graph> {
    b.g.set_outputs(outs);
    let specs: HashMap<String, crate::nn::ParamSpec> =
        b.params.into_iter().map(|p| (p.name.clone(), p)).collect();
    let g = rlx_ir::stateful::rebuild(&b.g, |out, _, node, _| {
        let Op::Param { name } = &node.op else {
            return Ok(None);
        };
        let spec = specs
            .get(name)
            .ok_or_else(|| format!("no spec for param {name}"))?;
        match &spec.init {
            Init::Data(d) => {
                held.insert(name.clone(), d.clone());
                Ok(Some(out.param(name.clone(), f32s(&spec.shape))))
            }
            Init::Tensor { key, op } => {
                refs.insert(key.clone());
                Ok(Some(match op {
                    TensorOp::Raw => out.param(key.clone(), f32s(&spec.shape)),
                    TensorOp::Transpose => {
                        let s = &spec.shape;
                        let p = out.param(key.clone(), f32s(&[s[1], s[0]]));
                        out.transpose_(p, vec![1, 0])
                    }
                    TensorOp::PlusOne => {
                        let p = out.param(key.clone(), f32s(&spec.shape));
                        let one = out.add_node(
                            Op::Constant {
                                data: 1f32.to_le_bytes().to_vec(),
                            },
                            vec![],
                            f32s(&[1]),
                        );
                        out.add(p, one)
                    }
                }))
            }
        }
    })
    .map_err(anyhow::Error::msg)?;
    Ok(g)
}

/// `g`'s RoPE inputs (`rope_cos` / `rope_sin`, `[n, half]`) the rows of the
/// tables `{table}.cos` / `{table}.sin`: rows `0..n` (`at` none) or the rows
/// at the input `at` (`f32[n]`).
fn rope_rows(g: &Graph, table: &str, at: Option<&str>) -> Result<Graph> {
    rope_rows_of(g, table, at, MAX_ROWS)
}

/// [`rope_rows`] from tables of `rows` rows.
pub(crate) fn rope_rows_of(g: &Graph, table: &str, at: Option<&str>, rows: usize) -> Result<Graph> {
    let mut index = None;
    rlx_ir::stateful::rebuild(g, |out, _, node, _| {
        let Op::Input { name } = &node.op else {
            return Ok(None);
        };
        let which = match name.as_str() {
            "rope_cos" => "cos",
            "rope_sin" => "sin",
            _ => return Ok(None),
        };
        let dims: Vec<usize> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let (n, half) = (dims[0], dims[1]);
        let t = out.param(format!("{table}.{which}"), f32s(&[rows, half]));
        Ok(Some(match at {
            None => out.narrow_(t, 0, 0, n),
            Some(at) => {
                let i = *index.get_or_insert_with(|| out.input(at, f32s(&[n])));
                out.gather_(t, i, 0)
            }
        }))
    })
    .map_err(anyhow::Error::msg)
}

/// A step graph's caches `{past}{l}` written in at the input `pos`: each new
/// row output (from `first`, K then V per layer) replaced by its cache with
/// the row in (`{past}{l}.next`).
fn write_rows(g: &mut Graph, first: usize, layers: usize, past: &str) -> Result<()> {
    let pos = match g.input_id("pos") {
        Some(p) => p,
        None => g.input("pos", f32s(&[1])),
    };
    let mut outs = g.outputs.clone();
    for l in 0..layers {
        for (j, kind) in ["k", "v"].iter().enumerate() {
            let name = format!("{past}{kind}_{l}");
            let cache = g.input_id(&name).with_context(|| format!("no {name}"))?;
            let shape = g.node(cache).shape.clone();
            let row = outs[first + 2 * l + j];
            let full = g.add_node(
                Op::DynamicUpdateSlice { axis: 1 },
                vec![cache, row, pos],
                shape,
            );
            g.node_mut(full).name = Some(format!("{name}.next"));
            outs[first + 2 * l + j] = full;
        }
    }
    g.set_outputs(outs);
    Ok(())
}

/// `prefill` at `seq` rows and `cap` cache rows.
fn prefill(
    m: &S2Model,
    seq: usize,
    cap: usize,
    held: &mut HashMap<String, Vec<f32>>,
    refs: &mut BTreeSet<String>,
) -> Result<Graph> {
    let (b, outs) = m.prefill_builder(seq);
    let mut g = package_graph(b, outs, held, refs)?;
    g = rope_rows(&g, "slow.rope", None)?;
    let s = &m.cfg.slow;
    let (dim, kvw) = (s.dim, s.n_kv * s.head_dim);
    let nb = m.cfg.codebook_size + 1;
    let outs = g.outputs.clone();
    let h = g.narrow_(outs[0], 1, seq - 1, 1);
    let h = g.reshape_(h, vec![1, dim as i64]);
    g.node_mut(h).name = Some("hidden".into());
    let head = g.param("fish.band_head", f32s(&[dim, nb]));
    let logits = g.mm(h, head);
    g.node_mut(logits).name = Some("logits".into());
    let origin = g.add_node(
        Op::Constant {
            data: 0f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let nought = g.add_node(
        Op::Constant {
            data: 0f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1, 1, 1]),
    );
    let cache = f32s(&[1, cap, kvw]);
    let mut new_outs = vec![h, logits];
    for (i, &kv) in outs[1..].iter().enumerate() {
        let z = g.add_node(
            Op::Expand {
                target_shape: vec![1, cap as i64, kvw as i64],
            },
            vec![nought],
            cache.clone(),
        );
        let full = g.add_node(
            Op::DynamicUpdateSlice { axis: 1 },
            vec![z, kv, origin],
            cache.clone(),
        );
        let kind = if i % 2 == 0 { "k" } else { "v" };
        g.node_mut(full).name = Some(format!("past_{kind}_{}", i / 2));
        new_outs.push(full);
    }
    g.set_outputs(new_outs);
    g.name = "prefill".into();
    Ok(g)
}

/// `step` at `cap` cache rows.
fn step(
    m: &S2Model,
    cap: usize,
    held: &mut HashMap<String, Vec<f32>>,
    refs: &mut BTreeSet<String>,
) -> Result<Graph> {
    let (b, outs) = m.decode_builder(cap);
    let mut g = package_graph(b, outs, held, refs)?;
    g = rope_rows(&g, "slow.rope", Some("pos"))?;
    let outs = g.outputs.clone();
    g.node_mut(outs[0]).name = Some("hidden".into());
    g.node_mut(outs[1]).name = Some("logits".into());
    write_rows(&mut g, 2, m.cfg.slow.n_layer, "past_")?;
    g.name = "step".into();
    Ok(g)
}

/// `fast`: one fast-decoder position.
fn fast(
    m: &S2Model,
    held: &mut HashMap<String, Vec<f32>>,
    refs: &mut BTreeSet<String>,
) -> Result<Graph> {
    let (b, outs) = m.fast_builder();
    let mut g = package_graph(b, outs, held, refs)?;
    g = rope_rows(&g, "fast.rope", Some("pos"))?;
    let outs = g.outputs.clone();
    g.node_mut(outs[0]).name = Some("logits".into());
    write_rows(&mut g, 1, m.cfg.fast.n_layer, "past_")?;
    g.name = "fast".into();
    Ok(g)
}

const EMBED: &str = "text_model.model.embeddings.weight";
const CODEBOOK_EMBED: &str = "audio_decoder.codebook_embeddings.weight";
const FAST_EMBED: &str = "audio_decoder.embeddings.weight";

/// `embed`: `tokens` `[t]` i32, `codes` `[t, 10]` i32 (an audio row's
/// `i · 4096 + code_i`), `audio` `[t]` (1 for an audio row) →
/// `embeds` `[1, t, dim]`.
fn embed(m: &S2Model, vocab: usize, cb_rows: usize, t: usize) -> Graph {
    let dim = m.cfg.slow.dim;
    let nc = m.cfg.num_codebooks;
    let mut g = Graph::new("embed");
    let tokens = g.input("tokens", Shape::new(&[t], DType::I32));
    let codes = g.input("codes", Shape::new(&[t, nc], DType::I32));
    let audio = g.input("audio", f32s(&[t, 1]));
    let table = g.param(EMBED, f32s(&[vocab, dim]));
    let tok = g.gather_(table, tokens, 0);
    let cb = g.param(CODEBOOK_EMBED, f32s(&[cb_rows, dim]));
    let flat = g.reshape_(codes, vec![(t * nc) as i64]);
    let rows = g.gather_(cb, flat, 0);
    let rows = g.reshape_(rows, vec![t as i64, nc as i64, dim as i64]);
    let sum = g.sum(rows, vec![1], false);
    let sum = g.mul(sum, audio);
    let x = g.add(tok, sum);
    // An audio row's sum scaled by 1/√(N+1), a text row's kept.
    let s = 1.0 / ((nc + 1) as f32).sqrt();
    let k = g.add_node(
        Op::Constant {
            data: (s - 1.0).to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let one = g.add_node(
        Op::Constant {
            data: 1f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let gain = g.mul(audio, k);
    let gain = g.add(gain, one);
    let x = g.mul(x, gain);
    let x = g.reshape_(x, vec![1, t as i64, dim as i64]);
    g.node_mut(x).name = Some("embeds".into());
    g.set_outputs(vec![x]);
    g
}

/// `fast_embed`: a code `id` `[1]` i32 → its fast-decoder row `[1, 1, dim]`.
fn fast_embed(m: &S2Model, rows: usize) -> Graph {
    let dim = m.cfg.fast.dim;
    let mut g = Graph::new("fast_embed");
    let id = g.input("id", Shape::new(&[1], DType::I32));
    let t = g.param(FAST_EMBED, f32s(&[rows, dim]));
    let x = g.gather_(t, id, 0);
    let x = g.reshape_(x, vec![1, 1, dim as i64]);
    g.node_mut(x).name = Some("x".into());
    g.set_outputs(vec![x]);
    g
}

/// `dequant`: `codes` `[f, 10]` i32 (each codebook's code) → `z`
/// `[1, f, 1024]`, the codes' decoded rows summed.
fn dequant(f: usize, rows: usize) -> Graph {
    let mut g = Graph::new("dequant");
    let codes = g.input("codes", Shape::new(&[f, NUM_CODEBOOKS], DType::I32));
    let codes = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![codes],
        f32s(&[f, NUM_CODEBOOKS]),
    );
    let off = g.param("codec.offsets", f32s(&[1, NUM_CODEBOOKS]));
    let ids = g.add(codes, off);
    let ids = g.reshape_(ids, vec![(f * NUM_CODEBOOKS) as i64]);
    let ids = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![ids],
        Shape::new(&[f * NUM_CODEBOOKS], DType::I32),
    );
    let table = g.param("codec.decoded", f32s(&[rows, 1024]));
    let picked = g.gather_(table, ids, 0);
    let picked = g.reshape_(picked, vec![f as i64, NUM_CODEBOOKS as i64, 1024]);
    let z = g.sum(picked, vec![1], false);
    let z = g.reshape_(z, vec![1, f as i64, 1024]);
    g.node_mut(z).name = Some("z".into());
    g.set_outputs(vec![z]);
    g
}

/// `codec` at `f` frames: [`build_decoder`]'s graph, its post transformer's
/// RoPE rows read from one table (`codec.rope.cos` / `.sin`), its mask the
/// sliding window at any length (the reference's causal mask below the
/// window is the same), each computed tensor baked as a constant (the fit's
/// to make symbolic) but for the upsampling weights, held.
fn codec(src: &rlx_torch_ckpt::PtModel, f: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (mut b, out) = build_decoder(src, f)?;
    b.g.set_outputs(vec![out]);
    let specs: HashMap<String, crate::nn::ParamSpec> =
        b.params.into_iter().map(|p| (p.name.clone(), p)).collect();
    let mut held = HashMap::new();
    let g = rlx_ir::stateful::rebuild(&b.g, |out, _, node, _| {
        match &node.op {
            Op::Param { name } => {
                let spec = specs
                    .get(name)
                    .ok_or_else(|| format!("no spec for param {name}"))?;
                if let Some(which) = name
                    .rsplit_once('.')
                    .and_then(|(head, _)| head.rsplit_once('.').map(|(_, w)| w))
                    .filter(|w| *w == "rope_cos" || *w == "rope_sin")
                {
                    let half = spec.shape[1];
                    let t = out.param(
                        format!("codec.{}", which.replace("rope_", "rope.")),
                        f32s(&[MAX_ROWS, half]),
                    );
                    return Ok(Some(out.narrow_(t, 0, 0, spec.shape[0])));
                }
                let data = match &spec.init {
                    Init::Data(d) => d.clone(),
                    Init::Tensor { key, op } => {
                        let (d, shape) =
                            crate::nn::TensorSource::tensor(src, key).map_err(|e| e.to_string())?;
                        match op {
                            TensorOp::Raw => d,
                            TensorOp::PlusOne => d.into_iter().map(|v| v + 1.0).collect(),
                            TensorOp::Transpose => crate::nn::transpose(&d, shape[0], shape[1]),
                        }
                    }
                };
                // Checkpoint tensors and the upsampling weights: params.
                let weight = matches!(spec.init, Init::Tensor { .. }) || data.len() >= 1 << 18;
                if weight {
                    held.insert(format!("codec.{name}"), data);
                    return Ok(Some(out.param(format!("codec.{name}"), f32s(&spec.shape))));
                }
                Ok(Some(out.add_node(
                    Op::Constant {
                        data: data.iter().flat_map(|x| x.to_le_bytes()).collect(),
                    },
                    vec![],
                    node.shape.clone(),
                )))
            }
            _ => Ok(None),
        }
    })
    .map_err(anyhow::Error::msg)?;
    // The window's mask at every length.
    let mut g = g;
    for node in g.nodes_mut() {
        if let Op::Attention { mask_kind, .. } = &mut node.op
            && *mask_kind == MaskKind::Causal
        {
            *mask_kind = MaskKind::SlidingWindow(CODEC_WINDOW - 1);
        }
    }
    let out = g.outputs[0];
    let audio = g.reshape_(out, vec![(f * FRAME_SAMPLES) as i64]);
    g.node_mut(audio).name = Some("audio".into());
    g.set_outputs(vec![audio]);
    g.name = "codec".into();
    Ok((g, held))
}

/// The codec of `dir`'s `codec.pth` as the functions `dequant` and `codec`
/// (symbolic in the frames) of `pkg`; the tensors they hold go to `held`.
pub(crate) fn codec_functions(
    pkg: &mut Package,
    dir: &Path,
    held: &mut HashMap<String, Vec<f32>>,
) -> Result<()> {
    let fc = FishCodec::open(dir, rlx_runtime::Device::Cpu)?;
    let decoded = fc.decoded_rows();
    let rows = decoded.len() / 1024;
    pkg.function(symbolic_in(FRAMES, (5, 7, 3), |f| Ok(dequant(f, rows)))?);
    let (cg, lifted) =
        symbolic_in_lifted(
            FRAMES,
            (5, 7, 3),
            1024,
            |f| Ok(codec(fc.checkpoint(), f)?.0),
        )?;
    pkg.function(cg);
    let (_, cparams) = codec(fc.checkpoint(), 3)?;
    held.extend(cparams);
    held.extend(lifted);
    held.insert("codec.decoded".into(), decoded);
    let mut off = vec![0f32];
    off.extend(
        (1..NUM_CODEBOOKS)
            .map(|i| (crate::codec::SEMANTIC_SIZE + (i - 1) * crate::codec::RESIDUAL_SIZE) as f32),
    );
    held.insert("codec.offsets".into(), off);
    let positions: Vec<usize> = (0..MAX_ROWS).collect();
    let (cos, sin) = crate::nn::rope_tables(&positions, 64, 10_000.0, true);
    held.insert("codec.rope.cos".into(), cos);
    held.insert("codec.rope.sin".into(), sin);
    Ok(())
}

/// Export the S2-Pro directory `dir` (`config.json`, `model*.safetensors`,
/// `tokenizer.json`, `codec.pth`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    // Expanded K/V heads: every backend's attention takes them.
    rlx_ir::env::set("RLX_FISH_GQA_EXPAND", "1");
    let m = S2Model::open(
        dir,
        rlx_runtime::Device::Cpu,
        S2Options {
            weight_storage: Some(rlx_flow::storage::WeightStorage::F32),
            ..S2Options::default()
        },
    )?;
    let cfg = m.cfg.clone();
    let mut pkg = Package::new(
        "fish-s2-pro",
        &opts.version,
        "fish-s2",
        "text-to-speech",
        "rlx-fish",
    );
    pkg.model["license"] = json!("fish-audio-research");
    let mut held: HashMap<String, Vec<f32>> = HashMap::new();
    let mut refs: BTreeSet<String> = BTreeSet::new();

    let (p, lifted) = symbolic_in_lifted_many(
        &[(PROMPT, vec![5, 7, 3]), (CAP, vec![37, 53, 20])],
        &[MAX_ROWS, MAX_ROWS],
        |n| prefill(&m, n[0], n[1], &mut HashMap::new(), &mut BTreeSet::new()),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the prompt graph lifted tables");
    prefill(&m, 3, 20, &mut held, &mut refs)?;
    pkg.function(p);
    pkg.function(symbolic_in(CAP, (37, 53, 20), |c| {
        step(&m, c, &mut HashMap::new(), &mut BTreeSet::new())
    })?);
    step(&m, 20, &mut held, &mut refs)?;
    pkg.function(fast(&m, &mut held, &mut refs)?);
    held.insert("fish.band_head".into(), m.band_head.clone());

    // The rows' graphs.
    let shard_dims = |k: &str| -> Result<Vec<usize>> {
        for f in std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())) {
            if f.extension().is_some_and(|e| e == "safetensors") {
                let w = WeightMap::from_file(f.to_str().context("path")?)?;
                if let Some((_, s)) = w.get(k) {
                    return Ok(s.to_vec());
                }
            }
        }
        anyhow::bail!("no {k}")
    };
    let vocab = shard_dims(EMBED)?[0];
    let cb_rows = shard_dims(CODEBOOK_EMBED)?[0];
    let fe_rows = shard_dims(FAST_EMBED)?[0];
    pkg.function(symbolic_in(PROMPT, (5, 7, 3), |t| {
        Ok(embed(&m, vocab, cb_rows, t))
    })?);
    pkg.function(fast_embed(&m, fe_rows));
    refs.extend([
        EMBED.to_string(),
        CODEBOOK_EMBED.to_string(),
        FAST_EMBED.to_string(),
    ]);

    codec_functions(&mut pkg, dir, &mut held)?;
    for (table, hd) in [
        ("slow.rope", cfg.slow.head_dim),
        ("fast.rope", cfg.fast.head_dim),
    ] {
        let positions: Vec<usize> = (0..MAX_ROWS).collect();
        let (cos, sin) = crate::nn::rope_tables(&positions, hd, cfg.rope_base, true);
        held.insert(format!("{table}.cos"), cos);
        held.insert(format!("{table}.sin"), sin);
    }

    pkg.dims = vec![
        ("frames".into(), FRAMES),
        ("prompt".into(), PROMPT),
        ("cap".into(), CAP),
    ];
    let pdims = pkg.param_dims();
    let mut names: Vec<&String> = held.keys().collect();
    names.sort();
    for n in names {
        let d = pdims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
        pkg.hold_f32(n.clone(), d, &held[n]);
    }
    drop(held);

    // The stacks' tensors, referenced shard by shard.
    let mut shards: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    for shard in shards {
        let source = WeightMap::from_file(shard.to_str().context("path")?)?;
        let params: HashMap<String, Vec<f32>> = refs
            .iter()
            .filter_map(|k| source.get(k).map(|(d, _)| (k.clone(), d.to_vec())))
            .collect();
        if !params.is_empty() {
            pkg.reference_map(&params, &source, "safetensors", &shard, out, false)?;
        }
    }

    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(dir.join("tokenizer.json"))?,
    );
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build Fish S2's program")?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": "sidecar:tokenizer.json",
            "slow_layers": cfg.slow.n_layer,
            "slow_kv": cfg.slow.n_kv * cfg.slow.head_dim,
            "fast_layers": cfg.fast.n_layer,
            "fast_kv": cfg.fast.n_kv * cfg.fast.head_dim,
            "dim": cfg.slow.dim,
            "fast_dim": cfg.fast.dim,
            "num_codebooks": cfg.num_codebooks,
            "codebook_size": cfg.codebook_size,
            "semantic_begin": cfg.semantic_begin,
            "semantic_end": cfg.semantic_end,
            "eos": cfg.eos,
            "max_seq_len": cfg.max_seq_len,
            "max_rows": MAX_ROWS,
            "max_new_tokens": crate::s2::DEFAULT_MAX_NEW_TOKENS,
            "chunk_length": crate::s2::DEFAULT_CHUNK_LENGTH,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 30,
            "seed": 0,
            "semantic_size": crate::codec::SEMANTIC_SIZE,
            "residual_size": crate::codec::RESIDUAL_SIZE,
            "sample_rate": SAMPLE_RATE,
        }),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("fish-s2", "text-to-speech", export_rlxp));
}
