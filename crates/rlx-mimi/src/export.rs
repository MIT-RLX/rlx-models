// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Export the Mimi decoder as an `.rlxp` package rlx runs alone, through
//! the one exporter ([`rlx_core::export`]): codes `[codebooks, frames]` in
//! — any number of each (codebook 0 the semantic one) — the waveform out.
//!
//! What [`crate::MimiCodec::decode_codes`] does on the host is in the
//! package: each code's codebook row through its branch's output
//! projection, summed (`SplitRvq::decode_frames`; one gather from the
//! codebooks stacked and one batched matmul). The decoder's transformer
//! reads its RoPE tables to the length, and computes its sliding-window
//! causal mask from positions, so the package runs any length up to
//! `max_position_embeddings` transformer steps.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_ir::op::{BinaryOp, CmpOp};
use rlx_ir::{DType, Dim, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::MimiCodec;
use crate::config::MimiConfig;
use crate::graph::{build_decode_graph, rope_tables, sliding_causal_mask};

/// The symbols: the number of frames and of codebooks.
const FRAMES: u32 = 0;
const CODEBOOKS: u32 = 1;

/// What a masked score gets: as far below any score as `-inf`, after the
/// softmax, and finite, so the mask computes from positions.
const MASKED: f32 = -1e30;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn scalar(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    )
}

/// `codes` `[n, t]` → `z` `[1, hidden, t, 1]`: codebook `k`'s row for each
/// code (`codes[k] + k · bins` in the codebooks stacked), through
/// projection `k`, summed over `k`.
fn lookup(cfg: &MimiConfig, books: usize, n: usize, t: usize) -> Graph {
    let (bins, dim, hidden) = (
        cfg.codebook_size,
        cfg.vector_quantization_hidden_dimension,
        cfg.hidden_size,
    );
    let mut g = Graph::new("lookup");
    let codes = g.input("codes", Shape::new(&[n, t], DType::I32));
    let codes = g.add_node(Op::Cast { to: DType::F32 }, vec![codes], f32s(&[n, t]));
    let offsets = g.param("codebook.offsets", f32s(&[books]));
    let offsets = g.narrow_(offsets, 0, 0, n);
    let offsets = g.reshape(offsets, vec![n as i64, 1], f32s(&[n, 1]));
    let offsets = g.add_node(
        Op::Expand {
            target_shape: vec![n as i64, t as i64],
        },
        vec![offsets],
        f32s(&[n, t]),
    );
    let rows = g.add(codes, offsets);
    let rows = g.reshape(rows, vec![(n * t) as i64], f32s(&[n * t]));
    let rows = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![rows],
        Shape::new(&[n * t], DType::I32),
    );
    let table = g.param("codebooks", f32s(&[books * bins, dim]));
    let picked = g.gather(table, rows, 0, f32s(&[n * t, dim]));
    let picked = g.reshape(
        picked,
        vec![n as i64, t as i64, dim as i64],
        f32s(&[n, t, dim]),
    );
    // Codebook `k`'s rows through its projection: `[n, t, dim] @ [n, dim, hidden]`.
    let proj = g.param("output_proj.weight", f32s(&[books, dim, hidden]));
    let proj = g.narrow_(proj, 0, 0, n);
    let x = g.mm(picked, proj);
    let sum = g.sum(x, vec![0], false);
    let z = g.transpose_(sum, vec![1, 0]);
    let z = g.reshape(
        z,
        vec![1, hidden as i64, t as i64, 1],
        f32s(&[1, hidden, t, 1]),
    );
    g.node_mut(z).name = Some("z".into());
    g.set_outputs(vec![z]);
    g
}

/// `graph` with its param `mask` (`[1, s, s]`, the sliding-window causal
/// mask at the length `s`) computed instead: position `i` sees `j` when
/// `0 ≤ i − j < window`, from the first `s` of the held `positions`
/// (`0, 1, …, steps − 1`).
fn mask_from_positions(graph: &Graph, mask: &str, window: usize, steps: usize) -> Result<Graph> {
    rlx_ir::stateful::rebuild(graph, |out, attrs, node, _| {
        let Op::Param { name } = &node.op else {
            return Ok(None);
        };
        if name != mask {
            return Ok(None);
        }
        let s_dim = node.shape.dims()[1];
        let (Dim::Dynamic(sym), Dim::Dynamic(_)) = (s_dim, node.shape.dims()[2]) else {
            return Err("the mask is not symbolic in the length".into());
        };
        let line = Shape::from_dims(&[s_dim], DType::F32);
        let square = Shape::from_dims(&[s_dim, s_dim], DType::F32);
        let positions = out.param("positions", f32s(&[steps]));
        let pos = out.add_node(
            Op::Narrow {
                axis: 0,
                start: 0,
                len: 1,
            },
            vec![positions],
            line,
        );
        attrs.push(rlx_ir::dynamic::SymAttr {
            node: pos,
            path: vec!["len".into()],
            sym,
            scale: 1,
            offset: 0,
        });
        let col = Shape::from_dims(&[s_dim, Dim::Static(1)], DType::F32);
        let row = Shape::from_dims(&[Dim::Static(1), s_dim], DType::F32);
        let i = out.add_node(
            Op::Reshape {
                new_shape: vec![-1, 1],
            },
            vec![pos],
            col,
        );
        let j = out.add_node(
            Op::Reshape {
                new_shape: vec![1, -1],
            },
            vec![pos],
            row,
        );
        let expand = |out: &mut Graph, x| {
            out.add_node(
                Op::Expand {
                    target_shape: vec![-1, -1],
                },
                vec![x],
                square.clone(),
            )
        };
        let (i, j) = (expand(out, i), expand(out, j));
        let back = out.binary(BinaryOp::Sub, i, j, square.clone());
        let zero = scalar(out, 0.0);
        let w = scalar(out, window as f32);
        let flags = Shape::from_dims(&[s_dim, s_dim], DType::Bool);
        let after = out.add_node(Op::Compare(CmpOp::Ge), vec![back, zero], flags.clone());
        let near = out.add_node(Op::Compare(CmpOp::Lt), vec![back, w], flags);
        let after = out.add_node(Op::Cast { to: DType::F32 }, vec![after], square.clone());
        let near = out.add_node(Op::Cast { to: DType::F32 }, vec![near], square.clone());
        let seen = out.binary(BinaryOp::Mul, after, near, square.clone());
        // 1 → 0, 0 → MASKED.
        let one = scalar(out, 1.0);
        let off = out.binary(BinaryOp::Sub, seen, one, square.clone());
        let big = scalar(out, -MASKED);
        let m = out.binary(BinaryOp::Mul, off, big, square);
        Ok(Some(out.add_node(
            Op::Reshape {
                new_shape: vec![1, -1, -1],
            },
            vec![m],
            node.shape.clone(),
        )))
    })
    .map_err(anyhow::Error::msg)
}

/// The Mimi decoder in `dir` as two functions — `lookup` (`codes`
/// `[codebooks, frames]` i32 → `z`) and `decode` (`z` → `audio`), symbolic
/// in the frames (symbol 0) and the codebooks (1) — with their params, the
/// config and the transformer steps they reach: for a package whose model
/// ends in Mimi (Sesame CSM).
pub fn mimi_functions(
    dir: &Path,
) -> Result<(Vec<Graph>, HashMap<String, Vec<f32>>, MimiConfig, usize)> {
    let ck = Checkpoint::open(dir)?;
    let codec = MimiCodec::open(dir)?;
    let cfg = codec.config().clone();
    let (weights, quantizer) = codec.decode_parts();
    let books = cfg.num_quantizers;
    if books < 3 {
        bail!("{books} codebooks: too few to tell the codebook count from the frames");
    }
    let steps = ck.config["max_position_embeddings"]
        .as_u64()
        .unwrap_or(8000) as usize;

    // The lookup, symbolic in the frames and the codebooks; checked at a
    // third size of each.
    let at = |n: usize, t: usize| lookup(&cfg, books, n, t);
    let mut lookup_g =
        rlx_ir::dynamic::generalize(&at(2, 5), &at(3, 7), &[(FRAMES, 5, 7), (CODEBOOKS, 2, 3)])
            .map_err(|e| anyhow::anyhow!("the lookup is not symbolic: {e}"))?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(FRAMES, 4);
    bind.set(CODEBOOKS, books);
    if format!("{}", rlx_ir::dynamic::specialize(&lookup_g, &bind)) != format!("{}", at(books, 4)) {
        bail!("the lookup generalized from two sizes is not the one built at a third");
    }
    lookup_g.name = "lookup".into();
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    params.insert(
        "codebook.offsets".into(),
        (0..books).map(|k| (k * cfg.codebook_size) as f32).collect(),
    );
    let (mut table, mut proj) = (Vec::new(), Vec::new());
    for (codebook, out_proj) in quantizer.decoding() {
        table.extend(codebook.iter());
        // `[hidden, dim]` → `[dim, hidden]`.
        proj.extend(out_proj.t().iter());
    }
    params.insert("codebooks".into(), table);
    params.insert("output_proj.weight".into(), proj);

    let decode = |t: usize| -> Result<(Graph, HashMap<String, Vec<f32>>)> {
        let (mut g, p, _) = build_decode_graph(&weights, t)?;
        g.name = "decode".into();
        Ok((g, p.into_iter().collect()))
    };
    let g = symbolic_in(FRAMES, (5, 7, 4), |t| Ok(decode(t)?.0))?;
    let (at4, mut decode_params) = decode(4)?;
    // The mask is the one param of the built graph equal to the mask at its
    // transformer length.
    let tf = weights.transformer;
    let s4 = transformer_steps(&at4)?;
    let want = sliding_causal_mask(s4, tf.sliding_window);
    let mask = decode_params
        .iter()
        .find(|(_, v)| **v == want)
        .map(|(k, _)| k.clone())
        .ok_or_else(|| anyhow::anyhow!("no param of the decoder is its mask"))?;
    decode_params.remove(&mask);
    let g = mask_from_positions(&g, &mask, tf.sliding_window, steps)?;
    let (cos, sin) = rope_tables(tf.inv_freq.as_slice().expect("contiguous"), steps);
    let d = 2 * tf.inv_freq.len();
    let g = broadcast_params(
        &g,
        &at4,
        &mut decode_params,
        &[("rope.cos", &cos, d), ("rope.sin", &sin, d)],
    )?;
    params.extend(decode_params);
    params.insert("positions".into(), (0..steps).map(|i| i as f32).collect());
    Ok((vec![lookup_g, g], params, cfg, steps))
}

/// Export the Mimi checkpoint in `dir` (`config.json`, `model.safetensors`)
/// as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let (graphs, params, cfg, steps) = mimi_functions(dir)?;
    let books = cfg.num_quantizers;
    let [lookup_g, g]: [Graph; 2] = graphs
        .try_into()
        .map_err(|_| anyhow::anyhow!("two functions"))?;
    let stages = vec![
        Stage::new(lookup_g).outputs(&["z"]),
        Stage::new(g).input("z", "z").outputs(&["audio"]),
    ];
    let mut p = Pipeline::new("mimi", "audio-decoding", stages);
    p.inputs = vec![json!({"name": "codes", "kind": "tensor", "dtype": "i32",
                           "shape": [null, null], "codebooks_at_most": books})];
    p.dims = vec![("frames".into(), FRAMES), ("codebooks".into(), CODEBOOKS)];
    p.metadata = vec![
        ("sample_rate".into(), json!(cfg.sampling_rate)),
        ("frame_rate".into(), json!(cfg.frame_rate)),
        ("max_transformer_steps".into(), json!(steps)),
    ];
    p.exported_by = "rlx-mimi".into();
    p.write(params, Some(&ck), out, opts)
}

/// The transformer's length in `graph` (the decoder built at a size): the
/// side of its one square param.
fn transformer_steps(graph: &Graph) -> Result<usize> {
    graph
        .nodes()
        .iter()
        .find_map(|n| match (&n.op, n.shape.dims()) {
            (Op::Param { .. }, [Dim::Static(1), Dim::Static(a), Dim::Static(b)]) if a == b => {
                Some(*a)
            }
            _ => None,
        })
        .ok_or_else(|| anyhow::anyhow!("the decoder has no mask"))
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("mimi", "audio-decoding", export_rlxp)
            .claims(&["mimi"])
            .architectures(&["MimiModel"]),
    );
}
