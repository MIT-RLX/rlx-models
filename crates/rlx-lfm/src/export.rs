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

//! Export a Hugging Face LFM2 / LFM2.5 checkpoint (ShortConv + attention
//! hybrid) as an `.rlxp` package rlx runs alone, through the one exporter
//! ([`rlx_core::export`]).
//!
//! The package's step is the runner's decode graph with its host inputs
//! moved inside: the attention mask and the one-hot row write are computed
//! from `position`, so each cache — an attention layer's K and V, a
//! ShortConv layer's convolution window — is a graph variable its output
//! replaces whole, and the step is symbolic in its cache capacity. The
//! RoPE rows are the runner's.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::{Checkpoint, Package};
use rlx_core::standard_decoder::{Lfm2Spec, build_lfm2_decode_ext};
use rlx_ir::op::CmpOp;
use rlx_ir::{DType, Graph, Op, Shape};
use serde_json::{Value, json};

pub use rlx_core::export::{ExportOptions, ExportReport};

use crate::lfm2_gguf::lfm2_spec_from_hf_config;

/// Cache capacities when the options list none.
const DEFAULT_UPPERS: [usize; 4] = [64, 256, 1024, 4096];

/// The step at cache capacity `cap`: the runner's decode graph, its mask
/// (`key_mask`: rows `0..=position`) and row write (`kv_write_oh`, and
/// `kv_write_inv` its complement) computed from the input `position` and
/// the first `cap` of the held `decode.iota` (`rows` long).
fn step(
    spec: &Lfm2Spec,
    ck: &Checkpoint,
    cap: usize,
    rows: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    let mut packed = HashMap::new();
    let mut w = ck.loader()?;
    let (g, params, names) = build_lfm2_decode_ext(spec, w.as_mut(), cap, &mut packed, false)?;
    if !packed.is_empty() {
        bail!("this checkpoint's weights are quantized: export from the f32/bf16 one");
    }
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let (mut position, mut iota) = (None, None);
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| {
        let Op::Input { name } = &node.op else {
            return Ok(None);
        };
        let cmp = match name.as_str() {
            "key_mask" => CmpOp::Le,
            "kv_write_oh" => CmpOp::Eq,
            "kv_write_inv" => CmpOp::Ne,
            _ => return Ok(None),
        };
        let pos = *position.get_or_insert_with(|| out.input("position", f32s(&[1])));
        let at = *iota.get_or_insert_with(|| {
            let t = out.param("decode.iota", f32s(&[rows]));
            out.add_node(
                Op::Narrow {
                    axis: 0,
                    start: 0,
                    len: cap,
                },
                vec![t],
                f32s(&[cap]),
            )
        });
        let b = out.add_node(
            Op::Compare(cmp),
            vec![at, pos],
            Shape::new(&[cap], DType::Bool),
        );
        let v = out.add_node(Op::Cast { to: DType::F32 }, vec![b], f32s(&[cap]));
        let dims: Vec<i64> = node
            .shape
            .dims()
            .iter()
            .map(|d| match d {
                rlx_ir::Dim::Static(n) => *n as i64,
                rlx_ir::Dim::Dynamic(_) => -1,
            })
            .collect();
        Ok(Some(out.reshape(v, dims, node.shape.clone())))
    })
    .map_err(anyhow::Error::msg)?;
    g.name = "step".into();
    Ok((g, params, names))
}

/// Export the LFM2 / LFM2.5 checkpoint in `dir` (`config.json`,
/// `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`) as the
/// package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    if ck.config.get("text_config").is_some() {
        bail!("an LFM2-VL checkpoint: its vision tower is not exported");
    }
    let intermediate = {
        let mut w = ck.loader()?;
        let (_, d) = w.take("model.layers.0.feed_forward.w1.weight")?;
        *d.first().context("feed_forward.w1 has no shape")?
    };
    let spec = lfm2_spec_from_hf_config(&ck.config, intermediate)?;
    let max_positions = ck
        .config
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .unwrap_or(128_000) as usize;
    let mut uppers: Vec<usize> = opts
        .uppers_or(&DEFAULT_UPPERS)
        .into_iter()
        .filter(|&u| u < max_positions)
        .collect();
    uppers.sort();
    let Some(&longest) = uppers.last() else {
        bail!("no cache length fits {max_positions} positions");
    };
    let reach = opts
        .max_context
        .unwrap_or(longest)
        .clamp(longest, max_positions - 1);
    let rows = reach + 1;

    // One step symbolic in its capacity, from two builds no other dim has.
    let mut refr = ck.referencer(opts.embed_weights)?;
    let (a, params, names) = step(&spec, &ck, 37, rows)?;
    refr.take(params)?;
    let (b, _, _) = step(&spec, &ck, 53, rows)?;
    let mut g = rlx_ir::dynamic::generalize(&a, &b, &[(0, 37, 53)])
        .map_err(|e| anyhow::anyhow!("the step is not symbolic in its capacity: {e}"))?;
    g.name = "step".into();
    // Each cache a variable its output replaces (`k_out_3` → `k_in_3`).
    let input = |g: &Graph, name: &str| {
        g.nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name: i } if i == name))
            .map(|n| n.id)
    };
    let mut vars = Vec::new();
    let mut kinds = serde_json::Map::new();
    for (j, out_name) in names.iter().enumerate() {
        let in_name = out_name.replace("_out_", "_in_");
        let id = input(&g, &in_name)
            .with_context(|| format!("the step has no input '{in_name}' for '{out_name}'"))?;
        vars.push((id, j + 1));
        if in_name.starts_with("conv_state_") {
            kinds.insert(in_name, json!("replace"));
        }
    }
    g.set_vars(vars);
    let first = g.outputs[0];
    g.node_mut(first).name = Some("logits".into());

    let family = ck.model_type().unwrap_or("lfm2").to_string();
    let mut pkg = Package::for_checkpoint(&ck, opts, &family, "text-generation", "rlx-lfm");
    pkg.function(g);
    pkg.dims = vec![("cap".into(), 0)];
    pkg.reference(refr, out)?;
    // RoPE rows as the runner computes them, and the positions the mask and
    // the row write compare against.
    let (dh, half) = (spec.head_dim, spec.head_dim / 2);
    let (mut cos, mut sin) = (
        Vec::with_capacity(rows * half),
        Vec::with_capacity(rows * half),
    );
    for pos in 0..rows {
        for i in 0..half {
            let fr = spec.rope_theta.powf(-(2.0 * i as f64) / dh as f64);
            let (s, c) = (pos as f64 * fr).sin_cos();
            cos.push(c as f32);
            sin.push(s as f32);
        }
    }
    pkg.hold_f32("rope_cos", vec![rows, half], &cos);
    pkg.hold_f32("rope_sin", vec![rows, half], &sin);
    let iota: Vec<f32> = (0..rows).map(|i| i as f32).collect();
    pkg.hold_f32("decode.iota", vec![rows], &iota);

    let tokenizer = pkg.tokenizer(&ck, None)?;
    let chat = pkg.chat(&ck);
    pkg.entry(
        "generate",
        json!({
            "functions": ["step"],
            "sizes": uppers.iter().map(|u| json!({"cap": u + 1})).collect::<Vec<_>>(),
            "tokenizer": tokenizer,
            "chat": chat,
            "stop_ids": ck.stop_ids(),
            "vocab_size": spec.vocab_size,
            "inputs": {
                "token_id": "token",
                "position": "position",
                "rope_cos": {"table": "rope_cos"},
                "rope_sin": {"table": "rope_sin"},
            },
            // Every cache's output is its next input; the convolution
            // windows are taken whole, the K/V caches by row.
            "state_update": "replace",
            "state_kinds": kinds,
            "logits": 0,
            "generation": ck.generation(),
            "max_context": reach,
        }),
    );
    pkg.metadata("context_length", json!(reach));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `lfm` runner's name.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("lfm", "text-generation", export_rlxp).claims(&["lfm2"]));
}
