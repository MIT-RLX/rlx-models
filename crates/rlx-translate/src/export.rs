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

//! Exports the Espresso graphs to an open, rlx-consumable form: f32 tensors in
//! **safetensors** (what `rlx_core::weight_map::WeightMap::from_file` reads)
//! plus a JSON description of the graph.
//!
//! # The int8 triple folds into one f32 linear
//!
//! Espresso spells a linear layer as three nodes:
//!
//! ```text
//!   dynamic_quantize   x        -> x.q, x.q_scale        (act_scale = 127/max|x|)
//!   inner_product      x.q      -> acc                   (int8 · int8 -> int32)
//!   dynamic_dequantize acc, scale -> y = acc/(act·w) + b
//! ```
//!
//! Since `x.q ≈ x · act_scale`, the activation scale cancels:
//!
//! ```text
//!   y ≈ x · (W_int8 / w_quantization_scale) + bias
//! ```
//!
//! so the export emits a single f32 weight `W_int8 / w_scale` and a bias, and
//! the graph becomes a plain `MatMul` + `Bias`. This drops the *activation*
//! quantization entirely, so the exported model is if anything slightly more
//! accurate than the shipped one — and it runs on every rlx backend without
//! needing int8 kernels.
//!
//! Embedding tables are dequantized through the quartile curve
//! ([`crate::exec::dequant_gather_row`]) into a dense f32 table: the four
//! `Q_meta` values per column are the 0/25/75/100th percentiles of a uniformly
//! coded byte.
//!
//! # Size
//!
//! f32 is 4× the int8 form: a 38 MB encoder becomes ~152 MB, and the
//! 168 000 × 512 embedding becomes ~344 MB. Export one bundle at a time.

use crate::exec::dequant_gather_row;
use crate::net::Graph;
use anyhow::{Context, Result, anyhow, bail};
use safetensors::tensor::{Dtype, TensorView};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

/// A tensor staged for export.
pub(crate) struct Staged {
    pub(crate) dims: Vec<usize>,
    pub(crate) data: Vec<f32>,
}

/// One node of the exported graph.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExportedOp {
    /// Op name in rlx terms: `matmul`, `gather`, `layernorm`, `add`, `mul`,
    /// `scale`, `recip`, `softmax`, `reshape`, `transpose`, `batch_matmul`,
    /// `copy`.
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    /// Weight tensor names this op consumes, if any.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub weights: Vec<String>,
    /// Op-specific attributes, verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attrs: Option<Value>,
}

/// The exported description of one graph.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExportedGraph {
    pub name: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub ops: Vec<ExportedOp>,
}

/// Converts one Espresso graph and writes `<out>/<name>.safetensors` plus
/// returning its JSON-serializable description.
pub fn export_graph(g: &Graph, out: &Path) -> Result<ExportedGraph> {
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let (tensors, exported) = stage_graph(g)?;
    write_safetensors(&tensors, &out.join(format!("{}.safetensors", g.name)))?;
    Ok(exported)
}

/// Materialises the f32 tensors of `g` and its op list, without writing.
///
/// Both the safetensors and the GGUF writers go through here, so the two
/// formats cannot drift apart in what they contain.
pub(crate) fn stage_graph(g: &Graph) -> Result<(BTreeMap<String, Staged>, ExportedGraph)> {
    let mut tensors: BTreeMap<String, Staged> = BTreeMap::new();
    let mut ops: Vec<ExportedOp> = Vec::new();
    // `dynamic_quantize` output name -> the f32 blob it quantized, so the
    // matmul can be rewired onto the original activation.
    let mut unquant: BTreeMap<String, String> = BTreeMap::new();
    // `inner_product` output -> (source blob, weight blob index, nB, nC)
    let mut pending: BTreeMap<String, (String, u64, usize, usize)> = BTreeMap::new();

    for l in &g.layers {
        match l.kind.as_str() {
            "dynamic_quantize" => {
                // Folded away: record what it consumed.
                let src = l.bottoms.first().cloned().unwrap_or_default();
                for t in &l.tops {
                    unquant.insert(t.clone(), src.clone());
                }
            }
            "inner_product" => {
                let q = l.bottoms.first().cloned().unwrap_or_default();
                let src = unquant.get(&q).cloned().unwrap_or(q);
                let n_in = l
                    .int("nB")
                    .ok_or_else(|| anyhow!("inner_product has no nB"))?
                    as usize;
                let n_out = l
                    .int("nC")
                    .ok_or_else(|| anyhow!("inner_product has no nC"))?
                    as usize;
                pending.insert(l.tops[0].clone(), (src, l.req_blob("W_int8")?, n_in, n_out));
            }
            "dynamic_dequantize" => {
                let acc = l.bottoms.first().cloned().unwrap_or_default();
                let (src, blob, n_in, n_out) = pending
                    .remove(&acc)
                    .ok_or_else(|| anyhow!("dequantize {acc:?} has no matching inner_product"))?;
                let w_scale = l.float("w_quantization_scale").unwrap_or(1.0) as f32;
                if w_scale == 0.0 {
                    bail!("layer {:?} has a zero weight scale", l.name);
                }
                let raw = g.weights.i8s(blob)?;
                let name = format!("{}.weight", l.tops[0]);
                tensors.insert(
                    name.clone(),
                    Staged {
                        dims: vec![n_out, n_in],
                        data: raw.iter().map(|v| f32::from(*v) / w_scale).collect(),
                    },
                );
                let mut weights = vec![name];
                if let Some(b) = l.blob("biases") {
                    let bias = g.weights.f32s(b)?;
                    if !bias.is_empty() {
                        let bn = format!("{}.bias", l.tops[0]);
                        tensors.insert(
                            bn.clone(),
                            Staged {
                                dims: vec![bias.len()],
                                data: bias,
                            },
                        );
                        weights.push(bn);
                    }
                }
                ops.push(ExportedOp {
                    op: "matmul".into(),
                    inputs: vec![src],
                    output: l.tops[0].clone(),
                    weights,
                    attrs: Some(json!({ "relu": l.flag("has_relu") })),
                });
            }
            "quantized_gather" => {
                let cols = l.int("nCol").ok_or_else(|| anyhow!("gather has no nCol"))? as usize;
                let rows = l.int("nRow").ok_or_else(|| anyhow!("gather has no nRow"))? as usize;
                let table = g.weights.raw(l.req_blob("weights_u8")?)?;
                let meta = g.weights.f32s(l.req_blob("Q_meta")?)?;
                let mut data = Vec::with_capacity(rows * cols);
                for r in 0..rows {
                    for c in 0..cols {
                        data.push(dequant_gather_row(
                            table[r * cols + c],
                            &meta[c * 4..c * 4 + 4],
                        ));
                    }
                }
                let name = format!("{}.table", l.tops[0]);
                tensors.insert(
                    name.clone(),
                    Staged {
                        dims: vec![rows, cols],
                        data,
                    },
                );
                ops.push(ExportedOp {
                    op: "gather".into(),
                    inputs: l.bottoms.clone(),
                    output: l.tops[0].clone(),
                    weights: vec![name],
                    attrs: None,
                });
            }
            "instancenorm_1d" => {
                let gamma = g.weights.f32s(l.req_blob("wGamma")?)?;
                let beta = g.weights.f32s(l.req_blob("wBeta")?)?;
                let (gn, bn) = (
                    format!("{}.gamma", l.tops[0]),
                    format!("{}.beta", l.tops[0]),
                );
                tensors.insert(
                    gn.clone(),
                    Staged {
                        dims: vec![gamma.len()],
                        data: gamma,
                    },
                );
                tensors.insert(
                    bn.clone(),
                    Staged {
                        dims: vec![beta.len()],
                        data: beta,
                    },
                );
                ops.push(ExportedOp {
                    op: "layernorm".into(),
                    inputs: l.bottoms.clone(),
                    output: l.tops[0].clone(),
                    weights: vec![gn, bn],
                    attrs: Some(json!({ "eps": l.float("eps").unwrap_or(1e-6) })),
                });
            }
            other => {
                let op = match other {
                    "elementwise" => match l.int("operation").unwrap_or(0) {
                        0 => "add",
                        1 => "mul",
                        3 => "scale",
                        10 => "recip",
                        code => bail!("layer {:?} has unmapped elementwise op {code}", l.name),
                    },
                    "softmax" => "softmax",
                    "reshape" => "reshape",
                    "transpose" => "transpose",
                    "batch_matmul" => "batch_matmul",
                    "copy" => "copy",
                    k => bail!("layer {:?} has unmapped type {k:?}", l.name),
                };
                ops.push(ExportedOp {
                    op: op.into(),
                    inputs: l.bottoms.clone(),
                    output: l.tops[0].clone(),
                    weights: Vec::new(),
                    attrs: Some(Value::Object(
                        l.attrs
                            .iter()
                            .filter(|(k, _)| {
                                !matches!(
                                    k.as_str(),
                                    "name" | "type" | "top" | "bottom" | "weights" | "debug_info"
                                )
                            })
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    )),
                });
            }
        }
    }

    Ok((
        tensors,
        ExportedGraph {
            name: g.name.clone(),
            inputs: g.inputs(),
            output: g.output()?.to_string(),
            ops,
        },
    ))
}

/// Writes staged tensors as f32 safetensors.
pub(crate) fn write_safetensors(tensors: &BTreeMap<String, Staged>, path: &Path) -> Result<()> {
    // safetensors borrows its buffers, so materialise the bytes first.
    let bytes: BTreeMap<String, (Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(k, t)| {
            let raw = t
                .data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>();
            (k.clone(), (t.dims.clone(), raw))
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = bytes
        .iter()
        .map(|(k, (dims, raw))| Ok((k.clone(), TensorView::new(Dtype::F32, dims.clone(), raw)?)))
        .collect::<Result<_, safetensors::SafeTensorError>>()
        .map_err(|e| anyhow!("building tensor views: {e}"))?;
    safetensors::serialize_to_file(views, &None, path)
        .map_err(|e| anyhow!("writing {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::Weights;
    use serde_json::json;

    /// Minimal graph: one quantize/inner_product/dequantize triple.
    fn linear_graph() -> Graph {
        // W = [[1,2],[3,4]] int8, w_scale 2 -> exported [[0.5,1.0],[1.5,2.0]]
        let w: Vec<u8> = vec![1u8, 2, 3, 4];
        let bias: Vec<u8> = [10.0f32, 20.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut c = 2u64.to_le_bytes().to_vec();
        for (i, p) in [(1u64, &w), (2u64, &bias)] {
            c.extend_from_slice(&i.to_le_bytes());
            c.extend_from_slice(&(p.len() as u64).to_le_bytes());
        }
        c.extend_from_slice(&w);
        c.extend_from_slice(&bias);

        let net = json!({
            "storage": "x.espresso.weights",
            "layers": [
                {"name":"q","type":"dynamic_quantize","bottom":"x","top":"x.q,x.s","weights":{}},
                {"name":"ip","type":"inner_product","bottom":"x.q","top":"acc","nB":2,"nC":2,
                 "weights":{"W_int8":1}},
                {"name":"dq","type":"dynamic_dequantize","bottom":"acc,x.s","top":"y",
                 "w_quantization_scale":2.0,"has_relu":1,"weights":{"biases":2}}
            ]
        });
        let layers = crate::net::Graph::layers_from_value(&net).expect("layers");
        Graph {
            name: "x".into(),
            layers,
            shapes: Default::default(),
            weights: Weights::parse(c).expect("weights"),
            storage: "x.espresso.weights".into(),
        }
    }

    #[test]
    fn the_int8_triple_folds_into_one_matmul() {
        let dir = std::env::temp_dir().join(format!("rlx-tr-exp-{}", std::process::id()));
        let g = linear_graph();
        let e = export_graph(&g, &dir).expect("export");
        assert_eq!(e.ops.len(), 1, "quantize and inner_product must fold away");
        assert_eq!(e.ops[0].op, "matmul");
        assert_eq!(
            e.ops[0].inputs,
            vec!["x"],
            "must consume the pre-quantize blob"
        );
        assert_eq!(e.ops[0].output, "y");
        assert_eq!(e.ops[0].weights, vec!["y.weight", "y.bias"]);
        assert_eq!(e.ops[0].attrs.as_ref().expect("attrs")["relu"], json!(true));
        assert!(dir.join("x.safetensors").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn exported_weights_are_the_int8_values_divided_by_the_scale() {
        let dir = std::env::temp_dir().join(format!("rlx-tr-exp2-{}", std::process::id()));
        let g = linear_graph();
        export_graph(&g, &dir).expect("export");
        let raw = std::fs::read(dir.join("x.safetensors")).expect("read");
        let st = safetensors::SafeTensors::deserialize(&raw).expect("parse");
        let w = st.tensor("y.weight").expect("weight");
        assert_eq!(w.shape(), &[2, 2]);
        let vals: Vec<f32> = w
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        assert_eq!(vals, vec![0.5, 1.0, 1.5, 2.0], "W_int8 / w_scale");
        let b = st.tensor("y.bias").expect("bias");
        assert_eq!(b.shape(), &[2]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unmapped_layer_type_is_rejected() {
        let net = json!({"storage":"x.espresso.weights",
            "layers":[{"name":"z","type":"some_new_op","bottom":"x","top":"y","weights":{}}]});
        let g = Graph {
            name: "x".into(),
            layers: crate::net::Graph::layers_from_value(&net).expect("layers"),
            shapes: Default::default(),
            weights: Weights::parse(0u64.to_le_bytes().to_vec()).expect("weights"),
            storage: String::new(),
        };
        let dir = std::env::temp_dir().join(format!("rlx-tr-exp3-{}", std::process::id()));
        let err = export_graph(&g, &dir).expect_err("must not silently drop an op");
        assert!(err.to_string().contains("some_new_op"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
