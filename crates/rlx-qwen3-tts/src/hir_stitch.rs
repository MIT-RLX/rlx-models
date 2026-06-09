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

//! Append tier-1 graph segments into one MIR graph (codec-frame megagraph).

use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

/// External bindings for a segment's `Input` nodes (by input name).
pub type InputBindMap = HashMap<String, NodeId>;

/// Remapped output node ids after append (same order as `src.outputs`).
pub type SegmentOutputs = Vec<NodeId>;

fn input_name(op: &Op) -> Option<&str> {
    match op {
        Op::Input { name } => Some(name.as_str()),
        _ => None,
    }
}

fn param_name(op: &Op) -> Option<&str> {
    match op {
        Op::Param { name } => Some(name.as_str()),
        _ => None,
    }
}

/// Append `src` onto `dst`, prefixing params and wiring `input_bind`.
pub fn append_graph_segment(
    dst: &mut Graph,
    dst_params: &mut HashMap<String, Vec<f32>>,
    src: &Graph,
    src_params: &HashMap<String, Vec<f32>>,
    param_prefix: &str,
    input_bind: &InputBindMap,
    shared_params: &mut HashMap<String, NodeId>,
) -> Result<SegmentOutputs> {
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in src.nodes() {
        if let Some(name) = input_name(&node.op) {
            let bound = input_bind
                .get(name)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing input bind for `{name}`"))?;
            id_map.insert(node.id, bound);
            continue;
        }

        if let Some(name) = param_name(&node.op) {
            let key = if param_prefix.is_empty() {
                name.to_string()
            } else {
                format!("{param_prefix}{name}")
            };
            if let Some(&existing) = shared_params.get(&key) {
                id_map.insert(node.id, existing);
                continue;
            }
            let data = src_params
                .get(name)
                .or_else(|| src_params.get(&key))
                .ok_or_else(|| anyhow::anyhow!("missing param `{name}` for stitch"))?;
            dst_params.insert(key.clone(), data.clone());
            let id = dst.append_node(
                Op::Param { name: key.clone() },
                vec![],
                node.shape.clone(),
                Some(key.clone()),
            );
            shared_params.insert(key, id);
            id_map.insert(node.id, id);
            continue;
        }

        let inputs: Vec<NodeId> = node.inputs.iter().map(|&id| id_map[&id]).collect();
        let appended = dst.append_node(
            node.op.clone(),
            inputs,
            node.shape.clone(),
            node.name.clone(),
        );
        id_map.insert(node.id, appended);
    }

    Ok(src.outputs.iter().map(|&id| id_map[&id]).collect())
}

/// Gather one codec-group row from a table param using a scalar token id input → `[1, 1, hidden]`.
pub fn gather_group_embed_3d(
    dst: &mut Graph,
    dst_params: &mut HashMap<String, Vec<f32>>,
    tok_input: NodeId,
    weight_key: &str,
    weight: &[f32],
    weight_shape: &[usize],
    hidden: usize,
    param_prefix: &str,
    shared_params: &mut HashMap<String, NodeId>,
) -> Result<NodeId> {
    ensure!(weight_shape.len() == 2, "embed table must be 2D");
    ensure!(weight_shape[1] == hidden, "embed hidden mismatch");
    let key = if param_prefix.is_empty() {
        weight_key.to_string()
    } else {
        format!("{param_prefix}{weight_key}")
    };
    let table_id = if let Some(&existing) = shared_params.get(&key) {
        existing
    } else {
        dst_params.insert(key.clone(), weight.to_vec());
        let id = dst.append_node(
            Op::Param { name: key.clone() },
            vec![],
            Shape::new(weight_shape, DType::F32),
            Some(key.clone()),
        );
        shared_params.insert(key, id);
        id
    };
    let gathered = dst.gather_(table_id, tok_input, 0);
    let embed_3d = dst.reshape_(gathered, vec![1, 1, hidden as i64]);
    Ok(embed_3d)
}
