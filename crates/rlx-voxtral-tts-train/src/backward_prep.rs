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

//! Lower training backward ops for Metal / MLX / wgpu / Vulkan before compile.

use rlx_autodiff::decompose_backward_ops_except;
use rlx_autodiff::legalize_reduce::legalize_multi_axis_reduce;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::OpKind;
use rlx_ir::op::ReduceOp;
use rlx_ir::{Graph, NodeId, Op, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

pub fn prepare_backward_for_device(graph: Graph, device: Device) -> Graph {
    if !needs_portable_backward_prep(device) {
        return graph;
    }
    let g = legalize_multi_axis_reduce(graph);
    let g = decompose_for_device(g, device);
    let g = lower_non_last_axis_reduce(g);
    if backward_graph_opt_from_env() {
        optimize_backward_graph(g)
    } else {
        g
    }
}

fn backward_graph_opt_from_env() -> bool {
    !std::env::var("RLX_VOXTRAL_TTS_TRAIN_BACKWARD_GRAPH_OPT")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "0" | "false" | "no"))
}

pub fn needs_portable_backward_prep(device: Device) -> bool {
    matches!(
        device,
        Device::Metal | Device::Mlx | Device::Gpu | Device::Vulkan
    )
}

fn decompose_for_device(g: Graph, device: Device) -> Graph {
    match device {
        // Native conv + attention backward in `rlx-mlx`; decompose RMS + other training ops only.
        Device::Mlx => decompose_backward_ops_except(
            g,
            &[
                OpKind::Conv2dBackwardInput,
                OpKind::Conv2dBackwardWeight,
                OpKind::AttentionBackward,
            ],
        ),
        // Native RMS, attention, and conv backward thunks on Metal; decompose the rest.
        Device::Metal => decompose_backward_ops_except(
            g,
            &[
                OpKind::RmsNormBackwardInput,
                OpKind::RmsNormBackwardGamma,
                OpKind::RmsNormBackwardBeta,
                OpKind::AttentionBackward,
                OpKind::Conv2dBackwardInput,
                OpKind::Conv2dBackwardWeight,
                OpKind::RopeBackward,
                OpKind::CumsumBackward,
                OpKind::GatherBackward,
            ],
        ),
        Device::Gpu | Device::Vulkan => {
            decompose_backward_ops_except(g, &[OpKind::AttentionBackward])
        }
        _ => g,
    }
}

fn lower_non_last_axis_reduce(graph: Graph) -> Graph {
    let needs = graph.nodes().iter().any(|n| {
        if let Op::Reduce { axes, .. } = &n.op {
            let rank = graph.shape(n.inputs[0]).rank();
            let mut ax: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if (a as i32) < 0 {
                        (rank as i32 + a as i32) as usize
                    } else {
                        a
                    }
                })
                .collect();
            ax.sort_unstable();
            ax.dedup();
            ax.len() > 1 || (rank > 0 && ax.as_slice() != [rank - 1])
        } else {
            false
        }
    });
    if !needs {
        return graph;
    }

    let mut out = Graph::new(format!("{}_reduce_prep", graph.name));
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if let Op::Reduce { op, axes, keep_dim } = &node.op {
            let input = inputs[0];
            lower_reduce_node(&mut out, input, *op, axes, *keep_dim, node.shape.clone())
        } else {
            out.add_node(node.op.clone(), inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }
    let outputs: Vec<NodeId> = graph.outputs.iter().map(|id| id_map[id]).collect();
    out.set_outputs(outputs);
    out
}

fn lower_reduce_node(
    g: &mut Graph,
    input: NodeId,
    op: ReduceOp,
    axes: &[usize],
    keep_dim: bool,
    out_shape: Shape,
) -> NodeId {
    let rank = g.shape(input).rank();
    let mut axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if (a as i32) < 0 {
                (rank as i32 + a as i32) as usize
            } else {
                a
            }
        })
        .collect();
    axes.sort_unstable();
    axes.dedup();
    if axes.is_empty() {
        return input;
    }
    if axes.len() == 1 && axes[0] == rank - 1 {
        return g.reduce(input, op, axes, keep_dim, out_shape);
    }
    axes.sort_unstable_by(|a, b| b.cmp(a));
    let mut h = input;
    for (step, &ax) in axes.iter().enumerate() {
        let last = step + 1 == axes.len();
        let kd = last && keep_dim;
        h = reduce_one_axis(g, h, op, ax, kd);
    }
    if g.shape(h) != &out_shape {
        let dims: Vec<i64> = out_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        h = g.reshape_(h, dims);
    }
    h
}

fn optimize_backward_graph(graph: Graph) -> Graph {
    let before = graph.len();
    let g = fold_identity_transpose(graph);
    let g = fold_inverse_transpose_pairs(g);
    let g = fold_identity_reshape(g);
    let g = fold_consecutive_reshape(g);
    let g = fold_identity_cast(g);
    let g = fold_identity_narrow(g);
    let g = fold_consecutive_narrow(g);
    let g = fold_concat_of_contiguous_narrows(g);
    let g = fold_single_input_concat(g);
    if graph_opt_verbose() {
        eprintln!("[backward_prep] graph opt: {} → {} nodes", before, g.len());
    }
    g
}

fn graph_opt_verbose() -> bool {
    std::env::var("RLX_VOXTRAL_TTS_TRAIN_BACKWARD_GRAPH_OPT_VERBOSE")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"))
}

/// Drop no-op `reshape` when the output shape matches the input.
fn fold_identity_reshape(graph: Graph) -> Graph {
    let consumers = consumer_counts(&graph);
    let mut bypass: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let Op::Reshape { .. } = &node.op else {
            continue;
        };
        if node.inputs.len() != 1 {
            continue;
        }
        if consumers.get(&node.id) != Some(&1) {
            continue;
        }
        let x = node.inputs[0];
        if graph.shape(x) == &node.shape {
            bypass.insert(node.id, x);
        }
    }

    if bypass.is_empty() {
        return graph;
    }
    rebuild_with_bypass(graph, bypass, "_reshape_id")
}

/// `reshape(reshape(x))` with single-use middle → one reshape from `x`.
fn fold_consecutive_reshape(graph: Graph) -> Graph {
    let consumers = consumer_counts(&graph);
    let mut bypass: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let Op::Reshape { .. } = &node.op else {
            continue;
        };
        if node.inputs.len() != 1 {
            continue;
        }
        let mid = node.inputs[0];
        if consumers.get(&mid) != Some(&1) {
            continue;
        }
        let mid_node = graph.node(mid);
        let Op::Reshape { .. } = &mid_node.op else {
            continue;
        };
        if mid_node.inputs.len() != 1 {
            continue;
        }
        bypass.insert(mid, mid_node.inputs[0]);
    }

    if bypass.is_empty() {
        return graph;
    }
    rebuild_with_bypass(graph, bypass, "_reshape_fold")
}

/// Drop no-op `cast` when input and output dtype match.
fn fold_identity_cast(graph: Graph) -> Graph {
    let consumers = consumer_counts(&graph);
    let mut bypass: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let Op::Cast { to } = &node.op else {
            continue;
        };
        if node.inputs.len() != 1 {
            continue;
        }
        if consumers.get(&node.id) != Some(&1) {
            continue;
        }
        let x = node.inputs[0];
        if graph.shape(x).dtype() == *to {
            bypass.insert(node.id, x);
        }
    }

    if bypass.is_empty() {
        return graph;
    }
    rebuild_with_bypass(graph, bypass, "_cast_id")
}

/// Drop `narrow` when the slice spans the full axis (equivalent to a copy).
fn fold_identity_narrow(graph: Graph) -> Graph {
    let consumers = consumer_counts(&graph);
    let mut bypass: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let Op::Narrow { axis, start, len } = &node.op else {
            continue;
        };
        if node.inputs.len() != 1 || *start != 0 {
            continue;
        }
        if consumers.get(&node.id) != Some(&1) {
            continue;
        }
        let x = node.inputs[0];
        let rank = graph.shape(x).rank();
        let ax = norm_axis(*axis, rank);
        if graph.shape(x).dim(ax).unwrap_static() == *len && graph.shape(x) == &node.shape {
            bypass.insert(node.id, x);
        }
    }

    if bypass.is_empty() {
        return graph;
    }
    rebuild_with_bypass(graph, bypass, "_narrow_id")
}

fn norm_axis(axis: usize, rank: usize) -> usize {
    if (axis as i32) < 0 {
        (rank as i32 + axis as i32) as usize
    } else {
        axis
    }
}

fn consumer_counts(graph: &Graph) -> HashMap<NodeId, usize> {
    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    for node in graph.nodes() {
        for &input in &node.inputs {
            *consumers.entry(input).or_insert(0) += 1;
        }
    }
    for &out in &graph.outputs {
        *consumers.entry(out).or_insert(0) += 1;
    }
    consumers
}

fn inverse_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

fn is_inverse_perm(perm_a: &[usize], perm_b: &[usize]) -> bool {
    perm_a.len() == perm_b.len() && perm_b == inverse_perm(perm_a)
}

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| p == i)
}

/// Drop no-op `transpose` when `perm` is the identity.
fn fold_identity_transpose(graph: Graph) -> Graph {
    let consumers = consumer_counts(&graph);
    let mut bypass: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let Op::Transpose { perm } = &node.op else {
            continue;
        };
        if !is_identity_perm(perm) || node.inputs.len() != 1 {
            continue;
        }
        if consumers.get(&node.id) != Some(&1) {
            continue;
        }
        bypass.insert(node.id, node.inputs[0]);
    }

    if bypass.is_empty() {
        return graph;
    }
    rebuild_with_bypass(graph, bypass, "_transpose_id")
}

/// Drop `transpose(P); transpose(inv(P))` when the middle value has a single use.
fn fold_inverse_transpose_pairs(graph: Graph) -> Graph {
    let consumers = consumer_counts(&graph);
    let mut bypass: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let Op::Transpose { perm: perm_b } = &node.op else {
            continue;
        };
        if node.inputs.len() != 1 {
            continue;
        }
        let mid = node.inputs[0];
        if consumers.get(&mid) != Some(&1) {
            continue;
        }
        let mid_node = graph.node(mid);
        let Op::Transpose { perm: perm_a } = &mid_node.op else {
            continue;
        };
        if mid_node.inputs.len() != 1 || !is_inverse_perm(perm_a, perm_b) {
            continue;
        }
        let src = mid_node.inputs[0];
        bypass.insert(mid, src);
        bypass.insert(node.id, src);
    }

    if bypass.is_empty() {
        return graph;
    }
    rebuild_with_bypass(graph, bypass, "_transpose_fold")
}

fn fold_consecutive_narrow(graph: Graph) -> Graph {
    let consumers = consumer_counts(&graph);
    let mut bypass: HashMap<NodeId, NodeId> = HashMap::new();
    let mut fused_narrow: HashMap<NodeId, (usize, usize, usize)> = HashMap::new();

    for node in graph.nodes() {
        let Op::Narrow {
            axis: ax2,
            start: s2,
            len: l2,
        } = &node.op
        else {
            continue;
        };
        if node.inputs.len() != 1 {
            continue;
        }
        let mid = node.inputs[0];
        if consumers.get(&mid) != Some(&1) {
            continue;
        }
        let mid_node = graph.node(mid);
        let Op::Narrow {
            axis: ax1,
            start: s1,
            len: l1,
        } = &mid_node.op
        else {
            continue;
        };
        if mid_node.inputs.len() != 1 || ax1 != ax2 {
            continue;
        }
        if s2 + l2 > *l1 {
            continue;
        }
        let src = mid_node.inputs[0];
        bypass.insert(mid, src);
        bypass.insert(node.id, src);
        fused_narrow.insert(node.id, (*ax1, s1 + s2, *l2));
    }

    if bypass.is_empty() {
        return graph;
    }
    rebuild_with_bypass_and_narrow(graph, bypass, fused_narrow, "_narrow_fold")
}

/// `concat([narrow(x, s_i, l_i), …])` that reassembles the full axis of `x` → `x`.
fn fold_concat_of_contiguous_narrows(graph: Graph) -> Graph {
    let consumers = consumer_counts(&graph);
    let mut bypass: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let Op::Concat { axis } = &node.op else {
            continue;
        };
        if node.inputs.is_empty() {
            continue;
        }
        let rank = node.shape.rank();
        let ax = norm_axis(*axis, rank);

        let mut src: Option<NodeId> = None;
        let mut slices: Vec<(usize, usize)> = Vec::with_capacity(node.inputs.len());
        let mut ok = true;

        for &inp in &node.inputs {
            if consumers.get(&inp) != Some(&1) {
                ok = false;
                break;
            }
            let narrow = graph.node(inp);
            let Op::Narrow {
                axis: nax,
                start,
                len,
            } = &narrow.op
            else {
                ok = false;
                break;
            };
            if narrow.inputs.len() != 1 {
                ok = false;
                break;
            }
            let x = narrow.inputs[0];
            let x_rank = graph.shape(x).rank();
            if norm_axis(*nax, x_rank) != ax {
                ok = false;
                break;
            }
            match src {
                None => src = Some(x),
                Some(s) if s == x => {}
                _ => {
                    ok = false;
                    break;
                }
            }
            slices.push((*start, *len));
        }
        if !ok {
            continue;
        }
        let src = src.expect("non-empty concat inputs");
        slices.sort_unstable_by_key(|(s, _)| *s);
        let mut end = 0usize;
        for (start, len) in &slices {
            if *start != end {
                ok = false;
                break;
            }
            end += len;
        }
        if !ok {
            continue;
        }
        let full = graph.shape(src).dim(ax).unwrap_static();
        if end != full || graph.shape(src) != &node.shape {
            continue;
        }

        bypass.insert(node.id, src);
        for &inp in &node.inputs {
            bypass.insert(inp, src);
        }
    }

    if bypass.is_empty() {
        return graph;
    }
    rebuild_with_bypass(graph, bypass, "_concat_fold")
}

/// `concat([x])` → `x`.
fn fold_single_input_concat(graph: Graph) -> Graph {
    let consumers = consumer_counts(&graph);
    let mut bypass: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let Op::Concat { .. } = &node.op else {
            continue;
        };
        if node.inputs.len() != 1 {
            continue;
        }
        if consumers.get(&node.id) != Some(&1) {
            continue;
        }
        bypass.insert(node.id, node.inputs[0]);
    }

    if bypass.is_empty() {
        return graph;
    }
    rebuild_with_bypass(graph, bypass, "_concat1")
}

fn effective_source(id: NodeId, bypass: &HashMap<NodeId, NodeId>) -> NodeId {
    let mut cur = id;
    while let Some(&next) = bypass.get(&cur) {
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

fn rebuild_with_bypass(graph: Graph, bypass: HashMap<NodeId, NodeId>, suffix: &str) -> Graph {
    rebuild_with_bypass_and_narrow(graph, bypass, HashMap::new(), suffix)
}

fn rebuild_with_bypass_and_narrow(
    graph: Graph,
    bypass: HashMap<NodeId, NodeId>,
    fused_narrow: HashMap<NodeId, (usize, usize, usize)>,
    suffix: &str,
) -> Graph {
    let mut out = Graph::new(format!("{}{}", graph.name, suffix));
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        if let Some(&(axis, start, len)) = fused_narrow.get(&node.id) {
            let src = bypass
                .get(&node.id)
                .copied()
                .expect("fused narrow must bypass to source");
            let input = id_map[&effective_source(src, &bypass)];
            let new_id = out.add_node(
                Op::Narrow { axis, start, len },
                vec![input],
                node.shape.clone(),
            );
            id_map.insert(node.id, new_id);
            continue;
        }
        if bypass.contains_key(&node.id) {
            let src = effective_source(node.id, &bypass);
            id_map.insert(node.id, id_map[&effective_source(src, &bypass)]);
            continue;
        }
        let inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|i| id_map[&effective_source(*i, &bypass)])
            .collect();
        let new_id = out.add_node(node.op.clone(), inputs, node.shape.clone());
        id_map.insert(node.id, new_id);
    }

    let outputs: Vec<NodeId> = graph.outputs.iter().map(|id| id_map[id]).collect();
    out.set_outputs(outputs);
    out
}

fn reduce_one_axis(
    g: &mut Graph,
    input: NodeId,
    op: ReduceOp,
    axis: usize,
    keep_dim: bool,
) -> NodeId {
    let in_shape = g.shape(input).clone();
    let rank = in_shape.rank();
    assert!(axis < rank);
    let mut out_dims: Vec<i64> = in_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    if keep_dim {
        out_dims[axis] = 1;
    } else {
        out_dims.remove(axis);
    }
    let out_shape = Shape::new(
        &out_dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        in_shape.dtype(),
    );
    g.reduce(input, op, vec![axis], keep_dim, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;

    fn f32_shape(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn fold_identity_transpose_to_input() {
        let mut g = Graph::new("t");
        let x = g.input("x", f32_shape(&[4, 8]));
        let t = g.add_node(
            Op::Transpose { perm: vec![0, 1] },
            vec![x],
            f32_shape(&[4, 8]),
        );
        g.set_outputs(vec![t]);

        let folded = super::fold_identity_transpose(g);
        assert_eq!(folded.len(), 1);
        assert!(matches!(
            folded.node(folded.outputs[0]).op,
            Op::Input { .. }
        ));
    }

    #[test]
    fn fold_inverse_transpose_pair() {
        let mut g = Graph::new("t");
        let x = g.input("x", f32_shape(&[4, 8]));
        let t1 = g.add_node(
            Op::Transpose { perm: vec![1, 0] },
            vec![x],
            f32_shape(&[8, 4]),
        );
        let t2 = g.add_node(
            Op::Transpose { perm: vec![1, 0] },
            vec![t1],
            f32_shape(&[4, 8]),
        );
        g.set_outputs(vec![t2]);

        let folded = fold_inverse_transpose_pairs(g);
        assert_eq!(folded.len(), 1, "transpose pair should fold to input");
        assert!(matches!(
            folded.node(folded.outputs[0]).op,
            Op::Input { .. }
        ));
    }

    #[test]
    fn fold_consecutive_narrow_same_axis() {
        let mut g = Graph::new("t");
        let x = g.input("x", f32_shape(&[2, 16]));
        let n1 = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 2,
                len: 10,
            },
            vec![x],
            f32_shape(&[2, 10]),
        );
        let n2 = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 3,
                len: 4,
            },
            vec![n1],
            f32_shape(&[2, 4]),
        );
        g.set_outputs(vec![n2]);

        let folded = fold_consecutive_narrow(g);
        let out = folded.outputs[0];
        match &folded.node(out).op {
            Op::Narrow { axis, start, len } => {
                assert_eq!(*axis, 1);
                assert_eq!(*start, 5);
                assert_eq!(*len, 4);
            }
            other => panic!("expected fused narrow, got {other:?}"),
        }
    }

    #[test]
    fn fold_concat_of_adjacent_narrows_to_input() {
        let mut g = Graph::new("t");
        let x = g.input("x", f32_shape(&[2, 12]));
        let n0 = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 0,
                len: 5,
            },
            vec![x],
            f32_shape(&[2, 5]),
        );
        let n1 = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 5,
                len: 7,
            },
            vec![x],
            f32_shape(&[2, 7]),
        );
        let c = g.add_node(Op::Concat { axis: 1 }, vec![n0, n1], f32_shape(&[2, 12]));
        g.set_outputs(vec![c]);

        let folded = fold_concat_of_contiguous_narrows(g);
        assert_eq!(folded.len(), 1);
        assert!(matches!(
            folded.node(folded.outputs[0]).op,
            Op::Input { .. }
        ));
    }

    #[test]
    fn fold_single_input_concat_to_input() {
        let mut g = Graph::new("t");
        let x = g.input("x", f32_shape(&[2, 8]));
        let c = g.add_node(Op::Concat { axis: 1 }, vec![x], f32_shape(&[2, 8]));
        g.set_outputs(vec![c]);

        let folded = fold_single_input_concat(g);
        assert_eq!(folded.len(), 1);
        assert!(matches!(
            folded.node(folded.outputs[0]).op,
            Op::Input { .. }
        ));
    }

    #[test]
    fn fold_consecutive_reshape_chain() {
        let mut g = Graph::new("t");
        let x = g.input("x", f32_shape(&[2, 12]));
        let r1 = g.add_node(
            Op::Reshape {
                new_shape: vec![24],
            },
            vec![x],
            f32_shape(&[24]),
        );
        let r2 = g.add_node(
            Op::Reshape {
                new_shape: vec![2, 12],
            },
            vec![r1],
            f32_shape(&[2, 12]),
        );
        g.set_outputs(vec![r2]);

        let folded = fold_consecutive_reshape(g);
        let out = folded.outputs[0];
        let x_in = folded
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Input { .. }))
            .expect("input")
            .id;
        assert!(matches!(folded.node(out).op, Op::Reshape { .. }));
        assert_eq!(folded.node(out).inputs, vec![x_in]);
        assert_eq!(folded.shape(out), &f32_shape(&[2, 12]));
    }

    #[test]
    fn fold_identity_cast_to_input() {
        let mut g = Graph::new("t");
        let x = g.input("x", f32_shape(&[4, 8]));
        let c = g.add_node(Op::Cast { to: DType::F32 }, vec![x], f32_shape(&[4, 8]));
        g.set_outputs(vec![c]);

        let folded = fold_identity_cast(g);
        assert_eq!(folded.len(), 1);
        assert!(matches!(
            folded.node(folded.outputs[0]).op,
            Op::Input { .. }
        ));
    }
}
