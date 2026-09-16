//! Finding the residual stream in a decoder graph.
//!
//! A lens taps the residual stream, so it has to locate it. Doing that
//! structurally — rather than asking each model crate to hand over node ids —
//! is what keeps [`crate::model::LensModel`] small: a model supplies a graph,
//! and the chain is read off it.
//!
//! The stream is the spine of a decoder: an embedding, then one `Add` per
//! sublayer folding that sublayer's delta back in. Walking *backwards* from the
//! final residual through those adds recovers the whole chain, and the residual
//! operand is the one whose shape matches the sum's — the delta branch has the
//! same shape too, so the tie is broken by which side continues the chain.
//!
//! This is deliberately narrow. It recognizes the `Add`-spine shape that
//! Qwen/Llama-family decoders have; a model whose residual is not built that
//! way should override [`crate::model::LensModel::stack`] and name its taps
//! directly.

use anyhow::{Result, bail};
use rlx_ir::op::BinaryOp;
use rlx_ir::{Graph, NodeId, Op, Shape};

/// The residual stream, earliest first.
///
/// `chain[0]` is where the stream starts (the embedding, or whatever feeds the
/// first residual add) and the last element is `from`. For a decoder with `L`
/// layers and `j` residual joins per layer the chain has `1 + j·L` entries.
pub fn residual_chain(graph: &Graph, from: NodeId) -> Result<Vec<NodeId>> {
    let shape = graph.node(from).shape.clone();
    let mut chain = vec![from];
    let mut cur = from;
    loop {
        let node = graph.node(cur);
        let Op::Binary(BinaryOp::Add) = &node.op else {
            break;
        };
        if node.inputs.len() != 2 {
            break;
        }
        // Both operands of a residual add have the residual's shape, so shape
        // alone cannot say which one continues the stream. Prefer the operand
        // that is itself an add — the previous join — and fall back to
        // `inputs[0]`, which is the residual by construction in every builder
        // this handles.
        let candidates: Vec<NodeId> = node
            .inputs
            .iter()
            .copied()
            .filter(|&i| graph.node(i).shape == shape)
            .collect();
        let next = match candidates.len() {
            0 => break,
            1 => candidates[0],
            _ => *candidates
                .iter()
                .find(|&&i| matches!(graph.node(i).op, Op::Binary(BinaryOp::Add)))
                .unwrap_or(&node.inputs[0]),
        };
        if next == cur {
            bail!("residual chain: {cur} is its own predecessor");
        }
        chain.push(next);
        cur = next;
    }
    chain.reverse();
    Ok(chain)
}

/// [`residual_chain`] from the graph's single output.
pub fn residual_stream(graph: &Graph) -> Result<Vec<NodeId>> {
    let [out] = graph.outputs[..] else {
        bail!(
            "residual_stream expects exactly one graph output, got {}; \
             call residual_chain with the residual output explicitly",
            graph.outputs.len()
        );
    };
    residual_stream_from(graph, out)
}

/// [`residual_chain`] with the shape and length checks a lens wants.
pub fn residual_stream_from(graph: &Graph, from: NodeId) -> Result<Vec<NodeId>> {
    let chain = residual_chain(graph, from)?;
    if chain.len() < 2 {
        bail!(
            "no residual stream found at {from}: the chain is {} node(s) long, so \
             the graph does not have the add-spine shape this recognizes",
            chain.len()
        );
    }
    Ok(chain)
}

/// Select one tap per layer from a residual chain, at each layer's **exit**.
///
/// With `joins_per_layer` residual adds per layer, layer `l` is entered at
/// `chain[joins_per_layer · l]` and *left* at `chain[joins_per_layer · (l + 1)]`.
/// This returns the latter, which is the convention the Python reference uses —
/// it hooks each block's forward **output** — and therefore the one a lens file
/// must follow to be interchangeable with it.
///
/// The distinction is a whole layer of index, not a detail: taking entry
/// residuals instead makes this crate's `J_l` equal the reference's `J_{l-1}`.
/// That was measured, not assumed — see `tests/reference_parity.rs`, where the
/// exit convention agrees with the reference to 2e-4 (the f16 storage floor) and
/// the entry convention disagrees by 0.67.
pub fn layer_exit_taps(
    chain: &[NodeId],
    layers: &[usize],
    joins_per_layer: usize,
) -> Result<Vec<NodeId>> {
    if joins_per_layer == 0 {
        bail!("joins_per_layer must be non-zero");
    }
    layers
        .iter()
        .map(|&l| {
            let idx = joins_per_layer * (l + 1);
            chain.get(idx).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "layer {l} exit is chain index {idx}, but the residual chain has \
                     only {} entries ({} joins per layer)",
                    chain.len(),
                    joins_per_layer
                )
            })
        })
        .collect()
}

/// The residual shape a chain carries, for cross-checking against a model's
/// declared `d_model`.
pub fn chain_shape(graph: &Graph, chain: &[NodeId]) -> Option<Shape> {
    chain.first().map(|&id| graph.node(id).shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    /// `embed → (+d0) → (+d1) → (+d2)`: a 3-join spine.
    fn spine(joins: usize) -> (Graph, Vec<NodeId>) {
        let f = DType::F32;
        let s = Shape::new(&[1, 4, 8], f);
        let mut g = Graph::new("spine");
        let embed = g.input("embed", s.clone());
        let mut chain = vec![embed];
        let mut cur = embed;
        for i in 0..joins {
            let delta = g.input(format!("delta{i}"), s.clone());
            cur = g.binary(BinaryOp::Add, cur, delta, s.clone());
            chain.push(cur);
        }
        g.set_outputs(vec![cur]);
        (g, chain)
    }

    #[test]
    fn walks_the_whole_spine_in_order() {
        let (g, expected) = spine(6);
        assert_eq!(residual_stream(&g).unwrap(), expected);
    }

    #[test]
    fn a_graph_without_a_spine_is_rejected() {
        let f = DType::F32;
        let mut g = Graph::new("flat");
        let x = g.input("x", Shape::new(&[4], f));
        g.set_outputs(vec![x]);
        assert!(residual_stream(&g).is_err());
    }

    #[test]
    fn multiple_outputs_need_an_explicit_start() {
        let (mut g, chain) = spine(4);
        let last = *chain.last().unwrap();
        g.set_outputs(vec![last, chain[0]]);
        assert!(residual_stream(&g).is_err());
        assert_eq!(residual_stream_from(&g, last).unwrap(), chain);
    }

    #[test]
    fn layer_exits_step_by_joins_per_layer() {
        let (g, _chain) = spine(6); // 3 layers × 2 joins
        let found = residual_stream(&g).unwrap();
        assert_eq!(found.len(), 7);
        // Layer `l` leaves at chain index `joins·(l+1)`: 2, 4, 6.
        let taps = layer_exit_taps(&found, &[0, 1, 2], 2).unwrap();
        assert_eq!(taps, vec![found[2], found[4], found[6]]);
    }

    #[test]
    fn a_layer_past_the_end_is_an_error() {
        let (g, _) = spine(6);
        let found = residual_stream(&g).unwrap();
        // Layer 4 would leave at chain index 10, past the end of a 3-layer
        // model's chain, so this must be caught rather than silently clamped.
        assert!(layer_exit_taps(&found, &[4], 2).is_err());
    }
}
