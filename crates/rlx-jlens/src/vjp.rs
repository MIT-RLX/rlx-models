//! Turning a model's forward graph into a lens VJP graph.
//!
//! A [`TappedGraph`] is a forward graph whose `outputs[0]` is the tensor the
//! cotangent is seeded on — `h_final`, the residual at the target layer — and
//! whose remaining outputs are the residual-stream cut points to differentiate
//! with respect to.
//!
//! [`TappedGraph::vjp_graph`] compiles that into a backward graph with a
//! `d_output` input: seed it with a cotangent and every tap's
//! `∂h_final/∂h_l` comes back in one pass.

use anyhow::{Result, ensure};
use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::Graph;

/// One residual-stream cut point.
#[derive(Debug, Clone)]
pub struct Tap {
    /// Layer index this tap reads, as the model numbers its layers.
    pub layer: usize,
    /// How to address the tensor across `prepare_graph_for_ad` renumbering.
    pub wrt: Wrt,
}

impl Tap {
    /// A tap on an interior activation published as `graph.outputs[output_index]`.
    pub fn at_output(layer: usize, output_index: usize) -> Self {
        Self {
            layer,
            wrt: Wrt::Output(output_index),
        }
    }

    /// A tap on a named graph input — the residual entering a single-block
    /// graph, for instance.
    pub fn at_input(layer: usize, name: impl Into<String>) -> Self {
        Self {
            layer,
            wrt: Wrt::Leaf(name.into()),
        }
    }
}

/// A forward graph prepared for lens fitting.
#[derive(Debug)]
pub struct TappedGraph {
    graph: Graph,
    taps: Vec<Tap>,
}

impl TappedGraph {
    /// `graph.outputs[0]` must be the residual at the target layer; taps name
    /// the source-layer residuals.
    pub fn new(graph: Graph, taps: Vec<Tap>) -> Result<Self> {
        ensure!(!taps.is_empty(), "a lens needs at least one tap");
        ensure!(
            !graph.outputs.is_empty(),
            "forward graph has no outputs; outputs[0] must be the target-layer residual"
        );
        for tap in &taps {
            if let Wrt::Output(idx) = &tap.wrt {
                ensure!(
                    *idx < graph.outputs.len(),
                    "tap for layer {} names output {idx}, but the graph has {} outputs",
                    tap.layer,
                    graph.outputs.len()
                );
                ensure!(
                    *idx != 0,
                    "tap for layer {} names outputs[0], which is the target-layer \
                     residual being differentiated, not a source",
                    tap.layer
                );
            }
        }
        Ok(Self { graph, taps })
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn taps(&self) -> &[Tap] {
        &self.taps
    }

    /// Layer indices, in tap order — the order gradients come back in.
    pub fn layers(&self) -> Vec<usize> {
        self.taps.iter().map(|t| t.layer).collect()
    }

    /// Build the backward graph.
    ///
    /// Its outputs are `[h_final, ∂h_final/∂h_tap0, ∂h_final/∂h_tap1, …]`, and
    /// it takes a `d_output` input shaped like `h_final`. Aux mirroring is off:
    /// the taps exist to *designate* the cut points, and reading their values
    /// back on every pass would cost a full residual tensor per tap for nothing.
    pub fn vjp_graph(&self) -> Graph {
        let wrt: Vec<Wrt> = self.taps.iter().map(|t| t.wrt.clone()).collect();
        grad_with_loss_wrt(
            &self.graph,
            &wrt,
            GradWithLossOptions::STRICT.with_aux(false),
        )
    }

    /// Index into the VJP graph's outputs for tap `i`. Output 0 is `h_final`.
    pub fn grad_output_index(&self, tap: usize) -> usize {
        tap + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Shape};

    fn two_output_graph() -> Graph {
        let f = DType::F32;
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[2], f));
        let w = g.param("w", Shape::new(&[2], f));
        let h = g.binary(BinaryOp::Mul, x, w, Shape::new(&[2], f));
        let y = g.binary(BinaryOp::Add, h, h, Shape::new(&[2], f));
        g.set_outputs(vec![y, h]);
        g
    }

    #[test]
    fn rejects_empty_taps() {
        assert!(TappedGraph::new(two_output_graph(), vec![]).is_err());
    }

    #[test]
    fn rejects_out_of_range_tap() {
        let err = TappedGraph::new(two_output_graph(), vec![Tap::at_output(0, 5)]).unwrap_err();
        assert!(err.to_string().contains("names output 5"), "{err}");
    }

    /// Tapping `outputs[0]` would differentiate the target w.r.t. itself and
    /// silently return the seed — a confusing identity rather than an error.
    #[test]
    fn rejects_tap_on_the_target_itself() {
        let err = TappedGraph::new(two_output_graph(), vec![Tap::at_output(0, 0)]).unwrap_err();
        assert!(err.to_string().contains("outputs[0]"), "{err}");
    }

    #[test]
    fn vjp_graph_emits_target_then_one_gradient_per_tap() {
        let tapped = TappedGraph::new(two_output_graph(), vec![Tap::at_output(0, 1)]).unwrap();
        let bwd = tapped.vjp_graph();
        // [y, dy/dh] — the `h` mirror is suppressed by with_aux(false).
        assert_eq!(bwd.outputs.len(), 2);
        assert_eq!(tapped.grad_output_index(0), 1);

        let mut compiled = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(bwd);
        compiled.set_param("w", &[1.0, 1.0]);
        let outs = compiled.run(&[("x", &[3.0f32, 5.0][..]), ("d_output", &[1.0f32, 1.0])]);
        // y = 2h ⇒ dy/dh = 2.
        assert_eq!(outs[1], vec![2.0, 2.0]);
    }
}
