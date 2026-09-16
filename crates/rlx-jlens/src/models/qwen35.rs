//! [`LensModel`] for the Qwen3.5 / Qwen3.6 hybrid trunk.
//!
//! One implementation of the interface, kept behind the `qwen35` feature so the
//! core crate depends on no model. Adding another model means another module
//! and another feature, not a change here.
//!
//! Qwen3.5/3.6 alternates two block types — a gated delta-net and full
//! attention, every `full_attention_interval`-th layer being attention — and
//! [`Qwen35LensModel::block_kind`] reports which, since the two have very
//! different residual dynamics and a lens over them is worth reading
//! separately.

use rlx_qwen35::{
    Qwen35Config, Qwen35Weights, build_qwen35_layer_probe_graph, build_qwen35_prefix_graph,
};

use crate::model::{BlockGraph, LensError, LensModel, Result, StackGraph, UnembedGraph};
use crate::taps::{layer_exit_taps, residual_stream};
use crate::vjp::{Tap, TappedGraph};

/// Which kind of block sits at a given layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Linear-attention scan with a recurrent state.
    GatedDeltaNet,
    /// Softmax attention.
    FullAttention,
}

/// A Qwen3.5/3.6 trunk, ready to be tapped.
pub struct Qwen35LensModel {
    cfg: Qwen35Config,
    weights: Qwen35Weights,
    name: String,
}

impl Qwen35LensModel {
    pub fn new(cfg: Qwen35Config, weights: Qwen35Weights) -> Self {
        Self {
            cfg,
            weights,
            name: "qwen35".to_string(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    /// Which block type sits at `layer`.
    pub fn block_kind(&self, layer: usize) -> BlockKind {
        let interval = self.cfg.full_attention_interval.max(1);
        if (layer + 1).is_multiple_of(interval) {
            BlockKind::FullAttention
        } else {
            BlockKind::GatedDeltaNet
        }
    }
}

/// The name the layer-probe graph gives its incoming residual.
pub const RESIDUAL_INPUT: &str = "trunk_h";

/// The name the trunk graph gives its token-id input.
pub const TOKEN_INPUT: &str = "input_ids";

/// The name the unembedding graph gives its residual input.
pub const UNEMBED_INPUT: &str = "resid";

impl LensModel for Qwen35LensModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_layers(&self) -> usize {
        self.weights.trunk_layers.len()
    }

    fn d_model(&self) -> usize {
        self.cfg.hidden_size
    }

    fn stack(
        &self,
        source_layers: &[usize],
        target_layer: usize,
        batch: usize,
        seq: usize,
    ) -> Result<StackGraph> {
        self.check_layer(target_layer)?;
        for &l in source_layers {
            self.check_layer(l)?;
            if l > target_layer {
                return Err(LensError::Other(anyhow::anyhow!(
                    "source layer {l} is after the target layer {target_layer};                      the lens transports forward, not backward"
                )));
            }
        }
        if source_layers.is_empty() {
            return Err(LensError::Other(anyhow::anyhow!(
                "stack needs at least one source layer"
            )));
        }

        // Trunk truncated after the target layer, so the graph's output *is*
        // the residual at that layer — no LM head, no final norm.
        let (graph, params, packed) = build_qwen35_prefix_graph(
            &self.cfg,
            self.weights.clone(),
            target_layer + 1,
            batch,
            seq,
        )
        .map_err(LensError::Other)?;
        if !packed.is_empty() {
            return Err(LensError::Unsupported {
                model: self.name.clone(),
                what: "packed/quantized parameters (needs set_param_typed feeding)".to_string(),
            });
        }

        let chain = residual_stream(&graph).map_err(LensError::Other)?;
        // Every trunk layer folds its attention/delta-net and its FFN back into
        // the stream, so the chain is `1 + 2·layers` long and layer `l` is
        // entered at `chain[2l]`. Derive it rather than hard-coding 2, so a
        // config that changes the block shape fails loudly here instead of
        // silently tapping the wrong point.
        let n_built = target_layer + 1;
        let joins = (chain.len() - 1) / n_built;
        if joins == 0 || 1 + joins * n_built != chain.len() {
            return Err(LensError::Other(anyhow::anyhow!(
                "residual chain has {} points for {n_built} layers, which is not                  1 + joins·layers for any whole `joins`",
                chain.len()
            )));
        }
        let tap_ids = layer_exit_taps(&chain, source_layers, joins).map_err(LensError::Other)?;
        let h_final = *chain.last().expect("chain is non-empty");

        let mut graph = graph;
        let mut outputs = vec![h_final];
        outputs.extend_from_slice(&tap_ids);
        graph.set_outputs(outputs);

        // `Wrt::Output` rather than raw ids: only output *positions* survive the
        // renumbering `prepare_graph_for_ad` performs.
        let taps: Vec<Tap> = source_layers
            .iter()
            .enumerate()
            .map(|(i, &layer)| Tap::at_output(layer, i + 1))
            .collect();
        let tapped = TappedGraph::new(graph, taps).map_err(LensError::Other)?;

        Ok(StackGraph {
            tapped,
            params,
            token_input: TOKEN_INPUT.to_string(),
            extra_feeds: Vec::new(),
            layers: source_layers.to_vec(),
            batch,
            seq,
        })
    }

    fn unembed(&self, rows: usize) -> Result<UnembedGraph> {
        use rlx_ir::{DType, Graph, Op, Shape};

        let d = self.cfg.hidden_size;
        let vocab = self.weights.lm_vocab_size(&self.cfg);
        let f = DType::F32;
        let mut params: crate::model::Params = std::collections::HashMap::new();

        let mut g = Graph::new("qwen35_unembed");
        let x = g.input(UNEMBED_INPUT, Shape::new(&[rows, d], f));

        // Final RMSNorm. Qwen3.5 GGUFs store the norm weight directly (no `-1`
        // unbake, unlike Gemma), and there is no beta.
        let gamma = g.param("output_norm.weight", Shape::new(&[d], f));
        params.insert(
            "output_norm.weight".into(),
            self.weights.output_norm.clone(),
        );
        let beta = g.param("output_norm.beta", Shape::new(&[d], f));
        params.insert("output_norm.beta".into(), vec![0.0; d]);
        let normed = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: self.cfg.rms_norm_eps as f32,
            },
            vec![x, gamma, beta],
            Shape::new(&[rows, d], f),
        );

        // LM head. `output` when the checkpoint ships an untied head, else the
        // embedding table transposed — the tie Qwen3.5 uses at this size.
        let head: Vec<f32> = match &self.weights.output {
            Some(rlx_qwen35::MatWeight::F32(w)) if !w.is_empty() => {
                // Stored `[vocab, d]`; the matmul wants `[d, vocab]`.
                let mut t = vec![0.0f32; d * vocab];
                for r in 0..vocab {
                    for c in 0..d {
                        t[c * vocab + r] = w[r * d + c];
                    }
                }
                t
            }
            _ => {
                let embd = &self.weights.token_embd;
                if embd.len() < vocab * d {
                    return Err(LensError::Other(anyhow::anyhow!(
                        "tied LM head needs {}·{d} embedding entries, found {}",
                        vocab,
                        embd.len()
                    )));
                }
                let mut t = vec![0.0f32; d * vocab];
                for r in 0..vocab {
                    for c in 0..d {
                        t[c * vocab + r] = embd[r * d + c];
                    }
                }
                t
            }
        };
        let w = g.param("lm_head.t", Shape::new(&[d, vocab], f));
        params.insert("lm_head.t".into(), head);
        let logits = g.matmul(normed, w, Shape::new(&[rows, vocab], f));
        g.set_outputs(vec![logits]);

        Ok(UnembedGraph {
            graph: g,
            params,
            residual_input: UNEMBED_INPUT.to_string(),
            rows,
            vocab,
        })
    }

    fn block(&self, layer: usize, batch: usize, seq: usize) -> Result<BlockGraph> {
        self.check_layer(layer)?;
        let (graph, params, packed) = build_qwen35_layer_probe_graph(
            &self.cfg,
            self.weights.clone(),
            layer,
            batch,
            seq,
            false,
        )
        .map_err(LensError::Other)?;
        if !packed.is_empty() {
            // Packed (quantized) parameters need `set_param_typed`, which this
            // path does not yet feed — fail rather than run on zeroed weights.
            return Err(LensError::Unsupported {
                model: self.name.clone(),
                what: "packed/quantized parameters (needs set_param_typed feeding)".to_string(),
            });
        }
        Ok(BlockGraph {
            graph,
            params,
            residual_input: RESIDUAL_INPUT.to_string(),
            batch,
            seq,
        })
    }
}
