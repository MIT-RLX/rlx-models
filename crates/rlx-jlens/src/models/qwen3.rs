//! [`LensModel`] for a plain Qwen3 trunk.
//!
//! The second implementation of the interface, and the reason for it is not
//! coverage — it is that a trait with one implementor is an untested guess. This
//! one is a dense attention-only decoder loaded from **HF safetensors**, where
//! [`super::qwen35`] is a delta-net/attention hybrid loaded from GGUF, so
//! between them they exercise both axes the interface was meant to abstract.
//!
//! It also unblocks comparing against the Python reference: `weights/Qwen3-0.6B`
//! ships an f32/bf16 `model.safetensors` that `transformers` and rlx can both
//! read, where the Qwen3.5 checkpoint on hand is quantized and would only ever
//! give the Jacobian *of the quantized model*.

use std::path::{Path, PathBuf};

use rlx_core::SafetensorsMmapLoader;
use rlx_core::weight_loader::WeightLoader;
use rlx_qwen3::{Qwen3Config, build_qwen3_graph_sized};

use crate::model::{BlockGraph, LensError, LensModel, Params, Result, StackGraph, UnembedGraph};
use crate::taps::{layer_exit_taps, residual_stream_from};
use crate::vjp::{Tap, TappedGraph};

const TOKEN_INPUT: &str = "input_ids";
const UNEMBED_INPUT: &str = "residual";

/// A Qwen3 trunk, ready to be tapped.
///
/// Holds the checkpoint *directory* rather than materialized weights: the graph
/// builders want a `WeightLoader`, and reopening the mmap per graph keeps one
/// copy of the checkpoint resident instead of two.
pub struct Qwen3LensModel {
    cfg: Qwen3Config,
    dir: PathBuf,
    name: String,
    /// Take an already-embedded `[batch, seq, hidden]` sequence instead of
    /// token ids.
    embeds_input: bool,
}

impl Qwen3LensModel {
    /// Open an HF checkpoint directory (`config.json` + `model.safetensors`).
    pub fn open(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let cfg = Qwen3Config::from_file(&dir.join("config.json"))?;
        Ok(Self {
            cfg,
            dir,
            name: "qwen3".to_string(),
            embeds_input: false,
        })
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Feed the trunk `inputs_embeds` rather than `input_ids`.
    ///
    /// This is what makes a vision-language model tappable without new
    /// machinery. A Qwen-VL projects image patches to the LM's hidden width —
    /// `projector_output_dim == LM hidden` is enforced by the runner — and
    /// splices them into the token sequence, so image content travels the *LM's*
    /// residual stream as ordinary positions. Tapping that stream reads image
    /// patches out through the LM's own vocabulary, with no caption list to
    /// choose and none of the rank collapse a ViT trunk shows.
    pub fn with_embeds_input(mut self, yes: bool) -> Self {
        self.embeds_input = yes;
        self
    }

    /// Name of the trunk's input, which depends on that choice.
    pub fn input_name(&self) -> &'static str {
        if self.embeds_input {
            "inputs_embeds"
        } else {
            TOKEN_INPUT
        }
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }

    fn check_layer(&self, layer: usize) -> Result<()> {
        if layer >= self.cfg.num_hidden_layers {
            return Err(LensError::LayerOutOfRange {
                layer,
                n_layers: self.cfg.num_hidden_layers,
            });
        }
        Ok(())
    }

    fn loader(&self) -> Result<SafetensorsMmapLoader> {
        SafetensorsMmapLoader::open(&self.dir)
            .map_err(|e| LensError::Other(anyhow::anyhow!("opening {}: {e}", self.dir.display())))
    }
}

impl LensModel for Qwen3LensModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
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
                    "source layer {l} is after the target layer {target_layer}; \
                     the lens transports forward, not backward"
                )));
            }
        }
        if source_layers.is_empty() {
            return Err(LensError::Other(anyhow::anyhow!(
                "stack needs at least one source layer"
            )));
        }

        // The whole trunk, no LM head. Unlike the Qwen3.5 model there is no
        // prefix builder to truncate at a layer, so the graph is built in full
        // and the residual at `target_layer` is published as the output —
        // everything after it is unreachable and gets dropped by DCE.
        let mut loader = self.loader()?;
        let (graph, params) = if self.embeds_input {
            let opts = rlx_qwen3::flow::Qwen3PrefillOpts {
                batch,
                seq,
                with_lm_head: false,
                with_kv_outputs: false,
                with_qk_outputs: false,
                last_logits_only: false,
                last_token_from_input: false,
                packed: false,
                profile: None,
                rope_cos: None,
                rope_sin: None,
            };
            let built =
                rlx_qwen3::flow::build_qwen3_prefill_embeds_built(&self.cfg, &mut loader, &opts)
                    .map_err(LensError::Other)?;
            rlx_core::flow_util::graph_from_built(built).map_err(LensError::Other)?
        } else {
            build_qwen3_graph_sized(&self.cfg, &mut loader, batch, seq, false, false)
                .map_err(LensError::Other)?
        };

        // The trunk builder ends with the final RmsNorm, but the lens transports
        // *residuals* — and the residual spine stops one node earlier. Walking
        // back from the graph output would find a norm, not an add, and report
        // a one-node chain. Qwen3.5 does not hit this because its prefix builder
        // stops at the residual; the difference is exactly the kind of thing a
        // second model exists to surface.
        let out = *graph
            .outputs
            .first()
            .ok_or_else(|| LensError::Other(anyhow::anyhow!("trunk graph has no output")))?;
        let spine_end = match &graph.node(out).op {
            rlx_ir::Op::RmsNorm { .. } | rlx_ir::Op::LayerNorm { .. } => graph.node(out).inputs[0],
            _ => out,
        };
        let chain = residual_stream_from(&graph, spine_end).map_err(LensError::Other)?;
        // Every layer folds its attention and its FFN back into the stream, so
        // the chain is `1 + joins·layers` long. Derive `joins` rather than
        // assuming 2, so a block shape this does not understand fails loudly
        // here instead of silently tapping the wrong point.
        let n_built = self.cfg.num_hidden_layers;
        let joins = (chain.len() - 1) / n_built;
        if joins == 0 || 1 + joins * n_built != chain.len() {
            return Err(LensError::Other(anyhow::anyhow!(
                "residual chain has {} points for {n_built} layers, which is not \
                 1 + joins·layers for any whole `joins`",
                chain.len()
            )));
        }
        let tap_ids = layer_exit_taps(&chain, source_layers, joins).map_err(LensError::Other)?;
        // The residual *leaving* the target layer, i.e. entering the one after.
        let h_target = chain[joins * (target_layer + 1)];

        let mut graph = graph;
        let mut outputs = vec![h_target];
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
            token_input: self.input_name().to_string(),
            extra_feeds: Vec::new(),
            layers: source_layers.to_vec(),
            batch,
            seq,
        })
    }

    fn unembed(&self, rows: usize) -> Result<UnembedGraph> {
        use rlx_ir::{DType, Graph, Op, Shape};

        let d = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let f = DType::F32;
        let mut params: Params = std::collections::HashMap::new();
        let mut loader = self.loader()?;
        let mut take = |key: &str| -> Result<Vec<f32>> {
            loader
                .take(key)
                .map(|(v, _)| v)
                .map_err(|e| LensError::Other(anyhow::anyhow!("reading {key}: {e}")))
        };

        let mut g = Graph::new("qwen3_unembed");
        let x = g.input(UNEMBED_INPUT, Shape::new(&[rows, d], f));

        let gamma = g.param("model.norm.weight", Shape::new(&[d], f));
        params.insert("model.norm.weight".into(), take("model.norm.weight")?);
        let beta = g.param("model.norm.beta", Shape::new(&[d], f));
        params.insert("model.norm.beta".into(), vec![0.0; d]);
        let normed = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: self.cfg.rms_norm_eps as f32,
            },
            vec![x, gamma, beta],
            Shape::new(&[rows, d], f),
        );

        // Both are stored `[vocab, d]` and the matmul wants `[d, vocab]`, so
        // read them transposed rather than transposing a 150M-entry table here.
        let key = if self.cfg.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        let head = loader
            .take_transposed(key)
            .map(|(v, _)| v)
            .map_err(|e| LensError::Other(anyhow::anyhow!("reading {key} transposed: {e}")))?;
        if head.len() != d * vocab {
            return Err(LensError::Other(anyhow::anyhow!(
                "{key} is {} entries, expected {d}·{vocab} = {}",
                head.len(),
                d * vocab
            )));
        }
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
        use rlx_qwen3::pipeline::{BlockSpec, build_qwen3_block_graph};

        self.check_layer(layer)?;
        let mut loader = self.loader()?;
        // Neither embedding nor logits: residual in, residual out, which is
        // exactly what a per-block Jacobian is taken over.
        let spec = BlockSpec {
            layers: layer..layer + 1,
            embed_input: false,
            produce_logits: false,
        };
        let (graph, params) = build_qwen3_block_graph(&self.cfg, &mut loader, batch, seq, &spec)
            .map_err(LensError::Other)?;
        Ok(BlockGraph {
            graph,
            params,
            residual_input: "hidden_states".to_string(),
            batch,
            seq,
        })
    }
}
