//! [`LensModel`] for a Qwen2.5-VL trunk — the vision-language case.
//!
//! This is the one that makes image patches readable as *words*, and it needs
//! no new lens machinery at all. A Qwen-VL projects image patches to the LM's
//! hidden width — the runner enforces `projector_output_dim == LM hidden` — and
//! splices them into the token sequence, so image content travels the **LM's**
//! residual stream as ordinary positions.
//!
//! Two consequences follow, and they are why this beats tapping a vision tower:
//!
//! * The stream is the one already shown to be identity-dominated, where `J`
//!   comes out high-rank with real diagonal energy. None of the rank-one
//!   collapse a DINOv3 trunk shows applies, because the ViT is upstream of the
//!   tap point.
//! * The vocabulary is the LM's own, so there is no candidate caption list to
//!   pick. A SigLIP-style readout tells you a patch aligns with *your* words;
//!   this tells you what the model is disposed to *say*.
//!
//! The trunk is fed `prefill_hidden` — an already-embedded `[batch, seq, d]`
//! sequence from `MultimodalPrompt::assemble` — and takes mRoPE `cos`/`sin` as
//! graph *inputs*, since they depend on where the image sits in the prompt.
//! Those ride along as [`StackGraph::extra_feeds`].

use std::path::{Path, PathBuf};

use rlx_qwen25_vl::config::Qwen25VlLmConfig;

use crate::model::{BlockGraph, LensError, LensModel, Params, Result, StackGraph, UnembedGraph};
use crate::taps::{layer_exit_taps, residual_stream_from};
use crate::vjp::{Tap, TappedGraph};

const HIDDEN: &str = "prefill_hidden";
const UNEMBED_INPUT: &str = "residual";

/// A Qwen2.5-VL language trunk, ready to be tapped.
pub struct Qwen25VlLensModel {
    cfg: Qwen25VlLmConfig,
    dir: PathBuf,
    name: String,
    /// Per-token mRoPE section positions from the assembled prompt. `None`
    /// falls back to plain 1-D positions, which is wrong whenever an image is
    /// present — image tokens carry 2-D grid positions, not a running counter.
    sections: Option<Vec<[usize; 4]>>,
}

impl Qwen25VlLensModel {
    pub fn open(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        // HF ships the LM fields at the top level with `vision_config` nested;
        // the runtime config is derived from that, not deserialized directly.
        let cfg = rlx_qwen25_vl::config::Qwen25VlHfConfig::from_file(&dir.join("config.json"))?
            .into_runtime()?
            .lm;
        Ok(Self {
            cfg,
            dir,
            name: "qwen25-vl".to_string(),
            sections: None,
        })
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Bind the mRoPE positions produced alongside the assembled prompt.
    pub fn with_sections(mut self, sections: Vec<[usize; 4]>) -> Self {
        self.sections = Some(sections);
        self
    }

    pub fn config(&self) -> &Qwen25VlLmConfig {
        &self.cfg
    }

    fn loader(&self) -> Result<rlx_core::SafetensorsMmapLoader> {
        rlx_core::SafetensorsMmapLoader::open(&self.dir)
            .map_err(|e| LensError::Other(anyhow::anyhow!("opening {}: {e}", self.dir.display())))
    }
}

impl LensModel for Qwen25VlLensModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_layers(&self) -> usize {
        self.cfg.lm.num_hidden_layers
    }

    fn d_model(&self) -> usize {
        self.cfg.lm.hidden_size
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

        // `vlm_prefill` sets `last_logits_only`, which gathers the final token
        // *before* the final norm — it cuts the residual spine and throws away
        // every other position, including all the image ones. The lens wants the
        // whole sequence.
        let opts = rlx_qwen25_vl::lm_flow::Qwen25VlPrefillOpts {
            batch,
            seq,
            with_lm_head: false,
            last_logits_only: false,
            export_aif_qk: false,
            profile: None,
        };
        let mut loader = self.loader()?;
        let built = rlx_qwen25_vl::lm_flow::build_qwen25_vl_prefill_mrope_built(
            &self.cfg,
            &mut loader,
            &opts,
            self.sections.as_deref(),
        )
        .map_err(LensError::Other)?;
        let (graph, params) =
            rlx_core::flow_util::graph_from_built(built).map_err(LensError::Other)?;

        // mRoPE tables are graph inputs, not params: they depend on where the
        // image sits in this prompt, so they are computed per prompt and fed.
        let half = self.cfg.lm.head_dim / 2;
        let (cos, sin) = rlx_qwen25_vl::mrope::mrope_prefill_feeds(
            &self.cfg,
            seq,
            self.sections.as_deref(),
            half,
        );

        let out = *graph
            .outputs
            .first()
            .ok_or_else(|| LensError::Other(anyhow::anyhow!("trunk graph has no output")))?;
        let spine_end = match &graph.node(out).op {
            rlx_ir::Op::RmsNorm { .. } | rlx_ir::Op::LayerNorm { .. } => graph.node(out).inputs[0],
            _ => out,
        };
        let chain = residual_stream_from(&graph, spine_end).map_err(LensError::Other)?;

        let n_built = self.cfg.lm.num_hidden_layers;
        let joins = (chain.len() - 1) / n_built;
        if joins == 0 || 1 + joins * n_built != chain.len() {
            return Err(LensError::Other(anyhow::anyhow!(
                "residual chain has {} points for {n_built} layers, which is not \
                 1 + joins·layers for any whole `joins`",
                chain.len()
            )));
        }
        let tap_ids = layer_exit_taps(&chain, source_layers, joins).map_err(LensError::Other)?;
        let h_target = chain[joins * (target_layer + 1)];

        let mut graph = graph;
        let mut outputs = vec![h_target];
        outputs.extend_from_slice(&tap_ids);
        graph.set_outputs(outputs);

        let taps: Vec<Tap> = source_layers
            .iter()
            .enumerate()
            .map(|(i, &layer)| Tap::at_output(layer, i + 1))
            .collect();
        let tapped = TappedGraph::new(graph, taps).map_err(LensError::Other)?;

        Ok(StackGraph {
            tapped,
            params,
            token_input: HIDDEN.to_string(),
            extra_feeds: vec![("rope_cos".to_string(), cos), ("rope_sin".to_string(), sin)],
            layers: source_layers.to_vec(),
            batch,
            seq,
        })
    }

    fn unembed(&self, rows: usize) -> Result<UnembedGraph> {
        use rlx_core::weight_loader::WeightLoader;
        use rlx_ir::{DType, Graph, Op, Shape};

        let d = self.cfg.lm.hidden_size;
        let vocab = self.cfg.lm.vocab_size;
        let f = DType::F32;
        let mut params: Params = std::collections::HashMap::new();
        let mut loader = self.loader()?;

        let mut g = Graph::new("qwen25_vl_unembed");
        let x = g.input(UNEMBED_INPUT, Shape::new(&[rows, d], f));
        let gamma = g.param("model.norm.weight", Shape::new(&[d], f));
        params.insert(
            "model.norm.weight".into(),
            loader
                .take("model.norm.weight")
                .map(|(v, _)| v)
                .map_err(|e| LensError::Other(anyhow::anyhow!("reading model.norm.weight: {e}")))?,
        );
        let beta = g.param("model.norm.beta", Shape::new(&[d], f));
        params.insert("model.norm.beta".into(), vec![0.0; d]);
        let normed = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: self.cfg.lm.rms_norm_eps as f32,
            },
            vec![x, gamma, beta],
            Shape::new(&[rows, d], f),
        );

        // Qwen2.5-VL-3B ties its head to the embedding table.
        let key = if self.cfg.lm.tie_word_embeddings {
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
                "{key} is {} entries, expected {d}·{vocab}",
                head.len()
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

    fn block(&self, layer: usize, _batch: usize, _seq: usize) -> Result<BlockGraph> {
        self.check_layer(layer)?;
        Err(LensError::Unsupported {
            model: self.name.clone(),
            what: "per-block graphs; use StackLens".to_string(),
        })
    }
}
