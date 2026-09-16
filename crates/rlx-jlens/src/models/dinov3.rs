//! [`LensModel`] for a DINOv3 vision transformer.
//!
//! The third implementation, and the first that is not a language model. It is
//! here to answer whether the lens is really modality-agnostic or just
//! text-shaped, and it exercised two places where the interface still assumed
//! language: `StackLens` validated its input as `batch · seq` token ids (a ViT
//! trunk takes an already-embedded `[batch, tokens, d]` sequence), and the
//! readout assumed a vocabulary.
//!
//! **There is no vocabulary.** DINOv3 ships a backbone with no head, no
//! prototypes and no text tower, so nothing plays the part `unembed` plays for
//! an LM. Rather than bolt on a probe — which would make the readout a
//! statement about the probe — `unembed` here is the model's own **final
//! LayerNorm**, and "logits" are the normed `d_model` representation. That is
//! exactly what a DINO feature *is*, so the lens question becomes:
//!
//! > transported forward, how close is patch *p*'s layer-*l* representation to
//! > the representation it actually ends up with?
//!
//! which is the cosine-to-final trajectory, and the direct analogue of "rank of
//! the answer" in the text case — with no vocabulary needed and nothing chosen
//! by hand.
//!
//! Tokens are `[CLS, register × 4, patch × 196]` for ViT-L/16 at 224², and the
//! patch embedding is host-side preprocessing outside the graph, so the trunk
//! this taps is pure transformer.

use std::path::{Path, PathBuf};

use rlx_dinov3::DinoV3Config;

use crate::model::{BlockGraph, LensError, LensModel, Params, Result, StackGraph, UnembedGraph};
use crate::taps::{layer_exit_taps, residual_stream_from};
use crate::vjp::{Tap, TappedGraph};

/// The trunk's input and output are both called `hidden`: an assembled
/// `[batch, seq, d]` token sequence in, the post-final-norm sequence out.
const HIDDEN: &str = "hidden";
const UNEMBED_INPUT: &str = "residual";

/// A DINOv3 trunk, ready to be tapped.
pub struct Dinov3LensModel {
    cfg: DinoV3Config,
    weights: PathBuf,
    name: String,
}

impl Dinov3LensModel {
    /// Open a checkpoint directory (`config.json` + `model.safetensors`).
    pub fn open(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let cfg = DinoV3Config::from_file(&dir.join("config.json"))?;
        Ok(Self {
            cfg,
            weights: dir.join("model.safetensors"),
            name: "dinov3".to_string(),
        })
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn config(&self) -> &DinoV3Config {
        &self.cfg
    }

    /// Tokens per image: `CLS + registers + patches`.
    pub fn seq_len(&self) -> usize {
        self.cfg.seq_len()
    }

    /// Index of the first patch token — everything before it is CLS/registers.
    pub fn first_patch(&self) -> usize {
        1 + self.cfg.num_register_tokens
    }

    fn build(&self, batch: usize) -> Result<(rlx_ir::Graph, Params)> {
        Ok(self.build_with_preprocess(batch)?.0)
    }

    /// The trunk graph plus the patch-embed / CLS / register weights.
    ///
    /// Patch embedding is host-side in DINOv3, so a caller that wants to feed a
    /// real image needs these to assemble `[CLS, registers, patches]` — the
    /// trunk itself starts from that sequence.
    pub fn build_with_preprocess(
        &self,
        batch: usize,
    ) -> Result<(
        (rlx_ir::Graph, Params),
        rlx_dinov3::preprocess::DinoV3PreprocessWeights,
    )> {
        let mut wm = rlx_core::load_weight_map(&self.weights, &[])
            .map_err(|e| LensError::Other(anyhow::anyhow!("loading {:?}: {e}", self.weights)))?;
        let built = rlx_dinov3::flow::build_dinov3_built(&self.cfg, &mut wm, batch)
            .map_err(LensError::Other)?;
        let pre = built.preprocess;
        let g = rlx_core::flow_util::graph_from_built(built.model).map_err(LensError::Other)?;
        Ok((g, pre))
    }

    /// Assemble the trunk input for one image: RGB bytes → `[seq, d]`.
    pub fn hidden_from_rgb(&self, rgb: &[u8], h_in: usize, w_in: usize) -> Result<Vec<f32>> {
        let (_, pre) = self.build_with_preprocess(1)?;
        let nchw =
            rlx_dinov3::preprocess::rgb_u8_to_imagenet_nchw(rgb, h_in, w_in, self.cfg.image_size);
        rlx_dinov3::preprocess::assemble_hidden(
            &pre,
            &nchw,
            1,
            self.cfg.patch_size,
            self.cfg.image_size,
        )
        .map_err(LensError::Other)
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
}

impl LensModel for Dinov3LensModel {
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
        if seq != self.cfg.seq_len() {
            return Err(LensError::Other(anyhow::anyhow!(
                "seq must be the token count {} (CLS + {} registers + {} patches), got {seq}",
                self.cfg.seq_len(),
                self.cfg.num_register_tokens,
                self.cfg.seq_len() - self.first_patch()
            )));
        }

        let (graph, params) = self.build(batch)?;
        // The graph's output is post-final-norm; the residual spine ends one
        // node earlier. Same shape as the Qwen3 trunk — walking back from the
        // output would find a norm and report a one-node chain.
        let out = *graph
            .outputs
            .first()
            .ok_or_else(|| LensError::Other(anyhow::anyhow!("trunk graph has no output")))?;
        let spine_end = match &graph.node(out).op {
            rlx_ir::Op::LayerNorm { .. } | rlx_ir::Op::RmsNorm { .. } => graph.node(out).inputs[0],
            _ => out,
        };
        let chain = residual_stream_from(&graph, spine_end).map_err(LensError::Other)?;

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
            extra_feeds: Vec::new(),
            layers: source_layers.to_vec(),
            batch,
            seq,
        })
    }

    /// The final LayerNorm, and nothing else.
    ///
    /// A DINO feature *is* the post-final-norm token, so this is the whole of
    /// the model's "output space" — "logits" are `d_model` wide and are the
    /// representation itself. There is no vocabulary to project into and none
    /// is invented here.
    fn unembed(&self, rows: usize) -> Result<UnembedGraph> {
        use rlx_ir::{DType, Graph, Op, Shape};

        let d = self.cfg.hidden_size;
        let f = DType::F32;
        let mut params: Params = std::collections::HashMap::new();
        let mut wm = rlx_core::load_weight_map(&self.weights, &[])
            .map_err(|e| LensError::Other(anyhow::anyhow!("loading {:?}: {e}", self.weights)))?;
        let mut take = |key: &str| -> Result<Vec<f32>> {
            wm.take(key)
                .map(|(v, _)| v)
                .map_err(|e| LensError::Other(anyhow::anyhow!("reading {key}: {e}")))
        };
        // HF DINOv3 names the trunk's trailing norm `norm`; older exports use
        // `layernorm`. Try both rather than guess.
        let (gamma_v, beta_v) = match (take("norm.weight"), take("norm.bias")) {
            (Ok(g), Ok(b)) => (g, b),
            _ => (take("layernorm.weight")?, take("layernorm.bias")?),
        };

        let mut g = Graph::new("dinov3_unembed");
        let x = g.input(UNEMBED_INPUT, Shape::new(&[rows, d], f));
        let gamma = g.param("norm.weight", Shape::new(&[d], f));
        params.insert("norm.weight".into(), gamma_v);
        let beta = g.param("norm.bias", Shape::new(&[d], f));
        params.insert("norm.bias".into(), beta_v);
        let normed = g.add_node(
            Op::LayerNorm {
                axis: -1,
                eps: self.cfg.layer_norm_eps as f32,
            },
            vec![x, gamma, beta],
            Shape::new(&[rows, d], f),
        );
        g.set_outputs(vec![normed]);

        Ok(UnembedGraph {
            graph: g,
            params,
            residual_input: UNEMBED_INPUT.to_string(),
            rows,
            vocab: d,
        })
    }

    fn block(&self, layer: usize, _batch: usize, _seq: usize) -> Result<BlockGraph> {
        self.check_layer(layer)?;
        Err(LensError::Unsupported {
            model: self.name.clone(),
            what: "per-block graphs; DINOv3 exposes a whole-trunk builder only, \
                   so use StackLens"
                .to_string(),
        })
    }
}
