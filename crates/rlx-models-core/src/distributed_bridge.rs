// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// GPLv3 — see the workspace license header.

//! **Model-side bridge** to the model-agnostic `rlx-distributed` API. This is
//! the "everything else" that rlx-models supplies on top of the RLX pipeline
//! primitives: adapters that turn rlx-models weight representations into a
//! [`rlx_distributed::ParamSource`], plus a one-call helper to partition a built
//! decoder graph and run it across pipeline stages.
//!
//! A model crate builds its graph as usual (`build_standard_decoder_packed`,
//! `build_deepseek_v4_prefill`, …), which yields the graph plus a dense-f32
//! `params` map and a `packed` (quantized-bytes) map. Feed those to
//! [`MapParamSource`] and hand the graph to
//! [`run_decoder_pipeline_local`] (single machine) or, on a cluster, partition
//! with [`rlx_distributed::graph::partition`] and serve each stage with
//! [`rlx_distributed::graph::serve_stage`] using a per-node source.

use crate::weight_loader::WeightLoader;
use rlx_distributed::{NamedTensor, Param, ParamSource};
use rlx_ir::{DType, Graph, quant::QuantScheme};
use rlx_runtime::{CompileOptions, Device};
use std::collections::HashMap;

/// A [`ParamSource`] over the two maps a builder emits: dense-f32 `params` and
/// `packed` quantized weights. Packed weights are served as raw `U8` bytes — the
/// graph's `Dequant*` op node already carries the quant scheme, so the runtime
/// dequantizes on use. Everything else is dense f32. Each pipeline stage asks
/// only for the param names in its own subgraph, so a worker materializes just
/// its shard.
pub struct MapParamSource {
    pub f32_params: HashMap<String, Vec<f32>>,
    pub packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
}

impl MapParamSource {
    pub fn new(
        f32_params: HashMap<String, Vec<f32>>,
        packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    ) -> Self {
        Self { f32_params, packed }
    }
}

impl ParamSource for MapParamSource {
    fn get(&mut self, name: &str) -> Option<Param> {
        // MOVE the data out (not clone): a pipeline worker holds tens of GB of
        // weights here, and the runtime copies each into the compiled graph on
        // `set_param`. Cloning would keep both copies resident (≈2× peak → OOM);
        // removing frees the map slot as soon as the runtime has taken its copy,
        // so only one weight is ever duplicated at a time. Params are sharded
        // one-to-a-stage, so no later reader needs the same key.
        if let Some((bytes, _scheme, _shape)) = self.packed.remove(name) {
            // MoE scale/bias slabs are stored BF16 to halve resident memory; the
            // packed weight CODES stay U8. The graph param node's dtype must match.
            let dt = if name.ends_with(".scales") || name.ends_with(".biases") {
                DType::BF16
            } else {
                DType::U8
            };
            return Some(Param::typed(bytes, dt));
        }
        self.f32_params.remove(name).map(Param::f32)
    }
}

/// A [`ParamSource`] backed by any rlx-models [`WeightLoader`]: a pipeline
/// worker loads only its stage's params straight from the checkpoint (dense f32
/// via [`WeightLoader::take`], which dequantizes affine/MXFP4 on the way out).
///
/// This is the RAM-streaming path for models too large to build whole on one
/// node: combined with a structure-only graph build (Param nodes without data),
/// each worker loads just its shard. Serves a **single-blob packed** weight
/// (GGUF K-quant `take_packed`) as raw `Param::Typed(U8)` — the graph's Dequant
/// op carries the scheme — and everything else (dense, or affine/MXFP4 that the
/// loader dequantizes on `take`) as `Param::F32`.
///
/// Gap: MLX **multi-entry affine** weights (one checkpoint key expands to
/// `key` codes + `key.scales` + `key.biases` via `take_packed_mlx`) are served
/// by [`MapParamSource`] from the built `packed`/`params` maps, not here — the
/// derived names don't map 1:1 to loader keys.
pub struct LoaderParamSource<'a> {
    pub loader: &'a mut dyn WeightLoader,
}

impl<'a> LoaderParamSource<'a> {
    pub fn new(loader: &'a mut dyn WeightLoader) -> Self {
        Self { loader }
    }
}

impl ParamSource for LoaderParamSource<'_> {
    fn get(&mut self, name: &str) -> Option<Param> {
        // Single-blob packed (GGUF K-quant): raw U8, scheme lives in the op node.
        if let Ok(Some((bytes, _scheme, _shape))) = self.loader.take_packed(name) {
            return Some(Param::typed(bytes, DType::U8));
        }
        // Dense f32, or affine/MXFP4 the loader dequantizes on take.
        self.loader
            .take(name)
            .ok()
            .map(|(data, _shape)| Param::f32(data))
    }
}

/// How a weight was loaded during a structure-only build — so a worker can
/// re-fetch its shard the same way (a transposed weight must come back
/// transposed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadKind {
    Raw,
    Transposed,
    /// MLX packed Linear whose large `w_q` codes were deferred; the worker
    /// re-fetches them via `take_packed_mlx`.
    PackedMlx,
}

/// Wraps a real [`WeightLoader`] to build a graph **structure-only**: it returns
/// each weight's correct SHAPE but drops the data (peak build RAM = one weight,
/// never the whole model — the fix for a coordinator OOMing while building a
/// checkpoint bigger than its RAM), and records a per-key [`LoadKind`] manifest
/// so a worker can re-load exactly its stage's shard. Synth constants (masks,
/// RoPE tables) are built in-graph and keep their data. F32 paths (Raw /
/// Transposed) only; quantized weights are passed through unchanged for now —
/// structure-only *packed* needs a metadata byte-length path (`packed_meta` on
/// the loader) so the graph can size the U8 param without loading its bytes.
pub struct StructureLoader<'a> {
    inner: &'a mut dyn WeightLoader,
    pub manifest: HashMap<String, LoadKind>,
}

impl<'a> StructureLoader<'a> {
    pub fn new(inner: &'a mut dyn WeightLoader) -> Self {
        Self {
            inner,
            manifest: HashMap::new(),
        }
    }
}

impl WeightLoader for StructureLoader<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
    fn take(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (_data, shape) = self.inner.take(key)?; // load + drop → keep only shape
        self.manifest.insert(key.to_string(), LoadKind::Raw);
        Ok((Vec::new(), shape))
    }
    fn take_transposed(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (_data, shape) = self.inner.take_transposed(key)?;
        self.manifest.insert(key.to_string(), LoadKind::Transposed);
        Ok((Vec::new(), shape))
    }
    fn take_packed(
        &mut self,
        key: &str,
    ) -> anyhow::Result<Option<crate::weight_map::PackedWeightTensor>> {
        self.inner.take_packed(key)
    }
    fn take_packed_mlx(
        &mut self,
        key: &str,
    ) -> anyhow::Result<Option<crate::weight_loader::MlxPackedLinear>> {
        self.inner.take_packed_mlx(key)
    }
    /// Defer the large `w_q` codes: measure them (via the inner loader), record
    /// a `PackedMlx` manifest entry, and return the metadata with the codes
    /// dropped — the graph sizes the U8 param from `w_q_len`.
    fn packed_mlx_meta(
        &mut self,
        key: &str,
    ) -> anyhow::Result<Option<crate::weight_loader::PackedMlxMeta>> {
        match self.inner.take_packed_mlx(key)? {
            Some(p) => {
                self.manifest.insert(key.to_string(), LoadKind::PackedMlx);
                let n_groups = p.n_groups();
                let w_q_len = p.w_q.len();
                Ok(Some(crate::weight_loader::PackedMlxMeta {
                    w_q_len,
                    scales: p.scales,
                    biases: p.biases,
                    scheme: p.scheme,
                    out_shape: p.out_shape,
                    n_groups,
                })) // p.w_q dropped → large codes freed
            }
            None => Ok(None),
        }
    }
    fn packed_meta(&self, key: &str) -> Option<(QuantScheme, Vec<usize>)> {
        self.inner.packed_meta(key)
    }
    fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        self.inner.tensor_bytes_borrowed(key)
    }
}

/// Worker-side [`ParamSource`] for a structure-only build: serves the small
/// in-graph synth constants from `synth` (the non-empty params the structure
/// build produced), and re-loads each real weight from `loader` via its
/// recorded [`LoadKind`] (`Transposed` → `take_transposed`, so it comes back
/// pre-transposed). Falls back to packed/dense `take` for names not in the
/// manifest.
pub struct ManifestParamSource<'a> {
    pub loader: &'a mut dyn WeightLoader,
    pub manifest: HashMap<String, LoadKind>,
    /// Small in-graph synth constants + retained f32 scales/biases (from the
    /// structure build's params map, non-empty entries).
    pub synth: HashMap<String, Vec<f32>>,
    /// Small retained packed entries (e.g. MXFP4 u8 scales) from the structure
    /// build's packed map (non-empty entries; the deferred codes are empty).
    pub synth_packed: HashMap<String, Vec<u8>>,
}

impl ParamSource for ManifestParamSource<'_> {
    fn get(&mut self, name: &str) -> Option<Param> {
        // MOVE out of the retained maps (frees as the runtime takes its copy).
        if let Some(v) = self.synth.remove(name) {
            if !v.is_empty() {
                return Some(Param::f32(v));
            }
        }
        if let Some(b) = self.synth_packed.remove(name) {
            if !b.is_empty() {
                // MoE scale/bias slabs are retained BF16 to halve resident memory;
                // any other retained packed bytes are U8. Dtype must match the node.
                let dt = if name.ends_with(".scales") || name.ends_with(".biases") {
                    DType::BF16
                } else {
                    DType::U8
                };
                return Some(Param::typed(b, dt));
            }
        }
        let served = match self.manifest.get(name) {
            Some(LoadKind::Transposed) => self
                .loader
                .take_transposed(name)
                .ok()
                .map(|(d, _)| Param::f32(d)),
            Some(LoadKind::Raw) => self.loader.take(name).ok().map(|(d, _)| Param::f32(d)),
            // Deferred MLX codes: re-fetch just this shard's w_q as raw U8.
            Some(LoadKind::PackedMlx) => self
                .loader
                .take_packed_mlx(name)
                .ok()
                .flatten()
                .map(|p| Param::typed(p.w_q, DType::U8)),
            None => {
                if let Ok(Some((b, _, _))) = self.loader.take_packed(name) {
                    return Some(Param::typed(b, DType::U8));
                }
                self.loader.take(name).ok().map(|(d, _)| Param::f32(d))
            }
        };
        // A silently-unserved param would be left ZERO in the arena (wrong output,
        // no panic). Loudly flag it so a manifest/naming gap can't pass unnoticed.
        if served.is_none() {
            eprintln!(
                "[ManifestParamSource] MISS: no data for param `{name}` (would zero the slot)"
            );
        }
        served
    }
}

/// Partition a built decoder `graph` into `n_stages` and run it in-process
/// through the pipeline — each stage compiles and loads only its parameter
/// shard from `(f32_params, packed)`. Peak weight RAM is one stage at a time.
/// Returns the final logits. This is the single-machine reference a real
/// cluster run (one `serve_stage` per node) must match.
#[allow(clippy::too_many_arguments)]
pub fn run_decoder_pipeline_local(
    graph: Graph,
    f32_params: HashMap<String, Vec<f32>>,
    packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    input_ids: &[u32],
    n_stages: usize,
    device: Device,
    opts: &CompileOptions,
) -> Vec<NamedTensor> {
    let seq = input_ids.len();
    let stages = rlx_distributed::graph::partition(&graph, n_stages);
    let mut src = MapParamSource::new(f32_params, packed);
    let ids_f: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
    let input = NamedTensor::new("input_ids", vec![1, seq], ids_f);
    rlx_distributed::graph::run_pipeline_local(stages, &mut src, vec![input], device, opts)
}
