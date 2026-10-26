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

//! RLX-compiled BERT encoder for text embeddings.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use rlx_runtime::{CompiledGraph, Device, Precision, PrecisionPolicy, Session};

use rlx_bert::flow::{BertFlow, build_bert_built};
use rlx_flow::WeightSource;

use crate::packed::PackedWeights;
use rlx_core::SafetensorsBytesLoader;
use rlx_core::config::BertConfig;
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::flow_util::{attach_built_params, compile_built, graph_from_built};
use rlx_core::weight_map::WeightMap;
use rlx_ir::logical_kernel::KernelDispatchConfig;

/// Where a model's weights come from: an f32 safetensors file, or bytes already
/// in memory (a browser has no filesystem), or an int8-packed checkpoint
/// ([`crate::packed`]) — which runs its projections as int8 and gathers its
/// token embeddings on the host. Cloning shares the bytes.
#[derive(Clone)]
enum WeightSrc {
    Path(String),
    Bytes(Arc<[u8]>),
    /// `int8`: run the projections as int8 matmuls; else widen them to f32
    Packed {
        w: PackedWeights,
        int8: bool,
    },
}

impl WeightSrc {
    fn open_path(p: &str) -> Result<Self> {
        let raw = std::fs::read(p)?;
        Ok(if crate::packed::is_packed(&raw) {
            WeightSrc::Packed {
                w: PackedWeights::new(raw)?,
                int8: true,
            }
        } else {
            WeightSrc::Path(p.to_string())
        })
    }

    fn open_bytes(b: Arc<[u8]>) -> Result<Self> {
        Ok(if crate::packed::is_packed(&b) {
            WeightSrc::Packed {
                w: PackedWeights::new(b)?,
                int8: true,
            }
        } else {
            WeightSrc::Bytes(b)
        })
    }

    fn weight_map(&self) -> Result<WeightMap> {
        match self {
            WeightSrc::Path(p) => WeightMap::from_file(p),
            WeightSrc::Bytes(b) => {
                WeightMap::from_weight_loader(&mut SafetensorsBytesLoader::new(b.clone())?)
            }
            WeightSrc::Packed { w, .. } => w.to_weight_map(),
        }
    }
}

/// RLX-compiled BERT model ready for inference.
pub struct RlxBertModel {
    compiled: CompiledGraph,
    config: BertConfig,
    weights: WeightSrc,
    /// The token-embedding table's key, when its rows are gathered here and
    /// fed as `inputs_embeds` (packed checkpoints).
    host_embed: Option<String>,
    /// The compiled graph's inputs, so `forward` feeds only those.
    input_names: Vec<String>,
    compiled_bs: (usize, usize),
    device: Device,
    precision: Precision,
    policy: Option<PrecisionPolicy>,
    /// First position id. Zero for BERT; `pad_token_id + 1` for the
    /// RoBERTa family. See [`RlxBertModel::position_offset`].
    position_offset: usize,
}

impl RlxBertModel {
    pub fn load_sized(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        seq: usize,
    ) -> Result<Self> {
        Self::load_sized_on(config_path, weights_path, batch, seq, Device::Cpu)
    }

    pub fn load_sized_on(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        seq: usize,
        device: Device,
    ) -> Result<Self> {
        Self::load_sized_with_policy(
            config_path,
            weights_path,
            batch,
            seq,
            device,
            Precision::F32,
            None,
        )
    }

    pub fn load_sized_with_policy(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        seq: usize,
        device: Device,
        precision: Precision,
        policy: Option<PrecisionPolicy>,
    ) -> Result<Self> {
        let config = BertConfig::from_file(config_path)?;
        let position_offset = roberta_position_offset(config_path);
        Self::assemble(
            config,
            WeightSrc::open_path(weights_path)?,
            position_offset,
            (batch, seq),
            device,
            precision,
            policy,
        )
    }

    /// For an int8-packed checkpoint: whether its projections run as int8
    /// matmuls (`true`, the default: a quarter of the memory, and faster where
    /// the backend has a good int8 kernel) or are widened to f32 once, here.
    /// Recompiles; no effect on an f32 checkpoint.
    pub fn set_int8_matmuls(&mut self, on: bool) -> Result<()> {
        if let WeightSrc::Packed { int8, .. } = &mut self.weights
            && *int8 != on
        {
            *int8 = on;
            let (b, s) = self.compiled_bs;
            self.compiled_bs = (0, 0);
            self.recompile(b, s)?;
        }
        Ok(())
    }

    /// Load from a `config.json` and a `model.safetensors` held in memory, as
    /// a browser fetched them.
    pub fn from_bytes(
        config_json: &[u8],
        weights: impl Into<Arc<[u8]>>,
        batch: usize,
        seq: usize,
        device: Device,
    ) -> Result<Self> {
        let config: BertConfig = serde_json::from_slice(config_json)?;
        Self::assemble(
            config,
            WeightSrc::open_bytes(weights.into())?,
            position_offset_of(config_json),
            (batch, seq),
            device,
            Precision::F32,
            None,
        )
    }

    fn assemble(
        config: BertConfig,
        weights: WeightSrc,
        position_offset: usize,
        (batch, seq): (usize, usize),
        device: Device,
        precision: Precision,
        policy: Option<PrecisionPolicy>,
    ) -> Result<Self> {
        let compiled =
            Self::compile_flow(&config, &weights, batch, seq, device, precision, &policy)?;
        let host_embed = match &weights {
            WeightSrc::Packed { w, .. } => Some(rlx_bert::flow::word_embeddings_key(|k| w.has(k))),
            _ => None,
        };
        Ok(Self {
            input_names: compiled.inputs().into_iter().map(|t| t.name).collect(),
            host_embed,
            compiled,
            config,
            weights,
            compiled_bs: (batch, seq),
            device,
            precision,
            policy,
            position_offset,
        })
    }

    /// First position id to feed the model.
    ///
    /// BERT numbers positions from 0, but the RoBERTa family — which includes
    /// every multilingual checkpoint in the registry, since `multilingual-e5`
    /// and `paraphrase-multilingual` are XLM-RoBERTa — reserves ids up to
    /// `pad_token_id` and starts real tokens at `pad_token_id + 1`. That is why
    /// their `max_position_embeddings` is 514 rather than 512. Feeding 0-based
    /// positions reads the wrong row of the position table for every token: the
    /// output is still plausible, just quietly worse, which is the kind of bug
    /// that survives a long time.
    pub fn position_offset(&self) -> usize {
        self.position_offset
    }

    pub fn load(config_path: &Path, weights_path: &str) -> Result<Self> {
        Self::load_sized(config_path, weights_path, 1, 1)
    }

    /// The same model compiled for another `[batch, seq]`, sharing this one's
    /// weight bytes (each compiled graph holds its own copy of the params; on
    /// a packed checkpoint that is the int8 projections only).
    pub fn with_shape(&self, batch: usize, seq: usize) -> Result<Self> {
        let compiled = Self::compile_flow(
            &self.config,
            &self.weights,
            batch,
            seq,
            self.device,
            self.precision,
            &self.policy,
        )?;
        Ok(Self {
            input_names: compiled.inputs().into_iter().map(|t| t.name).collect(),
            host_embed: self.host_embed.clone(),
            compiled,
            config: self.config.clone(),
            weights: self.weights.clone(),
            compiled_bs: (batch, seq),
            device: self.device,
            precision: self.precision,
            policy: self.policy.clone(),
            position_offset: self.position_offset,
        })
    }

    /// The `[batch, seq]` this model is compiled for.
    pub fn shape(&self) -> (usize, usize) {
        self.compiled_bs
    }

    pub fn recompile(&mut self, batch: usize, seq: usize) -> Result<()> {
        if self.compiled_bs == (batch, seq) {
            return Ok(());
        }
        self.compiled = Self::compile_flow(
            &self.config,
            &self.weights,
            batch,
            seq,
            self.device,
            self.precision,
            &self.policy,
        )?;
        self.input_names = self.compiled.inputs().into_iter().map(|t| t.name).collect();
        self.compiled_bs = (batch, seq);
        Ok(())
    }

    fn compile_flow(
        config: &BertConfig,
        weights: &WeightSrc,
        batch: usize,
        seq: usize,
        device: Device,
        precision: Precision,
        policy: &Option<PrecisionPolicy>,
    ) -> Result<CompiledGraph> {
        let built = match weights {
            // int8 projections as DequantMatMul; the vocabulary table stays
            // out of the graph (forward gathers its rows)
            WeightSrc::Packed { w, int8 } => {
                let flow = BertFlow::new(config, batch, seq).with_host_embeddings(true);
                let mut w = w.clone();
                if *int8 {
                    flow.build_from(&mut w)?
                } else {
                    flow.build_from(&mut crate::packed::DenseView(&mut w))?
                }
            }
            _ => build_bert_built(config, &mut weights.weight_map()?, batch, seq)?,
        };
        if device == Device::Cpu && precision == Precision::F32 && policy.is_none() {
            return compile_built(built, device);
        }
        let profile = built.profile().clone();
        let typed = built.typed_params.clone();
        let (graph, params) = graph_from_built(built)?;
        let mut opts =
            compile_options_from_profile(&profile, device, KernelDispatchConfig::default());
        opts.precision = precision;
        opts.policy = policy.clone();
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        attach_built_params(&mut compiled, params, &typed);
        Ok(compiled)
    }

    pub fn forward(
        &mut self,
        input_ids: &[f32],
        attention_mask: &[f32],
        token_type_ids: &[f32],
        position_ids: &[f32],
    ) -> Vec<f32> {
        let embeds = self.host_embeds(input_ids);
        let inputs = self.inputs(
            &embeds,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
        );
        let outputs = self.compiled.run(&inputs);
        outputs.into_iter().next().unwrap_or_default()
    }

    /// The token embeddings of `input_ids` gathered from the host table, when
    /// the graph takes them as `inputs_embeds`; empty otherwise.
    fn host_embeds(&self, input_ids: &[f32]) -> Vec<f32> {
        let (Some(key), WeightSrc::Packed { w, .. }) = (&self.host_embed, &self.weights) else {
            return Vec::new();
        };
        let h = self.config.hidden_size;
        let mut out = vec![0f32; input_ids.len() * h];
        for (row, &id) in out.chunks_exact_mut(h).zip(input_ids) {
            // an id past the table reads as [PAD] (0) rather than failing a query
            if w.add_row(key, id as usize, row).is_err() {
                let _ = w.add_row(key, 0, row);
            }
        }
        out
    }

    /// The named inputs this graph declares, from what a caller has.
    fn inputs<'a>(
        &self,
        embeds: &'a [f32],
        input_ids: &'a [f32],
        attention_mask: &'a [f32],
        token_type_ids: &'a [f32],
        position_ids: &'a [f32],
    ) -> Vec<(&'static str, &'a [f32])> {
        let all = [
            ("inputs_embeds", embeds),
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("token_type_ids", token_type_ids),
            ("position_ids", position_ids),
        ];
        all.into_iter()
            .filter(|(n, d)| {
                if self.input_names.is_empty() {
                    *n != "inputs_embeds" || !d.is_empty()
                } else {
                    self.input_names.iter().any(|x| x == n)
                }
            })
            .collect()
    }

    /// [`Self::forward`] for WebGPU in a browser, where reading results back
    /// cannot block; other backends run synchronously.
    #[cfg(target_arch = "wasm32")]
    pub async fn forward_async(
        &mut self,
        input_ids: &[f32],
        attention_mask: &[f32],
        token_type_ids: &[f32],
        position_ids: &[f32],
    ) -> Result<Vec<f32>> {
        let embeds = self.host_embeds(input_ids);
        let inputs = self.inputs(
            &embeds,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
        );
        let outputs = self
            .compiled
            .run_async(&inputs)
            .await
            .map_err(anyhow::Error::msg)?;
        Ok(outputs.into_iter().next().unwrap_or_default())
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}

/// Reads the position-id offset out of a raw `config.json`.
///
/// [`BertConfig`] carries neither `model_type` nor `pad_token_id`, and widening
/// a shared core type for one architecture's quirk is worse than reading the
/// two fields here. Unknown or absent fields give 0, i.e. BERT behaviour.
pub(crate) fn roberta_position_offset(config_path: &Path) -> usize {
    std::fs::read(config_path).map_or(0, |raw| position_offset_of(&raw))
}

/// [`roberta_position_offset`] of a `config.json` already read.
fn position_offset_of(raw: &[u8]) -> usize {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(raw) else {
        return 0;
    };
    let model_type = v
        .get("model_type")
        .and_then(|m| m.as_str())
        .unwrap_or_default();
    if !model_type.contains("roberta") {
        return 0;
    }
    v.get("pad_token_id")
        .and_then(|p| p.as_u64())
        .map_or(0, |p| p as usize + 1)
}

#[cfg(test)]
mod position_offset_tests {
    use super::roberta_position_offset;

    fn write(dir: &std::path::Path, body: &str) -> std::path::PathBuf {
        std::fs::create_dir_all(dir).expect("mkdir");
        let p = dir.join("config.json");
        std::fs::write(&p, body).expect("write");
        p
    }

    #[test]
    fn bert_numbers_positions_from_zero() {
        let d = std::env::temp_dir().join(format!("rlx-embed-pos-b{}", std::process::id()));
        let p = write(&d, r#"{"model_type":"bert","pad_token_id":0}"#);
        assert_eq!(roberta_position_offset(&p), 0);
        std::fs::remove_dir_all(&d).ok();
    }

    /// The multilingual checkpoints — `multilingual-e5-*`,
    /// `paraphrase-multilingual-*` — are all XLM-RoBERTa with `pad_token_id: 1`,
    /// so their first real position is 2. Measured on `multilingual-e5-base`,
    /// getting this wrong shrinks the margin between paraphrase and unrelated
    /// text from 0.61 to 0.20.
    #[test]
    fn xlm_roberta_starts_after_the_pad_token() {
        let d = std::env::temp_dir().join(format!("rlx-embed-pos-x{}", std::process::id()));
        let p = write(&d, r#"{"model_type":"xlm-roberta","pad_token_id":1}"#);
        assert_eq!(roberta_position_offset(&p), 2);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn an_unreadable_or_partial_config_falls_back_to_bert() {
        let d = std::env::temp_dir().join(format!("rlx-embed-pos-u{}", std::process::id()));
        assert_eq!(roberta_position_offset(&d.join("missing.json")), 0);
        let p = write(&d, r#"{"model_type":"roberta"}"#);
        assert_eq!(roberta_position_offset(&p), 0, "no pad_token_id");
        let p = write(&d, "not json");
        assert_eq!(roberta_position_offset(&p), 0);
        std::fs::remove_dir_all(&d).ok();
    }
}
