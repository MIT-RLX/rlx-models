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

use anyhow::Result;
use rlx_runtime::{CompiledGraph, Device, Precision, PrecisionPolicy, Session};

use rlx_bert::flow::build_bert_built;
use rlx_core::config::BertConfig;
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::flow_util::{compile_built, graph_from_built};
use rlx_core::weight_map::WeightMap;
use rlx_ir::logical_kernel::KernelDispatchConfig;

/// RLX-compiled BERT model ready for inference.
pub struct RlxBertModel {
    compiled: CompiledGraph,
    config: BertConfig,
    weights_path: String,
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
        let compiled = Self::compile_flow(
            &config,
            weights_path,
            batch,
            seq,
            device,
            precision,
            &policy,
        )?;
        Ok(Self {
            compiled,
            config,
            weights_path: weights_path.to_string(),
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

    pub fn recompile(&mut self, batch: usize, seq: usize) -> Result<()> {
        if self.compiled_bs == (batch, seq) {
            return Ok(());
        }
        self.compiled = Self::compile_flow(
            &self.config,
            &self.weights_path,
            batch,
            seq,
            self.device,
            self.precision,
            &self.policy,
        )?;
        self.compiled_bs = (batch, seq);
        Ok(())
    }

    fn compile_flow(
        config: &BertConfig,
        weights_path: &str,
        batch: usize,
        seq: usize,
        device: Device,
        precision: Precision,
        policy: &Option<PrecisionPolicy>,
    ) -> Result<CompiledGraph> {
        let mut wm = WeightMap::from_file(weights_path)?;
        let built = build_bert_built(config, &mut wm, batch, seq)?;
        if device == Device::Cpu && precision == Precision::F32 && policy.is_none() {
            return compile_built(built, device);
        }
        let profile = built.profile().clone();
        let (graph, params) = graph_from_built(built)?;
        let mut opts =
            compile_options_from_profile(&profile, device, KernelDispatchConfig::default());
        opts.precision = precision;
        opts.policy = policy.clone();
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        for (name, data) in params {
            compiled.set_param(&name, &data);
        }
        Ok(compiled)
    }

    pub fn forward(
        &mut self,
        input_ids: &[f32],
        attention_mask: &[f32],
        token_type_ids: &[f32],
        position_ids: &[f32],
    ) -> Vec<f32> {
        let batch = self.compiled_bs.0;
        let seq = self.compiled_bs.1;
        let _ = self.recompile(batch, seq);
        let outputs = self.compiled.run(&[
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("token_type_ids", token_type_ids),
            ("position_ids", position_ids),
        ]);
        outputs.into_iter().next().unwrap_or_default()
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
fn roberta_position_offset(config_path: &Path) -> usize {
    let Ok(raw) = std::fs::read(config_path) else {
        return 0;
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&raw) else {
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
