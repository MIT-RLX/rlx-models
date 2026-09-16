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

//! Resolving a language pair's Quasar graph into an executable plan.
//!
//! A [`TranslationPlan`] is the config's per-pair DAG flattened into
//! dependency order, with each stage bound to its block definition and its
//! implementation status. Building the plan needs only the config, so a pair
//! can be inspected before its weights are installed — which is exactly how
//! [`crate::assets::Availability`] reports what is still missing.

use crate::assets::Assets;
use crate::pdec::PDecParams;
use crate::quasar::{Block, BlockKind, Decoder, GraphNode, LangPair, QuasarConfig};
use anyhow::{Result, anyhow};
use std::collections::BTreeSet;

/// Whether this crate can execute a stage today.
///
/// This used to say "no block executor is implemented yet", which stopped being
/// true and then misled a reader of `rlx-translate plan` into thinking the
/// seven `MergerBlock` stages were unimplemented when [`crate::execute`] has
/// run them all along. The classification below is the executor's actual
/// coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Executed, and needs no file.
    Ready,
    /// Executed, and its data files are installed.
    HasAsset,
    /// Its data files are not installed on this machine.
    NeedsAsset,
    /// Not executed: the stage passes its input through unchanged. Says nothing
    /// about whether its files are present — see [`TranslationPlan::with_asset_status`].
    Pending,
}

/// What a graph stage does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageKind {
    /// Runs a block, either named in `block-definitions` or declared inline.
    Block(BlockKind),
    /// The terminal `graph-output` sink: it names the stage holding the final
    /// result and does no work of its own.
    Output,
}

impl StageKind {
    /// Display label.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Block(k) => k.as_str(),
            Self::Output => "graph-output",
        }
    }

    /// The block kind, when this stage runs one.
    pub fn block_kind(&self) -> Option<&BlockKind> {
        match self {
            Self::Block(k) => Some(k),
            Self::Output => None,
        }
    }
}

/// One resolved stage.
#[derive(Debug, Clone)]
pub struct Stage {
    /// Graph stage name (`pdec`, `spm_encode`, …).
    pub name: String,
    /// Upstream stage names, `:port` suffixes stripped.
    pub inputs: Vec<String>,
    /// Role each input was bound to (`in1`, `target`, `source`, …), parallel
    /// to [`Stage::inputs`]. Most blocks fan in anonymously; a few name their
    /// inputs semantically and the role is what distinguishes them.
    pub roles: Vec<String>,
    /// Output port each input was read from, parallel to [`Stage::inputs`].
    /// `None` is the stage's default result.
    pub ports: Vec<Option<String>>,
    /// Direction of a `SentencePieceBlock`: the same block type both encodes
    /// and decodes, and only its `action` says which.
    pub spm: Option<crate::quasar::SpmAction>,
    /// Decode parameters of *this* translator stage.
    ///
    /// A pivot pair carries several: `ar_AE-de_DE` has eight
    /// `PDecTranslatorBlock`s across two models, because the OS routes it
    /// ar->en->de. Using one set of parameters for every translator stage
    /// silently translates the wrong direction.
    pub pdec: Option<PDecParams>,
    /// What the stage does.
    pub kind: StageKind,
    /// Referenced block name, when the stage names one.
    pub block: Option<String>,
    /// Asset-relative files this stage needs.
    pub files: Vec<String>,
    pub support: Support,
}

/// A pair's pipeline, in dependency order.
#[derive(Debug, Clone)]
pub struct TranslationPlan {
    pub task: String,
    pub pair: LangPair,
    pub stages: Vec<Stage>,
    /// Decode parameters of the `PDecTranslatorBlock`, when the pair has one.
    pub pdec: Option<PDecParams>,
}

impl TranslationPlan {
    /// Flattens `pair`'s graph under `task`.
    pub fn build(config: &QuasarConfig, task: &str, pair: &LangPair) -> Result<Self> {
        let decoder = config
            .decoders
            .get(task)
            .ok_or_else(|| anyhow!("no task {task:?} in config"))?;
        let graph = decoder
            .graphs
            .get(pair)
            .ok_or_else(|| anyhow!("no graph for {pair} under task {task:?}"))?;

        let stages = graph
            .topo_order()?
            .into_iter()
            .map(|node| stage_for(decoder, node))
            .collect::<Vec<_>>();

        let pdec = decoder
            .translator(pair)
            .ok()
            .map(PDecParams::from_block)
            .transpose()?;

        Ok(Self {
            task: task.to_string(),
            pair: pair.clone(),
            stages,
            pdec,
        })
    }

    /// Every asset-relative file the plan touches.
    pub fn files(&self) -> BTreeSet<String> {
        self.stages
            .iter()
            .flat_map(|s| s.files.iter().cloned())
            .collect()
    }

    /// Stages missing at least one data file.
    pub fn missing_assets(&self) -> Vec<&Stage> {
        self.stages
            .iter()
            .filter(|s| s.support == Support::NeedsAsset)
            .collect()
    }

    /// Refines each stage's status against what is installed on this machine:
    /// a file-bearing stage becomes [`Support::HasAsset`] when every file
    /// resolves and [`Support::NeedsAsset`] when any does not.
    pub fn with_asset_status(mut self, assets: &Assets) -> Self {
        for stage in &mut self.stages {
            if stage.files.is_empty() {
                continue;
            }
            let installed = stage.files.iter().all(|f| assets.resolve(f).is_some());
            stage.support = match (installed, stage.support) {
                // A missing file is worth reporting whether or not the block is
                // executed, because it is an incomplete *install*.
                (false, _) => Support::NeedsAsset,
                // Files present, but nothing runs them: still pending. Marking
                // these `HasAsset` is what made `PDecForceAlign` read as
                // implemented.
                (true, Support::Pending) => Support::Pending,
                (true, _) => Support::HasAsset,
            };
        }
        self
    }

    /// Stages whose data files are all installed.
    pub fn with_assets(&self) -> Vec<&Stage> {
        self.stages
            .iter()
            .filter(|s| s.support == Support::HasAsset)
            .collect()
    }
}

fn stage_for(decoder: &Decoder, node: &GraphNode) -> Stage {
    let block = decoder.block_for(node);
    let kind = if node.is_output() && block.is_none() && node.block_type.is_none() {
        StageKind::Output
    } else {
        StageKind::Block(
            block
                .map(|b| b.kind.clone())
                .or_else(|| node.block_type.clone())
                .unwrap_or_else(|| BlockKind::Other("unknown".into())),
        )
    };
    Stage {
        name: node.name.clone(),
        inputs: node
            .receive_from
            .deps()
            .into_iter()
            .map(str::to_string)
            .collect(),
        roles: node
            .receive_from
            .named_deps()
            .into_iter()
            .map(|(r, _)| r.to_string())
            .collect(),
        ports: node
            .receive_from
            .ports()
            .into_iter()
            .map(|p| p.map(str::to_string))
            .collect(),
        spm: block.and_then(|b| b.spm_action()),
        pdec: block
            .filter(|b| b.kind == BlockKind::PDecTranslator)
            .and_then(|b| PDecParams::from_block(b).ok()),
        support: support_for(&kind),
        files: block.map(files_of).unwrap_or_default(),
        block: node.block.clone(),
        kind,
    }
}

fn files_of(block: &Block) -> Vec<String> {
    block.file_refs().into_iter().map(|(_, p)| p).collect()
}

/// What this crate executes today, before assets are considered.
fn support_for(kind: &StageKind) -> Support {
    let StageKind::Block(kind) = kind else {
        // The sink names the stage holding the result and does no work.
        return Support::Ready;
    };
    // The executor owns this classification; see `execute::handles`.
    if crate::execute::handles(kind) {
        Support::Ready
    } else {
        Support::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> QuasarConfig {
        let v = json!({
            "version-major": 278,
            "version-minor": 0,
            "mt-model-info": {
                "version": "MT-PL-v278",
                "language-pairs": ["en_US-fr_FR", "fr_FR-en_US"],
                "tasks": ["mt_app"]
            },
            "mt-decoders": {
                "mt_app": {
                    "engine-type": "PDEC",
                    "block-definitions": {
                        "en_US-fr_FR-SentencePieceBlockEncode": {
                            "block-type": "SentencePieceBlock",
                            "action": "encode",
                            "sentence-piece-file": "MT-x/MT/spm.model",
                            "asset-name": "MT-x"
                        },
                        "en_US-fr_FR-PDecTranslatorBlock": {
                            "block-type": "PDecTranslatorBlock",
                            "model-file": "MT-x/MT/pyespresso.mdl.bin",
                            "model-type": "espresso",
                            "beam": 3,
                            "source-token": "en_US",
                            "target-token": "fr_FR> <en_US-fr_FR-optimal",
                            "asset-name": "MT-x"
                        },
                        "en_US-fr_FR-CaseMapBlock": {
                            "block-type": "CaseMapBlock",
                            "locale": "en"
                        }
                    },
                    "language-pair-specific-settings": {
                        "en_US-fr_FR": {
                            "graph": {
                                "spm_encode": {
                                    "receive-from": "graph-input",
                                    "block": "en_US-fr_FR-SentencePieceBlockEncode"
                                },
                                "pdec": {
                                    "receive-from": "spm_encode",
                                    "block": "en_US-fr_FR-PDecTranslatorBlock"
                                },
                                "case": {
                                    "receive-from": "pdec",
                                    "block": "en_US-fr_FR-CaseMapBlock"
                                }
                            }
                        }
                    }
                }
            }
        });
        QuasarConfig::from_slice(&serde_json::to_vec(&v).expect("serialize")).expect("parses")
    }

    #[test]
    fn plan_is_in_dependency_order() {
        let cfg = config();
        let pair = LangPair::parse("en_US-fr_FR").expect("pair");
        let plan = TranslationPlan::build(&cfg, "mt_app", &pair).expect("builds");
        let names: Vec<_> = plan.stages.iter().map(|s| s.name.clone()).collect();
        assert_eq!(names, vec!["spm_encode", "pdec", "case"]);
    }

    #[test]
    fn plan_carries_decode_params_and_files() {
        let cfg = config();
        let pair = LangPair::parse("en_US-fr_FR").expect("pair");
        let plan = TranslationPlan::build(&cfg, "mt_app", &pair).expect("builds");
        let pdec = plan.pdec.as_ref().expect("has a translator");
        assert_eq!(pdec.beam, 3);
        assert_eq!(pdec.target_tokens(), vec!["fr_FR", "en_US-fr_FR-optimal"]);
        let files = plan.files();
        assert!(files.contains("MT-x/MT/spm.model"));
        assert!(files.contains("MT-x/MT/pyespresso.mdl.bin"));
    }

    #[test]
    fn stages_report_support_and_inputs() {
        let cfg = config();
        let pair = LangPair::parse("en_US-fr_FR").expect("pair");
        let plan = TranslationPlan::build(&cfg, "mt_app", &pair).expect("builds");
        let by = |n: &str| {
            plan.stages
                .iter()
                .find(|s| s.name == n)
                .cloned()
                .expect("stage present")
        };
        // `support_for` answers "does the executor run this block", and
        // SentencePiece it does; `with_asset_status` is what then folds in
        // whether the file is on the machine. This asserted `Pending` while the
        // classification still claimed no block executor existed.
        assert_eq!(by("spm_encode").support, Support::Ready);
        assert_eq!(
            by("spm_encode").kind,
            StageKind::Block(BlockKind::SentencePiece)
        );
        assert_eq!(by("case").support, Support::Ready);
        assert_eq!(by("pdec").inputs, vec!["spm_encode"]);
        assert!(by("spm_encode").inputs.contains(&"graph-input".to_string()));
    }

    #[test]
    fn unknown_task_or_pair_errors_clearly() {
        let cfg = config();
        let pair = LangPair::parse("en_US-fr_FR").expect("pair");
        let err = TranslationPlan::build(&cfg, "nope", &pair).expect_err("bad task");
        assert!(err.to_string().contains("nope"), "{err}");

        let other = LangPair::parse("de_DE-fr_FR").expect("pair");
        let err = TranslationPlan::build(&cfg, "mt_app", &other).expect_err("bad pair");
        assert!(err.to_string().contains("de_DE-fr_FR"), "{err}");
    }

    #[test]
    fn asset_status_downgrades_ready_stages_with_absent_files() {
        let cfg = config();
        let pair = LangPair::parse("en_US-fr_FR").expect("pair");
        let plan = TranslationPlan::build(&cfg, "mt_app", &pair)
            .expect("builds")
            .with_asset_status(&Assets::default());
        // CaseMap needs no files, so it stays Ready even with no assets at all.
        let case = plan
            .stages
            .iter()
            .find(|s| s.name == "case")
            .expect("stage");
        assert_eq!(case.support, Support::Ready);
    }
}
