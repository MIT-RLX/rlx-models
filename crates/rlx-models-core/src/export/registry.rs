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

//! Which family exports a checkpoint. A family registers its exporter
//! under its runner's name ([`crate::lm_registry`]) with the `model_type`s
//! and GGUF architectures it takes; [`export`] picks one for a checkpoint,
//! by those claims, then by the runner the model registry names for it.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{OnceLock, RwLock};

use anyhow::{Result, bail};

use super::{Checkpoint, ExportOptions, ExportReport, Format};

/// Export the checkpoint at `path` to the package `out`.
pub type ExporterFn = fn(&Path, &Path, &ExportOptions) -> Result<ExportReport>;

/// A family's exporter.
#[derive(Debug, Clone)]
pub struct Exporter {
    /// The family, as its runner is registered (`qwen3`, `gemma`, …).
    pub name: &'static str,
    /// The package's task (`text-generation`, `feature-extraction`, …).
    pub task: &'static str,
    /// `config.json` `model_type`s and GGUF `general.architecture`s it
    /// takes.
    pub claims: &'static [&'static str],
    /// `config.json` `architectures` it takes, where a `model_type` alone
    /// does not say (a BERT encoder vs. a BERT reranker).
    pub architectures: &'static [&'static str],
    pub export: ExporterFn,
}

impl Exporter {
    pub const fn new(name: &'static str, task: &'static str, export: ExporterFn) -> Self {
        Self {
            name,
            task,
            claims: &[],
            architectures: &[],
            export,
        }
    }

    /// The `model_type`s and GGUF architectures it takes.
    pub const fn claims(mut self, claims: &'static [&'static str]) -> Self {
        self.claims = claims;
        self
    }

    /// The `config.json` `architectures` it takes, ahead of an exporter
    /// that claims only the `model_type`.
    pub const fn architectures(mut self, architectures: &'static [&'static str]) -> Self {
        self.architectures = architectures;
        self
    }
}

/// The config-driven dense decoder ([`super::standard`]): the exporter
/// [`detect`] falls back to.
const STANDARD: &str = "standard";

fn table() -> &'static RwLock<BTreeMap<&'static str, Exporter>> {
    static T: OnceLock<RwLock<BTreeMap<&'static str, Exporter>>> = OnceLock::new();
    T.get_or_init(|| {
        let standard = Exporter::new(STANDARD, "text-generation", super::standard::export_rlxp);
        RwLock::new(BTreeMap::from([(STANDARD, standard)]))
    })
}

/// Register an exporter; the same name again replaces it.
pub fn register_exporter(e: Exporter) {
    table().write().expect("exporter table").insert(e.name, e);
}

/// The exporter registered as `name`.
pub fn exporter(name: &str) -> Option<Exporter> {
    table().read().expect("exporter table").get(name).cloned()
}

/// Every registered exporter, by name.
pub fn exporters() -> Vec<Exporter> {
    table()
        .read()
        .expect("exporter table")
        .values()
        .cloned()
        .collect()
}

/// The exporter for `ck`: one claiming its `architectures`, then its
/// `model_type` (or GGUF architecture), then the one named as the runner
/// the model registry gives that type, then — for a dense decoder its
/// config describes — the config-driven one ([`super::standard`]).
pub fn detect(ck: &Checkpoint) -> Result<Exporter> {
    let all = exporters();
    let archs = ck.architectures();
    if let Some(e) = all
        .iter()
        .find(|e| e.architectures.iter().any(|a| archs.contains(a)))
    {
        return Ok(e.clone());
    }
    let Some(kind) = ck.model_type() else {
        bail!(
            "{}: its config names no model_type (or GGUF architecture)",
            ck.path.display()
        );
    };
    if let Some(e) = all.iter().find(|e| e.claims.contains(&kind)) {
        return Ok(e.clone());
    }
    let runner = match ck.format {
        Format::Safetensors => crate::model_registry::runner_for_hf_model_type(kind),
        Format::Gguf => crate::model_registry::runner_for_gguf_arch(kind),
    };
    if let Some(e) = runner.and_then(exporter) {
        return Ok(e);
    }
    if super::standard::supports(ck)
        && let Some(e) = exporter(STANDARD)
    {
        return Ok(e);
    }
    let names: Vec<&str> = all.iter().map(|e| e.name).collect();
    match runner {
        Some(r) => bail!(
            "{kind}: runs with the `{r}` runner, which has no exporter yet \
             (exporters: {})",
            names.join(", ")
        ),
        None => bail!(
            "{kind}: no exporter takes it (exporters: {})",
            names.join(", ")
        ),
    }
}

/// Export the checkpoint at `path` (a Hugging Face directory or a GGUF) as
/// the package `out`, by the exporter [`detect`] picks, or `name`'s.
pub fn export(
    path: &Path,
    out: &Path,
    opts: &ExportOptions,
    name: Option<&str>,
) -> Result<(Exporter, ExportReport)> {
    let e = match name {
        Some(n) => match exporter(n) {
            Some(e) => e,
            None => bail!("no exporter is registered as `{n}`"),
        },
        None => detect(&Checkpoint::open(path)?)?,
    };
    let report = (e.export)(path, out, opts)?;
    Ok((e, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn none(_: &Path, _: &Path, _: &ExportOptions) -> Result<ExportReport> {
        Ok(ExportReport::default())
    }

    fn ck(config: serde_json::Value) -> Checkpoint {
        Checkpoint {
            path: "/nowhere".into(),
            format: Format::Safetensors,
            config,
        }
    }

    #[test]
    fn a_checkpoint_goes_to_the_exporter_claiming_it() {
        register_exporter(
            Exporter::new("test-enc", "feature-extraction", none).claims(&["testbert"]),
        );
        register_exporter(
            Exporter::new("test-rank", "text-ranking", none)
                .architectures(&["TestBertForSequenceClassification"]),
        );
        let enc = ck(json!({"model_type": "testbert", "architectures": ["TestBertModel"]}));
        assert_eq!(detect(&enc).unwrap().name, "test-enc");
        // `architectures` first: the same model_type, a reranker.
        let rank = ck(json!({"model_type": "testbert",
                             "architectures": ["TestBertForSequenceClassification"]}));
        assert_eq!(detect(&rank).unwrap().name, "test-rank");
        let err = detect(&ck(json!({"model_type": "nothing-takes-this"})))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no exporter takes it"), "{err}");
    }
}
