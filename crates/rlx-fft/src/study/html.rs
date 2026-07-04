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

//! Unified HTML study report — delegates to [`crate::study::full_html`].

use crate::ablation::AblationReport;
use crate::study::full_html::{FullStudyInputs, write_full_study_html};
use crate::study::telemetry::StudyTelemetryBundle;
use crate::train::multi::MultiTrainReport;
use anyhow::Result;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct StudyInputs {
    pub ablation: Option<AblationReport>,
    pub multi_train: Option<MultiTrainReport>,
    pub telemetry: Option<StudyTelemetryBundle>,
}

pub fn write_study_html(path: &Path, inputs: &StudyInputs) -> Result<()> {
    write_full_study_html(
        path,
        &FullStudyInputs {
            ablation: inputs.ablation.clone(),
            multi_train: inputs.multi_train.clone(),
            telemetry: inputs.telemetry.clone(),
        },
    )
}

pub fn render_study_html(inputs: &StudyInputs) -> String {
    crate::study::full_html::render_full_study_html(&FullStudyInputs {
        ablation: inputs.ablation.clone(),
        multi_train: inputs.multi_train.clone(),
        telemetry: inputs.telemetry.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ablation::AblationRow;

    #[test]
    fn render_study_html_smoke() {
        let inputs = StudyInputs {
            ablation: Some(crate::ablation::AblationReport {
                iters: 10,
                train_steps: 0,
                both_dirs: true,
                with_welch: true,
                limit_sweep: false,
                n_ffts: vec![256],
                elapsed_ms: 1.0,
                rows: vec![AblationRow {
                    tier: "baseline".into(),
                    variant: "rustfft".into(),
                    direction: "Forward".into(),
                    n_fft: 256,
                    batch: 8,
                    device: "cpu".into(),
                    iters: 10,
                    ms: 0.01,
                    max_err: 0.0,
                    train_steps: 0,
                    param_count: 0,
                    memory_bytes: 0,
                    status: "ok".into(),
                    note: None,
                }],
            }),
            multi_train: None,
            telemetry: None,
        };
        let html = render_study_html(&inputs);
        assert!(html.contains("RLX-FFT Full Study Report"));
        assert!(html.contains("rustfft"));
    }
}
