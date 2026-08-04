// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Leaderboard: collect metric rows across models/backends and render a
//! `LLM_BENCH.md`-style markdown table. Long-format (`model | device | metric |
//! value`) so any dimension — including ones added later — drops in without a
//! schema change.

use std::path::Path;

use anyhow::{Context, Result};

use crate::parity::ParityResult;
use crate::quality::{Gsm8kResult, MmluResult};
use crate::speed::SpeedResult;

/// One leaderboard cell.
#[derive(Debug, Clone)]
pub struct BenchRow {
    pub model: String,
    pub device: String,
    pub metric: String,
    pub value: String,
}

/// Accumulates [`BenchRow`]s and renders markdown.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub rows: Vec<BenchRow>,
}

impl Report {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, model: &str, device: &str, metric: &str, value: impl Into<String>) {
        self.rows.push(BenchRow {
            model: model.to_string(),
            device: device.to_string(),
            metric: metric.to_string(),
            value: value.into(),
        });
    }

    /// Add every speed metric.
    pub fn add_speed(&mut self, model: &str, device: &str, r: &SpeedResult) {
        self.push(
            model,
            device,
            "speed/prefill_toks_s",
            format!("{:.1}", r.prefill_toks_s),
        );
        self.push(
            model,
            device,
            "speed/decode_toks_s",
            format!("{:.1}", r.decode_toks_s),
        );
        self.push(model, device, "speed/ttft_ms", format!("{:.1}", r.ttft_ms));
        self.push(
            model,
            device,
            "speed/peak_rss_mb",
            format!("{}", r.peak_rss_mb),
        );
    }

    /// Add MMLU accuracy (raw + normalized).
    pub fn add_mmlu(&mut self, model: &str, device: &str, r: &MmluResult) {
        self.push(model, device, "mmlu/acc", format!("{:.4}", r.acc));
        self.push(model, device, "mmlu/acc_norm", format!("{:.4}", r.acc_norm));
    }

    /// Add GSM8K accuracy.
    pub fn add_gsm8k(&mut self, model: &str, device: &str, r: &Gsm8kResult) {
        self.push(model, device, "gsm8k/acc", format!("{:.4}", r.acc));
    }

    /// Add parity signals.
    pub fn add_parity(&mut self, model: &str, device: &str, r: &ParityResult) {
        if let Some(m) = r.argmax_match {
            self.push(
                model,
                device,
                "parity/argmax_match",
                if m { "yes" } else { "no" },
            );
        }
        if let Some(c) = r.cosine {
            self.push(model, device, "parity/cosine", format!("{c:.6}"));
        }
    }

    /// Render a GitHub-flavored markdown table.
    pub fn to_markdown(&self) -> String {
        let mut s = String::from("# RLX LLM benchmark\n\n");
        if self.rows.is_empty() {
            s.push_str("_No results._\n");
            return s;
        }
        s.push_str("| model | device | metric | value |\n");
        s.push_str("|---|---|---|---:|\n");
        for r in &self.rows {
            s.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                r.model, r.device, r.metric, r.value
            ));
        }
        s
    }

    /// Write the markdown report to `path`.
    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_markdown())
            .with_context(|| format!("writing report {}", path.display()))?;
        Ok(())
    }
}
