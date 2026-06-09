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

//! Shared `--weights` resolution for model CLIs (directory quants, split hints).

use anyhow::{Result, anyhow};
use rlx_core::gguf_support::DEFAULT_GGUF_PREFER_SUBSTR;
use rlx_core::{ResolveWeightsOptions, resolve_weights_file_with_options};
use std::path::{Path, PathBuf};

/// CLI options for resolving a weights path (file or directory).
#[derive(Debug, Clone, Default)]
pub struct WeightsResolveCli {
    pub prefer_gguf: Option<String>,
    pub gguf_index: Option<usize>,
}

impl WeightsResolveCli {
    pub fn to_resolve_opts(&self) -> ResolveWeightsOptions<'_> {
        ResolveWeightsOptions {
            prefer_gguf_substring: self
                .prefer_gguf
                .as_deref()
                .or(Some(DEFAULT_GGUF_PREFER_SUBSTR)),
            gguf_index: self.gguf_index,
        }
    }

    /// Parse `--prefer-quant` / `-p` and `--gguf-index` from argv (does not consume `--weights`).
    pub fn parse_optional_flags(args: &[String], i: &mut usize) -> Result<Self> {
        let mut out = Self::default();
        while *i < args.len() {
            match args[*i].as_str() {
                "--prefer-quant" | "--prefer" | "-p" => {
                    *i += 1;
                    out.prefer_gguf = Some(
                        args.get(*i)
                            .ok_or_else(|| anyhow!("missing value for --prefer-quant"))?
                            .clone(),
                    );
                    *i += 1;
                }
                "--gguf-index" => {
                    *i += 1;
                    out.gguf_index = Some(
                        args.get(*i)
                            .ok_or_else(|| anyhow!("missing value for --gguf-index"))?
                            .parse()?,
                    );
                    *i += 1;
                }
                _ => break,
            }
        }
        Ok(out)
    }
}

/// Resolve `path` using CLI defaults (`Q4_K_M` preference in multi-`.gguf` dirs).
pub fn resolve_weights_cli(path: &Path, cli: &WeightsResolveCli) -> Result<PathBuf> {
    resolve_weights_file_with_options(path, &cli.to_resolve_opts())
}
