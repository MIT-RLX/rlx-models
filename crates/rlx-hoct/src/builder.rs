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

//! Convenience builders for eager HOCT models.

use crate::config::HoctConfig;
use crate::flow::HoctFlow;
use crate::model::HoctModel;
use anyhow::Result;
use std::path::Path;

/// Load safetensors and return an eager [`HoctModel`] with the given config.
pub fn build_hoct_eager(cfg: &HoctConfig, weights_path: impl AsRef<Path>) -> Result<HoctModel> {
    HoctFlow::new(cfg.clone()).build_from_path(weights_path)
}
