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

//! Code predictor (5-layer Qwen3-shaped, groups 1–15).

mod bench;
mod compiled;
mod eager;
pub(crate) mod engine;

pub use bench::{CpBenchBackend, CpBenchReport, bench_cp_ab, bench_cp_predict_groups};
pub use compiled::CpCompiledEngine;
pub use eager::CpEagerModel;
pub use engine::CodePredictorEngine;
