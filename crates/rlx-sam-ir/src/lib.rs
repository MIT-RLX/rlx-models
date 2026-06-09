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

//! Shared IR builders for SAM v1/v2 mask decoders (two-way transformer, hyper matmul, MLP).

pub mod mask_hyper_matmul_ir;
pub mod mask_prompt_ir;
pub mod mlp_relu_ir;
pub mod twoway_transformer_ir;

pub use twoway_transformer_ir::TwoWayTransformerCompiled;
