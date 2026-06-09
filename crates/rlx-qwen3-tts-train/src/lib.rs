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

//! Native LoRA distillation on the Qwen3-TTS talker (MLX / Metal / CPU).

pub mod adam;
pub mod backward_prep;
pub mod codec_table;
pub mod compile;
pub mod config;
pub mod dataset;
pub mod device;
pub mod distill_cache;
pub mod jfk_lora;
pub mod talker_lora_graph;
pub mod teacher;
pub mod weights;
