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

//! Native RLX graphs for ocrs detection (U-Net) and recognition (CRNN + GRU).

mod detection;
mod recognition;
pub mod weights;

pub use detection::{DetectionGraphConfig, build_detection_graph, build_detection_graph_to_stage};
pub use recognition::{
    NUM_CLASSES, RecognitionGraphConfig, build_recognition_after_g1_graph,
    build_recognition_after_g2_graph, build_recognition_after_logits_graph,
    build_recognition_conv_graph, build_recognition_graph, log_softmax_last_axis,
};
