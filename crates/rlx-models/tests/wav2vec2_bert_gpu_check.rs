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

//! basic test: tiny synthetic Wav2Vec2-BERT graph on wgpu (Device::Gpu).
//!
//! Uses the default wgpu backend for this platform (Metal on macOS).

#[path = "wav2vec2_bert/support.rs"]
mod support;

#[cfg(feature = "gpu")]
#[test]
fn wav2vec2_bert_tiny_graph_runs_on_gpu() {
    mod compile_support;

    use rlx_runtime::Device;

    if !rlx_wgpu::is_available() {
        eprintln!("skip: no wgpu adapter");
        return;
    }
    support::run_tiny_graph(Device::Gpu);
}
