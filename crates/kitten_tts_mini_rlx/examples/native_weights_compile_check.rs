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

//! Native weights-only compile check (no `graph.json`).

fn main() -> anyhow::Result<()> {
    let weights = std::env::var("KITTEN_RLX_WEIGHTS").unwrap_or_else(|_| "weights".into());
    let opts = kitten_tts_mini_rlx::GraphOptions {
        sequence_length: 128,
        max_waveform_samples: 24_000,
    };
    kitten_tts_mini_rlx::compile_native(
        rlx_runtime::Device::Cpu,
        std::path::Path::new(&weights),
        &opts,
    )?;
    eprintln!("native weights compile ok");
    Ok(())
}
