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

//! Bundle path compile check (uses `compile_from_bundle`, not `graph.rs`).

fn main() -> anyhow::Result<()> {
    let weights = std::env::var("KITTEN_RLX_WEIGHTS").unwrap_or_else(|_| "weights".into());
    let bundle =
        std::env::var("KITTEN_RLX_BUNDLE").unwrap_or_else(|_| format!("{weights}/rlx_bundle"));
    let opts = kitten_tts_mini_rlx::GraphOptions {
        sequence_length: 128,
        max_waveform_samples: 24_000,
    };
    kitten_tts_mini_rlx::bundle_compile::compile_from_bundle(
        rlx_runtime::Device::Cpu,
        std::path::Path::new(&bundle),
        &opts,
    )?;
    eprintln!("bundle compile ok");
    Ok(())
}
