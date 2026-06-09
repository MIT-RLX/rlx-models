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

//! Fast native compile check (no inference). Usage:
//!   KITTEN_RLX_WEIGHTS=weights cargo run --example compile_check --release

fn main() -> anyhow::Result<()> {
    let weights = std::env::var("KITTEN_RLX_WEIGHTS").unwrap_or_else(|_| "weights".into());
    let opts = kitten_tts_mini_rlx::GraphOptions {
        sequence_length: 128,
        max_waveform_samples: 24_000,
    };
    let weights_data = kitten_tts_mini_rlx::load_weights(std::path::Path::new(&weights))?;
    let (hir, params) = kitten_tts_mini_rlx::build_hir(&weights_data, &opts)?;

    let mut compiled = rlx_runtime::Session::new(rlx_runtime::Device::Cpu)
        .compile_hir_with(hir, &rlx_runtime::CompileOptions::default())
        .map_err(|e| anyhow::anyhow!("{0}", e))?;
    for (name, data) in params {
        compiled.set_param(name.as_str(), &data);
    }
    eprintln!("compile ok");
    Ok(())
}
