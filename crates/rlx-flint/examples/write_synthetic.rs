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

//! Write the tiny synthetic checkpoint the tests use, for the torch reference:
//!
//! ```text
//! cargo run -p rlx-flint --example write_synthetic -- /tmp/flint-synth [seed]
//! python3 scripts/flint_reference.py --model-dir /tmp/flint-synth \
//!     --ids 3,17,42,5,88,13,21,60,7,33,2,71,50 --out /tmp/flint-synth-ref.safetensors
//! ```

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: write_synthetic DIR [SEED]"))?;
    let seed: u64 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(7);
    rlx_flint::synth::write_checkpoint(
        std::path::Path::new(&dir),
        &rlx_flint::synth::tiny_config_json(),
        rlx_flint::synth::SynthDtype::F32,
        3,
        seed,
    )?;
    println!("wrote {dir} (seed {seed})");
    Ok(())
}
