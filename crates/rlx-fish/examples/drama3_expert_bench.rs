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

//! Time Drama 3's host experts for one decode row across all 40 layers,
//! first touch (pages faulted from disk) then warm.
//!
//! `cargo run -p rlx-fish --release --example drama3_expert_bench -- weights/tts/fish-drama-3`

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use rlx_fish::moe::ExpertBank;
use rlx_fish::store::ShardStore;

fn main() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("model dir"));
    let store = Arc::new(ShardStore::open(&dir)?);
    let bank = ExpertBank::new(store, "model.language_model.layers.", 2048, 512, 8, true, 0);
    let x: Vec<f32> = (0..2048)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    for pass in ["cold", "warm"] {
        let t = Instant::now();
        for l in 0..40 {
            // A fixed, distinct pick of 8 experts per layer.
            let router: Vec<f32> = (0..256)
                .map(|e| if (e + l * 8) % 32 == 0 { 5.0 } else { 0.0 })
                .collect();
            bank.forward(l, &x, &router, 1)?;
        }
        eprintln!(
            "{pass}: {:.0} ms for 40 layers × 8 experts",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(())
}
