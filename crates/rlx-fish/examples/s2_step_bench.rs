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

//! Time S2-Pro's per-frame pieces: one slow decode step and one fast frame.
//!
//! `cargo run -p rlx-fish --release --features metal --example s2_step_bench --
//!  weights/tts/fish-s2-pro metal [steps] [kv capacity]` (`RLX_WEIGHT_STORAGE=f32|bf16`).

use std::time::Instant;

use anyhow::Result;
use rlx_fish::{Rng, S2Model, S2Options, SampleParams};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let device = rlx_cli::parse_device(args.get(2).map_or("cpu", String::as_str))?;
    let steps: usize = args.get(3).map_or(Ok(8), |s| s.parse())?;
    let cap: usize = args.get(4).map_or(Ok(256), |s| s.parse())?;
    let t = Instant::now();
    let mut m = S2Model::open(&dir, device, S2Options::default())?;
    let rows: Vec<Vec<u32>> = (0..24)
        .map(|i| {
            let mut r = vec![0u32; 11];
            r[0] = 1000 + i;
            r
        })
        .collect();
    let mut st = m.prefill(&rows, cap)?;
    eprintln!("open + prefill {:.1} s", t.elapsed().as_secs_f32());
    let embed = m.embed_rows(&rows[..1])?;
    let t = Instant::now();
    m.step(&mut st, &embed)?;
    eprintln!(
        "first decode step (compile + upload) {:.1} s",
        t.elapsed().as_secs_f32()
    );
    // Best of 4 batches: the box is shared and load comes and goes.
    let mut best = f64::MAX;
    for _ in 0..4 {
        let t = Instant::now();
        for _ in 0..steps {
            m.step(&mut st, &embed)?;
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / steps as f64);
    }
    eprintln!("slow step {best:.1} ms (best of 4 × {steps})");
    let mut rng = Rng::new(0);
    let h = st.hidden.clone();
    let t = Instant::now();
    m.fast_frame(
        &h,
        m.cfg.semantic_begin,
        SampleParams::default(),
        &mut rng,
        |_, _| {},
    )?;
    eprintln!(
        "first fast frame (compile + upload) {:.1} s",
        t.elapsed().as_secs_f32()
    );
    let mut best = f64::MAX;
    for _ in 0..4 {
        let t = Instant::now();
        for _ in 0..steps {
            m.fast_frame(
                &h,
                m.cfg.semantic_begin,
                SampleParams::default(),
                &mut rng,
                |_, _| {},
            )?;
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / steps as f64);
    }
    eprintln!("fast frame {best:.1} ms (best of 4 × {steps})");
    Ok(())
}
