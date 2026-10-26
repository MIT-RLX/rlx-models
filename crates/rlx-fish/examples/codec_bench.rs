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

//! Time the fish codec decoder: first call (build + compile + upload + run)
//! and a second call on the same bucket.
//!
//! `cargo run -p rlx-fish --release --features metal --example codec_bench --
//!  weights/tts/fish-s2-pro metal [frames]`

use std::time::Instant;

use anyhow::Result;
use rlx_fish::FishCodec;
use rlx_fish::codec::{NUM_CODEBOOKS, RESIDUAL_SIZE, SAMPLE_RATE, SEMANTIC_SIZE};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let device = rlx_cli::parse_device(args.get(2).map_or("cpu", String::as_str))?;
    let frames: usize = args.get(3).map_or(Ok(344), |s| s.parse())?;
    let codes: Vec<Vec<u32>> = (0..NUM_CODEBOOKS)
        .map(|cb| {
            let size = if cb == 0 {
                SEMANTIC_SIZE
            } else {
                RESIDUAL_SIZE
            } as u32;
            (0..frames as u32)
                .map(|t| (t * 2654435761u32.wrapping_add(cb as u32 * 97)) % size)
                .collect()
        })
        .collect();
    let mut codec = FishCodec::open(&dir, device)?;
    for pass in ["first", "second"] {
        let t = Instant::now();
        let pcm = codec.decode(&codes)?;
        let s = t.elapsed().as_secs_f32();
        let audio = pcm.len() as f32 / SAMPLE_RATE as f32;
        eprintln!(
            "{pass}: {s:.2} s for {audio:.1} s of audio ({:.1}x real time)",
            audio / s
        );
    }
    Ok(())
}
