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

//! `config.json` of the dual-AR checkpoints: S2-Pro and Audio8.
//! Streaming decode cost: an utterance decoded 16 frames at a time with
//! `FishCodec::decode_from`, as `rlx-fish audio8 --play` does, on one device.
//! Real time is 16 frames per 743 ms.
//!
//! cargo run -p rlx-fish --release --features metal --example codec_stream_bench -- DIR DEVICE [FRAMES]

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::PathBuf::from(&args[0]);
    let device = rlx_cli::parse_device(args.get(1).map_or("cpu", String::as_str))?;
    let frames: usize = args.get(2).map_or(Ok(215), |s| s.parse())?;
    let codes: Vec<Vec<u32>> = (0..10u32)
        .map(|i| {
            let n = if i == 0 { 4096 } else { 1024 };
            (0..frames as u32)
                .map(|t| (t * 7919 + i * 104_729) % n)
                .collect()
        })
        .collect();
    let mut codec = rlx_fish::FishCodec::open(&dir, device)?;
    // Build every graph this length needs, as `--play` does before it starts.
    codec.prepare(frames)?;
    let mut done = 0usize;
    let mut total = 0f64;
    let mut worst = 0f64;
    for f in (16..=frames).step_by(16).chain(std::iter::once(frames)) {
        let part: Vec<Vec<u32>> = codes.iter().map(|c| c[..f].to_vec()).collect();
        let t = Instant::now();
        let new = codec.decode_from(&part, done)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        done += new.len();
        total += ms;
        worst = worst.max(ms);
    }
    // The upsample stage alone: ask for samples past the end, so no DAC
    // window runs.
    let mut up = 0f64;
    for f in (16..=frames).step_by(16) {
        let part: Vec<Vec<u32>> = codes.iter().map(|c| c[..f].to_vec()).collect();
        let t = Instant::now();
        codec.decode_from(&part, usize::MAX)?;
        up += t.elapsed().as_secs_f64() * 1e3;
    }
    println!("of which the upsample stage: {up:.0} ms");
    let audio = done as f64 / 44_100.0;
    println!(
        "{frames} frames ({audio:.1} s) in 16-frame steps on {device:?}: decode {total:.0} ms total, worst step {worst:.0} ms (real time allows 743 ms a step); {:.2}x real time",
        audio * 1e3 / total
    );
    Ok(())
}
