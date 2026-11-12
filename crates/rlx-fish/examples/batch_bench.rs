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

//! Several sentences one after another (`S2Model::generate` each) against
//! all of them together (`S2Model::generate_batch`), on real weights: frames
//! per second, and whether every sentence's frames agree.
//!
//! cargo run -p rlx-fish --release --features metal --example batch_bench -- DIR DEVICE [N] [FRAMES] [greedy|sampled]

use std::time::Instant;

use rlx_fish::s2::{S2Model, S2Options};
use rlx_fish::sampling::{Rng, SampleParams};

const SENTENCES: [&str; 8] = [
    "The rain had stopped by the time we reached the station.",
    "She opened the letter slowly, afraid of what it might say.",
    "Our train leaves at seven, so please be ready by six thirty.",
    "He laughed, shook his head, and walked back into the kitchen.",
    "Nobody expected the old bridge to hold for another winter.",
    "Turn left at the bakery and the museum is right in front of you.",
    "I have never seen the lake this calm in the early morning.",
    "Thank you all for coming, and enjoy the rest of the evening.",
];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::PathBuf::from(&args[0]);
    let device = rlx_cli::parse_device(args.get(1).map_or("cpu", String::as_str))?;
    let n: usize = args.get(2).map_or(Ok(4), |s| s.parse())?;
    let frames: usize = args.get(3).map_or(Ok(64), |s| s.parse())?;
    let params = match args.get(4).map(String::as_str) {
        Some("sampled") => rlx_fish::audio8::SynthOptions::from_dir(&dir)?.params,
        _ => SampleParams::greedy(),
    };
    let tok = rlx_fish::prompt::load_tokenizer(&dir)?;
    let mut m = S2Model::open(&dir, device, S2Options::default())?;
    let prompts: Vec<_> = SENTENCES
        .iter()
        .cycle()
        .take(n)
        .map(|t| rlx_fish::audio8::prompt_rows(&tok, &m.cfg, t, None))
        .collect::<anyhow::Result<_>>()?;
    let seeds: Vec<u64> = (0..n as u64).collect();
    // Builds every graph once, untimed.
    m.generate(&prompts[0], 2, params, &mut Rng::new(0))?;
    m.generate_batch(&prompts, 2, params, &seeds)?;

    let t = Instant::now();
    let alone: Vec<_> = prompts
        .iter()
        .zip(&seeds)
        .map(|(p, &s)| m.generate(p, frames, params, &mut Rng::new(s)))
        .collect::<anyhow::Result<_>>()?;
    let t_alone = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let batch = m.generate_batch(&prompts, frames, params, &seeds)?;
    let t_batch = t.elapsed().as_secs_f64();

    let made: usize = alone.iter().map(Vec::len).sum();
    let made_b: usize = batch.iter().map(Vec::len).sum();
    let same = alone.iter().zip(&batch).filter(|(a, b)| a == b).count();
    let realtime = 44_100.0 / 2048.0;
    println!(
        "{n} sentences on {device:?}: one by one {:.1} frames/s ({made} frames), together {:.1} frames/s ({made_b} frames) = {:.2}x realtime, {same}/{n} identical",
        made as f64 / t_alone,
        made_b as f64 / t_batch,
        made_b as f64 / t_batch / realtime
    );
    for (i, (a, b)) in alone.iter().zip(&batch).enumerate() {
        if a != b {
            let first = a.iter().zip(b).position(|(x, y)| x != y);
            println!(
                "  sentence {i}: first differing frame {first:?} of {} / {}",
                a.len(),
                b.len()
            );
        }
    }
    Ok(())
}
