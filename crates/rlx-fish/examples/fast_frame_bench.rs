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
//! The fast decoder per frame: ten graph runs with host sampling
//! (`S2Model::fast_frame`) against one fused graph run
//! (`S2Model::fast_frame_fused`), on real weights — time per frame, and
//! whether the codes agree.
//!
//! cargo run -p rlx-fish --release --features metal --example fast_frame_bench -- DIR DEVICE [FRAMES]

use std::time::Instant;

use rlx_fish::s2::{S2Model, S2Options, Stage};
use rlx_fish::sampling::{Rng, SampleParams};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::PathBuf::from(&args[0]);
    let device = rlx_cli::parse_device(args.get(1).map_or("cpu", String::as_str))?;
    let frames: u64 = args.get(2).map_or(Ok(40), |s| s.parse())?;
    let cfg = rlx_fish::S2Config::from_dir(&dir)?;
    let n = cfg.slow.n_layer;
    let opts = S2Options {
        stage: Some(Stage {
            layers: n..n,
            last: false,
        }),
        ..S2Options::default()
    };
    let mut m = S2Model::open(&dir, device, opts)?;
    let dim = cfg.fast.dim;
    let sampled = rlx_fish::audio8::SynthOptions::from_dir(&dir)?.params;
    for (name, params) in [("greedy", SampleParams::greedy()), ("sampled", sampled)] {
        let (mut t_host, mut t_fused, mut same, mut codes_same, mut total) = (0f64, 0f64, 0, 0, 0);
        for i in 0..frames + 2 {
            let hidden: Vec<f32> = (0..dim)
                .map(|j| ((j as f32 + i as f32 * 7.0) * 0.37).sin() * 3.0)
                .collect();
            let main = cfg.semantic_begin + (i as u32 * 811) % cfg.codebook_size as u32;
            let t = Instant::now();
            let a = m.fast_frame(&hidden, main, params, &mut Rng::new(i), |_, _| {})?;
            let th = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let b = m.fast_frame_fused(&hidden, main, params, &mut Rng::new(i))?;
            let tf = t.elapsed().as_secs_f64();
            if i < 2 {
                continue; // graph builds
            }
            t_host += th;
            t_fused += tf;
            same += (a == b) as usize;
            codes_same += a.iter().zip(&b).filter(|(x, y)| x == y).count();
            total += a.len();
        }
        println!(
            "{name:8} {device:?}: host loop {:.1} ms/frame, fused {:.1} ms/frame; {same}/{frames} frames identical, {codes_same}/{total} codes",
            t_host * 1e3 / frames as f64,
            t_fused * 1e3 / frames as f64
        );
    }
    Ok(())
}
