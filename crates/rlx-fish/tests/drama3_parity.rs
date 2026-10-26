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

//! Drama 3 parity against `scripts/fish_reference.py drama3 --layers 4
//! --no-eos` (three Gated DeltaNet layers and one gated-attention layer, the
//! full codebook tower, greedy, repetition-aware sampling off).
//!
//! Needs `RLX_FISH_DRAMA3_DIR` (the Drama 3 directory) and `RLX_FISH_REF`.

mod common;

use common::{Fixture, close, setup_with};
use rlx_fish::drama3::StepObserver;
use rlx_fish::sampling::SampleParams;
use rlx_fish::{DramaModel, DramaOptions, Rng};

struct Check<'a> {
    hidden: &'a [f32],
    main: &'a [f32],
    heads: &'a [f32],
    d: usize,
}

impl StepObserver for Check<'_> {
    fn hidden(&mut self, step: usize, h: &[f32]) {
        close(
            &format!("hidden[{step}]"),
            h,
            &self.hidden[step * self.d..(step + 1) * self.d],
            5e-3,
        );
    }
    fn main_logits(&mut self, step: usize, l: &[f32]) {
        let n = l.len();
        close(
            &format!("main[{step}]"),
            l,
            &self.main[step * n..(step + 1) * n],
            5e-2,
        );
    }
    fn head_logits(&mut self, step: usize, head: usize, l: &[f32]) {
        let n = l.len();
        let base = (step * 9 + head) * n;
        close(
            &format!("head[{step}][{head}]"),
            l,
            &self.heads[base..base + n],
            5e-2,
        );
    }
}

#[test]
fn drama3_four_layers() {
    let file = "drama3_l4.safetensors";
    let Some((dir, reference, device)) = setup_with("RLX_FISH_DRAMA3_DIR", file) else {
        return;
    };
    let fx = Fixture::open(&reference.join(file));
    let (prompt, ps) = fx.i64("prompt");
    let rows: Vec<Vec<u32>> = prompt
        .chunks(ps[1])
        .map(|r| r.iter().map(|&v| v as u32).collect())
        .collect();
    let (hidden, hs) = fx.f32("hidden");
    let (main, _) = fx.f32("main_logits");
    let (heads, _) = fx.f32("head_logits");
    let (frames, fs) = fx.i64("frames");

    #[cfg(feature = "tokenizer")]
    {
        let tok = rlx_fish::prompt::load_tokenizer(&dir).unwrap();
        let segs =
            rlx_fish::drama3_text::prompt_segments("Hello from Fish Audio on RLX.", &[], &[])
                .unwrap();
        let built = rlx_fish::prompt::rows(&segs, &tok, 10, 248_099).unwrap();
        assert_eq!(built, rows, "prompt rows");
    }

    let opts = DramaOptions {
        layers: Some(4),
        ..Default::default()
    };
    let mut model = DramaModel::open(&dir, device, opts).unwrap();
    let mut check = Check {
        hidden: &hidden,
        main: &main,
        heads: &heads,
        d: hs[1],
    };
    let mut rng = Rng::new(0);
    let got = model
        .generate(
            &rows,
            fs[0],
            SampleParams::greedy(),
            &mut rng,
            false,
            &mut check,
        )
        .unwrap();
    let want: Vec<Vec<u32>> = frames
        .chunks(fs[1])
        .map(|r| r.iter().map(|&v| v as u32).collect())
        .collect();
    assert_eq!(got, want);
    let (hits, misses) = model.experts().cache_stats();
    eprintln!("expert cache: {hits} hits, {misses} misses");
}
