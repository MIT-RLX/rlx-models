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

//! S2-Pro parity against `scripts/fish_reference.py s2`.
//!
//! `s2_two_layers` reads `s2_l2.safetensors` (the slow stack cut to two
//! layers, `--layers 2 --no-eos`); `s2_full` reads `s2.safetensors` and loads
//! the whole 4.5 B model (~15 GB in f32 on the CPU). Both need `RLX_FISH_DIR`
//! (the S2-Pro directory) and `RLX_FISH_REF`.

mod common;

use common::{Fixture, close, setup};
use rlx_fish::sampling::SampleParams;
use rlx_fish::{Rng, S2Model, S2Options};

fn run(file: &str, layers: Option<usize>, allow_eos: bool) {
    let Some((dir, reference, device)) = setup(file) else {
        return;
    };
    let fx = Fixture::open(&reference.join(file));
    let (prompt, shape) = fx.i64("prompt");
    let (width, t) = (shape[0], shape[1]);
    let rows: Vec<Vec<u32>> = (0..t)
        .map(|j| (0..width).map(|i| prompt[i * t + j] as u32).collect())
        .collect();

    let mut model = S2Model::open(
        &dir,
        device,
        S2Options {
            layers,
            ..Default::default()
        },
    )
    .unwrap();
    #[cfg(feature = "tokenizer")]
    {
        let tok = rlx_fish::prompt::load_tokenizer(&dir).unwrap();
        let conv = {
            let mut c = rlx_fish::s2::Conversation::new(&[]).unwrap();
            c.push_user("Hello from Fish Audio on RLX.");
            c
        };
        let built = conv.prompt_rows(&tok, &model.cfg).unwrap();
        assert_eq!(built, rows, "prompt rows");
    }

    let (hidden, hs) = fx.f32("hidden");
    let (band, _) = fx.f32("band_logits");
    let (fast, _) = fx.f32("fast_logits");
    let (frames, fs) = fx.i64("frames");
    let (steps, dim, nb) = (hs[0], hs[1], 4096);
    let nc = fs[1] - 1;

    let cap = 256;
    let mut state = model.prefill(&rows, cap).unwrap();
    let mut rng = Rng::new(0);
    for step in 0..steps {
        close(
            &format!("hidden[{step}]"),
            &state.hidden,
            &hidden[step * dim..(step + 1) * dim],
            2e-2,
        );
        close(
            &format!("band[{step}]"),
            &state.logits[..nb],
            &band[step * nb..(step + 1) * nb],
            5e-2,
        );
        let want: Vec<u32> = frames[step * fs[1]..(step + 1) * fs[1]]
            .iter()
            .map(|&v| v as u32)
            .collect();
        let pick = if allow_eos { nb + 1 } else { nb };
        let main = model.band_token(rlx_fish::sampling::sample(
            &state.logits[..pick],
            SampleParams::greedy(),
            rlx_fish::sampling::Flavor::S2,
            &mut rng,
        ));
        assert_eq!(main, want[0], "main token at step {step}");
        let codes = model
            .fast_frame(
                &state.hidden,
                main,
                SampleParams::greedy(),
                &mut rng,
                |pos, logits| {
                    let base = (step * (nc - 1) + pos - 1) * nb;
                    close(
                        &format!("fast[{step}][{pos}]"),
                        logits,
                        &fast[base..base + nb],
                        5e-2,
                    );
                },
            )
            .unwrap();
        assert_eq!(&codes[..], &want[1..], "codes at step {step}");
        if step + 1 < steps {
            let embed = model.embed_rows(&[want]).unwrap();
            model.step(&mut state, &embed).unwrap();
        }
    }
}

#[test]
fn s2_two_layers() {
    run("s2_l2.safetensors", Some(2), false);
}

#[test]
#[ignore = "loads the full 4.5 B checkpoint"]
fn s2_full() {
    run("s2.safetensors", None, true);
}
