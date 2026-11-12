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
//! Audio8-TTS-Realistic-Air parity against `scripts/audio8_reference.py`.
//!
//! Needs `RLX_AUDIO8_DIR` (the checkpoint) and `RLX_FISH_REF` (the script's
//! `--out`); skips otherwise. `RLX_FISH_DEVICE` picks the backend (default
//! `cpu`). The prompts come from the published `ArkttsProcessor`, the frames
//! from the published model's greedy path, and what a sampled draw chooses
//! among from its logits processors: torch's random numbers are not this
//! crate's, so sampling is compared up to the draw.
//!
//! Those comparisons run in strict f32 ([`rlx_runtime::with_strict_f32`]).
//! A backend's faster default (CUDA's TF32) moves the hidden states past the
//! tolerances by design; what it must not move is the greedy path, which
//! `greedy_frames_at_the_backends_default_precision` checks with no float
//! tolerance at all.

mod common;

use common::{Fixture, close, setup_with};
use rlx_fish::sampling::{Flavor, SampleParams, candidates, sample};
use rlx_fish::{FishCodec, Rng, S2Model, S2Options};

const DIR: &str = "RLX_AUDIO8_DIR";
const TEXT: &str = "Welcome to Audio8 on RLX. It is wonderful to hear from you.";
const JFK: &str = "And so my fellow Americans, ask not what your country can do for you, \
                   ask what you can do for your country.";

/// Sampling settings the reference script dumps its pools at.
const DRAW: SampleParams = SampleParams {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 50,
};

/// `logits` drawn from at `params` keeps the reference's candidates, at its
/// probabilities. A candidate the two disagree on is off by its whole
/// probability, so `close` sees it.
fn same_pool(what: &str, logits: &[f32], params: SampleParams, want: &[f32]) {
    let mut got = vec![0f32; logits.len()];
    for (index, p) in candidates(logits, params, Flavor::S2) {
        got[index] = p as f32;
    }
    close(what, &got, want, 2e-3);
}

/// The reference's frames and prompt from `file`, compared against the
/// runner. With `strict`, also every hidden state, logit and sampling pool;
/// without, only the greedy tokens and codes.
fn run(file: &str, reference_text: Option<&str>, strict: bool) {
    if strict {
        rlx_runtime::with_strict_f32(|| compare(file, reference_text, true));
    } else {
        compare(file, reference_text, false);
    }
}

fn compare(file: &str, reference_text: Option<&str>, strict: bool) {
    let close = |what: &str, got: &[f32], want: &[f32], tol: f32| {
        if strict {
            close(what, got, want, tol);
        }
    };
    let same_pool = |what: &str, logits: &[f32], params: SampleParams, want: &[f32]| {
        if strict {
            same_pool(what, logits, params, want);
        }
    };
    let Some((dir, reference, device)) = setup_with(DIR, file) else {
        return;
    };
    let fx = Fixture::open(&reference.join(file));
    let (prompt, shape) = fx.i64("prompt");
    let (width, t) = (shape[0], shape[1]);
    let rows: Vec<Vec<u32>> = (0..t)
        .map(|j| (0..width).map(|i| prompt[i * t + j] as u32).collect())
        .collect();

    let mut model = S2Model::open(&dir, device, S2Options::default()).unwrap();
    let voice = reference_text.map(|text| {
        // The fixture's own audio rows, so this compares the text half.
        let audio: Vec<&Vec<u32>> = rows
            .iter()
            .filter(|r| model.cfg.is_semantic(r[0]))
            .collect();
        rlx_fish::s2::Reference {
            text: text.to_string(),
            codes: (1..width)
                .map(|i| audio.iter().map(|r| r[i]).collect())
                .collect(),
        }
    });
    #[cfg(feature = "tokenizer")]
    {
        let tok = rlx_fish::prompt::load_tokenizer(&dir).unwrap();
        let built = rlx_fish::audio8::prompt_rows(&tok, &model.cfg, TEXT, voice.as_ref()).unwrap();
        assert_eq!(built, rows, "prompt rows");
    }
    #[cfg(not(feature = "tokenizer"))]
    let _ = voice;

    let (hidden, hs) = fx.f32("hidden");
    let (band, bs) = fx.f32("band_logits");
    let (fast, fs) = fx.f32("fast_logits");
    let (frames, frame_shape) = fx.i64("frames");
    let (pool, _) = fx.f32("pool");
    let (ras_pool, _) = fx.f32("ras_pool");
    let (fast_pool, _) = fx.f32("fast_pool");
    let (steps, dim, nb) = (hs[0], hs[1], bs[1]);
    let (per_frame, book) = (fs[1], fs[2]);

    let cap = (t + steps).div_ceil(256) * 256;
    let mut state = model.prefill(&rows, cap).unwrap();
    let mut rng = Rng::new(0);
    for step in 0..steps {
        close(
            &format!("hidden[{step}]"),
            &state.hidden,
            &hidden[step * dim..(step + 1) * dim],
            2e-3,
        );
        close(
            &format!("band[{step}]"),
            &state.logits,
            &band[step * nb..(step + 1) * nb],
            5e-3,
        );
        let row = step * nb..(step + 1) * nb;
        same_pool(
            &format!("pool[{step}]"),
            &state.logits,
            DRAW,
            &pool[row.clone()],
        );
        let ras = model.cfg.ras;
        let redraw = SampleParams {
            temperature: ras.temperature,
            top_p: ras.top_p,
            ..DRAW
        };
        same_pool(
            &format!("ras[{step}]"),
            &state.logits,
            redraw,
            &ras_pool[row],
        );
        let want: Vec<u32> = frames[step * frame_shape[1]..(step + 1) * frame_shape[1]]
            .iter()
            .map(|&v| v as u32)
            .collect();
        let main = model.band_token(sample(
            &state.logits,
            SampleParams::greedy(),
            Flavor::S2,
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
                    let base = (step * per_frame + pos - 1) * book;
                    close(
                        &format!("fast[{step}][{pos}]"),
                        logits,
                        &fast[base..base + book],
                        5e-3,
                    );
                    same_pool(
                        &format!("fast_pool[{step}][{pos}]"),
                        logits,
                        DRAW,
                        &fast_pool[base..base + book],
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
fn greedy_frames_match_the_reference() {
    run("audio8.safetensors", None, true);
}

#[test]
fn cloned_greedy_frames_match_the_reference() {
    run("audio8_clone.safetensors", Some(JFK), true);
}

#[test]
fn greedy_frames_at_the_backends_default_precision() {
    run("audio8.safetensors", None, false);
    run("audio8_clone.safetensors", Some(JFK), false);
}

#[test]
fn codec_decode_matches_the_reference() {
    let file = "audio8_codec.safetensors";
    let Some((dir, reference, device)) = setup_with(DIR, file) else {
        return;
    };
    let fx = Fixture::open(&reference.join(file));
    let (codes, shape) = fx.i64("codes");
    let rows: Vec<Vec<u32>> = codes
        .chunks(shape[1])
        .map(|r| r.iter().map(|&c| c as u32).collect())
        .collect();
    let mut codec = FishCodec::open(&dir, device).unwrap();
    let wav = codec.decode(&rows).unwrap();
    let (want, _) = fx.f32("wav");
    assert_eq!(wav.len(), want.len());
    close("wav", &wav, &want, 2e-3);
}

#[test]
fn codec_encode_matches_the_reference() {
    let file = "audio8_codec.safetensors";
    let Some((dir, reference, device)) = setup_with(DIR, file) else {
        return;
    };
    let fx = Fixture::open(&reference.join(file));
    let (pcm, _) = fx.f32("enc_pcm");
    let mut codec = FishCodec::open(&dir, device).unwrap();

    let latent = codec.encode_latent(&pcm).unwrap();
    let (want, shape) = fx.f32("enc_latent");
    // Reference latent is `[C, T]`; ours is rows `[T, C]`.
    let (c, t) = (shape[0], shape[1]);
    let mut want_rows = vec![0f32; c * t];
    for ci in 0..c {
        for ti in 0..t {
            want_rows[ti * c + ci] = want[ci * t + ti];
        }
    }
    close("encode latent", &latent, &want_rows, 1e-3);

    let codes = codec.encode(&pcm).unwrap();
    let (want, ws) = fx.i64("enc_codes");
    let total = want.len();
    let same: usize = codes
        .iter()
        .flatten()
        .zip(&want)
        .filter(|(a, b)| **a as i64 == **b)
        .count();
    assert_eq!(codes.len(), ws[0]);
    assert_eq!(same, total, "{same}/{total} codes agree");
}

/// The DAC decoder runs window by window (`FishCodec::decode_latent`); over a
/// long utterance (300 frames, ten windows) that must be the whole-utterance
/// graph's audio.
#[test]
fn codec_windows_match_the_whole_graph() {
    let Some(dir) = rlx_ir::env::var(DIR).map(std::path::PathBuf::from) else {
        return;
    };
    let device = rlx_ir::env::var("RLX_FISH_DEVICE")
        .map(|d| rlx_cli::parse_device(&d).unwrap())
        .unwrap_or(rlx_runtime::Device::Cpu);
    let frames = 300;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = |n: u32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as u32
    };
    let codes: Vec<Vec<u32>> = (0..10)
        .map(|i| {
            let n = if i == 0 { 4096 } else { 1024 };
            (0..frames).map(|_| next(n)).collect()
        })
        .collect();
    let mut codec = FishCodec::open(&dir, device).unwrap();
    let z = codec.dequantize(&codes).unwrap();
    let windowed = codec.decode_latent(&z, frames).unwrap();
    let whole = codec.decode_latent_whole(&z, frames).unwrap();
    assert_eq!(windowed.len(), whole.len());
    close("windowed vs whole", &windowed, &whole, 1e-5);
}

/// The codec's safetensors export (`rlx-fish export-web`, in
/// `RLX_AUDIO8_WEB_DIR`, a directory without `codec.pth`) decodes and encodes
/// exactly as `codec.pth` does.
#[test]
fn the_exported_codec_is_the_checkpoints() {
    let (Some(dir), Some(web)) = (
        rlx_ir::env::var(DIR).map(std::path::PathBuf::from),
        rlx_ir::env::var("RLX_AUDIO8_WEB_DIR").map(std::path::PathBuf::from),
    ) else {
        return;
    };
    assert!(
        !web.join("codec.pth").exists(),
        "{} holds codec.pth",
        web.display()
    );
    let device = rlx_ir::env::var("RLX_FISH_DEVICE")
        .map(|d| rlx_cli::parse_device(&d).unwrap())
        .unwrap_or(rlx_runtime::Device::Cpu);
    let mut pth = FishCodec::open(&dir, device).unwrap();
    let mut st = FishCodec::open(&web, device).unwrap();
    let codes: Vec<Vec<u32>> = (0..10u32)
        .map(|i| {
            let n = if i == 0 { 4096 } else { 1024 };
            (0..70u32).map(|t| (t * 7919 + i * 104_729) % n).collect()
        })
        .collect();
    assert_eq!(st.decode(&codes).unwrap(), pth.decode(&codes).unwrap());
    let pcm: Vec<f32> = (0..3 * 44_100)
        .map(|i| ((i as f32) * 0.031).sin() * 0.3 + ((i as f32) * 0.0047).sin() * 0.2)
        .collect();
    assert_eq!(st.encode(&pcm).unwrap(), pth.encode(&pcm).unwrap());
}

/// The real model split the way the browser splits it — two slow stages and
/// a frame maker — samples the frames `S2Model::generate` does.
#[test]
fn a_pipeline_of_stages_samples_what_generate_does() {
    use rlx_fish::audio8::{FrameMaker, SlowStage, StageInput, cache_capacity};
    use rlx_fish::s2::Stage;
    let Some(dir) = rlx_ir::env::var(DIR).map(std::path::PathBuf::from) else {
        return;
    };
    let device = rlx_ir::env::var("RLX_FISH_DEVICE")
        .map(|d| rlx_cli::parse_device(&d).unwrap())
        .unwrap_or(rlx_runtime::Device::Cpu);
    let open = |stage: Option<Stage>| {
        S2Model::open(
            &dir,
            device,
            S2Options {
                stage,
                ..S2Options::default()
            },
        )
        .unwrap()
    };
    let params = rlx_fish::audio8::SynthOptions::from_dir(&dir)
        .unwrap()
        .params;
    let (max_new, seed) = (32, 3);
    let tok = rlx_fish::prompt::load_tokenizer(&dir).unwrap();
    let mut whole = open(None);
    let rows = rlx_fish::audio8::prompt_rows(&tok, &whole.cfg, TEXT, None).unwrap();
    let want: Vec<Vec<u32>> = whole
        .generate(&rows, max_new, params, &mut Rng::new(seed))
        .unwrap()
        .iter()
        .map(|f| {
            std::iter::once(f.main)
                .chain(f.codes.iter().copied())
                .collect()
        })
        .collect();
    drop(whole);

    let n = 24;
    let mut first = SlowStage::new(open(Some(Stage {
        layers: 0..10,
        last: false,
    })));
    let mut last = SlowStage::new(open(Some(Stage {
        layers: 10..n,
        last: true,
    })));
    let mut maker = FrameMaker::new(
        open(Some(Stage {
            layers: n..n,
            last: false,
        })),
        params,
        seed,
    );
    let cap = cache_capacity(&first.model.cfg, rows.len(), max_new);
    let h = first.prefill(StageInput::Rows(&rows), cap).unwrap();
    last.prefill(StageInput::Hidden(&h), cap).unwrap();
    let mut got = Vec::new();
    for step in 0..max_new {
        let Some(row) = maker.next(last.hidden(), last.logits()).unwrap() else {
            break;
        };
        got.push(row.clone());
        if step + 1 < max_new {
            let h = first.step(StageInput::Rows(&[row])).unwrap();
            last.step(StageInput::Hidden(&h)).unwrap();
        }
    }
    eprintln!("{} of {} frames", got.len(), want.len());
    assert_eq!(got, want);
}

/// What streaming relies on: decoding the frames so far gives the start of
/// the final audio, and `decode_from` returns exactly the full decode's
/// samples from a point on.
#[test]
fn the_codec_streams_what_it_decodes_whole() {
    let Some(dir) = rlx_ir::env::var(DIR).map(std::path::PathBuf::from) else {
        return;
    };
    let device = rlx_ir::env::var("RLX_FISH_DEVICE")
        .map(|d| rlx_cli::parse_device(&d).unwrap())
        .unwrap_or(rlx_runtime::Device::Cpu);
    let frames = 150;
    let codes: Vec<Vec<u32>> = (0..10u32)
        .map(|i| {
            let n = if i == 0 { 4096 } else { 1024 };
            (0..frames as u32)
                .map(|t| (t * 7919 + i * 104_729) % n)
                .collect()
        })
        .collect();
    let mut codec = FishCodec::open(&dir, device).unwrap();
    let whole = codec.decode(&codes).unwrap();
    for from in [0, 1, 70_000, 2048 * 64, whole.len() - 5] {
        assert_eq!(
            codec.decode_from(&codes, from).unwrap(),
            whole[from..],
            "from {from}"
        );
    }
    // A prefix of the frames: its own bucket and windows, the same samples.
    for f in [20, 64, 97] {
        let prefix: Vec<Vec<u32>> = codes.iter().map(|c| c[..f].to_vec()).collect();
        let part = codec.decode(&prefix).unwrap();
        close(
            &format!("prefix of {f} frames"),
            &part,
            &whole[..part.len()],
            1e-5,
        );
    }
}

/// `generate` — the fused frame graph, sampling in the graph — draws the
/// reference's greedy frames.
#[test]
fn greedy_generate_matches_the_reference() {
    let file = "audio8.safetensors";
    let Some((dir, reference, device)) = setup_with(DIR, file) else {
        return;
    };
    let fx = Fixture::open(&reference.join(file));
    let (prompt, shape) = fx.i64("prompt");
    let (width, t) = (shape[0], shape[1]);
    let rows: Vec<Vec<u32>> = (0..t)
        .map(|j| (0..width).map(|i| prompt[i * t + j] as u32).collect())
        .collect();
    let (frames, fs) = fx.i64("frames");
    let want: Vec<Vec<u32>> = frames
        .chunks(fs[1])
        .map(|r| r.iter().map(|&v| v as u32).collect())
        .collect();
    let mut model = S2Model::open(&dir, device, S2Options::default()).unwrap();
    let got = model
        .generate(&rows, want.len(), SampleParams::greedy(), &mut Rng::new(0))
        .unwrap();
    let got: Vec<Vec<u32>> = got
        .iter()
        .map(|f| {
            std::iter::once(f.main)
                .chain(f.codes.iter().copied())
                .collect()
        })
        .collect();
    assert_eq!(got, want);
}
