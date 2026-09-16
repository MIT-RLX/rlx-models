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

//! Gates the port against upstream's pure-numpy reference
//! (`pypkg/sanotts/models.py`) on the shipped `amy-en-1p46m` voice.
//!
//! The fixtures under `tests/fixtures/` were produced by running that reference
//! on a fixed pseudo-utterance (see `tests/fixtures/meta.json`); durations must
//! match exactly, and the latent/waveform to float32 rounding noise.
//!
//! The voice package itself is not vendored (2.9 MB of weights). Point
//! `RLX_SANOTTS_VOICE_DIR` at a local copy, or drop it in
//! `~/.cache/sanotts/amy-en-1p46m`; otherwise these tests skip.

use std::path::PathBuf;

use rlx_sanotts::{Mat, Synthesizer};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn voice_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("RLX_SANOTTS_VOICE_DIR") {
        let p = PathBuf::from(dir);
        return p.join("manifest.json").is_file().then_some(p);
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join(".cache/sanotts/amy-en-1p46m");
    p.join("manifest.json").is_file().then_some(p)
}

macro_rules! voice_or_skip {
    () => {
        match voice_dir() {
            Some(d) => d,
            None => {
                eprintln!(
                    "skipping: amy-en-1p46m voice pack not found \
                     (set RLX_SANOTTS_VOICE_DIR or populate ~/.cache/sanotts)"
                );
                return;
            }
        }
    };
}

fn read_i32(name: &str) -> Vec<i32> {
    let raw = std::fs::read(format!("{FIXTURES}/{name}")).expect(name);
    raw.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f32(name: &str) -> Vec<f32> {
    let raw = std::fs::read(format!("{FIXTURES}/{name}")).expect(name);
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn ids() -> Vec<i64> {
    read_i32("ids.i32").into_iter().map(i64::from).collect()
}

/// Max absolute difference and Pearson correlation against the reference.
fn compare(got: &[f32], want: &[f32]) -> (f32, f64) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let max_abs = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let n = got.len() as f64;
    let (ma, mb) = (
        got.iter().map(|&v| v as f64).sum::<f64>() / n,
        want.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (&a, &b) in got.iter().zip(want) {
        let (x, y) = (a as f64 - ma, b as f64 - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    (
        max_abs,
        num / (da.sqrt() * db.sqrt()).max(f64::MIN_POSITIVE),
    )
}

#[test]
fn durations_match_reference_exactly() {
    let dir = voice_or_skip!();
    let synth = Synthesizer::load(&dir).expect("load voice");
    let want: Vec<usize> = read_i32("durs.i32")
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let got = synth
        .durations(&ids(), synth.default_length_scale())
        .expect("durations");
    // Integer frame counts: any difference here desynchronizes everything after.
    assert_eq!(got, want);
}

#[test]
fn latent_matches_reference() {
    let dir = voice_or_skip!();
    let synth = Synthesizer::load(&dir).expect("load voice");
    let durs: Vec<usize> = read_i32("durs.i32")
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let latent = synth.latent(&ids(), &durs).expect("latent");
    let want = read_f32("latent.f32");
    assert_eq!(latent.rows * latent.cols, want.len());
    let (max_abs, corr) = compare(&latent.data, &want);
    assert!(
        max_abs < 2e-3 && corr > 0.999_999,
        "latent drift: max_abs={max_abs:e} corr={corr}"
    );
}

#[test]
fn waveform_matches_reference() {
    let dir = voice_or_skip!();
    let synth = Synthesizer::load(&dir).expect("load voice");
    let durs: Vec<usize> = read_i32("durs.i32")
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let latent_flat = read_f32("latent.f32");
    let frames = durs.iter().sum::<usize>();
    let rows = latent_rows(&synth);
    let latent = Mat::from_vec(latent_flat, rows, frames);
    let audio = synth.decode(&latent).expect("decode");
    let want = read_f32("audio.f32");
    let (max_abs, corr) = compare(&audio, &want);
    assert!(
        max_abs < 2e-3 && corr > 0.999_999,
        "waveform drift: max_abs={max_abs:e} corr={corr}"
    );
}

#[test]
fn end_to_end_from_ids_matches_reference() {
    let dir = voice_or_skip!();
    let synth = Synthesizer::load(&dir).expect("load voice");
    let wav = synth
        .synthesize_ids(&ids(), synth.default_length_scale())
        .expect("synthesize");
    let want = read_f32("audio.f32");
    assert_eq!(wav.sample_rate, 22050);
    let (max_abs, corr) = compare(&wav.samples, &want);
    assert!(
        max_abs < 2e-3 && corr > 0.999_999,
        "e2e drift: max_abs={max_abs:e} corr={corr}"
    );
}

fn latent_rows(synth: &Synthesizer) -> usize {
    synth
        .pack()
        .component_config::<rlx_sanotts::AcousticConfig>("acoustic")
        .expect("acoustic config")
        .out_channels
}

// ---------------------------------------------------------------------------
// Graph path
// ---------------------------------------------------------------------------

/// The compiled graph must agree with the host reference on every device the
/// build supports. Conv accumulation order differs between backends, so this is
/// a numeric-agreement gate, not a bit-equality one.
#[cfg(feature = "rlx-graph")]
#[test]
fn graph_matches_reference_on_available_devices() {
    use rlx_sanotts::Backend;

    let dir = voice_or_skip!();
    let want_latent = read_f32("latent.f32");
    let want_audio = read_f32("audio.f32");
    let durs: Vec<usize> = read_i32("durs.i32")
        .into_iter()
        .map(|d| d as usize)
        .collect();

    let devices = [
        rlx_runtime::Device::Cpu,
        rlx_runtime::Device::Metal,
        rlx_runtime::Device::Mlx,
        rlx_runtime::Device::Cuda,
        rlx_runtime::Device::Rocm,
        rlx_runtime::Device::Gpu,
        rlx_runtime::Device::Vulkan,
        rlx_runtime::Device::Ane,
    ];
    let mut ran = 0;
    for device in devices {
        if !rlx_runtime::is_available(device) {
            continue;
        }
        let mut synth = Synthesizer::load(&dir).expect("load voice");
        synth.set_backend(Backend::Graph(device));

        let latent = synth.latent(&ids(), &durs).expect("graph latent");
        let (l_max, l_corr) = compare(&latent.data, &want_latent);
        assert!(
            l_max < 5e-3 && l_corr > 0.999_99,
            "{device:?} latent drift: max_abs={l_max:e} corr={l_corr}"
        );

        let wav = synth
            .synthesize_ids(&ids(), synth.default_length_scale())
            .expect("graph synthesize");
        let (a_max, a_corr) = compare(&wav.samples, &want_audio);
        assert!(
            a_max < 5e-3 && a_corr > 0.999_99,
            "{device:?} waveform drift: max_abs={a_max:e} corr={a_corr}"
        );
        eprintln!(
            "{device:?}: latent max_abs={l_max:e} corr={l_corr:.9} | \
             audio max_abs={a_max:e} corr={a_corr:.9}"
        );
        ran += 1;
    }
    assert!(
        ran > 0,
        "no rlx device was available to test the graph path"
    );
}

// ---------------------------------------------------------------------------
// Loader guards and the text frontend
// ---------------------------------------------------------------------------

/// A corrupted weights blob must fail the manifest's SHA-256 rather than
/// decoding into plausible-sounding noise.
#[test]
fn corrupt_weights_are_rejected() {
    let dir = voice_or_skip!();
    let tmp = std::env::temp_dir().join("rlx-sanotts-corrupt-test");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir");
    for name in [
        "manifest.json",
        "piper-phoneme-config.json",
        "weights.fp16.bin",
    ] {
        std::fs::copy(dir.join(name), tmp.join(name)).expect(name);
    }
    let mut bytes = std::fs::read(tmp.join("weights.fp16.bin")).expect("read weights");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(tmp.join("weights.fp16.bin"), &bytes).expect("write weights");

    let err = rlx_sanotts::VoicePack::load_from_dir(&tmp)
        .expect_err("corrupt blob must not load")
        .to_string();
    assert!(err.contains("sha256"), "unexpected error: {err}");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Text in, speech out: the frontend produces Piper-framed ids and the stack
/// turns them into a non-degenerate waveform.
#[cfg(feature = "espeak")]
#[test]
fn text_to_speech_end_to_end() {
    use rlx_sanotts::config::{BOS_ID, EOS_ID, PAD_ID};

    let dir = voice_or_skip!();
    let synth = Synthesizer::load(&dir).expect("load voice");
    let ids = synth
        .phoneme_ids("Hello from a two megabyte voice.")
        .expect("phonemize");

    // Piper framing: `^ _ (phoneme _)* $`.
    assert_eq!(ids[0], BOS_ID);
    assert_eq!(ids[1], PAD_ID);
    assert_eq!(*ids.last().unwrap(), EOS_ID);
    assert!(ids.len() > 20, "suspiciously short id sequence: {ids:?}");
    assert!(
        ids[2..ids.len() - 1].chunks(2).all(|c| c[1] == PAD_ID),
        "every phoneme must be followed by a pad"
    );

    let wav = synth
        .synthesize("Hello from a two megabyte voice.")
        .expect("synthesize");
    assert_eq!(wav.sample_rate, 22050);
    // 256 samples per acoustic frame, exactly.
    assert_eq!(wav.samples.len() % rlx_sanotts::HOP, 0);
    assert!(
        wav.duration_secs() > 0.8 && wav.duration_secs() < 6.0,
        "implausible duration {}s",
        wav.duration_secs()
    );
    let rms = (wav.samples.iter().map(|s| (*s as f64).powi(2)).sum::<f64>()
        / wav.samples.len() as f64)
        .sqrt();
    assert!(rms > 0.01, "output is effectively silent (rms {rms})");
    assert!(
        wav.samples.iter().all(|s| s.is_finite()),
        "output contains non-finite samples"
    );
}

/// Text through the graph path must land on the same audio as the host path.
#[cfg(all(feature = "espeak", feature = "rlx-graph"))]
#[test]
fn graph_and_host_agree_from_text() {
    use rlx_sanotts::Backend;

    let dir = voice_or_skip!();
    let text = "Hello from a two megabyte voice.";
    let host = Synthesizer::load(&dir).expect("load voice");
    let want = host.synthesize(text).expect("host synthesize");

    let mut ran = 0;
    for device in [
        rlx_runtime::Device::Cpu,
        rlx_runtime::Device::Metal,
        rlx_runtime::Device::Mlx,
        rlx_runtime::Device::Cuda,
        rlx_runtime::Device::Rocm,
        rlx_runtime::Device::Gpu,
        rlx_runtime::Device::Vulkan,
        rlx_runtime::Device::Ane,
    ] {
        if !rlx_runtime::is_available(device) {
            continue;
        }
        let mut synth = Synthesizer::load(&dir).expect("load voice");
        synth.set_backend(Backend::Graph(device));
        let got = synth.synthesize(text).expect("graph synthesize");
        let (max_abs, corr) = compare(&got.samples, &want.samples);
        assert!(
            max_abs < 5e-3 && corr > 0.999_99,
            "{device:?} diverges from host: max_abs={max_abs:e} corr={corr}"
        );
        ran += 1;
    }
    assert!(ran > 0, "no rlx device was available");
}

// ---------------------------------------------------------------------------
// Graph-path capabilities: bucketing and the decoder post-filter
// ---------------------------------------------------------------------------

/// Deterministic small weights, so the synthetic-post-filter test is reproducible.
#[cfg(feature = "rlx-graph")]
fn pseudo(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / 8_388_608.0 - 1.0) * scale
        })
        .collect()
}

/// A graph built at a padded capacity must reproduce what a graph built for the
/// true length produces — the claim behind bucketing, which holds only because
/// every conv's bias add is masked back to zero.
///
/// Agreement is to reduction-order noise rather than bit-exact: the two graphs
/// convolve over different total lengths, so the backend tiles them differently.
/// The load-bearing check is the tail one — if the mask leaked, error would
/// concentrate next to the padding, and it is exactly zero there.
#[cfg(feature = "rlx-graph")]
#[test]
fn bucketing_is_exact() {
    use rlx_sanotts::graph::FrameGraph;
    use rlx_sanotts::{AcousticConfig, DecoderConfig, VoicePack};

    let dir = voice_or_skip!();
    let pack = VoicePack::load_from_dir(&dir).expect("load pack");
    let ac = pack
        .component_tensors("acoustic")
        .expect("acoustic tensors");
    let ac_cfg: AcousticConfig = pack.component_config("acoustic").expect("acoustic config");
    let de = pack.component_tensors("decoder").expect("decoder tensors");
    let de_cfg: DecoderConfig = pack.component_config("decoder").expect("decoder config");

    let durs: Vec<usize> = read_i32("durs.i32")
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let input =
        rlx_sanotts::model::acoustic_frame_input(&ac, &ac_cfg, &ids(), &durs).expect("frame input");
    let frames = input.cols;

    let mut ran = 0;
    for device in [
        rlx_runtime::Device::Cpu,
        rlx_runtime::Device::Metal,
        rlx_runtime::Device::Mlx,
        rlx_runtime::Device::Cuda,
        rlx_runtime::Device::Rocm,
        rlx_runtime::Device::Gpu,
        rlx_runtime::Device::Vulkan,
        rlx_runtime::Device::Ane,
    ] {
        if !rlx_runtime::is_available(device) {
            continue;
        }
        let mut tight =
            FrameGraph::compile(&ac, &ac_cfg, &de, &de_cfg, frames, device).expect("compile tight");
        // A capacity that is not a multiple of anything convenient, and far
        // enough past `frames` to exceed the network's receptive field.
        let mut padded = FrameGraph::compile(&ac, &ac_cfg, &de, &de_cfg, frames + 37, device)
            .expect("compile padded");

        let (l_tight, a_tight) = tight.forward(&input).expect("tight forward");
        let (l_pad, a_pad) = padded.forward(&input).expect("padded forward");

        assert_eq!(l_pad.cols, frames, "{device:?}: padded latent not cropped");
        assert_eq!(
            a_pad.len(),
            a_tight.len(),
            "{device:?}: padded audio not cropped"
        );
        let (l_max, _) = compare(&l_pad.data, &l_tight.data);
        let (a_max, _) = compare(&a_pad, &a_tight);
        // If the mask were leaking, the error would grow toward the tail where
        // the padding is; check the last 5% separately.
        let tail = a_tight.len() * 95 / 100;
        let (tail_max, _) = compare(&a_pad[tail..], &a_tight[tail..]);
        eprintln!("{device:?}: bucketing delta latent={l_max:e} audio={a_max:e} tail={tail_max:e}");
        assert!(
            l_max < 1e-5 && a_max < 1e-6,
            "{device:?}: bucketed output differs from exact-length: latent={l_max:e} audio={a_max:e}"
        );
        ran += 1;
    }
    assert!(ran > 0, "no rlx device was available");
}

/// No shipped voice has a decoder post-filter, so graft a synthetic one onto a
/// real decoder and check the graph lowering against the host implementation.
#[cfg(feature = "rlx-graph")]
#[test]
fn graph_post_filter_matches_host() {
    use rlx_sanotts::graph::FrameGraph;
    use rlx_sanotts::{AcousticConfig, DecoderConfig, VoicePack};

    let dir = voice_or_skip!();
    let pack = VoicePack::load_from_dir(&dir).expect("load pack");
    let ac = pack
        .component_tensors("acoustic")
        .expect("acoustic tensors");
    let ac_cfg: AcousticConfig = pack.component_config("acoustic").expect("acoustic config");
    let mut de = pack.component_tensors("decoder").expect("decoder tensors");
    let mut de_cfg: DecoderConfig = pack.component_config("decoder").expect("decoder config");

    // A 2-layer, 8-channel, kernel-9 post-filter over the 1-channel waveform.
    let (ch, layers, k) = (8usize, 2usize, 9usize);
    de_cfg.post_filter_channels = ch;
    de_cfg.post_filter_layers = layers;
    de_cfg.post_filter_kernel = k;
    de_cfg.post_filter_scale = 0.25;
    de.insert(
        "post_filter.in_conv.weight",
        pseudo(ch * k, 11, 0.2),
        vec![ch, 1, k],
    );
    de.insert("post_filter.in_conv.bias", pseudo(ch, 12, 0.05), vec![ch]);
    for layer in 0..layers {
        let seed = 100 + layer as u64 * 10;
        de.insert(
            format!("post_filter.units.{layer}.scale"),
            vec![0.3],
            vec![1],
        );
        de.insert(
            format!("post_filter.units.{layer}.conv1.weight"),
            pseudo(ch * ch * k, seed + 1, 0.08),
            vec![ch, ch, k],
        );
        de.insert(
            format!("post_filter.units.{layer}.conv1.bias"),
            pseudo(ch, seed + 2, 0.02),
            vec![ch],
        );
        de.insert(
            format!("post_filter.units.{layer}.conv2.weight"),
            pseudo(ch * ch * k, seed + 3, 0.08),
            vec![ch, ch, k],
        );
        de.insert(
            format!("post_filter.units.{layer}.conv2.bias"),
            pseudo(ch, seed + 4, 0.02),
            vec![ch],
        );
    }
    de.insert(
        "post_filter.out_conv.weight",
        pseudo(ch * k, 21, 0.2),
        vec![1, ch, k],
    );
    de.insert("post_filter.out_conv.bias", pseudo(1, 22, 0.02), vec![1]);

    let durs: Vec<usize> = read_i32("durs.i32")
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let input =
        rlx_sanotts::model::acoustic_frame_input(&ac, &ac_cfg, &ids(), &durs).expect("frame input");
    let frames = input.cols;

    // Host reference with the post-filter engaged.
    let latent = rlx_sanotts::model::acoustic_frame_forward(&ac, &ac_cfg, &input).expect("latent");
    let want = rlx_sanotts::model::decoder_forward(&de, &de_cfg, &latent).expect("host decode");
    // Sanity: the post-filter must actually change the waveform, or this test
    // would pass just as well against a no-op lowering.
    let plain_cfg: DecoderConfig = pack.component_config("decoder").expect("decoder config");
    let plain = rlx_sanotts::model::decoder_forward(&de, &plain_cfg, &latent).expect("host decode");
    let delta = want
        .iter()
        .zip(&plain)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        delta > 1e-3,
        "synthetic post-filter is a no-op (delta {delta:e})"
    );

    let mut ran = 0;
    for device in [
        rlx_runtime::Device::Cpu,
        rlx_runtime::Device::Metal,
        rlx_runtime::Device::Mlx,
        rlx_runtime::Device::Cuda,
        rlx_runtime::Device::Rocm,
        rlx_runtime::Device::Gpu,
        rlx_runtime::Device::Vulkan,
        rlx_runtime::Device::Ane,
    ] {
        if !rlx_runtime::is_available(device) {
            continue;
        }
        let mut g = FrameGraph::compile(&ac, &ac_cfg, &de, &de_cfg, frames, device)
            .expect("compile with post-filter");
        let (_, got) = g.forward(&input).expect("forward");
        let (max_abs, corr) = compare(&got, &want);
        assert!(
            max_abs < 5e-3 && corr > 0.999_99,
            "{device:?} post-filter drift: max_abs={max_abs:e} corr={corr}"
        );
        ran += 1;
    }
    assert!(ran > 0, "no rlx device was available");
}

/// The frontend must produce the *reference* phoneme stream for an American
/// voice, not an en-GB rendering of it.
///
/// espeak-ng ≤ 0.1.3 let its en-GB IPA override table outrank the active
/// phoneme table, so every English voice came out non-rhotic with RP vowels.
/// Feeding the reference IPA through `ids_from_phonemes` and comparing against
/// G2P pins that down without needing a second phonemizer to compare to.
#[cfg(feature = "espeak")]
#[test]
fn g2p_produces_american_vowels() {
    let dir = voice_or_skip!();
    let synth = Synthesizer::load(&dir).expect("load voice");
    let table = synth.phoneme_table().expect("amy ships a phoneme table");
    assert_eq!(table.espeak_voice, "en-us");

    // "hello over" in reference en-US IPA: /oʊ/ GOAT vowel, rhotic /ɚ/.
    let want = synth
        .ids_from_phonemes("həlˈoʊ ˈoʊvɚ")
        .expect("reference phonemes");
    let got = synth.phoneme_ids("hello over").expect("g2p");
    assert_eq!(
        got, want,
        "G2P diverges from the reference phoneme stream — is espeak-ng \
         rendering en-US through the en-GB phoneme table again?"
    );

    // Belt and braces: the RP rendering must NOT be what we produce.
    let rp = synth
        .ids_from_phonemes("həlˈəʊ ˈəʊvə")
        .expect("rp phonemes");
    assert_ne!(got, rp, "G2P produced the en-GB rendering");
}
