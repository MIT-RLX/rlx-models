//! Parity with `scripts/liquid_codec_reference.py` — LiquidAI's released
//! `modeling_liquid_codec.py` run in float32 over the bf16 checkpoint.
//!
//! `tests/fixtures/reference.safetensors` records, for
//! `tests/fixtures/clip_24k.wav` (4 s of `assets/harness/jfk_16k.wav` at
//! 24 kHz): the front-end-to-quantizer latents, the codes and the audio they
//! decode to. The tests load `LIQUID_CODEC_DIR` (default
//! `weights/audio/liquid-codec`; an LFM3-Audio directory works too) on
//! `LIQUID_CODEC_DEVICE` (default `cpu`), and skip when it is absent. CUDA
//! runs with TF32 off (`RLX_CUDA_NO_TF32`) unless that is set explicitly.

use std::path::PathBuf;

use rlx_core::weight_map::WeightMap;
use rlx_liquid_codec::{AudioCodec, LiquidCodec, RvqCodes, read_24k};

fn here(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn codec() -> Option<LiquidCodec> {
    let dir = rlx_ir::env::read("LIQUID_CODEC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| here("../../weights/audio/liquid-codec"));
    if rlx_liquid_codec::weights::resolve(&dir).is_err() {
        eprintln!(
            "skip: no checkpoint at {} (LIQUID_CODEC_DIR)",
            dir.display()
        );
        return None;
    }
    let device = rlx_ir::env::read("LIQUID_CODEC_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = rlx_cli::parse_device(&device).unwrap();
    // The tolerances are for f32 matmuls: CUDA's default TF32 moves the
    // decoded audio by ~2e-3 (codes stay identical). Set the flag to 0 to
    // measure TF32.
    if rlx_ir::env::var("RLX_CUDA_NO_TF32").is_none() {
        rlx_ir::env::set("RLX_CUDA_NO_TF32", "1");
    }
    Some(LiquidCodec::open(&dir, device).unwrap())
}

struct Reference {
    latents: Vec<f32>,
    codes: Vec<Vec<u32>>,
    decoded: Vec<f32>,
}

fn reference() -> Reference {
    let path = here("tests/fixtures/reference.safetensors");
    let mut m = WeightMap::from_file(path.to_str().unwrap()).unwrap();
    let (codes, shape) = m.take("codes").unwrap();
    let codes = codes
        .chunks(shape[1])
        .map(|f| f.iter().map(|&c| c as u32).collect())
        .collect();
    Reference {
        latents: m.take("latents").unwrap().0,
        codes,
        decoded: m.take("decoded").unwrap().0,
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

#[test]
fn encoder_matches_reference() {
    let Some(codec) = codec() else { return };
    let r = reference();
    let pcm = read_24k(&here("tests/fixtures/clip_24k.wav")).unwrap();
    let lat = codec.encode_latents(&pcm).unwrap();
    let err = max_abs(&lat, &r.latents);
    let cos = cosine(&lat, &r.latents);
    eprintln!("latents: max |Δ| {err:.3e}, cosine {cos:.8}");

    let codes = codec.encode(&pcm).unwrap();
    assert_eq!(codes.len(), r.codes.len());
    let total = codes.len() * codes[0].len();
    let same = codes
        .iter()
        .flatten()
        .zip(r.codes.iter().flatten())
        .filter(|(a, b)| a == b)
        .count();
    eprintln!("codes: {same}/{total} identical");
    assert!(cos > 0.99999, "latent cosine {cos}");
    assert!(err < 5e-3, "latent max |Δ| {err}");
    assert!(same * 100 >= total * 99, "{same}/{total} codes identical");
}

#[test]
fn decoder_matches_reference() {
    let Some(codec) = codec() else { return };
    let r = reference();
    let audio = codec.decode(&r.codes).unwrap();
    let err = max_abs(&audio, &r.decoded);
    let peak = r.decoded.iter().fold(0f32, |m, x| m.max(x.abs()));
    let cos = cosine(&audio, &r.decoded);
    eprintln!("decode: max |Δ| {err:.3e} (peak {peak:.3}), cosine {cos:.8}");
    assert!(cos > 0.99999, "decode cosine {cos}");
    assert!(err < 1e-3 * peak.max(1.0), "decode max |Δ| {err}");
}

#[test]
fn streaming_matches_whole() {
    let Some(codec) = codec() else { return };
    let r = reference();
    let whole = codec.decode(&r.codes).unwrap();
    let mut stream = codec.decode_stream();
    let mut out = Vec::new();
    for piece in r.codes.chunks(3) {
        out.extend(stream.push(piece).unwrap());
    }
    out.extend(stream.finish());
    let err = max_abs(&out, &whole);
    eprintln!("stream (3 frames per push): max |Δ| {err:.3e}");
    assert!(err < 1e-4, "stream vs whole max |Δ| {err}");
}

#[test]
fn audio_codec_trait() {
    let Some(codec) = codec() else { return };
    let info = codec.info();
    assert_eq!(
        (
            info.sample_rate,
            info.hop_length,
            info.max_quantizers,
            info.codebook_size
        ),
        (24_000, 1920, 12, 2016)
    );
    let pcm = vec![0.0f32; 3000];
    let codes = codec.encode_pcm(&pcm, None).unwrap();
    assert_eq!(codes.num_frames(), 2);
    assert!(codec.encode_pcm(&pcm, Some(6)).is_err());
    assert_eq!(codec.decode_codes(&codes).unwrap().len(), 2 * 1920);
    assert!(
        codec
            .decode_codes(&RvqCodes::new(vec![vec![0; 11]], 11))
            .is_err()
    );
}

/// The same decoder from LFM3-Audio's F16 mmproj (`rlx-lfm3-audio`, GGUF
/// names) against the same reference. `LFM3_AUDIO_GGUF` (default
/// `weights/lm/lfm3-audio-2b-gguf`); skips when absent.
#[test]
fn lfm3_audio_gguf_decoder_agrees() {
    let dir = rlx_ir::env::read("LFM3_AUDIO_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| here("../../weights/lm/lfm3-audio-2b-gguf"));
    let Ok((lm, mmproj)) = rlx_lfm3_audio::resolve(&dir, Some("Q4_K_M"), Some("F16")) else {
        eprintln!(
            "skip: no LFM3-Audio GGUF at {} (LFM3_AUDIO_GGUF)",
            dir.display()
        );
        return;
    };
    let device = rlx_ir::env::read("LIQUID_CODEC_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = rlx_cli::parse_device(&device).unwrap();
    let mut m = rlx_lfm3_audio::Lfm3Audio::open(&lm, &mmproj, device).unwrap();
    let r = reference();
    let audio = m.decode_frames(&r.codes).unwrap();
    let err = max_abs(&audio, &r.decoded);
    eprintln!("GGUF decoder: max |Δ| {err:.3e}");
    assert!(err < 1e-3, "GGUF decoder max |Δ| {err}");
}

fn same_codes(a: &[Vec<u32>], b: &[Vec<u32>]) -> (usize, usize) {
    assert_eq!(a.len(), b.len(), "frame count");
    let same = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .filter(|(x, y)| x == y)
        .count();
    (same, a.len() * a.first().map_or(0, Vec::len))
}

/// Long-clip encoding (KV-cached 64-frame chunks, here forced by a small
/// threshold) against the reference.
#[test]
fn chunked_encode_matches_reference() {
    let Some(mut codec) = codec() else { return };
    let r = reference();
    let pcm = read_24k(&here("tests/fixtures/clip_24k.wav")).unwrap();
    codec.set_max_whole_frames(16);
    let lat = codec.encode_latents(&pcm).unwrap();
    let err = max_abs(&lat, &r.latents);
    let (same, total) = same_codes(&codec.encode(&pcm).unwrap(), &r.codes);
    eprintln!("chunked encode: latents max |Δ| {err:.3e}, codes {same}/{total}");
    assert!(err < 5e-3, "chunked latent max |Δ| {err}");
    assert!(same * 100 >= total * 99, "{same}/{total} codes identical");
}

/// Streaming encode in uneven pushes against the reference: every frame is
/// emitted 480 samples after it ends, the rest on `finish`.
#[test]
fn streaming_encode_matches_reference() {
    let Some(codec) = codec() else { return };
    let r = reference();
    let pcm = read_24k(&here("tests/fixtures/clip_24k.wav")).unwrap();
    let mut s = codec.encode_stream();
    let mut codes = Vec::new();
    let (mut at, mut step) = (0, 1);
    while at < pcm.len() {
        let end = (at + step).min(pcm.len());
        let got = s.push(&pcm[at..end]).unwrap();
        // Frame f is out once 1920 f + 2400 samples are in.
        let ready = (end + 1920).saturating_sub(2400) / 1920;
        assert_eq!(
            codes.len() + got.len(),
            ready.min(r.codes.len()),
            "after {end} samples"
        );
        codes.extend(got);
        at = end;
        step = step * 7 % 3001 + 1;
    }
    codes.extend(s.finish().unwrap());
    let (same, total) = same_codes(&codes, &r.codes);
    eprintln!("streaming encode: codes {same}/{total}");
    assert!(same * 100 >= total * 99, "{same}/{total} codes identical");
}
