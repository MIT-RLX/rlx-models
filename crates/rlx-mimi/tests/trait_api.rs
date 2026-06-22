//! Exercises the unified `rlx_core::AudioCodec` view of `MimiCodec` on CPU.

use rlx_mimi::{AudioCodec, ChunkStreamer, MimiCodec, RvqCodes, default_mimi_dir};
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let dir = default_mimi_dir();
    if dir.join("model.safetensors").is_file() {
        Some(dir)
    } else {
        std::env::var("RLX_MIMI_DIR")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.join("model.safetensors").is_file())
    }
}

#[test]
fn codec_info_reports_expected_metadata() {
    let Some(dir) = model_dir() else {
        eprintln!("skip codec_info_reports_expected_metadata: no weights");
        return;
    };
    let codec = MimiCodec::open(&dir).expect("open");
    let info = codec.info();
    assert_eq!(info.sample_rate, 24_000);
    assert!((info.frame_rate - 12.5).abs() < 1e-3);
    assert_eq!(info.hop_length, 1920);
    assert_eq!(info.channels, 1);
    assert_eq!(info.max_quantizers, 32);
    // 8 codebooks bitrate is positive and below the full-rate bitrate.
    assert!(info.bitrate_bps(Some(8)) < info.bitrate_bps(None));
}

#[test]
fn trait_roundtrip_and_bitrate_control() {
    let Some(dir) = model_dir() else {
        eprintln!("skip trait_roundtrip_and_bitrate_control: no weights");
        return;
    };
    let codec = MimiCodec::open(&dir).expect("open");
    let pcm: Vec<f32> = (0..24_000 / 4)
        .map(|i| (i as f32 * 0.02).sin() * 0.3)
        .collect();

    // Encode/decode purely through the trait object.
    let dyn_codec: &dyn AudioCodec = &codec;
    let codes: RvqCodes = dyn_codec.encode_pcm(&pcm, Some(8)).expect("encode");
    assert_eq!(codes.num_quantizers, 8);
    assert!(codes.num_frames() >= 4);
    let recon = dyn_codec.decode_codes(&codes).expect("decode");
    assert!(recon.len() > pcm.len() / 4);

    // Bitrate-driven encode picks a codebook count and stays within budget.
    let target = codec.info().bitrate_bps(Some(8));
    let codes_b = codec
        .encode_pcm_bitrate(&pcm, target)
        .expect("bitrate encode");
    assert_eq!(codes_b.num_quantizers, 8);

    // Resampled front-end: 16 kHz input is resampled to 24 kHz internally.
    let pcm16: Vec<f32> = (0..16_000 / 4)
        .map(|i| (i as f32 * 0.03).sin() * 0.3)
        .collect();
    let codes16 = codec
        .encode_pcm_resampled(&pcm16, 16_000, Some(8))
        .expect("resampled encode");
    assert!(codes16.num_frames() >= 1);
}

#[test]
fn chunk_streamer_tracks_progress() {
    let Some(dir) = model_dir() else {
        eprintln!("skip chunk_streamer_tracks_progress: no weights");
        return;
    };
    let codec = MimiCodec::open(&dir).expect("open");
    let mut stream = ChunkStreamer::with_bitrate(&codec, codec.info().bitrate_bps(Some(8)));
    let chunk: Vec<f32> = (0..1920 * 3)
        .map(|i| (i as f32 * 0.05).sin() * 0.2)
        .collect();
    let codes = stream.encode_chunk(&chunk).expect("encode chunk");
    let pcm = stream.decode_chunk(&codes).expect("decode chunk");
    assert_eq!(stream.frames_emitted(), codes.num_frames());
    assert_eq!(stream.samples_emitted(), pcm.len());
}
