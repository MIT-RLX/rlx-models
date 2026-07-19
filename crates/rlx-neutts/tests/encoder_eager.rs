//! NeuCodec encoder eager forward — gated on exported weights (+ optional W2V-BERT).
//!
//! ```sh
//! NEUTTS_ENCODER_PATH=weights/tts/neutts/neucodec_encoder.safetensors \
//!   NEUTTS_ENCODER_STUB_SEMANTIC=1 \
//!   cargo test -p rlx-neutts --test encoder_eager encoder_encode_stub --release
//!
//! # Full path (needs facebook/w2v-bert-2.0 snapshot):
//! NEUTTS_ENCODER_PATH=... RLX_W2V_BERT_DIR=... \
//!   cargo test -p rlx-neutts --test encoder_eager --features w2v-bert encoder_encode_wav --release
//! ```

use rlx_neutts::NeuCodecEncoder;

#[test]
fn encoder_loads_exported_weights() {
    let Some(path) = std::env::var("NEUTTS_ENCODER_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
    else {
        eprintln!("skip encoder_loads_exported_weights: set NEUTTS_ENCODER_PATH");
        return;
    };

    let enc = NeuCodecEncoder::load(&path).expect("load encoder");
    assert_eq!(enc.codec_enc_strides(), [2, 2, 4, 4, 5]);
    assert_eq!(enc.semantic_w2v_layer(), 16);
}

#[test]
fn encoder_encode_stub() {
    if std::env::var("NEUTTS_ENCODER_STUB_SEMANTIC")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip encoder_encode_stub: set NEUTTS_ENCODER_STUB_SEMANTIC=1");
        return;
    }
    let Some(path) = std::env::var("NEUTTS_ENCODER_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
    else {
        eprintln!("skip encoder_encode_stub: set NEUTTS_ENCODER_PATH");
        return;
    };

    let mut enc = NeuCodecEncoder::load(&path).expect("load encoder");
    // 1 s of silence at 16 kHz → 50 tokens at 50 Hz
    let pcm = vec![0.0f32; 16_000];
    let codes = enc.encode_pcm(&pcm).expect("encode_pcm");
    assert_eq!(codes.len(), 50, "expected 50 tokens for 1 s @ 50 Hz");
    for &c in &codes {
        assert!((0..=65535).contains(&c), "FSQ code out of range: {c}");
    }
    eprintln!(
        "encoder_encode_stub: backend={} codes[0..5]={:?}",
        enc.backend_name(),
        &codes[..5.min(codes.len())]
    );
}

#[cfg(feature = "w2v-bert")]
#[test]
fn encoder_encode_w2v_silence() {
    let Some(enc_path) = std::env::var("NEUTTS_ENCODER_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
    else {
        eprintln!("skip encoder_encode_w2v_silence: set NEUTTS_ENCODER_PATH");
        return;
    };
    if std::env::var("RLX_W2V_BERT_DIR")
        .ok()
        .filter(|p| !p.is_empty())
        .is_none()
    {
        eprintln!("skip encoder_encode_w2v_silence: set RLX_W2V_BERT_DIR");
        return;
    }

    let mut enc = NeuCodecEncoder::load(&enc_path).expect("load encoder");
    assert_eq!(enc.backend_name(), "codec/encoder+w2v-bert");
    // 0.5 s silence @ 16 kHz → 25 tokens at 50 Hz
    let pcm = vec![0.0f32; 8_000];
    let codes = enc.encode_pcm(&pcm).expect("encode_pcm with w2v-bert");
    assert_eq!(codes.len(), 25, "expected 25 tokens for 0.5 s @ 50 Hz");
    for &c in &codes {
        assert!((0..=65535).contains(&c), "FSQ code out of range: {c}");
    }
    eprintln!(
        "encoder_encode_w2v_silence: backend={} codes[0..5]={:?}",
        enc.backend_name(),
        &codes[..5.min(codes.len())]
    );
}

#[cfg(feature = "w2v-bert")]
#[test]
fn encoder_encode_wav() {
    let Some(enc_path) = std::env::var("NEUTTS_ENCODER_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
    else {
        eprintln!("skip encoder_encode_wav: set NEUTTS_ENCODER_PATH");
        return;
    };
    if std::env::var("RLX_W2V_BERT_DIR")
        .ok()
        .filter(|p| !p.is_empty())
        .is_none()
    {
        eprintln!("skip encoder_encode_wav: set RLX_W2V_BERT_DIR");
        return;
    }
    let wav = std::env::var("NEUTTS_ENCODER_TEST_WAV")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists());
    let Some(wav) = wav else {
        eprintln!("skip encoder_encode_wav: set NEUTTS_ENCODER_TEST_WAV to a mono WAV");
        return;
    };

    let mut enc = NeuCodecEncoder::load(&enc_path).expect("load encoder");
    let codes = enc.encode_wav(&wav).expect("encode_wav");
    assert!(!codes.is_empty());
    eprintln!(
        "encoder_encode_wav: {} tokens, backend={}",
        codes.len(),
        enc.backend_name()
    );
}
