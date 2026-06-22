//! Streaming synthesis: the concatenation of overlap-trimmed chunks must equal a
//! full-utterance vocode, and every chunk must be produced faster than real time.

use std::path::PathBuf;

use rlx_inflect_nano::{InferOpts, InflectNano};

fn data_dir() -> Option<PathBuf> {
    let base = std::env::var("RLX_INFLECT_NANO_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/inflect-nano-rlx")
        });
    base.join("config.json").exists().then_some(base)
}

#[test]
fn streaming_matches_full_and_is_realtime() {
    let Some(dir) = data_dir() else {
        eprintln!("skip: bundle not found");
        return;
    };
    let model = InflectNano::load_from_dir(&dir).expect("load");
    let opts = InferOpts::default();
    let text = "The weather is nice today, and I feel very relaxed. \
        In nineteen ninety nine we found a thousand reasons to keep going.";

    // reference: full-utterance raw vocoder output
    let (p, t, l) = model.text_to_ids(text).unwrap();
    let mel = model
        .mel_from_ids(&p, &t, &l, model.cfg.default_speaker(), &opts)
        .unwrap();
    let full = model.wav_from_mel(&mel).unwrap();

    // streamed: 1-second chunks
    let mut streamed = Vec::new();
    let report = model
        .synthesize_stream(text, &opts, 1.0, |chunk| streamed.extend_from_slice(chunk))
        .expect("stream");

    eprintln!(
        "chunks={} audio={:.2}s compute={:.3}s rtf={:.1}x worst_chunk_rtf={:.1}x",
        report.chunks,
        report.audio_secs,
        report.compute_secs,
        report.rtf(),
        report.worst_chunk_rtf
    );

    assert_eq!(
        streamed.len(),
        full.len(),
        "streamed length {} != full {}",
        streamed.len(),
        full.len()
    );
    let d = streamed
        .iter()
        .zip(&full)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!("streamed-vs-full maxdiff = {d:.3e}");
    assert!(d < 1e-3, "streaming seam mismatch: {d:.3e}");

    // "one second of audio under one second of compute" — every chunk faster than real time.
    assert!(
        report.sustains_realtime(),
        "worst chunk RTF {:.2}x < 1.0",
        report.worst_chunk_rtf
    );
}
