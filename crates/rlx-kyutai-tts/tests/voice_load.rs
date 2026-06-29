//! Voice embedding fetch + tensor layout (env-gated).

use rlx_kyutai_tts::checkpoint::KyutaiTtsCheckpoint;
use rlx_kyutai_tts::download::{
    DEFAULT_VOICE_NAME, default_voices_dir, ensure_voice_embedding, voice_embedding_path,
};
use rlx_kyutai_tts::load_voice_speaker_wavs;

#[test]
fn voice_embedding_resolves_and_loads() {
    let checkpoint = KyutaiTtsCheckpoint::V1_6bEnFr;
    let voices_dir = default_voices_dir();
    let path = match ensure_voice_embedding(&voices_dir, checkpoint, DEFAULT_VOICE_NAME) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    assert_eq!(
        path,
        voice_embedding_path(&voices_dir, checkpoint, DEFAULT_VOICE_NAME)
    );
    let seq = load_voice_speaker_wavs(&path).expect("load speaker_wavs");
    assert_eq!(seq.ncols(), 512);
    assert!(seq.nrows() >= 1);
    assert!(seq.iter().any(|&v| v.abs() > 1e-6));
}
