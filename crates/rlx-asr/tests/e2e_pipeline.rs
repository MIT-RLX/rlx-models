// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Env-gated end-to-end: frontend → stub encoder → native AED → text.

use rlx_asr::pipeline::AsrSession;
use std::path::PathBuf;

fn asr_dir() -> Option<PathBuf> {
    let p = rlx_asr::asr_dir();
    p.is_dir().then_some(p)
}

#[test]
fn e2e_silence_runs() {
    let Some(dir) = asr_dir() else {
        eprintln!("skip: set RLX_ASR_DIR");
        return;
    };
    let mut asr = AsrSession::load(&dir).expect("load");
    // 0.5 s silence @ 16 kHz — exercises the full path without speech content.
    let pcm = vec![0.0f32; 8_000];
    let tr = asr.transcribe(&pcm, 16_000).expect("transcribe");
    eprintln!("e2e text={:?} tokens={}", tr.text, tr.token_ids.len());
    assert!(!tr.token_ids.is_empty());
}
