// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Streaming scores equal full-window scores, bit for bit: at every hop of a long stream, a
//! fresh network scoring just that window (a full forward) gives the same probabilities as
//! the streaming one, for several window lengths, hop sizes (including ones that must fall
//! back to a full forward) and a multi-output head.

use rlx_wakeword_core::{WakeCnn, WakeCnnConfig, WakeCnnWeights};

fn weights(n_out: usize, seed: u64) -> WakeCnnWeights {
    // Order-aware pooling (4 stretches), as trained now.
    let mut w = WakeCnnWeights::he_init(
        WakeCnnConfig {
            pool_chunks: 4,
            ..WakeCnnConfig::lite()
        },
        seed,
    );
    let h = w.cfg.hidden;
    w.fc2_w = (0..n_out * h)
        .map(|i| ((i * 7919 % 211) as f32 - 105.0) / 400.0)
        .collect();
    w.fc2_b = (0..n_out).map(|i| i as f32 * 0.1 - 0.2).collect();
    w
}

fn frames(n: usize, n_mels: usize) -> Vec<f32> {
    let mut s = 0x1234_5678u32;
    (0..n * n_mels)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
        })
        .collect()
}

#[test]
fn streaming_matches_the_full_window_exactly() {
    for (window, n_out) in [(120usize, 3usize), (200, 1), (37, 2)] {
        let w = weights(n_out, 7);
        let n_mels = w.cfg.n_mels;
        let all = frames(window + 400, n_mels);
        let mut stream = WakeCnn::new(w.clone()).with_window_frames(window);
        // Hops: mostly 4 frames (streamed), some 8, some odd (full recompute).
        let hops = [4usize, 4, 8, 4, 3, 4, 4, 12, 4, 5, 4];
        let (mut at, mut i, mut scored) = (0usize, 0usize, 0usize);
        while at + hops[i % hops.len()] <= all.len() / n_mels {
            let h = hops[i % hops.len()];
            let got = stream
                .push_mel_frames_all(&all[at * n_mels..(at + h) * n_mels])
                .map(|p| p.to_vec());
            at += h;
            i += 1;
            let Some(got) = got else { continue };
            let mut fresh = WakeCnn::new(w.clone()).with_window_frames(window);
            let want = fresh
                .push_mel_frames_all(&all[(at - window) * n_mels..at * n_mels])
                .unwrap()
                .to_vec();
            assert_eq!(got, want, "window {window}, hop {h} ending at frame {at}");
            scored += 1;
        }
        assert!(scored > 50, "{scored} windows scored");
    }
}

#[test]
fn appending_without_scoring_then_scoring_is_still_exact() {
    let w = weights(2, 3);
    let n_mels = w.cfg.n_mels;
    let all = frames(300, n_mels);
    let mut s = WakeCnn::new(w.clone()).with_window_frames(100);
    s.push_mel_frames_all(&all[..120 * n_mels]);
    s.append_mel_frames(&all[120 * n_mels..140 * n_mels]);
    let got = s
        .push_mel_frames_all(&all[140 * n_mels..144 * n_mels])
        .unwrap()
        .to_vec();
    let want = WakeCnn::new(w)
        .with_window_frames(100)
        .push_mel_frames_all(&all[44 * n_mels..144 * n_mels])
        .unwrap()
        .to_vec();
    assert_eq!(got, want);
}
