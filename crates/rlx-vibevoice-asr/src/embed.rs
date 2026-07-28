// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Host-side token embedding + speech-placeholder fusion (mirrors
// rlx-qwen3-asr::embed). Speech features overwrite the `<|speech_pad|>` rows;
// all other positions get their token-embedding row.

use crate::config::TOK_SPEECH_PAD;
use anyhow::{Result, ensure};

/// Build `inputs_embeds` `[seq, hidden]`: text rows from `token_embed`, and the
/// `<|speech_pad|>` rows overwritten (in order) with `audio_embeds`
/// `[n_frames, hidden]`.
pub fn fuse_inputs_embeds(
    hidden: usize,
    token_embed: &[f32],
    vocab: usize,
    token_ids: &[i64],
    audio_embeds: &[f32],
) -> Result<Vec<f32>> {
    ensure!(
        token_embed.len() == vocab * hidden,
        "token_embed len {} != vocab*hidden {}",
        token_embed.len(),
        vocab * hidden
    );
    let n_slots = token_ids.iter().filter(|&&t| t == TOK_SPEECH_PAD).count();
    let n_vecs = audio_embeds.len() / hidden;
    ensure!(
        n_slots == n_vecs,
        "speech placeholders ({n_slots}) != speech vectors ({n_vecs})"
    );

    let seq = token_ids.len();
    let mut out = vec![0f32; seq * hidden];
    let mut a = 0usize;
    for (pos, &tok) in token_ids.iter().enumerate() {
        let dst = &mut out[pos * hidden..(pos + 1) * hidden];
        if tok == TOK_SPEECH_PAD {
            dst.copy_from_slice(&audio_embeds[a * hidden..(a + 1) * hidden]);
            a += 1;
        } else {
            ensure!(
                (tok as usize) < vocab && tok >= 0,
                "token id {tok} out of range (vocab {vocab})"
            );
            let t = tok as usize;
            dst.copy_from_slice(&token_embed[t * hidden..(t + 1) * hidden]);
        }
    }
    Ok(out)
}

/// Greedy argmax over a `[vocab]` logits row.
pub fn argmax(logits: &[f32]) -> i64 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splices_speech_rows() {
        let hidden = 2;
        let vocab = 4;
        // token_embed[t] = [t, t]
        let mut te = vec![0f32; vocab * hidden];
        for t in 0..vocab {
            te[t * hidden] = t as f32;
            te[t * hidden + 1] = t as f32;
        }
        let ids = vec![1i64, TOK_SPEECH_PAD, TOK_SPEECH_PAD, 3];
        let audio = vec![9.0, 9.0, 8.0, 8.0]; // 2 frames
        let out = fuse_inputs_embeds(hidden, &te, vocab, &ids, &audio).unwrap();
        assert_eq!(&out[0..2], &[1.0, 1.0]);
        assert_eq!(&out[2..4], &[9.0, 9.0]);
        assert_eq!(&out[4..6], &[8.0, 8.0]);
        assert_eq!(&out[6..8], &[3.0, 3.0]);
    }

    #[test]
    fn argmax_picks_max() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
    }
}
