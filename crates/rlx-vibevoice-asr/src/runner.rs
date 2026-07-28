// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// End-to-end VibeVoice-ASR-BitNet runner: audio → two ConvNeXt VAE encoders →
// element-wise-summed speech features → fused into the Qwen2.5 prompt → BitNet
// Qwen2 decode → transcription. Mirrors VibeASR.cpp `demo/asr_infer.cpp`.

use anyhow::{Result, ensure};
use rlx_runtime::Device;
use std::path::Path;

use crate::audio::AudioData;
use crate::config::{COMPRESS_RATIO, LmConfig, TOK_ENDOFTEXT, TOK_IM_END, TOK_IM_START};
use crate::embed::fuse_inputs_embeds;
use crate::lm::{VibeLm, token_embed_matrix};
use crate::prompt::build_prompt_default;
use crate::tokenizer::VibeTokenizer;
use crate::vae::{VaeEncoderGraph, pad_to_multiple};
use crate::weights::{VaeEncoderWeights, load_vae};

/// Maximum transcription tokens to generate.
const MAX_NEW_TOKENS: usize = 2048;

pub struct VibeAsr {
    acoustic: VaeEncoderWeights,
    semantic: VaeEncoderWeights,
    lm: VibeLm,
    token_embed: (Vec<f32>, usize, usize), // (matrix, vocab, hidden)
    tok: VibeTokenizer,
    device: Device,
}

impl VibeAsr {
    /// Load the VAE + LM GGUFs and the tokenizer.
    pub fn load(
        vae_gguf: &Path,
        lm_gguf: &Path,
        tokenizer_json: &Path,
        device: Device,
    ) -> Result<Self> {
        let (acoustic, semantic) = load_vae(vae_gguf)?;
        let lm = VibeLm::load(lm_gguf, &LmConfig::default(), device)?;
        let token_embed = token_embed_matrix(lm.gguf())?;
        let tok = VibeTokenizer::from_file(tokenizer_json)?;
        Ok(Self {
            acoustic,
            semantic,
            lm,
            token_embed,
            tok,
            device,
        })
    }

    /// Transcribe mono audio at `src_rate` Hz. `json_format` selects the
    /// segment-JSON prompt (7B-style) instead of plain text.
    pub fn transcribe(
        &mut self,
        mono: &[f32],
        src_rate: usize,
        json_format: bool,
    ) -> Result<String> {
        let audio = AudioData::from_mono(mono, src_rate, true);
        ensure!(!audio.samples.is_empty(), "empty audio after resampling");
        let padded = pad_to_multiple(&audio.samples, COMPRESS_RATIO);

        // Encode acoustic + semantic (both → [n_frames, hidden]).
        let mut aenc = VaeEncoderGraph::compile_for(self.device, &self.acoustic, padded.len())?;
        let af = aenc.run(&padded)?;
        let mut senc = VaeEncoderGraph::compile_for(self.device, &self.semantic, padded.len())?;
        let sf = senc.run(&padded)?;

        let hidden = self.token_embed.2;
        let na = af.len() / hidden;
        let ns = sf.len() / hidden;
        let n_frames = na.min(ns);

        // Build the prompt; its <|speech_pad|> count is ceil(n_samples/3200),
        // which equals the encoder frame count.
        let prompt = build_prompt_default(
            |s| self.tok.encode_plain(s),
            audio.samples.len(),
            audio.duration_sec,
            None,
            json_format,
        );
        let target = prompt.speech_pad_count.min(n_frames);
        ensure!(
            target > 0,
            "no speech frames produced (n_frames={n_frames}, pads={})",
            prompt.speech_pad_count
        );

        // Element-wise sum of acoustic + semantic features (VibeASR.cpp
        // `build_speech_embeddings`), clamped to `target` frames.
        let mut audio_embeds = vec![0f32; target * hidden];
        for i in 0..target * hidden {
            audio_embeds[i] = af[i] + sf[i];
        }

        // If the prompt has more pads than we have frames, rebuild it to match
        // (keeps fuse's slot/vector counts equal). Normally they are identical.
        let prompt = if prompt.speech_pad_count != target {
            rebuild_prompt_with_pads(&prompt, target)
        } else {
            prompt
        };

        if std::env::var("VIBEASR_DEBUG").is_ok() {
            eprintln!(
                "[vibeasr] prompt {} tok, {} speech frames ({} acoustic / {} semantic)",
                prompt.tokens.len(),
                target,
                na,
                ns
            );
        }

        let (te, vocab, _) = &self.token_embed;
        let inputs_embeds = fuse_inputs_embeds(hidden, te, *vocab, &prompt.tokens, &audio_embeds)?;

        // Packed BitNet path by default (ternary Q2_0 DequantMatMul); set
        // VIBEASR_DENSE=1 for the dense-f32 rlx-flow prefill+KV-decode.
        let ids = if std::env::var("VIBEASR_DENSE").is_ok() {
            self.lm
                .generate(&inputs_embeds, prompt.tokens.len(), MAX_NEW_TOKENS)?
        } else {
            self.lm.generate_packed(
                &inputs_embeds,
                prompt.tokens.len(),
                te,
                *vocab,
                hidden,
                MAX_NEW_TOKENS,
            )?
        };

        let content = strip_header_and_eos(&ids);
        Ok(self.tok.decode(content, true))
    }
}

/// Drop the leading `<|im_start|>assistant\n` header (3 tokens the model emits
/// itself, since no generation prompt is appended) and any trailing EOS.
fn strip_header_and_eos(ids: &[i64]) -> &[i64] {
    let mut start = 0;
    if ids.first() == Some(&TOK_IM_START) {
        start = 3.min(ids.len());
    }
    let mut end = ids.len();
    if end > start && matches!(ids[end - 1], TOK_IM_END | TOK_ENDOFTEXT) {
        end -= 1;
    }
    &ids[start..end.max(start)]
}

/// Rebuild a prompt with a different speech-pad count (rare edge case where the
/// encoder frame count and `ceil(n_samples/3200)` disagree).
fn rebuild_prompt_with_pads(
    p: &crate::prompt::PromptTokens,
    target: usize,
) -> crate::prompt::PromptTokens {
    use crate::config::TOK_SPEECH_PAD;
    let head = &p.tokens[..p.speech_pad_start];
    let tail = &p.tokens[p.speech_pad_start + p.speech_pad_count..];
    let mut tokens = Vec::with_capacity(head.len() + target + tail.len());
    tokens.extend_from_slice(head);
    tokens.extend(std::iter::repeat_n(TOK_SPEECH_PAD, target));
    tokens.extend_from_slice(tail);
    crate::prompt::PromptTokens {
        tokens,
        speech_pad_start: p.speech_pad_start,
        speech_pad_count: target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TOK_SPEECH_START;

    #[test]
    fn strips_assistant_header() {
        // <|im_start|> assistant \n  Hello  <|im_end|>
        let ids = vec![TOK_IM_START, 77091, 198, 9906, TOK_IM_END];
        assert_eq!(strip_header_and_eos(&ids), &[9906]);
    }

    #[test]
    fn no_header_no_strip() {
        let ids = vec![9906, 1234];
        assert_eq!(strip_header_and_eos(&ids), &[9906, 1234]);
    }

    #[test]
    fn rebuild_pads_keeps_structure() {
        let p = crate::prompt::PromptTokens {
            tokens: vec![
                TOK_IM_START,
                TOK_SPEECH_START,
                151648,
                151648,
                151648,
                TOK_IM_END,
            ],
            speech_pad_start: 2,
            speech_pad_count: 3,
        };
        let r = rebuild_prompt_with_pads(&p, 1);
        assert_eq!(r.speech_pad_count, 1);
        assert_eq!(
            r.tokens,
            vec![TOK_IM_START, TOK_SPEECH_START, 151648, TOK_IM_END]
        );
    }
}
