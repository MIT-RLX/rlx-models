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

//! CustomVoice greedy synthesis (talker + code predictor + speech tokenizer decode).

use crate::config::{GenerationConfig, Qwen3TtsConfig};
use crate::load::Qwen3TtsWeightStore;
use crate::megakernel::Qwen3TtsMegakernel;
use crate::progress::Progress;
use crate::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use crate::speech_tokenizer::open_speech_decoder_for_frames;
use crate::text_embed::TextEmbedder;
use anyhow::Result;
use rlx_runtime::Device;
use std::path::Path;
use std::time::Instant;

pub struct SynthesisResult {
    pub codec_frames: Vec<Vec<u32>>,
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
}

pub fn synthesize_custom_voice_greedy(
    model_dir: &Path,
    cfg: &Qwen3TtsConfig,
    store: &Qwen3TtsWeightStore,
    device: Device,
    text: &str,
    speaker: &str,
    language: &str,
    gen_cfg: &GenerationConfig,
    skip_speech_decode: bool,
) -> Result<SynthesisResult> {
    crate::compile_opts::ensure_metal_lowering_env(device);
    let timing = crate::synth_opts::synth_timing_enabled();
    let step_timing = crate::synth_opts::step_timing_enabled();
    let t_total = Instant::now();
    let warmup_prog = Progress::new("warmup", 4);
    warmup_prog.set(0, "tokenizer + prompt");
    let t0 = Instant::now();
    let tokenizer = load_text_tokenizer(model_dir)?;
    let text_embedder = TextEmbedder::open(store)?;
    let prompt = build_custom_voice_prompt(
        cfg,
        store,
        &text_embedder,
        &tokenizer,
        text,
        speaker,
        language,
    )?;
    if timing {
        eprintln!(
            "[qwen3-tts timing] prompt: {:.2}s",
            t0.elapsed().as_secs_f64()
        );
    }

    warmup_prog.set(1, &format!("talker {:?} compile", device));
    let t_warm = Instant::now();
    let mut mk = Qwen3TtsMegakernel::open(store, cfg.talker(), cfg.code_predictor(), device)?;
    let max_frames = gen_cfg.max_new_tokens.max(2);
    let frame_budget = crate::synth_opts::codec_frame_budget(text, max_frames, 0);
    mk.warmup(prompt.embeds.view(), max_frames, Some(&warmup_prog))?;
    let mut speech_dec = if skip_speech_decode {
        None
    } else {
        warmup_prog.set(
            3,
            &format!(
                "speech decoder ({}, {} frames)",
                crate::speech_tokenizer::speech_conv_backend_label(device),
                frame_budget
            ),
        );
        Some(open_speech_decoder_for_frames(
            model_dir,
            device,
            frame_budget,
        )?)
    };
    warmup_prog.finish("graphs ready");
    if timing {
        eprintln!(
            "[qwen3-tts timing] warmup (horizon precompile={}): {:.2}s",
            crate::synth_opts::auto_precompile_horizon(max_frames),
            t_warm.elapsed().as_secs_f64()
        );
    }

    let frame_prog = Progress::new("synthesis", gen_cfg.max_new_tokens.max(2));
    frame_prog.set(0, "talker prefill");

    let t_synth = Instant::now();
    let t_frames = Instant::now();
    let talker_cfg = cfg.talker();
    let max_steps = gen_cfg.max_new_tokens.max(2);
    let min_frames = gen_cfg.min_new_tokens.max(1);
    let (codec_frames, ar_timings) = mk.synthesize_codec_ar(
        prompt.embeds.view(),
        talker_cfg,
        max_steps,
        min_frames,
        gen_cfg.repetition_penalty,
        &prompt.tts_pad_embed,
        Some(&frame_prog),
    )?;
    frame_prog.finish(&format!("{} codec frames", codec_frames.len()));
    if codec_frames.len() >= max_steps {
        let suggested = crate::synth_opts::codec_frame_budget(text, max_steps, 0);
        eprintln!(
            "[qwen3-tts] warning: hit codec frame ceiling ({max_steps}) before talker EOS — \
             audio may be cut short (try longer prompt budget ≈{suggested}, or raise --max-frames)"
        );
    }
    if timing {
        let total_ar = t_frames.elapsed().as_secs_f64();
        eprintln!(
            "[qwen3-tts timing] codec AR ({} frames): {:.2}s (prefill {:.2}s, talker {:.2}s, CP {:.2}s)",
            codec_frames.len(),
            total_ar,
            ar_timings.prefill_secs,
            ar_timings.talker_secs,
            ar_timings.cp_secs
        );
    }
    let _ = step_timing;

    let (pcm, _speech_secs) = if skip_speech_decode {
        (Vec::new(), 0f64)
    } else {
        let dec_prog = Progress::new("speech decode", 1);
        dec_prog.set(
            0,
            &format!(
                "12Hz decoder (pt {}, conv {})",
                crate::speech_tokenizer::speech_pt_backend_label(device),
                crate::speech_tokenizer::speech_conv_backend_label(device),
            ),
        );
        let t_dec = Instant::now();
        let pcm = speech_dec
            .as_mut()
            .expect("speech decoder")
            .decode(&codec_frames, device)?;
        let speech_secs = t_dec.elapsed().as_secs_f64();
        dec_prog.finish(&format!("{} samples", pcm.len()));
        if timing {
            eprintln!("[qwen3-tts timing] speech decode: {:.2}s", speech_secs);
        }
        (pcm, speech_secs)
    };
    if timing {
        let synth_secs = t_synth.elapsed().as_secs_f64();
        let audio_secs = pcm.len() as f64 / crate::tokens::SAMPLE_RATE_HZ as f64;
        if audio_secs > 0.0 {
            eprintln!(
                "[qwen3-tts timing] audio duration: {:.2}s ({} samples)",
                audio_secs,
                pcm.len()
            );
            eprintln!(
                "[qwen3-tts timing] synthesis rtf: {:.2} ({:.2}s synth / {:.2}s audio; target rtf≤1.0)",
                synth_secs / audio_secs,
                synth_secs,
                audio_secs
            );
        }
        eprintln!(
            "[qwen3-tts timing] total: {:.2}s (device={device:?})",
            t_total.elapsed().as_secs_f64()
        );
    }
    Ok(SynthesisResult {
        codec_frames,
        pcm,
        sample_rate: crate::tokens::SAMPLE_RATE_HZ,
    })
}
