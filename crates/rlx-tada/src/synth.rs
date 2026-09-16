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

//! The autoregressive synthesis loop (`TadaForCausalLM.generate`).
//!
//! The text is known, so this is not text generation — every step's token comes
//! from the transcript and the LM head is never built. What the loop actually
//! generates is, per token, an acoustic latent and a duration, sampled together
//! by the flow-matching head from the backbone's hidden state.
//!
//! Three offsets drive the whole thing and are easy to get subtly wrong:
//!
//! * **`shift_acoustic` (5).** The latent fed in at step `t` is the one
//!   belonging to token `t - 5`. The model sees text five tokens ahead of the
//!   audio it is producing, which is what lets it plan prosody.
//! * **`num_transition_steps` (5).** The last five prompt frames are dropped
//!   before the prompt is handed over, giving the model a hand-off region where
//!   it is already predicting while still inside the reference.
//! * **The prompt's text is masked out.** Only its acoustics condition the
//!   model; the reference transcript's *content* tokens are replaced with pad,
//!   keeping just the chat-structure tokens.

use crate::backbone::{Backbone, InputEmbedder};
use crate::codec::{CodecDecoder, expand_to_frames};
use crate::config::{FRAME_RATE, SAMPLE_RATE, TadaConfig};
use crate::gray;
use crate::head::{DiffusionHead, SolveOptions};
use crate::prof::{self, trace};
use crate::prompt::VoicePrompt;
use crate::rng::Normal;
use crate::text::normalize_text;
use crate::tokenizer::{PREFIX_TEMPLATE, TadaTokenizer};
use anyhow::{Context, Result, bail};
use ndarray::Array2;
use rlx_core::voice_clone::{OutputTrim, has_tts_runaway, trim_tts_output};
use rlx_runtime::Device;

/// Frames of the prompt handed back to the model before it takes over.
const NUM_TRANSITION_STEPS: usize = 5;

/// Knobs for one utterance.
#[derive(Debug, Clone)]
pub struct SynthOptions {
    pub solve: SolveOptions,
    /// Seed for the per-token initial noise. Fixed by default so a given
    /// (prompt, text, seed) always produces the same waveform.
    pub seed: u64,
    /// Trim the leading silence the first predicted gap asks for. Upstream
    /// always does; turning it off is useful when inspecting alignment.
    pub trim_leading_silence: bool,
    /// Cut anything after an over-long internal silence.
    ///
    /// TADA is autoregressive, so it can miss its stop condition and carry on
    /// with hallucinated speech or codec noise. The give-away is
    /// `[speech][long silence][more speech]`, which
    /// [`rlx_core::voice_clone::trim_tts_output`] detects and cuts. `None`
    /// leaves the decoder output exactly as produced.
    ///
    /// Leading silence is left alone here — TADA predicts its own leading gap
    /// and `trim_leading_silence` already handles it.
    pub runaway_trim: Option<OutputTrim>,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            solve: SolveOptions::default(),
            seed: 0x7ada_0000_0000_0001,
            trim_leading_silence: true,
            runaway_trim: Some(OutputTrim {
                trim_leading: false,
                ..OutputTrim::default()
            }),
        }
    }
}

/// A loaded TADA stack ready to synthesize.
pub struct Synthesizer {
    pub cfg: TadaConfig,
    pub device: Device,
    /// Checkpoint identity, used to key compiled graphs on disk.
    tag: String,
    embed: InputEmbedder,
    backbone: Backbone,
    head: DiffusionHead,
    /// Absent when only the autoregressive half is wanted — the loop can be
    /// exercised and compared without a vocoder, which is how the end-to-end
    /// parity test avoids a 650 MB dependency.
    decoder: Option<CodecDecoder>,
    tokenizer: TadaTokenizer,
}

impl Synthesizer {
    pub fn new(
        cfg: TadaConfig,
        device: Device,
        embed: InputEmbedder,
        backbone: Backbone,
        head: DiffusionHead,
        decoder: Option<CodecDecoder>,
        tokenizer: TadaTokenizer,
    ) -> Self {
        Self::with_tag(
            cfg, device, embed, backbone, head, decoder, tokenizer, "tada",
        )
    }

    /// As [`Self::new`], with an explicit compile-cache tag.
    #[allow(clippy::too_many_arguments)]
    pub fn with_tag(
        cfg: TadaConfig,
        device: Device,
        embed: InputEmbedder,
        backbone: Backbone,
        head: DiffusionHead,
        decoder: Option<CodecDecoder>,
        tokenizer: TadaTokenizer,
        tag: &str,
    ) -> Self {
        Self {
            cfg,
            device,
            tag: tag.to_string(),
            embed,
            backbone,
            head,
            decoder,
            tokenizer,
        }
    }

    /// Run the autoregressive loop and return the **generated** span only:
    /// `(latents [tokens, acoustic_dim], frame gaps [tokens + 1])`, with the
    /// encoder's mean/std already undone.
    ///
    /// This is `generate()`'s `encoded` / `time_before` pair — everything after
    /// it is the codec.
    pub fn synthesize_latents(
        &mut self,
        prompt: &VoicePrompt,
        text: &str,
        opts: &SynthOptions,
    ) -> Result<(Array2<f32>, Vec<u32>)> {
        prompt.validate()?;
        let plan = self.plan(prompt, text)?;
        let (latents, gaps) = self.run_loop(&plan, opts)?;

        let skip = plan.prompt_rows + NUM_TRANSITION_STEPS - 1;
        if latents.nrows() <= skip || gaps.len() <= skip {
            bail!(
                "synthesis produced {} latents but the prompt alone occupies {skip} — \
                 the target text is too short to leave any generated audio",
                latents.nrows()
            );
        }
        let out_gaps = gaps[skip..].to_vec();
        if std::env::var_os("RLX_TADA_TRACE").is_some() {
            // Duration blow-ups are TADA's characteristic failure, and they are
            // invisible in the waveform until it is already minutes long.
            eprintln!(
                "[tada] {} generated tokens, frame gaps {out_gaps:?} ({} frames total)",
                latents.nrows() - skip,
                out_gaps.iter().map(|g| (*g).max(1)).sum::<u32>()
            );
        }
        Ok((latents.slice(ndarray::s![skip.., ..]).to_owned(), out_gaps))
    }

    /// Synthesize `text` in the voice of `prompt`, returning 24 kHz mono PCM.
    pub fn synthesize(
        &mut self,
        prompt: &VoicePrompt,
        text: &str,
        opts: &SynthOptions,
    ) -> Result<Vec<f32>> {
        let (generated, gaps) = self.synthesize_latents(prompt, text, opts)?;
        let decoder = self
            .decoder
            .as_ref()
            .context("this Synthesizer was built without a codec decoder")?;

        let frames = expand_to_frames(&generated, &gaps, self.cfg.acoustic_dim)?;
        let t = std::time::Instant::now();
        let wav = decoder.decode(self.device, &frames)?;
        trace!(
            "  codec decode ({} frames) {:?}",
            frames.nrows(),
            t.elapsed()
        );

        let wav = if opts.trim_leading_silence {
            let trim = SAMPLE_RATE * gaps[0] as usize / FRAME_RATE;
            wav.get(trim..).unwrap_or(&[]).to_vec()
        } else {
            wav
        };

        let Some(cut) = opts.runaway_trim else {
            return Ok(wav);
        };
        if !has_tts_runaway(&wav, SAMPLE_RATE as u32, &cut) {
            return Ok(wav);
        }
        let kept = trim_tts_output(&wav, SAMPLE_RATE as u32, &cut);
        trace!(
            "  runaway: cut {:.2}s of post-silence output ({:.2}s -> {:.2}s)",
            (wav.len() - kept.len()) as f32 / SAMPLE_RATE as f32,
            wav.len() as f32 / SAMPLE_RATE as f32,
            kept.len() as f32 / SAMPLE_RATE as f32,
        );
        Ok(kept)
    }

    /// Everything derivable before the first forward pass.
    fn plan(&self, prompt: &VoicePrompt, text: &str) -> Result<Plan> {
        let target = self.tokenizer.encode(&normalize_text(text))?;
        if target.is_empty() {
            bail!("target text tokenized to nothing");
        }
        let prefix = self.tokenizer.encode(PREFIX_TEMPLATE)?;
        let special = self.tokenizer.special;
        let shift = self.cfg.shift_acoustic;

        // [bos] + prefix + prompt text + target text + eot × num_eos
        let mut input_ids = Vec::with_capacity(
            1 + prefix.len() + prompt.num_tokens() + target.len() + self.cfg.num_eos_tokens(),
        );
        input_ids.push(special.bos);
        input_ids.extend_from_slice(&prefix);
        input_ids.extend_from_slice(&prompt.token_ids);
        input_ids.extend_from_slice(&target);
        input_ids.extend(std::iter::repeat_n(special.eot, self.cfg.num_eos_tokens()));

        // Prompt-side conditioning: `prefix_len` zero rows for the chat
        // scaffold, then the reference tokens, then the transition region is
        // shaved off the end.
        let prefix_len = prefix.len();
        let total_rows = prefix_len + prompt.num_tokens();
        if total_rows <= NUM_TRANSITION_STEPS {
            bail!("reference audio is too short: {total_rows} prompt rows");
        }
        let rows = total_rows - NUM_TRANSITION_STEPS;

        let (before_p, after_p) = prompt.frame_gaps(self.cfg.num_time_classes as u32);
        let mut acoustic = Array2::<f32>::zeros((rows, self.cfg.acoustic_dim));
        let mut masks = vec![0u8; rows];
        let mut before = vec![0u32; rows];
        let mut after = vec![0u32; rows];
        for r in prefix_len..rows {
            let i = r - prefix_len;
            for c in 0..self.cfg.acoustic_dim {
                acoustic[[r, c]] = prompt.token_values[[i, c]];
            }
            masks[r] = 1;
            before[r] = before_p[i];
            after[r] = after_p[i];
        }
        // `_generate` receives the masks shifted one left with a trailing 1:
        // the mask at step t describes the latent that will be *consumed* next.
        let mut shifted = masks[1..].to_vec();
        shifted.push(1);

        // Hide the reference transcript: only chat structure survives.
        let mut masked_ids = input_ids.clone();
        let mut in_header = false;
        for slot in masked_ids.iter_mut().take(rows) {
            let tok = *slot;
            let keep = if tok == special.start_header {
                in_header = true;
                true
            } else if tok == special.end_header {
                in_header = false;
                true
            } else {
                in_header || tok == special.eot || tok == special.bos || tok == special.eos
            };
            if !keep {
                *slot = special.pad;
            }
        }

        let num_steps = masked_ids.len();
        // prefill_len, verbatim from `_generate`.
        let cap = num_steps.saturating_sub(shift + 1);
        let n_ac = cap.min(rows.saturating_sub(1));
        let n_t = cap.min(rows.saturating_sub(1));
        let n_frames_cap = rows.saturating_sub(2);
        let n_prefill_frames_max = if n_ac > 0 && n_t > 0 {
            n_ac.min(n_t).min(n_frames_cap)
        } else {
            0
        };
        if n_prefill_frames_max == 0 {
            bail!("reference audio is too short to prefill ({rows} prompt rows)");
        }
        let prefill_len = num_steps.min(shift + n_prefill_frames_max + 1);
        if prefill_len >= num_steps {
            bail!(
                "the reference prompt fills all {num_steps} steps — there is no room \
                 left to generate `{}`",
                text.chars().take(40).collect::<String>()
            );
        }

        Ok(Plan {
            input_ids: masked_ids,
            prompt_rows: rows,
            prefix_len,
            acoustic,
            masks: shifted,
            before,
            after,
            prefill_len,
            num_steps,
        })
    }

    fn run_loop(&mut self, plan: &Plan, opts: &SynthOptions) -> Result<(Array2<f32>, Vec<u32>)> {
        let shift = self.cfg.shift_acoustic;
        let hidden = self.cfg.hidden_size;
        let adim = self.cfg.acoustic_dim;
        let bits = self.cfg.num_time_bits();
        let latent = self.cfg.latent_dim();
        let guided = opts.solve.uses_guidance();
        let branches = self.backbone.batch;
        let special = self.tokenizer.special;

        // The solver is compiled lazily, inside the loop. Building it up front
        // would leave its ~1.4 GB arena resident through both backbone graph
        // builds, and those are what set the process peak.
        let mut solver: Option<rlx_runtime::CompiledGraph> = None;
        let mut rng = Normal::new(opts.seed);

        // ---- prefill ----
        let pl = plan.prefill_len;
        let zeros = vec![0f32; adim];
        let mut embeds = vec![0f32; branches * pl * hidden];
        let mut row = vec![0f32; hidden];
        for t in 0..pl {
            let (lat, mask, b, a) = plan.prefill_slot(t, shift, adim);
            let cond = self.embed.conditioning(lat.unwrap_or(&zeros), mask, b, a)?;
            self.embed.with_token(&cond, plan.input_ids[t], &mut row)?;
            for br in 0..branches {
                let base = br * pl * hidden + t * hidden;
                embeds[base..base + hidden].copy_from_slice(&row);
            }
        }

        let upper = crate::backbone::decode_bucket(plan.num_steps);
        let kv_dim = self.backbone.kv_dim();
        let layers = self.backbone.num_layers();
        let stride = kv_dim; // f32 values per (batch, position) KV row
        let mut kv_k: Vec<Vec<f32>> = Vec::with_capacity(layers);
        let mut kv_v: Vec<Vec<f32>> = Vec::with_capacity(layers);
        let t = std::time::Instant::now();
        {
            let out = self.backbone.prefill(&embeds, pl)?;
            trace!("  prefill (compile+run, seq {pl}) {:?}", t.elapsed());
            // Prefill rows are strided by its padded bucket, not by `pl`.
            let src_stride = out.row_stride();
            for l in 0..layers {
                let mut kb = vec![0f32; branches * upper * stride];
                let mut vb = vec![0f32; branches * upper * stride];
                for br in 0..branches {
                    let src = br * src_stride * stride;
                    let dst = br * upper * stride;
                    kb[dst..dst + pl * stride].copy_from_slice(&out.k(l)[src..src + pl * stride]);
                    vb[dst..dst + pl * stride].copy_from_slice(&out.v(l)[src..src + pl * stride]);
                }
                kv_k.push(kb);
                kv_v.push(vb);
            }
        }
        // Prefill and decode never overlap; keeping both arenas resident would
        // double the backbone's footprint for no benefit.
        self.backbone.drop_prefill();

        // ---- carry state, seeded exactly as `_generate` does after prefill ----
        let n_prefill_frames = pl - shift;
        let mut all_latents: Vec<f32> = Vec::with_capacity(plan.num_steps * adim);
        let mut all_gaps: Vec<u32> = Vec::with_capacity(plan.num_steps);
        for i in 0..n_prefill_frames {
            all_latents.extend_from_slice(plan.acoustic_row(i));
            all_gaps.push(plan.before[i + 1]);
        }
        let mut cur_latent = plan.acoustic_row(n_prefill_frames - 1).to_vec();
        let mut cur_mask = plan.masks[n_prefill_frames - 1];
        let mut cur_before = plan.before[n_prefill_frames];
        let mut cur_after = plan.after[n_prefill_frames];
        let mut last_gap = None;

        let mut noise = vec![0f32; latent];
        let mut step_embeds = vec![0f32; branches * hidden];

        let mut t_lm = std::time::Duration::ZERO;
        let mut t_solve = std::time::Duration::ZERO;
        let t_loop = std::time::Instant::now();
        for step in pl..plan.num_steps {
            let token = plan.input_ids[step.min(plan.input_ids.len() - 1)];
            let cond = self
                .embed
                .conditioning(&cur_latent, cur_mask, cur_before, cur_after)?;
            self.embed
                .with_token(&cond, token, &mut step_embeds[..hidden])?;
            if branches == 2 {
                // The negative branch keeps chat structure and drops content,
                // so guidance measures "what does the text add here".
                let neg_token = if special.is_structural(token) {
                    token
                } else {
                    special.pad
                };
                let (_, neg_slot) = step_embeds.split_at_mut(hidden);
                self.embed.with_token(&cond, neg_token, neg_slot)?;
            }

            let t0 = std::time::Instant::now();
            let out = self
                .backbone
                .step(&step_embeds, step, upper, &kv_k, &kv_v)?;
            t_lm += t0.elapsed();
            let hidden_states = out.hidden().to_vec();
            let pos_hidden = &hidden_states[..hidden];
            let neg_hidden = if branches == 2 {
                &hidden_states[hidden..2 * hidden]
            } else {
                pos_hidden
            };

            for l in 0..layers {
                for br in 0..branches {
                    let src = (br * (upper + 1) + upper) * stride;
                    let dst = (br * upper + step) * stride;
                    kv_k[l][dst..dst + stride].copy_from_slice(&out.k(l)[src..src + stride]);
                    kv_v[l][dst..dst + stride].copy_from_slice(&out.v(l)[src..src + stride]);
                }
            }

            rng.fill(&mut noise, opts.solve.noise_temperature);
            let mut inputs: Vec<(&str, &[f32])> = vec![("noise", &noise), ("cond", pos_hidden)];
            if guided {
                inputs.push(("neg_cond", neg_hidden));
            }
            if solver.is_none() {
                let t = std::time::Instant::now();
                solver = Some(self.head.compile_solver_tagged(
                    self.device,
                    &opts.solve,
                    &self.tag,
                )?);
                trace!("  solver compile {:?}", t.elapsed());
            }
            let t1 = std::time::Instant::now();
            let speech = solver
                .as_mut()
                .expect("solver")
                .run(&inputs)
                .into_iter()
                .next()
                .context("flow-matching solver produced no output")?;
            t_solve += t1.elapsed();
            if speech.len() != latent {
                bail!("solver returned {} values, expected {latent}", speech.len());
            }
            let pred_before = gray::decode(&speech[adim..adim + bits], bits);
            let pred_after = gray::decode(&speech[adim + bits..], bits);

            let i = step - shift;
            if i < plan.prompt_rows {
                cur_latent.copy_from_slice(plan.acoustic_row(i));
                cur_mask = plan.masks[i];
            } else {
                cur_latent.copy_from_slice(&speech[..adim]);
                cur_mask = 1;
            }
            all_latents.extend_from_slice(&cur_latent);

            if i + 1 < plan.prompt_rows {
                cur_before = plan.before[i + 1];
                cur_after = plan.after[i + 1];
            } else {
                cur_before = pred_before.min(self.cfg.num_time_classes as u32 - 1);
                cur_after = pred_after.min(self.cfg.num_time_classes as u32 - 1);
            }
            all_gaps.push(cur_before);
            last_gap = Some(cur_before);
        }

        trace!(
            "  decode loop {:?} over {} steps (lm {:?}, solve {:?}) rss {} MB",
            t_loop.elapsed(),
            plan.num_steps - pl,
            t_lm,
            t_solve,
            prof::rss_mb()
        );
        // Everything below is the codec, which wants its own large arena.
        drop(solver);
        self.backbone.release();
        // `_decode_wav` needs one more gap than latents: the trailing silence.
        if let Some(g) = last_gap {
            all_gaps.push(g);
        }

        let rows = all_latents.len() / adim;
        let mut latents = Array2::from_shape_vec((rows, adim), all_latents)
            .context("acoustic feature reshape")?;
        // Undo the encoder's normalization before the codec sees them.
        latents.mapv_inplace(|v| v * self.cfg.acoustic_std + self.cfg.acoustic_mean);
        Ok((latents, all_gaps))
    }
}

/// Everything the loop needs that does not depend on model outputs.
struct Plan {
    input_ids: Vec<u32>,
    /// Rows of prompt-side conditioning (chat prefix + reference tokens, minus
    /// the transition region).
    prompt_rows: usize,
    #[allow(dead_code)]
    prefix_len: usize,
    acoustic: Array2<f32>,
    masks: Vec<u8>,
    before: Vec<u32>,
    after: Vec<u32>,
    prefill_len: usize,
    num_steps: usize,
}

impl Plan {
    fn acoustic_row(&self, i: usize) -> &[f32] {
        let w = self.acoustic.ncols();
        &self.acoustic.as_slice().expect("contiguous")[i * w..(i + 1) * w]
    }

    /// Prompt conditioning for prefill position `t`.
    ///
    /// Position `t` carries the values that the step-by-step path would have
    /// set at step `t - 1`: the latent of token `t - shift - 1` and the gaps of
    /// token `t - shift`. Positions at or below `shift` carry nothing.
    fn prefill_slot(&self, t: usize, shift: usize, adim: usize) -> (Option<&[f32]>, u8, u32, u32) {
        if t <= shift {
            return (None, 0, 0, 0);
        }
        let ac_idx = t - shift - 1;
        let n_ac = (self.prefill_len.saturating_sub(shift + 1)).min(self.prompt_rows);
        let (lat, mask) = if ac_idx < n_ac {
            (Some(&self.acoustic_row(ac_idx)[..adim]), self.masks[ac_idx])
        } else {
            (None, 0)
        };
        let n_t = (self.prefill_len.saturating_sub(shift + 1)).min(self.before.len() - 1);
        let (b, a) = if ac_idx < n_t {
            (self.before[ac_idx + 1], self.after[ac_idx + 1])
        } else {
            (0, 0)
        };
        (lat, mask, b, a)
    }
}
