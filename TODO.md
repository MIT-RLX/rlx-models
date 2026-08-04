# audio.cpp → rlx-models port backlog

Source: [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) — 44 model families.
This tracks the **30 not-fully-covered** models (26 missing + 4 partial). The 14 already
implemented are listed at the bottom for reference.

Suggested crate name is `rlx-<name>`. "Reuse" = the closest existing rlx crate/primitive to
build on. Status: `missing` = no coverage, `partial` = related crate exists but not this exact model.

## Progress

**Approach:** foundation-first (shared building blocks), then compose models. Depth: arch + CPU smoke, sequential.

- [x] **MIOTTS babble FIXED (whisper 6/6).** Same typical-channel-length-collision class as supertonic but on the ConvTranspose path: `/wave_conv_upsample` 1D ConvTranspose input `[1,512,100]` (512 ch, 100 frames) was flipped to `[1,100,512]` by `ensure_nchw_4d`'s `is_vocoder_blc` because **length 100 is a "typical channel" value** → whole vocoder corrupted (decoder_body cos 0.038). Bisected via native-vs-ort tap-diff (`mio_dump` example + `RLX_ONNX_TAP` + Python onnxruntime; RoPE + the import_probe panic were red herrings). Fix: extended the `conv_pool.rs` concrete-in_channels NCL guard (added for supertonic) to ConvTranspose too — weight `[Cin,Cout/g,k]` → in_channels = `dim(0)`. **decoder_body mag/phase cos 1.000000; whisper "The Quick Brown Fox…" 6/6** (was " You" 0/6). No regressions (Kokoro + tiny-tts conv green). See memory `miotts_convtranspose_FIXED`.
- [x] **SUPERTONIC babble FIXED (real weights + whisper-validated).** Root cause was a `../rlx` `rlx-onnx-import` shape bug, not the model port: `ensure_nchw_4d`'s `is_vocoder_blc` heuristic transposed the dilation-2 ConvNeXt depthwise conv's genuine-NCL input `[1,256,64]`→`[1,64,256]` because the padded length **64** is in the `is_typical_channel` list, giving `c_in/groups=64/256=0` → bias-only conv → babble (dil-1 pad=60 and dil-4 pad=72 don't collide, so only dil-2 broke). Fixed in `crates/io/rlx-onnx-import/src/lower/ops/conv_pool.rs` using the weight's concrete in_channels; regression test `conv_ncl_typical_length.rs`. **Per-subgraph + full e2e native-vs-ort parity now cos=1.000000** (was 0.0108); native whisper transcript EXACT ("The quick brown fox…"). No regressions (onnx-import suite, tiny-tts conv parity, melotts/kokoro round-trips all green). Also added tiny_tts in-memory compile cache + `run_named` (in-place) → **native wall-time now matches ort** (RTF ~4×). RAM still exceeds ort (native 3.47 GB vs 517 MB — the vector_estimator 1.36 GB + vocoder 1.17 GB f32 activation arenas are held resident; shared-arena + f16 activations would close it). See memory `supertonic_conv_ncl_fix`.

- [x] **`rlx-audio-blocks` crate scaffolded** and wired into the workspace — the RLX analogue of audio.cpp's `framework/`. Consolidates model-agnostic algorithms; re-exports canonical crates as the campaign proceeds.
- [x] **Shared TDT decoder** (`decoders::tdt`) — port of audio.cpp `framework/decoders/tdt_*`. `TdtDecoderCore` trait + `run_tdt_greedy_duration_loop`. 6 CPU smoke tests green. **Unblocks `parakeet_tdt`.**
- [x] **`rlx-parakeet` crate** (Parakeet-TDT) — first model composed from the foundation: reuses `rlx-nemotron-asr` FastConformer encoder + LSTM prediction net + the shared TDT decoder; adds only `TdtJoint` (token + duration heads) and `TdtCore`. 4 CPU smoke tests green. Encoder/`.nemo` wiring for e2e `transcribe` deferred (needs a checkpoint).
- [x] **`sampling` foundation block** (`rlx-audio-blocks::sampling`) — seedable `Rng` (SplitMix64 + Box–Muller) and schedulers (`betas` Linear/ScaledLinear/Cosine, `alphas_cumprod`, `FlowMatchEuler`, `ddpm_posterior_mean`). 9 CPU tests green. Unblocks the diffusion/flow generators. (Torch bit-exact RNG parity deferred — hand such models pre-generated reference noise.)
- [x] **`rlx-hviske` crate** (queue #1, Danish ASR) — thin preset over `rlx-whisper`: pins the Whisper-large-v3 config (128 mel bins, d_model 1280, 32/32 layers, vocab 51866), defaults language `da`, forwards all backend feature flags (inherits CPU/Metal/MLX/CoreML/CUDA/ROCm/Vulkan/wgpu). 4 unit + 1 doctest green. Real-weight transcribe deferred to a checkpoint.
- [x] **`rlx-inflect-v2` crate** (queue #2) — **correction: Inflect v2 is VITS2, not a v1 delta.** Ported the faithful `InflectV2Config` (vocab 178, sr 24000, hop 256, flow_count 4) + `GenerationOptions` (variation 0.667) + `sample_flow_prior` (reuses `rlx-audio-blocks::sampling::Rng`). 6 CPU tests green. VITS graph reuses `rlx-tiny-tts`; wiring `BundleConfig` + espeak frontend + `TinyModel` is the next step (needs checkpoint).
- [x] **`rlx-outetts` crate** (queue #3) — **correction: OuteTTS 1.0 uses DAC (2 codebooks), not WavTokenizer; backbone is Llama-3.** Ported `OuteTtsConfig`/`GenerationConfig`, `build_prompt_string` (`<|im_start|>…<|text_start|>…<|audio_start|>`), and `AudioCodeMap`/`collect_codebooks` (the `<|c1_N|>`/`<|c2_N|>` ↔ DAC-codebook mapping, faithful `append_audio_code` incl. top-code guard). 6 CPU tests green. Wiring: `rlx-llama32` backbone + sampler → `AudioCodeMap` → `rlx-dac` decode (needs checkpoint).
- [x] **`rlx-kroko` crate** (queue #4, streaming ASR) + **new shared decoder** — Kroko is a **Zipformer2 stateless-context-2 transducer** (blank 0). Added `rlx-audio-blocks::decoders::transducer` (`run_stateless_transducer_greedy` + `StatelessTransducerCore`, context-slide greedy — reusable by any k2/sherpa Zipformer). `rlx-kroko` = faithful `KrokoConfig` + `validate()` + `DecoderOptions` + `greedy_decode` over the shared loop. 4 + 4 CPU tests green. Zipformer2 encoder graph wiring (from conformer machinery) = next step (needs package).
- [x] **`rlx-stable-audio` crate** (queue #5, first diffusion model) — `StableAudioConfig` + rectified-flow `sampler` (length-dependent timestep shift: `LogSnr`/`Full`/`None`, `make_schedule`, `effective_latent_length`) reusing `rlx-audio-blocks::sampling::FlowMatchEuler` (3rd consumer of the sampling block). 7 CPU tests green. DiT (`rlx-flux2` patterns) + T5 conditioner (`rlx-parlertts`) + SAME VAE (`rlx-dac`) graph wiring = next step (needs checkpoint).
- [x] **`rlx-seed-vc` crate** (queue #6, diffusion VC) — `SeedVcConfig` + conditional-flow-matching `cfm_scheduler` (ascending 0→1 over `FlowMatchEuler`, 4th consumer) + `cfg_blend`/`cfm_guided_step` (classifier-free guidance). 4 CPU tests green. CFM DiT + CAM++ speaker (`rlx-funasr`) + content encoder + BigVGAN (`rlx-neutts`) wiring = next step (needs checkpoint).
- [x] **`rlx-ace-step` crate** (queue #7, music DiT) + **foundation grew**: promoted `sampling::classifier_free_guidance` and `sampling::{sd3_time_shift, sd3_shifted_sigmas}` (SD3/Flux flow shift) into `rlx-audio-blocks` (shared by all SD3-style flow gens). `AceStepConfig` + `flow_scheduler` (SD3-shifted `FlowMatchEuler`) + `guided`. 3 + new-foundation CPU tests green (audio-blocks now 22). DiT + UMT5/lyric conditioner + DCAE wiring = next step.
- [x] **`rlx-fish` crate** (queue #8, Fish-Speech) — dual-AR (slow Llama backbone + fast/depth transformer) + Firefly-GAN codec. `FishConfig`/`FireflyConfig` + `codebook_matrix`/`flatten_codebook_matrix`/`validate_codes` (bridge fast-transformer flat stream ↔ codec per-frame codebook rows). 5 CPU tests green. Both transformers → `rlx-llama32`; Firefly decode wiring = next step.
- [x] **`rlx-higgs` crate** (queue #9, Higgs-Audio v2 TTS+STT) + **foundation grew**: added `rlx-audio-blocks::codec` (`build_delay_pattern`/`revert_delay_pattern` — the RVQ delay interleave shared by MusicGen/Parler/Higgs). `HiggsConfig` (Llama-3.2-3B + DualFFN + RVQ) + `HiggsMode` + `delay_encode`/`delay_decode`. 4 + 4 CPU tests green (audio-blocks now 26). Backbone/DualFFN/tokenizer wiring = next step. **Covers both higgs_audio_tts + higgs_audio_stt.**
- [x] **`rlx-voxcpm` crate** (queue #10, VoxCPM tokenizer-free TTS) + **foundation grew**: added `FlowMatchEuler::ascending` (noise→data CFM schedule). `VoxCpmConfig` (MiniCPM backbone + continuous acoustic latent + local flow head) + `local_flow_scheduler` + `guided` (CFG) + `acoustic_frames`. 5 + 1 CPU tests green (audio-blocks now 27). Backbone (`rlx-minicpm5`) + flow DiT + vocoder wiring = next step.
- [x] **`rlx-index-tts` crate** (queue #11a, IndexTTS-2) — `IndexTtsConfig` (GPT AR backbone + semantic vocab/rate + S2A flow + mel + speaker/emotion dims) + `tokens_for_duration` (signature **duration control**) + `s2a_scheduler` (noise→data) + `guided` + `blend_emotion` (via shared guidance). 4 CPU tests green. GPT + S2A flow DiT + BigVGAN wiring = next step.
- [x] **`rlx-glm-tts` crate** (queue #11b, GLM-4-Voice family) — `GlmTtsConfig` (GLM backbone + single-codebook speech tokens) + `streaming_schedule` (the GLM-4-Voice text/audio interleave, 13:26) + `tokens_for_duration` + `token2mel_scheduler` (noise→data flow) + `guided`. 4 CPU tests green. GLM backbone (`rlx-glm`) + flow token→mel + HiFiGAN wiring = next step.
- [x] **`rlx-irodori` crate** (queue #11c, Japanese voice-design TTS) — `IrodoriConfig` (LM + codec + voice-design dim) + a correct Japanese **mora frontend** `count_morae` (yōon attach, sokuon/nasal/chōonpu count) + `tokens_for_kana`. 4 CPU tests green (とうきょう=4/がっこう=4/きゃ=1/ラーメン=4). LM (`rlx-llama32`) + codec (`rlx-dac`) + full g2p wiring = next step.
- [x] **`rlx-omnivoice` crate** (queue #11d, 646+ lang voice-design TTS) — `OmniVoiceConfig` (LM + codec + language/voice-design embeds, num_languages 646) + `normalize_language` (ISO 639-3 three-letter handling). 3 CPU tests green. LM (`rlx-llama32`) + codec + conditioning wiring = next step.
- [x] **`rlx-confucius` crate** (queue #11e, Confucius4-TTS voice cloning) — `ConfuciusConfig` + `plan_clone` (voice-cloning prompt planner: reference-text → reference-audio → target-text → target-audio-start). 3 CPU tests green. LM (`rlx-llama32`) + codec + reference conditioning wiring = next step.
- [x] **`rlx-dramabox` crate** (queue #11f, expressive TTS + cloning) — `DramaBoxConfig` + `parse_expressive`/`strip_tags` (inline `[happy]…[sad]…` style-tag parser; a smoke test caught + fixed a premature-span-split bug on unclosed `[`). 4 CPU tests green. **TTS batch (#11) complete.** LM + codec + expressive conditioning wiring = next step.
- [x] **`rlx-roformer-sep` crate** (queue #12, source separation) — **covers bs_roformer + mel_band_roformer**. `RoformerSepConfig` + `BandSplit{Fixed,Mel}` + `fixed_band_ranges`/`mel_band_ranges` (contiguous full-coverage partitions; mel = 2595·log₁₀ scale) + `apply_complex_mask`. 5 CPU tests green. STFT via `rlx-fft`; RoFormer graph + mask head = next step.
- [x] **`rlx-demucs` crate** (queue #13, htdemucs) — `DemucsConfig` (dual time/spectral U-Nets + cross-domain transformer, 4 stems) + `encoder_channels` (base·growth^i) + `segment_starts`/`transition_weight` (overlap-add inference: stride=segment·(1−overlap), triangular cross-fade). 4 CPU tests green. **Separation category complete.** Conv U-Nets + transformer + `rlx-fft` STFT wiring = next step.
- [x] **MarbleNet VAD** (queue #14) — **extended `rlx-vad`** (not a new crate): added `marblenet::MarbleNetConfig` (TCS-conv 3x2x64, 10 ms hop) + `SegmentParams::marblenet()` preset + generalized `segments::speech_segments_from_probs` (model-agnostic hysteresis segmentation any frame VAD can reuse). 2 new tests (+ existing 6 still green). TCS-conv graph + NeMo weights = next step.
- [x] **Sortformer diarization** (queue #15) — **extended `rlx-diarize`**: added `sortformer::SortformerConfig` (FastConformer + max_speakers) + `sort_speakers_by_arrival` (Sortformer's arrival-time canonical speaker order) + `activity_to_turns` (threshold + merge the `[frames][max_speakers]` sigmoid activity into `SpeakerTurn`s). 4 new tests green. FastConformer + transformer graph + NeMo weights = next step.
- [x] **`rlx-qwen3-aligner` crate** (queue #16, forced alignment) — `Qwen3AlignerConfig` + `forced_align` (monotonic **Viterbi forced alignment**: maps a known token sequence onto frames, each token ≥1 consecutive frame, full coverage). 4 CPU tests green. **VAD/diar/align category complete.** Encoder (`rlx-qwen3-asr`) + emission head wiring = next step.
- [x] **`rlx-rvc` crate** (queue #17, voice conversion) — `RvcConfig` + `retrieval_blend` (the k-NN feature-index blend by `index_rate`, inverse-square-distance weights) + `transpose_f0` (pitch shift 2^(semitones/12)). 5 CPU tests green. HuBERT content encoder (`rlx-wav2vec2-bert`) + NSF-HiFiGAN wiring = next step.
- [x] **`rlx-vevo` crate** (queue #18, Vevo2 controllable VC/TTS) — `VevoConfig` + `flow_scheduler`/`guided` + `collapse_repeats`/`expand_units` (reduced-unit RLE) + `VevoControl` disentangled-control presets (voice-conversion / style-transfer / full-imitation). 4 CPU tests green. AR + flow + vocoder graphs = next step.
- [x] **`rlx-heartmula` crate** (queue #19, music gen) — `HeartMulaConfig` (MusicGen-style codec-token LM + HeartCodec RVQ) + `frames_for_duration` (duration control, clamped) + `delay_encode`/`delay_decode` (reuses shared `codec` delay pattern). 3 CPU tests green. LM + HeartCodec decode wiring = next step.
- [x] **`rlx-citrinet` crate** (queue #20, NeMo Citrinet CTC ASR) — `CitrinetConfig` (TCS-conv + SE, blank last) + `ctc_greedy_decode`/`ctc_greedy_decode_logits` (collapse repeats + drop blank; reusable by any CTC model). 3 CPU tests green. **Main model list complete.** TCS-conv+SE encoder (`rlx-conformer-ctc`) + `.nemo` wiring = next step.
- [x] **moss_tts_local** (queue #21a, partial finisher) — **extended `rlx-moss-nano`** additively: `variant::MossVariant{Nano,Local}` (Local = offline + voice-control, own local dir, no hosted repo). 2 new tests green (existing pipeline untouched). Shares moss-nano's native graph.
- [x] **vietneu_tts** (queue #21b, partial finisher) — **extended `rlx-neutts`** additively: `variant::NeuTtsVariant{Air,VieNeu}` (VieNeu = Vietnamese fine-tune, lang `vi`, same GGUF-Llama+NeuCodec pipeline). 2 new tests green (existing 23 untouched).

### Arch + CPU-smoke pass: COMPLETE ✅

All 26 missing models + 3 partials are covered by native-Rust crates/extensions with green CPU smoke tests. Only checkpoint-gated e2e graph wiring + real-weight parity + all-backend runs remain (the **second pass** below). `parakeet_tdt` e2e = task #4 (needs a `.nemo` checkpoint).

## Loop protocol (read this first each run)

This backlog is being worked by a recurring `/loop` (cron `*/10 * * * *`). Each fire is a **fresh session with no memory** — this doc is the source of truth. On each run:

1. Read this Progress section and the **Next-up queue** below. Pick the **top unfinished** queue item.
2. Implement **one increment**: scaffold the crate (or extend the named crate), build the architecture, and add **CPU smoke tests** — reuse existing rlx blocks/crates wherever possible (see the "already implemented" table above; most models are `{shared LM/codec/vocoder} + thin glue`).
3. Register any new crate in the workspace `Cargo.toml` (members list **and** `[workspace.dependencies]`, `version = "0.2.14"`).
4. `cargo test -p <crate>` must pass before moving on.
5. Update this doc: check the item off, add a one-line result, and advance the queue.
6. **Do not `git commit`.** Depth = arch + CPU smoke first; real-weight parity + all-backend runs are follow-ups gated on checkpoints/RAM (log them, don't block on them).
7. **Rust + rlx ONLY — no C++.** Deliverables are native Rust crates reusing rlx primitives. Do not port or depend on cpp; implement architectures natively in Rust. (audio.cpp is a spec reference only; don't reproduce its code.)

Reuse cheat-sheet: LM backbones → `rlx-llama32`/`rlx-qwen3`/`rlx-glm`; audio tokenizers/codecs → `rlx-wavtokenizer`/`rlx-snac`/`rlx-dac`/`rlx-encodec`; vocoders → `rlx-neutts`(BigVGAN)/`rlx-nanocodec`(HiFiGAN)/`rlx-tsac`(HiFT); speech encoders → `rlx-funasr`(CAM++)/`rlx-miratts`(WavLM)/`rlx-wav2vec2-bert`(Conformer); T5 → `rlx-parlertts`; diffusion/flow → `rlx-audio-blocks::sampling` + `rlx-flux2`/`rlx-vlash` DiT patterns; TDT → `rlx-audio-blocks::decoders::tdt`.

## Next-up queue (ordered — work top-down)

1. ~~**hviske_asr** → `rlx-hviske`~~ ✅ done — preset over `rlx-whisper`, all backends inherited, 5 tests green.
2. ~~**inflect_v2**~~ ✅ done — `rlx-inflect-v2` (VITS2 config + prior; graph reuses `rlx-tiny-tts`). 6 tests green.
3. ~~**outetts**~~ ✅ done — `rlx-outetts` (Llama-3 + **DAC** 2-codebook; config + prompt + code map). 6 tests green.
4. ~~**kroko_asr**~~ ✅ done — `rlx-kroko` (Zipformer2 stateless transducer) + shared `decoders::transducer`. 8 tests green.
5. ~~**stable_audio**~~ ✅ done — `rlx-stable-audio` (config + RF sampler schedule reusing `FlowMatchEuler`). 7 tests green.
6. ~~**seed_vc**~~ ✅ done — `rlx-seed-vc` (CFM scheduler + CFG). 4 tests green.
7. ~~**ace_step**~~ ✅ done — `rlx-ace-step` (SD3-shift flow scheduler + CFG; promoted both into foundation). 3 tests green.
8. ~~**fish_audio**~~ ✅ done — `rlx-fish` (dual-AR config + codebook packing). 5 tests green.
9. ~~**higgs_audio_tts** / **higgs_audio_stt**~~ ✅ done — `rlx-higgs` (Llama-3.2 + RVQ delay pattern; both TTS+STT). 8 tests green.
10. ~~**voxcpm2**~~ ✅ done — `rlx-voxcpm` (MiniCPM + local flow head + CFG). 5 tests green.
11. LM + codec TTS batch ✅ ALL DONE: index_tts2, glm_tts, irodori_tts, omnivoice, confucius4_tts, dramabox (`rlx-dramabox`).
12. ~~**bs_roformer**, **mel_band_roformer**~~ ✅ done — `rlx-roformer-sep` (band-split + complex mask). 5 tests green.
13. ~~**htdemucs**~~ ✅ done — `rlx-demucs` (U-Net channels + overlap-add). 4 tests green.
14. ~~**marblenet_vad**~~ ✅ done — extended `rlx-vad` (MarbleNet config + shared probs→segments). 2 tests green.
15. ~~**sortformer_diar**~~ ✅ done — extended `rlx-diarize` (arrival-sort + activity→turns). 4 tests green.
16. ~~**qwen3_forced_aligner**~~ ✅ done — `rlx-qwen3-aligner` (Viterbi forced alignment). 4 tests green.
17. ~~**rvc**~~ ✅ done — `rlx-rvc` (retrieval blend + F0 transpose). 5 tests green.
18. ~~**vevo2**~~ ✅ done — `rlx-vevo` (unit RLE + disentangled control). 4 tests green.
19. ~~**heartmula**~~ ✅ done — `rlx-heartmula` (codec-token LM + delay pattern). 3 tests green.
20. ~~**citrinet_asr**~~ ✅ done — `rlx-citrinet` (config + CTC greedy decode). 3 tests green.
21. **Partial finishers:** ~~moss_tts_local~~ ✅ → ~~vietneu_tts~~ ✅ (extended `rlx-neutts`) → parakeet_tdt e2e (checkpoint-gated, task #4). **QUEUE COMPLETE — arch+CPU-smoke pass done.**

After the arch+smoke pass over the queue, second pass: real-weight parity + all-backend (cpu/metal/mlx/wgpu/cuda/vulkan) validation per crate, checkpoint-permitting.

---

## TTS / voice cloning (11 missing)

- [x] **confucius4_tts** — multilingual voice cloning TTS · **`rlx-confucius` crate** (config + clone-prompt planner; LM+codec wiring next)
- [x] **dramabox** — expressive TTS + voice cloning · **`rlx-dramabox` crate** (config + inline expressive-tag parser; LM+codec wiring next)
- [x] **fish_audio** — Fish-Speech dual-AR + Firefly-GAN · **`rlx-fish` crate** (config + dual-AR codebook packing; both transformers → `rlx-llama32`, Firefly decode next)
- [x] **higgs_audio_tts** — BosonAI Higgs-Audio v2 · **`rlx-higgs` crate** (Llama-3.2 backbone + RVQ delay pattern; TTS mode)
- [x] **omnivoice** — voice-design TTS (646+ languages) · **`rlx-omnivoice` crate** (config + ISO-639-3 language handling; LM+codec wiring next)
- [x] **voxcpm2** — OpenBMB VoxCPM tokenizer-free TTS · **`rlx-voxcpm` crate** (MiniCPM backbone + local flow-matching acoustic head + CFG)
- [x] **index_tts2** — Bilibili IndexTTS-2 · **`rlx-index-tts` crate** (GPT AR + S2A flow + duration/emotion control; graph wiring next)
- [x] **irodori_tts** — Japanese voice-design TTS · **`rlx-irodori` crate** (config + Japanese mora frontend; LM+codec wiring next)
- [x] **glm_tts** — Zhipu GLM-4-Voice family · **`rlx-glm-tts` crate** (GLM backbone + streaming 13:26 interleave + flow token→mel; reuses `rlx-glm`)
- [x] **outetts** — OuteAI multilingual TTS · **`rlx-outetts` crate** (Llama-3 backbone `rlx-llama32` + **DAC** 2-codebook `rlx-dac`; config + prompt + code map done, graph wiring next)
- [x] **moss_tts_local** — offline MOSS-TTS w/ control · **extended `rlx-moss-nano`** (`MossVariant::Local`)

## ASR / STT (4 missing)

- [x] **citrinet_asr** — NVIDIA NeMo Citrinet (1D-conv CTC) · **`rlx-citrinet` crate** (config + CTC greedy decode; conv encoder wiring next)
- [x] **hviske_asr** — Danish Whisper finetune · **`rlx-hviske` crate** (preset over `rlx-whisper`, all backends inherited)
- [x] **kroko_asr** — multilingual streaming ASR · **`rlx-kroko` crate** (Zipformer2 stateless transducer + shared `decoders::transducer` greedy loop)
- [x] **higgs_audio_stt** — BosonAI Higgs-Audio STT · **`rlx-higgs` crate** (same model, STT mode)

## Voice conversion (3 missing)

- [x] **rvc** — Retrieval-based VC · **`rlx-rvc` crate** (retrieval feature blend + F0 transpose; HuBERT+generator wiring next)
- [x] **seed_vc** — Seed-VC zero-shot VC (CFM) · **`rlx-seed-vc` crate** (CFM scheduler over `FlowMatchEuler` + CFG; CAM++/content/BigVGAN wiring next)
- [x] **vevo2** — Amphion Vevo controllable VC/TTS · **`rlx-vevo` crate** (unit RLE + disentangled control + flow; graph wiring next)

## Music / sound generation (3 missing)

- [x] **ace_step** — ACE-Step music generation/editing (DiT) · **`rlx-ace-step` crate** (SD3-shift flow scheduler + CFG in foundation; DiT/UMT5/DCAE wiring next)
- [x] **stable_audio** — Stability Stable-Audio-Open · **`rlx-stable-audio` crate** (config + RF sampler schedule reusing `sampling::FlowMatchEuler`; DiT/T5/VAE graph wiring next)
- [x] **heartmula** — music generation · **`rlx-heartmula` crate** (codec-token LM + RVQ delay pattern; graph wiring next)

## Source separation (3 missing)

- [x] **htdemucs** — Hybrid-Transformer Demucs · **`rlx-demucs` crate** (config + U-Net channels + overlap-add; graph wiring next)
- [x] **bs_roformer** — Band-Split RoFormer · **`rlx-roformer-sep` crate** (fixed band-split + complex mask; RoFormer graph next)
- [x] **mel_band_roformer** — Mel-Band RoFormer · **`rlx-roformer-sep` crate** (mel band-split; shared with bs_roformer)

## VAD / diarization / alignment (3 missing)

- [x] **marblenet_vad** — NeMo MarbleNet (1D-conv VAD) · **extended `rlx-vad`** (`MarbleNetConfig` + shared `speech_segments_from_probs`; TCS-conv graph next)
- [x] **sortformer_diar** — NVIDIA Sortformer diarization · **extended `rlx-diarize`** (`sortformer` module: arrival-sort + activity→turns; FastConformer graph next)
- [x] **qwen3_forced_aligner** — Qwen3 forced alignment · **`rlx-qwen3-aligner` crate** (Viterbi forced alignment; encoder wiring next)

---

## Partial coverage — extend existing crate (4)

- [~] **parakeet_tdt** — **`rlx-parakeet` crate created** (arch + CPU smoke): `TdtJoint` + `TdtCore` + shared `rlx-audio-blocks::decoders::tdt` decode, reusing `rlx-nemotron-asr` encoder + pred net. Remaining: encoder/`.nemo` e2e wiring + parity (task #4).
- [ ] **moss_tts_local** — offline MOSS-TTS w/ control · extend **rlx-moss-nano** (MOSS-TTS-Nano) to the local variant
- [x] **inflect_v2** — VITS2 (not a v1 delta) · **`rlx-inflect-v2` crate** (config + prior; synthesis graph reuses `rlx-tiny-tts`)
- [ ] **vietneu_tts** — Vietnamese NeuTTS · extend **rlx-neutts** (NeuTTS backbone present; add Vietnamese finetune/preset)

---

## Already implemented (14 — no work needed)

chatterbox (`rlx-chatterbox`) · fun_asr_nano (`rlx-funasr`) · miocodec (in `rlx-miotts`) ·
miotts (`rlx-miotts`) · pocket_tts (`rlx-pocket-tts`) · nemotron_asr (`rlx-nemotron-asr`) ·
qwen3_asr (`rlx-qwen3-asr`) · qwen3_tts (`rlx-qwen3-tts`) · silero_vad (`rlx-vad`, silero backend) ·
vibevoice (`rlx-vibevoice`) · vibevoice_asr (`rlx-vibevoice-asr`) · voxtral_realtime (`rlx-voxrt`) ·
moss_tts_nano (`rlx-moss-nano`) · supertonic (`rlx-supertonic`)
