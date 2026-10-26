# Exporting a model as an `.rlxp`

Any family with an exporter writes its checkpoint as an `.rlxp` package that
rlx runs alone — `Model.open("model.rlxp").chat(…)` in every language, no
rlx-models code — and every exporter goes through one place:
`rlx_core::export` (`rlx-models-core`, feature `export`).

```sh
cargo run --release -p rlx-models --bin rlx-export --features export,runner,metal -- \
    weights/Qwen3-0.6B -o qwen3.rlxp --check
```

`rlx-export <checkpoint> -o <out.rlxp>` picks the family as its runner is
picked (`config.json`'s `model_type`, a GGUF's `general.architecture`),
writes the package, and prints what it holds. `--check` then opens the
package as rlx alone does and decodes against the family's own `LmRunner`
(feature `runner`): the logits after each prompt and the greedy tokens after
it, each prompt a user's message in the package's chat template (`--raw`:
as given). `--list` prints the exporters this build has.

| Exporter | Takes | Package |
|----------|-------|---------|
| `qwen3` | `qwen3`, `qwen2` (a Qwen2 / Qwen3 GGUF too, with a `tokenizer.json` beside it) | `generate`: a step symbolic in its cache and batch, its state inside; prompt functions (one per batch size); a speculative `check` |
| `llama32` | `llama` (Llama 3.x's scaled RoPE included), or a Llama GGUF (its K-quant tensors read from the file, `rope_freqs.weight` the RoPE) | as `qwen3`; any other Llama-shaped GGUF the runner builds (`granite`, `exaone`, `olmo2`, `nemotron`, `glm4`, `chatglm`, `phi3`, `mistral3`, `muse-glimmer`) as `llama4`: the runner's packed full-sequence graph, every arch delta in it; a Hugging Face `model_type` other than Llama's goes to `standard` |
| `nanbeige` | `nanbeige` (its layers looped `num_loops` times over shared weights) | as `llama4`, the loops unrolled as the runner unrolls them |
| `hy-mt` | `hunyuan_v1_dense` (HY-MT), or a `hunyuan-dense` GGUF | as `qwen3`: the config mapped onto Qwen3's as the runner maps it |
| `mistral` | `mistral` (context stops at the sliding window) | as `qwen3` |
| `gemma` | `gemma3`, `gemma3_text`, `gemma2` | as `qwen3`, sliding-window layers masked in the step |
| `qwen3-vl`, `qwen25-vl` | `qwen3_vl`, `qwen2_5_vl` | as `qwen3`, plus the vision tower (`generate.media`) |
| `qwen35` | Qwen3.5 GGUF | `generate`: the hybrid step, its K-quant projections read packed |
| `lfm` | `lfm2` | `generate`: a step symbolic in its cache, the convolution windows and K/V caches its variables |
| `embed`, `rerank` | BERT / XLM-R / NomicBERT (`nomic_bert`, its GPT-style config keys read), ClinicalBERT (a BERT with `bert.` names); `…ForSequenceClassification` | `embed`; `score` |
| `whisper` | `whisper` | `transcribe`: log-mel front end, encoder, decoder |
| `conformer-ctc` | a NeMo Conformer-CTC `.nemo` (`striding` subsampling), `--family conformer-ctc` | the frontend (`audio::nemo_features_with`: pre-emphasis, centred STFT zero-padded as NeMo pads, the checkpoint's window and filterbank, `ln(x + 2^-24)`, per-bin standardization over the audio's frames, the frame past it zeroed, symbolic in the samples), the encoder at each mel bucket the runner pads to with the CTC head's argmax appended, and the shared CTC program (`export::program::ctc`) answering `transcribe`: the padding, the collapse, the SentencePiece pieces |
| `moonshine` | `moonshine` (v1: tiny, base) | `transcribe`: the encoder over the raw PCM, one graph symbolic in its length (the conv frontend's three nested quotients; its RoPE rows a held table), then a decoder with no step — each token, the whole prefix through the full-prefix decoder at the runner's bucket, the encoder's states its context, the tied head on its last row |
| `moonshine-streaming` | `moonshine_streaming` (tiny, small, medium) | `transcribe`: the CMVN/asinh front end and the masks from the audio's real length, the encoder, then the decoder with no step (the whole sequence each token, as the runner decodes), symbolic in the encoder's length |
| `siglip2` | `siglip` (fixed resolution) | `pipeline`: Pillow resize, patch embedding, vision tower |
| `llama4` | `llama4`, `llama4_text` (the language model) | `generate` with no step: the full-sequence graph (MoE, RoPE and NoPE layers) over the whole context each token, as the runner runs it |
| `minimax-m3` | `minimax_m3` (the language model) | as `llama4`, one forward function per size (block-sparse attention holds a length-dependent constant) |
| `dinov2`, `dinov3`, `uni2` | DINOv2 / UNI2 (timm names; the config is read from the tensors), DINOv3 (HF) | `pipeline`: an RGB image of any size, resized and normalized as the runner does (`vision::resize_normalize`), its patch tokens assembled as the runner does on the host (`vision::patch_tokens`), the encoder; its tokens and class embedding (or a head's logits) out |
| `bioclip2` | an OpenCLIP directory (`open_clip_config.json`, `open_clip_model.safetensors`), `--family bioclip2` | `image`, a pipeline: an RGB image of any size → OpenCLIP's preprocessing (`vision::clip_resize_crop`: Pillow's antialiased bicubic on the shortest side, center crop) → patch stem → vision tower → embedding; `text`, an `embed` entry: the end-of-text row through the projection, normalized. One model: `run` an image, `embed` texts |
| `embeddinggemma` | `Gemma3TextModel` (sentence-transformers layout) | `embed`: embeddings, both attention masks, mean pooling and the dense head in the graph |
| `jina-v5` | jina-embeddings-v5, `--variant <task>` (`retrieval` by default) | `embed`: its task adapter merged, last-token pooling in the graph |
| `snac` | a SNAC decoder `.safetensors` | `pipeline`: each level's codes → the codebook lookup → the decoder (noise silent) → audio, for any number of codes |
| `nanocodec` | a NanoCodec decoder `.safetensors` | `pipeline`: Group-FSQ codes → dequantization → decoder → audio |
| `wavtokenizer`, `xcodec2` | their decoder `.safetensors` | `pipeline`: features / embeddings → decoder → the Vocos inverse STFT (`audio::vocos_istft`) → audio |
| `vibevoice` | its acoustic decoder `.safetensors` | `pipeline`: latent → audio |
| `facodec` | its decoder `.safetensors` | `pipeline`: latent and speaker embedding → the timbre (`W·spk + b`) in the graph → audio |
| `encodec` | the 24 kHz EnCodec `.safetensors` | `pipeline`: codes `[codebooks, frames]` (any bandwidth) → the residual codebooks' rows summed → decoder, its LSTM as `Op::Lstm` → audio |
| `speechtokenizer` | its 16 kHz `.safetensors` | as `encodec` (codebook 0 the semantic tokens) |
| `dac` | a DAC directory (`model.safetensors`, `config.json`), `--family dac` | `pipeline`: codes → each codebook's row through its projection, summed → decoder (weight norm folded) → audio |
| `mimi` | `MimiModel` | `pipeline`: codes → the split RVQ's rows through their projections → decoder; its transformer reads RoPE tables to the length and computes its sliding-window mask from positions |
| `kokoro` | a Kokoro-82M directory (`tokenizer.json`, `voices/`, the split bundle `onnx/rlx-split/`) | a program answering `synthesize` (espeak-ng in WebAssembly, its English data in the package; the vocabulary, the voice's style row, the length regulator, the ISTFT normalization), its encoder and decoder rlx graphs symbolic in the phonemes and the frames, imported once from the runner's ONNX at export |
| `piper` | a Piper voice directory (`<voice>.onnx.json`, the split bundle `rlx-split/`) | a program answering `synthesize` (espeak-ng in WebAssembly, the `phoneme_id_map` tokenizer, the flow noise, the length regulator); its encoder and decoder rlx graphs symbolic in the phonemes and the frames, imported once from the runner's ONNX at export; its stochastic duration predictor an rlx graph (`sdp_graph`), the spline's inverse computed for every element; `encoder_3`…`encoder_5` the encoder where its relative attention pads nothing |
| `supertonic` | a Supertonic-3 directory (`onnx/`, `voice_styles/`) | a program answering `synthesize` (the text preprocessor and unicode tokenizer, the voice styles, the noise, the flow-matching loop; the crate's own `config`/`tokenize`/`voices` reading the package's files); its four graphs rlx graphs symbolic in the text and latent lengths |
| `luxtts` | a LuxTTS directory (`tokens.txt`, `encoder_body.onnx`, `fm_decoder.onnx`, `onnx/vocoder_spec.onnx`) | a program answering `synthesize` with a voice to clone (`options.prompt: {audio, sample_rate, text}`; espeak-ng, the length regulator, the noise, the flow update); the prompt's log-mel and the vocoder's ISTFT rlx graphs (`audio::log_mel`, `audio::istft_center`); the flow decoder one graph per frames mod 4 (`fm_decoder_<r>`, its Zipformer downsampling) |
| `zipvoice` | a ZipVoice directory (LuxTTS' layout) | LuxTTS' package (`rlx_luxtts::export::export_family`) with ZipVoice's sampler defaults |
| `f5tts` | an F5-TTS directory (`F5_Preprocess.onnx`, `F5_Transformer.onnx`, `F5_Decode.onnx`, `vocab.txt`) | a program answering `synthesize` with a voice to clone (the reference trim, the tokenizer, the chunking, the flow loop, f16 between graphs as the runner's); its three graphs symbolic in the reference's samples, the ids and the frames (the STFT's frames and the frames past the reference's quotients of those) |
| `melotts` | a TinyTTS / MeloTTS bundle (`config.json`, `onnx/`, `frontend/`) | a program answering `synthesize` (rlx-inflect-nano's English frontend — its WordPiece groups the host's tokenizer — and the VITS glue); its four graphs symbolic in the phones and the frames, and the frontend's neural G2P one GRU step each (`g2p_encode`, `g2p_decode`) |
| `openvoice` | an OpenVoice v2 directory (`tone_extract.onnx`, `tone_color.onnx`) with a MeloTTS bundle in or beside it | MeloTTS' package (`rlx_tiny_tts::export::export_with`) cloning `options.prompt` (`{audio, sample_rate}`): the spectrogram (`audio::spectrogram`), the tone extractor and the converter rlx graphs symbolic in the spectrogram's frames, built at seven sizes |
| `parler-tts` | a Parler-TTS directory (`onnx/text_encoder.onnx`, `onnx/decoder.onnx`, `tokenizer.json`; DAC beside it as `parler-dac/`) | a program answering `synthesize` (`options.description` the voice): the delayed-codebook sampling loop, the runner's seeded sampler; the T5 encoder and the decoder symbolic in the description's and the transcript's ids (the decoder's causal mask rebuilt from the length), `pick` (one step's logits, the rest kept host-side), DAC's lookup and decoder (`rlx_dac::export::dac_functions`) |
| `miotts` | a MioTTS directory (Qwen3's files, `presets/*.f32`; MioCodec beside it) | Qwen3's package (`rlx_qwen3::export::export_rlxp_with`: its `generate` entry) with MioCodec's `decoder_body` and its ISTFT (`audio::istft_same`) as graphs and a program running the LM (`entry.run`, `seed_per_step`) and the codec; `options.codes` decodes given codes |
| `moss-tts-nano` | a MOSS-TTS-Nano directory (its ONNX, `codec/`, `browser_poc_manifest.json`) | a program answering `synthesize` and `run` (the frames): the prompt rows, the frame loop at the runner's buckets and random draws, the pause trimming; the prefill (its hidden states alone) symbolic in the rows, the sampled frame, the codec symbolic in the frames; `options.codes` decodes given frames |
| `orpheus`, `maya1` | an Orpheus / Maya1 GGUF (SNAC's decoder in `snac/` or `snac-24khz/` beside it, or `ORPHEUS_SNAC_PATH`) | the GGUF Llama's package, SNAC's lookup and decoder (symbolic in the frames) beside it and a program answering `synthesize` and `run` (the codes): the prompt (`{voice}: {text}`, Maya1 `<description="…"> {text}`), the language model a token at a time, its logits penalized by `penalty^count` and sampled by the host as the runner samples, the seven codes of each frame, SNAC, the peak normalization; `options.codes` decodes given codes |
| `neutts` | a NeuTTS GGUF (Air's Qwen2, Nano's Llama) with `neucodec_decoder.safetensors` (or `NEUTTS_DECODER_PATH`) and reference voices (`samples/<name>.npy` + `.txt`, or `NEUTTS_SAMPLES`) beside it | the GGUF's language-model package, NeuCodec's FSQ lookup, decoder and inverse STFT (symbolic in the frames) and a program answering `synthesize` and `run` (the codes): espeak-ng phonemizing the text and the voice's reference text (`options.ipa`, `options.ref_codes` with `ref_ipa` / `ref_text` instead), the prompt, top-k 50 / top-p 0.9 sampling through the host, the speech tokens' codes, NeuCodec; `options.codes` decodes given codes |
| `soprano` | a Soprano 1.1 directory (`tokenizer.json`, `onnx/soprano_backbone_kv_fp32.onnx`, `onnx/soprano_decoder_fp32.onnx`) | the backbone (its past caches empty, symbolic in the tokens: the logits and the last hidden states) and the Vocos decoder (symbolic in the frames) as rlx IR, and a program answering `synthesize` and `run` (the audio token ids): the prompt, the runner's sampler, the chunking past 80 prompt tokens, the offline crop |
| `chatterbox` | a ChatterBox directory (`tokenizer.json`, `onnx/`, `native/t3_lm.safetensors` with `t3_config.json`, `default_voice.wav`) | `speech_encoder`, `embed_tokens`, `flow_encoder`, `cfm_estimator` and `hift_head` as rlx IR (each symbolic in its lengths), T3 as rlx-llama32's graph (its weights referenced), the default voice's conditioning (the encoder run by rlx at its length) and a program answering `synthesize` and `run` (the speech tokens; `stage: "mel"` the flow's mel): the prompt embeds, T3 over the context padded as the runner pads it, the runner's sampler, the CFM solver with guidance, HiFT's inverse STFT, the onset polish; a `reference` tensor clones a voice (2 to 4 s of it: past 100 units the encoder's speaker branch pads to multiples of 200 frames, a graph per step) |
| `kittentts2` | a KittenTTS 2 repository (`config.json`, `lm/` and its ternary checkpoint, transcoded to `Q2_0` once; `voices/`, `cpp/default/voices.json`) with `chatterbox-turbo/s3gen_meanflow.safetensors` beside it (or `RLX_KITTENTTS2_S3GEN`) | the `Q2_0` GGUF's language-model package with a `speaker` medium (`spk_proj`, `prefill_embeds`: the speaker's row at the prompt's start token), S3Gen's encoder, meanflow estimator, HiFT f0 and decoder (symbolic in the tokens / frames; the encoder's relative positions slices of one table), each built-in voice's conditioning, and a program answering `synthesize` and `run`: the crate's NeMo normalizer, chunking and prompts, its penalties and sampler, the two flow steps, the sine source, the inverse STFT, the fade; `options.codes` decodes given codec tokens |
| `miratts` | a MiraTTS directory (its Qwen2 checkpoint, `tokenizer.json`, BiCodec's ONNX in `decoders/`) | the Qwen2 package, BiCodec's `detokenizer` (249 codes, 32 globals) and `s_encoder` as rlx IR at the runner's sizes, the reference's mel as a graph (`s_mel`: DFT and filterbank as matmuls), and a program answering `synthesize` and `run` (a reference's global tokens, or `options.text`'s codes): the prompt, the LM sampled (`penalty^count`, then the host's `sample` with min-p), the codes decoded; the voice `options.context` or a 16 kHz `reference` |
| `sesame` | a CSM-1B directory (`config.json`, `model.safetensors`, `tokenizer.json`; Mimi from `RLX_MIMI_DIR`) | the backbone and the depth decoder (rlx-llama32 over embedded rows, the checkpoint's weights referenced), `frame_embed`, the heads and the depth projection as graphs, Mimi's `lookup` and `decode` (`rlx_mimi::export::mimi_functions`), and a program answering `synthesize` and `run` (the frames' codes): the prompt's frames, each frame's 32 codes sampled top-k from the runner's `fastrand` stream, fed back until a frame of zeros, Mimi; `options.frames` decodes given frames |
| `zonos` | a Zonos v0.1 directory (`config.json`, `model.safetensors`; DAC from `parler-dac/` beside it, or `ZONOS_DAC_DIR`) | the backbone over the CFG pair (symbolic in the context, its RoPE tables params), the prefix conditioner's projection (`prefix`), the frames' rows, the heads, DAC's `lookup` and `decode`, the conditioning's small tensors as files, and a program answering `synthesize` and `run` (the aligned codes): espeak-ng, the six conditioning rows, the delay-pattern loop with guidance and the runner's sampler and EOS rules; `options.phonemes` / `options.codes` skip the frontend / the backbone |
| `metavoice` | a MetaVoice-1B directory (`first_stage.safetensors`, `second_stage.safetensors`, `speaker_encoder.safetensors`, their args, `tokenizer_metavoice.json`, `bria_16k.wav`; EnCodec from `RLX_ENCODEC_PATH`, else `weights/tts/encodec24`) | the speaker encoder over a 160-frame partial (DFT mel and the 3-layer LSTM), the first stage's prompt and step over the guidance pair (symbolic in the prompt and the caches' rows, each step's row written in place; its weights referenced), the second stage, EnCodec's `lookup`, `pre`, `lstm` and `post`, the tokenizer and the Bria clip as files, and a program answering `synthesize` and `run` (the tokens and codes; `stage: "speaker"`, the voice's embedding): the BPE, the voice's partials, the guided loop with the runner's argmax / nucleus sampler (its caches in slots, one length a request), the coarse books, the fine ones, post-processing; `options.tokens` / `options.speaker` skip the first stage / the voice |
| `gepard` | a Gepard directory (`model.safetensors`, `gepard_config.json`, `tokenizer.json`, `nano_dec_1.89kbps.safetensors`) | the Qwen3.5 backbone's prompt and step (symbolic in the prompt and the caches' rows; the step embeds the last frame's 32 codes itself; both end in the 32 codebook heads and the stop head; the weights referenced), the NanoCodec HiFi-GAN decoder (symbolic in the frames), the tokenizer, and a program answering `synthesize` and `run` (the frames): the repeated-text prompt, the runner's `fastrand` sampler and stop rule (its caches in slots), the FSQ dequantization; `options.frames` skips the backbone |
| `pocket-tts` | a Pocket TTS language directory (`model.safetensors`, `tokenizer.json`, `embeddings/`; named as its bundled config, e.g. `english_2026-10`, or `RLX_POCKET_TTS_LANGUAGE`) | the crate's FlowLM step (symbolic in the caches' rows, each new key and value written into its cache, its RoPE rows a table read at `pos`) and its 8-frame Mimi decoder block (streaming states in and out, RoPE rows read at `positions`), the released voices' key / value states, the text embeddings and `input_linear` as files, and a program answering `synthesize` and `run` (the latents): `text.rs`'s preparation and chunking over the host's tokenizer, the runner's host embeddings, `StdRng` / `Normal` noise and EOS rule, the decoder's blocks; `options.voice` picks a voice |
| `voxtral-tts` | a Voxtral-4B-TTS directory (`consolidated.safetensors`, `params.json`, `tekken.json`, `voice_embedding/*.f32`) | `embed`, the Ministral backbone's prompt and step (symbolic in the prompt and the caches' rows, GPT-J RoPE over the Mistral-native q/k, the step embedding the last frame's 37 codes; referenced), `acoustic` (the semantic head and the flow-matching transformer's 7 Euler steps with guidance, from the given noise), `codec` (the audio tokenizer's decoder, symbolic in the frames), the preset voices and `tekken.json` as files, and a program answering `synthesize` and `run` (the frames): Tekken BPE, the speech prompt, the runner's per-frame `StdRng` noise, the semantic pick and acoustic quantization, end of audio; `options.voice` / `options.frames` |
| `fish-s2` | a Fish Audio S2-Pro directory (`config.json`, `model*.safetensors`, `tokenizer.json`, `codec.pth`) | the crate's slow stack as a prompt and a step graph (symbolic in the prompt and the caches' rows, its RoPE rows a table, each new key and value written in place; the band logits), the fast decoder's step and embedding, the rows' embedding, the codec's dequantization and decoder (symbolic in the frames: its RoPE rows one table, its window mask at every length); the stacks' weights referenced, the codec's held; a program answering `synthesize` and `run` (the codes): the conversation (`prompt.rs`), the crate's sampler (`sampling.rs`, with the repetition-aware redraw), end at `<|im_end|>`; `options.codes` decodes given codes |
| `fish-drama3` | a Fish Audio Drama 3 directory (`config.json`, `model*.safetensors` and their index, `tokenizer.json`, `codec.pth`) | each backbone layer as a prompt and a step graph (symbolic in the prompt and the caches' rows; a DeltaNet layer's window and state carried, an attention layer's new key and value written in place), the routed experts as one `experts` graph whose FP8 weights and block scales the program has the host read for each run (`weights` inputs, eight experts a run), the final norm and main-token head, each tower layer's prompt and step graph (its code embeddings gathered in the graph), the rows' embedding, S2-Pro's codec; the 39 GB checkpoint referenced by name, nothing read at export; a program answering `synthesize` and `run` (the codes): the prompt and chunk history (`drama3_text.rs`, `prompt.rs`), routing (softmax, top 8, renormalized), the crate's sampler with the repetition-aware redraw, end at `<|im_end|>`; `options.codes` decodes given codes |
| `tada` | a TADA root (`tada-1b/`, `tada-codec/decoder/`, `tokenizer/`, a voice prompt `harness.tadaprompt`) | the rows' embedding (token row plus acoustic conditioning, both guidance branches), the Llama-3.2 backbone as a prompt and a step graph (symbolic in the prompt and the caches' rows, each new key and value written in place), the flow-matching head's whole ODE solve as one graph, the codec decoder's local-attention stack (its segment mask computed from the frames' token mask) and DAC stack symbolic in the frames; the checkpoint's tensors referenced by fingerprint, the folded convolutions held; a program answering `synthesize` and `run` (latents and frame gaps): the prompt plan around the voice (`voice.json`, or `options.voice`), the crate's noise, Gray-coded durations and text normalization, the frame grid, the leading-gap and runaway trims |
| `nemotron-asr` | a Nemotron 3.5 ASR `.nemo` (beside it the Hugging Face `model.safetensors`, which it references) | NeMo's frontend from the checkpoint's window and filterbank (zero-padded, the frame past the audio zeroed), the FastConformer's subsampling symbolic in the mel frames and its layers in the encoder frames (relative positions and the `chunked_limited` mask rows of tables), the language prompt kernel and the joint's projection over every frame, the prediction LSTMs' step, one frame's joint and argmax; a program answering `transcribe`: the greedy RNN-T search (`options.language` the prompt, `auto` by default), the SentencePiece pieces |
| `parakeet-tdt` | a Parakeet-TDT `.nemo` (beside it the Hugging Face `model.safetensors`) | as `nemotron-asr` (no language prompt, full attention, per-feature normalization over the audio's frames), the joint's duration head too; the program's search moves each frame by `durations[argmax]` and keeps the start-of-sequence state, as NeMo does |
| `funasr` | a FunASR SenseVoiceSmall or Paraformer directory (`model.pt`, `config.yaml`, `am.mvn`, the tokens), as its `config.yaml` says | SenseVoice: the SAN-M encoder with its four prompt rows gathered in the graph and the CTC argmax; Paraformer: the SAN-M encoder with the CIF predictor's alphas, and the decoder symbolic in the frames and the tokens with each token's argmax; positions rows of one table; the weights held (a `.pt` has no reference scheme); one program answering `transcribe`: the crate's Kaldi fbank, LFR and CMVN (`frontend.rs`), SenseVoice's `options.language` / `options.itn` and CTC collapse or Paraformer's integrate-and-fire (`cif.rs`), and the pieces |
| `qwen3-asr` | a Qwen3-ASR directory (`config.json`, `model.safetensors`, `vocab.json`, `merges.txt`) | Whisper's log-mel as a graph (`whisper_log_mel`, symbolic in the frames); the audio tower symbolic in its chunks and its audio rows, each row's attention window an input; the prompt's embedding with the audio's rows written in; the Qwen3 decoder's prompt graph (symbolic in the prompt and the caches' rows) and its stateful step; the weights referenced; one program answering `transcribe`: the reflect padding, the chunk layout, the chat prompt (`options.context`, `options.language` forced as transformers forces it), greedy decoding, the text after `<asr_text>` |
| `voxtral` | a Voxtral directory (`config.json`, the safetensors shards, `tekken.json`) | Whisper's log-mel (`whisper_log_mel`) over whole 30 s windows; the Whisper encoder (its batch-broadcast biases held as one slice) and the projector, symbolic in the windows; the prompt's embedding with the audio rows written in; the Llama prompt graph and step (its caches carry the in-place append's spare row, the token's row last); the weights referenced from the shards; one program answering `transcribe`: the padding, the transcription prompt (`options.language`), greedy decoding to `</s>`, tekken's bytes decoded in wasm |
| `nanowakeword`, `porcupine`, `voxrt` | a wake CNN `.safetensors` (`WakeCnnWeights`), `--family` the engine | `mel` (`rlx_wake`'s frontend, symbolic in the frames) and `score_{n}` (the CNN over `n` frames, one graph each `n` from 8 to 40: its strided convolutions' lengths are quotients no symbol fits); the weights held; one program answering `detect` (and `run`): the engines' 80 ms pushes, each scored as `push_pcm` scores it, the threshold and cooldown (`options.threshold`, `options.cooldown_ms`) |
| `openwakeword` | an openWakeWord directory (`embedding.safetensors`, `phrase.safetensors`, `config.json`), `--family openwakeword` | `mel`, `embed` (the embedding net over 76 frames) and `phrase` (the head over the last 16 embeddings); the wake words' program, embedding every 8 frames as the engine's queue does |
| `sanotts` | a sanoTTS voice package (`manifest.json`, `weights.fp16.bin`, `piper-phoneme-config.json`), `--family sanotts` | `duration` and `tokens` (the duration student and the acoustic token stack, symbolic in the tokens) and `frames` (the frame stack and the `piperlite` decoder, symbolic in the frames); the weights held; one program answering `synthesize`: espeak-ng (its English data under `espeak/`), the voice's codepoint table and framing, the durations rounded half to even, the expansion to frames with its positional rows |
| `inflect-nano` | an Inflect-Nano bundle (`config.json`, `acoustic.safetensors`, `vocoder.safetensors`, `frontend/`), `--family inflect-nano` | `encode` (embeddings, speaker projection, encoder blocks, prosody heads; symbolic in the tokens), `frames` (the length-regulated rows, local context, position and pitch projections, decoder blocks, the bidirectional GRU, mel head and postnet; symbolic in the tokens and the frames), `vocoder` (its zero-insertion blocks held as one slice); the weights held; one program answering `synthesize`: the crate's own text frontend built to wasm (its assets under `frontend/`), the durations, the frame layout and the loudness normalization |
| `kyutai-tts` | a Kyutai TTS directory (`dsm_tts_*.safetensors`, `tokenizer_spm_*.model`), Mimi from `RLX_MIMI_DIR` (Hugging Face `kyutai/mimi`), the default voice from `RLX_KYUTAI_TTS_VOICES_DIR` (`rlx_kyutai_tts::export::export_with` names both) | graphs over the checkpoint's tensors by their own names (referenced, none copied): `embed` (multiplexed text, delayed codes, the CFG bin's conditioning sum), `speaker` (a voice's projection, learnt padding, positions), `step` (the temporal row with its cross-attention; symbolic in the caches' rows), `dep_{k}` for each of the 32 codebooks, and Mimi's `mimi_lookup` / `mimi_decode`; the SentencePiece model as a `tokenizer.json` (`rlx_core::export::spm`); one program answering `synthesize` and `run` (the codes): the crate's own `generate.rs`, `state_machine.rs`, `delays.rs` and `sampling.rs` built to wasm |
| `moshi` | a Moshiko one-way directory (`model.safetensors`, `tokenizer_spm_*.model`), Mimi from `RLX_MIMI_DIR` (`rlx_moshi::export::export_with` names both) | graphs over the checkpoint's bf16 tensors by their own names (referenced, none copied): `embed` (the text token and the delayed codes), `step` (the temporal transformer, symbolic in the caches' rows), `dep_{k}` for each DepFormer slice (its weights narrowed from the stacked projections), Mimi's `mimi_lookup` / `mimi_decode`; the SentencePiece model as a `tokenizer.json`; one program answering `synthesize` and `run` (the codes): the prompt a token a frame, the delayed codes, the crate's sampler (`sampling.rs`) built to wasm |
| `qwen3-tts` | a Qwen3-TTS CustomVoice directory (`config.json`, `model.safetensors`, `tokenizer.json`, `speech_tokenizer/`) | `text_rows` / `codec_rows` (the projected text and codec embeddings, symbolic in the rows), `head`, the code predictor's `cp_head_{k}` / `cp_embed_{k}`, the talker's `prompt` (symbolic in the prompt and the caches' rows) and stateful `step`, the code predictor's `cp_prompt` / `cp_step` (its weights `cp.`), `speech` (the 12 Hz decoder, symbolic in the frames); the checkpoint's tensors referenced; one program answering `synthesize` and `run` (the codes): the CustomVoice prompt (speaker, language), the repetition penalty, greedy first codes with the reserved range masked, the code predictor's groups, the decoder |
| `rlx-tts` | an RLX TTS bundle (`rlx-tts.rlxp`, `.gguf`, or a directory) | FastSpeech2's `encode` (symbolic in the phones; the predictors' outputs) and `decode` (each frame's phone gathered; symbolic in the phones and frames), WaveRNN's 120 steps a mel frame as one `Op::Scan` (f16-rounded activations as the portable path; Gumbel-max of the given noise), the frontend's files, and a program answering `synthesize` and `run` (the mel; `options.mel` the vocoder's samples): the crate's frontend (`src/frontend`, `src/ops.rs`), durations and buckets, the BNNS AES-CTR noise (`WaveRnnRng::Bnns`, seed 16807), the output post |
| `neuralhash` | The vendor's `NeuralHashv3b*.espresso.net` with `neuralhash_128x96_seed1.dat` beside it (the copy macOS ships), `--family neuralhash` | `pipeline`: an RGB image of any size → Pillow's antialiased bicubic to 360 × 360 (`vision::pillow_preprocess`, any `ImagePreprocessor` as a graph) → the descriptor network → the seed projection and the `score ≥ 0` step: `embedding`, `scores`, `bits` |
| `upscale` | any checkpoint `Upscaler::open` takes (ESRGAN, Compact, SPAN, PLKSR, SwinIR, HAT, …), `--family upscale` | the network symbolic in its tile side (a convolutional net; a windowed transformer one graph per side), and a program answering `run`: the runner's tile plan (`tile.rs` itself), the stitch, the 8-bit conversion and alpha; `u8 [h, w, 3 or 4]` in and out |
| `birefnet` | a BiRefNet directory (`model.safetensors`; `BiRefNet_lite` or `BiRefNet`), `--family birefnet` | `pipeline`: an RGB image of any size → `imageops::resize` (Triangle) to 1024 and ImageNet normalization (`vision::image_resize_normalize`, the `image` crate's resize as a graph) → the network → sigmoid → the mask back to the image's size (`vision::image_resize_mask`): `alpha` `[h, w]` |
| `subject-lift` | The vendor's subject-lifting `.espresso.net` (the copy macOS ships, by default), `--family subject-lift` | `pipeline`: an RGB image → `imageops::resize` to 512 and `[-1, 1]` → the network → the saliency feathered (σ 0.75, a matmul each way) and resized back: `alpha` `[h, w]`, `gating` |
| `wespeaker` | a WeSpeaker ResNet34-LM directory (`graphs/wespeaker.rlxp`), `--family wespeaker` | `pipeline`: 16 kHz audio → the Kaldi fbank as a graph (16-bit scale, mean removed, pre-emphasis, Povey window, `ln(max(e, 1e-10))`) at the network's 148 frames (later ones dropped, missing ones zero) → the network → the first 256 values L2-normalized: `embedding` |
| `timestral` | a Timestral directory (`config.json`, `model.safetensors`), `--family timestral` | `program` (`forecast`): the crate's own `model.rs` and `forecaster.rs` built to wasm — the packed rows' causal scaling, masks and positions, the block-by-block decode, covariates (past, known, categorical), symmetric averaging, clipping and the quantile map — around graphs symbolic in the rows and context patches, one per decoder length a block pads to and the covariate window's. `run({context: [B, V, T]}, {horizon})` returns `quantiles` `[B, V, H, Q]` and `levels` |
| `sam` | a SAM v1 ViT-B/L/H checkpoint (safetensors or GGUF), `--family sam` | `program` (`segment`): the crate's own `image.rs` (the long side resized to 1024, normalized, padded) and `points.rs` (the points' and boxes' Fourier embeddings) built to wasm, around `encode` (the patch embedding, encoder and neck), `mask_embed` (a mask prompt's downscaling) and `decode` (the two-way transformer, upscaling, hypernetwork MLPs and IoU head as one graph symbolic in the prompt tokens), the image embedding kept in the runtime between them. `run({image: u8 [h, w, 3], points, labels, boxes, mask}, {multimask})` returns `masks` `[k, 256, 256]`, `iou` and `size` |
| `pplx-decider` | a pplx-decider GGUF (protocol v3), `--family pplx-decider` | `program` (`decide`): the crate's own `protocol.rs` built to wasm — the request validated, the prompt rendered, tokenized by the package's tokenizer, split into question chunks past the longest bucket, each answer's distribution — around one graph per sequence bucket (the token and RoPE rows gathered, the Qwen3.5 trunk with bidirectional full attention and its Q8_0 projections read from the GGUF, the hidden state at each marker, the readout's 255 rows), symbolic in the markers and the image rows; with the mmproj GGUF beside it, the vision tower as one graph symbolic in the merged grid (the position table resampled in-graph), the program preprocessing each picture with the crate's `vision.rs` and placing its rows at their M-RoPE positions. `run([], {request: "<JSON text>"})` returns `decision`, the reference's JSON; pictures go as `data:` URLs |
| `d1` | a d1 (LFM2.5-VL) checkpoint directory, `--family d1` | `program` (`decide`): the crate's own `prompt.rs` built to wasm — the questions validated, the prompt rendered and tokenized by the package's tokenizer, the state once and each question after it, each answer's distribution — around one graph per length bucket (the token embeddings, the LFM2 text model, the last position against the tied head's rows of the answer tokens, symbolic in those and the image rows); the SigLIP2 NaFlex tower at the processor's patch maximum and the projector, the program processing each picture with the crate's `processor.rs` (the position grid resampled by SigLIP2's `pos_grid.rs`) and placing its features over the image tokens. `run([], {request: "{state, questions, images}"})` returns `decision`, `system_one`'s JSON; pictures go as `data:` URLs |
| `kev` | a kev run directory (`head.pt`, `adapter_model.safetensors`, `tokenizer.json`) with its Qwen3.5 base beside it (or `base/` in it), `--family kev` | `program` (`decide`): the crate's own `api.rs`, `encode.rs`, `json.rs` and `pointer.rs` built to wasm — the request read in order, encoded, one causal row per question bucketed as the runner does, the pointer head over the states at each `<decide>` and `</opt>` — around one graph symbolic in the rows, their length and the positions read (the backbone with the LoRA folded in, the folded projections held, the rest referenced from the base). `run([], {request: "<body>"})` returns `decision` and `probabilities` |
| `carbon` | a Carbon directory (with Qwen3's `tokenizer.json` beside it, which the release leaves to its base), `--family carbon` | the Llama backbone's `generate` entry (as `llama32`), and a `program` (`complete`, open option `entry`) answering `generate` with the crate's own `dna_config.rs` and `hybrid.rs` built to wasm: text through the package's BPE, nucleotides through the 6-mer table, a DNA prompt left open as `CarbonRunner::complete` leaves it |
| `minicpm-v` | a MiniCPM-V 4.6 / 4.7 directory (`minicpmv4_6`, `minicpmv4_7`) | `program` (`chat`): the crate's own `preprocess.rs` (slicing), `prompt.rs` (chat prompt, placeholders), `feeds.rs` (each crop's feeds) and `positions.rs` (canvas M-RoPE) built to wasm, around a vision graph per crop bucket and the Qwen3.5 trunk's decode step with its state inside (`rlx_ir::stateful`: the KV caches written in place, the recurrent states replaced) taking a token or an image row and its M-RoPE positions; the prompt runs through the step token by token, the caches staying in the runtime's slots, then the reply. `run([image0, …], {prompt, max_tokens, thinking})` returns `text` |
| `silero-vad`, `earshot` | Silero VAD 16 kHz (rlx-vad's embedded weights, or a `silero_vad_16k.safetensors`), `--family silero-vad`; Earshot's embedded weights, `--family earshot` | one frame of the network as a graph (`step`: frame and recurrent state in, probability and state out; Earshot's mel frontend and ring buffer in it too) and a program answering `run`: the frames, Silero's 64-sample context and the state carried between them, the segments (the crate's `segments.rs`): `probs`, `segments` |
| `ten-vad` | its embedded weights (or a `.safetensors`), `--family ten-vad` | `pipeline`: 16 kHz PCM (int16 units, whole 256-sample hops) → the frontend and pitch track as graphs (the anti-alias biquad and the Viterbi each an `Op::Scan`) → the network, an `Op::Scan` over the hops with its periodic LSTM reset → one voice probability per hop |
| `standard` | any dense decoder `standard_decoder` accepts (Llama, Qwen2/3, Mistral, Gemma 2/3, Phi-3, OLMoE, Qwen MoE, …) | as `llama4`; the exporter `detect` falls back to for a checkpoint no family claims |

Checked against their runners with `--check` (greedy tokens identical,
logits within 3e-6): Qwen3-0.6B (CPU and Metal), Qwen3.5-0.8B Q4_K_M,
LFM2.5-230M, and small random-weight Llama 3 (scaled RoPE) and Mistral
checkpoints. The same Llama package with its scaling dropped diverges from
the runner on every prompt. Each crate's `tests/export_rlxp.rs` (and
`rlx-llama32/tests/export_standard.rs`) checks its exporter against its own
runner on random weights it writes with `check::write_checkpoint`, no
download: Llama 4, MiniMax M3, the `standard` exporter against
`Llama32Runner` (72 tokens, logits within 3e-7), DINOv2 (with and without
a classifier head and registers), DINOv3 (plain and gated MLP) and UNI2,
each for an image at the model's size and one of another shape, within
1e-4. EmbeddingGemma-300M and jina-embeddings-v5-text-nano are checked on
the real checkpoints (opt-in: `RLX_EMBEDDINGGEMMA_DIR`, `RLX_JINA_V5_DIR`):
`Model.embed` from the package within 1.5e-7 of the runner's, a text past
the sliding window included; the two Jina tasks' packages embed apart.
The codecs are checked on the real weights each crate's `tests/fixtures`
holds, at two lengths (symbolic): SNAC within 6e-7, NanoCodec and
VibeVoice exactly, WavTokenizer within 8e-5 of a 1.3 peak (its host DFT
rounds cosines in f32), XCodec2 within 3e-6, FACodec within 2.2e-5 for two
speakers, EnCodec within 9e-6 of a 2.7 peak at all 8 codebooks and at 4,
SpeechTokenizer within 6e-6 of a 2.2 peak at the same two cuts. DAC
44.1 kHz (`RLX_DAC_DIR`, or `weights/tts/parler-dac`) within 4e-7 and Mimi
(`RLX_MIMI_DIR`, or the crate's `.cache/mimi`) within 7e-6, 140 frames
included — 280 transformer steps, past its 250-step window. BioCLIP-2 on its real
weights (`RLX_BIOCLIP2_MODEL`, or `weights/bioclip-2`): texts within
1.2e-7, images at cosine 1.0 and within 1.2e-4 of their peak — the
preprocessing graph and the runner, both Pillow's bicubic in floats, put
a few pixels a level apart (each within 154 in 150,528 of Pillow itself
on photo sizes). Kokoro-82M on its real checkpoint: two texts, an American and a British
voice, the same samples as `NativeKokoro` (off by 0). A Piper voice (en_US-lessac-medium): two texts and two punctuation-only inputs, the sample counts of `NativeVits` and the samples within 7e-6. Supertonic-3: English and Spanish, two voices, off by 0. LuxTTS: three texts cloning the fixture's voice, cosine ≥ 0.999999998 (the prompt's log-mel a matmul DFT, 7e-5 off the runner's FFT in the log domain). ZipVoice-distill: cosine 0.99999999975. F5-TTS (8 flow steps): off by 0. TinyTTS / MeloTTS: two texts (one with a word only the G2P knows), the sample counts of `TinyTts` and the samples within 3e-6. OpenVoice: cosine 0.999999999 against `OpenVoice`, with and without flow noise. Parler-TTS: greedy and sampled, cosine 1.0 against `NativeParler`. MioTTS: the first 16 sampled codes the runner's (the two LMs' rounding then tips a near tie), the codec on the runner's codes cosine 1.0. MOSS-TTS-Nano: all 96 frames the runner's; its codec onnxruntime's within 8.5e-6 (the runner's own decode returns NaN). Orpheus 3B (Q4_K_M): the first 28 sampled codes the runner's, SNAC on them cosine 0.99997. Maya1: its speech, greedy, cosine 0.99995. NeuTTS Air: NeuCodec transformers' within 2e-6 (cosine 1.0), the first 28 sampled codes the runner's. Soprano 1.1: its audio tokens the runner's, sampled and greedy, its speech cosine 1.0. ChatterBox: its 40 speech tokens the runner's (native T3), sampled and greedy, the flow's mel and the speech at cosine 1.0. Zonos v0.1: 24 of 24 delay-pattern frames the runner's, greedy and sampled, DAC on them cosine 1.0. MetaVoice-1B: the Bria voice's embedding cosine 0.9999999997, 40 of 40 first-stage tokens the runner's (greedy and sampled), its 160 fine codes and the speech (cosine 0.999999999) on the runner's tokens. Gepard: 40 of 40 frames the runner's, greedy and sampled, the decoder on them cosine 1.0. Pocket TTS (english_2026-10, voice alba): a one-chunk text sampled and at temperature 0 and a two-chunk text, the sample counts of `PocketTts::generate` and its speech at cosine 1.0. Voxtral-4B-TTS (neutral_female): 24 of 24 frames the runner's, the codec on them cosine 1.0 (the runner first fixed against vLLM-Omni: see CHANGELOG). Fish S2-Pro: 24 of 24 frames the runner's, the codec on them cosine 1.0. Paraformer (zh): its example and SenseVoice's Chinese one transcribed as the runner does. SenseVoiceSmall: the JFK Rust speech (language detected and given) and its Chinese example transcribed as the runner does (the runner first fixed: it never applied its CMVN; see CHANGELOG). Parakeet-TDT 0.6B v3: the JFK Rust speech transcribed as the runner does, punctuated, word for word (the runner first fixed: see CHANGELOG). Qwen3-TTS 0.6B CustomVoice (vivian, English): greedy, 30 of 30 frames `qwen-tts`'s, its speech the fixed runner decoder's at cosine 1.0. Kyutai TTS 1.6B (alba-mackenna): greedy, 35 of 35 Mimi frames the runner's and its speech at cosine 0.9999999995; sampled with the runner's seed, 22 of 34 frames before the graphs' rounding tips a near tie. Moshiko (one-way, 12 steps): 10 of 10 Mimi frames the runner's, greedy and sampled with its seeds (the package's CPU step is ~50× the runner's time: open). Inflect-Nano: two texts (numbers and a homograph, one at speed 1.2), the host reference's sample counts at cosine ≥ 0.99999996 (the graph vocoder's 2e-4 of the README, through the loudness gain). sanoTTS (amy-en-1p46m): two texts, one at length scale 1.3, the host reference's sample counts and samples within 5.1e-6. The wake CNN (`student_lite`'s layout, weights of both signs) and openWakeWord: every 80 ms push of a JFK clip scored as the engine scores it, within 3e-5 and 1.5e-4 (the engines' naive f32 DFT). Voxtral Mini 3B: both JFK clips transcribed as the runner does, which transformers' `VoxtralForConditionalGeneration` does too. Qwen3-ASR 0.6B: the JFK clips (one attention window and two) transcribed as the runner does, which transformers' `Qwen3ASRForConditionalGeneration` does too. Nemotron 3.5 ASR: the JFK Rust speech transcribed as the runner does, word for word (the runner first fixed against Hugging Face transformers: see CHANGELOG), the frontend graph's mel the runner's within 3e-4. TADA (tada-1b, the JFK harness voice): its 13 frame gaps the runner's, the latents and the speech at cosine 1.0. Fish Drama 3: 8 of 8 frames the runner's (its experts the runner's within 6e-6 on a layer's real weights), the codec on them cosine 1.0. RLX TTS: the mel within 1e-6 (cosine 0.9999999997); the vocoder the runner's portable path for ~570 samples (its fused GRU arithmetic and the scan's separate ops then tip a decision; greedy alike), Whisper reading the package's speech exactly. Sesame CSM-1B: greedy, 118 of the first 128 codes the runner's (then its eager f32 and the graphs tip a near tie), Mimi on the runner's frames cosine 1.0. MiraTTS: a reference's 32 global tokens, the LM's 40 sampled codes and BiCodec's audio (cosine 1.0) the runner's. KittenTTS 2: its LM's prompt logits the runner's within 8e-5 (the first 14 sampled tokens the same), the greedy text path's first 42 codec tokens, S3Gen on the runner's tokens exact (mu and mel 0 off, speech cosine 1.0). Moonshine v2 streaming tiny, small and medium on
their real weights (`weights/asr`): the JFK clip and a cut of it that is
no whole number of frames, word for word. TEN-VAD on its fixture clip, part of it, and it
repeated to 2000 hops (past the LSTM reset): within 4e-6 of the runner,
no voice decision apart. Moonshine (v1) tiny on its real weights: the JFK clip and a cut of it, word for word the runner's. Silero VAD: the JFK clip and the clip with a second of silence in it, every frame's probability within 9.4e-6 of `SileroSession`'s, the segments the same; Earshot on 3 s of it, within 6.6e-6 of `Detector`'s, the segments the same. WeSpeaker on its real weights: the JFK clip cut to 1 s, 1.5 s and 4.4 s, cosine 1.0, within 3.9e-6. Timestral, on synthetic weights and the real 5M checkpoint: one series, two series of two variates with gaps and a late start, past and known covariates, and a raw forecast over 65 decode blocks; on synthetic weights exactly `TimestralForecaster`'s, on 5M within 8e-6 of the largest value. SAM v1 ViT-B on its real weights and a photo: one point, two points, a box and a point with a mask prompt, three masks and one, logits within 1.3e-3 of values up to 33 (no pixel on the other side of 0), IoU within 2.3e-5 of `Sam::forward`'s. pplx-decider (the Q8_0 release) on the nine text requests of its reference fixtures and one past the package's longest bucket (two sequences): every answer within 7.2e-6 of `PplxDecider::decide`'s, the same choices; its five image requests (one picture and two, a document past 1,700 tokens) within 8.1e-6. d1 on its bf16 checkpoint and the five text cases of its reference fixtures (one question and several, many options, a JSON state, Unicode): within 6e-7 of `D1::system_one`'s; its four picture cases (one picture under one question and several, a tiled 1400 × 900 one, a small one) within 1.2e-6. kev-0.8b over Qwen3.5-0.8B-Base: the example ticket, one question alone and an object state, every probability equal to `DecisionModel`'s. Carbon-500M: an open DNA prompt, `<dna>` inside text and plain text, 24 greedy tokens each, the same text as `CarbonRunner::complete`. MiniCPM-V-4.7-1B: a photo and a question, and a question alone, 24 greedy tokens each, the same reply as `MiniCpmVRunner::chat`. Conformer-CTC small (English) on its real `.nemo`: the JFK clip and a cut of it that is no whole number of frames, word for word the runner's. Every Llama-shaped GGUF arch above, written as a two-layer Q8_0 GGUF (`rlx-llama32/tests/export_gguf_arches.rs`): the runner's tokens, logits within 6.2e-6 (four layers; Muse Glimmer's sliding and NoPE layers both). Nanbeige (`num_loops = 2`) and HY-MT on random weights: the runner's tokens. NeuralHash on the model macOS ships: the runner's hash at four image sizes and aspects, its embedding within 1.7e-5 of its scale (a few resized pixels in ten thousand a level apart: the graph's bicubic coefficients are f32, Pillow's f64). The upscaler on the real Compact (`realesr-animevideov3`) and ESRGAN (`RealESRGAN_x4plus_anime_6B`) checkpoints: byte for byte the runner's, one tile and several, RGB and RGBA. BiRefNet_lite on its real weights, the network at 512: alpha within 1.8e-7 of the runner's at two image sizes. Subject lifting on the model macOS ships: alpha within 3e-7 at 300 × 200; at 640 × 640, where the resize graph's tap sums (another order than the `image` crate's) put 16 of 786,432 pixels a level apart, within 1.4e-3 at worst and 1.3e-7 on average. NomicBERT, and ClinicalBERT
against `ClinicalBertRunner` (`rlx-clinicalbert/tests/export_rlxp.rs`), on
random weights.

A package with no decode step (`generate.prefill` and no `functions`) runs
its prompt function over the whole context for each token, as a runner
with no cache does: O(n²) in the context, exactly the runner's numbers.
`prefill.functions` lists one function per length when the graph is not
symbolic in it.

## What the one exporter does

- A model's host logic that graphs say badly (a phonemizer, a search, a
  decode loop) goes in a **program**: `export::program::build` compiles a
  program crate (a `cdylib` over `rlx-program`) for wasm32, and
  `export::program::entry` adds it as an entry of kind `program` beside the
  graphs it runs (rlx's `docs/rlxp.md`, "Programs").
- A package runs rlx graphs only. Graphs a runner imports from ONNX at
  each length are imported by the exporter, at a few lengths, and made one
  graph symbolic in it with `graph::symbolic_in_lifted`: each constant the
  import bakes from the length is rebuilt from it — a run of one value; a
  block repeated, each repeat a step more (an `arange`, a
  `repeat_interleave`'s indices, an upsampler's staircase, clamped or not,
  and its interpolation weights), built in the graph; or the start of one
  longer table, held. Each graph is built at three lengths and the third
  tells a line from a square (an attention's relative positions,
  `2·l² + l − 1`) where two cannot. A constant that holds a length (a
  scalar, a shape) is counted out in the graph (`sum(ones[n])`); a
  countdown is repeats reversed. `symbolic_in_lifted_sizes` takes several
  symbols (a flow decoder's text and latent lengths). A graph whose
  structure depends on the length modulo a factor (a Zipformer's
  downsampling) is exported once per residue, each symbolic in the
  quotient.
- Build sizes: different remainders by any stride or hop (a quotient is
  fit only where they differ), and no two symbols' in proportion across
  the builds (`generalize_checked` refuses them). Where strided convs nest,
  build at more sizes (`symbolic_in_lifted_many`): a few points fit a
  quotient by chance. A causal mask baked as a constant (`[L, L+1]`, ones past
  the diagonal) is rebuilt from row and column indices; a param as long as
  the length (a fill the import made a param) is refused — bake it. A run of one value
  expands to the constant's own shape.
- A language model inside a TTS is the LM exporter's package
  (`export_decoder_with` / `rlx_qwen3::export::export_rlxp_with`, a hook to
  add the codec's graphs and the program); the program generates through
  `entry.run` (`seed_per_step` samples as a runner that reseeds each step).
  A sampled decoder's tokens agree with the runner's until the two
  implementations' rounding tips a near tie: test the LM's first tokens and
  the codec on the runner's codes apart.
- A graph that reads one param twice (a tied head beside the embedding
  lookup) is merged to one node when the package is written: a runtime
  binds a name to one node, and a second would read zeros.
- A voice to clone reaches a program as `options.prompt`: `{audio:
  [samples], sample_rate, text}`. `rlx_tiny_tts::model::import_rlx_graph`
  imports an ONNX graph as a package holds it. ONNX stays the reference the
  package is checked against.

- `Checkpoint` reads a Hugging Face directory or a GGUF: its config,
  tokenizer, chat template, stop ids, sampling defaults
  (`generation_config.json`), card and license, and a weight loader.
- `Package` writes the package: functions, entries, sidecars, held tensors.
  Each param that equals a checkpoint tensor (as stored or transposed; a
  GGUF's as rlx dequantizes it) is referenced from the checkpoint instead of
  copied (`Referencer`), so a package is a few MB beside its weights.
- `decoder::Decoder` is a decoder-only language model: its prompt graph and
  decode step built at a size, its RoPE rows. `decoder::export_decoder` makes
  them symbolic, moves the cache into the step (`rlx_ir::stateful`), adds the
  prompt, batch and check functions, and writes a `generate` entry.
- `graph::export_stages` / `export_graph` write any graphs as a `pipeline`
  entry — tensors in, tensors out — so whatever a family builds runs from the
  package with `Model.run`.
- `rope::Rope` computes RoPE tables from a config, scaled as its
  `rope_scaling` says (`llama3`, `linear`, `yarn`, `longrope`): the tables
  are data, so a scaled model needs no graph of its own.
- `decoder::Forward` is a language model as one full-sequence graph;
  `decoder::export_forward` writes it as a `generate` entry with no step,
  symbolic in its length or, failing that, one function per length (a
  param whose values depend on the length gets one copy per size).
- `standard` exports any dense decoder `standard_decoder` accepts from its
  `config.json` alone.
- `vision` is a vision transformer's host steps as graphs: the resize and
  normalization, the patch tokens (class, register and position
  variants), the class row; `vision::export_vit` writes a ViT package.
- `graph::Pipeline` is a `pipeline` package: stages (each on a device of
  its own if it must), symbolic dims, input contracts, outputs from any
  stage, metadata.
- `audio::vocos_istft` is a Vocos head's inverse STFT as a graph (DFT
  bases as held params, overlap-add, the window envelope).
- `graph::symbolic_in` makes a graph built at fixed lengths symbolic in
  its length — a dim may be any multiple of it plus a constant (a
  convolution's padding), checked at a third length — and
  `graph::broadcast_params` turns params a builder materialized at the
  length into data a package holds: a block repeated along it (zero
  padding) as one slice expanded, a table (RoPE) as its full rows, read up
  to the length; `graph::per_size` names the length-dependent params of
  per-length graphs apart.
- `check::check_generate` compares a package's decode with a runner's;
  `check::write_checkpoint` writes a random-weight checkpoint for a test.

## Adding an exporter

A decoder-only family implements `Decoder` — usually a hundred lines over
the graphs its runner already builds (see `rlx-gemma/src/export.rs`) — and
registers it under its runner's name:

```rust
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("gemma", "text-generation", export_rlxp)
            .claims(&["gemma3", "gemma3_text", "gemma2"]),
    );
}
```

then adds a line to `register_builtin_exporters` in
`crates/rlx-models/src/export.rs` and its crate's `export` feature to the
facade's. A family whose step is not a Qwen-shaped one writes its own entry
over `Package` (`rlx-lfm/src/export.rs`, `rlx-qwen35/src/export.rs`). Run
`rlx-export … --check` on a real checkpoint before listing it above.

## Not exported yet

Language models:

- **Llama 3.2 Vision (`mllama`)**: it generates only from an image, which its
  runner splits into tiles on the host before the vision tower. Exporting it
  needs that tiling as a package stage.
- **qwen4exp**: each step reads per-layer embeddings the host looks up by an
  n-gram hash of the recent tokens, from a 51-billion-weight table outside
  the checkpoint. Neither a graph nor an entry holds that yet.
- **A mixture-of-experts Qwen3 GGUF**: the Qwen3 exporter takes a dense
  Qwen2 / Qwen3 GGUF and refuses one with experts.
- **Mixture-of-experts Qwen3 and Gemma** through the decode-step exporter;
  OLMoE / Qwen MoE export through `standard`, without a step.
- The `standard` fallback covers a family only as far as its runner is the
  plain config-driven decoder: run `rlx-export … --check` before relying on
  it for a family with a runner of its own.

The other model crates, by what an exporter needs from each:

| Kind | Crates | What it takes |
|------|--------|---------------|
| Vision encoders | vjepa2, vision, hoct, denoise | Their host steps as graph stages (`vision` has the ViT ones), then a `pipeline`. `vision` (Nomic Vision) first needs its runner checked against the reference: it takes v1.5's `selector.norm1` as the last norm and pools the class row, leaving the rest of the `selector` attention pooler unused |
| Audio codecs | every codec's encoder; tsac (its codes come from a range coder on the host; its neural decoder is DAC's) | Their host steps as graph stages; for tsac, the entropy decoder as an entry |
| Speech recognition | citrinet, kroko, wav2vec2-asr | `transcribe` (as Whisper's); transducers need their search loop as an entry |
| Speech synthesis | kittentts (the ONNX v1) (styletts2 runs Kokoro, exported as `kokoro`); the small crates (styletts2, outetts, index-tts, …) hold no runnable model yet | Each one's front end, sampling loop and codec as a program and graphs, as the speech synthesizers exported above have |
| Diffusion and flow models | flux1, flux2, stable-audio, ace-step, trellis2, … | `pipeline` with its `loop` and the scheduler's arithmetic in the graph |
| VAD, small nets | timesfm3 | A `pipeline` over each graph, the feature front end (mel, fbank) as a stage |
