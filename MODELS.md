# Models

Every model family in this workspace, one crate per architecture. This catalog is generated from each crate's `Cargo.toml` (`description` + backend feature set).

**152 model families** across 15 categories, plus 4 training crates. Shared infrastructure, servers, and benchmark crates are not listed here.

## Backends

The **Backends** column lists the build features each crate exposes. **CPU is always available**; the labels below name the accelerator backends a crate wires up:

- **All 7** = CPU · Metal · MLX · CUDA · ROCm · wgpu · Vulkan — RLX's full standard backend set.
- A comma list (e.g. `CPU, Metal, MLX, CUDA`) means only those backends are feature-gated.
- `(+CoreML)` / `(+ONNX)` mark optional Apple-ANE (CoreML) or ONNX-Runtime paths.
- Rows showing only **CPU** are CPU-native pipelines or crates whose GPU backends are not yet feature-gated (typically newer config/scaffolding stages).

Enable GPU backends at build time with matching features, e.g. `cargo build -p rlx-qwen3 --features all-backends` (or `metal` / `mlx` / `cuda` / `rocm` / `gpu` / `vulkan`). See the main [README](README.md) for routing details and per-family runbooks.

## Categories

| Category | Count |
|---|--:|
| [Language models (text LLMs & reasoning)](#language-models-text-llms--reasoning) | 26 |
| [Vision-language & multimodal (VLM / omni)](#vision-language--multimodal-vlm--omni) | 12 |
| [Vision encoders, detection & segmentation](#vision-encoders-detection--segmentation) | 9 |
| [Biomedical & scientific](#biomedical--scientific) | 4 |
| [Text & vision embeddings](#text--vision-embeddings) | 3 |
| [Image, video & 3D generation](#image-video--3d-generation) | 3 |
| [OCR & document understanding](#ocr--document-understanding) | 5 |
| [Speech recognition (ASR)](#speech-recognition-asr) | 14 |
| [Text-to-speech & speech LMs](#text-to-speech--speech-lms) | 43 |
| [Voice conversion](#voice-conversion) | 3 |
| [Music & audio generation](#music--audio-generation) | 3 |
| [Audio source separation](#audio-source-separation) | 2 |
| [Neural audio codecs](#neural-audio-codecs) | 11 |
| [Speech front-end, wake-word & DSP](#speech-front-end-wake-word--dsp) | 13 |
| [Robotics (vision-language-action)](#robotics-vision-language-action) | 1 |
| **Total** | **152** |

## Language models (text LLMs & reasoning)

Decoder LMs, MoE, hybrid SSM/attention, ternary, diffusion LMs, and speculative decoding.

| Crate | Description | Backends |
|---|---|---|
| `rlx-bonsai` | Bonsai small-reasoning runner — STUB (PLAN.md M4) | **All 7** |
| `rlx-cohere` | Cohere Command-R runner — STUB (PLAN.md M4) | **All 7** |
| `rlx-deepseek` | DeepSeek-V3 / V3.1 (MLA + fine-grained MoE), incl. Kimi-K2, for RLX | **All 7** (+CoreML) |
| `rlx-eagle3` | EAGLE3 speculative-decoding draft + scheduler primitives for RLX | CPU, Metal, MLX, CUDA |
| `rlx-gemma` | Gemma / Gemma 2 causal LMs for RLX | **All 7** (+CoreML) |
| `rlx-glm` | GLM 5.1 runner (delegates to rlx-llama32; GLM-specific RoPE/RMSNorm pending) | CPU |
| `rlx-glm4moe` | GLM-4.5 / GLM-4.6 (glm4_moe: partial-RoPE attention + DeepSeek-style MoE) for RLX | **All 7** (+CoreML) |
| `rlx-gpt-oss` | gpt-oss-20b runner (delegates to rlx-llama32) | CPU |
| `rlx-granite` | Granite (IBM) Llama-shaped runner — STUB (PLAN.md M4) | **All 7** |
| `rlx-jamba` | Jamba (Mamba-1 + attention + MoE hybrid) for RLX | **All 7** (+CoreML) |
| `rlx-laguna` | Laguna MoE (poolside Laguna S/XS) for RLX — packed GGUF generate + synth reference | CPU, Metal, MLX, CUDA, wgpu, Vulkan (+CoreML) |
| `rlx-lfm` | LiquidAI LFM2.5 runner (text) — LFM SSM LM (decode-step + state wiring) | **All 7** |
| `rlx-llada2` | LLaDA2 MoE diffusion LM + TIDE offload for RLX | **All 7** |
| `rlx-llama32` | LLaMA 3.2 for RLX | **All 7** (+CoreML) |
| `rlx-mamba` | Mamba1 selective-SSM block and multi-backend driver; SSM core via rlx-ssm flow (SelectiveScan / mamba1_step) | CPU, Metal, MLX, CUDA, ROCm, wgpu |
| `rlx-minicpm5` | MiniCPM5 causal LM runner (Llama-shaped; openbmb/MiniCPM5-1B) | **All 7** (+CoreML) |
| `rlx-minimax` | MiniMax runners for RLX — M2.5/M2.7 Lightning Attention LM, and M3 (MSA block-sparse MoE + vision tower) | **All 7** |
| `rlx-mistral` | Mistral 3+ / Ministral runner — STUB (PLAN.md M4) | **All 7** (+CoreML) |
| `rlx-nanbeige` | Nanbeige4.2 Looped Transformer LM (Nanbeige/Nanbeige4.2-3B) | **All 7** (+CoreML) |
| `rlx-nemotron` | NVIDIA Nemotron 3 Nano runner — text + hybrid Mamba2/attention LM | **All 7** |
| `rlx-neutrino` | Neutrino-8B (Fermion Research) — Qwen3 topology with FV5 ternary weights for RLX | **All 7** (+CoreML) |
| `rlx-omnicoder` | OmniCoder Qwen3-coder-shaped runner — STUB (PLAN.md M4) | **All 7** |
| `rlx-phi` | Phi 3 / Phi 4 runner on rlx-llama32 (partial RoPE, NeoX GGUF) | **All 7** (+CoreML) |
| `rlx-qwen3` | Qwen3 decoder LM for RLX | **All 7** (+CoreML) |
| `rlx-qwen35` | Qwen3.5 / Qwen3.6 hybrid trunk for RLX | **All 7** (+CoreML) |
| `rlx-tinyllama` | TinyLlama-1.1B causal LM runner (Llama-shaped; TinyLlama/TinyLlama-1.1B-Chat-v1.0) | **All 7** (+CoreML) |

## Vision-language & multimodal (VLM / omni)

Image/video (and audio) understanding, grounding, and computer-use agents.

| Crate | Description | Backends |
|---|---|---|
| `rlx-fara` | Microsoft Fara1.5 computer-use agent (Qwen3.5 multimodal) for RLX | **All 7** (+CoreML) |
| `rlx-florence2` | Microsoft Florence-2 (DaViT + BART) vision-language model for RLX | **All 7** |
| `rlx-inkling` | Inkling multimodal MoE (thinkingmachines/Inkling) for RLX | CPU |
| `rlx-kimi-k3` | Kimi-K3 (Moonshot AI) — hybrid KDA + MLA linear attention, LatentMoE, multimodal — for RLX | **All 7** (+CoreML) |
| `rlx-lfm-vl` | LFM2.5-VL runner (vision + LFM2.5 LM) | CPU |
| `rlx-llama4` | Llama-4 (MoE text + iRoPE, early-fusion vision) for RLX | **All 7** (+CoreML) |
| `rlx-locateanything` | NVIDIA LocateAnything-3B VLM (MoonViT + Qwen2.5-3B) for RLX | **All 7** |
| `rlx-mistral-vl` | Ministral / Mistral Medium VL runner (Pixtral mmproj + mistral3/4 LM) | **All 7** (+CoreML) |
| `rlx-mllama` | Llama-3.2-Vision (mllama) cross-attention vision-language model for RLX | **All 7** (+CoreML) |
| `rlx-nemotron-omni` | Nemotron-3 Nano Omni runner (text + vision + audio) | CPU |
| `rlx-qwen25-vl` | Qwen2.5-VL vision-language model for RLX (AIF / VLMEvalKit target) | **All 7** |
| `rlx-qwen3-vl` | Qwen3-VL runner (vision + Qwen3 MoE LM) — STUB (PLAN.md M7) | CPU |

## Vision encoders, detection & segmentation

ViT encoders, video encoders, open-vocabulary detection, and Segment Anything.

| Crate | Description | Backends |
|---|---|---|
| `rlx-dinov2` | DINOv2 ViT encoder for RLX | **All 7** |
| `rlx-dinov3` | DINOv3 ViT (2D-axial RoPE, register tokens, LayerScale, optional gated MLP) encoder for RLX | **All 7** |
| `rlx-grounding-dino` | Grounding DINO (IDEA-Research/grounding-dino-base) for RLX | **All 7** |
| `rlx-sam` | Segment Anything Model (SAM v1) for RLX | **All 7** |
| `rlx-sam2` | SAM 2 (Hiera) for RLX | **All 7** |
| `rlx-sam3` | SAM 3 for RLX | **All 7** |
| `rlx-siglip2` | SigLIP 2 (fixed-resolution + NaFlex) image + text encoder for RLX | **All 7** |
| `rlx-vit-elastic` | SnapViT elastic structured pruning + GLARE continual SSL pre-training for ViTs on RLX | CPU, Metal, MLX, CUDA, wgpu, Vulkan |
| `rlx-vjepa2` | V-JEPA 2 video encoder for RLX | **All 7** |

## Biomedical & scientific

Pathology / microscopy / clinical models.

| Crate | Description | Backends |
|---|---|---|
| `rlx-bioclip2` | BioCLIP-2 (OpenCLIP ViT-L-14) image + text encoder for RLX | **All 7** |
| `rlx-clinicalbert` | ClinicalBERT encoder runner (Huang / Bio_ClinicalBERT) on top of rlx-bert | **All 7** |
| `rlx-hoct` | Higher-Order Cell Tracking Transformer (HOCT) for RLX | **All 7** |
| `rlx-uni2` | UNI2-h pathology ViT-H/14 (packed SwiGLU + registers) encoder for RLX | **All 7** |

## Text & vision embeddings

BERT-family and Nomic encoders behind `RlxEmbed`.

| Crate | Description | Backends |
|---|---|---|
| `rlx-bert` | BERT graph builder for RLX | **All 7** |
| `rlx-nomic` | NomicBERT graph builder for RLX | **All 7** |
| `rlx-vision` | NomicVision encoder graphs for RLX | **All 7** |

## Image, video & 3D generation

Rectified-flow / diffusion image and image-to-3D, plus flow reward alignment.

| Crate | Description | Backends |
|---|---|---|
| `rlx-diamond` | Diamond Maps reward alignment — flow matching value functions and GLASS sampling (arXiv:2602.05993) | CPU |
| `rlx-flux2` | FLUX.2 rectified-flow image model for RLX | **All 7** |
| `rlx-trellis2` | Microsoft TRELLIS.2-4B image-to-3D (flow-matching DiTs + sparse-3D-conv VAEs + dual-grid mesh extraction) native port for RLX | **All 7** |

## OCR & document understanding

Text detection + recognition, VLM OCR, and terminal-screen extraction.

| Crate | Description | Backends |
|---|---|---|
| `rlx-ocr` | OCR engine for RLX — text detection + recognition | **All 7** |
| `rlx-ocr2` | Native RLX OCR: CRAFT-style text detector + CRNN/CTC recognizer + n-gram/lexicon correction | CPU, Metal, MLX, CUDA, wgpu, Vulkan (+CoreML) |
| `rlx-ppocrv6` | PP-OCRv6 tiny/small OCR — native RLX HIR + safetensors (no runtime ONNX) | **All 7** (+CoreML) |
| `rlx-termclean` | Fast TUI text extraction: strip chrome from terminal screens, classify line types, and reconstruct scrolled documents (pure-std, batched, multicore; optional Metal tagger) | CPU |
| `rlx-unlimited-ocr` | baidu/Unlimited-OCR (SAM + CLIP DeepEncoder + MoE LM) for RLX | **All 7** |

## Speech recognition (ASR)

Whisper, transducers, CTC, Conformers, and speech LMs.

| Crate | Description | Backends |
|---|---|---|
| `rlx-asr` | Native RLX streaming Conformer ASR (GGUF weights: encoder, AED, CTC, FSTs) | **All 7** (+CoreML) |
| `rlx-citrinet` | NeMo Citrinet 1D-conv CTC ASR on RLX — config + CTC greedy decode | CPU |
| `rlx-conformer-ctc` | NVIDIA NeMo Conformer-CTC ASR (e.g. stt_en_conformer_ctc_small) on RLX, loaded natively from .nemo | **All 7** (+CoreML) |
| `rlx-funasr` | FunASR (Paraformer, SenseVoiceSmall, FSMN-VAD, CT-Transformer punctuation, CAM++ speaker) ported to Rust on RLX — all backends | **All 7** (+CoreML) |
| `rlx-hviske` | Hviske Danish ASR on RLX — a Whisper-large-v3 finetune preset over rlx-whisper | **All 7** (+CoreML) |
| `rlx-kroko` | Kroko streaming ASR (Zipformer2 stateless transducer) on RLX — config + greedy decode | CPU |
| `rlx-nemotron-asr` | NVIDIA Nemotron 3.5 ASR Streaming (cache-aware FastConformer + RNN-T) runner for RLX, loaded natively from .nemo | **All 7** (+CoreML) |
| `rlx-parakeet` | NVIDIA Parakeet-TDT FastConformer transducer on RLX (Token-and-Duration Transducer) | CPU |
| `rlx-qwen3-asr` | Qwen3-ASR speech recognition for RLX (Qwen3-Omni audio encoder + Qwen3 decoder) | **All 7** (+CoreML) |
| `rlx-vibevoice-asr` | VibeVoice-ASR-BitNet speech recognition for RLX (I8_S ConvNeXt VAE encoders + BitNet I2_S Qwen2 LM decoder), loading Microsoft's shipped GGUFs natively | **All 7** (+CoreML) |
| `rlx-voxtral` | Mistral Voxtral speech LM for RLX (Whisper encoder + Llama decoder) | **All 7** (+CoreML) |
| `rlx-wav2vec2-asr` | Wav2Vec2 CTC forced alignment for WhisperX-style word timestamps | CPU |
| `rlx-wav2vec2-bert` | Wav2Vec2-BERT speech encoder for RLX | **All 7** |
| `rlx-whisper` | OpenAI Whisper ASR for RLX | **All 7** (+CoreML) |

## Text-to-speech & speech LMs

Voice cloning, expressive/controllable TTS, and speech-to-speech LMs.

| Crate | Description | Backends |
|---|---|---|
| `rlx-chatterbox` | ChatterBox (Resemble AI 0.5B Llama + S3Gen) zero-shot voice-cloning TTS for RLX (MIT) | **All 7** (+CoreML) |
| `rlx-confucius` | Confucius4-TTS multilingual voice-cloning TTS on RLX — config + clone-prompt planner | CPU |
| `rlx-dramabox` | DramaBox expressive TTS + voice cloning on RLX — config + inline expressive-tag parser | CPU |
| `rlx-f5tts` | F5-TTS (flow-matching DiT voice cloning) for RLX | CPU, Metal, MLX, CUDA, wgpu, Vulkan (+CoreML) |
| `rlx-fish` | Fish-Speech (dual-AR Llama backbone + Firefly-GAN codec) on RLX — config + codebook packing | CPU |
| `rlx-gemma-inflect-nano` | Gemma 3 270M + Inflect-Nano TTS pairing demo (unpublished) | **All 7** (+CoreML) |
| `rlx-gepard` | Gepard (~556M) autoregressive decoder-only TTS — Qwen3.5 backbone + NanoCodec FSQ | **All 7** (+CoreML) |
| `rlx-glm-tts` | GLM-TTS / GLM-4-Voice (GLM backbone + streaming speech tokens + flow token2mel) on RLX | CPU |
| `rlx-higgs` | Higgs-Audio v2 (Llama-3.2 backbone + RVQ audio tokenizer) on RLX — TTS + STT config and codebook delay | CPU |
| `rlx-index-tts` | IndexTTS-2 (GPT AR + flow-matching S2A + BigVGAN) on RLX — config, duration control, emotion blend | CPU |
| `rlx-inflect-nano` | Inflect-Nano-v1 English text-to-speech (FastSpeech-style + Snake HiFi-GAN) for RLX | **All 7** |
| `rlx-inflect-v2` | Inflect v2 (VITS-style end-to-end flow TTS) on RLX — config, generation options, and flow prior | CPU |
| `rlx-irodori` | Irodori-TTS Japanese voice-design TTS on RLX — config + mora-based Japanese frontend | CPU |
| `rlx-kittentts` | KittenTTS native RLX text-to-speech | **All 7** (+CoreML) |
| `rlx-kokoro` | Kokoro-82M (StyleTTS2 + ISTFTNet) text-to-speech for RLX | **All 7** (+CoreML) |
| `rlx-kyutai-tts` | Kyutai TTS (1.6B en/fr) — depth-multiplexed Helium-style TTS + Mimi codec for RLX | **All 7** |
| `rlx-luxtts` | LuxTTS (ZipVoice-distill flow-matching voice cloning) for RLX | CPU, Metal, MLX, CUDA, wgpu, Vulkan (+CoreML) |
| `rlx-maya1` | Maya1 (Llama-3B + SNAC expressive voice-design TTS, Apache-2.0) for RLX | CPU, Metal, MLX, CUDA, wgpu, Vulkan (+CoreML) |
| `rlx-melotts` | MeloTTS (~52M) multi-lingual VITS2 text-to-speech for RLX | **All 7** (+CoreML) |
| `rlx-metavoice` | MetaVoice-1B (1.2B) zero-shot voice cloning TTS for RLX | **All 7** (+CoreML) |
| `rlx-miotts` | MioTTS-0.6B (Qwen3 + MioCodec 25Hz/24kHz, Apache-2.0) for RLX | **All 7** (+CoreML) |
| `rlx-miratts` | MiraTTS (0.5B, Qwen2-based LLM + 48kHz neural codec, CC-BY-NC-SA-4.0) for RLX | **All 7** (+CoreML) |
| `rlx-moshi` | Kyutai Moshi speech-to-speech LM (Helium + depth transformer) for RLX | **All 7** (+CoreML) |
| `rlx-moss-nano` | MOSS-TTS-Nano (OpenMOSS hierarchical AR codec-LM TTS, 48 kHz, Apache-2.0) for RLX | CPU, Metal, MLX, CUDA, wgpu, Vulkan (+CoreML) |
| `rlx-neutts` | NeuTTS voice-cloning TTS — GGUF backbone + NeuCodec decoder for RLX | **All 7** (+CoreML) |
| `rlx-omnivoice` | OmniVoice massively-multilingual voice-design TTS on RLX — config + ISO-639-3 language handling | CPU |
| `rlx-openvoice` | OpenVoice v2 (~100M) zero-shot voice-cloning TTS for RLX (MIT) — native on all RLX backends | CPU, Metal, MLX, CUDA, wgpu, Vulkan (+CoreML) |
| `rlx-orpheus` | Orpheus TTS — Llama-3B speech LM + SNAC decoder for RLX | **All 7** (+CoreML) |
| `rlx-outetts` | OuteTTS (Llama-3 backbone + DAC 2-codebook) on RLX — config, prompt format, and audio-code mapping | CPU |
| `rlx-parlertts` | Parler-TTS Mini v1 (878M) voice description controlled TTS for RLX | **All 7** (+CoreML) |
| `rlx-piper` | Piper VITS text-to-speech for RLX | CPU, Metal, MLX, CUDA, wgpu, Vulkan (+CoreML) |
| `rlx-pocket-tts` | Pocket TTS — Kyutai's lightweight CPU TTS (FlowLM + Mimi codec) for RLX | **All 7** (+CoreML) |
| `rlx-qwen3-tts` | Qwen3-TTS for RLX — talker (Qwen3-shaped) + code predictor + 12Hz codec path | **All 7** (+CoreML) |
| `rlx-sesame` | Sesame CSM-1B (Llama-3.2-1B backbone + depth decoder → Mimi) for RLX | **All 7** (+CoreML) |
| `rlx-soprano` | Soprano 1.1 (80M Qwen3 AR TTS + 32 kHz vocoder, Apache-2.0) for RLX | **All 7** (+CoreML) |
| `rlx-styletts2` | StyleTTS2-family TTS for RLX (native Kokoro-82M over RLX backends) | **All 7** (+CoreML) |
| `rlx-supertonic` | Supertonic-3 (flow-matching latent TTS) for RLX | **All 7** (+CoreML, ONNX) |
| `rlx-tiny-tts` | TinyTTS English text-to-speech (VITS2/MeloTTS, 44.1 kHz) for RLX — all backends | **All 7** (+CoreML) |
| `rlx-tts` | RLX FastSpeech2 + WaveRNN text-to-speech (local GGUF or directory bundle) | **All 7** (+CoreML) |
| `rlx-voxcpm` | VoxCPM tokenizer-free TTS (MiniCPM backbone + local flow-matching acoustic head) on RLX | CPU |
| `rlx-voxtral-tts` | Mistral Voxtral-4B-TTS for RLX — codec decoder + acoustic head (vLLM-Omni port) | **All 7** |
| `rlx-zipvoice` | ZipVoice (k2-fsa flow-matching voice cloning) for RLX | CPU, Metal, MLX, CUDA, wgpu |
| `rlx-zonos` | Zonos v0.1 (1.6B transformer + DAC 44kHz, Apache-2.0) for RLX | **All 7** (+CoreML) |

## Voice conversion

Zero-shot and retrieval-based voice conversion.

| Crate | Description | Backends |
|---|---|---|
| `rlx-rvc` | RVC retrieval-based voice conversion on RLX — config + retrieval feature blend + F0 pitch-shift | CPU |
| `rlx-seed-vc` | Seed-VC zero-shot voice conversion on RLX — config + conditional-flow-matching sampler with CFG | CPU |
| `rlx-vevo` | Vevo2 (Amphion) controllable VC/TTS/style imitation on RLX — config, unit RLE, disentangled control | CPU |

## Music & audio generation

Flow/diffusion music and general audio generation.

| Crate | Description | Backends |
|---|---|---|
| `rlx-ace-step` | ACE-Step music generation (flow-matching DiT) on RLX — config + SD3-shift flow scheduler | CPU |
| `rlx-heartmula` | HeartMula music generation on RLX — codec-token LM config + RVQ delay pattern + duration control | CPU |
| `rlx-stable-audio` | Stable Audio Open (rectified-flow DiT) on RLX — config + RF timestep-shift sampler schedule | CPU |

## Audio source separation

Music stem separation.

| Crate | Description | Backends |
|---|---|---|
| `rlx-demucs` | Hybrid-Transformer Demucs (htdemucs) music source separation on RLX — config, U-Net channels, overlap-add | CPU |
| `rlx-roformer-sep` | BS-RoFormer / Mel-Band-RoFormer music source separation on RLX — config, band-split, complex masking | CPU |

## Neural audio codecs

RVQ / FSQ / VAE codecs and speech tokenizers.

| Crate | Description | Backends |
|---|---|---|
| `rlx-dac` | Descript Audio Codec (DAC) neural audio codec for RLX | **All 7** |
| `rlx-encodec` | Meta EnCodec neural audio codec (facebook/encodec_24khz) for RLX — multi-backend | **All 7** |
| `rlx-facodec` | FACodec (amphion/naturalspeech3_facodec) factorized codec for RLX — multi-backend HiFi-GAN/BigVGAN decoder | **All 7** |
| `rlx-mimi` | Kyutai Mimi neural audio codec (12.5 Hz, 24 kHz) for RLX | **All 7** |
| `rlx-nanocodec` | NVIDIA NanoCodec (nvidia/nemo-nano-codec) FSQ codec for RLX — multi-backend causal HiFi-GAN decoder | **All 7** |
| `rlx-snac` | SNAC multi-scale neural audio codec (hubertsiuzdak/snac) for RLX — multi-backend | **All 7** |
| `rlx-speechtokenizer` | SpeechTokenizer (fnlp/SpeechTokenizer) RVQ speech codec for RLX — multi-backend | **All 7** |
| `rlx-tsac` | Fabrice Bellard TSAC very-low-bitrate audio codec (44.1 kHz) for RLX | **All 7** |
| `rlx-vibevoice` | VibeVoice (microsoft/VibeVoice) acoustic σ-VAE codec for RLX — multi-backend ConvNeXt decoder | **All 7** |
| `rlx-wavtokenizer` | WavTokenizer (novateur/WavTokenizer) single-token codec for RLX — multi-backend Vocos decoder | **All 7** |
| `rlx-xcodec` | XCodec2 (HKUSTAudio/xcodec2) decoder for RLX — multi-backend RoFormer-Vocos decoder | **All 7** |

## Speech front-end, wake-word & DSP

VAD, echo cancellation, diarization, forced alignment, wake-word, learned FFT.

| Crate | Description | Backends |
|---|---|---|
| `rlx-aec` | Acoustic echo cancellation (FDAF-NLMS + RLX residual suppression) at 16 kHz | **All 7** |
| `rlx-diarize` | Native RLX speaker diarization (embedding + clustering) | CPU |
| `rlx-fft` | Learned FFT via butterfly networks — train for reference precision, run compiled on RLX backends | **All 7** (+CoreML) |
| `rlx-nanowakeword` | Native nanowakeword CNN wake-word detection on RLX (ONNX parity optional) | **All 7** (+ONNX) |
| `rlx-openwakeword` | Native openWakeWord wake-word pipeline on RLX (ONNX parity optional) | **All 7** (+ONNX) |
| `rlx-porcupine` | Porcupine-style wake-word CNN on RLX | **All 7** |
| `rlx-qwen3-aligner` | Qwen3 forced aligner on RLX — config + monotonic Viterbi forced alignment | CPU |
| `rlx-vad` | Voice activity detection (Earshot + Silero) on RLX | **All 7** |
| `rlx-voxrt` | VoxRT-style wake-word CNN on RLX | **All 7** |
| `rlx-wake` | Shared wake-word streaming API, mel frontend, and CNN primitives for RLX | **All 7** |
| `rlx-wakeword` | First-party RLX wakeword: event session, multi-phrase train/pack, ternary, optional VAD/speaker-id | **All 7** |
| `rlx-wakeword-core` | no_std-ready mel + WakeCnn for RLX wakeword (f32 + optional ternary fused kernels) | CPU |
| `rlx-wakeword-wasm` | WASM wakeword for Node, browser, and Web Workers (wasm-bindgen) | CPU |

## Robotics (vision-language-action)

Flow-matching manipulation policies.

| Crate | Description | Backends |
|---|---|---|
| `rlx-vlash` | VLASH π₀ / π₀.₅ Vision-Language-Action policies (PaliGemma + Gemma-300M flow matching) for RLX | **All 7** |

## Training crates

Fine-tuning / from-scratch training harnesses (not counted in the model total above).

| Crate | Description | Backends |
|---|---|---|
| `rlx-qwen3-tts-train` | RLX MLX/Metal LoRA training for Qwen3-TTS talker (JFK custom voice) | CPU, Metal, MLX, CUDA |
| `rlx-tiny` | Train a small nanoGPT-style LLM from scratch on TinyStories — a showcase of the RLX rlx! DSL + autodiff training flow (unpublished) | CPU, Metal |
| `rlx-tinystories` | Train a small nanoGPT-style LLM from scratch on TinyStories — a showcase of the RLX rlx! DSL + autodiff training flow (unpublished) | CPU, Metal, CUDA |
| `rlx-voxtral-tts-train` | RLX autodiff training for Voxtral voice cloning — codec encoder + LoRA | **All 7** |

---

_Generated from workspace `Cargo.toml` metadata. To regenerate after adding a crate, re-run the catalog script and update the counts in [README.md](README.md)._
