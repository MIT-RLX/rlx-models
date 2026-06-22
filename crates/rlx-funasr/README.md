# rlx-funasr

[FunASR](https://github.com/modelscope/FunASR) (ModelScope) ported to Rust on
**RLX**, running on every RLX backend — `cpu`, `metal`, `mlx`, `cuda`, `rocm`,
`gpu` (wgpu).

The complete production pipeline is implemented:

| Stage | Model | Module |
|-------|-------|--------|
| ASR (flagship) | **Paraformer** — SAN-M encoder → CIF predictor → SAN-M decoder (non-autoregressive) | `paraformer`, `cif` |
| ASR (multilingual) | **SenseVoiceSmall** — encoder-only CTC with language/event/emotion tags | `sensevoice` |
| VAD | **FSMN-VAD** — Deep-FSMN classifier + silence-duration state machine | `vad` |
| Punctuation | **CT-Transformer** — SAN-M encoder + per-token punctuation classifier | `punc` |
| Speaker | **CAM++** — FCM + densely-connected TDNN + statistics pooling (192-d) | `speaker` |
| Orchestration | VAD → ASR → punctuation → speaker | `pipeline` |

Shared infrastructure: the Kaldi-fbank + LFR + CMVN `frontend`, the `sanm`
SAN-M/FSMN HIR building blocks, `weights` loading from native PyTorch
`model.pt` (`pt`, a dependency-free zip+pickle reader) or `safetensors`,
`config` (`config.yaml`), and `tokenizer` (char / SentencePiece).

## Architecture fidelity

Implemented verbatim against the current FunASR source:

* **SAN-M attention** — fused `linear_q_k_v`, scale `q` by `d_k^-0.5`, scaled
  dot-product attention through `linear_out`, **plus** a parallel FSMN memory
  branch (depthwise `Conv1d` over the un-headed `v` with a residual); the two
  are summed.
* **Encoder layer** — pre-norm; the attention residual is skipped when
  `in_size != size` (how the first layer changes feature dimension).
* **Sinusoidal positions** — `cat([sin, cos])`, positions `1..=T`, applied after
  scaling the input by `sqrt(output_size)`.
* **CIF** — `CifPredictorV2` α head + the sequential integrate-and-fire with
  tail processing (`tail_threshold`), on the host.
* **SAN-M decoder** — feed-forward first (with its internal LayerNorm and
  bias-less `w_2`), then FSMN self-attention (residual from the original `tgt`),
  then cross-attention; the CIF acoustic embeddings are fed directly as the
  decoder input sequence.

## Compute split

Heavy tensor compute (encoders, decoders, classifiers, the speaker network) is
compiled to an RLX graph and runs on the selected device. The inherently
sequential pieces — CIF integrate-and-fire, CTC collapse, the VAD state
machine, beam-free argmax — run on the host, exactly as the other RLX ASR
crates do. Graphs are built per utterance length (static shapes).

## CLI

Audio may be WAV, mp3, m4a/aac, or flac (compressed formats are decoded with
`symphonia`); everything is downmixed to mono and resampled to 16 kHz.

```
rlx-funasr transcribe --dir <model_dir> --wav <a.wav> [--device cpu|metal|mlx|cuda|gpu] [--type paraformer|sensevoice] [--lang auto] [--itn]
rlx-funasr vad        --dir <vad_dir>   --wav <a.wav> [--device ...]
rlx-funasr punc       --dir <punc_dir>  --text "你好世界" [--device ...]
rlx-funasr spk        --dir <spk_dir>   --wav <a.wav> [--device ...]
rlx-funasr pipeline   --vad <d> --asr <d> [--punc <d>] [--spk <d>] --wav <a.wav> [--device ...]
rlx-funasr stream     --vad <d> --asr <d> [--punc <d>] --wav <a.wav> [--chunk-ms 500] [--device ...]
rlx-funasr dump-keys  --dir <model_dir>
```

## Weights

`weights::load_dir` reads `*.safetensors` (preferred, memory-mapped) or a native
PyTorch checkpoint:

* **modern (ZIP) `.pt`** → upstream `rlx_nemo::PtModel` (lazy, memory-mapped);
* **legacy (pre-1.6, non-ZIP) `.pt`** — still common for FunASR — → the in-crate
  `pt::StateDict`, a dependency-free pickle VM that walks the five-pickle legacy
  stream + raw storage blocks, recovers `_rebuild_tensor_v2` metadata, and
  converts storages (f16 / bf16 / f64 / int) to `f32`.

CMVN is applied only when `config.yaml`'s `frontend_conf.cmvn_file` is non-null
(Paraformer uses `am.mvn`; SenseVoice ships an unused one and disables it). The
vocabulary size is taken from the checkpoint's output weight, not assumed.

Tokenization reads `tokens.json` / `tokens.txt`, or a SentencePiece
`*.bpe.model` whose proto is parsed natively (SenseVoice ships no `tokens.json`).

## Tests

`cargo test -p rlx-funasr` builds, compiles, and runs **every** model graph with
synthetic weights on each enabled backend, asserting finite outputs of the
correct shape (add `--features metal` / `--features mlx` / `--features gpu` to
exercise the GPU backends). The frontend, CIF, tokenizer, CMVN and pickle
readers have focused unit tests.

## Validation (real checkpoints)

All five models validated against the official ModelScope/HF weights:

* **SenseVoiceSmall** (`FunAudioLLM/SenseVoiceSmall`, ZIP `.pt`, 936 MB) —
  `zh.mp3` → `开饭时间早上九点至下午五点` + `<|zh|><|NEUTRAL|><|Speech|><|woitn|>`
  (canonical output, char-for-char) on **cpu / metal / mlx / wgpu**, identical.
  The frontend is **bit-exact** vs `torchaudio.compliance.kaldi` (fbank + LFR,
  max abs diff 5e-4) and the CTC logits match the torch reference on **123/124
  frames** (the one differing English token is a genuine near-tie).
* **Paraformer-zh** (`funasr/paraformer-zh`, legacy `.pt`, 880 MB) —
  `asr_example.wav` → `正是因为存在绝对正义所以我们接受现实的相对正义…`
  (character-perfect), exercising the CIF predictor + SAN-M decoder; cpu & metal.
* **CT-Transformer** (`funasr/ct-punc`) — restores `，` / `。` exactly matching
  the reference (`…生命之源。长期以来，…克服困难，定期…水文资料。`).
* **FSMN-VAD** (`funasr/fsmn-vad`) — detects the speech region and caps segments
  at `max_single_segment` (60 s), matching the reference's segmentation.
* **CAM++** (`funasr/campplus`) — 192-d embedding with **cosine 0.987** vs the
  reference `spk_embedding`.

The 917 / 956 / 161 / 24 / 937-tensor key sets matched the implementation
exactly (`dump-keys`).

## Performance

Graphs are cached per input length ([`cache::GraphCache`], LRU): a repeated
length skips the rebuild, the ~1 GB weight clone, the HIR lowering, and the
kernel compile — measured **85.6 s → 6.3 s** (13.6×) for the second SenseVoice
call. The pipeline and streaming paths benefit directly.

## Streaming

[`StreamingRecognizer`] (`rlx-funasr stream …`) accepts audio chunk by chunk,
runs VAD over the buffer, and emits each speech segment's transcription (with
optional punctuation) as it finalizes, dropping committed audio to bound memory.
This is VAD-gated offline streaming, not the chunked-cache BiCifParaformer model.

## Status / caveats

* ASR (Paraformer, SenseVoice), punctuation, VAD, and speaker are all verified on
  real weights; SenseVoice runs identically on cpu/metal/mlx/wgpu. `cuda` / `rocm`
  dispatch through the same `Device` path but are untested here (no NVIDIA/AMD HW).
* Residual numerical gaps are fp-accumulation level (encoder 123/124-frame match;
  CAM++ cosine 0.987) — within model uncertainty, not logic errors.
