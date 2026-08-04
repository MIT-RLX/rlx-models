# rlx-kroko

**Kroko** streaming ASR on RLX — a k2/sherpa-style **Zipformer2** transducer with a
**stateless context-2 predictor** (blank id `0`), 80-dim fbank features, chunked
streaming (chunk 269 / shift 256, subsampling factor 4).

Composition:

| Component | Reuse |
|-----------|-------|
| Greedy decode loop (stateless predictor) | `rlx-audio-blocks::decoders::transducer` (added here) |
| Streaming Zipformer2 encoder | to wire from `rlx-wav2vec2-bert` / `rlx-nemotron-asr` conformer machinery |

## What's here (checkpoint-free, tested)

- `KrokoConfig` — faithful port of audio.cpp `KrokoASRConfig` (zipformer2, sr 16000,
  fbank 80, chunk 269/256, subsampling 4, context-2, blank 0) + `validate()` for the
  package invariants audio.cpp asserts.
- `DecoderOptions` / `DecodingMethod` — greedy / modified-beam tuning.
- `greedy_decode` — drives the shared stateless-transducer greedy loop with Kroko's
  context size + blank id.

Plus the new shared decoder in `rlx-audio-blocks`:
`run_stateless_transducer_greedy` + `StatelessTransducerCore` (context-slide greedy
search), which any k2/sherpa Zipformer/transducer model can reuse.

4 CPU smoke tests here (+ 4 for the shared decoder in `rlx-audio-blocks`).

## Next step

Build the streaming Zipformer2 encoder graph (Conv2dSubsampling + Zipformer2 stacks)
and load the stateless decoder/joint weights, exposing a `StatelessTransducerCore`
for end-to-end `transcribe`, then per-backend parity. Needs a Kroko package.
