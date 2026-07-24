# rlx-melotts

MeloTTS — MyShell's ~52M multilingual VITS2 TTS for RLX (**MIT**).

Inference uses the shared [`rlx-tiny-tts`](../rlx-tiny-tts/) engine. Hub weights are
the same pack as TinyTTS:

- [`eugenehp/tiny-tts-rlx`](https://huggingface.co/eugenehp/tiny-tts-rlx) → `tiny-tts.rlxp`
- Local alias: `weights/tts/melotts` → `weights/tts/tiny-tts-rlx`

```bash
just fetch-tiny-tts          # or just fetch-melotts
just melotts-demo "Hi." metal
```

```rust,ignore
use rlx_melotts::{MeloTts, InferOpts};
let tts = MeloTts::load(rlx_melotts::resolve_bundle_dir()?)?;
let wav = tts.synthesize("The quick brown fox.", &InferOpts::default())?;
```

See [`rlx-tiny-tts` README](../rlx-tiny-tts/README.md) for pack layout, backends, and
`AssetSource` loading. There is no separate MeloTTS weight dump on Hub
([`eugenehp/melotts`](https://huggingface.co/eugenehp/melotts) is a redirect card).
