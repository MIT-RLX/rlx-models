"""VibeVoice-ASR-Streaming as the official code runs it, for the crate's checks.

    python -I streaming_reference.py <VibeVoice source> <model dir> <clip.wav> <out.json>

The source is a checkout of github.com/microsoft/VibeVoice; the model
directory holds `microsoft/VibeVoice-ASR-Streaming-7B`. The clip is a 24 kHz
mono 16-bit WAV, read as is (no resampling, as the checkpoint's
`normalize_audio` is off). It runs `streaming_generate` as the demo does
(f32 on the CPU, greedy, the checkpoint's chunk and lookahead, its default
`split_then_encode`) and writes the prompt's ids, each chunk's ids and text,
and the first chunk's speech features' first and last rows. The acoustic
latent is taken at its mean: the official `encode_speech` adds Gaussian
noise to it (the acoustic tokenizer's `std_dist_type`), which no fixed
transcript can follow; the runner takes the mean.
"""

import json
import sys
import wave

import numpy as np
import torch


def main() -> None:
    src, model_dir, wav, out = sys.argv[1:5]
    sys.path.insert(0, src)
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    with wave.open(wav) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 24000
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    cfg = json.load(open(f"{model_dir}/preprocessor_config.json"))
    frame = cfg["speech_tok_compress_ratio"] / cfg["target_sample_rate"]

    processor = VibeVoiceASRProcessor.from_pretrained(model_dir)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    tok = processor.tokenizer
    model.model.acoustic_tokenizer.std_dist_type = "none"

    # Each chunk's ids, as `streaming_generate` decodes them.
    chunk_ids = []

    class Recording:
        def __getattr__(self, name):
            return getattr(tok, name)

        def decode(self, ids, **kw):
            chunk_ids.append(list(ids))
            return tok.decode(ids, **kw)

        def encode(self, *a, **kw):
            return tok.encode(*a, **kw)

    first = {}
    encode = model.encode_speech

    def encode_first(seg):
        f = encode(seg)
        if not first:
            first["features"] = f[0].float()
        return f

    model.encode_speech = encode_first
    texts = []
    with torch.no_grad():
        for _, _, text in model.streaming_generate(
            audio_tensor=torch.from_numpy(pcm),
            tokenizer=Recording(),
            chunk_duration=cfg["chunk_frames"] * frame,
            text_audio_delay=cfg["lookahead_frames"] * frame,
            sample_rate=cfg["target_sample_rate"],
            max_new_tokens_per_chunk=256,
            temperature=0.0,
        ):
            texts.append(text)
    keys = "speaker, content"
    prompt = (
        "You are a helpful assistant that transcribes audio input into text output. "
        f"Please transcribe the following audios streamingly with these keys: {keys}\n"
    )
    f = first["features"]
    json.dump(
        {
            "clip": wav.rsplit("/", 1)[-1],
            "acoustic_latent": "mean",
            "prompt_ids": tok.encode(prompt, add_special_tokens=False),
            "chunk_ids": chunk_ids,
            "chunk_texts": texts,
            "features": {"shape": list(f.shape), "first": f[0].tolist(), "last": f[-1].tolist()},
        },
        open(out, "w"),
    )
    for t in texts:
        print(repr(t))


if __name__ == "__main__":
    main()
