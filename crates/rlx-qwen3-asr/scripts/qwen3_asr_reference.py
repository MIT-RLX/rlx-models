"""The `transformers` Qwen3-ASR (`Qwen3ASRForConditionalGeneration`) on a
converted (`-hf`) checkpoint: the reference the rlx runner is checked against.

    python scripts/qwen3_asr_reference.py --model-dir weights/asr/qwen3-asr-0.6b-hf \
        --wav clip16k.wav [--language English]

Prints the generated token ids, then the raw decoded text. Greedy decoding.
"""

import argparse

import librosa
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--language", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).eval()
    audio, _ = librosa.load(args.wav, sr=16000)
    inputs = processor.apply_transcription_request(audio, language=args.language)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    ids = out[0, inputs["input_ids"].shape[1] :].tolist()
    print(" ".join(map(str, ids)))
    print(processor.decode(ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
