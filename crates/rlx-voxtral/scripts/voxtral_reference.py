"""The `transformers` Voxtral (`VoxtralForConditionalGeneration`) transcribing
a clip: the reference the rlx runner is checked against.

    python scripts/voxtral_reference.py --model-dir weights/asr/voxtral-mini-3b \
        --wav clip16k.wav [--language en]

Prints the generated token ids, then the decoded text. Greedy decoding.
"""

import argparse

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).eval()
    inputs = processor.apply_transcription_request(
        language=args.language, audio=args.wav, model_id=args.model_dir
    )
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    ids = out[0, inputs["input_ids"].shape[1] :].tolist()
    print(" ".join(map(str, ids)))
    print(processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
