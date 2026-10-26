"""The official `qwen-tts` CustomVoice talker generating codec frames
greedily (talker and sub-talker): the reference the crate's synthesis is
checked against.

    python scripts/custom_voice_reference.py --model-dir weights/tts/qwen3-tts-0.6b-customvoice \
        --text "Hello from native Rust." --speaker vivian --language English \
        --out tests/fixtures/custom_voice_greedy

Writes `case.json` (`text`, `speaker`, `language`, `codes`: `[frames][16]`).
"""

import argparse
import json
import os

import torch
from qwen_tts import Qwen3TTSModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--language", default="English")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tts = Qwen3TTSModel.from_pretrained(args.model_dir, device_map="cpu", dtype=torch.float32)
    input_ids = tts._tokenize_texts([tts._build_assistant_text(args.text)])
    gen = tts._merge_generate_kwargs(
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=args.max_new_tokens,
    )
    codes, _ = tts.model.generate(
        input_ids=input_ids,
        instruct_ids=[None],
        languages=[args.language],
        speakers=[args.speaker],
        non_streaming_mode=True,
        **gen,
    )
    c = codes[0].tolist()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "case.json"), "w") as f:
        json.dump(
            {"text": args.text, "speaker": args.speaker, "language": args.language,
             "repetition_penalty": gen["repetition_penalty"], "codes": c},
            f,
        )
    print(len(c), "frames; rep penalty", gen["repetition_penalty"])


if __name__ == "__main__":
    main()
