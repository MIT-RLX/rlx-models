"""FunASR's own Python (the `funasr` package) on a model directory: the
reference the rlx runner is checked against.

    python scripts/funasr_reference.py --model-dir weights/asr/sensevoice-small \
        --wav clip16k.wav [--language auto] [--itn]

Prints the raw transcript (the model's tags kept). The frontend's dither
is turned off: FunASR dithers at inference too (`dither: 1.0`), so its own
output changes from run to run.
"""

import argparse

import funasr.frontends.wav_frontend as wav_frontend
from funasr import AutoModel

_forward = wav_frontend.WavFrontend.forward


def _no_dither(self, *args, **kwargs):
    self.dither = 0.0
    return _forward(self, *args, **kwargs)


wav_frontend.WavFrontend.forward = _no_dither


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--language", default="auto")
    ap.add_argument("--itn", action="store_true")
    args = ap.parse_args()
    model = AutoModel(model=args.model_dir, device="cpu", disable_update=True)
    out = model.generate(input=args.wav, language=args.language, use_itn=args.itn)
    print(out[0]["text"])


if __name__ == "__main__":
    main()
