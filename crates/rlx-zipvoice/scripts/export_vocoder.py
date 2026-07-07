#!/usr/bin/env python3
"""Export ZipVoice's Vocos vocoder (`charactr/vocos-mel-24khz`) to
`vocoder_spec.onnx` — the backbone + spectral head up to (but not including) the
ISTFT (ONNX has no ISTFT op); rlx does the ISTFT in Rust.

  pip install vocos onnxscript          # in a venv
  python export_vocoder.py weights/tts/zipvoice-distill/onnx/vocoder_spec.onnx
"""
import sys

import torch
import torch.nn as nn
from vocos import Vocos


def main(out_onnx: str) -> None:
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()

    class SpecHead(nn.Module):
        def __init__(self, out):
            super().__init__()
            self.out = out

        def forward(self, x):                 # [B, L, dim]
            x = self.out(x).transpose(1, 2)   # [B, n_fft+2, L]
            mag, p = x.chunk(2, dim=1)
            mag = torch.exp(mag).clamp(max=1e2)
            return mag * torch.cos(p), mag * torch.sin(p)

    class Voc(nn.Module):
        def __init__(self):
            super().__init__()
            self.b, self.h = voc.backbone, SpecHead(voc.head.out)

        def forward(self, mel):
            return self.h(self.b(mel))

    torch.onnx.export(
        Voc().eval(), torch.randn(1, 100, 50), out_onnx,
        input_names=["mel"], output_names=["real", "imag"],
        dynamic_axes={"mel": {0: "b", 2: "l"}, "real": {0: "b", 2: "l"}, "imag": {0: "b", 2: "l"}},
        opset_version=17, dynamo=False,
    )
    print("exported", out_onnx)


if __name__ == "__main__":
    main(sys.argv[1])
