#!/usr/bin/env python3
"""Export LuxTTS's Vocos vocoder to `vocoder_spec.onnx` — the backbone + spectral
head *up to but not including* the ISTFT (which ONNX can't represent). Output is
the STFT real/imag coefficients; rlx-luxtts does the ISTFT in Rust.

The Vocos weights (`vocoder/vocos.bin`) are not published as ONNX, so run this
once after downloading the LuxTTS repo:

    pip install vocos onnxscript          # in a venv
    python export_vocoder.py <luxtts_dir>/vocoder/vocos.bin <luxtts_dir>/onnx/vocoder_spec.onnx
"""
import sys
import torch
import torch.nn as nn
from vocos.models import VocosBackbone


def main(vocos_bin: str, out_onnx: str) -> None:
    sd = torch.load(vocos_bin, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    backbone = VocosBackbone(input_channels=100, dim=512, intermediate_dim=1536, num_layers=8)
    backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})

    class SpecHead(nn.Module):
        def __init__(self, dim: int = 512, n_fft: int = 1024):
            super().__init__()
            self.out = nn.Linear(dim, n_fft + 2)

        def forward(self, x):                 # x [B, L, dim]
            x = self.out(x).transpose(1, 2)   # [B, n_fft+2, L]
            mag, p = x.chunk(2, dim=1)        # [B, 513, L] each
            mag = torch.exp(mag).clamp(max=1e2)
            return mag * torch.cos(p), mag * torch.sin(p)   # real, imag

    head = SpecHead()
    head.out.weight.data = sd["head.out.weight"]
    head.out.bias.data = sd["head.out.bias"]

    class Voc(nn.Module):
        def __init__(self):
            super().__init__()
            self.b, self.h = backbone, head

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
    main(sys.argv[1], sys.argv[2])
