#!/usr/bin/env python3
"""PyLaia reference dumps for rlx-pylaia parity.

A plain-torch transcription of PyLaia 1.0.7's `LaiaCRNN` (`laia/models/htr/
laia_crnn.py`, `conv_block.py`, `nn/image_pooling_sequencer.py`,
`nn/image_to_sequence.py`) built from stock `torch.nn` layers, so the
reference needs no PyLaia install (which pins torch 1.13 and a CUDA-only
helper). The checkpoint loads with `strict=True` into the same module names,
and is read through a restricted unpickler (no checkpoint code runs).

Per conv block: Conv2d (same padding, bias only without batchnorm) →
BatchNorm2d → activation → MaxPool2d(poolsize); then the `none-N` sequencer
(`permute(3, 0, 1, 2).view(W, N, C·H)`), a bidirectional RNN, and a Linear.
Prediction transform: grayscale → invert → ToTensor ([0, 1]), no resize.

Writes raw little-endian f32 `.bin` files + `meta.json` into --out:
  input     the network input [1, H, W]
  conv      CNN output [C, H', W']
  logits    linear output [W', num_labels] (pre-softmax)
and the greedy CTC transcription in `meta.json`.

  python3 scripts/pylaia_reference.py --model weights/ocr/pylaia-rimes \\
      --image line.jpg --out DIR
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htr_safe_unpickle as hsu  # noqa: E402

ACTS = {"LeakyReLU": torch.nn.LeakyReLU, "ReLU": torch.nn.ReLU, "Tanh": torch.nn.Tanh}
RNNS = {"LSTM": torch.nn.LSTM, "GRU": torch.nn.GRU, "RNN": torch.nn.RNN}


def pair(v):
    return tuple(v) if isinstance(v, (list, tuple)) else (v, v)


class ConvBlock(torch.nn.Module):
    def __init__(self, ni, nh, ks, st, di, act, ps, bn):
        super().__init__()
        ks, st, di, ps = pair(ks), pair(st), pair(di), pair(ps)
        pad = tuple((ks[d] - 1) // 2 * di[d] for d in (0, 1))
        self.conv = torch.nn.Conv2d(ni, nh, ks, stride=st, padding=pad, dilation=di, bias=not bn)
        self.batchnorm = torch.nn.BatchNorm2d(nh) if bn else None
        self.activation = act() if act else None
        self.pool = torch.nn.MaxPool2d(ps) if ps[0] * ps[1] >= 2 else None

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class LaiaCRNN(torch.nn.Module):
    def __init__(self, kw):
        super().__init__()
        seq_kind, fix = kw["image_sequencer"].split("-")
        if seq_kind != "none":
            raise SystemExit(f"sequencer {kw['image_sequencer']!r} not covered by this port")
        if kw.get("vertical_text"):
            raise SystemExit("vertical_text models are not covered by this port")
        self.fix = int(fix)
        blocks, ni = [], kw["num_input_channels"]
        for i, nh in enumerate(kw["cnn_num_features"]):
            blocks.append(
                ConvBlock(
                    ni,
                    nh,
                    kw["cnn_kernel_size"][i],
                    kw["cnn_stride"][i],
                    kw["cnn_dilation"][i],
                    kw["cnn_activation"][i],
                    kw["cnn_poolsize"][i],
                    kw["cnn_batchnorm"][i],
                )
            )
            ni = nh
        self.conv = torch.nn.Sequential(*blocks)
        self.rnn = kw["rnn_type"](
            ni * self.fix, kw["rnn_units"], kw["rnn_layers"], bidirectional=True
        )
        self.linear = torch.nn.Linear(2 * kw["rnn_units"], kw["num_output_labels"])

    def forward(self, x):
        x = self.conv(x)
        n, c, h, w = x.shape
        assert h == self.fix, f"CNN output height {h} != sequencer size {self.fix}"
        x = x.permute(3, 0, 1, 2).contiguous().view(w, n, h * c)
        x, _ = self.rnn(x)
        return self.linear(x)


def model_kwargs(model_dir):
    obj = hsu.load(os.path.join(model_dir, "model"))
    raw = obj["kwargs"]
    kw = hsu.unwrap(raw)
    kw["cnn_activation"] = [ACTS[c.__name__] for c in raw["cnn_activation"]]
    kw["rnn_type"] = RNNS[raw["rnn_type"].__name__]
    return kw


def save(out, name, t):
    a = t.detach().float().cpu().numpy().astype(np.float32)
    a.tofile(os.path.join(out, name + ".bin"))
    return list(a.shape)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="dir with model, weights.ckpt, syms.txt")
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    net = LaiaCRNN(model_kwargs(a.model)).eval()
    ckpt = hsu.load(os.path.join(a.model, "weights.ckpt"))
    net.load_state_dict(hsu.state_dict(ckpt, "state_dict.model."), strict=True)

    syms = {}
    for line in open(os.path.join(a.model, "syms.txt"), encoding="utf-8"):
        sym, idx = line.rstrip("\n").rsplit(" ", 1)
        syms[int(idx)] = sym

    img = ImageOps.invert(Image.open(a.image).convert("L"))
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None, None]

    conv = net.conv(x)
    logits = net(x)[:, 0, :]
    out, prev = [], None
    for i in logits.argmax(-1).tolist():
        if i != prev and i != 0:
            out.append(syms[i])
        prev = i
    text = "".join(" " if s == "<space>" else s for s in out)

    meta = {
        "image": a.image,
        "input": save(a.out, "input", x[0]),
        "conv": save(a.out, "conv", conv[0]),
        "logits": save(a.out, "logits", logits),
        "text": text,
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
