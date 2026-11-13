#!/usr/bin/env python3
"""DAN reference dumps for rlx-dan parity.

A plain-torch transcription of Coquenet's Document Attention Network
(github.com/FactoDeepLearning/DAN: `basic/models.py` FCN_Encoder,
`OCR/document_OCR/dan/models_dan.py` GlobalHTADecoder and the greedy loop
of `trainer_dan.py::evaluate_batch`), kept literal — including the
decoder's cache of per-layer *outputs* and its 100-token attention window —
so the port is checked against the original algorithm, not a restatement
of it. The checkpoint (`dan_rimes_page.pt`, Zenodo 7244382) loads with
`strict=True` through a restricted unpickler (no checkpoint code runs).

Input: the page as float RGB in [0, 255] (`predict_example.py`), optionally
normalized `(x − mean) / std` per channel (`--mean`, `--std`).

Writes raw little-endian f32 `.bin` files + `meta.json` into --out:
  input     the network input [3, H, W]
  features  encoder output [256, H/32, W/8]
  logits    decision layer for every generated step [T, vocab + 1]
and the greedy token ids + transcription (layout tokens kept) in meta.json.

  python3 scripts/dan_reference.py --model weights/ocr/dan-rimes/dan_rimes_page.pt \\
      --image page.png --out DIR
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htr_safe_unpickle as hsu  # noqa: E402

ENC_DIM, H_MAX, W_MAX, L_MAX = 256, 500, 1000, 15000
LAYERS, HEADS, FFN, WIN = 8, 4, 256, 100


class DepthSepConv2D(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.depth_conv = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin)
        self.point_conv = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.conv3 = nn.Conv2d(cout, cout, 3, padding=1, stride=stride)
        self.norm_layer = nn.InstanceNorm2d(cout, eps=0.001)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.relu(self.conv3(self.norm_layer(x)))


class DSCBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = DepthSepConv2D(cin, cout)
        self.conv2 = DepthSepConv2D(cout, cout)
        self.conv3 = DepthSepConv2D(cout, cout)
        self.norm_layer = nn.InstanceNorm2d(cout, eps=0.001)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.conv3(self.norm_layer(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [(3, 16, (1, 1)), (16, 32, (2, 2)), (32, 64, (2, 2)), (64, 128, (2, 2)),
                 (128, 128, (2, 1)), (128, 128, (2, 1))]
        self.init_blocks = nn.ModuleList([ConvBlock(a, b, s) for a, b, s in chans])
        self.blocks = nn.ModuleList([DSCBlock(128, 128), DSCBlock(128, 128), DSCBlock(128, 128),
                                     DSCBlock(128, 256)])

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        return x


class MHA(nn.Module):
    """`CustomMultiHeadAttention` (scaled q, additive bool masks)."""

    def __init__(self):
        super().__init__()
        self.in_proj_q = nn.Linear(ENC_DIM, ENC_DIM)
        self.in_proj_k = nn.Linear(ENC_DIM, ENC_DIM)
        self.in_proj_v = nn.Linear(ENC_DIM, ENC_DIM)
        self.out_proj = nn.Linear(ENC_DIM, ENC_DIM)
        self.hd = ENC_DIM // HEADS

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        tl, b, c = query.shape
        sl = key.shape[0]
        q = self.in_proj_q(query) * self.hd ** -0.5
        k = self.in_proj_k(key)
        v = self.in_proj_v(value)
        q = q.reshape(tl, b * HEADS, self.hd).transpose(0, 1)
        k = k.reshape(sl, b * HEADS, self.hd).transpose(0, 1)
        v = v.reshape(sl, b * HEADS, self.hd).transpose(0, 1)
        w = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            w.masked_fill_(attn_mask.unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            w = w.view(b, HEADS, tl, sl).masked_fill(key_padding_mask[:, None, None], float("-inf"))
            w = w.view(b * HEADS, tl, sl)
        w = torch.softmax(w, dim=-1)
        o = torch.bmm(w, v).transpose(0, 1).contiguous().view(tl, b, c)
        return self.out_proj(o)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_att, self.att = MHA(), MHA()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(ENC_DIM) for _ in range(3))
        self.linear1 = nn.Linear(ENC_DIM, FFN)
        self.linear2 = nn.Linear(FFN, ENC_DIM)

    def forward(self, tgt, memory, tgt_mask, mem_pad_mask, last_n):
        q = tgt[-last_n:]
        t = self.norm1(q + self.self_att(q, tgt, tgt, attn_mask=tgt_mask))
        t = self.norm2(t + self.att(t, memory, memory, key_padding_mask=mem_pad_mask))
        return self.norm3(t + self.linear2(torch.relu(self.linear1(t))))


class AttDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(LAYERS)])


class Decoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.att_decoder = AttDecoder()
        self.emb = nn.Embedding(vocab + 3, ENC_DIM)
        self.end_conv = nn.Conv1d(ENC_DIM, vocab + 1, 1)


def pe_1d():
    pe = torch.zeros((1, ENC_DIM, L_MAX))
    div = torch.exp(-torch.arange(0., ENC_DIM, 2) / ENC_DIM * torch.log(torch.tensor(10000.0))).unsqueeze(1)
    pos = torch.arange(0., L_MAX)
    pe[:, ::2, :] = torch.sin(pos * div).unsqueeze(0)
    pe[:, 1::2, :] = torch.cos(pos * div).unsqueeze(0)
    return pe


def pe_2d(h, w):
    d = ENC_DIM
    pe = torch.zeros((1, d, H_MAX, W_MAX))
    div = torch.exp(-torch.arange(0., d // 2, 2) / d * torch.log(torch.tensor(10000.0))).unsqueeze(1)
    wp, hp = torch.arange(0., W_MAX), torch.arange(0., H_MAX)
    pe[:, :d // 2:2] = torch.sin(hp * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, W_MAX)
    pe[:, 1:d // 2:2] = torch.cos(hp * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, W_MAX)
    pe[:, d // 2::2] = torch.sin(wp * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, H_MAX, 1)
    pe[:, d // 2 + 1::2] = torch.cos(wp * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, H_MAX, 1)
    return pe[:, :, :h, :w]


def save(out, name, t):
    a = t.detach().float().cpu().numpy().astype(np.float32)
    a.tofile(os.path.join(out, name + ".bin"))
    return list(a.shape)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mean", type=float, nargs=3)
    p.add_argument("--std", type=float, nargs=3)
    p.add_argument("--max-chars", type=int, default=3000)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ck = hsu.load(a.model)
    charset = list(ck["charset"])
    vocab = len(charset)
    enc, dec = Encoder().eval(), Decoder(vocab).eval()
    enc.load_state_dict(hsu.state_dict(ck, "encoder_state_dict."), strict=True)
    dec.load_state_dict(hsu.state_dict(ck, "decoder_state_dict."), strict=True)
    end, start = vocab, vocab + 1

    img = np.array(Image.open(a.image).convert("RGB")).astype(np.float32)
    x = torch.from_numpy(img).permute(2, 0, 1)[None]
    if a.mean:
        x = (x - torch.tensor(a.mean)[None, :, None, None]) / torch.tensor(a.std)[None, :, None, None]
    H, W = img.shape[:2]
    reduced = (H // 32, W // 8)

    feats = enc(x)
    _, c, fh, fw = feats.shape
    mem = (feats + pe_2d(fh, fw)).flatten(2).permute(2, 0, 1)  # [fh·fw, 1, C]
    pad = torch.ones((1, fh, fw), dtype=torch.bool)
    pad[0, :reduced[0], :reduced[1]] = False
    pad = pad.flatten(1)
    pe1 = pe_1d()

    tokens = [start]
    cache = None
    logits_all = []
    for _ in range(a.max_chars):
        L = len(tokens)
        t = dec.emb(torch.tensor([tokens])).permute(0, 2, 1) + pe1[:, :, :L]
        t = t.permute(2, 0, 1)  # [L, 1, C]
        if cache is not None:
            cache = cache[:, -WIN + 1:]
        keep = min(1 + WIN - 1, L)
        t = t[-keep:]
        full = torch.logical_not(torch.logical_and(
            torch.tril(torch.ones((L, L), dtype=torch.bool)),
            torch.triu(torch.ones((L, L), dtype=torch.bool), diagonal=-WIN + 1)))
        tgt_mask = full[-1:, -keep:]
        out, new = t, []
        for i, layer in enumerate(dec.att_decoder.decoder_layers):
            out = layer(out, mem, tgt_mask, pad, 1)
            new.append(out)
            if cache is not None:
                out = torch.cat([cache[i], out], dim=0)
        cache = torch.cat([cache, torch.stack(new)], dim=1) if cache is not None else torch.stack(new)
        out = out[-1:]
        pred = dec.end_conv(torch.relu(out).permute(1, 2, 0))[0, :, -1]
        logits_all.append(pred)
        nxt = int(pred.argmax())
        tokens.append(nxt)
        if nxt == end:
            break

    ids = tokens[1:]
    text = "".join(charset[i] for i in ids if i < vocab)
    meta = {
        "image": a.image,
        "mean": a.mean,
        "std": a.std,
        "input": save(a.out, "input", x[0]),
        "features": save(a.out, "features", feats[0]),
        "logits": save(a.out, "logits", torch.stack(logits_all)),
        "ids": ids,
        "text": text,
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"steps": len(ids), "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
