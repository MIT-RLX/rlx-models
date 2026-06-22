#!/usr/bin/env python3
"""Export Inflect-Nano-v1 to an RLX asset bundle.

Converts the released PyTorch checkpoints to safetensors (folding the vocoder's
weight-norm exactly like ``remove_weight_norm()``), and bundles every asset the
standalone Rust text frontend needs to match Python bit-for-bit:

  - acoustic.safetensors      MicroFastSpeech weights
  - vocoder.safetensors       Snake HiFi-GAN generator (weight-norm folded)
  - config.json               acoustic + vocoder configs, sample_rate, speaker map
  - frontend/cmudict_rep.txt  the repo CMUdict (checked before g2p_en)
  - frontend/g2p_checkpoint.safetensors  g2p_en OOV seq2seq weights (from npz)
  - frontend/homographs.en    g2p_en homograph table
  - frontend/g2p_cmudict.txt   nltk cmudict (g2p_en's internal dict)
  - frontend/perceptron_tagger.json  nltk averaged-perceptron POS tagger
  - frontend/bert/             bert-base-uncased tokenizer (vocab.txt + tokenizer.json)

Run with the project venv that has torch + g2p_en + nltk + transformers:
    .venv-inflect/bin/python scripts/export_inflect_nano.py \
        --repo /tmp/inflect-nano --out weights/inflect-nano-rlx
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file


def _t2np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float32).contiguous().numpy()


def export_acoustic(repo: Path, out: Path) -> dict:
    ck = torch.load(repo / "weights" / "inflect_nano_v1_acoustic.pt",
                    map_location="cpu", weights_only=True)
    sd = ck["model"]
    tensors = {k: _t2np(v) for k, v in sd.items()}
    save_file(tensors, str(out / "acoustic.safetensors"))
    print(f"acoustic: {len(tensors)} tensors, "
          f"{sum(v.size for v in tensors.values()):,} params")
    return dict(ck["config"]), dict(ck.get("speakers") or {"mark": 0})


def fold_weight_norm(sd: dict, g_key: str, v_key: str) -> np.ndarray:
    """Reproduce torch.nn.utils.remove_weight_norm with default dim=0."""
    g = sd[g_key].to(torch.float32)          # (out, 1, 1)
    v = sd[v_key].to(torch.float32)          # (out, in, k)
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
    w = v * (g / norm)
    return _t2np(w)


def export_vocoder(repo: Path, out: Path) -> dict:
    ck = torch.load(repo / "weights" / "inflect_nano_v1_vocoder.pt",
                    map_location="cpu", weights_only=True)
    g = ck["generator"]
    tensors: dict[str, np.ndarray] = {}
    folded = 0
    for key in list(g.keys()):
        if key.endswith(".weight_v"):
            base = key[: -len(".weight_v")]
            tensors[base + ".weight"] = fold_weight_norm(g, base + ".weight_g", base + ".weight_v")
            folded += 1
        elif key.endswith(".weight_g"):
            continue  # consumed with its matching weight_v
        else:
            tensors[key] = _t2np(g[key])  # bias, log_alpha
    save_file(tensors, str(out / "vocoder.safetensors"))
    print(f"vocoder: {len(tensors)} tensors ({folded} weight-norm folded), "
          f"{sum(v.size for v in tensors.values()):,} params")
    cfg = dict(ck.get("config") or {})
    cfg.setdefault("variant", "snake_v2mid")
    return cfg


# Fixed mel-frame length for the static CoreML model. CoreML's MIL requires
# bounded dims, so dynamic-axis ONNX silently falls back to CPU; the Rust CoreML
# path chunks the mel into this fixed length (with overlap). Keep in sync with
# `onnx_vocoder::COREML_STATIC_FRAMES`.
COREML_STATIC_FRAMES = 256


def export_vocoder_onnx(repo: Path, out: Path) -> None:
    """Export the (weight-norm-folded) vocoder to ONNX for the ONNX Runtime path.
    Emits both a dynamic-axis `vocoder.onnx` (CPU EP) and a static-shape
    `vocoder_static.onnx` (CoreML EP — bounded dims so it runs on CoreML)."""
    import sys

    sys.path.insert(0, str(repo))
    from train_hifigan_oracle_v1 import HifiGanGenerator, make_config

    ck = torch.load(repo / "weights" / "inflect_nano_v1_vocoder.pt", map_location="cpu", weights_only=True)
    cfg = make_config((ck.get("config") or {}).get("variant", "snake_v2mid"))
    gen = HifiGanGenerator(cfg)
    gen.load_state_dict(ck["generator"])
    gen.remove_weight_norm()
    gen.eval()
    with torch.no_grad():
        torch.onnx.export(
            gen, torch.zeros(1, cfg.num_mels, 64), str(out / "vocoder.onnx"),
            input_names=["mel"], output_names=["wav"],
            dynamic_axes={"mel": {2: "frames"}, "wav": {2: "samples"}},
            opset_version=17, dynamo=False,
        )
        torch.onnx.export(
            gen, torch.zeros(1, cfg.num_mels, COREML_STATIC_FRAMES), str(out / "vocoder_static.onnx"),
            input_names=["mel"], output_names=["wav"], opset_version=17, dynamo=False,
        )
    print(f"vocoder.onnx (dynamic) + vocoder_static.onnx (static {COREML_STATIC_FRAMES}) exported")


def export_g2p(out_fe: Path) -> None:
    import g2p_en
    gdir = Path(g2p_en.__file__).parent
    npz = np.load(gdir / "checkpoint20.npz")
    tensors = {k: np.ascontiguousarray(npz[k].astype(np.float32)) for k in npz.files}
    save_file(tensors, str(out_fe / "g2p_checkpoint.safetensors"))
    shutil.copyfile(gdir / "homographs.en", out_fe / "homographs.en")
    print(f"g2p_en: {len(tensors)} arrays, homographs.en copied")


def export_nltk_cmudict(out_fe: Path) -> None:
    from nltk.corpus import cmudict
    d = cmudict.dict()
    # one word per line: "word  P1 P2 P3" (multiple prons separated by " | ")
    with (out_fe / "g2p_cmudict.txt").open("w", encoding="utf-8") as f:
        for word in sorted(d.keys()):
            prons = [" ".join(p) for p in d[word]]
            f.write(f"{word}\t{' | '.join(prons)}\n")
    print(f"nltk cmudict: {len(d)} words")


def export_perceptron_tagger(out_fe: Path) -> None:
    """Copy the nltk averaged-perceptron tagger (the `_eng` JSON that nltk.pos_tag
    uses at runtime) into one combined JSON the Rust tagger loads."""
    import nltk
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    from nltk.data import find
    base = Path(str(find("taggers/averaged_perceptron_tagger_eng")))
    weights = json.loads((base / "averaged_perceptron_tagger_eng.weights.json").read_text())
    tagdict = json.loads((base / "averaged_perceptron_tagger_eng.tagdict.json").read_text())
    classes = json.loads((base / "averaged_perceptron_tagger_eng.classes.json").read_text())
    payload = {"weights": weights, "tagdict": tagdict, "classes": classes}
    (out_fe / "perceptron_tagger.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"perceptron tagger: {len(weights)} features, {len(tagdict)} tagdict, {len(classes)} classes")


def export_symbols(repo: Path, out_fe: Path) -> None:
    """Dump the exact symbol table + language/tone maps so the Rust frontend
    maps phonemes→ids identically (avoids re-deriving the sorted-set logic)."""
    import importlib
    sys.path.insert(0, str(repo))
    S = importlib.import_module("tiny_tts.text.symbols")

    payload = {
        "symbols": list(S.symbols),
        "num_tones": int(S.num_tones),
        "num_languages": int(S.num_languages),
        "language_id_map": dict(S.language_id_map),
        "language_tone_start_map": dict(S.language_tone_start_map),
        "punctuation": list(S.punctuation),
        "pu_symbols": list(S.pu_symbols),
    }
    (out_fe / "symbols.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"symbols: {len(S.symbols)} ids, {S.num_tones} tones, {S.num_languages} languages")


def export_bert(out_fe: Path) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bdir = out_fe / "bert"
    bdir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(bdir))
    print("bert-base-uncased tokenizer saved")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="cloned Inflect-Nano-v1 repo")
    ap.add_argument("--out", type=Path, required=True, help="output bundle dir")
    args = ap.parse_args()

    out: Path = args.out
    out_fe = out / "frontend"
    out.mkdir(parents=True, exist_ok=True)
    out_fe.mkdir(parents=True, exist_ok=True)

    acoustic_cfg, speakers = export_acoustic(args.repo, out)
    vocoder_cfg = export_vocoder(args.repo, out)
    export_vocoder_onnx(args.repo, out)

    config = {
        "model": "Inflect-Nano-v1",
        "sample_rate": 24000,
        "n_mels": 80,
        "add_blank": True,
        "language": "EN",
        "speakers": speakers,
        "acoustic": acoustic_cfg,
        "vocoder": vocoder_cfg,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    shutil.copyfile(args.repo / "tiny_tts" / "text" / "cmudict.rep", out_fe / "cmudict_rep.txt")
    print("repo cmudict.rep copied")
    export_g2p(out_fe)
    export_nltk_cmudict(out_fe)
    export_perceptron_tagger(out_fe)
    export_symbols(args.repo, out_fe)
    export_bert(out_fe)
    print(f"\nDone. Bundle written to {out}")


if __name__ == "__main__":
    main()
