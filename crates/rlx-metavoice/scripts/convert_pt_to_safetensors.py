#!/usr/bin/env python3
"""Convert MetaVoice-1B HF `.pt` checkpoints → safetensors + tokenizer JSON.

  python3 crates/rlx-metavoice/scripts/convert_pt_to_safetensors.py \
      weights/tts/metavoice
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def untie(sd: dict) -> dict:
    seen: dict[int, str] = {}
    out = {}
    for k, v in sd.items():
        v = v.detach().contiguous().to(torch.float32)
        ptr = v.data_ptr()
        if ptr in seen:
            v = v.clone()
        seen[ptr] = k
        out[k] = v
    return out


def main(root: Path) -> None:
    obj = torch.load(root / "first_stage.pt", map_location="cpu", weights_only=False)
    sd = untie({k.replace("_orig_mod.", ""): v for k, v in obj["model"].items()})
    save_file(sd, root / "first_stage.safetensors")
    (root / "first_stage_args.json").write_text(
        json.dumps(
            {
                "model_args": obj["model_args"],
                "meta": {k: v for k, v in obj["meta"].items() if k != "tokenizer"},
            },
            indent=2,
        )
    )
    tok = obj["meta"]["tokenizer"]
    ranks = [
        [base64.b64encode(k).decode("ascii"), int(v)]
        for k, v in tok["mergeable_ranks"].items()
    ]
    (root / "tokenizer_metavoice.json").write_text(
        json.dumps(
            {
                "pat_str": tok["pat_str"],
                "mergeable_ranks_b64": ranks,
                "special_tokens": {str(k): int(v) for k, v in tok["special_tokens"].items()},
                "offset": int(tok["offset"]),
            }
        )
    )
    print(f"first_stage: {len(sd)} tensors")

    obj2 = torch.load(root / "second_stage.pt", map_location="cpu", weights_only=False)
    sd2 = untie(dict(obj2["model"]))
    save_file(sd2, root / "second_stage.safetensors")
    (root / "second_stage_args.json").write_text(
        json.dumps({"model_args": obj2["model_args"]}, indent=2)
    )
    print(f"second_stage: {len(sd2)} tensors")

    se = torch.load(root / "speaker_encoder.pt", map_location="cpu", weights_only=False)
    ms = se.get("model_state") or se.get("model") or se
    sd3 = untie({k: v for k, v in ms.items() if torch.is_tensor(v)})
    save_file(sd3, root / "speaker_encoder.safetensors")
    print(f"speaker_encoder: {len(sd3)} tensors")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "weights/tts/metavoice"))
