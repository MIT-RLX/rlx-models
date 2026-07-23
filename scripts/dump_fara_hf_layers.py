"""Dump HF Fara last-token hiddens for every language layer (parity vs RLX).

.venv-hf/bin/python scripts/dump_fara_hf_layers.py \\
  --model-dir .cache/fara/4b --out /tmp/fara_hf_layers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=".cache/fara/4b")
    ap.add_argument("--out", default="/tmp/fara_hf_layers")
    ap.add_argument("--prompt", default=None, help="override full ChatML; default matches rlx probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if args.prompt is None:
        # Match rlx-qwen35 examples/fara_text_probe.rs
        prompt = (
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
    else:
        prompt = args.prompt
    ids = tok.encode(prompt, add_special_tokens=False)
    print(f"[hf] {len(ids)} ids: {ids}")
    inp = torch.tensor([ids], dtype=torch.long)

    lm = model.model.language_model
    captures: dict[str, torch.Tensor] = {}

    def save_last(name: str, h: torch.Tensor) -> None:
        # h: [B, S, D] or [S, D]
        if h.dim() == 2:
            row = h[-1].detach().float().cpu().numpy()
        else:
            row = h[0, -1].detach().float().cpu().numpy()
        captures[name] = row

    hooks = []
    hooks.append(
        lm.embed_tokens.register_forward_hook(
            lambda _m, _i, o: save_last("embed", o)
        )
    )
    for i, layer in enumerate(lm.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda _m, _i, o, i=i: save_last(
                    f"layer_{i:02d}", o[0] if isinstance(o, tuple) else o
                )
            )
        )

    with torch.no_grad():
        out_lm = lm(input_ids=inp, use_cache=False)
        hidden = out_lm.last_hidden_state  # post final norm
        save_last("final_norm", hidden)
        # logits via tied embed
        w = lm.embed_tokens.weight
        logits = hidden[0, -1] @ w.T
        top = torch.topk(logits, 5)
        print("[hf] top5", top.indices.tolist(), top.values.tolist())
        print("[hf] toks", [tok.decode([int(i)]) for i in top.indices.tolist()])
        np.save(out / "logits.npy", logits.detach().float().cpu().numpy().reshape(1, -1))

    for name, row in sorted(captures.items()):
        np.save(out / f"{name}.npy", row.reshape(1, -1))
        print(
            f"[hf] {name}: dim={row.shape[0]} mean={row.mean():.6f} "
            f"absmax={np.abs(row).max():.4f} l2={np.linalg.norm(row):.4f}"
        )

    for h in hooks:
        h.remove()
    print(f"[hf] wrote {len(captures)+1} tensors under {out}")


if __name__ == "__main__":
    main()
