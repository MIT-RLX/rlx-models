#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Full forward logits reference for RLX LLaDA2 e2e parity.

Requires:
  - torch, transformers, safetensors
  - TIDE modeling code on PYTHONPATH (`TIDE_MODEL_CODE` or `/Users/Shared/TIDE/model`)
  - `LLADA2_MODEL_DIR` with config.json + HF safetensors shards

Usage:
  export LLADA2_MODEL_DIR=/path/to/checkpoint
  export TIDE_MODEL_CODE=/Users/Shared/TIDE/model
  export LLADA2_E2E_MAX_LAYERS=2
  python3 rlx-models/examples/llada2_full_parity_reference.py --seq-len 8 --block-length 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _tide_model_code_dir() -> str:
    return os.environ.get("TIDE_MODEL_CODE", "/Users/Shared/TIDE/model")


def _load_tide_modules():
    import importlib.util
    import types

    base = _tide_model_code_dir()
    pkg = types.ModuleType("tide_model")
    pkg.__path__ = [base]
    sys.modules["tide_model"] = pkg

    def load_module(qualname: str, filename: str):
        path = os.path.join(base, filename)
        spec = importlib.util.spec_from_file_location(qualname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "tide_model"
        sys.modules[qualname] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = load_module(
        "tide_model.configuration_llada2_moe", "configuration_llada2_moe.py"
    )
    model_mod = load_module("tide_model.modeling_llada2_moe", "modeling_llada2_moe.py")
    return cfg_mod, model_mod


def _load_sharded_state(model_dir: str, max_layers: int) -> dict:
    from safetensors.torch import load_file

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    state: dict = {}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted({weight_map[k] for k in weight_map})
        for shard in shard_files:
            path = os.path.join(model_dir, shard)
            for k, v in load_file(path, device="cpu").items():
                if _keep_key(k, max_layers):
                    state[k] = v
    else:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.isfile(single):
            raise FileNotFoundError(f"no safetensors under {model_dir}")
        for k, v in load_file(single, device="cpu").items():
            if _keep_key(k, max_layers):
                state[k] = v
    return state


def _keep_key(key: str, max_layers: int) -> bool:
    prefix = "model.layers."
    if not key.startswith(prefix):
        return True
    rest = key[len(prefix) :]
    layer_str = rest.split(".", 1)[0]
    if not layer_str.isdigit():
        return True
    return int(layer_str) < max_layers


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.environ.get("LLADA2_MODEL_DIR", ""))
    p.add_argument("--prompt-ids", default="1,2,3")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--block-length", type=int, default=4)
    p.add_argument("--max-layers", type=int, default=int(os.environ.get("LLADA2_E2E_MAX_LAYERS", "2")))
    args = p.parse_args()

    if not args.model_dir:
        print(json.dumps({"error": "LLADA2_MODEL_DIR not set"}), file=sys.stderr)
        return 2

    try:
        import torch
        import transformers.modeling_rope_utils as rope_utils

        def _llada2_default_rope(config, device=None):
            """Standard RoPE inv_freq when `config.rope_scaling` is null (TIDE default path)."""
            dim = int(
                getattr(config, "head_dim", 128)
                * float(getattr(config, "partial_rotary_factor", 1.0))
            )
            theta = float(getattr(config, "rope_theta", 10_000.0))
            inv_freq = 1.0 / (
                theta
                ** (
                    torch.arange(0, dim, 2, dtype=torch.float32, device=device).float()
                    / dim
                )
            )
            return inv_freq, 1.0

        rope_utils.ROPE_INIT_FUNCTIONS["default"] = _llada2_default_rope

        cfg_mod, model_mod = _load_tide_modules()
        LLaDA2MoeConfig = cfg_mod.LLaDA2MoeConfig
        LLaDA2MoeModelLM = model_mod.LLaDA2MoeModelLM
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2

    cfg_path = os.path.join(args.model_dir, "config.json")
    with open(cfg_path) as f:
        raw = json.load(f)
    raw.pop("dtype", None)
    cfg = LLaDA2MoeConfig(**raw)
    max_layers = min(args.max_layers, cfg.num_hidden_layers)
    cfg.num_hidden_layers = max_layers

    device = "cpu"
    model = LLaDA2MoeModelLM(cfg).to(device).eval()
    state = _load_sharded_state(args.model_dir, max_layers)
    model.load_state_dict(state, strict=False)

    prompt = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
    seq_len = args.seq_len
    block_length = args.block_length
    mask_id = getattr(cfg, "mask_token_id", 156895)

    ids = [mask_id] * seq_len
    for i, t in enumerate(prompt):
        if i < seq_len:
            ids[i] = t

    num_blocks = (seq_len + block_length - 1) // block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
        .log()[:, :, :seq_len, :seq_len]
    )
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
        )
        logits = out.logits[0].float().cpu().tolist()

    emit = {
        "test": "forward_logits",
        "seq_len": seq_len,
        "vocab_size": cfg.vocab_size,
        "logits": logits,
    }
    print(json.dumps(emit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
