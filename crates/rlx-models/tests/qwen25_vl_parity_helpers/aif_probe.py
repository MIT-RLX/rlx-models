#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# AIF (Liu et al., CVPR 2026) — token dynamics + adaptive mask (Eq. 2–5).
#
# Dynamics mode (RLX_AIF_DYNAMICS):
#   prefill_v2t  — Eq. 2 image-to-text: d_v^l = max_{j,h} a_{i,j,h}^l  (default)
#   decode_step  — Fig. 6 one-step decode: text query → visual keys (probe forward)

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

MASK_RATIO_CANDIDATES = [0.1 * i for i in range(1, 10)]


def dynamics_mode() -> str:
    return os.environ.get("RLX_AIF_DYNAMICS", "prefill_v2t")


def compute_dynamics_eq2_prefill(
    attentions,
    vision_start: int,
    n_vision: int,
    seq_len: int,
) -> np.ndarray | None:
    """Eq. 2 — image-to-text: visual query i, text keys j, d_v^l = max_{j,h} a_{i,j,h}^l."""
    if not attentions or n_vision <= 0:
        return None

    vision_range = set(range(vision_start, vision_start + n_vision))
    text_indices = [j for j in range(seq_len) if j not in vision_range]
    n_layers = len(attentions)
    dynamics = np.zeros((n_vision, n_layers), dtype=np.float64)
    for l, layer in enumerate(attentions):
        if layer is None:
            continue
        attn = layer[0].float().cpu()  # [heads, seq, seq]
        for vi in range(n_vision):
            qi = vision_start + vi
            valid_text = [j for j in text_indices if j <= qi]
            if not valid_text:
                dynamics[vi, l] = 0.0
            else:
                sl = attn[:, qi, valid_text]
                dynamics[vi, l] = float(sl.max())
    return dynamics


def compute_dynamics_decode_step(
    attentions,
    vision_start: int,
    n_vision: int,
) -> np.ndarray | None:
    """Fig. 6 — one-step decode probe: first generated token attends to visual keys."""
    if not attentions or n_vision <= 0:
        return None
    n_layers = len(attentions)
    dynamics = np.zeros((n_vision, n_layers), dtype=np.float64)
    for l, layer in enumerate(attentions):
        if layer is None:
            continue
        attn = layer[0].float().cpu()  # [heads, 1, past_len+1]
        for vi in range(n_vision):
            ki = vision_start + vi
            dynamics[vi, l] = float(attn[:, 0, ki].max())
    return dynamics


def compute_mu(dynamics: np.ndarray) -> np.ndarray:
    """Eq. 3."""
    return dynamics.mean(axis=1)


def compute_token_entropies(dynamics: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Eq. 4."""
    n, l = dynamics.shape
    ent = np.zeros(n, dtype=np.float64)
    for i in range(n):
        denom = l * mu[i]
        if denom <= 0.0:
            continue
        p = np.maximum(dynamics[i] / denom, 0.0)
        ent[i] = float(-(p * np.log(p + 1e-12)).sum())
    return ent


def distribution_entropy(mu: np.ndarray) -> float:
    """Eq. 5."""
    mu = np.maximum(mu.astype(np.float64), 0.0)
    total = mu.sum()
    if total <= 0.0:
        return 0.0
    p = mu / total
    return float(-(p * np.log(p + 1e-12)).sum())


def select_adaptive_mask_ratio(mu: np.ndarray, token_entropy: np.ndarray) -> float:
    """Sec. 4.3."""
    s0 = distribution_entropy(mu)
    n = mu.size
    if n == 0:
        return 0.5
    order = np.argsort(-token_entropy)
    best_ratio = 0.5
    best_dist = -1.0
    for ratio in MASK_RATIO_CANDIDATES:
        block_n = int(math.floor(n * ratio))
        if block_n <= 0 or block_n >= n:
            continue
        keep = np.ones(n, dtype=bool)
        keep[order[:block_n]] = False
        mu_keep = mu[keep]
        s = distribution_entropy(mu_keep)
        dist = abs(s - s0)
        if dist > best_dist:
            best_dist = dist
            best_ratio = ratio
    return float(best_ratio)


def blocked_keys_from_entropy(
    token_entropy: np.ndarray, vision_start: int, ratio: float
) -> list[int]:
    n = token_entropy.size
    if n == 0 or ratio <= 0.0:
        return []
    block_n = int(math.floor(n * min(max(ratio, 0.0), 1.0)))
    order = np.argsort(-token_entropy)[:block_n]
    return [vision_start + int(i) for i in order]


def build_aif_probe(dynamics: np.ndarray) -> dict[str, Any]:
    mu = compute_mu(dynamics)
    token_entropy = compute_token_entropies(dynamics, mu)
    s0 = distribution_entropy(mu)
    ratio = select_adaptive_mask_ratio(mu, token_entropy)
    return {
        "dynamics": dynamics,
        "mu": mu,
        "token_entropy": token_entropy,
        "s0": s0,
        "mask_ratio": ratio,
    }


def find_vision_span(ids_list: list[int], image_pad_id: int) -> tuple[int, int]:
    i = 0
    while i < len(ids_list):
        if ids_list[i] == image_pad_id:
            start = i
            while i < len(ids_list) and ids_list[i] == image_pad_id:
                i += 1
            return start, i - start
        i += 1
    return 0, 0


def resolve_model_dir() -> str:
    d = os.environ.get("RLX_QWEN25_VL_HF_DIR")
    if d:
        return d
    if os.environ.get("RLX_QWEN25_VL_DOWNLOAD") == "1":
        from huggingface_hub import snapshot_download

        return snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
    raise RuntimeError("set RLX_QWEN25_VL_HF_DIR or RLX_QWEN25_VL_DOWNLOAD=1")


def run_hf_prefill_probe(
    *,
    model_dir: str,
    image_path: str,
    question: str,
    device: str = "cpu",
    vlmevalkit: bool = False,
) -> dict[str, Any]:
    """Paper probe forward (Fig. 6b): prefill + optional one-step decode attentions."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.float32
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.bfloat16
    elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_dir)
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    ids_list = inputs["input_ids"][0].tolist()
    vision_start_idx, n_vision = find_vision_span(ids_list, image_pad_id)
    seq_len = int(inputs["input_ids"].shape[-1])
    mode = dynamics_mode()

    dynamics = None
    probe = None
    blocked = None

    with torch.no_grad():
        prefill = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )
        if mode == "decode_step" and n_vision > 0:
            next_id = prefill.logits[:, -1:].argmax(dim=-1)
            decode = model(
                input_ids=next_id,
                past_key_values=prefill.past_key_values,
                output_attentions=True,
                return_dict=True,
            )
            if decode.attentions is not None:
                dynamics = compute_dynamics_decode_step(
                    decode.attentions, vision_start_idx, n_vision
                )
        elif prefill.attentions is not None and n_vision > 0:
            dynamics = compute_dynamics_eq2_prefill(
                prefill.attentions, vision_start_idx, n_vision, seq_len
            )

    if dynamics is not None:
        probe = build_aif_probe(dynamics)
        blocked = blocked_keys_from_entropy(
            probe["token_entropy"], vision_start_idx, probe["mask_ratio"]
        )

    patch_size = processor.image_processor.patch_size
    grid = None
    resized_w = None
    resized_h = None
    if "image_grid_thw" in inputs:
        g = inputs["image_grid_thw"][0].tolist()
        grid = {"t": g[0], "h": g[1], "w": g[2]}
        resized_h = int(grid["h"] * patch_size)
        resized_w = int(grid["w"] * patch_size)

    meta: dict[str, Any] = {
        "model_dir": model_dir,
        "image": image_path,
        "question": question,
        "chat_text": chat_text,
        "seq_len": seq_len,
        "vision_start_idx": vision_start_idx,
        "n_vision_tokens": n_vision,
        "image_pad_id": int(image_pad_id),
        "vlmevalkit": vlmevalkit,
        "aif_dynamics_mode": mode,
    }
    if grid is not None:
        meta["image_grid_thw"] = grid
        meta["resized_w"] = resized_w
        meta["resized_h"] = resized_h
    if probe is not None:
        meta["aif_mask_ratio"] = float(probe["mask_ratio"])
        meta["aif_s0"] = float(probe["s0"])
        meta["aif_n_layers"] = int(dynamics.shape[1])
        meta["aif_blocked_keys"] = blocked

    return {
        "dynamics": dynamics,
        "probe": probe,
        "meta": meta,
        "input_ids": inputs["input_ids"][0].cpu().numpy(),
    }


def save_probe_sample(out_dir: os.PathLike | str, sample_id: str, result: dict[str, Any]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / sample_id
    meta = result["meta"]
    with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if result.get("dynamics") is not None:
        np.save(f"{prefix}_vision_dynamics.npy", result["dynamics"])
    probe = result.get("probe")
    if probe is not None:
        np.save(f"{prefix}_vision_mu_scores.npy", probe["mu"])
        np.save(f"{prefix}_vision_token_entropy.npy", probe["token_entropy"])
