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

"""HF reference for LocateAnything parity (projector + MoonViT + LM prefill).

Avoids loading the full LocateAnythingForConditionalGeneration graph (peft / qwen3 deps).
Loads real safetensors weights from the checkpoint directory.

Protocol:

  META key=value ...
  VISION_IN / PROJECTOR / PATCHES / MOONVIT / PROJECTOR_FROM_VIT
  INPUTS_EMBEDS / LOGITS_LAST (lm_prefill probe)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

LM_PREFILL_SEQ = 8
LM_MTP_SEQ = 18
LM_MTP_PAST_LEN = LM_MTP_SEQ - 6  # block_size=6 in LocateAnything-3B
LM_DECODE_PAST_LEN = 12
LM_DECODE_TOKEN = 1000
LM_MTP_DECODE_PAST_LEN = 17
LM_MTP_DECODE_TOKEN = 1001
LM_MTP_DECODE_BLOCK = 6
LM_GREEDY_SEQ = 8
LM_GREEDY_NEW = 5
LM_FUSE_GREEDY_NEW = 3
E2E_GENERATE_NEW = 3
E2E_GENERATE_LONG = 8
PROMPT_N_IMAGE = 4
PROMPT_PHRASE = "Locate a single instance that matches the following description: red backpack."
REAL_PHRASE = "person"


def install_decord_stub() -> None:
    if "decord" not in sys.modules:
        sys.modules["decord"] = types.ModuleType("decord")


def install_hf_model_deps(model_dir: Path) -> None:
    """peft stub + `mask_magi_utils.py` in checkpoint dir for dynamic HF import."""
    import importlib.util
    import shutil

    try:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "to_legacy_cache"):

            def _to_legacy_cache(self):
                return tuple(
                    (self.layers[i].keys, self.layers[i].values)
                    for i in range(len(self.layers))
                )

            DynamicCache.to_legacy_cache = _to_legacy_cache
    except ImportError:
        pass

    spec = importlib.util.spec_from_loader("peft", loader=None)
    peft = importlib.util.module_from_spec(spec)
    sys.modules["peft"] = peft

    class LoraConfig:
        def __init__(self, **kw):
            pass

    def get_peft_model(module, _cfg):
        return module

    peft.LoraConfig = LoraConfig
    peft.get_peft_model = get_peft_model

    magi = types.ModuleType("la_ckpt.mask_magi_utils")
    magi.build_magi_ranges = lambda **kw: {}
    sys.modules["la_ckpt.mask_magi_utils"] = magi

    stub_src = Path(__file__).parent / "mask_magi_utils_stub.py"
    dst = model_dir / "mask_magi_utils.py"
    if not dst.is_file():
        shutil.copy(stub_src, dst)


def _patch_locateanything_pretrained(la_mod) -> None:
    orig = la_mod.LocateAnythingPreTrainedModel._check_and_adjust_attn_implementation

    def _patched(self, attn_implementation, is_init_check=False, **kw):
        kw.pop("allow_all_kernels", None)
        if attn_implementation == "magi":
            return "magi"
        return orig(self, attn_implementation, is_init_check)

    la_mod.LocateAnythingPreTrainedModel._check_and_adjust_attn_implementation = _patched
    la_mod.LocateAnythingPreTrainedModel.post_init = lambda self: None


def load_full_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shards = sorted({weight_map[k] for k in weight_map})
    out: dict[str, torch.Tensor] = {}
    for shard in shards:
        path = model_dir / shard
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
    return out


def build_locateanything_hf(model_dir: Path):
    install_hf_model_deps(model_dir)
    _load_ckpt_module(model_dir, "mask_sdpa_utils")
    mq = _load_ckpt_module(model_dir, "modeling_qwen2")
    _patch_qwen2_pretrained(mq)
    la = _load_ckpt_module(model_dir, "modeling_locateanything")
    _patch_locateanything_pretrained(la)
    cfg_mod = _load_ckpt_module(model_dir, "configuration_locateanything")
    q2_cfg_mod = _load_ckpt_module(model_dir, "configuration_qwen2")
    top = json.loads((model_dir / "config.json").read_text())
    tc = {k: v for k, v in top["text_config"].items() if not k.startswith("_")}
    tc.setdefault("pad_token_id", tc.get("eos_token_id", 0))
    text_cfg = q2_cfg_mod.Qwen2Config(**tc)
    text_cfg._attn_implementation = "sdpa"
    cfg = cfg_mod.LocateAnythingConfig.from_pretrained(str(model_dir))
    cfg.text_config = text_cfg
    cfg.vision_config._attn_implementation = "sdpa"
    model = la.LocateAnythingForConditionalGeneration(cfg)
    model.load_state_dict(load_full_state_dict(model_dir), strict=True)
    model.eval().float()
    return model


def hf_hybrid_generate_new_ids(
    model,
    model_dir: Path,
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
    *,
    generation_mode: str,
    max_new_tokens: int,
    n_future_tokens: int,
    generate_kwargs: dict,
) -> list[int]:
    """Run HF `LocateAnythingForConditionalGeneration.generate` loop; return new token ids."""
    sys.path.insert(0, str(model_dir))
    from generate_utils import handle_pattern, sample_tokens

    batch_size, seq_len = input_ids.shape
    generated = input_ids.clone()
    total_gen_length = seq_len + max_new_tokens
    past_key_values = None
    token_ids = model.token_ids
    use_mtp = generation_mode in ("fast", "hybrid")
    default_mask_token_id = token_ids["default_mask_token_id"]
    pre_mask_tokens = torch.full(
        (batch_size, n_future_tokens - 1),
        default_mask_token_id,
        dtype=generated.dtype,
        device=generated.device,
    )
    max_possible_len = total_gen_length + n_future_tokens
    full_position_ids = torch.arange(0, max_possible_len, device=generated.device).unsqueeze(0)

    while generated.size(1) < seq_len + max_new_tokens:
        if use_mtp:
            generated_with_mask = torch.cat(
                (generated, generated[:, -1].unsqueeze(1), pre_mask_tokens), dim=1
            )
            start_idx = past_key_values[0][0].size(2) if past_key_values is not None else 0
            position_ids = full_position_ids[:, start_idx : generated_with_mask.size(1)].clone()
            position_ids[0, -n_future_tokens:] -= 1
            prepare_inputs = model.language_model.prepare_inputs_for_generation(
                generated_with_mask,
                past_key_values,
                None,
                inputs_embeds=None,
                use_cache=True,
                position_ids=position_ids,
            )
        else:
            start_idx = past_key_values[0][0].size(2) if past_key_values is not None else 0
            position_ids = full_position_ids[:, start_idx : generated.size(1)]
            prepare_inputs = model.language_model.prepare_inputs_for_generation(
                generated,
                past_key_values,
                None,
                inputs_embeds=None,
                use_cache=True,
                position_ids=position_ids,
            )

        if past_key_values is None:
            prepare_inputs.update(
                {
                    "visual_features": vit_embeds,
                    "image_token_index": model.config.image_token_index,
                }
            )

        outputs = model.language_model(**prepare_inputs)
        pkv = outputs.past_key_values
        end = generated.shape[1]
        if hasattr(pkv, "layers"):
            past_key_values = tuple(
                (pkv.layers[i].keys[:, :, :end, :], pkv.layers[i].values[:, :, :end, :])
                for i in range(len(pkv.layers))
            )
        else:
            past_key_values = tuple(
                (pkv[i][0][:, :, :end, :], pkv[i][1][:, :, :end, :]) for i in range(len(pkv))
            )

        if use_mtp:
            next_token_logits = outputs.logits[:, -n_future_tokens:, :]
            _probs, _conf, x0, box_avg = sample_tokens(
                next_token_logits, generated, token_ids, keep_k=5, **generate_kwargs
            )
            is_box_empty = (box_avg[0] == 0).all()
            new_tokens = x0[0] if is_box_empty else box_avg[0]
            out_pattern = handle_pattern(new_tokens, token_ids, generation_mode)
            out_token = torch.tensor(
                out_pattern["tokens"], dtype=x0.dtype, device=x0.device
            )
            out_type = out_pattern["type"]
        else:
            next_token_logits = outputs.logits[:, -1:, :]
            _probs, _conf, x0, _ = sample_tokens(
                next_token_logits, generated, token_ids, **generate_kwargs
            )
            out_token = x0[0]
            out_type = "continue_ar"
            token_val = out_token[0].item()
            if generation_mode == "hybrid":
                if token_val == token_ids["box_end_token_id"]:
                    out_type = "box_end_ar"
                elif (
                    token_ids["coord_start_token_id"]
                    <= token_val
                    <= token_ids["coord_end_token_id"]
                    or token_val == token_ids["none_token_id"]
                ):
                    out_type = "coord_ar"
                else:
                    out_type = "im_end"
            elif token_val == token_ids["im_end_token_id"]:
                out_type = "im_end"

        generated = torch.cat([generated, out_token.unsqueeze(0)], dim=1)
        if out_type == "im_end":
            break
        if generation_mode == "hybrid":
            if out_type == "error_box":
                use_mtp = False
            elif out_type == "box_end_ar":
                use_mtp = True

    new_ids = generated[0, seq_len:].tolist()
    return new_ids


def emit_line(tag: str, values) -> None:
    flat = []
    for v in values:
        if isinstance(v, (float, np.floating)):
            flat.append(f"{float(v):.17g}")
        else:
            flat.append(str(int(v)))
    print(f"{tag} {len(flat)}", " ".join(flat))


def patches_rlx_flat(patches: torch.Tensor) -> list[float]:
    L, c, ph, pw = patches.shape
    out: list[float] = []
    for i in range(L):
        for ch in range(c):
            for dy in range(ph):
                for dx in range(pw):
                    out.append(float(patches[i, ch, dy, dx].item()))
    return out


def default_fixture_image() -> Path:
    # Bundled in rlx-locateanything (single canonical sample for tests + CLI).
    return (
        Path(__file__).resolve().parents[3]
        / "rlx-locateanything"
        / "fixtures"
        / "sample.jpg"
    )


def resolve_probe_image(image_arg: Path | None) -> Path:
    if image_arg is not None:
        return image_arg
    env = os.environ.get("RLX_LOCATEANYTHING_IMAGE")
    if env:
        return Path(env)
    return default_fixture_image()


def load_probe_image(image_arg: Path | None) -> Image.Image:
    path = resolve_probe_image(image_arg)
    if not path.is_file():
        raise FileNotFoundError(f"probe image not found: {path}")
    return Image.open(path).convert("RGB")


def synth_image(size: int = 56) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            arr[y, x, 0] = (x * 5) % 256
            arr[y, x, 1] = (y * 7) % 256
            arr[y, x, 2] = ((x + y) * 3) % 256
    return Image.fromarray(arr, mode="RGB")


def load_tensors_with_prefix(model_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard_names = sorted({weight_map[k] for k in weight_map if k.startswith(prefix)})
    out: dict[str, torch.Tensor] = {}
    for shard in shard_names:
        path = model_dir / shard
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    out[key[len(prefix) :]] = f.get_tensor(key)
    return out


def build_mlp1(model_dir: Path) -> nn.Sequential:
    cfg = json.loads((model_dir / "config.json").read_text())
    vit_h = cfg["vision_config"]["hidden_size"]
    llm_h = cfg["text_config"]["hidden_size"]
    merge = cfg["vision_config"]["merge_kernel_size"]
    in_dim = vit_h * merge[0] * merge[1]
    mlp1 = nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, llm_h),
        nn.GELU(),
        nn.Linear(llm_h, llm_h),
    )
    weights = load_tensors_with_prefix(model_dir, "mlp1.")
    mlp1.load_state_dict(weights, strict=True)
    mlp1.eval()
    return mlp1


def _ensure_ckpt_pkg(model_dir: Path) -> str:
    pkg = "la_ckpt"
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(model_dir)]
        sys.modules[pkg] = mod
    return pkg


def _load_ckpt_module(model_dir: Path, name: str):
    pkg = _ensure_ckpt_pkg(model_dir)
    full = f"{pkg}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    path = model_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch_qwen2_pretrained(qwen2_mod) -> None:
    orig = qwen2_mod.Qwen2PreTrainedModel._check_and_adjust_attn_implementation

    def _patched(self, attn_implementation, is_init_check=False, **_kw):
        return orig(self, attn_implementation, is_init_check)

    qwen2_mod.Qwen2PreTrainedModel._check_and_adjust_attn_implementation = _patched
    qwen2_mod.Qwen2PreTrainedModel.post_init = lambda self: None


def build_qwen2_lm(model_dir: Path):
    magi = types.ModuleType("la_ckpt.mask_magi_utils")
    magi.build_magi_ranges = lambda **kw: {}
    sys.modules["la_ckpt.mask_magi_utils"] = magi
    cfg_mod = _load_ckpt_module(model_dir, "configuration_qwen2")
    _load_ckpt_module(model_dir, "mask_sdpa_utils")
    mq = _load_ckpt_module(model_dir, "modeling_qwen2")
    _patch_qwen2_pretrained(mq)

    cfg = json.loads((model_dir / "config.json").read_text())["text_config"]
    tc = {k: v for k, v in cfg.items() if not k.startswith("_")}
    tc.setdefault("pad_token_id", tc.get("eos_token_id", 0))
    qcfg = cfg_mod.Qwen2Config(**tc)
    qcfg._attn_implementation = "sdpa"
    model = mq.Qwen2ForCausalLM(qcfg)
    weights = load_tensors_with_prefix(model_dir, "language_model.")
    model.load_state_dict(weights, strict=True)
    model.eval().float()
    return model, cfg


def build_mtp_prefill_mask_2d(
    input_ids: list[int],
    text_mask_token_id: int,
    block_size: int,
    use_cache: bool,
    causal_attn: bool,
    mask_utils,
) -> torch.Tensor:
    """Same steps as `rlx_locateanything::mask::mtp_prefill_mask_2d`."""
    seq = len(input_ids)
    m = torch.zeros(seq, seq)
    for q in range(seq):
        for k in range(seq):
            if k > q:
                m[q, k] = float("-inf")
    ids_t = torch.tensor(input_ids, dtype=torch.long)
    mask_utils.update_causal_mask_for_one_gen_window_2d(
        ids_t, m, block_size, use_cache, causal_attn
    )
    mask_utils.update_causal_mask_with_pad_non_visible_2d(
        ids_t, m, text_mask_token_id, block_size, causal_attn
    )
    return m


def expand_attn_bias_rlx(mask_2d: torch.Tensor, batch: int, num_heads: int) -> torch.Tensor:
    """RLX `MaskKind::Bias` layout `[batch, num_heads, seq, seq]`."""
    return mask_2d.unsqueeze(0).unsqueeze(0).expand(batch, num_heads, -1, -1).contiguous()


def expand_attn_bias_hf_sdpa(mask_2d: torch.Tensor, batch: int) -> torch.Tensor:
    """HF Qwen2 SDPA expects `[batch, 1, seq, seq]`."""
    return mask_2d.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1).contiguous()


def hf_lm_forward_last_logits(
    model,
    cfg: dict,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch, seq_length, _ = inputs_embeds.shape
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
    hidden_states = inputs_embeds
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
    hidden_states = model.model.norm(hidden_states)
    return model.lm_head(hidden_states)[:, -1, :].flatten()


def hf_lm_prefill_last_logits(model, cfg: dict, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Causal prefill on `inputs_embeds` (matches RLX `build_locateanything_prefill_built`)."""
    batch, seq_length, _ = inputs_embeds.shape
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
    attention_mask = _prepare_4d_causal_attention_mask(
        None,
        (batch, seq_length),
        inputs_embeds,
        0,
        sliding_window=cfg.get("sliding_window"),
    )
    return hf_lm_forward_last_logits(model, cfg, inputs_embeds, attention_mask)


def incremental_mtp_mask_qk(mask_2d: torch.Tensor, past_len: int, q_len: int) -> torch.Tensor:
    """Query rows `past_len..` × keys `0..past_len+q_len` from full `[seq, seq]` mask."""
    mask_qk = torch.zeros(q_len, past_len + q_len)
    for qi in range(q_len):
        full_q = past_len + qi
        for ki in range(past_len + q_len):
            mask_qk[qi, ki] = mask_2d[full_q, ki]
    return mask_qk


def hf_lm_mtp_kv_last_logits(
    model,
    cfg: dict,
    inputs_embeds: torch.Tensor,
    input_ids: list[int],
    text_mask: int,
    block_size: int,
    past_len: int,
    mask_utils,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal prefix cache + MTP query block (matches RLX `prefill_with_kv` + `mtp_logits`)."""
    batch, seq, hidden = inputs_embeds.shape
    q_len = seq - past_len
    assert q_len == block_size

    mask_2d = build_mtp_prefill_mask_2d(
        input_ids, text_mask, block_size, use_cache=True, causal_attn=False, mask_utils=mask_utils
    )
    mask_qk = incremental_mtp_mask_qk(mask_2d, past_len, q_len)
    nh = cfg["num_attention_heads"]
    attn_hf = mask_qk.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1).contiguous()

    prefix = inputs_embeds[:, :past_len, :]
    query = inputs_embeds[:, past_len:, :]

    past_kv = DynamicCache()
    hidden = prefix
    mask_prefix = _prepare_4d_causal_attention_mask(
        None,
        (batch, past_len),
        prefix,
        0,
        sliding_window=cfg.get("sliding_window"),
    )
    pos_prefix = torch.arange(past_len, dtype=torch.long).unsqueeze(0)
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden,
            attention_mask=mask_prefix,
            position_ids=pos_prefix,
            past_key_value=past_kv,
            use_cache=True,
        )
        hidden = layer_outputs[0]
        if len(layer_outputs) > 2 and layer_outputs[2] is not None:
            past_kv = layer_outputs[2]

    hidden = query
    pos_query = torch.arange(past_len, past_len + q_len, dtype=torch.long).unsqueeze(0)
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden,
            attention_mask=attn_hf,
            position_ids=pos_query,
            past_key_value=past_kv,
            use_cache=False,
        )
        hidden = layer_outputs[0]
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)[:, -1, :].flatten()
    attn_rlx = expand_attn_bias_rlx(mask_qk, batch, nh)
    return logits, prefix, query, attn_rlx


def hf_lm_mtp_kv_block_logits(
    model,
    cfg: dict,
    inputs_embeds: torch.Tensor,
    input_ids: list[int],
    text_mask: int,
    block_size: int,
    past_len: int,
    mask_utils,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like [`hf_lm_mtp_kv_last_logits`] but returns all `block_size` query logits."""
    batch, seq, hidden = inputs_embeds.shape
    q_len = seq - past_len
    assert q_len == block_size

    mask_2d = build_mtp_prefill_mask_2d(
        input_ids, text_mask, block_size, use_cache=True, causal_attn=False, mask_utils=mask_utils
    )
    mask_qk = incremental_mtp_mask_qk(mask_2d, past_len, q_len)
    nh = cfg["num_attention_heads"]
    attn_hf = mask_qk.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1).contiguous()

    prefix = inputs_embeds[:, :past_len, :]
    query = inputs_embeds[:, past_len:, :]

    past_kv = DynamicCache()
    hidden = prefix
    mask_prefix = _prepare_4d_causal_attention_mask(
        None,
        (batch, past_len),
        prefix,
        0,
        sliding_window=cfg.get("sliding_window"),
    )
    pos_prefix = torch.arange(past_len, dtype=torch.long).unsqueeze(0)
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden,
            attention_mask=mask_prefix,
            position_ids=pos_prefix,
            past_key_value=past_kv,
            use_cache=True,
        )
        hidden = layer_outputs[0]
        if len(layer_outputs) > 2 and layer_outputs[2] is not None:
            past_kv = layer_outputs[2]

    hidden = query
    pos_query = torch.arange(past_len, past_len + q_len, dtype=torch.long).unsqueeze(0)
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden,
            attention_mask=attn_hf,
            position_ids=pos_query,
            past_key_value=past_kv,
            use_cache=False,
        )
        hidden = layer_outputs[0]
    hidden = model.model.norm(hidden)
    logits_block = model.lm_head(hidden).reshape(-1)
    attn_rlx = expand_attn_bias_rlx(mask_qk, batch, nh)
    return logits_block, prefix, query, attn_rlx


def fuse_inputs_embeds_hf(
    model,
    token_ids: list[int],
    vision_embeds: list[float],
    image_token_id: int,
    hidden: int,
) -> torch.Tensor:
    """Match `rlx_locateanything::embed::fuse_inputs_embeds`."""
    embed_w = model.model.embed_tokens.weight
    vocab, h = embed_w.shape
    seq = len(token_ids)
    n_slots = sum(1 for t in token_ids if t == image_token_id)
    n_vecs = len(vision_embeds) // h
    assert n_slots == n_vecs, f"image slots {n_slots} != vision vecs {n_vecs}"
    out = torch.zeros(seq, h)
    img_idx = 0
    for pos, tok in enumerate(token_ids):
        if tok == image_token_id:
            src = torch.tensor(
                vision_embeds[img_idx * h : (img_idx + 1) * h], dtype=torch.float32
            )
            out[pos] = src
            img_idx += 1
        else:
            out[pos] = embed_w[tok]
    return out.unsqueeze(0)


def hf_lm_greedy_ar_tokens(
    model,
    cfg: dict,
    inputs_embeds: torch.Tensor,
    n_new: int,
) -> list[int]:
    """Greedy AR continuation after prefill (matches RLX `LmSessionCaches` slow path)."""
    batch, seq, _ = inputs_embeds.shape
    past_kv = DynamicCache()
    hidden = inputs_embeds
    mask_prefix = _prepare_4d_causal_attention_mask(
        None,
        (batch, seq),
        inputs_embeds,
        0,
        sliding_window=cfg.get("sliding_window"),
    )
    pos_prefix = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden,
            attention_mask=mask_prefix,
            position_ids=pos_prefix,
            past_key_value=past_kv,
            use_cache=True,
        )
        hidden = layer_outputs[0]
        if len(layer_outputs) > 2 and layer_outputs[2] is not None:
            past_kv = layer_outputs[2]

    hidden = model.model.norm(hidden)
    token = int(model.lm_head(hidden)[:, -1, :].argmax(dim=-1).item())
    out = [token]
    curr_past = seq
    for _ in range(n_new - 1):
        query = model.model.embed_tokens(torch.tensor([[token]], dtype=torch.long))
        pos_query = torch.tensor([[curr_past]], dtype=torch.long)
        mask_decode = _prepare_4d_causal_attention_mask(
            None,
            (batch, 1),
            query,
            curr_past,
            sliding_window=cfg.get("sliding_window"),
        )
        hidden = query
        for layer in model.model.layers:
            layer_outputs = layer(
                hidden,
                attention_mask=mask_decode,
                position_ids=pos_query,
                past_key_value=past_kv,
                use_cache=True,
            )
            hidden = layer_outputs[0]
            if len(layer_outputs) > 2 and layer_outputs[2] is not None:
                past_kv = layer_outputs[2]
        hidden = model.model.norm(hidden)
        token = int(model.lm_head(hidden)[:, -1, :].argmax(dim=-1).item())
        out.append(token)
        curr_past += 1
    return out


def probe_lm_greedy_ar(model_dir: Path) -> None:
    model, cfg = build_qwen2_lm(model_dir)
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    seq = LM_GREEDY_SEQ
    n_new = LM_GREEDY_NEW

    gen = torch.Generator().manual_seed(42)
    inputs = torch.randn(1, seq, hidden, generator=gen, dtype=torch.float32)

    with torch.no_grad():
        tokens = hf_lm_greedy_ar_tokens(model, cfg, inputs, n_new)

    print(f"META seq={seq} n_new={n_new} hidden={hidden} vocab={vocab}")
    emit_line("INPUTS_EMBEDS", inputs.flatten().tolist())
    emit_line("GENERATED_IDS", tokens)


def probe_lm_greedy_fused(model_dir: Path) -> None:
    model, cfg = build_qwen2_lm(model_dir)
    top = json.loads((model_dir / "config.json").read_text())
    image_token = int(top["image_token_index"])
    bos = int(top["text_config"]["bos_token_id"])
    hidden = cfg["hidden_size"]
    n_new = LM_FUSE_GREEDY_NEW

    img = synth_image(56)
    proc = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    batch = proc.preprocess([img], return_tensors="pt")
    vit = build_moonvit(model_dir)
    mlp1 = build_mlp1(model_dir)
    with torch.no_grad():
        vit_out = vit(batch["pixel_values"].float(), torch.tensor(batch["image_grid_hws"]))
        merged = torch.cat(list(vit_out), dim=0) if isinstance(vit_out, (list, tuple)) else vit_out
        vision = mlp1(merged).flatten().tolist()

    n_image = merged.shape[0]
    prompt_ids = [bos, 100, 200] + [image_token] * n_image + [300, 400]

    with torch.no_grad():
        fused = fuse_inputs_embeds_hf(model, prompt_ids, vision, image_token, hidden)
        tokens = hf_lm_greedy_ar_tokens(model, cfg, fused, n_new)

    print(
        f"META seq={len(prompt_ids)} n_new={n_new} n_image={n_image} "
        f"hidden={hidden} image_token={image_token}"
    )
    emit_line("INPUT_IDS", prompt_ids)
    emit_line("VISION_EMBEDS", vision)
    emit_line("FUSED_EMBEDS", fused.flatten().tolist())
    emit_line("GENERATED_IDS", tokens)


def hf_build_rlx_style_prompt_ids(
    tokenizer, image_token: int, user_text: str, n_image: int
) -> list[int]:
    """Match `rlx_locateanything::tokenizer::build_user_prompt_ids` layout."""
    ids: list[int] = []
    ids.extend(tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
    ids.extend([image_token] * n_image)
    ids.extend(tokenizer.encode(user_text, add_special_tokens=False))
    ids.extend(tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False))
    return ids


def probe_prompt_tokenizer(model_dir: Path) -> None:
    from transformers import AutoTokenizer

    top = json.loads((model_dir / "config.json").read_text())
    image_token = int(top["image_token_index"])
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    ids = hf_build_rlx_style_prompt_ids(tok, image_token, PROMPT_PHRASE, PROMPT_N_IMAGE)
    print(
        f"META n_image={PROMPT_N_IMAGE} image_token={image_token} "
        f"phrase_len={len(PROMPT_PHRASE)} seq={len(ids)}"
    )
    emit_line("PROMPT_IDS", ids)


def probe_task_ground_single(_model_dir: Path) -> None:
    phrase = "red backpack"
    user_text = (
        f"Locate a single instance that matches the following description: {phrase}."
    )
    print("META task=ground-single")
    print(f"PHRASE {phrase}")
    print(f"USER_TEXT {user_text}")


def probe_task_ground_multi(_model_dir: Path) -> None:
    phrase = "traffic light"
    user_text = (
        f"Locate all the instances that match the following description: {phrase}."
    )
    print("META task=ground-multi")
    print(f"PHRASE {phrase}")
    print(f"USER_TEXT {user_text}")


def probe_task_detect(_model_dir: Path) -> None:
    cats = "person</c>car"
    user_text = (
        f"Locate all the instances that matches the following description: {cats}."
    )
    print("META task=detect")
    print(f"CATEGORIES {cats}")
    print(f"USER_TEXT {user_text}")


def probe_processor_prompt(model_dir: Path) -> None:
    """HF LocateAnythingProcessor chat template + `<image-1>` expansion."""
    install_decord_stub()
    from transformers import AutoProcessor

    img = synth_image(56)
    phrase = "red backpack"
    user_text = (
        f"Locate a single instance that matches the following description: {phrase}."
    )
    user_with_ph = f"<image-1>{user_text}"
    proc = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_with_ph},
            ],
        }
    ]
    prompt_str = proc.py_apply_chat_template(messages, add_generation_prompt=True)
    batch = proc(text=prompt_str, images=[img], return_tensors="pt")
    ids = batch["input_ids"][0].tolist()
    top = json.loads((model_dir / "config.json").read_text())
    image_token = int(top["image_token_index"])
    n_image = ids.count(image_token)
    print(
        f"META n_image={n_image} image_token={image_token} seq={len(ids)} "
        f"task=ground-single layout=processor"
    )
    emit_line("PROMPT_IDS", ids)
    print(f"USER_TEXT {user_with_ph}")


def probe_e2e_processor_hybrid(model_dir: Path) -> None:
    """Hybrid generate with HF processor prompt layout."""
    install_decord_stub()
    from transformers import AutoProcessor

    top = json.loads((model_dir / "config.json").read_text())
    block_size = int(top["text_config"]["block_size"])
    n_new = E2E_GENERATE_NEW
    phrase = "red backpack"
    user_text = (
        f"Locate a single instance that matches the following description: {phrase}."
    )
    user_with_ph = f"<image-1>{user_text}"

    img = synth_image(56)
    proc = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_with_ph},
            ],
        }
    ]
    prompt_str = proc.py_apply_chat_template(messages, add_generation_prompt=True)
    batch = proc(text=prompt_str, images=[img], return_tensors="pt")
    pv = batch["pixel_values"].float()
    grid = torch.tensor(batch["image_grid_hws"], dtype=torch.int32)
    prompt_ids = batch["input_ids"][0].tolist()

    model = build_locateanything_hf(model_dir)
    with torch.no_grad():
        vit_list = model.extract_feature(pv, grid)
        vit_embeds = torch.cat(vit_list, dim=0)
        vit_embeds = model.mlp1(vit_embeds)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    gkw = {
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "generation_mode": "hybrid",
    }
    with torch.no_grad():
        new_ids = hf_hybrid_generate_new_ids(
            model,
            model_dir,
            input_ids,
            vit_embeds,
            generation_mode="hybrid",
            max_new_tokens=n_new,
            n_future_tokens=block_size,
            generate_kwargs=gkw,
        )

    image_token = int(top["image_token_index"])
    n_image = prompt_ids.count(image_token)
    print(
        f"META seq={len(prompt_ids)} n_new={n_new} n_image={n_image} "
        f"task=ground-single mode=hybrid layout=processor"
    )
    emit_line("INPUT_IDS", prompt_ids)
    emit_line("GENERATED_IDS", new_ids)


def probe_e2e_ground_single_hybrid(model_dir: Path) -> None:
    """Hybrid generate with RLX-style prompt + synth image (ground-single task)."""
    top = json.loads((model_dir / "config.json").read_text())
    image_token = int(top["image_token_index"])
    bos = int(top["text_config"]["bos_token_id"])
    block_size = int(top["text_config"]["block_size"])
    n_new = E2E_GENERATE_NEW
    phrase = "red backpack"
    user_text = (
        f"Locate a single instance that matches the following description: {phrase}."
    )

    img = synth_image(56)
    proc = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    batch = proc.preprocess([img], return_tensors="pt")
    pv = batch["pixel_values"].float()
    grid = torch.tensor(batch["image_grid_hws"], dtype=torch.int32)

    model = build_locateanything_hf(model_dir)
    with torch.no_grad():
        vit_list = model.extract_feature(pv, grid)
        vit_embeds = torch.cat(vit_list, dim=0)
        vit_embeds = model.mlp1(vit_embeds)

    n_image = vit_embeds.shape[0]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt_ids = hf_build_rlx_style_prompt_ids(tok, image_token, user_text, n_image)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    gkw = {
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "generation_mode": "hybrid",
    }
    with torch.no_grad():
        new_ids = hf_hybrid_generate_new_ids(
            model,
            model_dir,
            input_ids,
            vit_embeds,
            generation_mode="hybrid",
            max_new_tokens=n_new,
            n_future_tokens=block_size,
            generate_kwargs=gkw,
        )

    print(
        f"META seq={len(prompt_ids)} n_new={n_new} n_image={n_image} task=ground-single "
        f"mode=hybrid"
    )
    emit_line("INPUT_IDS", prompt_ids)
    emit_line("GENERATED_IDS", new_ids)


def probe_e2e_hybrid_long(model_dir: Path) -> None:
    top = json.loads((model_dir / "config.json").read_text())
    image_token = int(top["image_token_index"])
    bos = int(top["text_config"]["bos_token_id"])
    block_size = int(top["text_config"]["block_size"])
    n_new = E2E_GENERATE_LONG

    img = synth_image(56)
    proc = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    batch = proc.preprocess([img], return_tensors="pt")
    pv = batch["pixel_values"].float()
    grid = torch.tensor(batch["image_grid_hws"], dtype=torch.int32)

    model = build_locateanything_hf(model_dir)
    with torch.no_grad():
        vit_list = model.extract_feature(pv, grid)
        vit_embeds = torch.cat(vit_list, dim=0)
        vit_embeds = model.mlp1(vit_embeds)

    n_image = vit_embeds.shape[0]
    prompt_ids = [bos, 100, 200] + [image_token] * n_image + [300, 400]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    gkw = {
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "generation_mode": "hybrid",
    }
    with torch.no_grad():
        new_ids = hf_hybrid_generate_new_ids(
            model,
            model_dir,
            input_ids,
            vit_embeds,
            generation_mode="hybrid",
            max_new_tokens=n_new,
            n_future_tokens=block_size,
            generate_kwargs=gkw,
        )

    print(
        f"META seq={len(prompt_ids)} n_new={n_new} n_image={n_image} "
        f"image_token={image_token} mode=hybrid_long"
    )
    emit_line("INPUT_IDS", prompt_ids)
    emit_line("GENERATED_IDS", new_ids)


def probe_e2e_fast_generate(model_dir: Path) -> None:
    top = json.loads((model_dir / "config.json").read_text())
    image_token = int(top["image_token_index"])
    bos = int(top["text_config"]["bos_token_id"])
    block_size = int(top["text_config"]["block_size"])
    n_new = E2E_GENERATE_NEW

    img = synth_image(56)
    proc = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    batch = proc.preprocess([img], return_tensors="pt")
    pv = batch["pixel_values"].float()
    grid = torch.tensor(batch["image_grid_hws"], dtype=torch.int32)

    model = build_locateanything_hf(model_dir)
    with torch.no_grad():
        vit_list = model.extract_feature(pv, grid)
        vit_embeds = torch.cat(vit_list, dim=0)
        vit_embeds = model.mlp1(vit_embeds)

    n_image = vit_embeds.shape[0]
    prompt_ids = [bos, 100, 200] + [image_token] * n_image + [300, 400]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    gkw = {
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "generation_mode": "fast",
    }
    with torch.no_grad():
        new_ids = hf_hybrid_generate_new_ids(
            model,
            model_dir,
            input_ids,
            vit_embeds,
            generation_mode="fast",
            max_new_tokens=n_new,
            n_future_tokens=block_size,
            generate_kwargs=gkw,
        )

    print(
        f"META seq={len(prompt_ids)} n_new={n_new} n_image={n_image} "
        f"image_token={image_token} mode=fast"
    )
    emit_line("INPUT_IDS", prompt_ids)
    emit_line("GENERATED_IDS", new_ids)


def probe_e2e_hybrid_generate(model_dir: Path) -> None:
    top = json.loads((model_dir / "config.json").read_text())
    image_token = int(top["image_token_index"])
    bos = int(top["text_config"]["bos_token_id"])
    block_size = int(top["text_config"]["block_size"])
    n_new = E2E_GENERATE_NEW

    img = synth_image(56)
    proc = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    batch = proc.preprocess([img], return_tensors="pt")
    pv = batch["pixel_values"].float()
    grid = torch.tensor(batch["image_grid_hws"], dtype=torch.int32)

    model = build_locateanything_hf(model_dir)
    with torch.no_grad():
        vit_list = model.extract_feature(pv, grid)
        vit_embeds = torch.cat(vit_list, dim=0)
        vit_embeds = model.mlp1(vit_embeds)

    n_image = vit_embeds.shape[0]
    prompt_ids = [bos, 100, 200] + [image_token] * n_image + [300, 400]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    gkw = {
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "generation_mode": "hybrid",
    }
    with torch.no_grad():
        new_ids = hf_hybrid_generate_new_ids(
            model,
            model_dir,
            input_ids,
            vit_embeds,
            generation_mode="hybrid",
            max_new_tokens=n_new,
            n_future_tokens=block_size,
            generate_kwargs=gkw,
        )

    print(
        f"META seq={len(prompt_ids)} n_new={n_new} n_image={n_image} "
        f"image_token={image_token} mode=hybrid"
    )
    emit_line("INPUT_IDS", prompt_ids)
    emit_line("GENERATED_IDS", new_ids)


def hf_lm_decode_ar_last_logits(
    model,
    cfg: dict,
    prefix_embeds: torch.Tensor,
    token: int,
    past_len: int,
) -> torch.Tensor:
    """Causal prefix cache + single-token AR decode (matches RLX `decode_step` causal path)."""
    batch = prefix_embeds.shape[0]
    past_kv = DynamicCache()
    hidden = prefix_embeds
    mask_prefix = _prepare_4d_causal_attention_mask(
        None,
        (batch, past_len),
        prefix_embeds,
        0,
        sliding_window=cfg.get("sliding_window"),
    )
    pos_prefix = torch.arange(past_len, dtype=torch.long).unsqueeze(0)
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden,
            attention_mask=mask_prefix,
            position_ids=pos_prefix,
            past_key_value=past_kv,
            use_cache=True,
        )
        hidden = layer_outputs[0]
        if len(layer_outputs) > 2 and layer_outputs[2] is not None:
            past_kv = layer_outputs[2]

    query = model.model.embed_tokens(torch.tensor([[token]], dtype=torch.long))
    pos_query = torch.tensor([[past_len]], dtype=torch.long)
    mask_decode = _prepare_4d_causal_attention_mask(
        None,
        (batch, 1),
        query,
        past_len,
        sliding_window=cfg.get("sliding_window"),
    )
    hidden = query
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden,
            attention_mask=mask_decode,
            position_ids=pos_query,
            past_key_value=past_kv,
            use_cache=False,
        )
        hidden = layer_outputs[0]
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden).flatten()


def mtp_decode_additive_row(block_size: int, past_len: int) -> list[float]:
    """Additive mask row for one MTP decode step (matches `rlx_locateanything::mask::last_row_decode_mask`)."""
    total = past_len + 1
    row = [0.0] * total
    q = past_len
    for k in range(total):
        if k > q:
            row[k] = float("-inf")
    win_start = max(0, past_len - (block_size - 1))
    for k in range(win_start, past_len + 1):
        row[k] = 0.0
    return row


def hf_lm_decode_mtp_last_logits(
    model,
    cfg: dict,
    prefix_embeds: torch.Tensor,
    token: int,
    past_len: int,
    block_size: int,
) -> tuple[torch.Tensor, list[float]]:
    """Prefix cache + MTP-window decode mask (additive row → HF SDPA `[1,1,1,kv]`)."""
    batch = prefix_embeds.shape[0]
    past_kv = DynamicCache()
    hidden = prefix_embeds
    mask_prefix = _prepare_4d_causal_attention_mask(
        None,
        (batch, past_len),
        prefix_embeds,
        0,
        sliding_window=cfg.get("sliding_window"),
    )
    pos_prefix = torch.arange(past_len, dtype=torch.long).unsqueeze(0)
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden,
            attention_mask=mask_prefix,
            position_ids=pos_prefix,
            past_key_value=past_kv,
            use_cache=True,
        )
        hidden = layer_outputs[0]
        if len(layer_outputs) > 2 and layer_outputs[2] is not None:
            past_kv = layer_outputs[2]

    row = mtp_decode_additive_row(block_size, past_len)
    attn = torch.tensor(row, dtype=torch.float32).view(1, 1, 1, -1)
    query = model.model.embed_tokens(torch.tensor([[token]], dtype=torch.long))
    pos_query = torch.tensor([[past_len]], dtype=torch.long)
    hidden = query
    for layer in model.model.layers:
        layer_outputs = layer(
            hidden,
            attention_mask=attn,
            position_ids=pos_query,
            past_key_value=past_kv,
            use_cache=False,
        )
        hidden = layer_outputs[0]
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden).flatten()
    return logits, row


def probe_lm_decode_mtp(model_dir: Path) -> None:
    model, cfg = build_qwen2_lm(model_dir)
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    past_len = LM_MTP_DECODE_PAST_LEN
    token = LM_MTP_DECODE_TOKEN
    block_size = LM_MTP_DECODE_BLOCK

    gen = torch.Generator().manual_seed(42)
    prefix = torch.randn(1, past_len, hidden, generator=gen, dtype=torch.float32)

    with torch.no_grad():
        logits, mask_row = hf_lm_decode_mtp_last_logits(
            model, cfg, prefix, token, past_len, block_size
        )

    print(
        f"META past_len={past_len} token={token} block_size={block_size} hidden={hidden} vocab={vocab}"
    )
    emit_line("INPUTS_PREFIX", prefix.flatten().tolist())
    emit_line("TOKEN", [token])
    emit_line("MASK_ROW", mask_row)
    emit_line("LOGITS_LAST", logits.tolist())


def probe_lm_decode_ar(model_dir: Path) -> None:
    model, cfg = build_qwen2_lm(model_dir)
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    past_len = LM_DECODE_PAST_LEN
    token = LM_DECODE_TOKEN

    gen = torch.Generator().manual_seed(42)
    prefix = torch.randn(1, past_len, hidden, generator=gen, dtype=torch.float32)

    with torch.no_grad():
        logits = hf_lm_decode_ar_last_logits(model, cfg, prefix, token, past_len)

    print(
        f"META past_len={past_len} token={token} hidden={hidden} vocab={vocab} "
        f"rope_theta={cfg['rope_theta']}"
    )
    emit_line("INPUTS_PREFIX", prefix.flatten().tolist())
    emit_line("TOKEN", [token])
    emit_line("LOGITS_LAST", logits.tolist())


def probe_lm_mtp_decode(model_dir: Path) -> None:
    import torch.nn.functional as F

    sys.path.insert(0, str(model_dir))
    from generate_utils import decode_bbox_avg, get_token_ids_from_config, handle_pattern

    model, cfg = build_qwen2_lm(model_dir)
    mask_utils = _load_ckpt_module(model_dir, "mask_sdpa_utils")
    top = json.loads((model_dir / "config.json").read_text())
    text_mask = int(top["text_config"]["text_mask_token_id"])
    block_size = int(top["text_config"]["block_size"])
    vocab = cfg["vocab_size"]
    hidden = cfg["hidden_size"]
    seq = LM_MTP_SEQ
    past_len = LM_MTP_PAST_LEN

    prefix = [10 + i * 10 for i in range(past_len)]
    tail = [text_mask] * (block_size - 2) + [1000, 1001]
    input_ids = prefix + tail

    gen = torch.Generator().manual_seed(42)
    inputs = torch.randn(1, seq, hidden, generator=gen, dtype=torch.float32)

    token_ids = get_token_ids_from_config(top)
    token_ids["box_start_token_id"] = int(top["box_start_token_id"])
    token_ids["box_end_token_id"] = int(top["box_end_token_id"])
    token_ids["coord_start_token_id"] = int(top["coord_start_token_id"])
    token_ids["coord_end_token_id"] = int(top["coord_end_token_id"])
    token_ids["ref_start_token_id"] = int(top["ref_start_token_id"])
    token_ids["ref_end_token_id"] = int(top["ref_end_token_id"])
    token_ids["none_token_id"] = int(top["none_token_id"])

    with torch.no_grad():
        logits_block, prefix_emb, query_emb, attn_inc = hf_lm_mtp_kv_block_logits(
            model,
            cfg,
            inputs,
            input_ids,
            text_mask,
            block_size,
            past_len,
            mask_utils,
        )
        logits_2d = logits_block.reshape(block_size, vocab)
        probs = F.softmax(logits_2d, dim=-1)
        box = decode_bbox_avg(
            logits_2d, probs, token_ids, generation_mode="hybrid"
        )
        if box is None:
            box_list: list[int] = []
            pattern_tokens: list[int] = []
            pattern_kind = "none"
        else:
            box_list = box.tolist()
            pat = handle_pattern(box, token_ids, generation_mode="hybrid")
            pattern_tokens = pat["tokens"]
            pattern_kind = pat["type"]

    nh = cfg["num_attention_heads"]
    print(
        f"META seq={seq} past_len={past_len} block_size={block_size} vocab={vocab} "
        f"num_heads={nh} text_mask={text_mask} pattern_kind={pattern_kind}"
    )
    emit_line("INPUT_IDS", input_ids)
    emit_line("INPUTS_PREFIX", prefix_emb.flatten().tolist())
    emit_line("INPUTS_QUERY", query_emb.flatten().tolist())
    emit_line("ATTN_BIAS_INC", attn_inc.flatten().tolist())
    emit_line("LOGITS_BLOCK", logits_block.tolist())
    emit_line("BOX_TOKENS", box_list)
    emit_line("PATTERN_TOKENS", pattern_tokens)


def probe_lm_mtp_kv(model_dir: Path) -> None:
    model, cfg = build_qwen2_lm(model_dir)
    mask_utils = _load_ckpt_module(model_dir, "mask_sdpa_utils")
    top = json.loads((model_dir / "config.json").read_text())
    text_mask = int(top["text_config"]["text_mask_token_id"])
    block_size = int(top["text_config"]["block_size"])
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    nh = cfg["num_attention_heads"]
    seq = LM_MTP_SEQ
    past_len = LM_MTP_PAST_LEN
    assert past_len + block_size == seq

    prefix = [10 + i * 10 for i in range(past_len)]
    tail = [text_mask] * (block_size - 2) + [1000, 1001]
    input_ids = prefix + tail

    gen = torch.Generator().manual_seed(42)
    inputs = torch.randn(1, seq, hidden, generator=gen, dtype=torch.float32)

    with torch.no_grad():
        logits, prefix_emb, query_emb, attn_inc = hf_lm_mtp_kv_last_logits(
            model,
            cfg,
            inputs,
            input_ids,
            text_mask,
            block_size,
            past_len,
            mask_utils,
        )

    print(
        f"META seq={seq} past_len={past_len} q_len={block_size} hidden={hidden} vocab={vocab} "
        f"block_size={block_size} num_heads={nh} text_mask={text_mask}"
    )
    emit_line("INPUT_IDS", input_ids)
    emit_line("INPUTS_PREFIX", prefix_emb.flatten().tolist())
    emit_line("INPUTS_QUERY", query_emb.flatten().tolist())
    emit_line("ATTN_BIAS_INC", attn_inc.flatten().tolist())
    emit_line("LOGITS_LAST", logits.tolist())


def probe_lm_mtp_prefill(model_dir: Path) -> None:
    model, cfg = build_qwen2_lm(model_dir)
    mask_utils = _load_ckpt_module(model_dir, "mask_sdpa_utils")
    top = json.loads((model_dir / "config.json").read_text())
    text_mask = int(top["text_config"]["text_mask_token_id"])
    block_size = int(top["text_config"]["block_size"])
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    nh = cfg["num_attention_heads"]
    seq = LM_MTP_SEQ

    prefix = [10 + i * 10 for i in range(seq - block_size)]
    tail = [text_mask] * (block_size - 2) + [1000, 1001]
    input_ids = prefix + tail
    assert len(input_ids) == seq

    gen = torch.Generator().manual_seed(42)
    inputs = torch.randn(1, seq, hidden, generator=gen, dtype=torch.float32)
    mask_2d = build_mtp_prefill_mask_2d(
        input_ids, text_mask, block_size, use_cache=False, causal_attn=False, mask_utils=mask_utils
    )
    attn_rlx = expand_attn_bias_rlx(mask_2d, 1, nh)
    attn_hf = expand_attn_bias_hf_sdpa(mask_2d, 1)

    with torch.no_grad():
        logits = hf_lm_forward_last_logits(model, cfg, inputs, attn_hf)

    print(
        f"META seq={seq} hidden={hidden} vocab={vocab} block_size={block_size} "
        f"num_heads={nh} text_mask={text_mask}"
    )
    emit_line("INPUT_IDS", input_ids)
    emit_line("INPUTS_EMBEDS", inputs.flatten().tolist())
    emit_line("ATTN_BIAS", attn_rlx.flatten().tolist())
    emit_line("LOGITS_LAST", logits.tolist())


def probe_lm_prefill(model_dir: Path) -> None:
    model, cfg = build_qwen2_lm(model_dir)
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    gen = torch.Generator().manual_seed(42)
    inputs = torch.randn(
        1, LM_PREFILL_SEQ, hidden, generator=gen, dtype=torch.float32
    )
    with torch.no_grad():
        logits = hf_lm_prefill_last_logits(model, cfg, inputs)
    print(
        f"META seq={LM_PREFILL_SEQ} hidden={hidden} vocab={vocab} "
        f"rope_theta={cfg['rope_theta']}"
    )
    emit_line("INPUTS_EMBEDS", inputs.flatten().tolist())
    emit_line("LOGITS_LAST", logits.tolist())


def build_moonvit(model_dir: Path):
    sys.path.insert(0, str(model_dir))
    from configuration_locateanything import MoonViTConfig  # noqa: E402
    from modeling_vit import MoonVitPretrainedModel  # noqa: E402

    cfg = json.loads((model_dir / "config.json").read_text())
    vit_cfg = MoonViTConfig(**cfg["vision_config"])
    model = MoonVitPretrainedModel(vit_cfg)
    weights = load_tensors_with_prefix(model_dir, "vision_model.")
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def probe_projector(model_dir: Path, n_tokens: int) -> None:
    cfg = json.loads((model_dir / "config.json").read_text())
    vit_h = cfg["vision_config"]["hidden_size"]
    merge = cfg["vision_config"]["merge_kernel_size"]
    projector_in = vit_h * merge[0] * merge[1]
    hidden = cfg["text_config"]["hidden_size"]

    mlp1 = build_mlp1(model_dir)
    gen = torch.Generator().manual_seed(42)
    vision_in = torch.randn(n_tokens, projector_in, generator=gen, dtype=torch.float32)
    with torch.no_grad():
        out = mlp1(vision_in)

    print(f"META n_tokens={n_tokens} hidden={hidden} projector_in={projector_in}")
    emit_line("VISION_IN", vision_in.flatten().tolist())
    emit_line("PROJECTOR", out.flatten().tolist())


def probe_preprocess_real(model_dir: Path, image_arg: Path | None) -> None:
    img = load_probe_image(image_arg)
    proc = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    batch = proc.preprocess([img], return_tensors="pt")
    pv = batch["pixel_values"].float()
    gh, gw = int(batch["image_grid_hws"][0, 0]), int(batch["image_grid_hws"][0, 1])
    print(f"META grid_h={gh} grid_w={gw} image_w={img.size[0]} image_h={img.size[1]} layout=real")
    emit_line("PATCHES", patches_rlx_flat(pv))


def probe_moonvit(model_dir: Path) -> None:
    img = synth_image(56)
    proc = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    batch = proc.preprocess([img], return_tensors="pt")
    pv = batch["pixel_values"].float()
    grid = batch["image_grid_hws"]
    grid_t = torch.tensor(grid, dtype=torch.int32)

    gh, gw = int(grid_t[0, 0]), int(grid_t[0, 1])
    cfg = json.loads((model_dir / "config.json").read_text())
    merge = cfg["vision_config"]["merge_kernel_size"]
    kh, kw = int(merge[0]), int(merge[1])
    n_merged = (gh // kh) * (gw // kw)
    hidden = cfg["vision_config"]["hidden_size"]
    out_dim = hidden * kh * kw

    vit = build_moonvit(model_dir)
    mlp1 = build_mlp1(model_dir)

    with torch.no_grad():
        vit_out = vit(pv, grid_t)

    if isinstance(vit_out, (list, tuple)):
        merged = torch.cat([t for t in vit_out], dim=0)
    else:
        merged = vit_out

    with torch.no_grad():
        proj = mlp1(merged)

    print(
        f"META grid_h={gh} grid_w={gw} n_merged={n_merged} "
        f"hidden={hidden} merge_h={kh} merge_w={kw} out_dim={out_dim} "
        f"lm_hidden={cfg['text_config']['hidden_size']}"
    )
    emit_line("PATCHES", patches_rlx_flat(pv))
    emit_line("MOONVIT", merged.flatten().tolist())
    emit_line("PROJECTOR_FROM_VIT", proj.flatten().tolist())


def probe_moonvit_real(model_dir: Path, image_arg: Path | None) -> None:
    img = load_probe_image(image_arg)
    proc = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    batch = proc.preprocess([img], return_tensors="pt")
    pv = batch["pixel_values"].float()
    grid = batch["image_grid_hws"]
    grid_t = torch.tensor(grid, dtype=torch.int32)

    gh, gw = int(grid_t[0, 0]), int(grid_t[0, 1])
    cfg = json.loads((model_dir / "config.json").read_text())
    merge = cfg["vision_config"]["merge_kernel_size"]
    kh, kw = int(merge[0]), int(merge[1])
    n_merged = (gh // kh) * (gw // kw)
    hidden = cfg["vision_config"]["hidden_size"]
    out_dim = hidden * kh * kw

    vit = build_moonvit(model_dir)
    mlp1 = build_mlp1(model_dir)

    with torch.no_grad():
        vit_out = vit(pv, grid_t)

    if isinstance(vit_out, (list, tuple)):
        merged = torch.cat([t for t in vit_out], dim=0)
    else:
        merged = vit_out

    with torch.no_grad():
        proj = mlp1(merged)

    print(
        f"META grid_h={gh} grid_w={gw} n_merged={n_merged} "
        f"hidden={hidden} merge_h={kh} merge_w={kw} out_dim={out_dim} "
        f"lm_hidden={cfg['text_config']['hidden_size']} image_w={img.size[0]} image_h={img.size[1]} "
        f"layout=real"
    )
    emit_line("PATCHES", patches_rlx_flat(pv))
    emit_line("MOONVIT", merged.flatten().tolist())
    emit_line("PROJECTOR_FROM_VIT", proj.flatten().tolist())


def probe_e2e_processor_real(model_dir: Path, image_arg: Path | None) -> None:
    install_decord_stub()
    from transformers import AutoProcessor

    top = json.loads((model_dir / "config.json").read_text())
    block_size = int(top["text_config"]["block_size"])
    n_new = E2E_GENERATE_NEW
    phrase = REAL_PHRASE
    user_text = (
        f"Locate a single instance that matches the following description: {phrase}."
    )
    user_with_ph = f"<image-1>{user_text}"

    img = load_probe_image(image_arg)
    proc = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_with_ph},
            ],
        }
    ]
    prompt_str = proc.py_apply_chat_template(messages, add_generation_prompt=True)
    batch = proc(text=prompt_str, images=[img], return_tensors="pt")
    pv = batch["pixel_values"].float()
    grid = torch.tensor(batch["image_grid_hws"], dtype=torch.int32)
    prompt_ids = batch["input_ids"][0].tolist()

    model = build_locateanything_hf(model_dir)
    with torch.no_grad():
        vit_list = model.extract_feature(pv, grid)
        vit_embeds = torch.cat(vit_list, dim=0)
        vit_embeds = model.mlp1(vit_embeds)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    gkw = {
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "generation_mode": "hybrid",
    }
    with torch.no_grad():
        new_ids = hf_hybrid_generate_new_ids(
            model,
            model_dir,
            input_ids,
            vit_embeds,
            generation_mode="hybrid",
            max_new_tokens=n_new,
            n_future_tokens=block_size,
            generate_kwargs=gkw,
        )

    image_token = int(top["image_token_index"])
    n_image = prompt_ids.count(image_token)
    print(
        f"META seq={len(prompt_ids)} n_new={n_new} n_image={n_image} "
        f"task=ground-single mode=hybrid layout=processor-real "
        f"image_w={img.size[0]} image_h={img.size[1]}"
    )
    emit_line("INPUT_IDS", prompt_ids)
    emit_line("GENERATED_IDS", new_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument(
        "--probe",
        choices=(
            "projector",
            "moonvit",
            "lm_prefill",
            "lm_mtp_prefill",
            "lm_mtp_kv",
            "lm_decode_ar",
            "lm_decode_mtp",
            "lm_greedy_ar",
            "lm_greedy_fused",
            "lm_mtp_decode",
            "e2e_hybrid",
            "e2e_fast",
            "prompt_tokenizer",
            "processor_prompt",
            "task_ground_single",
            "task_ground_multi",
            "task_detect",
            "e2e_ground_single",
            "e2e_processor",
            "e2e_hybrid_long",
            "preprocess_real",
            "moonvit_real",
            "e2e_processor_real",
            "all",
        ),
        default="all",
    )
    ap.add_argument("--n-tokens", type=int, default=4)
    ap.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Real photo for *_real probes (default: rlx-locateanything/fixtures/sample.jpg or RLX_LOCATEANYTHING_IMAGE)",
    )
    args = ap.parse_args()

    if args.probe in ("projector", "all"):
        probe_projector(args.model_dir, args.n_tokens)
    if args.probe in ("moonvit", "all"):
        probe_moonvit(args.model_dir)
    if args.probe in ("lm_prefill", "all"):
        probe_lm_prefill(args.model_dir)
    if args.probe in ("lm_mtp_prefill", "all"):
        probe_lm_mtp_prefill(args.model_dir)
    if args.probe in ("lm_mtp_kv", "all"):
        probe_lm_mtp_kv(args.model_dir)
    if args.probe in ("lm_decode_ar", "all"):
        probe_lm_decode_ar(args.model_dir)
    if args.probe in ("lm_decode_mtp", "all"):
        probe_lm_decode_mtp(args.model_dir)
    if args.probe in ("lm_greedy_ar", "all"):
        probe_lm_greedy_ar(args.model_dir)
    if args.probe in ("lm_greedy_fused", "all"):
        probe_lm_greedy_fused(args.model_dir)
    if args.probe in ("lm_mtp_decode", "all"):
        probe_lm_mtp_decode(args.model_dir)
    if args.probe in ("e2e_hybrid", "all"):
        probe_e2e_hybrid_generate(args.model_dir)
    if args.probe in ("e2e_fast", "all"):
        probe_e2e_fast_generate(args.model_dir)
    if args.probe in ("prompt_tokenizer", "all"):
        probe_prompt_tokenizer(args.model_dir)
    if args.probe in ("processor_prompt", "all"):
        probe_processor_prompt(args.model_dir)
    if args.probe in ("task_ground_single", "all"):
        probe_task_ground_single(args.model_dir)
    if args.probe in ("task_ground_multi", "all"):
        probe_task_ground_multi(args.model_dir)
    if args.probe in ("task_detect", "all"):
        probe_task_detect(args.model_dir)
    if args.probe in ("e2e_ground_single", "all"):
        probe_e2e_ground_single_hybrid(args.model_dir)
    if args.probe in ("e2e_processor", "all"):
        probe_e2e_processor_hybrid(args.model_dir)
    if args.probe in ("e2e_hybrid_long", "all"):
        probe_e2e_hybrid_long(args.model_dir)
    if args.probe in ("preprocess_real", "all"):
        probe_preprocess_real(args.model_dir, args.image)
    if args.probe in ("moonvit_real", "all"):
        probe_moonvit_real(args.model_dir, args.image)
    if args.probe in ("e2e_processor_real", "all"):
        probe_e2e_processor_real(args.model_dir, args.image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
