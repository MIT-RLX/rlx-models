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

#
"""PyTorch SAM3 image-pipeline benchmark for head-to-head comparison.

Loads the same SAM3 checkpoint the Rust parity tests use and runs the
forward path the public processor does, measuring per-stage wall time.
"""

import os
import sys
import time

import numpy as np
import torch


def env(name):
    v = os.environ.get(name)
    if v is None:
        print(f"missing env var: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def main():
    n_warmup = int(os.environ.get("BENCH_WARMUP", "1"))
    n_iters = int(os.environ.get("BENCH_ITERS", "3"))
    threads = int(os.environ.get("BENCH_THREADS", "0"))
    if threads > 0:
        torch.set_num_threads(threads)

    # CPU redirect for SAM3's hard-coded device="cuda" sites.
    def _r(fn):
        def w(*a, **k):
            d = k.get("device")
            if isinstance(d, str) and d.startswith("cuda"):
                k["device"] = "cpu"
            elif isinstance(d, torch.device) and d.type == "cuda":
                k["device"] = torch.device("cpu")
            return fn(*a, **k)

        return w

    for n in ("zeros", "ones", "empty", "full", "arange", "linspace", "rand", "randn", "tensor", "as_tensor"):
        setattr(torch, n, _r(getattr(torch, n)))

    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

    import torch.nn.functional as F
    import sam3.perflib.fused as _fused

    def _addmm_act_f32(activation, linear, mat1):
        out = F.linear(mat1, linear.weight.detach(), linear.bias.detach())
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(out)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(out)
        raise ValueError(activation)

    _fused.addmm_act = _addmm_act_f32
    import sam3.model.vitdet as _v
    _v.addmm_act = _addmm_act_f32

    from sam3.model_builder import build_sam3_image_model
    from safetensors.torch import load_file

    weights_path = env("RLX_SAM3_WEIGHTS")
    image_bin = env("RLX_SAM3_IMAGE_BIN")
    prompt_str = os.environ.get("RLX_SAM3_TEXT_PROMPT", "person")

    print(f"threads={torch.get_num_threads()}", file=sys.stderr)

    t_build_0 = time.perf_counter()
    image_model = build_sam3_image_model(
        device="cpu",
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_inst_interactivity=False,
        compile=False,
    )
    state = load_file(weights_path)
    state = {(k.replace("detector.", "") if "detector." in k else k): v for k, v in state.items()}
    image_model.load_state_dict(state, strict=False)
    image_model.float()
    image_model.backbone.vision_backbone.trunk.float()
    image_model.transformer.float()
    image_model.segmentation_head.float()
    image_model.dot_prod_scoring.float()
    t_build = time.perf_counter() - t_build_0

    img = np.fromfile(image_bin, dtype=np.float32).reshape(1, 3, 1008, 1008)
    x = torch.from_numpy(img).float()

    # Pre-tokenize text once.
    txt_enc = image_model.backbone.language_backbone
    tokens = txt_enc.tokenizer([prompt_str], context_length=txt_enc.context_length)

    timings = {k: [] for k in ("trunk", "neck", "text", "encoder", "decoder", "seg", "scoring", "total")}

    def step():
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast(device_type="cpu", enabled=False):
            t = time.perf_counter()
            trunk_out = image_model.backbone.vision_backbone.trunk(x)
            timings["trunk"].append(time.perf_counter() - t)

            t = time.perf_counter()
            sam3_out, sam3_pos, _, _ = image_model.backbone.vision_backbone(x)
            timings["neck"].append(time.perf_counter() - t)

            t = time.perf_counter()
            _, text_memory = txt_enc.encoder(tokens)
            text_memory = text_memory.transpose(0, 1)
            text_memory_resized = txt_enc.resizer(text_memory)
            prompt = text_memory_resized.float()
            prompt_mask = (tokens == 0).bool()
            timings["text"].append(time.perf_counter() - t)

            src_level = sam3_out[-2]
            pos_level = sam3_pos[-2]
            seq_src = src_level.flatten(2).permute(2, 0, 1).contiguous()
            seq_pos = pos_level.flatten(2).permute(2, 0, 1).contiguous()

            t = time.perf_counter()
            enc_out = image_model.transformer.encoder(
                src=[seq_src],
                prompt=prompt,
                src_pos=[seq_pos],
                src_key_padding_mask=[None],
                prompt_key_padding_mask=prompt_mask,
                feat_sizes=[(72, 72)],
            )
            timings["encoder"].append(time.perf_counter() - t)

            t = time.perf_counter()
            tgt = image_model.transformer.decoder.query_embed.weight.unsqueeze(1).contiguous()
            hs, ref_boxes, _, _ = image_model.transformer.decoder(
                tgt=tgt,
                memory=enc_out["memory"],
                memory_key_padding_mask=enc_out["padding_mask"],
                pos=enc_out["pos_embed"],
                reference_boxes=None,
                level_start_index=enc_out["level_start_index"],
                spatial_shapes=enc_out["spatial_shapes"],
                valid_ratios=enc_out["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=False,
            )
            timings["decoder"].append(time.perf_counter() - t)

            t = time.perf_counter()
            hs_bf = hs.transpose(1, 2).contiguous()
            seg_out = image_model.segmentation_head(
                backbone_feats=list(sam3_out[:-1]),
                obj_queries=hs_bf,
                image_ids=torch.zeros(1, dtype=torch.long),
                encoder_hidden_states=enc_out["memory"],
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            timings["seg"].append(time.perf_counter() - t)

            t = time.perf_counter()
            _ = image_model.dot_prod_scoring(hs_bf, prompt, prompt_mask)
            timings["scoring"].append(time.perf_counter() - t)

        timings["total"].append(time.perf_counter() - t0)

    for _ in range(n_warmup):
        step()
    timings = {k: [] for k in timings}
    for _ in range(n_iters):
        step()

    def fmt(vs):
        ms = [v * 1000.0 for v in vs]
        return f"avg={sum(ms)/len(ms):8.1f}ms  min={min(ms):8.1f}ms  max={max(ms):8.1f}ms"

    print(f"# pytorch bench (build+load={t_build:.1f}s, threads={torch.get_num_threads()})")
    for k in ("trunk", "neck", "text", "encoder", "decoder", "seg", "scoring", "total"):
        print(f"  pytorch {k:<8}: {fmt(timings[k])}")


if __name__ == "__main__":
    main()
