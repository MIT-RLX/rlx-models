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
# Per-stage SAM3 bench on host PyTorch — supports cpu / mps via
# BENCH_DEVICE=cpu|mps. Mirrors `bench.py` but runs on the host so
# we can hit Apple Silicon GPU (MPS).

import os, sys, time, types

# Stub triton BEFORE any torch imports (torch._inductor touches it at import).
if 'triton' not in sys.modules:
    tr = types.ModuleType('triton')
    tl = types.ModuleType('triton.language')
    backends = types.ModuleType('triton.backends')
    backends_compiler = types.ModuleType('triton.backends.compiler')
    compiler = types.ModuleType('triton.compiler')
    compiler_compiler = types.ModuleType('triton.compiler.compiler')
    runtime = types.ModuleType('triton.runtime')
    class dtype: pass
    class constexpr:
        def __init__(self, v=None): self.value = v
    class GPUTarget: pass
    class BaseBackend: pass
    class CompiledKernel: pass
    class ASTSource: pass
    tl.dtype = dtype
    tl.constexpr = constexpr
    backends_compiler.GPUTarget = GPUTarget
    backends_compiler.BaseBackend = BaseBackend
    compiler_compiler.CompiledKernel = CompiledKernel
    compiler_compiler.ASTSource = ASTSource
    compiler_compiler.compile = lambda *a, **k: None
    runtime.driver = type('d', (), {'active': None})
    tr.language = tl
    tr.backends = backends
    tr.compiler = compiler
    tr.runtime = runtime
    tr.jit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    tr.autotune = lambda *a, **k: lambda f: f
    tr.heuristics = lambda *a, **k: lambda f: f
    tr.Config = lambda *a, **k: None
    backends.compiler = backends_compiler
    compiler.compiler = compiler_compiler
    for n, m in [
        ('triton', tr), ('triton.language', tl),
        ('triton.backends', backends), ('triton.backends.compiler', backends_compiler),
        ('triton.compiler', compiler), ('triton.compiler.compiler', compiler_compiler),
        ('triton.runtime', runtime),
    ]:
        sys.modules[n] = m

import numpy as np
import torch


def main():
    n_warmup = int(os.environ.get("BENCH_WARMUP", "2"))
    n_iters = int(os.environ.get("BENCH_ITERS", "5"))
    device = os.environ.get("BENCH_DEVICE", "cpu")
    weights_path = os.environ["RLX_SAM3_WEIGHTS"]
    image_bin = os.environ["RLX_SAM3_IMAGE_BIN"]
    prompt_str = os.environ.get("RLX_SAM3_TEXT_PROMPT", "person")

    if device == "mps":
        assert torch.backends.mps.is_available(), "MPS not available"
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    # CPU-redirect patch only applies for `cpu`; MPS can construct cuda
    # placeholders directly only when explicitly redirected too. Patch
    # always since SAM3 hard-codes device="cuda" in some constructors.
    def _r(fn):
        def w(*a, **k):
            d = k.get("device")
            if isinstance(d, str) and d.startswith("cuda"):
                k["device"] = dev
            elif isinstance(d, torch.device) and d.type == "cuda":
                k["device"] = dev
            return fn(*a, **k)
        return w
    for n in ("zeros","ones","empty","full","arange","linspace","rand","randn","tensor","as_tensor"):
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

    print(f"# device={device} torch={torch.__version__}", file=sys.stderr)
    t0 = time.perf_counter()
    model = build_sam3_image_model(
        device="cpu", eval_mode=True, checkpoint_path=None,
        load_from_HF=False, enable_inst_interactivity=False, compile=False,
    )
    state = load_file(weights_path)
    state = {(k.replace("detector.", "") if "detector." in k else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.float()
    if device == "mps":
        model.to(dev)
    print(f"build+load+move: {time.perf_counter()-t0:.1f}s", file=sys.stderr)

    img = np.fromfile(image_bin, dtype=np.float32).reshape(1, 3, 1008, 1008)
    x = torch.from_numpy(img).float().to(dev)

    txt = model.backbone.language_backbone
    tokens = txt.tokenizer([prompt_str], context_length=txt.context_length).to(dev)

    timings = {k: [] for k in ("trunk","neck","text","encoder","decoder","seg","total")}

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    def step():
        t_total = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast(device_type="cpu", enabled=False):
            sync()
            t = time.perf_counter()
            trunk_out = model.backbone.vision_backbone.trunk(x)
            sync()
            timings["trunk"].append(time.perf_counter()-t)

            t = time.perf_counter()
            sam3_out, sam3_pos, _, _ = model.backbone.vision_backbone(x)
            sync()
            timings["neck"].append(time.perf_counter()-t)

            t = time.perf_counter()
            _, text_memory = txt.encoder(tokens)
            text_memory = text_memory.transpose(0, 1)
            text_memory_resized = txt.resizer(text_memory)
            prompt = text_memory_resized.float()
            prompt_mask = (tokens == 0).bool()
            sync()
            timings["text"].append(time.perf_counter()-t)

            src_level = sam3_out[-2]
            pos_level = sam3_pos[-2]
            seq_src = src_level.flatten(2).permute(2, 0, 1).contiguous()
            seq_pos = pos_level.flatten(2).permute(2, 0, 1).contiguous()

            t = time.perf_counter()
            enc_out = model.transformer.encoder(
                src=[seq_src], prompt=prompt, src_pos=[seq_pos],
                src_key_padding_mask=[None], prompt_key_padding_mask=prompt_mask,
                feat_sizes=[(72, 72)],
            )
            sync()
            timings["encoder"].append(time.perf_counter()-t)

            t = time.perf_counter()
            tgt = model.transformer.decoder.query_embed.weight.unsqueeze(1).contiguous()
            hs, ref_boxes, _, _ = model.transformer.decoder(
                tgt=tgt, memory=enc_out["memory"],
                memory_key_padding_mask=enc_out["padding_mask"], pos=enc_out["pos_embed"],
                reference_boxes=None, level_start_index=enc_out["level_start_index"],
                spatial_shapes=enc_out["spatial_shapes"], valid_ratios=enc_out["valid_ratios"],
                tgt_mask=None, memory_text=prompt, text_attention_mask=prompt_mask,
                apply_dac=False,
            )
            sync()
            timings["decoder"].append(time.perf_counter()-t)

            t = time.perf_counter()
            hs_bf = hs.transpose(1, 2).contiguous()
            _ = model.segmentation_head(
                backbone_feats=list(sam3_out[:-1]), obj_queries=hs_bf,
                image_ids=torch.zeros(1, dtype=torch.long, device=dev),
                encoder_hidden_states=enc_out["memory"],
                prompt=prompt, prompt_mask=prompt_mask,
            )
            sync()
            timings["seg"].append(time.perf_counter()-t)

        sync()
        timings["total"].append(time.perf_counter()-t_total)

    for _ in range(n_warmup):
        step()
    timings = {k: [] for k in timings}
    for _ in range(n_iters):
        step()

    def fmt(vs):
        ms = [v*1000 for v in vs]
        return f"avg={sum(ms)/len(ms):8.1f}ms  min={min(ms):8.1f}ms  max={max(ms):8.1f}ms"

    print(f"# pytorch host bench (device={device})")
    for k in ("trunk","neck","text","encoder","decoder","seg","total"):
        print(f"  pytorch/{device} {k:<8}: {fmt(timings[k])}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    # Stub triton even on host (sam3 imports it on init).
    import types
    class _Stub:
        def __getattr__(self, n): return _Stub()
        def __call__(self, *a, **k): return lambda f: f
    if 'triton' not in sys.modules:
        tr = types.ModuleType('triton')
        tl = types.ModuleType('triton.language')
        class dtype: pass
        tl.dtype = dtype
        tl.constexpr = type('c', (), {'__init__': lambda s,v=None: setattr(s,'value',v)})
        tr.language = tl
        tr.jit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
        sys.modules['triton'] = tr
        sys.modules['triton.language'] = tl
    main()
