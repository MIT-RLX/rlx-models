#!/usr/bin/env python3
"""Reference dump for `rlx-minicpm-sala` parity: OpenBMB's own
`modeling_minicpm_sala.py`, run on CPU.

That file imports `fla` (flash-linear-attention, Triton kernels) at module
scope. This script installs a pure-torch stand-in first, transcribed from
fla's `naive_recurrent_simple_gla`:

    S_t = exp(g) * S_{t-1} + k_t^T v_t ,   o_t = scale * q_t S_t

with `g` the per-head log decay (`g_gamma`). On CPU the remote code builds the
`minicpm4` layers as SDPA attention, which is exactly what its InfLLM-v2
kernel computes below `sparse_config.dense_len` tokens.

Modes:

  tiny OUTDIR
      Random tiny config (both mixer kinds, GQA, output gate, qk-norm, RoPE on
      the linear layers), saved as an HF directory, plus a reference dump.
  real MODEL_DIR OUT [--layers N] [--dtype float32|bfloat16] [--widen]
      The released checkpoint, optionally truncated to its first N layers so
      an f32 reference fits in memory (the full 9B model is ~36 GB in f32).
      --widen keeps the weights in --dtype (bfloat16: ~18 GB) and widens one
      top-level module at a time to f32 while it runs, so every product is
      f32 with the checkpoint's own values.

  sparse MODEL_DIR OUT [--layers N] [--records R] [--needle K]   (CUDA)
      InfLLM-v2's block-sparse attention, which the model switches to at
      `sparse_config.dense_len` (8192) keys: a prompt of R numbered records
      plus a question about record K, run through OpenBMB's `infllm_v2` CUDA
      kernels (scripts/infllm_reference_env.sh builds them) in bf16 — they have
      no other implementation. The `minicpm4` layers take the sparse class only
      with `flash_attention_2`; a `flash_attn` stand-in (SDPA) satisfies the
      import, and the dense functions it provides are not reached past
      `dense_len`. Also dumps every query's selected blocks in the first
      attention layer (<out>.safetensors, `topk.{layer}`).

Each writes <out>.json: prompt ids, last-position logits, greedy ids and the
reference's top-2 margins. Greedy decoding re-runs the whole sequence each
step (`use_cache=False`): the remote code's cache path needs its CUDA cache
class.
"""

import argparse
import json
import math
import shutil
import sys
import types
from pathlib import Path

import torch


def install_fla_stand_in() -> None:
    def simple_gla(q, k, v, g_gamma, scale=None, initial_state=None,
                   output_final_state=True, cu_seqlens=None, **_):
        # q, k, v: (b, t, h, d); g_gamma: (h,) log decay per step. With
        # `cu_seqlens` the batch is one packed row of several sequences, each
        # run on its own (fla's varlen convention).
        if cu_seqlens is not None:
            outs, finals = [], []
            for n in range(len(cu_seqlens) - 1):
                a, b = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
                init = None if initial_state is None else initial_state[n:n + 1]
                o, f = simple_gla(q[:, a:b], k[:, a:b], v[:, a:b], g_gamma, scale, init,
                                  output_final_state)
                outs.append(o)
                finals.append(f)
            final = torch.cat(finals, 0) if output_final_state else None
            return torch.cat(outs, 1), final
        dtype = q.dtype
        # float64 on CPU; f32 on a GPU, where memory is the limit (the GPU
        # reference itself runs in bf16).
        acc = torch.float64 if q.device.type == "cpu" else torch.float32
        q, k, v = (x.transpose(1, 2).to(acc) for x in (q, k, v))
        b, h, t, dk = q.shape
        if scale is None:
            scale = dk ** -0.5
        gate = g_gamma.to(acc).exp().view(1, h, 1, 1)
        s = q.new_zeros(b, h, dk, v.shape[-1])
        if initial_state is not None:
            s = s + initial_state.to(acc)
        out = torch.zeros_like(v)
        for i in range(t):
            s = s * gate + k[:, :, i, :, None] * v[:, :, i, None, :]
            out[:, :, i] = (q[:, :, i, :, None] * scale * s).sum(-2)
        final = s.float() if output_final_state else None
        return out.transpose(1, 2).to(dtype), final

    def tensor_cache(fn):
        return fn

    def prepare_lens_from_mask(mask):
        return mask.sum(dim=-1, dtype=torch.int32)

    def prepare_cu_seqlens_from_mask(mask, out_dtype=torch.int32):
        lens = prepare_lens_from_mask(mask)
        return torch.nn.functional.pad(lens.cumsum(0, dtype=out_dtype), (1, 0))

    mods = {
        "fla": types.ModuleType("fla"),
        "fla.ops": types.ModuleType("fla.ops"),
        "fla.ops.simple_gla": types.ModuleType("fla.ops.simple_gla"),
        "fla.ops.simple_gla.fused_recurrent": types.ModuleType("fla.ops.simple_gla.fused_recurrent"),
        "fla.ops.utils": types.ModuleType("fla.ops.utils"),
        "fla.ops.utils.index": types.ModuleType("fla.ops.utils.index"),
        "fla.utils": types.ModuleType("fla.utils"),
    }
    mods["fla.ops.simple_gla"].chunk_simple_gla = simple_gla
    mods["fla.ops.simple_gla.fused_recurrent"].fused_recurrent_simple_gla = simple_gla
    mods["fla.ops.utils.index"].prepare_cu_seqlens_from_mask = prepare_cu_seqlens_from_mask
    mods["fla.ops.utils.index"].prepare_lens_from_mask = prepare_lens_from_mask
    mods["fla.utils"].tensor_cache = tensor_cache
    sys.modules.update(mods)


install_fla_stand_in()

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

CODE_FILES = ("modeling_minicpm_sala.py", "configuration_minicpm_sala.py")


def greedy(model, ids, n_new):
    seq = list(ids)
    out, margins = [], []
    last_logits = None
    with torch.no_grad():
        for step in range(n_new):
            logits = model(torch.tensor([seq]), use_cache=False).logits[0, -1].float()
            if step == 0:
                last_logits = logits.clone()
            top2 = torch.topk(logits, 2)
            margins.append(float(top2.values[0] - top2.values[1]))
            tok = int(top2.indices[0])
            out.append(tok)
            seq.append(tok)
    return last_logits, out, margins


def install_flash_attn_stand_in() -> None:
    """`flash_attn` for the import in modeling_minicpm_sala.py: exact SDPA
    equivalents of the two dense functions (the sparse path never calls them
    past `dense_len`); the padding helpers are redefined by the modeling file."""
    import torch.nn.functional as F

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=softmax_scale)
        return o.transpose(1, 2)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                               max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                               causal=False, **_):
        outs = []
        for b in range(len(cu_seqlens_q) - 1):
            qs, qe = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
            ks, ke = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
            qq, kk, vv = (x.transpose(0, 1)[None] for x in (q[qs:qe], k[ks:ke], v[ks:ke]))
            rep = qq.shape[1] // kk.shape[1]
            kk, vv = kk.repeat_interleave(rep, 1), vv.repeat_interleave(rep, 1)
            lq, lk = qq.shape[2], kk.shape[2]
            mask = None
            if causal:  # bottom-right aligned, as flash-attn >= 2.1
                mask = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(lk - lq)
            o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=softmax_scale)
            outs.append(o[0].transpose(0, 1))
        return torch.cat(outs, 0)

    fa = types.ModuleType("flash_attn")
    fa.flash_attn_func = flash_attn_func
    fa.flash_attn_varlen_func = flash_attn_varlen_func
    fa.__version__ = "2.8.0"
    bp = types.ModuleType("flash_attn.bert_padding")
    bp.index_first_axis = bp.pad_input = bp.unpad_input = None
    fa.bert_padding = bp
    import importlib.machinery
    fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    bp.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", None)
    sys.modules.update({"flash_attn": fa, "flash_attn.bert_padding": bp})


def needle_prompt(records: int, needle: int) -> str:
    """R numbered records with pseudo-random codes, then a question."""
    lines = [f"Record {i}: the code for item {i} is {(i * 7919 + 104729) % 100000:05d}."
             for i in range(records)]
    return ("Here is a list of records.\n" + "\n".join(lines)
            + f"\n\nWhat is the code for item {needle}? Answer with the code only.")


def sparse(args) -> None:
    install_flash_attn_stand_in()
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    dev = torch.device("cuda")
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    if args.layers:
        config.num_hidden_layers = args.layers
        config.mixer_types = config.mixer_types[: args.layers]
    config._attn_implementation = "flash_attention_2"
    # transformers would resolve `flash_attention_2` to an installed
    # flash-attn (version check, lazy import); the sparse class calls
    # infllm_v2 itself, so keep the name and skip the resolution.
    import transformers.modeling_utils as mu
    mu.PreTrainedModel._check_and_adjust_attn_implementation = (
        lambda self, *a, **k: "flash_attention_2"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=config, trust_remote_code=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(dev).eval()
    m = sys.modules[type(model).__module__]
    assert hasattr(m, "infllmv2_attn_varlen_func"), "infllm_v2 did not import"

    # Every query's selected blocks, per attention layer.
    selected = {}
    original = m.compressed_attention

    def recording(*a, **kw):
        out = original(*a, **kw)
        selected.setdefault(len(selected), out.detach().to("cpu", torch.int32).clone())
        return out

    m.compressed_attention = recording

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    question = needle_prompt(args.records, args.needle)
    text = tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= config.sparse_config["dense_len"], f"prompt {len(ids)} below dense_len"
    seq = list(ids)
    out, margins, last = [], [], None
    with torch.no_grad():
        for step in range(args.n_new):
            x = torch.tensor([seq], device=dev)
            logits = model(x, attention_mask=torch.ones_like(x), use_cache=False,
                           logits_to_keep=1).logits[0, -1].float()
            if step == 0:
                last = logits.cpu().clone()
                first_selected = dict(selected)
            top2 = torch.topk(logits, 2)
            margins.append(float(top2.values[0] - top2.values[1]))
            t = int(top2.indices[0])
            out.append(t)
            seq.append(t)
    save_file({f"topk.{k}": v.contiguous() for k, v in first_selected.items()},
              args.out + ".safetensors")
    dump(args.out, ids, last, out, margins, {
        "prompt": text, "question": question, "records": args.records,
        "needle": args.needle, "layers": args.layers or config.num_hidden_layers,
        "dtype": "bfloat16", "device": torch.cuda.get_device_name(),
        "text": tok.decode(out, skip_special_tokens=True),
    })


def dump(out, ids, last_logits, new_ids, margins, extra):
    meta = {
        "prompt_ids": ids,
        "last_logits": last_logits.tolist(),
        "greedy": new_ids,
        "margins": margins,
        **extra,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out + ".json", "w") as f:
        json.dump(meta, f)
    print(json.dumps({k: meta[k] for k in ("prompt_ids", "greedy")}))


def tiny(args) -> None:
    src = Path(__file__).resolve().parent.parent / ".cache" / "sala"
    code_dir = Path(args.code_dir) if args.code_dir else src
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in CODE_FILES:
        shutil.copy(code_dir / f, outdir / f)
    cfg = json.loads((code_dir / "config.json").read_text())
    cfg.update(
        hidden_size=64, intermediate_size=96, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, lightning_head_dim=16,
        lightning_nh=4, lightning_nkv=4, num_hidden_layers=4,
        mixer_types=["minicpm4", "lightning-attn", "lightning-attn", "minicpm4"],
        vocab_size=97, max_position_embeddings=256, dim_model_base=16,
        torch_dtype="float32", dtype="float32",
    )
    (outdir / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(outdir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).float().eval()
    # The default init leaves norms at 1 and gates near 0.5; spread every
    # parameter so a wrong norm or gate cannot hide.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / math.sqrt(p.shape[-1]))
    model.save_pretrained(outdir, safe_serialization=True)
    ids = [int(x) for x in torch.randint(3, 97, (21,))]
    last, new_ids, margins = greedy(model, ids, args.n_new)
    with torch.no_grad():
        all_logits = model(torch.tensor([ids]), use_cache=False).logits[0].float()
    dump(str(outdir / "reference"), ids, last, new_ids, margins,
         {"all_logits": all_logits.flatten().tolist(), "vocab": 97})


def real(args) -> None:
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    if args.layers:
        config.num_hidden_layers = args.layers
        config.mixer_types = config.mixer_types[: args.layers]
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=config, trust_remote_code=True, dtype=dtype,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).eval()
    if args.widen:
        widen_per_module(model, dtype)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    msgs = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    last, new_ids, margins = greedy(model, ids, args.n_new)
    dump(args.out, ids, last, new_ids, margins, {
        "prompt": text, "question": args.prompt,
        "layers": args.layers or config.num_hidden_layers,
        "dtype": "float32" if args.widen else args.dtype,
        "text": tok.decode(new_ids, skip_special_tokens=True),
    })


def widen_per_module(model, storage) -> None:
    """Widen each layer (and the embedding, final norm, LM head) to f32 just
    for its own forward; activations stay f32 between them."""
    inner = model.model
    mods = [inner.embed_tokens, *inner.layers, inner.norm, model.lm_head]

    # Parameters only: the buffers (RoPE tables) are f32 already, and a
    # round trip through bf16 would round them.
    def cast(module, dtype):
        for p in module.parameters():
            p.data = p.data.to(dtype)

    def pre(module, args, kwargs):
        cast(module, torch.float32)
        args = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a
                     for a in args)
        kwargs = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                  for k, v in kwargs.items()}
        return args, kwargs

    def post(module, _args, _kwargs, out):
        cast(module, storage)
        return out

    for m in mods:
        m.register_forward_pre_hook(pre, with_kwargs=True)
        m.register_forward_hook(post, with_kwargs=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tiny")
    t.add_argument("outdir")
    t.add_argument("--code-dir", default=None)
    t.add_argument("--n-new", type=int, default=8)
    r = sub.add_parser("real")
    r.add_argument("model_dir")
    r.add_argument("out")
    r.add_argument("--layers", type=int, default=0)
    r.add_argument("--dtype", default="float32")
    r.add_argument("--widen", action="store_true")
    r.add_argument("--prompt", default="What is the capital of France? Answer in one word.")
    r.add_argument("--n-new", type=int, default=8)
    sp = sub.add_parser("sparse")
    sp.add_argument("model_dir")
    sp.add_argument("out")
    sp.add_argument("--layers", type=int, default=4)
    sp.add_argument("--records", type=int, default=600)
    sp.add_argument("--needle", type=int, default=417)
    sp.add_argument("--n-new", type=int, default=8)
    args = ap.parse_args()
    {"tiny": tiny, "real": real, "sparse": sparse}[args.mode](args)


if __name__ == "__main__":
    main()
