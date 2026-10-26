#!/usr/bin/env python3
"""Check a plain-torch transcription of InfLLM-v2's block selection against
OpenBMB's `infllm_v2` CUDA kernels on a real layer.

Runs MiniCPM-SALA's first `minicpm4` layer through the reference (see
`minicpm_sala_reference.py sparse`), captures the query, the compressed keys
and the kernels' chosen blocks, and recomputes the choice with the rule
`rlx-minicpm-sala` implements (src/sparse.rs): softmax over the visible
32/16 windows summed per KV group, max-pooled to 64-key blocks with the
initial and local blocks forced in, top-k, future blocks dropped. Done in f32
and in bf16 (the kernels' precision), so a disagreement can be told apart
from rounding at the top-k edge.

Usage (CUDA, in the env scripts/infllm_reference_env.sh builds):
  python -I scripts/infllm_selection_check.py MODEL_DIR [--records 560]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minicpm_sala_reference as ref  # noqa: E402  (installs the fla stand-in)


def transcription(q, ck, ck2, n_prompt, kernel=32, stride=16, block=64, init=1, local=32,
                  topk=96, dtype=torch.float32, round_sum=None, return_scores=False,
                  ties=None):
    """q: (N, nh, d); ck: (C, nkv, d) 32/16 windows; ck2: (C2, nkv, d) 128/64
    windows. Returns a bool (nkv, N, nb) choice.

    Stage 1 is the kernels' compressed-LSE softmax: the normalizer of each
    (head, query) is the log-sum-exp over the visible coarse windows, and a
    fine window scores exp(q·ck/√d − lse)."""
    n, nh, d = q.shape
    c, nkv, _ = ck.shape
    group = nh // nkv
    q = q.to(dtype).view(n, nkv, group, d).permute(1, 0, 2, 3)  # nkv, N, g, d
    k = ck.to(dtype).permute(1, 2, 0)  # nkv, d, C
    k2 = ck2.to(dtype).permute(1, 2, 0)  # nkv, d, C2
    c2 = ck2.shape[0]
    k_len = (n_prompt - kernel) // stride + 1
    shift = k_len - (n_prompt - 15) // 16
    i = torch.arange(n, device=q.device)
    seen = ((i - 15).div(16, rounding_mode="floor") + shift).clamp(0, k_len)
    s2 = 4 * stride
    k2_len = (n_prompt - 4 * kernel) // s2 + 1
    shift2 = k2_len - (n_prompt - (s2 - 1)) // s2
    seen2 = ((i - (s2 - 1)).div(s2, rounding_mode="floor") + shift2).clamp(0, k2_len)
    parts = []
    for a in range(0, n, 1024):  # queries in blocks: the full score tensor is GBs
        qa = q[:, a:a + 1024]
        t2 = torch.einsum("hngd,hdc->hngc", qa, k2).float() * d ** -0.5
        m2 = torch.arange(c2, device=q.device)[None, :] >= seen2[a:a + 1024, None]
        t2 = t2.masked_fill(m2[None, :, None, :], float("-inf"))
        lse = torch.logsumexp(t2, -1, keepdim=True)
        s = torch.einsum("hngd,hdc->hngc", qa, k).float() * d ** -0.5
        mask = torch.arange(c, device=q.device)[None, :] >= seen[a:a + 1024, None]
        s = s.masked_fill(mask[None, :, None, :], float("-inf"))
        pa = torch.nan_to_num(torch.exp(s - lse), nan=0.0, posinf=0.0)
        if round_sum is None:
            parts.append(pa.to(dtype).sum(2))
        elif round_sum == "trunc":  # bf16 by dropping the low 16 bits
            v = pa.sum(2).float().contiguous()
            parts.append((v.view(torch.int32) & -65536).view(torch.float32))
        else:  # the kernel: f32 scores, f32 head sum, rounded output
            parts.append(pa.sum(2).to(round_sum))
    p = torch.cat(parts, 1)  # nkv, N, C
    nb = (n_prompt + block - 1) // block
    per = block // stride
    out = torch.full((nkv, n, nb), float("-inf"), device=q.device, dtype=p.dtype)
    for kb in range(nb):
        lo, hi = max(kb * per - 1, 0), min(kb * per + per, c)
        if hi > lo:
            out[:, :, kb] = p[:, :, lo:hi].max(-1).values
    bq = i // block
    kb = torch.arange(nb, device=q.device)
    forced = (kb[None, :] < init) | ((bq[:, None] >= kb[None, :]) & (bq[:, None] <= kb[None, :] + local))
    out = out.masked_fill(forced[None], float("inf"))
    if ties is None:
        idx = out.topk(min(topk, nb), -1).indices
        choice = torch.zeros(nkv, n, nb, dtype=torch.bool, device=q.device)
        choice.scatter_(-1, idx, True)
    else:  # rank count, as the graph does: higher scores, then ties by index
        choice = torch.zeros(nkv, n, nb, dtype=torch.bool, device=q.device)
        order = kb[:, None] < kb[None, :] if ties == "lower" else kb[:, None] > kb[None, :]
        for a in range(0, n, 512):
            o = out[:, a:a + 512].float()
            gt = (o[..., None, :] > o[..., :, None]).sum(-1)
            eq = ((o[..., None, :] == o[..., :, None]) & order.T[None, None]).sum(-1)
            choice[:, a:a + 512] = (gt + eq) < topk
    choice &= (kb[None, None, :] <= bq[None, :, None])
    return (choice, out) if return_scores else choice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--records", type=int, default=560)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    ref.install_flash_attn_stand_in()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers.modeling_utils as mu
    mu.PreTrainedModel._check_and_adjust_attn_implementation = lambda self, *a, **k: "flash_attention_2"

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    config.num_hidden_layers = 1
    config.mixer_types = config.mixer_types[:1]
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=config, trust_remote_code=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True).cuda().eval()
    m = sys.modules[type(model).__module__]
    seen = {}
    original = m.compressed_attention

    def capture(q, k, k2, *a, **kw):
        out = original(q, k, k2, *a, **kw)
        seen.update(q=q.detach().clone(), ck=k.detach().clone(), ck2=k2.detach().clone(),
                    topk=out.detach().clone())
        return out

    m.compressed_attention = capture
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    text = tok.apply_chat_template(
        [{"role": "user", "content": ref.needle_prompt(args.records, 417)}],
        tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device="cuda")
    with torch.no_grad():
        model(ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    n = ids.shape[1]
    q, ck, topk = seen["q"], seen["ck"], seen["topk"].long()  # topk: nkv, N, k
    nb = (n + 63) // 64
    kernel = torch.zeros(topk.shape[0], n, nb + 1, dtype=torch.bool, device="cuda")
    kernel.scatter_(-1, torch.where(topk >= 0, topk, nb), True)
    kernel = kernel[..., :nb]
    print(f"prompt {n} tokens, {nb} blocks, compressed keys {tuple(ck.shape)}")
    variants = (("bf16", dict(dtype=torch.bfloat16)), ("f32", dict(dtype=torch.float32)),
                ("f32 sum -> bf16", dict(dtype=torch.float32, round_sum=torch.bfloat16)),
                ("f32 sum -> bf16, rank, lower index wins ties",
                 dict(dtype=torch.float32, round_sum=torch.bfloat16, ties="lower")),
                ("f32 sum -> bf16, rank, higher index wins ties",
                 dict(dtype=torch.float32, round_sum=torch.bfloat16, ties="higher")))
    for name, kw in variants:
        ours, scores = transcription(q, ck, seen["ck2"], n, return_scores=True, **kw)
        rows = (ours != kernel).any(-1).sum().item()
        swapped = (kernel & ~ours).sum().item()
        print(f"{name}: {rows} of {kernel.shape[0] * n} rows differ, "
              f"{swapped} of {kernel.sum().item()} chosen blocks swapped")
        if args.analyze:
            bq = (torch.arange(n, device="cuda") // 64)[None, :, None]
            kb = torch.arange(nb, device="cuda")[None, None, :]
            dist = (bq - kb).expand_as(kernel)
            only_k = kernel & ~ours
            only_o = ours & ~kernel
            print("  kernel-only blocks by (query block - block):",
                  torch.bincount(dist[only_k].clamp(0, 200)).nonzero().flatten()[:20].tolist())
            print("  ours-only   blocks by (query block - block):",
                  torch.bincount(dist[only_o].clamp(0, 200)).nonzero().flatten()[:20].tolist())
            diff_rows = (ours != kernel).any(-1)  # nkv, N
            qi = torch.arange(n, device="cuda")
            print("  differing rows by i % 16:", torch.bincount(qi[diff_rows.any(0)] % 16, minlength=16).tolist())
            print("  differing rows by i % 64:", torch.bincount((qi[diff_rows.any(0)] % 64) // 4, minlength=16).tolist())
            print("  kernel count per row (min/max):", kernel.sum(-1).min().item(), kernel.sum(-1).max().item(),
                  " ours:", ours.sum(-1).min().item(), ours.sum(-1).max().item())
            first = diff_rows.any(0).nonzero().flatten()[:5].tolist()
            for i0 in first:
                h = int(diff_rows[:, i0].nonzero()[0])
                ko = (only_k[h, i0]).nonzero().flatten().tolist()
                oo = (only_o[h, i0]).nonzero().flatten().tolist()
                print(f"  i={i0} h={h} kernel-only {ko} {scores[h, i0, ko].float().tolist()} "
                      f"ours-only {oo} {scores[h, i0, oo].float().tolist()}")
            ties = 0
            for h in range(kernel.shape[0]):
                for i0 in diff_rows[h].nonzero().flatten().tolist():
                    a = scores[h, i0][only_k[h, i0]].float()
                    b = scores[h, i0][only_o[h, i0]].float()
                    if a.numel() and b.numel() and torch.equal(a.sort().values, b.sort().values):
                        ties += 1
            print(f"  differing rows whose swapped blocks score exactly equal here: {ties}")


if __name__ == "__main__" and "--rlx-dump" not in sys.argv:
    main()


def compare_dump() -> None:
    """`--rlx-dump FILE`: the choice `rlx-minicpm-sala` made (written by its
    test with RLX_SALA_SPARSE_DUMP), against this transcription fed the f32
    query and keys of OpenBMB's code on CPU — the same rule on the same
    precision as the graph."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--rlx-dump", required=True)
    ap.add_argument("--records", type=int, default=560)
    args = ap.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    config.num_hidden_layers = 1
    config.mixer_types = config.mixer_types[:1]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=config, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="sdpa", low_cpu_mem_usage=True).eval()
    att = model.model.layers[0].self_attn
    got = {}
    att.q_proj.register_forward_hook(lambda _m, _i, o: got.__setitem__("q", o[0].detach()))
    att.k_proj.register_forward_hook(lambda _m, _i, o: got.__setitem__("k", o[0].detach()))
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    text = tok.apply_chat_template(
        [{"role": "user", "content": ref.needle_prompt(args.records, 417)}],
        tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        model.model.embed_tokens(ids)  # keep the import path warm
        h = model.model.embed_tokens(ids) * config.scale_emb
        x = model.model.layers[0].input_layernorm(h)
        att.q_proj(x), att.k_proj(x)
    n = ids.shape[1]
    nh, nkv, d = config.num_attention_heads, config.num_key_value_heads, 128
    q = got["q"].view(n, nh, d)
    k = got["k"].view(n, nkv, d)

    def windows(size, stride):
        count = (n - size) // stride + 1
        return torch.stack([k[j * stride:j * stride + size].mean(0) for j in range(count)])

    ck, ck2 = windows(32, 16), windows(128, 64)
    nb = (n + 63) // 64
    import numpy as np
    raw = np.fromfile(args.rlx_dump, dtype=np.uint8)
    width = raw.size // (nkv * n)
    ours_rlx = torch.from_numpy(raw.reshape(nkv, n, width)[..., :nb].astype(bool))
    for name, rnd in (("bf16 round-to-nearest", torch.bfloat16), ("bf16 truncation", "trunc"),
                      ("no rounding", None)):
        choice = transcription(q, ck, ck2, n, dtype=torch.float32, round_sum=rnd, ties="lower")
        rows = (choice != ours_rlx).any(-1).sum().item()
        swapped = (choice & ~ours_rlx).sum().item()
        print(f"rlx vs transcription (f32 inputs, {name}): {rows} of {nkv * n} rows differ, "
              f"{swapped} of {choice.sum().item()} chosen blocks swapped")


if __name__ == "__main__" and "--rlx-dump" in sys.argv:
    compare_dump()
    sys.exit(0)
