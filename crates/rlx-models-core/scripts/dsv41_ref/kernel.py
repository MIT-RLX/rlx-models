"""Pure-torch CPU stand-in for the tilelang kernels in
deepseek-ai/DeepSeek-V4.1-Flash `inference/kernel.py`.

Semantics transliterated 1:1 from the tilelang prim_funcs so the released
`model.py` runs unmodified on CPU. Only the *numerics* are reproduced; the
tiling/pipelining is irrelevant here.
"""
import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP4_MAX = 6.0

# OCP E2M1 value table (convert.py FP4_TABLE order).
_E2M1 = torch.tensor([0.0, .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.])


def _round_scale(amax, max_inv):
    """fast_round_scale: 2^ceil(log2(amax * max_inv)) via IEEE bit tricks."""
    x = (amax * max_inv).float()
    bits = x.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & ((1 << 23) - 1)
    e = exp + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(x), e)


def _to_fp8(x):
    return x.to(torch.float8_e4m3fn).float()


def _to_fp4(x):
    """Round-to-nearest-even onto the E2M1 grid (what T.Cast(FP4, .) does)."""
    tbl = _E2M1[:8].to(x.device)                      # magnitudes, ascending
    a = x.abs().unsqueeze(-1)
    idx = (a - tbl).abs().argmin(dim=-1)
    # ties -> even index, matching RNE on the 1-bit mantissa grid
    return torch.sign(x) * tbl[idx]


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float().view(-1, N).unflatten(-1, (-1, block_size))
    amax = z.abs().amax(-1).clamp_min(1e-4)
    s = _round_scale(amax, 1.0 / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    q = torch.clamp(z / s.unsqueeze(-1), -FP8_MAX, FP8_MAX)
    if inplace:
        y = (_to_fp8(q) * s.unsqueeze(-1)).flatten(-2).view_as(x)
        x.copy_(y.to(x.dtype))
        return x
    return _to_fp8(q).flatten(-2).view_as(x), s.view(*x.shape[:-1], N // block_size)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float().view(-1, N).unflatten(-1, (-1, block_size))
    amax = z.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0 ** -9)
        s = _to_fp8(amax / FP4_MAX)
    else:
        amax = amax.clamp_min(6 * 2.0 ** -126)
        s = _round_scale(amax, 1.0 / FP4_MAX)
    q = torch.clamp(z / s.unsqueeze(-1), -FP4_MAX, FP4_MAX)
    if inplace:
        y = (_to_fp4(q) * s.unsqueeze(-1)).flatten(-2).view_as(x)
        x.copy_(y.to(x.dtype))
        return x
    return _to_fp4(q).flatten(-2).view_as(x), s.view(*x.shape[:-1], N // block_size)


def _deq_block2d(w, s, block_size):
    """w [N,K] fp8 codes (as float), s [ceil(N/b), K/b] -> dequantized [N,K]."""
    n, k = w.shape
    sr = s.float().repeat_interleave(block_size, 0)[:n]
    sr = sr.repeat_interleave(block_size, 1)[:, :k]
    return w.float() * sr


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    K = a.size(-1)
    am = a.reshape(-1, K).float()
    a_sr = a_s.reshape(am.size(0), -1).float().repeat_interleave(block_size, 1)[:, :K]
    bd = _deq_block2d(b.float(), b_s, block_size)
    c = (am * a_sr) @ bd.t()
    return c.view(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """`b` here is already-unpacked E2M1 magnitudes [N,K] (the CPU shim keeps fp4
    weights unpacked); `b_s` is [N, K//32]."""
    K = a.size(-1)
    am = a.reshape(-1, K).float()
    a_sr = a_s.reshape(am.size(0), -1).float().repeat_interleave(act_block_size, 1)[:, :K]
    bd = b.float() * b_s.float().repeat_interleave(32, 1)[:, :K]
    c = (am * a_sr) @ bd.t()
    return c.view(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b,m,h,d]; kv [b,n,d]; attn_sink [h]; topk_idxs [b,m,topk] int32 (-1 = empty)."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    gathered = kv.gather(1, idx.clamp_min(0).reshape(b, -1, 1).expand(-1, -1, d))
    gathered = gathered.view(b, m, -1, d) * valid.unsqueeze(-1)          # [b,m,topk,d]
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -float("inf"))
    # the sink participates in the denominator only
    mx = scores.amax(-1, keepdim=True).clamp_min(-1e30)
    e = (scores - mx).exp()
    den = e.sum(-1, keepdim=True) + (attn_sink.float().view(1, 1, h, 1) - mx).exp()
    o = torch.einsum("bmht,bmtd->bmhd", e / den, gathered.float())
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# ── parity switch ────────────────────────────────────────────────────────────
# RLX's graph builder omits the fp8/fp4 activation round-trips (they are
# precision simulation, not semantics). Setting RLX_REF_NOQUANT=1 turns the
# in-place quantizers into no-ops so the reference and the port are comparable
# to f32 tolerance.
import os as _os
if _os.environ.get("RLX_REF_NOQUANT") == "1":
    _aq, _fq = act_quant, fp4_act_quant

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):  # noqa: F811
        return x if inplace else _aq(x, block_size, scale_fmt, scale_dtype, inplace)

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):  # noqa: F811
        return x if inplace else _fq(x, block_size, inplace, scale_dtype)
