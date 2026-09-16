"""Run the toy DeepSeek-V4.1 reference and dump inputs/outputs for the rlx port."""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RLX_REF_NOQUANT", "1")
import toy, prng
import model as M

# the port emits logits for every position; the reference head defaults to the
# last one only, so widen it for the dump
_orig_head_forward = M.ParallelHead.forward
M.ParallelHead.forward = lambda self, x, full_logits=True: _orig_head_forward(self, x, True)

SEQ = 12


def small_args(**kw):
    a = toy.toy_args(**kw)
    a.vocab_size = 64
    a.dim = 32
    a.moe_inter_dim = 16
    a.n_layers = 6
    a.n_heads = 2
    a.head_dim = 32
    a.rope_head_dim = 16
    a.q_lora_rank = 16
    a.o_groups = 2
    a.o_lora_rank = 8
    a.window_size = 4
    a.n_routed_experts = 4
    a.n_activated_experts = 2
    a.n_shared_experts = 1
    a.index_n_heads = 2
    a.index_head_dim = 32
    a.index_topk = 3
    a.candidate_topk_blocks = 2
    a.candidate_block_size = 2
    a.candidate_source_layer = 4
    a.max_seq_len = 32
    a.original_seq_len = 16
    a.rope_factor = 4
    if a.engram_layer_ids:
        a.engram_max_ngram_size = 3
        a.engram_n_heads = 2
        a.engram_head_dim = 32
        a.engram_vocab_size = 11
        a.engram_compressed_vocab_size = toy.COMPRESSED_VOCAB
        import engram as engram_mod
        layout = engram_mod.EngramLayout.from_args(a)
        a.engram_num_embeddings = tuple(
            int(sum(p for per in layer for p in per)) for layer in layout.primes
        )
    return a


def build(**kw):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    args = small_args(**kw)
    m = M.Transformer(args, tokenizer=object())
    shapes = {}
    for name, p in m.named_parameters():
        shape = tuple(p.shape)
        shapes[name] = list(shape)
        if name.endswith("engram.embed.scale"):
            p.data = torch.ones(shape, dtype=torch.float32)
        else:
            p.data = torch.from_numpy(prng.param_values(name, shape).copy())
    m.float().eval()
    return args, m, shapes


def main():
    out = {}
    args, m, shapes = build()
    # `engram.embed.scale` is folded into the table by the rlx loader, so the
    # port never asks for it; keep it out of the manifest it drives.
    out["shapes"] = {k: v for k, v in shapes.items() if not k.endswith("engram.embed.scale")}
    ids = torch.tensor([[(i * 7 + 3) % args.vocab_size for i in range(SEQ)]], dtype=torch.long)
    out["input_ids"] = ids[0].tolist()

    caught = {}
    hooks = []

    def grab(tag):
        def hook(mod, inp, o):
            t = o[0] if isinstance(o, tuple) else o
            caught[tag] = t.detach().float().reshape(-1).tolist()
        return hook

    # finer taps inside attention: the compressed KV actually attended to, the
    # index keys, and the top-k selection the Indexer made
    orig_ck = M.Attention._compress_kv
    orig_idx = M.Indexer.forward

    def ck(self, x, qr, start_pos, offset):
        out = orig_ck(self, x, qr, start_pos, offset)
        caught[f"compkv.{self.layer_id}"] = out[0].detach().float().reshape(-1).tolist()
        caught[f"topk.{self.layer_id}"] = out[1].detach().reshape(-1).tolist()
        return out

    def idxf(self, x, qr, latent, start_pos, offset):
        out = orig_idx(self, x, qr, latent, start_pos, offset)
        if getattr(self, "owns_k", False):
            caught[f"indexk.{self.layer_id if hasattr(self,'layer_id') else 'x'}"] = (
                shared_k(self).detach().float().reshape(-1).tolist()
            )
        return out

    def shared_k(self):
        return M.shared_attn.index_k[:, : M.shared_attn.index_k.size(1)]

    # record the exact tensor every `topk` sees, so ties are visible
    orig_topk = torch.Tensor.topk
    topk_calls = []

    class _TopkOut:
        def __init__(self, values, indices):
            self.values, self.indices = values, indices

        def __getitem__(self, i):
            return (self.values, self.indices)[i]

    def rec_topk(self, k, dim=-1, largest=True, sorted=True):
        # `torch.topk` leaves the order among EQUAL scores unspecified, and the
        # Indexer produces exact ties constantly (it rectifies its head scores,
        # so any position every head dislikes scores exactly 0). Pin it to
        # lowest-index-wins, which is what rlx's Op::TopK does, so the two
        # implementations are comparable at all.
        assert largest
        order = torch.argsort(-self, dim=dim, stable=True)
        idx = order.narrow(dim, 0, k)
        out = _TopkOut(self.gather(dim, idx), idx)
        topk_calls.append((tuple(self.shape), k, self.detach().float().reshape(-1).tolist(),
                           out.indices.detach().reshape(-1).tolist()))
        return out

    torch.Tensor.topk = rec_topk
    M.Attention._compress_kv = ck
    M.Indexer.forward = idxf

    for i, layer in enumerate(m.layers):
        hooks.append(layer.attn.register_forward_hook(grab(f"attn_out.{i}")))
        hooks.append(layer.ffn.register_forward_hook(grab(f"ffn_out.{i}")))
        hooks.append(layer.register_forward_hook(grab(f"block_out.{i}")))
        if layer.engram is not None:
            hooks.append(layer.engram.register_forward_hook(grab(f"engram_out.{i}")))
        if layer.attn.compressor is not None:
            hooks.append(layer.attn.compressor.register_forward_hook(grab(f"comp.{i}")))

    _, logits, _ = m(ids)
    M.Attention._compress_kv = orig_ck
    M.Indexer.forward = orig_idx
    torch.Tensor.topk = orig_topk
    import math

    def finite(xs):
        return [(-1e30 if x == -math.inf else (1e30 if x == math.inf else x)) for x in xs]

    out["topk_calls"] = [
        {"shape": list(sh), "k": k, "input": finite(inp), "indices": idx}
        for sh, k, inp, idx in topk_calls
    ]
    for h in hooks:
        h.remove()
    out["logits"] = logits.detach().float().reshape(-1).tolist()
    out["inter"] = caught
    out["config"] = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in vars(args).items()
    }
    # the engram hash ids the port must reproduce on the host side
    if m.engram_hash is not None:
        hashes = m.engram_hash(ids, 0, None)
        out["engram_rows"] = hashes.reshape(-1).tolist()
    json.dump(out, open("dsv41_ref.json", "w"))
    print("logits", logits.shape, "absmean", float(logits.abs().mean()))
    print("wrote dsv41_ref.json", os.path.getsize("dsv41_ref.json"), "bytes")


if __name__ == "__main__":
    main()
