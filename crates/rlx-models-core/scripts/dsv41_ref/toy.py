"""Build a small DeepSeek-V4.1 reference model on CPU with seeded f32 weights,
run it, and dump weights + activations for the rlx port to match."""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engram as engram_mod

VOCAB = 256
COMPRESSED_VOCAB = 64
engram_mod.build_compressed_token_map = lambda tok: ([i % COMPRESSED_VOCAB for i in range(VOCAB)], COMPRESSED_VOCAB)

import model as M
import torch.nn.functional as _F


def _pee_forward(self, indices):
    """float32 stand-in for ParallelEngramEmbedding.forward (same math, no bf16 cast)."""
    values = _F.embedding(indices, self.weight.float())
    scales = _F.embedding(indices, self.scale.float())
    values = values.unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)
    return values.flatten(-2)


M.ParallelEngramEmbedding.forward = _pee_forward


def toy_args(vision=False, engram=True, dspark=False):
    kw = dict(
        max_batch_size=1, max_seq_len=64, temperature=0, dtype="bf16", expert_dtype=None,
        vocab_size=VOCAB, dim=64, moe_inter_dim=32, n_layers=6, n_mtp_layers=0,
        n_heads=4, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
        score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0, norm_topk_prob=True,
        q_lora_rank=32, head_dim=32, rope_head_dim=16, norm_eps=1e-20, o_groups=2, o_lora_rank=16,
        window_size=8, compress_ratios=(0, 0, 2, 2, 1, 1),
        kv_source_layers=(2, 4), index_source_layers=(2, 4, 5),
        compress_rope_theta=40000.0, original_seq_len=32, rope_theta=10000.0,
        rope_factor=4, beta_fast=32, beta_slow=1,
        index_n_heads=2, index_head_dim=32, index_topk=4,
        candidate_source_layer=4, candidate_topk_blocks=2, candidate_block_size=2,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    )
    if engram:
        kw.update(engram_layer_ids=(1, 3), engram_max_ngram_size=4, engram_vocab_size=97,
                  engram_n_heads=2, engram_head_dim=32, engram_pad_id=2,
                  engram_compressed_vocab_size=COMPRESSED_VOCAB)
    if vision:
        kw.update(vision_n_layers=2, vision_dim=32, vision_n_heads=2, vision_inter_dim=48,
                  vision_patch_size=2, vision_downsample_ratio=2, vision_rope_theta=10000.0)
    if dspark:
        kw["compress_ratios"] = kw["compress_ratios"] + (0, 0)
        kw.update(n_mtp_layers=2, dspark_block_size=3, dspark_noise_token_id=5,
                  dspark_target_layer_ids=(4, 5), dspark_markov_rank=16,
                  dspark_n_routed_experts=4, dspark_n_activated_experts=2)
    args = M.ModelArgs(**kw)
    if engram:
        # size each table so every (n-gram, head) prime bucket range fits
        layout = engram_mod.EngramLayout.from_args(args)
        rows = []
        for per_layer in layout.primes:
            flat = [p for per_ngram in per_layer for p in per_ngram]
            rows.append(int(sum(flat)))
        args.engram_num_embeddings = tuple(rows)
    return args


def seeded_init(mod, seed=1234):
    g = torch.Generator().manual_seed(seed)
    for name, p in sorted(mod.named_parameters()):
        t = torch.empty(p.shape, dtype=torch.float32)
        if "engram" in name and name.endswith(".scale"):
            t = torch.ones(p.shape, dtype=torch.float32)
        elif p.dim() >= 2:
            t.normal_(0.0, (1.0 / p.shape[-1]) ** 0.5, generator=g)
        else:
            t.normal_(0.0, 0.2, generator=g)
        p.data = t
    return mod


def build(**kw):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    args = toy_args(**kw)
    m = M.Transformer(args, tokenizer=object())
    m = seeded_init(m).float()
    m.eval()
    return args, m


if __name__ == "__main__":
    os.environ.setdefault("RLX_REF_NOQUANT", "1")
    args, m = build()
    ids = torch.arange(1, 25, dtype=torch.long).unsqueeze(0) % VOCAB
    out, logits, main_hidden = m(ids)
    print("logits", tuple(logits.shape), float(logits.float().abs().mean()))
    print("argmax", out.tolist())
