"""Dump the DeepSeek-V4.1 DSpark draft head reference outputs.

The draft head is isolated from the main model by feeding it a synthetic
`main_hidden`, so the fixture pins DSpark alone.
"""
import json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RLX_REF_NOQUANT", "1")
import prng, dump as D
import model as M

SEQ, POS = 12, 12
ID_SEED, ID_STEP = 3, 7


def main():
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    args = D.small_args(dspark=True)
    args.n_mtp_layers = 2
    args.dspark_block_size = 3
    args.dspark_noise_token_id = 5
    args.dspark_target_layer_ids = (4, 5)
    args.dspark_markov_rank = 16
    args.dspark_n_routed_experts = 2
    args.dspark_n_activated_experts = 2
    args.compress_ratios = tuple(args.compress_ratios[:args.n_layers]) + (0, 0)
    m = M.Transformer(args, tokenizer=object())
    shapes = {}
    for name, p in m.named_parameters():
        shapes[name] = list(p.shape)
        if name.endswith("engram.embed.scale"):
            p.data = torch.ones(tuple(p.shape), dtype=torch.float32)
        else:
            p.data = torch.from_numpy(prng.param_values(name, tuple(p.shape)).copy())
    m.float().eval()

    caught = {}
    orig_head = M.DSparkBlock.forward_head

    def head(self, x, pre_mix, input_ids):
        h = self.hc_pre(x, pre_mix)
        caught["hidden"] = h.detach().reshape(-1).tolist()
        caught["logits"] = self.head(self.norm(h), full_logits=True).detach().reshape(-1).tolist()
        return orig_head(self, x, pre_mix, input_ids)

    M.DSparkBlock.forward_head = head

    n_t = len(args.dspark_target_layer_ids)
    mh_seed = torch.from_numpy(prng.param_values("mh_seed", (1, SEQ, args.dim * n_t)).copy())
    mh_step = torch.from_numpy(prng.param_values("mh_step", (1, 1, args.dim * n_t)).copy())

    with torch.inference_mode():
        assert m.forward_spec(torch.tensor([[ID_SEED]]), mh_seed, 0) is None
        rings = [
            s.attn.window_kv_cache[0].detach().clone().reshape(-1).tolist() for s in m.mtp
        ]
        out_ids, logits, conf = m.forward_spec(torch.tensor([[ID_STEP]]), mh_step, POS)

    json.dump(
        {
            "shapes": shapes,
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
            "seq": SEQ,
            "pos": POS,
            "id_seed": ID_SEED,
            "id_step": ID_STEP,
            "mh_seed": mh_seed.reshape(-1).tolist(),
            "mh_step": mh_step.reshape(-1).tolist(),
            "rings": rings,
            "hidden": caught["hidden"],
            "logits": caught["logits"],
            "output_ids": out_ids.reshape(-1).tolist(),
            "biased_logits": logits.detach().reshape(-1).tolist(),
            "confidence": conf.detach().reshape(-1).tolist(),
        },
        open("dsv41_dspark_ref.json", "w"),
        separators=(",", ":"),
    )
    print("ring", len(rings), len(rings[0]), "logits", tuple(logits.shape), "ids", out_ids.tolist())


if __name__ == "__main__":
    main()
