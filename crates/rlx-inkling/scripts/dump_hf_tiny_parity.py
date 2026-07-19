#!/usr/bin/env python3
# Dump a tiny InklingForCausalLM (transformers) reference for Rust eager parity.
#
#   pip install 'transformers>=5.14' torch
#   python crates/rlx-inkling/scripts/dump_hf_tiny_parity.py
#
# Writes crates/rlx-inkling/tests/fixtures/hf_tiny_parity/
#   config.json meta.json weights.bin weights_index.json logits.bin
#   (+ optional weights.npz / logits.npy for Python debugging)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import InklingForCausalLM, InklingTextConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "hf_tiny_parity"
IDS = [1, 2, 3, 4]


def build_config() -> InklingTextConfig:
    return InklingTextConfig(
        vocab_size=32,
        unpadded_vocab_size=32,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=4,
        sliding_window_size=4,
        d_rel=4,
        rel_extent=8,
        log_scaling_n_floor=None,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        conv_kernel_size=3,
        dense_mlp_idx=1,
        local_layer_ids=[0, 1],
        dense_intermediate_size=32,
        intermediate_size=32,
        moe_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        shared_expert_sink=True,
        route_scale=1.0,
        logits_mup_width_multiplier=1.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def squeeze_conv(w: torch.Tensor) -> np.ndarray:
    # [C, 1, K] → [C, K] row-major flat
    assert w.ndim == 3 and w.shape[1] == 1, w.shape
    return w.squeeze(1).detach().float().cpu().numpy().reshape(-1)


def to_flat(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().contiguous().numpy().reshape(-1)


def export_weights(model: InklingForCausalLM) -> dict[str, np.ndarray]:
    sd = model.state_dict()
    out: dict[str, np.ndarray] = {}
    out["embed"] = to_flat(sd["model.embed_tokens.weight"])
    out["embed_norm"] = to_flat(sd["model.embed_norm.weight"])
    out["norm"] = to_flat(sd["model.norm.weight"])
    out["unembed"] = to_flat(sd["lm_head.weight"])

    n_layers = model.config.num_hidden_layers
    for layer in range(n_layers):
        p = f"model.layers.{layer}"
        out[f"layers.{layer}.attn_norm"] = to_flat(sd[f"{p}.input_layernorm.weight"])
        out[f"layers.{layer}.mlp_norm"] = to_flat(sd[f"{p}.post_attention_layernorm.weight"])
        out[f"layers.{layer}.attn_sconv"] = squeeze_conv(sd[f"{p}.attn_sconv.conv1d.weight"])
        out[f"layers.{layer}.mlp_sconv"] = squeeze_conv(sd[f"{p}.mlp_sconv.conv1d.weight"])
        out[f"layers.{layer}.q_norm"] = to_flat(sd[f"{p}.self_attn.q_norm.weight"])
        out[f"layers.{layer}.k_norm"] = to_flat(sd[f"{p}.self_attn.k_norm.weight"])
        out[f"layers.{layer}.k_sconv"] = squeeze_conv(sd[f"{p}.self_attn.k_sconv.conv1d.weight"])
        out[f"layers.{layer}.v_sconv"] = squeeze_conv(sd[f"{p}.self_attn.v_sconv.conv1d.weight"])
        out[f"layers.{layer}.rel_proj"] = to_flat(sd[f"{p}.self_attn.rel_logits_proj.proj"])
        out[f"layers.{layer}.wq"] = to_flat(sd[f"{p}.self_attn.q_proj.weight"])
        out[f"layers.{layer}.wk"] = to_flat(sd[f"{p}.self_attn.k_proj.weight"])
        out[f"layers.{layer}.wv"] = to_flat(sd[f"{p}.self_attn.v_proj.weight"])
        out[f"layers.{layer}.wr"] = to_flat(sd[f"{p}.self_attn.r_proj.weight"])
        out[f"layers.{layer}.wo"] = to_flat(sd[f"{p}.self_attn.o_proj.weight"])

        if f"{p}.mlp.gate_proj.weight" in sd:
            out[f"layers.{layer}.gate"] = to_flat(sd[f"{p}.mlp.gate_proj.weight"])
            out[f"layers.{layer}.up"] = to_flat(sd[f"{p}.mlp.up_proj.weight"])
            out[f"layers.{layer}.down"] = to_flat(sd[f"{p}.mlp.down_proj.weight"])
            out[f"layers.{layer}.mlp_global_scale"] = to_flat(sd[f"{p}.mlp.global_scale"])
        else:
            # experts.gate_up_proj: [E, 2I, H] — flatten as-is (expert-major)
            out[f"layers.{layer}.expert_w13"] = to_flat(sd[f"{p}.mlp.experts.gate_up_proj"])
            out[f"layers.{layer}.expert_w2"] = to_flat(sd[f"{p}.mlp.experts.down_proj"])
            out[f"layers.{layer}.gate_weight"] = to_flat(sd[f"{p}.mlp.gate.weight"])
            out[f"layers.{layer}.gate_bias"] = to_flat(sd[f"{p}.mlp.gate.e_score_correction_bias"])
            out[f"layers.{layer}.gate_global_scale"] = to_flat(sd[f"{p}.mlp.gate.global_scale"])
            out[f"layers.{layer}.shared_gate"] = to_flat(sd[f"{p}.mlp.shared_experts.gate_proj"])
            out[f"layers.{layer}.shared_up"] = to_flat(sd[f"{p}.mlp.shared_experts.up_proj"])
            out[f"layers.{layer}.shared_down"] = to_flat(sd[f"{p}.mlp.shared_experts.down_proj"])
    return out


def main() -> None:
    torch.manual_seed(0)
    cfg = build_config()
    model = InklingForCausalLM(cfg)
    model.eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % (2**31 - 1)
            g = torch.Generator().manual_seed(seed)
            p.normal_(0.0, 0.02, generator=g)
            if "global_scale" in name:
                p.fill_(1.0)
            if "e_score_correction_bias" in name:
                p.zero_()

    ids = torch.tensor([IDS], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu().numpy()

    weights = export_weights(model)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "weights.npz", **weights)
    np.save(OUT / "logits.npy", logits)
    # Rust-friendly flat dump (no ndarray dependency).
    index: dict[str, dict[str, int]] = {}
    blob = bytearray()
    for k in sorted(weights):
        arr = weights[k].astype(np.float32).reshape(-1)
        index[k] = {"offset": len(blob) // 4, "len": int(arr.size)}
        blob.extend(arr.tobytes())
    (OUT / "weights.bin").write_bytes(blob)
    (OUT / "weights_index.json").write_text(json.dumps(index, indent=2) + "\n")
    (OUT / "logits.bin").write_bytes(logits.astype(np.float32).tobytes())
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "input_ids": IDS,
                "vocab_size": cfg.vocab_size,
                "hidden_size": cfg.hidden_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "transformers_version": __import__("transformers").__version__,
                "logits_sum": float(logits.sum()),
                "logits_max": float(logits.max()),
            },
            indent=2,
        )
        + "\n"
    )
    # Minimal text_config JSON for Rust loaders.
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["InklingForConditionalGeneration"],
                "model_type": "inkling_mm_model",
                "eos_token_id": 2,
                "text_config": {
                    "vocab_size": 32,
                    "unpadded_vocab_size": 32,
                    "hidden_size": 16,
                    "num_hidden_layers": 3,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "swa_num_attention_heads": 4,
                    "swa_num_key_value_heads": 2,
                    "swa_head_dim": 4,
                    "sliding_window_size": 4,
                    "d_rel": 4,
                    "rel_extent": 8,
                    "model_max_length": 64,
                    "rms_norm_eps": 1e-6,
                    "sconv_kernel_size": 3,
                    "use_embed_norm": True,
                    "local_layer_ids": [0, 1],
                    "dense_mlp_idx": 1,
                    "dense_intermediate_size": 32,
                    "intermediate_size": 16,
                    "moe_intermediate_size": 16,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "n_shared_experts": 1,
                    "shared_expert_sink": True,
                    "route_scale": 1.0,
                    "logits_mup_width_multiplier": 1.0,
                },
                "audio_config": {"n_mel_bins": 4, "mel_vocab_size": 4, "decoder_dmodel": 16},
                "vision_config": {
                    "patch_size": 8,
                    "temporal_patch_size": 2,
                    "n_channels": 3,
                    "n_layers": 2,
                    "decoder_dmodel": 16,
                    "vision_encoder_type": "hmlp",
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT}")
    print(f"logits sum={logits.sum():.6f} max={logits.max():.6f}")


if __name__ == "__main__":
    main()
