#!/usr/bin/env python3
"""Export NeuCodec encoder weights to safetensors for rlx-neutts.

Extracts the acoustic codec encoder (CodecEnc), semantic encoder head
(SemanticEncoder_module), latent fusion (fc_prior), and FSQ project_in from
the full NeuCodec PyTorch checkpoint.

Usage:
  python3 scripts/export_neucodec_encoder.py \\
      --checkpoint ~/.skill/models/neutts/hub/models--neuphonic--neucodec/blobs/<sha> \\
      --out weights/tts/neutts

  # Or download via Hugging Face:
  python3 scripts/export_neucodec_encoder.py --repo neuphonic/neucodec --out weights/tts/neutts

Requires: pip install torch safetensors huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENC_PREFIXES = (
    "CodecEnc.",
    "SemanticEncoder_module.",
    "fc_prior.",
    "generator.quantizer.project_in.",
)

# BigCodec strides wired in the NeuCodec reference implementation.
CODEC_ENC_STRIDES = [2, 2, 4, 4, 5]
SEMANTIC_LAYER = 16


def fold_weight_norm(state: dict, prefix: str, export: dict, torch) -> None:
    """Fold PyTorch weight_norm (g * v / ||v||) into a single .weight tensor."""
    g_key = f"{prefix}.weight_g"
    v_key = f"{prefix}.weight_v"
    b_key = f"{prefix}.bias"
    if g_key not in state or v_key not in state:
        return
    g = state[g_key].float()
    v = state[v_key].float()
    norm = torch.linalg.vector_norm(v, dim=tuple(range(1, v.ndim)), keepdim=True)
    export[f"{prefix}.weight"] = (g * (v / norm)).contiguous()
    if b_key in state:
        export[f"{prefix}.bias"] = state[b_key].float().contiguous()


def resolve_checkpoint(repo: str | None, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        return checkpoint
    if repo is None:
        repo = "neuphonic/neucodec"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            f"error: {e}\ninstall: pip install huggingface_hub torch safetensors"
        ) from e
    path = hf_hub_download(repo, "pytorch_model.bin")
    return Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=None,
        help="Hugging Face repo id (default: neuphonic/neucodec when --checkpoint omitted)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to pytorch_model.bin (HF cache blob or local file)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("weights/tts/neutts"),
        help="Output directory (writes neucodec_encoder.safetensors + config json)",
    )
    args = parser.parse_args()

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"error: {e}\ninstall: pip install torch safetensors", file=sys.stderr)
        return 1

    ckpt = resolve_checkpoint(args.repo, args.checkpoint)
    print(f"loading {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    raw = {k: v for k, v in state.items() if k.startswith(ENC_PREFIXES)}
    if not raw:
        print("error: no encoder tensors matched prefixes", file=sys.stderr)
        return 1

    export: dict = {}

    # Snake activations, resampling filters, and ordinary biases/weights.
    for k, v in raw.items():
        if ".weight_g" in k or ".weight_v" in k:
            continue
        if k.endswith(".weight") or k.endswith(".bias") or k.endswith(".alpha") or k.endswith(
            ".beta"
        ) or ".filter" in k:
            export[k] = v.float().contiguous()

    # Fold weight_norm conv layers (CodecEnc uses weight_norm throughout).
    wn_prefixes = {
        k[: -len(".weight_g")]
        for k in raw
        if k.endswith(".weight_g")
    }
    for prefix in sorted(wn_prefixes):
        fold_weight_norm(raw, prefix, export, torch)

    meta = {
        "codec_enc_strides": ",".join(str(s) for s in CODEC_ENC_STRIDES),
        "semantic_w2v_layer": str(SEMANTIC_LAYER),
        "encoder_sample_rate": "16000",
        "tokens_per_second": "50",
        "source_checkpoint": str(ckpt),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    st_path = args.out / "neucodec_encoder.safetensors"
    cfg_path = args.out / "neucodec_encoder_config.json"

    save_file(export, st_path, metadata=meta)
    cfg = {
        "codec_enc_strides": CODEC_ENC_STRIDES,
        "semantic_w2v_layer": SEMANTIC_LAYER,
        "encoder_sample_rate": 16000,
        "tokens_per_second": 50,
        "tensor_count": len(export),
        "prefixes": list(ENC_PREFIXES),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    by_prefix: dict[str, int] = {}
    for k in export:
        top = k.split(".", 1)[0]
        by_prefix[top] = by_prefix.get(top, 0) + 1

    print(f"wrote {len(export)} tensors -> {st_path}")
    print(f"wrote config -> {cfg_path}")
    print("tensor counts by prefix:", dict(sorted(by_prefix.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
