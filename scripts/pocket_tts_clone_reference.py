#!/usr/bin/env python3
"""Voice-cloning reference for `rlx-pocket-tts` (Mimi encoder + speaker projection).

The ungated `kyutai/pocket-tts-without-voice-cloning` checkpoints ship the Mimi
encoder (SEANet + encoder transformer) zeroed, so cloning cannot be checked on
the released weights without the gated repo. This writes a copy of a
checkpoint whose zeroed encoder tensors are replaced by seeded random values of
a sensible scale, runs upstream `get_state_for_audio_prompt` on a WAV with it,
and dumps the latents, the speaker-projected conditioning and the resulting
voice state for the Rust side to compare against.

    .venv-pocket/bin/python -I scripts/pocket_tts_clone_reference.py \
        --lang-dir weights/tts/pocket-tts-kyutai/languages/english_2026-10 \
        --language english_2026-10 \
        --wav weights/tts/pocket-tts-kyutai/reference/english_2026-10-alba/reference.wav \
        --out weights/tts/pocket-tts-kyutai/reference/clone-english_2026-10
"""

import argparse
import pathlib
import tempfile

import safetensors.torch
import torch
import yaml

import pocket_tts.utils.config as pt_config
from pocket_tts.models.tts_model import TTSModel


def randomized_checkpoint(src: pathlib.Path, dst: pathlib.Path, seed: int):
    g = torch.Generator().manual_seed(seed)
    tensors = safetensors.torch.load_file(str(src))
    for name, t in tensors.items():
        if not name.startswith(("mimi.encoder.", "mimi.encoder_transformer.")):
            continue
        shape = t.shape
        if name.endswith("layer_scale_1.scale") or name.endswith("layer_scale_2.scale"):
            v = torch.full(shape, 0.1)
        elif name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            v = 1.0 + 0.1 * torch.randn(shape, generator=g)
        elif name.endswith(".bias"):
            v = 0.05 * torch.randn(shape, generator=g)
        else:
            fan_in = int(torch.tensor(shape[1:]).prod().item()) if len(shape) > 1 else shape[0]
            v = torch.randn(shape, generator=g) / fan_in**0.5
        tensors[name] = v.to(torch.float32).contiguous()
    safetensors.torch.save_file(tensors, str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-dir", required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lang_dir = pathlib.Path(args.lang_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "model.safetensors"
    randomized_checkpoint(lang_dir / "model.safetensors", weights, args.seed)

    cfg = yaml.safe_load((pt_config.CONFIGS_DIR / f"{args.language}.yaml").read_text())
    cfg["weights_path"] = str(weights.resolve())
    cfg.pop("weights_path_without_voice_cloning", None)
    tok = cfg["flow_lm"]["lookup_table"]
    name = "tokenizer.json" if tok["tokenizer"] in ("tokenizers", "json") else "tokenizer.model"
    tok["tokenizer_path"] = str((lang_dir / name).resolve())
    cfg_path = pathlib.Path(tempfile.mkdtemp()) / f"{args.language}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))

    model = TTSModel.load_model(config=cfg_path)
    model.eval()
    captured = {}
    real_encode = model.mimi.encode_to_latent

    def encode(x):
        captured["audio"] = x.detach().clone()
        z = real_encode(x)
        captured["latents"] = z.detach().clone()
        return z

    model.mimi.encode_to_latent = encode
    with torch.no_grad():
        state = model.get_state_for_audio_prompt(pathlib.Path(args.wav))
    tensors = {
        "audio": captured["audio"].reshape(-1).contiguous(),
        "latents": captured["latents"][0].contiguous(),
        "conditioning": torch.nn.functional.linear(
            captured["latents"].to(torch.float32), model.flow_lm.speaker_proj_weight
        )[0].contiguous(),
    }
    for module, st in state.items():
        offset = int(st["offset"].view(-1)[0])
        tensors[f"{module}/k"] = st["cache"][0, 0, :offset].contiguous()
        tensors[f"{module}/v"] = st["cache"][1, 0, :offset].contiguous()
    safetensors.torch.save_file(tensors, str(out / "clone_reference.safetensors"))
    print(f"wrote {out}: {tensors['latents'].shape[0]} latent frames, offset {offset}")


if __name__ == "__main__":
    main()
