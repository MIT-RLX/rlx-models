#!/usr/bin/env python3
"""Reference dumps for rlx-fish parity (Fish Audio S2-Pro / Drama 3 + codec).

Runs the official fish-speech PyTorch code (the `fish_speech/` package shipped in
the fishaudio/Drama-3 Space, or a fish-speech checkout) and writes one
safetensors file per mode into --out:

  codec   seeded random codes -> waveform (from_indices), and a speech clip ->
          codes (encode); weights from <model-dir>/codec.pth
  s2      S2-Pro dual-AR: prompt rows, slow hidden/logits at the last prompt
          row, teacher-forced fast logits, and N greedy frames
  drama3  Drama 3 backbone + codebook tower: prompt rows and N greedy frames

Every tensor is f32 (codes as i64). Needs torch, einops, tokenizers,
safetensors, numpy, soundfile. `loguru` is replaced by a stub when missing.

  python3 scripts/fish_reference.py codec --src /tmp/fishref/space \
      --model-dir weights/tts/fish-s2-pro --wav assets/jfk/jfk_voice_clone.wav \
      --out /tmp/fish_ref
"""

import argparse
import logging
import os
import sys
import types

import numpy as np
import torch
from safetensors.torch import save_file


def import_fish(src):
    try:
        import loguru  # noqa: F401
    except ImportError:
        stub = types.ModuleType("loguru")
        stub.logger = logging.getLogger("fish")
        sys.modules["loguru"] = stub
    sys.path.insert(0, os.path.abspath(src))


def codec_mode(args):
    from fish_speech.codec import load_codec

    codec = load_codec(os.path.join(args.model_dir, "codec.pth"), torch.float32)
    gen = torch.Generator().manual_seed(args.seed)
    frames = args.frames
    semantic = torch.randint(0, 4096, (1, 1, frames), generator=gen)
    residual = torch.randint(0, 1024, (1, 9, frames), generator=gen)
    codes = torch.cat((semantic, residual), dim=1)
    out = {"codes": codes[0].to(torch.int64)}
    with torch.no_grad():
        z = codec.quantizer.decode(codes.clone())
        out["latent"] = z[0].float()
        wav = codec.decoder(z)
        out["wav"] = wav[0, 0].float()
        if args.wav:
            import soundfile as sf
            import torchaudio

            audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
            audio = torch.from_numpy(audio.mean(axis=1))
            audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
            audio = audio[: int(args.seconds * codec.sample_rate)]
            out["enc_pcm"] = audio.float()
            x = audio[None, None]
            lengths = torch.tensor([x.shape[-1]])
            indices, feature_lengths = codec.encode(x, lengths)
            out["enc_codes"] = indices[0, :, : feature_lengths[0]].to(torch.int64)
            # What the RVQ searches: encoder, 4x downsample, pre transformer.
            length = x.shape[-1]
            pad = -(-length // codec.frame_length) * codec.frame_length - length
            z = codec.encoder(torch.nn.functional.pad(x, (0, pad)))
            z = codec.quantizer.pre_module(codec.quantizer.downsample(z))
            out["enc_latent"] = z[0]
    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(args.out, "codec.safetensors"))
    print({k: tuple(v.shape) for k, v in out.items()})


def s2_prompt(tokenizer, text, num_codebooks, ref_text=None, ref_codes=None):
    """Rows [1 + num_codebooks, T] exactly as Conversation.encode_for_inference."""
    semantic_begin = tokenizer.token_to_id("<|semantic:0|>")
    parts = ["<|im_start|>system\n"]
    if ref_codes is not None:
        parts += [
            "convert the provided text to speech reference to the following:\n\nText:\n",
            ref_text if "<|speaker:" in ref_text else f"<|speaker:0|>{ref_text}",
            "\n\nSpeech:\n",
            ref_codes,
        ]
    else:
        parts.append("convert the provided text to speech")
    parts += ["<|im_end|>\n", "<|im_start|>user\n", text, "<|im_end|>\n",
              "<|im_start|>assistant\n<|voice|>"]
    rows = []
    for part in parts:
        if isinstance(part, str):
            ids = torch.tensor(tokenizer.encode(part, add_special_tokens=False).ids, dtype=torch.long)
            r = torch.zeros((1 + num_codebooks, len(ids)), dtype=torch.long)
            r[0] = ids
        else:
            r = torch.cat((part[:1] + semantic_begin, part)).long()
        rows.append(r)
    return torch.cat(rows, dim=1)


def s2_mode(args):
    from tokenizers import Tokenizer

    from fish_speech.models.text2semantic.llama import BaseModelArgs, DualARTransformer
    from fish_speech.models.text2semantic.llama import precompute_freqs_cis
    from safetensors import safe_open

    config = BaseModelArgs.from_pretrained(args.model_dir)
    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    config.semantic_begin_id = tokenizer.token_to_id("<|semantic:0|>")
    config.semantic_end_id = tokenizer.token_to_id("<|semantic:4095|>")
    if args.layers:
        config.n_layer = args.layers
    config.max_seq_len = 1024
    # Allocate once and fill tensor by tensor: a 36-layer f32 copy is ~18 GB,
    # and a second (random-init or bf16) copy next to it would double that.
    with torch.device("meta"):
        model = DualARTransformer(config)
    model.to_empty(device="cpu")
    model.freqs_cis = precompute_freqs_cis(config.max_seq_len, config.head_dim, config.rope_base)
    model.fast_freqs_cis = precompute_freqs_cis(config.num_codebooks, config.fast_head_dim, config.rope_base)
    model.causal_mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
    params = dict(model.named_parameters())
    filled = set()
    for shard in sorted(f for f in os.listdir(args.model_dir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(args.model_dir, shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith("text_model.model."):
                    name = key[len("text_model.model."):]
                elif key.startswith("audio_decoder."):
                    rest = key[len("audio_decoder."):]
                    name = rest if rest.startswith("codebook_embeddings.") else "fast_" + rest
                else:
                    name = key
                if name.startswith("layers.") and int(name.split(".")[1]) >= config.n_layer:
                    continue
                with torch.no_grad():
                    params[name].copy_(f.get_tensor(key).float())
                filled.add(name)
    assert filled == set(params), sorted(set(params) - filled)[:8]
    model.eval()
    model.setup_caches(1, 1024, dtype=torch.float32)

    nc = config.num_codebooks
    prompt = s2_prompt(tokenizer, args.text, nc)
    t = prompt.shape[1]
    out = {"prompt": prompt.clone()}
    im_end = tokenizer.token_to_id("<|im_end|>")
    band = torch.full((config.vocab_size,), float("-inf"))
    band[config.semantic_begin_id: config.semantic_end_id + 1] = 0
    if not args.no_eos:
        band[im_end] = 0

    frames, hidden_rows, logit_rows, fast_logits = [], [], [], []
    with torch.no_grad():
        x = prompt[None]
        pos = torch.arange(t)
        for step in range(args.steps):
            res = model.forward_generate(x, pos)
            logits, hidden = res.logits[0, -1], res.hidden_states[0, -1]
            hidden_rows.append(hidden.clone())
            logit_rows.append(logits[config.semantic_begin_id: config.semantic_end_id + 1].clone())
            main = int((logits + band).argmax())
            if main == im_end:
                break
            c = [main - config.semantic_begin_id]
            model.forward_generate_fast(hidden[None], torch.tensor([0]))
            h = model.fast_embeddings(torch.tensor([c[0]]))
            per = []
            for i in range(1, nc):
                lg = model.forward_generate_fast(h, torch.tensor([i]))[0, 0]
                per.append(lg.clone())
                c.append(int(lg.argmax()))
                h = model.fast_embeddings(torch.tensor([c[-1]]))
            fast_logits.append(torch.stack(per))
            frame = torch.tensor([main] + c)
            frames.append(frame)
            x = frame[None, :, None]
            pos = torch.tensor([t + step])
    out["hidden"] = torch.stack(hidden_rows)
    out["band_logits"] = torch.stack(logit_rows)
    out["fast_logits"] = torch.stack(fast_logits)
    out["frames"] = torch.stack(frames)
    os.makedirs(args.out, exist_ok=True)
    name = f"s2_l{args.layers}.safetensors" if args.layers else "s2.safetensors"
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(args.out, name))
    print({k: tuple(v.shape) for k, v in out.items()}, "frames", out["frames"][:, :3].tolist())


def drama3_mode(args):
    import json

    from tokenizers import Tokenizer

    import fish_speech.models.text2semantic.fish_omni.tower as tower_mod
    from fish_speech.models.text2semantic.fish_omni.backbone import (
        FishOmniBackbone,
        load_backbone_weights,
    )
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
    from fish_speech.models.text2semantic.fish_omni.model import iter_checkpoint_tensors
    from fish_speech.models.text2semantic.fish_omni.tower import (
        FishOmniAudioDecoder,
        load_audio_decoder_weights,
    )

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = FishOmniConfig.from_dict(json.load(f))
    layers = args.layers or config.text.num_hidden_layers
    config.text.num_hidden_layers = layers
    config.text.layer_types = config.text.layer_types[:layers]
    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    dtype = torch.float32
    # Experts stay block-FP8 and are dequantized per forward (the reference's
    # CPU path): 4 layers of dense f32 experts alone would be 12 GB.
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=dtype)
        tower = FishOmniAudioDecoder(config, dtype=dtype)
    backbone.materialize(["cpu"])
    path = __import__("pathlib").Path(args.model_dir)

    def keep(item):
        key = item[0]
        parts = key.split(".")
        if key.startswith("model.language_model.layers."):
            return int(parts[3]) < layers
        return True

    load_backbone_weights(backbone, filter(keep, iter_checkpoint_tensors(
        path, skip_prefixes=("audio_decoder.", "audio_decoder_mtp_shared.", "audio_decoder_mtp."))))
    tower.to_empty(device="cpu")
    load_audio_decoder_weights(tower, iter_checkpoint_tensors(
        path, skip_prefixes=("model.language_model.", "lm_head.weight")))
    backbone.eval()
    tower.eval()

    prompt = build_drama3_prompt(args.text, tokenizer, config)
    rows = len(prompt)
    capacity = rows + args.steps + 1
    backbone.setup_caches(1, capacity)
    tower.setup_caches(capacity)

    recorded = []
    original = tower_mod.sample_top_k_top_p

    def record(logits, *a, **k):
        recorded.append(logits[0].float().clone())
        return original(logits, *a, **k)

    tower_mod.sample_top_k_top_p = record
    out = {"prompt": prompt.clone()}
    hidden_rows, main_logits, head_logits, frames_out = [], [], [], []
    with torch.no_grad():
        frames = torch.zeros((capacity, prompt.shape[1]), dtype=torch.long)
        frames[:rows] = prompt
        positions = torch.arange(capacity)
        emb = tower.embed_backbone(prompt, backbone.embed(prompt[:, 0]))
        hidden = backbone(inputs_embeds=emb[None], input_pos=positions[:rows])[0]
        if rows > 1:
            tower.prefill(hidden[: rows - 1], positions[: rows - 1], frames[: rows - 1], frames[1:rows])
        hidden = hidden[-1:]
        no_history = torch.full((16,), -1, dtype=torch.long)
        for step in range(args.steps):
            pos = rows - 1 + step
            hidden_rows.append(hidden[0].clone())
            logits = backbone.compute_logits(hidden)
            recorded.clear()
            sampled = tower.decode_step(logits, hidden, positions[pos: pos + 1], frames[pos: pos + 1],
                                        no_history, temperature=1.0, top_k=1, top_p=1.0)
            main_logits.append(recorded[0])  # candidate order: eos, then the band
            head_logits.append(torch.stack(recorded[2:]))  # [main, RAS retry, heads...]
            main = int(sampled[0, 0])
            if main == config.eos_token_id and not args.no_eos:
                break
            frames[pos + 1] = sampled[0]
            frames_out.append(sampled[0].clone())
            emb = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
            hidden = backbone(inputs_embeds=emb[None], input_pos=positions[pos + 1: pos + 2])[0]
    out["hidden"] = torch.stack(hidden_rows)
    out["main_logits"] = torch.stack(main_logits)
    out["head_logits"] = torch.stack(head_logits)
    out["frames"] = torch.stack(frames_out)
    os.makedirs(args.out, exist_ok=True)
    name = f"drama3_l{layers}.safetensors"
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(args.out, name))
    print({k: tuple(v.shape) for k, v in out.items()}, "frames", out["frames"][:, :3].tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["codec", "s2", "drama3"])
    p.add_argument("--src", required=True, help="directory holding the fish_speech package")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--wav")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--text", default="Hello from Fish Audio on RLX.")
    p.add_argument("--steps", type=int, default=6, help="greedy frames to generate")
    p.add_argument("--layers", type=int, default=0, help="truncate the slow backbone (0 = all)")
    p.add_argument("--no-eos", action="store_true", help="never pick <|im_end|> (truncated models)")
    args = p.parse_args()
    import_fish(args.src)
    torch.manual_seed(args.seed)
    {"codec": codec_mode, "s2": s2_mode, "drama3": drama3_mode}[args.mode](args)


if __name__ == "__main__":
    main()
