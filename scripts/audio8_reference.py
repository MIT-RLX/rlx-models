#!/usr/bin/env python3
"""Reference dumps for Audio8-TTS-Realistic-Air parity (crates/rlx-fish).

Runs the code published with the checkpoint (`modeling_arktts.py`,
`modeling_arktts_codec.py`, `processing_arktts.py`) on the CPU in f32 and
writes one safetensors file per mode into --out:

  codec  seeded random codes -> waveform, a speech clip -> codes
         (audio8_codec.safetensors)
  lm     greedy frames: prompt rows from the real processor, per-step slow
         hidden state, band logits, fast logits and what sampling would choose
         among at the default settings (audio8.safetensors, or
         audio8_clone.safetensors with --wav/--ref-text)

--src is a directory holding those files as a package named `arktts` (with an
__init__.py), so nothing is loaded through trust_remote_code. Read them first.
Needs torch, transformers, safetensors, soundfile, torchaudio, numpy.

  python3 scripts/audio8_reference.py codec --src SRC --model-dir DIR \
      --wav assets/jfk/jfk_voice_clone.wav --out /tmp/audio8_ref
  python3 scripts/audio8_reference.py lm --src SRC --model-dir DIR --out /tmp/audio8_ref
  python3 scripts/audio8_reference.py lm --src SRC --model-dir DIR --out /tmp/audio8_ref \
      --wav assets/jfk/jfk_voice_clone.wav --ref-text "the clip's transcript"
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file


def load_model(args):
    sys.path.insert(0, os.path.abspath(args.src))
    from arktts.configuration_arktts import ArkttsConfig
    from arktts.modeling_arktts import ArkttsModel

    with open(os.path.join(args.model_dir, "config.json")) as f:
        raw = json.load(f)
    raw.pop("auto_map", None)
    raw.pop("architectures", None)
    raw.pop("dtype", None)
    config = ArkttsConfig(**raw)
    config._name_or_path = os.path.abspath(args.model_dir)
    model = ArkttsModel(config)
    state = {k: v.float() for k, v in load_file(os.path.join(args.model_dir, "model.safetensors")).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing[:8], unexpected[:8])
    return model.float().eval()


def read_wav(path, rate):
    import soundfile as sf
    import torchaudio

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(axis=1))
    return torchaudio.functional.resample(audio, sr, rate)


def codec_mode(args):
    model = load_model(args)
    codec = model.load_codec(device="cpu", dtype=torch.float32)
    gen = torch.Generator().manual_seed(args.seed)
    frames = args.frames
    semantic = torch.randint(0, 4096, (1, 1, frames), generator=gen)
    residual = torch.randint(0, 1024, (1, 9, frames), generator=gen)
    codes = torch.cat((semantic, residual), dim=1)
    out = {"codes": codes[0].to(torch.int64)}
    with torch.inference_mode():
        out["wav"] = codec.decode(codes)[0, 0].float()
        if args.wav:
            audio = read_wav(args.wav, codec.sample_rate)[: int(args.seconds * codec.sample_rate)]
            out["enc_pcm"] = audio.float()
            x = audio[None, None]
            indices, lengths = codec.encode(x)
            out["enc_codes"] = indices[0, :, : lengths[0]].to(torch.int64)
            pad = -(-x.shape[-1] // codec.frame_length) * codec.frame_length - x.shape[-1]
            z = codec.encoder(torch.nn.functional.pad(x, (0, pad)))
            z = codec.quantizer.pre_module(codec.quantizer.downsample(z))
            out["enc_latent"] = z[0]
    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.clone().contiguous() for k, v in out.items()},
              os.path.join(args.out, "audio8_codec.safetensors"))
    print({k: tuple(v.shape) for k, v in out.items()})


def lm_mode(args):
    from transformers.generation import LogitsProcessorList

    model = load_model(args)
    sys.path.insert(0, os.path.abspath(args.src))
    from arktts.modeling_arktts import ArkttsSemanticLogitsProcessor
    from arktts.processing_arktts import ArkttsProcessor

    processor = ArkttsProcessor.from_pretrained(args.model_dir)
    config = model.config
    kwargs = {}
    if args.wav:
        kwargs = {
            "reference_audio": [read_wav(args.wav, config.codec_sample_rate).numpy()],
            "sampling_rate": [config.codec_sample_rate],
            "reference_text": [args.ref_text],
        }
    inputs = dict(processor(text=[args.text], return_tensors="pt", **kwargs))

    nc = config.num_codebooks
    begin, end = config.semantic_begin_id, config.semantic_end_id
    out = {}
    hidden_rows, logit_rows, fast_rows, frames = [], [], [], []
    regular_rows, ras_rows, fast_pool_rows = [], [], []
    temperature, top_p, top_k = 0.7, 0.9, 50

    def pool(scores):
        return torch.softmax(scores.float(), dim=-1)[0]

    def band_only(row):
        return torch.cat((row[begin:end + 1], row[config.eos_token_id:config.eos_token_id + 1]))
    with torch.inference_mode():
        prompt, mask = model._prepare_prompt(**inputs)
        out["prompt"] = prompt[0].clone()
        width = prompt.shape[-1]
        model._setup_generation_caches(1, width + args.steps, torch.float32)
        logits, hidden = model._slow_step(
            prompt, torch.arange(width), mask.cumsum(-1).sub(1).clamp_min(0), mask
        )
        fast_step = model._fast_step

        def recording_fast_step(h, position):
            scores = fast_step(h, position)
            if position > 0:
                fast_rows.append(scores[0].clone())
                fast_pool_rows.append(pool(model._processed_scores(
                    None, scores, LogitsProcessorList(), top_k, top_p, temperature)))
            return scores

        model._fast_step = recording_fast_step
        none = LogitsProcessorList()
        history = prompt[:, 0]
        for step in range(args.steps):
            hidden_rows.append(hidden[0, 0].clone())
            logit_rows.append(band_only(logits[0]))
            # The reference's `_sample_semantic` filters, before the draw.
            band = LogitsProcessorList([ArkttsSemanticLogitsProcessor(begin, end, config.eos_token_id)])
            regular_rows.append(band_only(pool(model._processed_scores(
                history, logits, band, top_k, top_p, temperature))))
            ras_rows.append(band_only(pool(model._processed_scores(
                history, logits, band, top_k, config.ras_top_p, config.ras_temperature))))
            semantic = model._sample_semantic(history, logits, none, 50, 0.9, 1.0, None, False)
            if int(semantic) == config.eos_token_id:
                break
            codes = model._generate_codebooks(hidden, semantic, none, 50, 0.9, 1.0, False)
            frames.append(torch.cat((semantic, codes[0])))
            history = torch.cat((history, semantic[:, None]), dim=1)
            mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype)), dim=1)
            column = torch.cat((semantic[:, None], codes), dim=1).unsqueeze(-1)
            logits, hidden = model._slow_step(
                column, torch.tensor([width + step]), torch.tensor([[width + step]]), mask
            )
        model._fast_step = fast_step

        # The capture above must be what `generate` itself does.
        whole = model.generate(**inputs, do_sample=False, max_new_tokens=args.steps,
                               temperature=1.0, top_p=0.9, top_k=50)
    got = torch.stack(frames)
    assert whole.shape[-1] >= len(frames), (whole.shape, len(frames))
    assert torch.equal(whole[0, :, : len(frames)].T, got[:, 1:]), "capture differs from generate()"

    out["hidden"] = torch.stack(hidden_rows[: len(frames)])
    out["band_logits"] = torch.stack(logit_rows[: len(frames)])
    out["fast_logits"] = torch.stack(fast_rows).reshape(len(frames), nc - 1, -1)
    out["frames"] = got.to(torch.int64)
    out["pool"] = torch.stack(regular_rows[: len(frames)])
    out["ras_pool"] = torch.stack(ras_rows[: len(frames)])
    out["fast_pool"] = torch.stack(fast_pool_rows).reshape(len(frames), nc - 1, -1)
    os.makedirs(args.out, exist_ok=True)
    name = "audio8_clone.safetensors" if args.wav else "audio8.safetensors"
    save_file({k: v.clone().contiguous() for k, v in out.items()}, os.path.join(args.out, name))
    print({k: tuple(v.shape) for k, v in out.items()}, "main", got[:, 0].tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["codec", "lm"])
    p.add_argument("--src", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--wav")
    p.add_argument("--ref-text")
    p.add_argument("--text", default="Welcome to Audio8 on RLX. It is wonderful to hear from you.")
    args = p.parse_args()
    {"codec": codec_mode, "lm": lm_mode}[args.mode](args)


if __name__ == "__main__":
    main()
